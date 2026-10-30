"""Native C counterpart of the resident frame controller; kernels do all physics."""

import itertools
from pathlib import Path

import numpy as np
from frame_manifest import validate_frame, validate_growth
from ordered import CompilationSession
from replay import execute_c


class CapacityError(RuntimeError):
    def __init__(self, gate, required, capacity):
        super().__init__(f"{gate}: required {required}, capacity {capacity}")
        self.gate, self.required, self.capacity = gate, required, capacity


class CpuFrameSession:
    """Sources and gate placement have the same trusted-code contract as Rust."""

    def __init__(self, variants, directory):
        if not variants:
            raise ValueError("No frame specialization")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.variants = variants
        self.compiler = CompilationSession()
        self.current = 0
        self.steps = 0
        self.poisoned = False
        self.environment = {}
        for v in variants:
            self._validate_variant(v)
        for old, new in itertools.pairwise(variants):
            validate_growth(old["manifest"], new["manifest"])
        self.compiled = self._compile(0)
        self.storage = {n: a.copy() for n, a in variants[0]["initial"].items()}

    @staticmethod
    def _validate_variant(v):
        validate_frame(v["manifest"])
        sizes = v["manifest"]["program"]["buffer_bytes"]
        if set(v["initial"]) != set(sizes) or any(
            a.dtype != np.uint32 or a.ndim != 1 or a.nbytes != sizes[n]
            for n, a in v["initial"].items()
        ):
            raise ValueError("Invalid frame upload")

    def extend(self, variant):
        """Append trusted capacity specialization without changing live state."""
        if self.poisoned:
            raise RuntimeError("Poisoned frame")
        self._validate_variant(variant)
        validate_growth(self.variants[-1]["manifest"], variant["manifest"])
        self.variants.append(variant)

    def _compile(self, index):
        v = self.variants[index]
        return [
            self.compiler.compile_c(
                v["sources"][s["source"]], s["bindings"], self.directory
            )
            for s in v["manifest"]["program"]["stages"]
        ]

    def _range(self, start, end):
        v = self.variants[self.current]
        stages = v["manifest"]["program"]["stages"]
        if not 0 <= start <= end <= len(stages):
            raise ValueError("Invalid stage range")
        for i in range(start, end):
            s = stages[i]
            local = {n: self.storage[p] for n, p in s["buffer_map"].items()}
            self.environment.update(
                execute_c(
                    v["sources"][s["source"]],
                    s["bindings"],
                    self.directory,
                    local,
                    compiled=self.compiled[i],
                )
            )

    def _grow(self, index):
        v = self.variants[index]
        compiled = self._compile(index)
        storage = {n: a.copy() for n, a in v["initial"].items()}
        for name, old in self.variants[self.current]["manifest"]["persistent"].items():
            new = v["manifest"]["persistent"][name]
            n = old["word_len"]
            storage[new["buffer"]][new["word_start"] : new["word_start"] + n] = (
                self.storage[old["buffer"]][old["word_start"] : old["word_start"] + n]
            )
        self.compiled, self.storage, self.current = compiled, storage, index

    def step(self):
        if self.poisoned:
            raise RuntimeError("Poisoned frame")
        limit = self.variants[self.current]["manifest"]["max_steps"]
        if self.steps == 0xFFFFFFFFFFFFFFFF or (
            limit is not None and self.steps >= limit
        ):
            raise RuntimeError("Frame step limit")
        retries = 0
        while True:
            m = self.variants[self.current]["manifest"]
            names = [s["name"] for s in m["program"]["stages"]]
            start = 0
            counts = {}
            overflow = None
            for gate in m["gates"]:
                end = names.index(gate["after_stage"]) + 1
                try:
                    self._range(start, end)
                except Exception:
                    self.poisoned = True
                    raise
                needed = sum(
                    int(self.storage[c["buffer"]][c["word"]]) for c in gate["counters"]
                )
                counts[gate["id"]] = needed
                if needed > gate["capacity"]:
                    overflow = (gate, needed)
                    break
                start = end
            if overflow is not None:
                gate, needed = overflow
                next_index = next(
                    (
                        i
                        for i in range(self.current + 1, len(self.variants))
                        if any(
                            g["id"] == gate["id"] and g["capacity"] >= needed
                            for g in self.variants[i]["manifest"]["gates"]
                        )
                    ),
                    None,
                )
                if next_index is None:
                    raise CapacityError(gate["id"], needed, gate["capacity"])
                self._grow(next_index)
                retries += 1
                continue
            try:
                self._range(start, len(names))
            except Exception:
                self.poisoned = True
                raise
            self.steps += 1
            return {
                "step": self.steps,
                "variant": self.current,
                "retries": retries,
                "counts": counts,
            }
