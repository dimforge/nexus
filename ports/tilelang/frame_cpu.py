"""Shared resident frame controller with native C storage/execution defaults."""

import itertools
from pathlib import Path

import numpy as np
from frame_manifest import validate_frame, validate_growth
from ordered import CompilationSession
from replay import execute_c, load_c_library


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
        self.loaded_programs = {}
        self.loaded_libraries = {}
        self.current = 0
        self.steps = 0
        self.poisoned = False
        self.environment = {}
        for v in variants:
            self._validate_variant(v)
        for old, new in itertools.pairwise(variants):
            validate_growth(old["manifest"], new["manifest"])
        self.compiled = self._compile(0)
        self.storage = self._allocate(variants[0]["initial"])

    def _allocate(self, initial):
        return {n: a.copy() for n, a in initial.items()}

    def _discard(self, storage):
        """Device subclasses release the allocations owned by this mapping."""

    def _synchronize(self):
        """CPU calls are synchronous; device subclasses wait only at boundaries."""

    def _read(self, buffer, start, count):
        return buffer[start : start + count].copy()

    def _write(self, buffer, start, words):
        buffer[start : start + len(words)] = words

    def _copy(self, destination, dst_start, source, src_start, count):
        destination[dst_start : dst_start + count] = source[
            src_start : src_start + count
        ]

    def read(self, name):
        if self.poisoned:
            raise RuntimeError("Poisoned frame")
        count = (
            self.variants[self.current]["manifest"]["program"]["buffer_bytes"][name]
            // 4
        )
        self._synchronize()
        return self._read(self.storage[name], 0, count)

    def close(self):
        self._discard(self.storage)
        self.storage = {}
        self.poisoned = True

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

    def write_persistent(self, region, word_offset, words):
        """Caller supplies valid kernel input words; checked bounds preserve guards."""
        if self.poisoned:
            raise RuntimeError("Poisoned frame")
        r = self.variants[self.current]["manifest"]["persistent"].get(region)
        if (
            r is None
            or type(word_offset) is not int
            or word_offset < 0
            or words.dtype != np.uint32
            or words.ndim != 1
            or word_offset + words.size > r["word_len"]
        ):
            raise ValueError("Invalid persistent write range or words")
        start = r["word_start"] + word_offset
        self._write(self.storage[r["buffer"]], start, words)

    def checkpoint(self):
        if self.poisoned:
            raise RuntimeError("Poisoned frame")
        m = self.variants[self.current]["manifest"]
        return {
            "schema": 1,
            "steps": self.steps,
            "capacities": {g["id"]: g["capacity"] for g in m["gates"]},
            "regions": {
                name: self._read(
                    self.storage[r["buffer"]], r["word_start"], r["word_len"]
                ).tolist()
                for name, r in m["persistent"].items()
            },
        }

    def restore(self, checkpoint):
        """Restore trusted solver state after complete shape checks, before publishing."""
        from frame_checkpoint import fits, validate_checkpoint

        if self.poisoned:
            raise RuntimeError("Poisoned frame")
        validate_checkpoint(checkpoint)
        index = next(
            (i for i, v in enumerate(self.variants) if fits(checkpoint, v["manifest"])),
            None,
        )
        if index is None:
            raise ValueError("Checkpoint has no compatible capacity specialization")
        v = self.variants[index]
        compiled = self._compile(index)
        storage = self._allocate(v["initial"])
        try:
            for name, words in checkpoint["regions"].items():
                region = v["manifest"]["persistent"][name]
                start = region["word_start"]
                self._write(
                    storage[region["buffer"]], start, np.array(words, np.uint32)
                )
            self._synchronize()
        except Exception:
            self._discard(storage)
            raise
        self._discard(self.storage)
        self.compiled, self.storage, self.current, self.steps = (
            compiled,
            storage,
            index,
            checkpoint["steps"],
        )
        return index

    def reset(self):
        """Return to the original upload without compiling or running reference physics."""
        if self.poisoned:
            raise RuntimeError("Poisoned frame")
        v = self.variants[0]
        compiled = self._compile(0)
        storage = self._allocate(v["initial"])
        try:
            self._synchronize()
        except Exception:
            self._discard(storage)
            raise
        self._discard(self.storage)
        self.compiled, self.storage, self.current, self.steps = compiled, storage, 0, 0

    def _compile(self, index):
        if index in self.loaded_programs:
            return self.loaded_programs[index]
        v = self.variants[index]
        result = []
        for stage in v["manifest"]["program"]["stages"]:
            source, bindings = stage["source"], stage["bindings"]
            library = v.get("libraries", {}).get(source)
            if library is None:
                compiled = self.compiler.compile_c(
                    v["sources"][source], bindings, self.directory
                )
            else:
                # The same kernel appears in many solver passes. Loading copied
                # library files separately repeats loader/signature work. Exact
                # bytes and ABI equality let all those passes share one handle.
                key = (
                    Path(library).read_bytes(),
                    bindings["entry"],
                    tuple(bindings["buffers"]),
                )
                if key not in self.loaded_libraries:
                    self.loaded_libraries[key] = load_c_library(
                        library, bindings, {"runtime": "cached native TileLang C++"}
                    )
                compiled = self.loaded_libraries[key]
            result.append(compiled)

        self.loaded_programs[index] = result
        return result

    def _range(self, start, end, counts=None):
        v = self.variants[self.current]
        stages = v["manifest"]["program"]["stages"]
        if not 0 <= start <= end <= len(stages):
            raise ValueError("Invalid stage range")
        for i in range(start, end):
            s = stages[i]
            condition = v["manifest"].get("conditions", {}).get(s["name"])
            if (
                condition is not None
                and counts[condition["gate"]] < condition["minimum"]
            ):
                continue
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
        storage = self._allocate(v["initial"])
        try:
            for name, old in self.variants[self.current]["manifest"][
                "persistent"
            ].items():
                new = v["manifest"]["persistent"][name]
                n = old["word_len"]
                self._copy(
                    storage[new["buffer"]],
                    new["word_start"],
                    self.storage[old["buffer"]],
                    old["word_start"],
                    n,
                )
            self._synchronize()
        except Exception:
            self._discard(storage)
            raise
        self._discard(self.storage)
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
                    self._synchronize()
                except Exception:
                    self.poisoned = True
                    raise
                needed = sum(
                    int(self._read(self.storage[c["buffer"]], c["word"], 1)[0])
                    for c in gate["counters"]
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
                self._range(start, len(names), counts)
                self._synchronize()
            except Exception:
                self.poisoned = True
                raise
            for a, b in m.get("swaps", []):
                self.storage[a], self.storage[b] = self.storage[b], self.storage[a]
            self.steps += 1
            return {
                "step": self.steps,
                "variant": self.current,
                "retries": retries,
                "counts": counts,
            }
