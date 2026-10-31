"""Resident CUDA/Ascend control using the same transactional gates as C.

The device interface owns allocations, binaries and a single ordered stream.
Only capacity counters and explicit application/checkpoint reads reach the CPU.
"""

from frame_cpu import CpuFrameSession


class DeviceFrameSession(CpuFrameSession):
    def __init__(self, variants, directory, device):
        self.device = device
        try:
            super().__init__(variants, directory)
            self.environment.update(device.environment)
        except Exception:
            device.close()
            raise

    def _allocate(self, initial):
        result = {}
        try:
            for name, words in initial.items():
                result[name] = self.device.upload(words)
            return result
        except Exception:
            self._discard(result)
            raise

    def _discard(self, storage):
        for allocation in storage.values():
            self.device.free(allocation)

    def _synchronize(self):
        try:
            self.device.synchronize()
        except Exception:
            self.poisoned = True
            raise

    def _read(self, buffer, start, count):
        return self.device.read(buffer, start, count)

    def _write(self, buffer, start, words):
        self.device.write(buffer, start, words)

    def _copy(self, destination, dst_start, source, src_start, count):
        self.device.copy(destination, dst_start, source, src_start, count)

    def _compile(self, index):
        if index not in self.loaded_programs:
            v = self.variants[index]
            self.loaded_programs[index] = [
                self.device.compile(
                    v["sources"][s["source"]], s["bindings"], self.directory
                )
                for s in v["manifest"]["program"]["stages"]
            ]
        return self.loaded_programs[index]

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
            # CUDA/Ascend indirect descriptors must already be lowered to a
            # device-side virtual grid, never read to choose a host launch size.
            if "indirect" in s:
                raise ValueError(
                    "Native indirect dispatch requires virtual-grid lowering"
                )
            args = [self.storage[s["buffer_map"][n]] for n in s["bindings"]["buffers"]]
            for _ in range(s["bindings"]["iterations"]):
                self.device.launch(self.compiled[i], args, s["bindings"])

    def close(self):
        try:
            super().close()
        finally:
            self.device.close()
