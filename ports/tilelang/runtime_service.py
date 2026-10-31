"""JSON-line control for a resident C/CUDA/Ascend frame session.

Only explicit read requests copy body state to the Rust caller. Physics remains
inside compiled TileLang kernels; this process contains no host physics fallback.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from frame_cpu import CapacityError, CpuFrameSession
from runtime_compile import load_variant


def serve(directory, input_stream, output_stream, target="c"):
    directory = Path(directory)
    variant = load_variant(directory, expected_target=target)
    if target == "c":
        runtime = CpuFrameSession([variant], directory / "native-c")
    else:
        from device_frame import DeviceFrameSession
        from device_replay import create_device

        runtime = DeviceFrameSession(
            [variant], directory / "device-build", create_device(target)
        )

    def reply(value):
        output_stream.write(json.dumps(value) + "\n")
        output_stream.flush()

    try:
        reply({"ready": True})
        for line in input_stream:
            try:
                request = json.loads(line)
                command = request["command"]
                if command == "step":
                    result = runtime.step()
                elif command == "checkpoint":
                    result = runtime.checkpoint()
                elif command == "restore":
                    result = runtime.restore(request["checkpoint"])
                elif command == "reset":
                    result = runtime.reset()
                elif command == "read":
                    result = runtime.read(request["buffer"]).tolist()
                elif command == "read_binary":
                    if not hasattr(output_stream, "buffer"):
                        raise ValueError(
                            "Binary readback requires a binary-capable output stream"
                        )
                    words = runtime.read(request["buffer"])
                    reply({"ok": {"binary_words": int(words.size)}})
                    output_stream.buffer.write(
                        words.astype("<u4", copy=False).tobytes()
                    )
                    output_stream.buffer.flush()
                    continue
                elif command == "write":
                    words = request["words"]
                    if not isinstance(words, list) or any(
                        type(w) is not int or not 0 <= w <= 0xFFFFFFFF for w in words
                    ):
                        raise ValueError("Invalid native word upload")
                    runtime.write_persistent(
                        request["region"],
                        request["word_offset"],
                        np.array(words, np.uint32),
                    )
                    result = None
                elif command == "extend":
                    runtime.extend(
                        load_variant(Path(request["directory"]), expected_target=target)
                    )
                    result = None
                elif command == "close":
                    reply({"ok": None})
                    return
                else:
                    raise ValueError("Unknown runtime command")
                reply({"ok": result})
            except CapacityError as error:
                reply(
                    {
                        "error": str(error),
                        "kind": "capacity",
                        "gate": error.gate,
                        "required": error.required,
                        "capacity": error.capacity,
                    }
                )
            except (KeyError, ValueError, OSError, RuntimeError) as error:
                reply({"error": str(error), "kind": "runtime"})
    finally:
        runtime.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--target", choices=("c", "cuda", "ascend"), default="c")
    args = p.parse_args()
    serve(args.directory, sys.stdin, sys.stdout, args.target)


if __name__ == "__main__":
    main()
