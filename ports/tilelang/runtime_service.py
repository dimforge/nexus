"""JSON-line control for a resident generated-C frame session.

Only explicit read requests copy body state to the Rust caller. Physics remains
inside compiled TileLang kernels; this process contains no host physics fallback.
"""

import argparse
import json
import sys
from pathlib import Path

from frame_cpu import CapacityError, CpuFrameSession
from runtime_compile import load_variant


def serve(directory, input_stream, output_stream):
    directory = Path(directory)
    runtime = CpuFrameSession([load_variant(directory)], directory / "native-c")

    def reply(value):
        output_stream.write(json.dumps(value) + "\n")
        output_stream.flush()

    reply({"ready": True})
    for line in input_stream:
        try:
            request = json.loads(line)
            command = request["command"]
            if command == "step":
                result = runtime.step()
            elif command == "read":
                result = runtime.storage[request["buffer"]].tolist()
            elif command == "extend":
                runtime.extend(load_variant(Path(request["directory"])))
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    serve(args.directory, sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
