import os
from pathlib import Path
import sys
import tempfile
import unittest

from run_tilelang_testbed import interpreter_path


class InterpreterPathTests(unittest.TestCase):
    def test_virtualenv_symlink_keeps_its_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "venv/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            selected = interpreter_path(python)
            self.assertEqual(selected, os.path.abspath(python))
            self.assertNotEqual(selected, str(python.resolve()))

    def test_missing_interpreter_fails_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                interpreter_path(Path(directory) / "missing")


if __name__ == "__main__":
    unittest.main()
