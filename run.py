"""Standalone entry point: python run.py infer|finetune [options]."""
import importlib
from pathlib import Path
import sys

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"infer", "finetune"}:
        print("Usage: python run.py infer|finetune [options]")
        print("Run either command with --help for its arguments.")
        return
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.parent))
    command = sys.argv.pop(1)
    module = "inference" if command == "infer" else "finetune"
    importlib.import_module(root.name + "." + module).main()

if __name__ == "__main__":
    main()
