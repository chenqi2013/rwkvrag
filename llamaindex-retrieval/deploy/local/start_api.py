"""Launch the local API from project sources and an explicit JSON settings file."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=ROOT/"data/services/local-app/settings.json")
    parser.add_argument("--port", type=int, default=18440)
    args = parser.parse_args()
    # Explicit deployment settings take precedence over unrelated shell settings.
    env = {k: v for k, v in os.environ.items() if not k.startswith("RWKVRAG_")}
    env["RWKVRAG_SETTINGS_FILE"] = str(args.settings.resolve(strict=True))
    env["PYTHONPATH"] = str(ROOT/"llamaindex-retrieval/src")
    python = str(ROOT/"llamaindex-retrieval/.venv/bin/python")
    os.execve(python, [python, "-m", "uvicorn", "llamaindex_retrieval.api:app", "--host", "127.0.0.1", "--port", str(args.port)], env)

if __name__ == "__main__":
    main()
