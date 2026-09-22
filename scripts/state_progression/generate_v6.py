"""Run the V6 repair supplement with its separately frozen prompts and jobs."""
import argparse
import asyncio
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'state_progression_generate_base', Path(__file__).with_name('generate.py'))
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
generator.FROZEN = ROOT / 'llamaindex-retrieval/statetune/progression-repair-v6-20260922'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--stop', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--credentials', type=Path,
                        default=Path.home() / '.config/rwkvrag/teacher-deepseek-20260922.json')
    asyncio.run(generator.main(parser.parse_args()))
