"""Install pinned offline OCR language packs; no system Tesseract executable needed.

Run with the project's Python: setup_ocr.py --output /absolute/path/to/tessdata
Then set ocr_tessdata_dir in the API settings to that path.
"""
import argparse
import hashlib
import json
from pathlib import Path

import httpx

COMMIT = '87416418657359cb625c412a48b6e1d6d41c29bd'
PINS = {
    'eng': '7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2',
    'chi_sim': 'a5fcb6f0db1e1d6d8522f39db4e848f05984669172e584e8d76b6b3141e1f730',
    'chi_tra': '529c5b5797d64b126065cd55f2bb4c7fd7b15790798091b1ff259941a829330b',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for name, digest in PINS.items():
            path = args.output / (name + '.traineddata')
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
                continue
            response = client.get(f'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/{COMMIT}/{name}.traineddata')
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != digest:
                raise ValueError(f'OCR language hash mismatch: {name}')
            temporary = path.with_suffix('.download')
            temporary.write_bytes(response.content)
            temporary.replace(path)
    (args.output / 'MANIFEST.json').write_text(json.dumps({'commit': COMMIT, 'sha256': PINS}, indent=2))
    print(json.dumps({'ocr_tessdata_dir': str(args.output.resolve()), 'ocr_languages': 'chi_sim+eng'}))


if __name__ == '__main__':
    main()
