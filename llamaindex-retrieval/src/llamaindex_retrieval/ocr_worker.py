"""Isolated CPU document extraction. No network or answer-model calls."""
import hashlib
import json
from pathlib import Path
import sys

import fitz


def extract(path, options):
    source = fitz.open(path)
    if not source.is_pdf:
        pdf_bytes = source.convert_to_pdf()
        source.close()
        source = fitz.open(stream=pdf_bytes, filetype="pdf")
    with source:
        if source.needs_pass:
            raise ValueError("加密文档需要先解密")
        if len(source) > options['max_pages']:
            raise ValueError(f"文档超过 {options['max_pages']} 页解析上限")
        pages = []
        model_hashes = None
        for i, page in enumerate(source):
            number = i + 1
            Path(options['progress_file']).write_text(str(number))
            native = page.get_text('text', sort=True)
            images = bool(page.get_image_info())
            drawings = bool(page.get_drawings()) if not native.strip() else False
            needed = images or '\ufffd' in native or (not native.strip() and drawings)
            if not native.strip() and not needed:
                pages.append({'page': number, 'status': 'blank', 'text': '', 'method': 'native'})
                continue
            try:
                tp = None
                if needed:
                    if not options['enabled']:
                        raise ValueError("此页需要 OCR，但 OCR 已关闭")
                    if not model_hashes:
                        folder = Path(options['tessdata']) if options['tessdata'] else None
                        if folder is None:
                            raise ValueError("OCR 语言包未配置，请先安装语言包并配置 ocr_tessdata_dir")
                        model_hashes = {}
                        for lang in options['languages'].split('+'):
                            file = folder / (lang + '.traineddata')
                            if not file.is_file():
                                raise ValueError(f"缺少 OCR 语言包：{lang}")
                            model_hashes[lang] = hashlib.sha256(file.read_bytes()).hexdigest()
                    pixels = page.rect.width * page.rect.height * (options['dpi'] / 72) ** 2
                    if pixels > options['max_pixels']:
                        raise ValueError("页面渲染尺寸超过 OCR 上限")
                    tp = page.get_textpage_ocr(language=options['languages'], dpi=options['dpi'],
                                               full=False, tessdata=options['tessdata'])
                text = page.get_text('text', sort=True, textpage=tp)
                if not text.strip():
                    raise ValueError("可见页面未识别到文字；请检查扫描质量或文档是否只有图片")
                words = page.get_text('words', sort=True, textpage=tp)
                pages.append({'page': number, 'status': 'complete', 'text': text,
                              'method': ('native+ocr' if native.strip() else 'ocr') if needed else 'native',
                              'native_text': native, 'page_rect': list(page.rect),
                              'words': [{'bbox': list(w[:4]), 'text': w[4]} for w in words]})
            except Exception as exc:
                raise ValueError(f"第 {number} 页解析失败：{exc}") from exc
        return {'schema': 'rwkvrag-page-extraction-v1', 'pages': pages,
                'engine': 'PyMuPDF/Tesseract', 'pymupdf_version': fitz.VersionBind,
                'languages': options['languages'], 'language_sha256': model_hashes or {}}


if __name__ == '__main__':
    try:
        result = extract(sys.argv[1], json.loads(Path(sys.argv[2]).read_text()))
    except Exception as error:
        Path(sys.argv[3]).write_text(json.dumps({'error': str(error)}, ensure_ascii=False))
        sys.exit(1)
    Path(sys.argv[3]).write_text(json.dumps(result, ensure_ascii=False))
