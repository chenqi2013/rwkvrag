import hashlib
from pathlib import Path
import subprocess

import fitz
import pytest

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.ocr import extract_pages
from llamaindex_retrieval.parsers import parse_uploaded_file
from llamaindex_retrieval.source_revisions import prepare_revision, original_snapshot


def scanned_pdf(path):
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 80), 'Offline archive has 8 users.', fontsize=24)
    png = p.get_pixmap(dpi=150).tobytes('png')
    mixed = fitz.open()
    p = mixed.new_page()
    p.insert_text((50, 80), 'Native first page.', fontsize=20)
    p = mixed.new_page()
    p.insert_image(p.rect, stream=png)
    mixed.new_page()
    mixed.save(path)
    return png


def test_mixed_pdf_never_silently_drops_scanned_page_without_language_pack(tmp_path):
    path = tmp_path / 'mixed.pdf'
    scanned_pdf(path)
    with pytest.raises(ValueError, match='第 2 页.*语言包未配置'):
        parse_uploaded_file(path, 'f', 'kb', settings=Settings(_env_file=None))


def test_disabled_ocr_reports_required_page(tmp_path):
    path = tmp_path / 'mixed.pdf'
    scanned_pdf(path)
    with pytest.raises(ValueError, match='第 2 页.*OCR 已关闭'):
        parse_uploaded_file(path, 'f', 'kb', settings=Settings(_env_file=None, ocr_enabled=False))


def test_timeout_is_bounded_and_never_returns_partial_documents(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('ocr', 5)
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(ValueError, match='未发布部分内容'):
        extract_pages(tmp_path / 'p.pdf', Settings(_env_file=None))


def test_page_limit_fails_before_partial_index(tmp_path):
    path = tmp_path / 'mixed.pdf'
    scanned_pdf(path)
    with pytest.raises(ValueError, match='超过 1 页'):
        parse_uploaded_file(path, 'f', 'kb', settings=Settings(_env_file=None, ocr_max_pages=1))


def test_original_snapshot_preserves_old_version_and_rejects_tampering(tmp_path):
    path = tmp_path / 'original.md'
    path.write_text('# Version one')
    settings = Settings(_env_file=None, upload_dir=tmp_path / 'uploads')
    _, revision = prepare_revision(settings, path, 'f', 'kb')
    item = {'id': 'f', 'knowledge_base_id': 'kb'}
    snapshot = original_snapshot(settings, item, revision['source_sha256'])
    path.write_text('# Version two')
    prepare_revision(settings, path, 'f', 'kb')
    assert snapshot.read_text() == '# Version one'
    with pytest.raises(ValueError):
        original_snapshot(settings, item, '../../etc/passwd')
    snapshot.write_text('tampered')
    with pytest.raises(ValueError, match='完整性'):
        original_snapshot(settings, item, revision['source_sha256'])


def test_real_ocr_image_and_mixed_pdf_preserve_pages_and_word_coordinates(tmp_path):
    import os
    directory = os.environ.get('RWKVRAG_TEST_TESSDATA')
    if not directory:
        pytest.skip('set RWKVRAG_TEST_TESSDATA for real offline OCR')
    path = tmp_path / 'mixed.pdf'
    png = scanned_pdf(path)
    image = tmp_path / 'image.png'
    image.write_bytes(png)
    settings = Settings(_env_file=None, ocr_tessdata_dir=Path(directory), ocr_languages='eng',
                        upload_dir=tmp_path / 'uploads')
    docs, rev = prepare_revision(settings, path, 'f', 'kb')
    assert len(docs) == 2
    assert 'Native first page' in docs[0].text
    assert 'Offline archive has 8 users' in docs[1].text
    assert [d.metadata['page'] for d in docs] == [1, 2]
    assert rev['extraction_summary'] == {'pages': 3, 'ocr_pages': [2], 'blank_pages': [3], 'status': 'complete'}
    assert docs[1].metadata['recognized_words']
    assert docs[1].metadata['extraction_method'] == 'ocr'
    assert docs[1].metadata['extraction_engine']['language_sha256']['eng'] == hashlib.sha256((Path(directory)/'eng.traineddata').read_bytes()).hexdigest()
    parsed = parse_uploaded_file(image, 'image', 'kb', settings=settings)
    assert 'Offline archive has 8 users' in parsed[0].text
    assert parsed[0].metadata['kind'] == 'image'


def test_real_same_page_native_and_image_text_are_both_retained(tmp_path):
    import os
    directory = os.environ.get('RWKVRAG_TEST_TESSDATA')
    if not directory:
        pytest.skip('set RWKVRAG_TEST_TESSDATA for real offline OCR')
    raster = fitz.open()
    p = raster.new_page(width=400, height=120)
    p.insert_text((20, 60), 'Embedded image evidence.', fontsize=24)
    png = p.get_pixmap(dpi=180).tobytes('png')
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((40, 60), 'Native header stays intact.', fontsize=20)
    p.insert_image(fitz.Rect(40, 120, 440, 240), stream=png)
    path = tmp_path / 'same-page.pdf'
    doc.save(path)
    docs = parse_uploaded_file(path, 'f', 'kb', settings=Settings(_env_file=None,
        ocr_tessdata_dir=Path(directory), ocr_languages='eng'))
    assert 'Native header stays intact' in docs[0].text
    assert 'Embedded image evidence' in docs[0].text
    assert docs[0].text.count('Native header') == 1
    assert docs[0].metadata['extraction_method'] == 'native+ocr'
