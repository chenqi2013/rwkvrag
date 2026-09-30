"""Bounded extraction in a separate process, leaving old published indexes intact."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def extract_pages(path, settings):
    options = {'enabled': settings.ocr_enabled, 'tessdata': str(settings.ocr_tessdata_dir or ''),
               'languages': settings.ocr_languages, 'dpi': settings.ocr_dpi,
               'max_pages': settings.ocr_max_pages, 'max_pixels': settings.ocr_max_pixels}
    with TemporaryDirectory(prefix='rwkvrag-ocr-') as work:
        work = Path(work)
        options['progress_file'] = str(work / 'page')
        config, output = work / 'config.json', work / 'result.json'
        config.write_text(json.dumps(options))
        env = {**os.environ, 'OMP_THREAD_LIMIT': '2'}
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('ocr_worker.py')),
                                     str(path), str(config), str(output)],
                                    capture_output=True, timeout=settings.ocr_timeout_seconds, env=env)
        except subprocess.TimeoutExpired as exc:
            page = (work / 'page').read_text() if (work / 'page').exists() else '?'
            raise ValueError(f'文档解析超时（第 {page} 页），未发布部分内容；原有知识保持不变') from exc
        data = json.loads(output.read_text()) if output.exists() else {}
        if result.returncode or 'error' in data:
            raise ValueError(data.get('error', '文档解析进程异常退出，未发布部分内容'))
        return data
