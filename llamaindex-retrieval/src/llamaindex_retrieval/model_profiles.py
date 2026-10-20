"""Resolve a trial's model identity, endpoint and measured limits in one place."""
from copy import deepcopy
from pathlib import Path


def resolve(config):
    result = deepcopy(config)
    name = result.get('active_model')
    if name is None:
        return result
    profile = result.get('model_profiles', {}).get(name)
    fields = {'model', 'base_url', 'context_window_tokens', 'concurrency'}
    optional = {'api_key_file', 'require_fp32_runtime', 'tokenizer_file', 'tokenizer_sha256'}
    if not isinstance(profile, dict) or not fields <= set(profile) or set(profile) - fields - optional:
        raise ValueError('model profile requires model, base_url, context_window_tokens, concurrency')
    if (not isinstance(profile['model'], str) or not profile['model']
            or not isinstance(profile['base_url'], str) or not profile['base_url'].startswith(('http://', 'https://'))
            or type(profile['context_window_tokens']) is not int or profile['context_window_tokens'] < 1024
            or type(profile['concurrency']) is not int or not 1 <= profile['concurrency'] <= 256):
        raise ValueError('invalid model profile')
    result['settings'].update(native_model=profile['model'], native_base_url=profile['base_url'],
                              native_context_window_tokens=profile['context_window_tokens'],
                              native_max_concurrency=profile['concurrency'])
    if 'api_key_file' in profile:
        path = profile['api_key_file']
        if path is not None and (not isinstance(path, str) or not Path(path).is_absolute()):
            raise ValueError('profile api_key_file requires an absolute path or null')
        result['settings']['native_api_key'] = Path(path).read_text().strip() if path else ''
    if 'require_fp32_runtime' in profile:
        if type(profile['require_fp32_runtime']) is not bool:
            raise ValueError('profile require_fp32_runtime must be a boolean')
        result['settings']['native_require_fp32_runtime'] = profile['require_fp32_runtime']
    if 'tokenizer_file' in profile or 'tokenizer_sha256' in profile:
        if not {'tokenizer_file', 'tokenizer_sha256'} <= set(profile):
            raise ValueError('profile tokenizer requires file and hash together')
        result['settings'].update(native_tokenizer_file=profile['tokenizer_file'],
                                  native_tokenizer_sha256=profile['tokenizer_sha256'])
    return result
