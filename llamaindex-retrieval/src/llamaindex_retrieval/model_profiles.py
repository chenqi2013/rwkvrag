"""Resolve a trial's model identity, endpoint and measured limits in one place."""
from copy import deepcopy


def resolve(config):
    result = deepcopy(config)
    name = result.get('active_model')
    if name is None:
        return result
    profile = result.get('model_profiles', {}).get(name)
    fields = {'model', 'base_url', 'context_window_tokens', 'concurrency'}
    if not isinstance(profile, dict) or set(profile) != fields:
        raise ValueError('model profile requires model, base_url, context_window_tokens, concurrency')
    if (not isinstance(profile['model'], str) or not profile['model']
            or not isinstance(profile['base_url'], str) or not profile['base_url'].startswith(('http://', 'https://'))
            or type(profile['context_window_tokens']) is not int or profile['context_window_tokens'] < 1024
            or type(profile['concurrency']) is not int or not 1 <= profile['concurrency'] <= 256):
        raise ValueError('invalid model profile')
    result['settings'].update(native_model=profile['model'], native_base_url=profile['base_url'],
                              native_context_window_tokens=profile['context_window_tokens'],
                              native_max_concurrency=profile['concurrency'])
    return result
