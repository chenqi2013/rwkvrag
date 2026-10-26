"""Studio assets and advertised capabilities; no model inference."""
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from llamaindex_retrieval import interactive_app, model_profiles
from llamaindex_retrieval.config import Settings


async def test_studio_is_bundled_and_served_without_admin_build():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=interactive_app.app),
                                base_url='http://test') as client:
        page = await client.get('/studio/')
        assert page.status_code == 200
        assert './app.js' in page.text and './style.css' in page.text
        for asset in ['app.js', 'style.css']:
            response = await client.get('/studio/' + asset)
            assert response.status_code == 200 and response.content
        assert (await client.get('/studio/missing.json')).status_code == 404


@pytest.mark.parametrize('enabled', [False, True])
async def test_meta_advertises_observation_capability_without_quality_claim(monkeypatch, enabled):
    async def knowledge_bases():
        return []
    trial = SimpleNamespace(settings=Settings(_env_file=None),
                            config={'gpu': None, 'live_progress': enabled},
                            knowledge_bases=knowledge_bases, model_readiness='available')
    monkeypatch.setattr(interactive_app.app.state, 'trial', trial, raising=False)
    response = await interactive_app.meta()
    assert response['live_progress'] is enabled
    assert response['quality_verified'] is False


@pytest.mark.parametrize('concurrency', [1, 32, 544, 1024])
def test_profile_and_settings_accept_explicit_instance_pool_capacity(concurrency):
    config = model_profiles.resolve({'settings': {}, 'active_model': 'pool', 'model_profiles': {
        'pool': {'model': 'rwkv-test', 'base_url': 'http://127.0.0.1:9000/v1',
                 'context_window_tokens': 16384, 'concurrency': concurrency}}})
    assert Settings(_env_file=None, **config['settings']).native_max_concurrency == concurrency
    assert Settings(_env_file=None).native_max_concurrency == 32


@pytest.mark.parametrize('concurrency', [0, 1025])
def test_settings_reject_capacity_outside_supported_range(concurrency):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, native_max_concurrency=concurrency)
