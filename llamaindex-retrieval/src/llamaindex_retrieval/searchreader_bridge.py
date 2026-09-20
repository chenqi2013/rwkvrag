"""Executed by the configured SearchReader virtualenv; stdout is JSON only.

Reuse search and page acquisition, never the other project's generated answer.
Credentials stay in that project's existing configuration and this process.
"""
import contextlib
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic
from uuid import uuid4


def route(root, request):
    from rwkv_search_reader.config import get_settings
    from rwkv_search_reader.routing import build_decision_prompt
    from rwkv_search_reader.model_client import build_completion_prompt
    import requests

    settings = get_settings()
    base_url = request.get("router_base_url") or settings.router_base_url
    model = request.get("router_model") or settings.router_model
    if not base_url or not model:
        raise RuntimeError("router_not_configured")
    # Exactly the deployed StateTune training/serving prompt; no routing keywords here.
    prompt = build_decision_prompt(request["messages"])
    if settings.router_protocol == "completions":
        body = {"model": model, "prompt": build_completion_prompt("", prompt),
            "stream": False, "max_tokens": 8, "temperature": 0.0,
            "stop": ["\nUser:", "\nSystem:", "\nAssistant:"]}
        endpoint = "/completions"
    elif settings.router_protocol == "chat":
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "max_tokens": 8, "temperature": 0.0}
        endpoint = "/chat/completions"
    else:
        raise ValueError("invalid_router_protocol")
    started = monotonic()
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(base_url.rstrip("/") + endpoint, json=body,
            headers={"Authorization": "Bearer " + settings.router_api_key},
            timeout=(5, settings.router_timeout_seconds))
        response.raise_for_status()
        payload = response.json()
    choice = payload["choices"][0]
    raw = choice.get("text") if endpoint == "/completions" else choice["message"]["content"]
    parsed = raw.strip() if isinstance(raw, str) else ""
    return {"status": "completed" if parsed in {"true", "false"} else "invalid_decision",
        "needs_search": parsed == "true" if parsed in {"true", "false"} else None,
        "stage": "routing", "call_id": str(uuid4()), "evidence_ids": [],
        "model": model, "request": body, "response": payload,
        "raw_text": raw, "prompt_sha256": hashlib.sha256(body.get("prompt", prompt).encode()).hexdigest(),
        "request_payload_sha256": hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest(),
        "raw_text_sha256": hashlib.sha256((raw if isinstance(raw, str) else json.dumps(raw)).encode()).hexdigest(),
        "elapsed_ms": round((monotonic() - started) * 1000),
        "source_hashes": {name: hashlib.sha256((root / "src/rwkv_search_reader" / name).read_bytes()).hexdigest()
                          for name in ("routing.py", "model_client.py")}}


def main():
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root / "src"))
    request = json.load(sys.stdin)
    if request.get("action") == "route":
        with contextlib.redirect_stdout(sys.stderr):
            output = route(root, request)
        print(json.dumps(output, ensure_ascii=False))
        return
    with contextlib.redirect_stdout(sys.stderr):
        from rwkv_search_reader.config import get_settings
        from rwkv_search_reader.retrieval import fetch_page, normalize_url
        from rwkv_search_reader.search import create_search_provider

        settings = replace(get_settings(), max_evidence_chars=request["material_characters"])
        provider = create_search_provider(settings)
        try:
            hits = provider.search(request["query"], request["max_results"])
            materials = []
            for hit in hits[:request["max_results"]]:
                hit.url = normalize_url(hit.url)
                if not hit.url:
                    continue
                if not provider.results_are_material:
                    hit = fetch_page(hit, settings)
                # Retain full provider material separately from bounded reader input.
                snapshot = hit.content_markdown or hit.content or hit.snippet
                text = (hit.content or hit.snippet)[:request["material_characters"]]
                if not text.strip():
                    continue
                value = asdict(hit)
                # Provider error strings may contain transport credentials.
                value["error"] = "page_fetch_failed" if hit.error else ""
                value.update(snapshot=snapshot, reader_text=text,
                    material_limited=len(text) < len(snapshot))
                materials.append(value)
            versions = {name: hashlib.sha256((root / "src/rwkv_search_reader" / name).read_bytes()).hexdigest()
                        for name in ("config.py", "search.py", "retrieval.py", "models.py")}
            output = {"provider": provider.name, "hits": materials, "source_hashes": versions}
        finally:
            provider.session.close()
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Never return upstream response bodies, key pools or exception messages.
        print(json.dumps({"error": type(error).__name__}))
        sys.exit(1)
