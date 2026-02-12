import pytest
from utils import *
import tempfile

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()


def _get_router_models() -> list[str]:
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    data = res.body.get("data", [])
    models: list[str] = []
    for item in data:
        model_id = item.get("id") or item.get("model")
        if isinstance(model_id, str) and model_id:
            models.append(model_id)
    return models


def _require_router_model() -> str:
    models = _get_router_models()
    if not models:
        pytest.skip("router model registry is empty in this environment")
    return models[0]


@pytest.mark.parametrize(
    "requested_model,success",
    [
        ("__AUTO__", True),
        ("non-existent/model", False),
    ]
)
def test_router_chat_completion_stream(requested_model: str, success: bool):
    global server
    server.start()
    model = _require_router_model() if requested_model == "__AUTO__" else requested_model
    content = ""
    ex: ServerError | None = None
    try:
        res = server.make_stream_request("POST", "/chat/completions", data={
            "model": model,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hello"},
            ],
            "stream": True,
        })
        for data in res:
            if data["choices"]:
                choice = data["choices"][0]
                if choice["finish_reason"] in ["stop", "length"]:
                    assert "content" not in choice["delta"]
                else:
                    assert choice["finish_reason"] is None
                    content += choice["delta"]["content"] or ''
    except ServerError as e:
        ex = e

    if success:
        assert ex is None
        assert len(content) > 0
    else:
        assert ex is not None
        assert content == ""


def _get_model_status(model_id: str) -> str:
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    for item in res.body.get("data", []):
        if item.get("id") == model_id or item.get("model") == model_id:
            return item["status"]["value"]
    raise AssertionError(f"Model {model_id} not found in /models response")


def _wait_for_model_status(model_id: str, desired: set[str], timeout: int = 60) -> str:
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        last_status = _get_model_status(model_id)
        if last_status in desired:
            return last_status
        time.sleep(1)
    raise AssertionError(
        f"Timed out waiting for {model_id} to reach {desired}, last status: {last_status}"
    )


def _load_model_and_wait(
    model_id: str, timeout: int = 60, headers: dict | None = None
) -> None:
    load_res = server.make_request(
        "POST", "/models/load", data={"model": model_id}, headers=headers
    )
    assert load_res.status_code == 200
    assert isinstance(load_res.body, dict)
    assert load_res.body.get("success") is True
    _wait_for_model_status(model_id, {"loaded"}, timeout=timeout)


def test_router_unload_model():
    global server
    server.start()
    model_id = _require_router_model()

    _load_model_and_wait(model_id)

    unload_res = server.make_request("POST", "/models/unload", data={"model": model_id})
    assert unload_res.status_code == 200
    assert unload_res.body.get("success") is True
    _wait_for_model_status(model_id, {"unloaded"})


def test_router_models_max_evicts_lru():
    global server
    server.models_max = 2
    server.start()

    candidate_models = _get_router_models()
    if len(candidate_models) < 3:
        pytest.skip("need at least three router models to validate LRU eviction")

    # Load only the first 2 models to fill the cache
    first, second, third = candidate_models[:3]

    _load_model_and_wait(first, timeout=120)
    _load_model_and_wait(second, timeout=120)

    # Verify both models are loaded
    assert _get_model_status(first) == "loaded"
    assert _get_model_status(second) == "loaded"

    # Load the third model - this should trigger LRU eviction of the first model
    _load_model_and_wait(third, timeout=120)

    # Verify eviction: third is loaded, first was evicted
    assert _get_model_status(third) == "loaded"
    assert _get_model_status(first) == "unloaded"


def test_router_no_models_autoload():
    global server
    server.no_models_autoload = True
    server.start()
    model_id = _require_router_model()

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 400
    assert "error" in res.body

    _load_model_and_wait(model_id)

    success_res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert success_res.status_code == 200
    assert "error" not in success_res.body


def test_router_default_slot_lifecycle_mode_is_conservative():
    global server
    server.start()

    model_id = _require_router_model()

    _load_model_and_wait(model_id)

    props = server.make_request("GET", f"/props?model={model_id}")
    assert props.status_code == 200
    assert props.body["slot_lifecycle_mode"] == "conservative"


def test_router_explicit_slot_lifecycle_off_is_propagated():
    global server
    server.slot_lifecycle = "off"
    server.start()

    model_id = _require_router_model()
    _load_model_and_wait(model_id)

    props = server.make_request("GET", f"/props?model={model_id}")
    assert props.status_code == 200
    assert props.body["slot_lifecycle_mode"] == "off"


def test_router_api_key_required():
    global server
    server.api_key = "sk-router-secret"
    server.start()

    model_id = _require_router_model()
    auth_headers = {"Authorization": f"Bearer {server.api_key}"}

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 401
    assert res.body.get("error", {}).get("type") == "authentication_error"

    _load_model_and_wait(model_id, headers=auth_headers)

    authed = server.make_request(
        "POST",
        "/v1/chat/completions",
        headers=auth_headers,
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert authed.status_code == 200
    assert "error" not in authed.body


def test_router_lifecycle_auto_restore_after_idle_unload():
    global server

    with tempfile.TemporaryDirectory() as slot_dir:
        server.slot_lifecycle = "conservative"
        server.slot_save_path = slot_dir
        server.sleep_idle_seconds = 1
        server.start()
        model_id = _require_router_model()

        warmup = server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model_id,
                "messages": [{"role": "user", "content": "Write one short sentence about caching."}],
                "max_tokens": 8,
                "stream": False,
                "cache_prompt": True,
            },
        )
        assert warmup.status_code == 200

        # Let the child model go idle and unload, and allow async save to complete.
        time.sleep(3)

        second = server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model_id,
                "messages": [{"role": "user", "content": "Write one short sentence about caching."}],
                "max_tokens": 8,
                "stream": False,
                "cache_prompt": True,
            },
        )
        assert second.status_code == 200
        lifecycle = second.body.get("slot_lifecycle")
        assert isinstance(lifecycle, dict)
        assert lifecycle.get("restore_success") is True
