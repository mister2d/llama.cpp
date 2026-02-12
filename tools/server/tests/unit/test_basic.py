import pytest
import requests
import os
from utils import *

server = ServerPreset.tinyllama2()

def _skip_if_offline_tinyllama_missing() -> None:
    if os.environ.get("SKIP_SERVER_PRESET_PRELOAD", "").lower() not in {"1", "true", "yes"}:
        return
    model_path = "./tmp/ggml-org_test-model-stories260K_stories260K-f32.gguf"
    if not os.path.exists(model_path):
        pytest.skip(
            "tinyllama test model not cached locally; unset SKIP_SERVER_PRESET_PRELOAD or preload presets first"
        )


@pytest.fixture(autouse=True)
def create_server():
    global server
    _skip_if_offline_tinyllama_missing()
    server = ServerPreset.tinyllama2()


def test_server_start_simple():
    global server
    server.start()
    res = server.make_request("GET", "/health")
    assert res.status_code == 200


def test_server_props():
    global server
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert ".gguf" in res.body["model_path"]
    assert res.body["total_slots"] == server.n_slots
    default_val = res.body["default_generation_settings"]
    assert server.n_ctx is not None and server.n_slots is not None
    assert default_val["n_ctx"] == server.n_ctx / server.n_slots
    assert default_val["params"]["seed"] == server.seed


def test_server_models():
    global server
    server.start()
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    assert res.body["data"][0]["id"] == server.model_alias


def test_server_slots():
    global server

    # without slots endpoint enabled, this should return error
    server.server_slots = False
    server.start()
    res = server.make_request("GET", "/slots")
    assert res.status_code == 501 # ERROR_TYPE_NOT_SUPPORTED
    assert "error" in res.body
    server.stop()

    # with slots endpoint enabled, this should return slots info
    server.server_slots = True
    server.n_slots = 2
    server.start()
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    assert len(res.body) == server.n_slots
    assert server.n_ctx is not None and server.n_slots is not None
    assert res.body[0]["n_ctx"] == server.n_ctx / server.n_slots
    assert "params" not in res.body[0]
    assert "lifecycle" in res.body[0]

    res_diag = server.make_request("GET", "/slots?diagnostics=1")
    assert res_diag.status_code == 200
    assert "slots" in res_diag.body
    assert "diagnostics" in res_diag.body
    assert "n_slot_restore_total" in res_diag.body["diagnostics"]


def test_server_metrics_include_lifecycle_counters():
    global server
    server.server_metrics = True
    server.slot_save_path = "./tmp"
    server.start()

    # Create some slot activity so lifecycle counters are emitted with non-zero values.
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "metrics-slot.bin"})
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "metrics-slot.bin"})
    assert res.status_code == 200

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert isinstance(metrics.body, str)
    assert "llamacpp:slot_save_total" in metrics.body
    assert "llamacpp:slot_restore_total" in metrics.body
    assert "llamacpp:slot_restore_full_total" in metrics.body


def test_load_split_model():
    global server
    if os.environ.get("SKIP_SERVER_PRESET_PRELOAD", "").lower() in {"1", "true", "yes"}:
        pytest.skip("split-model download test requires preset preload/network access")
    server.offline = False
    server.model_hf_repo = "ggml-org/models"
    server.model_hf_file = "tinyllamas/split/stories15M-q8_0-00001-of-00003.gguf"
    server.model_alias = "tinyllama-split"
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 16,
        "prompt": "Hello",
        "temperature": 0.0,
    })
    assert res.status_code == 200
    assert match_regex("(little|girl)+", res.body["content"])


def test_no_webui():
    global server
    # default: webui enabled
    server.start()
    url = f"http://{server.server_host}:{server.server_port}"
    res = requests.get(url)
    assert res.status_code == 200
    assert "<!doctype html>" in res.text
    server.stop()

    # with --no-webui
    server.no_webui = True
    server.start()
    res = requests.get(url)
    assert res.status_code == 404
