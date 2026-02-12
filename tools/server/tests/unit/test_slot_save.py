import pytest
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
    server.slot_save_path = "./tmp"
    server.temperature = 0.0


def test_slot_save_restore():
    global server
    server.start()

    # First prompt in slot 1 should be fully processed
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # Save state of slot 1
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84
    assert "n_checkpoints" in res.body
    assert os.path.exists("./tmp/slot1.bin.ctxchk")

    # Since we have cache, this should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # Loading the saved cache into slot 0
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84
    assert "n_checkpoints" in res.body
    assert res.body["restore_quality"] in ["full", "partial_legacy"]

    # Since we have cache, slot 0 should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # For verification that slot 1 was not corrupted during slot 0 load, same thing should work
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_restore_legacy_without_checkpoint_sidecar():
    global server
    server.start()

    # Prime slot 1 and save it with sidecar data.
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    filename = "slot-legacy.bin"
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": filename,
    })
    assert res.status_code == 200
    assert res.body["n_saved"] > 0
    assert os.path.exists(f"./tmp/{filename}.ctxchk")

    # Force legacy restore path by removing checkpoint sidecar.
    os.remove(f"./tmp/{filename}.ctxchk")
    assert not os.path.exists(f"./tmp/{filename}.ctxchk")

    # Restore should succeed with partial_legacy quality and still reuse prompt cache.
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": filename,
    })
    assert res.status_code == 200
    assert res.body["n_restored"] > 0
    assert res.body["restore_quality"] == "partial_legacy"

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > 0


def test_slot_erase():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed


def test_slot_lifecycle_strict_fails_when_restore_file_missing():
    global server

    missing_file = "./tmp/tinyllama-2.slot-0.bin"
    if os.path.exists(missing_file):
        os.remove(missing_file)

    server.slot_lifecycle = "strict"
    server.slot_lifecycle_strict_status_code = 503
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "hello",
        "id_slot": 0,
        "cache_prompt": True,
    })

    assert res.status_code == 503
    assert res.body["error"]["type"] == "unavailable_error"


def test_slot_lifecycle_conservative_emits_restore_metadata_and_skip_reason():
    global server
    server.slot_lifecycle = "conservative"
    server.slot_lifecycle_save_min_restored_tokens = 1
    server.slot_lifecycle_save_min_ratio = 0.5
    server.start()

    default_slot_file = "tinyllama-2.slot-0.bin"

    # Prime slot 0 and persist a known state under the lifecycle default filename.
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": default_slot_file,
    })
    assert res.status_code == 200
    assert res.body["n_saved"] > 0

    # Clear current in-memory slot so lifecycle restore must run on next request.
    res = server.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200

    # Use a short prompt so prompt/restored ratio falls below guard threshold and save is skipped.
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hi",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "slot_lifecycle" in res.body
    assert res.body["slot_lifecycle"]["restore_success"] is True
    assert "cache_reused_tokens" in res.body["slot_lifecycle"]
    assert "restore_effective" in res.body["slot_lifecycle"]
    assert res.body["slot_lifecycle"]["save_decision"] == "skipped_guard_low_reuse"


def test_slot_lifecycle_conservative_bootstraps_when_restore_missing():
    global server

    default_slot_file = "./tmp/tinyllama-2.slot-0.bin"
    sidecar_file = default_slot_file + ".ctxchk"
    if os.path.exists(default_slot_file):
        os.remove(default_slot_file)
    if os.path.exists(sidecar_file):
        os.remove(sidecar_file)

    server.slot_lifecycle = "conservative"
    server.slot_lifecycle_save_min_restored_tokens = 1
    server.slot_lifecycle_save_min_ratio = 0.5
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "Bootstrap state file for conservative lifecycle mode.",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "slot_lifecycle" in res.body
    assert res.body["slot_lifecycle"]["restore_success"] is False
    assert res.body["slot_lifecycle"]["restore_quality"] == "missing"
    assert res.body["slot_lifecycle"]["restore_effective"] is False
    assert res.body["slot_lifecycle"]["save_decision"] == "save_succeeded"
    assert os.path.exists(default_slot_file)


def test_slot_lifecycle_stream_includes_final_metadata():
    global server
    server.slot_lifecycle = "conservative"
    server.slot_lifecycle_save_min_restored_tokens = 1
    server.slot_lifecycle_save_min_ratio = 0.5
    server.start()

    default_slot_file = "tinyllama-2.slot-0.bin"

    # Prime and persist slot state used by automatic restore.
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": default_slot_file,
    })
    assert res.status_code == 200
    assert res.body["n_saved"] > 0

    # Erase in-memory slot so lifecycle restore must execute.
    res = server.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200

    stream_chunks = list(server.make_stream_request("POST", "/completion", data={
        "prompt": "Hi",
        "id_slot": 0,
        "cache_prompt": True,
        "stream": True,
    }))
    assert stream_chunks

    lifecycle_chunks = [c for c in stream_chunks if "slot_lifecycle" in c]
    assert lifecycle_chunks, "expected slot_lifecycle in final stream payload"
    lifecycle = lifecycle_chunks[-1]["slot_lifecycle"]
    assert lifecycle["enabled"] is True
    assert lifecycle["restore_success"] is True
    assert "cache_reused_tokens" in lifecycle
    assert "restore_effective" in lifecycle
    assert lifecycle["save_decision"] == "skipped_guard_low_reuse"
