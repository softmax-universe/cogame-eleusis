"""Complete standard native games with exact local provider evidence; no paid calls."""

import hashlib
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAME, PLAYER, OUTPUT = (Path(arg).resolve() for arg in sys.argv[1:4])
OUTPUT.mkdir(parents=True, exist_ok=False, mode=0o700)
for mode in (
    "accepted",
    "invalid",
    "double-object",
    "provider-429",
    "provider-503",
    "provider-error",
    "missing-credentials",
    "normalized",
    "applied-text",
    "scripted",
):
    if (
        os.environ.get("ELEUSIS_FIXTURE_MODE")
        and mode != os.environ["ELEUSIS_FIXTURE_MODE"]
    ):
        continue
    calls = {}

    class Provider(http.server.BaseHTTPRequestHandler):
        def do_POST(self, mode=mode, calls=calls):
            assert self.path == "/v1/messages"
            request = json.loads(self.rfile.read(int(self.headers["content-length"])))
            assert request["model"] == "fixture/pinned"
            seat = int(self.headers["X-Coworld-Player-Slot"])
            testing = request["messages"][0]["content"].startswith("PREDICTION TEST")
            action = {
                "publish": True,
                "hypothesis": "fixture",
                "notes": "private-fixture-notes",
            }
            action.update({"answers": ["pass"] * 6} if testing else {"experiment": ""})
            if mode == "normalized" and seat == 0:
                action["notes"] = "fixture-note-" * 80
                if testing:
                    action["answers"] = ["PASS"] * 6
            if mode == "applied-text" and seat == 0:
                action["hypothesis"] = "fixture\rprivate"
            if mode == "invalid" and seat == 0:
                text = "not JSON"
            elif mode == "double-object" and seat == 0:
                text = json.dumps(action) + " " + json.dumps(action)
            elif mode == "accepted" and seat == 0:
                text = "```json\n" + json.dumps(action) + "\n```"
            else:
                text = json.dumps(action)
            body = {
                "model": "fixture/served",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            status_code = 200
            if mode == "provider-error" and seat == 0:
                status_code = 500
                body = {"error": {"message": "fixture provider error"}}
            elif mode == "provider-503" and seat == 2:
                status_code = 503
                body = {"error": {"message": "fixture service unavailable"}}
            elif mode == "provider-429" and seat == 0:
                status_code = 429
                body = {"error": {"message": "fixture rate limited"}}
            call_id = str(uuid.uuid4())
            raw = json.dumps(body)
            calls[call_id] = (request, raw)
            self.send_response(status_code)
            self.send_header("content-type", "application/json")
            self.send_header("x-softmax-llm-call-id", call_id)
            self.send_header("content-length", str(len(raw.encode())))
            self.end_headers()
            self.wfile.write(raw.encode())

        def log_message(self, *_):
            pass

    provider = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    owner = threading.Thread(target=provider.serve_forever)
    owner.start()
    output = OUTPUT / mode
    output.mkdir(mode=0o700)
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    config = {
        "tokens": [f"t{s}" for s in range(5)],
        "players": [{"name": f"p{s}"} for s in range(5)],
        "seed": int(os.environ.get("ELEUSIS_FIXTURE_SEED", "7")),
        "rounds": 24,
        "testEvery": 6,
        "testStrips": 6,
        "promptVariant": "mixed",
        "promptSeed": 4,
        "minBatchSpacingMs": 0,
        "player_connect_timeout_seconds": 5,
        "llmTimeoutSeconds": 5,
    }
    if mode != "missing-credentials":
        config["hostedModel"] = "fixture/pinned"
    (output / "config.json").write_text(json.dumps(config))
    env = {
        **os.environ,
        "COGAME_HOST": "127.0.0.1",
        "COGAME_PORT": str(port),
        "COGAME_CONFIG_URI": (output / "config.json").as_uri(),
        "COGAME_RESULTS_URI": (output / "results.json").as_uri(),
        "COGAME_SAVE_REPLAY_URI": (output / "replay.json").as_uri(),
        "COGAME_SAVE_TRAJECTORY_URI": (output / "trajectory.jsonl").as_uri(),
        "COGAME_PLAYER_FAILURE_URI": (output / "player_failure.json").as_uri(),
        "COGAME_LOG_URI": (output / "diagnostic.log").as_uri(),
        "COWORLD_EPISODE_ID": str(uuid.uuid4()),
        "COWORLD_GAME_VERSION": "native-fixture",
        "COWORLD_SOURCE_REVISION": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "PLAYER_PROMPT": "fixture strategy",
        "PLAYER_SCRIPTED": "openbook" if mode == "scripted" else "",
    }
    if mode == "missing-credentials":
        for key in (
            "COWORLD_LLM_ENDPOINT",
            "COWORLD_LLM_MODEL",
            "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
            "AWS_BEARER_TOKEN_BEDROCK",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_API_KEY_URI",
        ):
            env.pop(key, None)
    else:
        env["COWORLD_LLM_ENDPOINT"] = f"http://127.0.0.1:{provider.server_port}"
        env["COWORLD_LLM_MODEL"] = "fixture/requested"
    failed_modes = {"invalid", "double-object", "provider-429", "provider-503", "provider-error", "missing-credentials"}
    player_fail_modes = {"invalid", "double-object"}
    infra_fail_modes = {"provider-429", "provider-503", "provider-error", "missing-credentials"}
    expected_fail_seat = 2 if mode == "provider-503" else 0
    processes = []
    try:
        with (output / "game.log").open("w") as log:
            processes.append(
                subprocess.Popen([GAME], cwd=ROOT, env=env, stdout=log, stderr=log)
            )
            for _ in range(200):
                assert processes[0].poll() is None, (output / "game.log").read_text()
                with socket.socket() as probe:
                    if probe.connect_ex(("127.0.0.1", port)) == 0:
                        break
                time.sleep(0.05)
            else:
                raise AssertionError("game failed to listen")
            for seat in range(5):
                player_env = {
                    **env,
                    "COWORLD_PLAYER_WS_URL": f"ws://127.0.0.1:{port}/player?slot={seat}&token=t{seat}",
                }
                processes.append(
                    subprocess.Popen(
                        [PLAYER], cwd=ROOT, env=player_env, stdout=log, stderr=log
                    )
                )
            game_rc = processes[0].wait(timeout=90)
            if mode in failed_modes:
                assert game_rc != 0, f"{mode}: expected non-zero game exit, got 0"
            else:
                assert game_rc == 0, (output / "game.log").read_text()
            for process in processes[1:]:
                assert process.wait(timeout=90) == 0, (output / "game.log").read_text()
        events = [
            json.loads(line)
            for line in (output / "trajectory.jsonl").read_text().splitlines()
        ]
        results_path = output / "results.json"
        replay_path = output / "replay.json"
        failure_path = output / "player_failure.json"
        diag_path = output / "diagnostic.log"
        if mode in failed_modes:
            assert not results_path.exists(), f"{mode}: results.json must not be written on failure"
            assert not replay_path.exists(), f"{mode}: replay.json must not be written on failure"
            assert diag_path.exists(), f"{mode}: diagnostic log missing"
            diag = json.loads(diag_path.read_text())
            assert diag["status"] == "failed"
            assert "scores" not in diag
            assert len(events) == 6 and events[-1]["status"] == "failed"
            assert events[-1]["participant_outcomes"] == {}
            assert "scores" not in events[-1]["outcome"]
            assert events[-1]["outcome"]["rounds"] == 0
            assert failure_path.exists(), f"{mode}: COGAME_PLAYER_FAILURE_URI must be written on failure"
            failure_info = json.loads(failure_path.read_text())
            assert set(failure_info.keys()) == {"failed_policy_index", "message", "kind"}
            assert failure_info["failed_policy_index"] == expected_fail_seat
            assert isinstance(failure_info["message"], str) and 1 <= len(failure_info["message"]) <= 2000
            if mode in player_fail_modes:
                assert events[-1]["outcome"]["failureKind"] == "player"
                assert failure_info["kind"] == "player_error"
            else:
                assert events[-1]["outcome"]["failureKind"] == "infra"
                assert failure_info["kind"] == "episode_inconclusive"
                if mode == "provider-429":
                    assert "429" in diag["reason"] and "429" in failure_info["message"]
                elif mode == "provider-503":
                    assert "503" in diag["reason"] and "503" in failure_info["message"]
                elif mode == "provider-error":
                    assert "500" in diag["reason"] and "500" in failure_info["message"]
                elif mode == "missing-credentials":
                    assert "credentials" in diag["reason"] and "credentials" in failure_info["message"]
        else:
            assert not failure_path.exists()
            assert results_path.exists() and replay_path.exists()
            results_data = json.loads(results_path.read_text())
            assert results_data["reason"] == "complete"
            assert len(results_data["scores"]) == 5
            replay_data = json.loads(replay_path.read_text())
            expected_family = f"eleusis-rule-{replay_data['config']['ruleId']}"
            assert events[-1]["seed_family"] == expected_family
            assert len(events) == 141 and events[-1]["status"] == "completed"
            assert (
                events[-1]["outcome"]["rounds"] == 24
                and events[-1]["outcome"]["tests"] == 4
            )
            for start in range(0, 140, 5):
                batch = events[start : start + 5]
                for key in ("board", "seats", "test_strips"):
                    assert all(
                        event["observation"][key] == batch[0]["observation"][key]
                        for event in batch
                    )
        joined = set()
        normalized_proposals = 0
        applied_text_changes = 0
        if mode not in failed_modes:
            replay_data = json.loads(replay_path.read_text())
            native_actions = [
                event
                for event in replay_data["events"]
                if event["kind"] in {"skip", "experiment", "answer"}
            ]
            assert len(native_actions) == len(events) - 1
            for event, native_action in zip(events[:-1], native_actions, strict=True):
                action = event["executed_action"]
                if action["hypothesis"]:
                    assert action["hypothesis"] == native_action.get("hypothesis", "")
                if action["notes"]:
                    assert action["notes"] == native_action.get("text", "")
                if "answers" in action:
                    assert action["answers"] == native_action["answers"]
                else:
                    assert action["experiment"] == native_action.get("strip", "")
        for event in events[:-1]:
            assert event["visibility"] == "private"
            assert (
                "rule" not in event["observation"]
                and "ruleId" not in event["observation"]
            )
            for attempt in event["attempts"]:
                if mode == "scripted":
                    assert (
                        attempt["origin"] == "teacher"
                        and attempt["policy"] == "openbook"
                    )
                    assert (
                        attempt["platform_call_id"] is None
                        and attempt["request"] is None
                    )
                    assert attempt["parsed_action"] == event["executed_action"]
                    continue
                call_id = attempt["platform_call_id"]
                request, raw = calls[call_id]
                joined.add(call_id)
                assert attempt["request"] == request and attempt["raw_response"] == raw
                assert attempt["decoder"] == {"max_tokens": request["max_tokens"]}
                metadata = next(
                    item["prompt_template"]
                    for item in event["observation"]["prompt_attempts"]
                    if item["attempt_id"] == attempt["attempt_id"]
                )
                assert metadata["variant"] == "reordered"
                assert (
                    metadata["system_sha256"]
                    == hashlib.sha256(request["system"].encode()).hexdigest()
                )
                assert (
                    metadata["user_sha256"]
                    == hashlib.sha256(
                        request["messages"][0]["content"].encode()
                    ).hexdigest()
                )
                if mode == "applied-text" and event["seat"] == "0":
                    assert not attempt["accepted"]
                    assert attempt["parsed_action"]["hypothesis"] == "fixture\rprivate"
                    assert event["executed_action"]["hypothesis"] == "fixture private"
                    assert (
                        attempt["rejection_reason"]
                        == "simulator normalized the applied action"
                    )
                    applied_text_changes += 1
                elif attempt["accepted"]:
                    assert attempt["parsed_action"] == event["executed_action"]
                    raw_text = attempt["response"]
                    if raw_text.startswith("```"):
                        raw_text = raw_text.split("\n", 1)[1].rsplit("\n```", 1)[0]
                    if json.loads(raw_text) != attempt["parsed_action"]:
                        normalized_proposals += 1
                        assert mode == "normalized" and event["seat"] == "0"
                else:
                    assert attempt["rejection_reason"]
            if mode in player_fail_modes:
                assert event["selected_attempt_id"] is None
                assert event["executed_action"] is None
                assert event["fallback_origin"] is None
                if event["seat"] == str(expected_fail_seat):
                    assert event["action_status"] == "rejected"
                    assert len(event["attempts"]) == 1
                else:
                    assert event["action_status"] == "missing"
            elif mode in infra_fail_modes:
                assert event["selected_attempt_id"] is None
                assert event["executed_action"] is None
                assert event["fallback_origin"] is None
                assert event["action_status"] == "missing"
                if mode != "missing-credentials" and event["seat"] == str(expected_fail_seat):
                    assert len(event["attempts"]) == 1
            elif mode == "applied-text" and event["seat"] == "0":
                assert event["action_status"] == "rejected"
                assert event["selected_attempt_id"] is None
                assert event["fallback_origin"] is None
            else:
                assert event["action_status"] == "accepted"
        assert joined == set(calls)
        if mode not in failed_modes:
            replay = replay_path.read_text()
            assert "platform_call_id" not in replay and "prompt_attempts" not in replay
        assert normalized_proposals == (28 if mode == "normalized" else 0)
        assert applied_text_changes == (28 if mode == "applied-text" else 0)
        proof = {
            "mode": mode,
            "requested_seed": config["seed"],
            "scenario_family": events[-1]["seed_family"],
            "decisions": len(events) - 1,
            "provider_calls": len(joined),
            "normalized_proposals": normalized_proposals,
            "applied_text_changes": applied_text_changes,
            "rejected_decisions": sum(
                e["action_status"] == "rejected" for e in events[:-1]
            ),
            "fallback_decisions": sum(
                e["action_status"] == "fallback" for e in events[:-1]
            ),
            "outcome": events[-1]["outcome"],
            "model_calls_are_local_fixtures": True,
            "game_sha256": hashlib.sha256(GAME.read_bytes()).hexdigest(),
            "player_sha256": hashlib.sha256(PLAYER.read_bytes()).hexdigest(),
        }
        (output / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
        print(
            mode,
            len(events) - 1,
            "decisions",
            len(joined),
            "exact provider joins",
            events[-1]["seed_family"],
            flush=True,
        )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        provider.shutdown()
        owner.join(timeout=5)
        provider.server_close()
