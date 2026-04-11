"""Gateway STT config tests — honor stt.enabled: false from config.yaml."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def test_gateway_config_stt_disabled_from_dict_nested():
    config = GatewayConfig.from_dict({"stt": {"enabled": False}})
    assert config.stt_enabled is False


def test_load_gateway_config_bridges_stt_enabled_from_config_yaml(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.dump({"stt": {"enabled": False}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    config = load_gateway_config()

    assert config.stt_enabled is False


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_skips_when_stt_disabled():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=False)

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=AssertionError("transcribe_audio should not be called when STT is disabled"),
    ):
        result, transcripts = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/voice.ogg"],
        )

    assert "transcription is disabled" in result.lower()
    assert "caption" in result
    assert transcripts == []


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_avoids_bogus_no_provider_message_for_backend_key_errors():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": False, "error": "VOICE_TOOLS_OPENAI_KEY not set"},
    ):
        result, transcripts = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/voice.ogg"],
        )

    assert "No STT provider is configured" not in result
    assert "trouble transcribing" in result
    assert "caption" in result
    assert transcripts == []


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_transcribes_queued_voice_event():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False
    runner._echo_transcribed_text = AsyncMock()

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/queued-voice.ogg"],
        media_types=["audio/ogg"],
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "queued voice transcript",
            "provider": "local_command",
            "elapsed_seconds": 1.25,
        },
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "queued voice transcript" in result
    assert "voice message" in result.lower()
    runner._echo_transcribed_text.assert_awaited_once_with(
        event,
        [{"text": "queued voice transcript", "elapsed_seconds": 1.25}],
    )


@pytest.mark.asyncio
async def test_handle_message_echoes_transcript_before_running_agent(monkeypatch):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.adapters = {Platform.TELEGRAM: type("Adapter", (), {"send": AsyncMock()})()}
    runner.hooks = type("Hooks", (), {"emit": AsyncMock()})()
    runner.session_store = type("Store", (), {})()
    runner.session_store._generate_session_key = lambda source: "telegram:chat-1"
    runner.session_store.get_or_create_session = lambda source: type(
        "Entry", (), {"session_key": "telegram:chat-1", "session_id": "sid-1", "created_at": 2, "updated_at": 3}
    )()
    runner.session_store.load_transcript = lambda session_id: [{"role": "user", "content": "hi"}]
    runner.session_store.has_any_sessions = lambda: True
    runner.session_store.append_to_transcript = lambda *args, **kwargs: None
    runner.session_store.update_session = lambda *args, **kwargs: None
    runner._is_user_authorized = lambda source: True
    runner._set_session_env = lambda context: None
    runner._has_setup_skill = lambda: False
    runner._run_process_watcher = AsyncMock()
    runner._run_processing_hook = AsyncMock()
    runner._should_send_voice_reply = lambda *args, **kwargs: False
    runner._deliver_media_from_response = AsyncMock()
    runner._send_voice_reply = AsyncMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._voice_mode = {}
    runner._show_reasoning = False
    runner._session_db = None
    runner._base_url = ""
    runner._model = "test-model"

    run_agent_mock = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "last_prompt_tokens": 0,
        }
    )
    runner._run_agent = run_agent_mock

    monkeypatch.setattr("gateway.run.build_session_context", lambda *args, **kwargs: object())
    monkeypatch.setattr("gateway.run.build_session_context_prompt", lambda *args, **kwargs: "ctx")

    event = MessageEvent(
        text="",
        message_id="msg-1",
        message_type=MessageType.VOICE,
        media_urls=["/tmp/voice.ogg"],
        media_types=["audio/ogg"],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", user_id="u1", user_name="Li Ge"),
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hello world", "elapsed_seconds": 2.34},
    ):
        await runner._handle_message(event)

    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[0] == "chat-1"
    assert adapter.send.await_args.args[1] == "Transcribed (2.3s): hello world"
    assert adapter.send.await_args.kwargs["reply_to"] == "msg-1"
    assert run_agent_mock.await_count == 1
    assert "hello world" in run_agent_mock.await_args.kwargs["message"]


@pytest.mark.asyncio
async def test_handle_message_continues_when_transcript_echo_send_fails(monkeypatch):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.adapters = {Platform.TELEGRAM: type("Adapter", (), {"send": AsyncMock(side_effect=RuntimeError("network down"))})()}
    runner.hooks = type("Hooks", (), {"emit": AsyncMock()})()
    runner.session_store = type("Store", (), {})()
    runner.session_store._generate_session_key = lambda source: "telegram:chat-1"
    runner.session_store.get_or_create_session = lambda source: type(
        "Entry", (), {"session_key": "telegram:chat-1", "session_id": "sid-1", "created_at": 2, "updated_at": 3}
    )()
    runner.session_store.load_transcript = lambda session_id: [{"role": "user", "content": "hi"}]
    runner.session_store.has_any_sessions = lambda: True
    runner.session_store.append_to_transcript = lambda *args, **kwargs: None
    runner.session_store.update_session = lambda *args, **kwargs: None
    runner._is_user_authorized = lambda source: True
    runner._set_session_env = lambda context: None
    runner._has_setup_skill = lambda: False
    runner._run_process_watcher = AsyncMock()
    runner._run_processing_hook = AsyncMock()
    runner._should_send_voice_reply = lambda *args, **kwargs: False
    runner._deliver_media_from_response = AsyncMock()
    runner._send_voice_reply = AsyncMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._voice_mode = {}
    runner._show_reasoning = False
    runner._session_db = None
    runner._base_url = ""
    runner._model = "test-model"

    run_agent_mock = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "last_prompt_tokens": 0,
        }
    )
    runner._run_agent = run_agent_mock

    monkeypatch.setattr("gateway.run.build_session_context", lambda *args, **kwargs: object())
    monkeypatch.setattr("gateway.run.build_session_context_prompt", lambda *args, **kwargs: "ctx")

    event = MessageEvent(
        text="",
        message_id="msg-1",
        message_type=MessageType.VOICE,
        media_urls=["/tmp/voice.ogg"],
        media_types=["audio/ogg"],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", user_id="u1", user_name="Li Ge"),
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hello world", "elapsed_seconds": 0.8},
    ):
        await runner._handle_message(event)

    assert run_agent_mock.await_count == 1
    assert "hello world" in run_agent_mock.await_args.kwargs["message"]
