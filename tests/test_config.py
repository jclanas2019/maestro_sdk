"""tests/test_config.py — maestro.config tests"""
import sys, os, tempfile, io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from pathlib import Path
from unittest.mock import patch

import maestro
from maestro.config import (
    load_env, get_config, MaestroConfig,
    global_observer, reset_global_observer, configure_logging,
    make_anthropic_from_config, make_openai_from_config,
    _parse_dotenv,
)


# ════════════════════════════════════════════════════════════════════════════
#  _parse_dotenv — internal parser
# ════════════════════════════════════════════════════════════════════════════

class TestParseDotenv:
    def _write(self, content, suffix=".env"):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
        f.write(content); f.flush(); f.close()
        return f.name

    def test_simple_key_value(self):
        p = self._write("FOO=bar\nBAZ=qux\n")
        assert _parse_dotenv(p) == {"FOO": "bar", "BAZ": "qux"}
        os.unlink(p)

    def test_strips_comments(self):
        p = self._write("# comment\nKEY=value\n# another\n")
        assert _parse_dotenv(p) == {"KEY": "value"}
        os.unlink(p)

    def test_blank_lines_skipped(self):
        p = self._write("\n\nKEY=val\n\n")
        assert _parse_dotenv(p) == {"KEY": "val"}
        os.unlink(p)

    def test_double_quoted_value(self):
        p = self._write('KEY="hello world"\n')
        assert _parse_dotenv(p)["KEY"] == "hello world"
        os.unlink(p)

    def test_single_quoted_value(self):
        p = self._write("KEY='hello world'\n")
        assert _parse_dotenv(p)["KEY"] == "hello world"
        os.unlink(p)

    def test_export_prefix_stripped(self):
        p = self._write("export KEY=value\n")
        assert _parse_dotenv(p)["KEY"] == "value"
        os.unlink(p)

    def test_empty_value(self):
        p = self._write("KEY=\n")
        assert _parse_dotenv(p)["KEY"] == ""
        os.unlink(p)

    def test_inline_comment_on_unquoted_value(self):
        p = self._write("KEY=value # this is a comment\n")
        assert _parse_dotenv(p)["KEY"] == "value"
        os.unlink(p)

    def test_inline_comment_NOT_stripped_in_quoted_value(self):
        p = self._write('KEY="value # not a comment"\n')
        assert _parse_dotenv(p)["KEY"] == "value # not a comment"
        os.unlink(p)

    def test_missing_file_returns_empty(self):
        assert _parse_dotenv("/nonexistent/.env.xyz") == {}

    def test_equals_in_value(self):
        p = self._write("KEY=base64==\n")
        assert _parse_dotenv(p)["KEY"] == "base64=="
        os.unlink(p)

    def test_multiple_keys(self):
        content = (
            "ANTHROPIC_API_KEY=sk-ant-abc\n"
            "OPENAI_API_KEY=sk-oai-def\n"
            "MAESTRO_LOG_LEVEL=DEBUG\n"
        )
        p = self._write(content)
        result = _parse_dotenv(p)
        assert result["ANTHROPIC_API_KEY"] == "sk-ant-abc"
        assert result["OPENAI_API_KEY"]    == "sk-oai-def"
        assert result["MAESTRO_LOG_LEVEL"] == "DEBUG"
        os.unlink(p)

    def test_real_dotenv_example(self):
        """Parse the .env.example file from the project."""
        example = Path(__file__).parent.parent / ".env.example"
        if not example.exists():
            pytest.skip(".env.example not found")
        result = _parse_dotenv(example)
        # Should parse at least the documented keys
        assert "ANTHROPIC_API_KEY" in result
        assert "OPENAI_API_KEY"    in result
        assert "MAESTRO_LOG_LEVEL" in result


# ════════════════════════════════════════════════════════════════════════════
#  load_env
# ════════════════════════════════════════════════════════════════════════════

class TestLoadEnv:
    def _write_env(self, content):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False, encoding="utf-8")
        f.write(content); f.flush(); f.close()
        return f.name

    def _cleanup_keys(self, *keys):
        for k in keys:
            os.environ.pop(k, None)

    def test_loads_variables_into_os_environ(self):
        path = self._write_env("TEST_MAESTRO_LOAD=hello\n")
        try:
            self._cleanup_keys("TEST_MAESTRO_LOAD")
            load_env(path)
            assert os.environ.get("TEST_MAESTRO_LOAD") == "hello"
        finally:
            self._cleanup_keys("TEST_MAESTRO_LOAD")
            os.unlink(path)

    def test_does_not_overwrite_existing_by_default(self):
        path = self._write_env("TEST_MAESTRO_KEEP=from_file\n")
        try:
            os.environ["TEST_MAESTRO_KEEP"] = "from_env"
            load_env(path)
            assert os.environ["TEST_MAESTRO_KEEP"] == "from_env"
        finally:
            self._cleanup_keys("TEST_MAESTRO_KEEP")
            os.unlink(path)

    def test_override_flag_overwrites_existing(self):
        path = self._write_env("TEST_MAESTRO_OVER=new_value\n")
        try:
            os.environ["TEST_MAESTRO_OVER"] = "old_value"
            load_env(path, override=True)
            assert os.environ["TEST_MAESTRO_OVER"] == "new_value"
        finally:
            self._cleanup_keys("TEST_MAESTRO_OVER")
            os.unlink(path)

    def test_returns_dict_of_loaded_keys(self):
        path = self._write_env("TEST_MAESTRO_RET=yes\n")
        try:
            self._cleanup_keys("TEST_MAESTRO_RET")
            loaded = load_env(path)
            assert "TEST_MAESTRO_RET" in loaded
            assert loaded["TEST_MAESTRO_RET"] == "yes"
        finally:
            self._cleanup_keys("TEST_MAESTRO_RET")
            os.unlink(path)

    def test_missing_file_returns_empty_dict(self):
        result = load_env("/nonexistent/.env.xyz")
        assert result == {}

    def test_loads_all_maestro_config_keys(self):
        content = (
            "ANTHROPIC_API_KEY=test-key-1\n"
            "OPENAI_API_KEY=test-key-2\n"
            "MAESTRO_ANTHROPIC_MODEL=claude-opus-4-6\n"
            "MAESTRO_OPENAI_MODEL=gpt-4o\n"
            "MAESTRO_LOG_LEVEL=DEBUG\n"
            "MAESTRO_BATCH_SIZE=500\n"
        )
        path = self._write_env(content)
        keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MAESTRO_ANTHROPIC_MODEL",
                "MAESTRO_OPENAI_MODEL", "MAESTRO_LOG_LEVEL", "MAESTRO_BATCH_SIZE"]
        try:
            for k in keys: os.environ.pop(k, None)
            load_env(path)
            assert os.environ.get("MAESTRO_ANTHROPIC_MODEL") == "claude-opus-4-6"
            assert os.environ.get("MAESTRO_BATCH_SIZE") == "500"
        finally:
            for k in keys: os.environ.pop(k, None)
            os.unlink(path)

    def test_accessible_via_maestro_namespace(self):
        path = self._write_env("TEST_MAESTRO_NS=works\n")
        try:
            self._cleanup_keys("TEST_MAESTRO_NS")
            maestro.load_env(path)
            assert os.environ.get("TEST_MAESTRO_NS") == "works"
        finally:
            self._cleanup_keys("TEST_MAESTRO_NS")
            os.unlink(path)


# ════════════════════════════════════════════════════════════════════════════
#  get_config / MaestroConfig
# ════════════════════════════════════════════════════════════════════════════

class TestGetConfig:
    def test_returns_maestro_config(self):
        cfg = get_config()
        assert isinstance(cfg, MaestroConfig)

    def test_default_anthropic_model(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAESTRO_ANTHROPIC_MODEL", None)
            cfg = get_config()
            assert cfg.anthropic_model == "claude-haiku-4-5-20251001"

    def test_reads_anthropic_api_key(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-123"}):
            cfg = get_config()
            assert cfg.anthropic_api_key == "sk-test-123"

    def test_reads_openai_api_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-oai-test"}):
            cfg = get_config()
            assert cfg.openai_api_key == "sk-oai-test"

    def test_reads_custom_model(self):
        with patch.dict(os.environ, {"MAESTRO_ANTHROPIC_MODEL": "claude-opus-4-6"}):
            cfg = get_config()
            assert cfg.anthropic_model == "claude-opus-4-6"

    def test_reads_openai_base_url(self):
        with patch.dict(os.environ, {"MAESTRO_OPENAI_BASE_URL": "http://localhost:11434/v1"}):
            cfg = get_config()
            assert cfg.openai_base_url == "http://localhost:11434/v1"

    def test_reads_retry_settings(self):
        with patch.dict(os.environ, {"MAESTRO_RETRY_MAX_ATTEMPTS": "5",
                                      "MAESTRO_RETRY_MAX_DELAY": "120.0"}):
            cfg = get_config()
            assert cfg.retry_max_attempts == 5
            assert cfg.retry_max_delay    == 120.0

    def test_reads_batch_settings(self):
        with patch.dict(os.environ, {"MAESTRO_BATCH_SIZE": "500",
                                      "MAESTRO_ERROR_THRESHOLD": "10"}):
            cfg = get_config()
            assert cfg.batch_size      == 500
            assert cfg.error_threshold == 10

    def test_reads_scheduler_settings(self):
        with patch.dict(os.environ, {"MAESTRO_SCHEDULER_TICK": "0.5",
                                      "MAESTRO_SCHEDULER_MAX_WORKERS": "8"}):
            cfg = get_config()
            assert cfg.scheduler_tick        == 0.5
            assert cfg.scheduler_max_workers == 8

    def test_metrics_enabled_true_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAESTRO_METRICS_ENABLED", None)
            cfg = get_config()
            assert cfg.metrics_enabled is True

    def test_metrics_disabled(self):
        for falsy in ("false", "0", "no", "off"):
            with patch.dict(os.environ, {"MAESTRO_METRICS_ENABLED": falsy}):
                cfg = get_config()
                assert cfg.metrics_enabled is False, f"Expected False for {falsy!r}"

    def test_metrics_enabled_truthy_values(self):
        for truthy in ("true", "1", "yes", "TRUE"):
            with patch.dict(os.environ, {"MAESTRO_METRICS_ENABLED": truthy}):
                cfg = get_config()
                assert cfg.metrics_enabled is True, f"Expected True for {truthy!r}"

    def test_reads_log_level(self):
        with patch.dict(os.environ, {"MAESTRO_LOG_LEVEL": "DEBUG"}):
            cfg = get_config()
            assert cfg.log_level == "DEBUG"

    def test_prometheus_path(self):
        with patch.dict(os.environ, {"MAESTRO_PROMETHEUS_PATH": "/tmp/metrics.prom"}):
            cfg = get_config()
            assert cfg.prometheus_path == "/tmp/metrics.prom"

    def test_none_for_unset_optional(self):
        with patch.dict(os.environ, {}, clear=False):
            for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MAESTRO_OPENAI_BASE_URL"):
                os.environ.pop(k, None)
            cfg = get_config()
            assert cfg.anthropic_api_key is None
            assert cfg.openai_api_key    is None
            assert cfg.openai_base_url   is None

    def test_accessible_via_maestro_namespace(self):
        cfg = maestro.get_config()
        assert isinstance(cfg, MaestroConfig)

    def test_repr_masks_api_keys(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-secret-abc"}):
            cfg  = get_config()
            text = repr(cfg)
            assert "sk-secret-abc" not in text
            assert "***" in text


# ════════════════════════════════════════════════════════════════════════════
#  global_observer
# ════════════════════════════════════════════════════════════════════════════

class TestGlobalObserver:
    def setup_method(self):
        reset_global_observer()

    def teardown_method(self):
        reset_global_observer()

    def test_returns_observer_when_enabled(self):
        with patch.dict(os.environ, {"MAESTRO_METRICS_ENABLED": "true"}):
            obs = global_observer()
            assert obs is not None

    def test_returns_none_when_disabled(self):
        with patch.dict(os.environ, {"MAESTRO_METRICS_ENABLED": "false"}):
            reset_global_observer()
            obs = global_observer()
            assert obs is None

    def test_same_instance_on_repeated_calls(self):
        with patch.dict(os.environ, {"MAESTRO_METRICS_ENABLED": "true"}):
            obs1 = global_observer()
            obs2 = global_observer()
            assert obs1 is obs2

    def test_reset_clears_instance(self):
        with patch.dict(os.environ, {"MAESTRO_METRICS_ENABLED": "true"}):
            obs1 = global_observer()
            reset_global_observer()
            obs2 = global_observer()
            assert obs1 is not obs2

    def test_accessible_via_maestro_namespace(self):
        with patch.dict(os.environ, {"MAESTRO_METRICS_ENABLED": "true"}):
            obs = maestro.global_observer()
            assert obs is not None


# ════════════════════════════════════════════════════════════════════════════
#  Config-aware factory functions
# ════════════════════════════════════════════════════════════════════════════

class TestConfigAwareFactories:
    def test_make_anthropic_from_config_uses_env_model(self):
        with patch.dict(os.environ, {
            "MAESTRO_ANTHROPIC_MODEL":    "claude-opus-4-6",
            "MAESTRO_ANTHROPIC_MAX_TOKENS": "8192",
        }), patch("anthropic.Anthropic"):
            adapter = make_anthropic_from_config()
            assert adapter.model == "claude-opus-4-6"
            assert adapter._max_tokens == 8192

    def test_make_openai_from_config_uses_env_model(self):
        with patch.dict(os.environ, {
            "MAESTRO_OPENAI_MODEL":   "gpt-4o",
            "MAESTRO_OPENAI_BASE_URL": "http://localhost:11434/v1",
        }), patch("openai.OpenAI"):
            adapter = make_openai_from_config()
            assert adapter.model == "gpt-4o"

    def test_make_anthropic_from_config_overrides_work(self):
        with patch.dict(os.environ, {"MAESTRO_ANTHROPIC_MODEL": "claude-haiku-4-5-20251001"}), \
             patch("anthropic.Anthropic"):
            adapter = make_anthropic_from_config(model="claude-opus-4-6")
            assert adapter.model == "claude-opus-4-6"

    def test_make_anthropic_uses_config_defaults(self):
        """make_anthropic() (no args) picks up config values after load_env."""
        with patch.dict(os.environ, {"MAESTRO_ANTHROPIC_MODEL": "claude-sonnet-4-6"}), \
             patch("anthropic.Anthropic"):
            from maestro.agents import make_anthropic
            adapter = make_anthropic()
            assert adapter.model == "claude-sonnet-4-6"

    def test_make_openai_uses_config_defaults(self):
        with patch.dict(os.environ, {"MAESTRO_OPENAI_MODEL": "o1-mini"}), \
             patch("openai.OpenAI"):
            from maestro.agents import make_openai
            adapter = make_openai()
            assert adapter.model == "o1-mini"


# ════════════════════════════════════════════════════════════════════════════
#  configure_logging
# ════════════════════════════════════════════════════════════════════════════

class TestConfigureLogging:
    def test_sets_log_level(self):
        import logging
        configure_logging("WARNING")
        assert logging.getLogger("maestro").level == logging.WARNING
        configure_logging("INFO")  # reset

    def test_uses_env_level_by_default(self):
        import logging
        with patch.dict(os.environ, {"MAESTRO_LOG_LEVEL": "ERROR"}):
            configure_logging()
            assert logging.getLogger("maestro").level == logging.ERROR
        configure_logging("INFO")  # reset


# ════════════════════════════════════════════════════════════════════════════
#  CLI --env flag
# ════════════════════════════════════════════════════════════════════════════

class TestCLIEnvFlag:
    def _write_env(self, content):
        import tempfile
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False, encoding="utf-8")
        f.write(content); f.flush(); f.close()
        return f.name

    def test_config_command_shows_defaults(self, capsys):
        from maestro.cli import main
        main(["--no-env", "config"])
        out = capsys.readouterr().out
        assert "ANTHROPIC_API_KEY" in out
        assert "MAESTRO_LOG_LEVEL" in out

    def test_config_check_reports_missing_keys(self, capsys):
        from maestro.cli import main
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY",    None)
            rc = main(["--no-env", "config", "--check"])
            out = capsys.readouterr().out
            assert "Missing" in out or rc == 1

    def test_env_flag_loads_file(self, capsys):
        path = self._write_env("MAESTRO_LOG_LEVEL=DEBUG\n")
        try:
            from maestro.cli import main
            os.environ.pop("MAESTRO_LOG_LEVEL", None)
            main(["--env", path, "--no-env", "config"])
        finally:
            os.environ.pop("MAESTRO_LOG_LEVEL", None)
            os.unlink(path)

    def test_no_env_flag_skips_dotenv(self, capsys):
        from maestro.cli import main
        # Should not fail even if .env doesn't exist
        rc = main(["--no-env", "info"])
        out = capsys.readouterr().out
        assert "Maestro SDK" in out


# ════════════════════════════════════════════════════════════════════════════
#  End-to-end: load .env → get_config → make_anthropic
# ════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_full_flow(self):
        """
        1. Write a .env with model and key.
        2. load_env() → sets os.environ.
        3. get_config() → reads from environ.
        4. make_anthropic() → uses config defaults.
        """
        content = (
            "ANTHROPIC_API_KEY=sk-e2e-test-key\n"
            "MAESTRO_ANTHROPIC_MODEL=claude-opus-4-6\n"
            "MAESTRO_ANTHROPIC_MAX_TOKENS=2048\n"
        )
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False, encoding="utf-8")
        f.write(content); f.flush(); f.close()
        keys = ["ANTHROPIC_API_KEY", "MAESTRO_ANTHROPIC_MODEL", "MAESTRO_ANTHROPIC_MAX_TOKENS"]
        try:
            for k in keys: os.environ.pop(k, None)
            loaded = maestro.load_env(f.name)
            assert "MAESTRO_ANTHROPIC_MODEL" in loaded

            cfg = maestro.get_config()
            assert cfg.anthropic_api_key   == "sk-e2e-test-key"
            assert cfg.anthropic_model     == "claude-opus-4-6"
            assert cfg.anthropic_max_tokens == 2048

            with patch("anthropic.Anthropic"):
                from maestro.agents import make_anthropic
                adapter = make_anthropic()
                assert adapter.model == "claude-opus-4-6"
                assert adapter._max_tokens == 2048

        finally:
            for k in keys: os.environ.pop(k, None)
            os.unlink(f.name)

    def test_dotenv_example_parses_without_error(self):
        """The .env.example file should parse cleanly."""
        example = Path(__file__).parent.parent / ".env.example"
        if not example.exists():
            pytest.skip(".env.example not found")
        result = _parse_dotenv(example)
        assert len(result) > 0
        assert "ANTHROPIC_API_KEY" in result
        assert "MAESTRO_LOG_LEVEL" in result
