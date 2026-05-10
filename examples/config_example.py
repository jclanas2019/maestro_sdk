"""
examples/config_example.py — maestro.config patterns.

Shows every configuration pattern without requiring real API keys.

Run: python examples/config_example.py
"""
import sys, os, tempfile, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import maestro
from maestro.config import (
    load_env, get_config, MaestroConfig,
    global_observer, reset_global_observer,
    configure_logging, make_anthropic_from_config, make_openai_from_config,
    _parse_dotenv,
)

SEP = "═" * 62


# ════════════════════════════════════════════════════════════════════════════
#  Helper — write a temporary .env file
# ════════════════════════════════════════════════════════════════════════════

def tmp_env(content: str) -> str:
    """Write content to a temp .env file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False, encoding="utf-8")
    f.write(content); f.flush(); f.close()
    return f.name

def cleanup(*paths): [os.unlink(p) for p in paths if os.path.exists(p)]
def clean_keys(*keys): [os.environ.pop(k, None) for k in keys]


# ════════════════════════════════════════════════════════════════════════════
#  1. .env file syntax — what's supported
# ════════════════════════════════════════════════════════════════════════════
print(SEP); print("1. .env file syntax — supported formats"); print(SEP)

env_content = """
# ── API Keys ──────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-demo-key
OPENAI_API_KEY=sk-oai-demo-key

# ── Plain values ──────────────────────────────────────────────────────────
MAESTRO_LOG_LEVEL=DEBUG
MAESTRO_BATCH_SIZE=500

# ── Quoted values ─────────────────────────────────────────────────────────
MAESTRO_ANTHROPIC_MODEL="claude-haiku-4-5-20251001"
SERVICE_DESC='order processing pipeline'

# ── export prefix (supported) ─────────────────────────────────────────────
export MAESTRO_OPENAI_MODEL=gpt-4o-mini

# ── Values with equals signs (base64, URLs) ───────────────────────────────
DB_URL=postgresql://user:pass@host/db?ssl=true
ENCODED=abc123==

# ── Inline comment on unquoted value ──────────────────────────────────────
MAESTRO_RETRY_MAX_ATTEMPTS=3 # max retries before giving up

# ── Empty value ───────────────────────────────────────────────────────────
OPTIONAL_KEY=
"""
path1 = tmp_env(env_content)
parsed = _parse_dotenv(path1)
cleanup(path1)

keys_to_show = [
    "ANTHROPIC_API_KEY", "MAESTRO_LOG_LEVEL", "MAESTRO_BATCH_SIZE",
    "MAESTRO_ANTHROPIC_MODEL", "SERVICE_DESC", "MAESTRO_OPENAI_MODEL",
    "DB_URL", "ENCODED", "MAESTRO_RETRY_MAX_ATTEMPTS", "OPTIONAL_KEY",
]
for k in keys_to_show:
    v = parsed.get(k, "(missing)")
    display = "***" if "KEY" in k else v
    print(f"  {k:<36} = {display!r}")


# ════════════════════════════════════════════════════════════════════════════
#  2. load_env() — basic loading
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("2. load_env() — loading into os.environ"); print(SEP)

path2 = tmp_env(
    "TEST_DEMO_MODEL=claude-opus-4-6\n"
    "TEST_DEMO_BATCH=250\n"
    "TEST_DEMO_LOG=WARNING\n"
)
clean_keys("TEST_DEMO_MODEL", "TEST_DEMO_BATCH", "TEST_DEMO_LOG")

loaded = maestro.load_env(path2)
cleanup(path2)

print(f"  Keys loaded: {list(loaded.keys())}")
print(f"  TEST_DEMO_MODEL = {os.environ.get('TEST_DEMO_MODEL')}")
print(f"  TEST_DEMO_BATCH = {os.environ.get('TEST_DEMO_BATCH')}")
clean_keys("TEST_DEMO_MODEL", "TEST_DEMO_BATCH", "TEST_DEMO_LOG")


# ════════════════════════════════════════════════════════════════════════════
#  3. Priority — env vars beat .env
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("3. Priority — environment beats .env"); print(SEP)

path3 = tmp_env("DEMO_PRIORITY=from_file\n")
os.environ["DEMO_PRIORITY"] = "from_environment"

loaded3 = maestro.load_env(path3)
cleanup(path3)
print(f"  .env says: 'from_file'")
print(f"  env says:  'from_environment'")
print(f"  result:    {os.environ.get('DEMO_PRIORITY')!r}  ← env wins")

# override=True reverses this
path3b = tmp_env("DEMO_PRIORITY=from_file_overridden\n")
maestro.load_env(path3b, override=True)
cleanup(path3b)
print(f"\n  With override=True:")
print(f"  result:    {os.environ.get('DEMO_PRIORITY')!r}  ← file wins")
clean_keys("DEMO_PRIORITY")


# ════════════════════════════════════════════════════════════════════════════
#  4. get_config() — typed config object
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("4. get_config() — typed MaestroConfig"); print(SEP)

path4 = tmp_env(
    "ANTHROPIC_API_KEY=sk-ant-example\n"
    "MAESTRO_ANTHROPIC_MODEL=claude-opus-4-6\n"
    "MAESTRO_ANTHROPIC_MAX_TOKENS=8192\n"
    "MAESTRO_OPENAI_MODEL=gpt-4o\n"
    "MAESTRO_OPENAI_BASE_URL=http://localhost:11434/v1\n"
    "MAESTRO_RETRY_MAX_ATTEMPTS=5\n"
    "MAESTRO_BATCH_SIZE=1000\n"
    "MAESTRO_LOG_LEVEL=WARNING\n"
    "MAESTRO_METRICS_ENABLED=true\n"
    "MAESTRO_SCHEDULER_MAX_WORKERS=8\n"
)
clean_keys("ANTHROPIC_API_KEY", "MAESTRO_ANTHROPIC_MODEL", "MAESTRO_ANTHROPIC_MAX_TOKENS",
           "MAESTRO_OPENAI_MODEL", "MAESTRO_OPENAI_BASE_URL", "MAESTRO_RETRY_MAX_ATTEMPTS",
           "MAESTRO_BATCH_SIZE", "MAESTRO_LOG_LEVEL", "MAESTRO_METRICS_ENABLED",
           "MAESTRO_SCHEDULER_MAX_WORKERS")
maestro.load_env(path4)
cleanup(path4)

cfg = maestro.get_config()

print(f"  API keys:")
print(f"    anthropic_api_key  = {'***' if cfg.anthropic_api_key else '(not set)'}")
print(f"    openai_api_key     = {'***' if cfg.openai_api_key else '(not set)'}")
print()
print(f"  Anthropic defaults:")
print(f"    model              = {cfg.anthropic_model}")
print(f"    max_tokens         = {cfg.anthropic_max_tokens}")
print(f"    temperature        = {cfg.anthropic_temperature}")
print()
print(f"  OpenAI defaults:")
print(f"    model              = {cfg.openai_model}")
print(f"    base_url           = {cfg.openai_base_url or '(default)'}")
print()
print(f"  Retry:")
print(f"    max_attempts       = {cfg.retry_max_attempts}")
print(f"    max_delay          = {cfg.retry_max_delay}s")
print()
print(f"  Batch:              size={cfg.batch_size}  error_threshold={cfg.error_threshold}")
print(f"  Logging:            level={cfg.log_level}")
print(f"  Metrics:            enabled={cfg.metrics_enabled}")
print(f"  Scheduler:          tick={cfg.scheduler_tick}s  workers={cfg.scheduler_max_workers}")

clean_keys("ANTHROPIC_API_KEY", "MAESTRO_ANTHROPIC_MODEL", "MAESTRO_ANTHROPIC_MAX_TOKENS",
           "MAESTRO_OPENAI_MODEL", "MAESTRO_OPENAI_BASE_URL", "MAESTRO_RETRY_MAX_ATTEMPTS",
           "MAESTRO_BATCH_SIZE", "MAESTRO_LOG_LEVEL", "MAESTRO_METRICS_ENABLED",
           "MAESTRO_SCHEDULER_MAX_WORKERS")


# ════════════════════════════════════════════════════════════════════════════
#  5. make_anthropic() / make_openai() — config-aware factories
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("5. Config-aware factories — make_anthropic() / make_openai()"); print(SEP)

from unittest.mock import patch

path5 = tmp_env(
    "ANTHROPIC_API_KEY=sk-ant-from-env\n"
    "MAESTRO_ANTHROPIC_MODEL=claude-opus-4-6\n"
    "MAESTRO_ANTHROPIC_MAX_TOKENS=2048\n"
    "OPENAI_API_KEY=sk-oai-from-env\n"
    "MAESTRO_OPENAI_MODEL=gpt-4o\n"
)
clean_keys("ANTHROPIC_API_KEY", "OPENAI_API_KEY",
           "MAESTRO_ANTHROPIC_MODEL", "MAESTRO_ANTHROPIC_MAX_TOKENS", "MAESTRO_OPENAI_MODEL")
maestro.load_env(path5)
cleanup(path5)

with patch("anthropic.Anthropic"), patch("openai.OpenAI"):
    from maestro.agents import make_anthropic, make_openai

    llm_a = make_anthropic()   # no args — uses .env values
    llm_o = make_openai()      # no args — uses .env values
    llm_a_override = make_anthropic(model="claude-haiku-4-5-20251001")  # override model only

    print("  make_anthropic() — no args, reads from .env:")
    print(f"    model      = {llm_a.model}")
    print(f"    max_tokens = {llm_a._max_tokens}")
    print()
    print("  make_openai() — no args, reads from .env:")
    print(f"    model = {llm_o.model}")
    print()
    print("  make_anthropic(model='claude-haiku-4-5-20251001') — explicit override:")
    print(f"    model = {llm_a_override.model}")

    # Config-specific factories
    llm_a2 = make_anthropic_from_config()
    llm_o2 = make_openai_from_config(model="gpt-4o-mini")
    print()
    print("  make_anthropic_from_config():")
    print(f"    model = {llm_a2.model}")
    print("  make_openai_from_config(model='gpt-4o-mini'):")
    print(f"    model = {llm_o2.model}")

clean_keys("ANTHROPIC_API_KEY", "OPENAI_API_KEY",
           "MAESTRO_ANTHROPIC_MODEL", "MAESTRO_ANTHROPIC_MAX_TOKENS", "MAESTRO_OPENAI_MODEL")


# ════════════════════════════════════════════════════════════════════════════
#  6. with_retry() reads config defaults
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("6. with_retry() — config-driven retry policy"); print(SEP)

path6 = tmp_env(
    "MAESTRO_RETRY_MAX_ATTEMPTS=4\n"
    "MAESTRO_RETRY_MAX_DELAY=45.0\n"
)
clean_keys("MAESTRO_RETRY_MAX_ATTEMPTS", "MAESTRO_RETRY_MAX_DELAY")
maestro.load_env(path6)
cleanup(path6)

cfg6 = maestro.get_config()
print(f"  Config says: max_attempts={cfg6.retry_max_attempts}  max_delay={cfg6.retry_max_delay}")
print()

# Combine .env model selection + retry + observability
obs6 = maestro.global_observer()
reset_global_observer()

from maestro.agents._providers import _RetryingAdapter, _ObservingAdapter, LLMResponse, TokenUsage
from maestro.observe import InMemoryObserver

base_mock = __import__('unittest.mock', fromlist=['MagicMock']).MagicMock()
type(base_mock).model    = property(lambda s: "claude-haiku-4-5-20251001")
type(base_mock).provider = property(lambda s: "anthropic")
base_mock.chat.return_value = LLMResponse("OK", usage=TokenUsage(10, 20))

obs_local = InMemoryObserver()
resilient = _ObservingAdapter(_RetryingAdapter(base_mock, max_attempts=cfg6.retry_max_attempts), obs_local)

from maestro.agents._providers import Message
result = resilient.chat([Message.user("hello")])
print(f"  Observed call:  {obs_local.counter('agents', 'llm_call', model='claude-haiku-4-5-20251001', provider='anthropic'):.0f}")
print(f"  Tokens input:   {obs_local.gauge('agents', 'tokens_input', model='claude-haiku-4-5-20251001', provider='anthropic'):.0f}")
print(f"  Result:         {result.content}")
clean_keys("MAESTRO_RETRY_MAX_ATTEMPTS", "MAESTRO_RETRY_MAX_DELAY")


# ════════════════════════════════════════════════════════════════════════════
#  7. global_observer() — shared metrics collector
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("7. global_observer() — shared metrics across the SDK"); print(SEP)

reset_global_observer()
os.environ["MAESTRO_METRICS_ENABLED"] = "true"

obs7 = maestro.global_observer()
assert obs7 is maestro.global_observer()   # same instance
print(f"  global_observer() returns same instance: True")

# Emit some metrics
from maestro.observe import MetricEvent
obs7.on_event(MetricEvent("rules", "rule_fired", 3.0, {"rule": "vip"}, kind="counter"))
obs7.on_event(MetricEvent("batch", "records_written", 150.0, {}, kind="gauge"))
obs7.on_event(MetricEvent("agents", "llm_call", 1.0, {}, kind="counter"))

print()
for line in obs7.summary().split('\n'):
    if line.strip(): print(f"  {line.rstrip()}")

# Prometheus export
print()
print("  Prometheus snippet:")
for line in obs7.export_prometheus().split('\n')[:6]:
    if line: print(f"    {line}")

reset_global_observer()
clean_keys("MAESTRO_METRICS_ENABLED")


# ════════════════════════════════════════════════════════════════════════════
#  8. Multiple .env files — dev / staging / production
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("8. Multiple environments — .env.dev / .env.prod"); print(SEP)

dev_env  = tmp_env(
    "MAESTRO_ANTHROPIC_MODEL=claude-haiku-4-5-20251001\n"
    "MAESTRO_LOG_LEVEL=DEBUG\n"
    "MAESTRO_BATCH_SIZE=10\n"
    "MAESTRO_RETRY_MAX_ATTEMPTS=1\n"
)
prod_env = tmp_env(
    "MAESTRO_ANTHROPIC_MODEL=claude-opus-4-6\n"
    "MAESTRO_LOG_LEVEL=WARNING\n"
    "MAESTRO_BATCH_SIZE=1000\n"
    "MAESTRO_RETRY_MAX_ATTEMPTS=5\n"
)

for name, path in [("dev", dev_env), ("prod", prod_env)]:
    clean_keys("MAESTRO_ANTHROPIC_MODEL", "MAESTRO_LOG_LEVEL",
               "MAESTRO_BATCH_SIZE", "MAESTRO_RETRY_MAX_ATTEMPTS")
    maestro.load_env(path, override=True)
    cfg8 = maestro.get_config()
    print(f"  [{name:<4}]  model={cfg8.anthropic_model:<32} "
          f"log={cfg8.log_level:<8} batch={cfg8.batch_size:<6} retries={cfg8.retry_max_attempts}")

cleanup(dev_env, prod_env)
clean_keys("MAESTRO_ANTHROPIC_MODEL", "MAESTRO_LOG_LEVEL",
           "MAESTRO_BATCH_SIZE", "MAESTRO_RETRY_MAX_ATTEMPTS")


# ════════════════════════════════════════════════════════════════════════════
#  9. configure_logging
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("9. configure_logging() — from config or explicit"); print(SEP)

import logging

path9 = tmp_env("MAESTRO_LOG_LEVEL=WARNING\n")
clean_keys("MAESTRO_LOG_LEVEL")
maestro.load_env(path9)
cleanup(path9)

maestro.configure_logging()  # reads MAESTRO_LOG_LEVEL=WARNING
level_name = logging.getLevelName(logging.getLogger("maestro").level)
print(f"  After load_env(MAESTRO_LOG_LEVEL=WARNING):  level={level_name}")

maestro.configure_logging("DEBUG")
level_name = logging.getLevelName(logging.getLogger("maestro").level)
print(f"  After configure_logging('DEBUG'):           level={level_name}")
maestro.configure_logging("INFO")   # reset
clean_keys("MAESTRO_LOG_LEVEL")


# ════════════════════════════════════════════════════════════════════════════
#  10. CLI: maestro config
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("10. CLI — maestro config"); print(SEP)

from maestro.cli import main as cli_main
import contextlib, io

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    cli_main(["--no-env", "config"])
output = buf.getvalue()
print("  $ maestro config")
for line in output.split('\n')[:12]:
    if line.strip(): print(f"    {line.rstrip()}")
print("    ...")

print()
print("  $ maestro --env .env.example config --check")
buf2 = io.StringIO()
example_env = str(__import__('pathlib').Path(__file__).parent.parent / ".env.example")
with contextlib.redirect_stdout(buf2):
    cli_main(["--env", example_env, "--no-env", "config", "--check"])
for line in buf2.getvalue().split('\n')[-5:]:
    if line.strip(): print(f"    {line.rstrip()}")


# ════════════════════════════════════════════════════════════════════════════
#  11. OpenAI-compatible endpoints via .env
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("11. Alternative endpoints via .env"); print(SEP)

examples = [
    ("Ollama (local)",  "MAESTRO_OPENAI_BASE_URL=http://localhost:11434/v1",  "llama3"),
    ("Groq",            "MAESTRO_OPENAI_BASE_URL=https://api.groq.com/openai/v1", "llama-3.1-8b-instant"),
    ("Together AI",     "MAESTRO_OPENAI_BASE_URL=https://api.together.xyz/v1", "meta-llama/Llama-3-8b"),
    ("Azure OpenAI",    "MAESTRO_OPENAI_BASE_URL=https://my-resource.openai.azure.com/", "gpt-4o"),
]
print("  Configure alternative providers entirely from .env:")
for provider, env_line, model in examples:
    print(f"\n  # {provider}")
    print(f"    {env_line}")
    print(f"    MAESTRO_OPENAI_MODEL={model}")
    print(f"    → make_openai()  # connects to {provider.split('(')[0].strip()}")


print(); print(SEP); print("All config examples completed."); print(SEP)
print("""
Quick reference:
  import maestro
  maestro.load_env()           # load .env from cwd (call once at startup)
  cfg = maestro.get_config()   # typed config — re-reads env on every call
  llm = maestro.make_anthropic() # uses config model + api key automatically
  llm = maestro.make_openai()    # same for OpenAI
  obs = maestro.global_observer() # shared InMemoryObserver when metrics enabled

  maestro configure_logging()    # apply MAESTRO_LOG_LEVEL from env

CLI:
  maestro config                 # show resolved config
  maestro config --check         # verify required keys are set
  maestro --env prod.env config  # load a specific .env then show config
""")
