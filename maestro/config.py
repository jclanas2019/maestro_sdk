"""
maestro.config — environment configuration and .env loading.

Loads variables from a ``.env`` file into ``os.environ`` and exposes them
as a typed ``MaestroConfig`` dataclass.

Quick start::

    import maestro
    maestro.load_env()          # loads .env from current directory
    cfg = maestro.get_config()  # typed config object

    print(cfg.anthropic_api_key)        # reads ANTHROPIC_API_KEY
    print(cfg.anthropic_model)          # reads MAESTRO_ANTHROPIC_MODEL

    # Providers auto-pick up the config:
    llm = maestro.make_anthropic()      # uses cfg.anthropic_model, cfg.anthropic_api_key
    llm = maestro.make_openai()         # uses cfg.openai_model, cfg.openai_api_key

Design
------
* Variables already in ``os.environ`` are **never overwritten** by default.
  Pass ``override=True`` to force-overwrite from the file.
* All ``MAESTRO_*`` variables are read on every ``get_config()`` call so that
  changes to ``os.environ`` are always reflected.
* ``python-dotenv`` is used when installed; otherwise the built-in parser
  handles the common subset of ``.env`` syntax.
* A global ``InMemoryObserver`` is created when ``MAESTRO_METRICS_ENABLED=true``
  and is accessible as ``maestro.global_observer``.

Supported ``.env`` syntax::

    # comment
    KEY=plain value
    KEY="double-quoted value"
    KEY='single-quoted value'
    export KEY=value
    KEY=               # empty value
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_UNSET = object()   # sentinel for "not provided"


# ════════════════════════════════════════════════════════════════════════════
#  .env parser (no external dependency)
# ════════════════════════════════════════════════════════════════════════════

def _parse_dotenv(path: str | Path) -> dict[str, str]:
    """
    Parse a ``.env`` file and return a ``{key: value}`` dict.

    Handles: comments, export prefix, single/double quotes, empty values.
    Does NOT handle: multi-line values, variable interpolation, shell escapes.
    """
    result: dict[str, str] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return result
    except OSError as exc:
        logger.warning("config: cannot read %r — %s", str(path), exc)
        return result

    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()

        # Skip blanks and comments
        if not line or line.startswith("#"):
            continue

        # Strip optional 'export' prefix
        if line.startswith("export "):
            line = line[7:].lstrip()

        # Must contain '='
        if "=" not in line:
            logger.debug("config: line %d has no '=' — skipped", lineno)
            continue

        key, _, value = line.partition("=")
        key   = key.strip()
        value = value.strip()

        if not key:
            continue

        # Strip inline comments only when value is unquoted
        if value and value[0] not in ('"', "'"):
            hash_pos = value.find(" #")
            if hash_pos >= 0:
                value = value[:hash_pos].rstrip()

        # Strip matching outer quotes
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]

        result[key] = value

    return result


# ════════════════════════════════════════════════════════════════════════════
#  load_env
# ════════════════════════════════════════════════════════════════════════════

def load_env(
    path:     str | Path = ".env",
    override: bool = False,
    verbose:  bool = False,
) -> dict[str, str]:
    """
    Load a ``.env`` file into ``os.environ``.

    Parameters
    ----------
    path:
        Path to the ``.env`` file (default: ``".env"`` in the current directory).
    override:
        If ``True``, overwrite variables that already exist in ``os.environ``.
        Default: ``False`` (environment variables take priority).
    verbose:
        If ``True``, log each loaded key at DEBUG level.

    Returns
    -------
    dict
        Keys and values that were loaded (whether or not they were set in the
        environment — useful for debugging).

    Example::

        import maestro
        maestro.load_env()                  # .env in cwd
        maestro.load_env(".env.production") # custom path
        maestro.load_env(override=True)     # force-overwrite env vars
    """
    # Try python-dotenv first (handles more edge cases)
    try:
        from dotenv import dotenv_values
        parsed = dict(dotenv_values(dotenv_path=str(path)))
    except ImportError:
        parsed = _parse_dotenv(path)

    if not parsed and not Path(path).exists():
        logger.debug("config: %r not found — skipping", str(path))
        return {}

    loaded: dict[str, str] = {}
    for key, value in parsed.items():
        if value is None:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
            if verbose:
                # Never log the value for keys that look like secrets
                safe = "***" if any(s in key.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD")) else value
                logger.debug("config: %s = %s", key, safe)
        else:
            logger.debug("config: %s already set — skipped", key)

    # Apply log level immediately if present
    if "MAESTRO_LOG_LEVEL" in loaded:
        _apply_log_level(os.environ["MAESTRO_LOG_LEVEL"])

    count = len(loaded)
    logger.info("config: loaded %d variable(s) from %r", count, str(path))
    return loaded


# ════════════════════════════════════════════════════════════════════════════
#  MaestroConfig
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class MaestroConfig:
    """
    Typed view of all Maestro-relevant environment variables.

    Instantiated by :func:`get_config`.  All values are read from
    ``os.environ`` at construction time, so calling ``get_config()`` again
    after modifying the environment yields fresh values.

    All fields have sensible defaults so ``get_config()`` works even without
    a ``.env`` file.
    """

    # ── API keys ──────────────────────────────────────────────────────── #
    anthropic_api_key: Optional[str] = None
    """Value of ``ANTHROPIC_API_KEY``."""

    openai_api_key:    Optional[str] = None
    """Value of ``OPENAI_API_KEY``."""

    # ── Anthropic defaults ────────────────────────────────────────────── #
    anthropic_model:        str   = "claude-haiku-4-5-20251001"
    """``MAESTRO_ANTHROPIC_MODEL`` — default Anthropic model identifier."""

    anthropic_max_tokens:   int   = 4096
    """``MAESTRO_ANTHROPIC_MAX_TOKENS`` — max response tokens for Anthropic."""

    anthropic_temperature:  float = 1.0
    """``MAESTRO_ANTHROPIC_TEMPERATURE`` — Anthropic sampling temperature."""

    # ── OpenAI defaults ───────────────────────────────────────────────── #
    openai_model:       str            = "gpt-4o-mini"
    """``MAESTRO_OPENAI_MODEL`` — default OpenAI model identifier."""

    openai_base_url:    Optional[str]  = None
    """``MAESTRO_OPENAI_BASE_URL`` — override API endpoint (Ollama, Azure …)."""

    openai_temperature: float          = 1.0
    """``MAESTRO_OPENAI_TEMPERATURE`` — OpenAI sampling temperature."""

    # ── Retry ─────────────────────────────────────────────────────────── #
    retry_max_attempts: int   = 3
    """``MAESTRO_RETRY_MAX_ATTEMPTS`` — default retry attempts for LLM calls."""

    retry_max_delay:    float = 60.0
    """``MAESTRO_RETRY_MAX_DELAY`` — cap on exponential backoff (seconds)."""

    retry_base_delay:   float = 1.0
    """``MAESTRO_RETRY_BASE_DELAY`` — first-retry delay (seconds)."""

    # ── Logging ───────────────────────────────────────────────────────── #
    log_level: str = "INFO"
    """``MAESTRO_LOG_LEVEL`` — Python logging level for the maestro namespace."""

    # ── Batch ─────────────────────────────────────────────────────────── #
    batch_size:        int = 100
    """``MAESTRO_BATCH_SIZE`` — default records-per-write-batch."""

    error_threshold:   int = -1
    """``MAESTRO_ERROR_THRESHOLD`` — failed records before job aborts (-1=∞)."""

    # ── Scheduler ─────────────────────────────────────────────────────── #
    scheduler_tick:        float = 1.0
    """``MAESTRO_SCHEDULER_TICK`` — seconds between scheduler checks."""

    scheduler_max_workers: int   = 4
    """``MAESTRO_SCHEDULER_MAX_WORKERS`` — thread pool size for the scheduler."""

    # ── Observability ─────────────────────────────────────────────────── #
    metrics_enabled:  bool          = True
    """``MAESTRO_METRICS_ENABLED`` — enable the global in-memory observer."""

    prometheus_path: Optional[str] = None
    """``MAESTRO_PROMETHEUS_PATH`` — write Prometheus metrics to this file."""

    def __repr__(self) -> str:
        def _safe(k, v):
            if isinstance(v, str) and any(s in k for s in ("key", "secret", "token", "password")):
                return "***" if v else "(not set)"
            return repr(v) if v is not None else "(not set)"
        lines = [f"  {k} = {_safe(k, v)}" for k, v in self.__dict__.items()]
        return "MaestroConfig(\n" + "\n".join(lines) + "\n)"


# ════════════════════════════════════════════════════════════════════════════
#  get_config
# ════════════════════════════════════════════════════════════════════════════

def get_config() -> MaestroConfig:
    """
    Return a :class:`MaestroConfig` built from the current environment.

    Call :func:`load_env` first to populate variables from a ``.env`` file.
    Every call re-reads ``os.environ``, so changes are always reflected.

    Example::

        import maestro
        maestro.load_env()
        cfg = maestro.get_config()

        print(cfg.anthropic_model)
        print(cfg.retry_max_attempts)
        print(cfg.metrics_enabled)
    """
    e = os.environ.get

    def _bool(key: str, default: bool) -> bool:
        v = e(key)
        if v is None: return default
        return v.strip().lower() not in ("0", "false", "no", "off", "")

    def _int(key: str, default: int) -> int:
        v = e(key)
        if v is None: return default
        try:   return int(v.strip())
        except ValueError: return default

    def _float(key: str, default: float) -> float:
        v = e(key)
        if v is None: return default
        try:   return float(v.strip())
        except ValueError: return default

    def _str(key: str, default: str) -> str:
        v = e(key)
        return v.strip() if v else default

    def _opt(key: str) -> Optional[str]:
        v = e(key)
        return v.strip() if v else None

    return MaestroConfig(
        anthropic_api_key      = _opt("ANTHROPIC_API_KEY"),
        openai_api_key         = _opt("OPENAI_API_KEY"),
        anthropic_model        = _str("MAESTRO_ANTHROPIC_MODEL",   "claude-haiku-4-5-20251001"),
        anthropic_max_tokens   = _int("MAESTRO_ANTHROPIC_MAX_TOKENS", 4096),
        anthropic_temperature  = _float("MAESTRO_ANTHROPIC_TEMPERATURE", 1.0),
        openai_model           = _str("MAESTRO_OPENAI_MODEL",   "gpt-4o-mini"),
        openai_base_url        = _opt("MAESTRO_OPENAI_BASE_URL"),
        openai_temperature     = _float("MAESTRO_OPENAI_TEMPERATURE", 1.0),
        retry_max_attempts     = _int("MAESTRO_RETRY_MAX_ATTEMPTS", 3),
        retry_max_delay        = _float("MAESTRO_RETRY_MAX_DELAY", 60.0),
        retry_base_delay       = _float("MAESTRO_RETRY_BASE_DELAY", 1.0),
        log_level              = _str("MAESTRO_LOG_LEVEL", "INFO"),
        batch_size             = _int("MAESTRO_BATCH_SIZE", 100),
        error_threshold        = _int("MAESTRO_ERROR_THRESHOLD", -1),
        scheduler_tick         = _float("MAESTRO_SCHEDULER_TICK", 1.0),
        scheduler_max_workers  = _int("MAESTRO_SCHEDULER_MAX_WORKERS", 4),
        metrics_enabled        = _bool("MAESTRO_METRICS_ENABLED", True),
        prometheus_path        = _opt("MAESTRO_PROMETHEUS_PATH"),
    )


# ════════════════════════════════════════════════════════════════════════════
#  Global observer (auto-created when MAESTRO_METRICS_ENABLED=true)
# ════════════════════════════════════════════════════════════════════════════

_global_observer = None


def global_observer():
    """
    Return the global :class:`~maestro.observe.InMemoryObserver`.

    Created lazily on first access if ``MAESTRO_METRICS_ENABLED=true``.
    Returns ``None`` if metrics are disabled.

    Example::

        import maestro
        maestro.load_env()
        obs = maestro.global_observer()
        if obs:
            print(obs.summary())
    """
    global _global_observer
    if _global_observer is None and get_config().metrics_enabled:
        from maestro.observe import InMemoryObserver
        _global_observer = InMemoryObserver()
    return _global_observer


def reset_global_observer() -> None:
    """Clear the global observer (useful in tests)."""
    global _global_observer
    _global_observer = None


# ════════════════════════════════════════════════════════════════════════════
#  Logging setup
# ════════════════════════════════════════════════════════════════════════════

def _apply_log_level(level: str) -> None:
    numeric = getattr(logging, level.upper(), None)
    if isinstance(numeric, int):
        logging.getLogger("maestro").setLevel(numeric)
        logger.debug("config: log level set to %s", level.upper())
    else:
        logger.warning("config: unknown log level %r — ignoring", level)


def configure_logging(level: Optional[str] = None) -> None:
    """
    Configure the ``maestro`` logger.

    If *level* is not provided, reads ``MAESTRO_LOG_LEVEL`` from the environment
    (or the default ``"INFO"``).

    Example::

        import maestro
        maestro.configure_logging("DEBUG")
        # or let the env decide:
        maestro.load_env()
        maestro.configure_logging()
    """
    effective = level or get_config().log_level
    _apply_log_level(effective)

    # Ensure at least a basic handler exists so messages appear
    root_logger = logging.getLogger("maestro")
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
        root_logger.addHandler(handler)


# ════════════════════════════════════════════════════════════════════════════
#  Config-aware factory helpers
# ════════════════════════════════════════════════════════════════════════════

def make_anthropic_from_config(**overrides):
    """
    Create an :class:`~maestro.agents.AnthropicAdapter` using values from
    ``get_config()`` as defaults, with any keyword arguments overriding them.

    Raises :exc:`ValueError` with a clear message if ``ANTHROPIC_API_KEY``
    is not set and not provided as an override.

    Example::

        import maestro
        maestro.load_env()
        llm = maestro.make_anthropic_from_config()
    """
    from maestro.agents._providers import AnthropicAdapter
    cfg    = get_config()
    api_key = overrides.pop("api_key", None) or cfg.anthropic_api_key
    if not api_key:
        raise ValueError(
            "Anthropic API key not set. "
            "Either set ANTHROPIC_API_KEY in your .env file / environment, "
            "or pass api_key= explicitly to make_anthropic_from_config()."
        )
    kwargs = {
        "model":       cfg.anthropic_model,
        "api_key":     api_key,
        "max_tokens":  cfg.anthropic_max_tokens,
        "temperature": cfg.anthropic_temperature,
    }
    kwargs.update(overrides)
    return AnthropicAdapter(**kwargs)


def make_openai_from_config(**overrides):
    """
    Create an :class:`~maestro.agents.OpenAIAdapter` using values from
    ``get_config()`` as defaults.

    Raises :exc:`ValueError` if ``OPENAI_API_KEY`` is not set and the
    ``base_url`` is also not overridden (base_url may not need a key for
    local endpoints like Ollama).

    Example::

        import maestro
        maestro.load_env()
        llm = maestro.make_openai_from_config()
        llm = maestro.make_openai_from_config(model="gpt-4o")
    """
    from maestro.agents._providers import OpenAIAdapter
    cfg     = get_config()
    api_key  = overrides.pop("api_key",  None) or cfg.openai_api_key
    base_url = overrides.pop("base_url", None) or cfg.openai_base_url
    if not api_key and not base_url:
        raise ValueError(
            "OpenAI API key not set. "
            "Either set OPENAI_API_KEY in your .env file / environment, "
            "pass api_key= explicitly, or set MAESTRO_OPENAI_BASE_URL for "
            "a local endpoint (e.g. Ollama) that does not require a key."
        )
    kwargs = {
        "model":       cfg.openai_model,
        "api_key":     api_key,
        "base_url":    base_url,
        "temperature": cfg.openai_temperature,
    }
    kwargs.update(overrides)
    return OpenAIAdapter(**kwargs)


__all__ = [
    "load_env",
    "get_config",
    "MaestroConfig",
    "global_observer",
    "reset_global_observer",
    "configure_logging",
    "make_anthropic_from_config",
    "make_openai_from_config",
]
