"""
maestro.cli — command-line interface for the Maestro SDK.

Install the ``maestro`` command::

    pip install -e .

Then use from your terminal::

    maestro info
    maestro rules validate examples/weather-rule.yaml
    maestro rules fire   examples/weather-rule.yaml --facts '{"rain": true}'
    maestro batch run    examples/batch-job.yaml
    maestro saga describe examples/book-trip.yaml
    maestro schedule demo
    maestro fsm dot      examples/turnstile.yaml

All commands support ``--help`` for details.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from typing import Optional


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

def _load_env_for_cli(args) -> None:
    """Load .env file based on CLI flags."""
    from maestro.config import load_env
    no_env  = getattr(args, "no_env", False)
    env_path = getattr(args, "env",    None)
    if no_env:
        return
    if env_path:
        loaded = load_env(env_path, verbose=True)
        print(f"  Loaded {len(loaded)} variable(s) from {env_path!r}")
    else:
        import os; from pathlib import Path
        if Path(".env").exists():
            load_env(".env")


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    # Auto-load .env
    _load_env_for_cli(args)

    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if "--debug" in (argv or sys.argv):
            import traceback; traceback.print_exc()
        return 1


# ════════════════════════════════════════════════════════════════════════════
#  Parser tree
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maestro",
        description="Maestro SDK — automation toolkit for rules, batch, flows and states.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Subcommands:
              info          Show SDK version and installed modules
              rules         Rules engine utilities
              batch         Batch pipeline utilities
              saga          Saga pattern utilities
              schedule      Scheduler utilities
              fsm           Finite state machine utilities

            Run 'maestro <subcommand> --help' for details.
        """),
    )
    parser.add_argument("--version", action="version", version=_version_string())
    parser.add_argument("--debug",   action="store_true", help="Show full tracebacks")
    parser.add_argument("--env",     default=None, metavar="FILE",
                        help="Load environment variables from FILE (default: .env if it exists)")
    parser.add_argument("--no-env",  action="store_true",
                        help="Do not auto-load .env from the current directory")

    sub = parser.add_subparsers(title="subcommands")
    _add_info(sub)
    _add_config(sub)
    _add_rules(sub)
    _add_batch(sub)
    _add_saga(sub)
    _add_schedule(sub)
    _add_fsm(sub)
    return parser


def _version_string() -> str:
    try:
        from maestro import __version__
        return f"maestro {__version__}"
    except Exception:
        return "maestro (version unknown)"


# ════════════════════════════════════════════════════════════════════════════
#  info
# ════════════════════════════════════════════════════════════════════════════

def _add_config(sub) -> None:
    p = sub.add_parser("config", help="Show resolved configuration from env / .env")
    p.add_argument("--all",    action="store_true", help="Show all fields, including defaults")
    p.add_argument("--check",  action="store_true", help="Check that required keys are set")
    p.set_defaults(func=_cmd_config)


def _cmd_config(args) -> None:
    from maestro.config import get_config
    cfg = get_config()
    _print_header("Maestro Configuration")

    fields = {
        "ANTHROPIC_API_KEY":         cfg.anthropic_api_key,
        "OPENAI_API_KEY":            cfg.openai_api_key,
        "MAESTRO_ANTHROPIC_MODEL":   cfg.anthropic_model,
        "MAESTRO_ANTHROPIC_MAX_TOKENS": cfg.anthropic_max_tokens,
        "MAESTRO_ANTHROPIC_TEMPERATURE": cfg.anthropic_temperature,
        "MAESTRO_OPENAI_MODEL":      cfg.openai_model,
        "MAESTRO_OPENAI_BASE_URL":   cfg.openai_base_url,
        "MAESTRO_OPENAI_TEMPERATURE": cfg.openai_temperature,
        "MAESTRO_RETRY_MAX_ATTEMPTS": cfg.retry_max_attempts,
        "MAESTRO_RETRY_MAX_DELAY":   cfg.retry_max_delay,
        "MAESTRO_LOG_LEVEL":         cfg.log_level,
        "MAESTRO_BATCH_SIZE":        cfg.batch_size,
        "MAESTRO_ERROR_THRESHOLD":   cfg.error_threshold,
        "MAESTRO_SCHEDULER_TICK":    cfg.scheduler_tick,
        "MAESTRO_SCHEDULER_MAX_WORKERS": cfg.scheduler_max_workers,
        "MAESTRO_METRICS_ENABLED":   cfg.metrics_enabled,
        "MAESTRO_PROMETHEUS_PATH":   cfg.prometheus_path,
    }
    secret_keys = {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}
    missing = []

    print()
    for key, value in fields.items():
        is_secret = key in secret_keys
        if value is None or value == "":
            display = "(not set)"
            if is_secret: missing.append(key)
        elif is_secret:
            display = f"{'*' * 8}  (set)"
        else:
            display = str(value)
        print(f"  {key:<36} {display}")

    if args.check:
        print()
        if missing:
            print(f"  ✗ Missing required keys: {', '.join(missing)}")
            print("    Set them in .env or as environment variables.")
            return 1
        else:
            print("  ✓ All required keys are set.")

    print()
    print("  Tip: run 'maestro --env /path/to/.env config' to load a specific file.")


def _add_info(sub) -> None:
    p = sub.add_parser("info", help="Show SDK version, modules and capabilities")
    p.set_defaults(func=_cmd_info)


def _cmd_info(args) -> None:
    from maestro import __version__
    _print_header("Maestro SDK")
    print(f"  Version : {__version__}")
    print(f"  Python  : {sys.version.split()[0]}")
    print()

    modules = [
        ("maestro.rules",      "Declarative rules engine"),
        ("maestro.batch",      "Record-oriented ETL batch processing"),
        ("maestro.flows",      "Composable workflow orchestration"),
        ("maestro.states",     "Deterministic finite state machine"),
        ("maestro.integration","Cross-module bridges"),
        ("maestro.retry",      "Resilience policies (retry, circuit breaker)"),
        ("maestro.observe",    "Unified observability (metrics, tracing)"),
        ("maestro.validate",   "Schema validation for Facts and Records"),
        ("maestro.events",     "Reactive pub/sub event bus"),
        ("maestro.graph",      "DAG-based parallel workflow"),
        ("maestro.async_",     "AsyncIO-native flows and batch"),
        ("maestro.saga",       "Distributed saga with compensation"),
        ("maestro.schedule",   "Cron-style scheduler"),
        ("maestro.cli",        "Command-line interface"),
    ]

    print("  Modules:")
    for mod, desc in modules:
        try:
            __import__(mod)
            status = "✓"
        except ImportError:
            status = "✗"
        print(f"    {status}  {mod:<30} {desc}")


# ════════════════════════════════════════════════════════════════════════════
#  rules
# ════════════════════════════════════════════════════════════════════════════

def _add_rules(sub) -> None:
    p = sub.add_parser("rules", help="Rules engine utilities")
    rs = p.add_subparsers(title="commands")

    # validate
    pv = rs.add_parser("validate", help="Validate a YAML rule file")
    pv.add_argument("file", help="Path to YAML rule file")
    pv.set_defaults(func=_cmd_rules_validate)

    # fire
    pf = rs.add_parser("fire", help="Fire rules against JSON facts and show results")
    pf.add_argument("file", help="Path to YAML rule file")
    pf.add_argument("--facts", "-f", default="{}", help='JSON facts object (default: {})')
    pf.add_argument("--engine", choices=["default","inference","first"], default="default")
    pf.set_defaults(func=_cmd_rules_fire)

    p.set_defaults(func=lambda a: (p.print_help(), 0))


def _cmd_rules_validate(args) -> None:
    try:
        from maestro.rules import YamlRuleFactory
        factory = YamlRuleFactory()
        rules   = factory.from_strings(open(args.file).read())
        _print_header(f"Validating: {args.file}")
        print(f"  Found {len(rules)} rule(s):")
        for r in rules:
            print(f"    ✓  {r.name!r}  (priority={r.priority})")
        print(f"\n  Result: VALID")
    except ImportError:
        print("Error: PyYAML required — pip install maestro-sdk[yaml]", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"  Result: INVALID — {exc}", file=sys.stderr)
        return 1


def _cmd_rules_fire(args) -> None:
    try:
        facts_data = json.loads(args.facts)
    except json.JSONDecodeError as e:
        print(f"Error: invalid --facts JSON: {e}", file=sys.stderr)
        return 1

    try:
        from maestro.rules import (YamlRuleFactory, Facts, Rules,
                                    DefaultRulesEngine, InferenceRulesEngine,
                                    FirstApplicableRulesEngine)
        factory     = YamlRuleFactory()
        rule_list   = factory.from_strings(open(args.file).read())
        rules       = Rules(*rule_list)
        facts       = Facts(**facts_data)

        engine_map  = {
            "default":   DefaultRulesEngine,
            "inference": InferenceRulesEngine,
            "first":     FirstApplicableRulesEngine,
        }
        engine = engine_map[args.engine]()

        _print_header(f"Firing rules from: {args.file}")
        print(f"  Engine : {type(engine).__name__}")
        print(f"  Facts  : {facts_data}")
        print(f"  Rules  : {len(rule_list)}")
        print()

        result = engine.fire(rules, facts) if args.engine in ("default","first") else None
        if args.engine == "inference":
            engine.fire(rules, facts)
            result = {}

        print("  Results:")
        if result:
            for rule, fired in result.items():
                sym = "✓ FIRED" if fired else "  skipped"
                print(f"    {sym}  {rule.name!r}")
        print()
        print("  Facts after:")
        for k, v in facts.as_map().items():
            orig = facts_data.get(k, "—")
            changed = " ← changed" if str(v) != str(orig) else ""
            print(f"    {k}: {v!r}{changed}")

    except ImportError:
        print("Error: PyYAML required — pip install maestro-sdk[yaml]", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


# ════════════════════════════════════════════════════════════════════════════
#  batch
# ════════════════════════════════════════════════════════════════════════════

def _add_batch(sub) -> None:
    p = sub.add_parser("batch", help="Batch pipeline utilities")
    bs = p.add_subparsers(title="commands")

    # run
    pr = bs.add_parser("run", help="Run a batch job defined in a YAML config")
    pr.add_argument("file", help="Path to batch YAML config")
    pr.set_defaults(func=_cmd_batch_run)

    # stats
    ps = bs.add_parser("stats", help="Show metrics from a completed job report JSON")
    ps.add_argument("file", help="Path to job report JSON")
    ps.set_defaults(func=_cmd_batch_stats)

    p.set_defaults(func=lambda a: (p.print_help(), 0))


def _cmd_batch_run(args) -> None:
    """
    YAML format::

        name: my-job
        batch_size: 100
        reader:
          type: flat_file
          path: data/input.csv
        filter:
          - header
        mapper:
          type: delimited
          fields: [id, name, total]
          delimiter: ","
        writer:
          type: stdout
    """
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML required — pip install maestro-sdk[yaml]", file=sys.stderr)
        return 1

    try:
        cfg = yaml.safe_load(open(args.file))
    except Exception as e:
        print(f"Error reading {args.file}: {e}", file=sys.stderr)
        return 1

    from maestro.batch import (
        JobBuilder, JobExecutor,
        FlatFileRecordReader, StringRecordReader, IterableRecordReader,
        HeaderRecordFilter, PassThroughRecordMapper, DelimitedRecordMapper,
        StandardOutputRecordWriter, FileRecordWriter, DevNullRecordWriter,
        CollectionRecordWriter,
    )

    name       = cfg.get("name", "cli-job")
    batch_size = int(cfg.get("batch_size", 100))

    # Reader
    reader_cfg = cfg.get("reader", {})
    rtype      = reader_cfg.get("type", "flat_file")
    if rtype == "flat_file":
        reader = FlatFileRecordReader(reader_cfg["path"],
                                      encoding=reader_cfg.get("encoding", "utf-8"))
    elif rtype == "string":
        reader = StringRecordReader(reader_cfg["data"])
    else:
        print(f"Unknown reader type: {rtype!r}", file=sys.stderr)
        return 1

    # Filters
    filters_cfg = cfg.get("filter", [])
    filters = []
    for f in (filters_cfg if isinstance(filters_cfg, list) else [filters_cfg]):
        if f == "header":
            filters.append(HeaderRecordFilter())

    # Mapper
    mapper_cfg  = cfg.get("mapper", {})
    mapper_type = mapper_cfg.get("type", "passthrough")
    if mapper_type == "delimited":
        fields    = mapper_cfg.get("fields", [])
        delimiter = mapper_cfg.get("delimiter", ",")
        mapper    = DelimitedRecordMapper(field_names=fields, delimiter=delimiter)
    else:
        mapper = PassThroughRecordMapper()

    # Writer
    writer_cfg  = cfg.get("writer", {})
    writer_type = writer_cfg.get("type", "stdout")
    if writer_type == "stdout":
        writer = StandardOutputRecordWriter()
    elif writer_type == "file":
        writer = FileRecordWriter(writer_cfg["path"])
    elif writer_type == "devnull":
        writer = DevNullRecordWriter()
    else:
        print(f"Unknown writer type: {writer_type!r}", file=sys.stderr)
        return 1

    job = (JobBuilder()
           .named(name)
           .batch_size(batch_size)
           .reader(reader)
           .mapper(mapper)
           .writer(writer)
           .build())

    for f in filters:
        job._filters.append(f)

    _print_header(f"Running batch job: {name}")
    report = JobExecutor().execute(job)
    _print_job_report(report)


def _cmd_batch_stats(args) -> None:
    try:
        data = json.load(open(args.file))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    _print_header("Batch Job Stats")
    for k, v in data.items():
        print(f"  {k}: {v}")


# ════════════════════════════════════════════════════════════════════════════
#  saga
# ════════════════════════════════════════════════════════════════════════════

def _add_saga(sub) -> None:
    p = sub.add_parser("saga", help="Saga pattern utilities")
    ss = p.add_subparsers(title="commands")

    pd = ss.add_parser("describe", help="Describe a saga (steps and compensations)")
    pd.add_argument("file", help="Path to saga YAML description")
    pd.set_defaults(func=_cmd_saga_describe)

    p.set_defaults(func=lambda a: (p.print_help(), 0))


def _cmd_saga_describe(args) -> None:
    """
    YAML saga description format::

        name: book-trip
        steps:
          - name: book-flight
            compensation: cancel-flight
          - name: book-hotel
            compensation: cancel-hotel
          - name: charge-card
            compensation: refund-card
          - name: send-confirmation
    """
    try:
        import yaml
        cfg = yaml.safe_load(open(args.file))
    except ImportError:
        print("Error: PyYAML required — pip install maestro-sdk[yaml]", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading {args.file}: {e}", file=sys.stderr)
        return 1

    name  = cfg.get("name", "saga")
    steps = cfg.get("steps", [])

    _print_header(f"Saga: {name}")
    print(f"  Steps ({len(steps)}):\n")
    for i, step in enumerate(steps, 1):
        sname = step.get("name", f"step-{i}")
        comp  = step.get("compensation")
        print(f"    {i:>2}. {sname}")
        if comp:
            print(f"         ↩  compensation: {comp}")
        else:
            print(f"         ─  (no compensation)")

    print()
    print("  On failure at step N, compensations run in reverse:")
    compensatable = [(s["name"], s["compensation"])
                     for s in reversed(steps)
                     if s.get("compensation")]
    if compensatable:
        for sname, comp in compensatable:
            print(f"    {comp} (undoes {sname!r})")
    else:
        print("    (no compensations defined)")


# ════════════════════════════════════════════════════════════════════════════
#  schedule
# ════════════════════════════════════════════════════════════════════════════

def _add_schedule(sub) -> None:
    p = sub.add_parser("schedule", help="Scheduler utilities")
    ss = p.add_subparsers(title="commands")

    pd = ss.add_parser("demo", help="Run a live scheduler demo for 10 seconds")
    pd.add_argument("--duration", type=float, default=10.0,
                    help="Demo duration in seconds (default: 10)")
    pd.set_defaults(func=_cmd_schedule_demo)

    pc = ss.add_parser("cron", help="Show next N fire times for a cron expression")
    pc.add_argument("expression", help='Cron expression (e.g. "*/5 * * * *")')
    pc.add_argument("--count", "-n", type=int, default=5, help="Number of fire times to show")
    pc.set_defaults(func=_cmd_schedule_cron)

    p.set_defaults(func=lambda a: (p.print_help(), 0))


def _cmd_schedule_demo(args) -> None:
    import time
    from maestro.schedule import Scheduler, IntervalTrigger, ImmediateTrigger
    from maestro.flows._work import WorkContext, LambdaWork

    log = []
    counter = [0]

    def tick(ctx):
        counter[0] += 1
        msg = f"[tick #{counter[0]}] at {__import__('datetime').datetime.now().strftime('%H:%M:%S')}"
        log.append(msg)
        print(f"  {msg}")

    _print_header(f"Scheduler demo ({args.duration:.0f}s)")
    print(f"  Task 'ticker' fires every 2 seconds\n")

    with Scheduler(tick_seconds=0.5) as sched:
        sched.add("ticker",
                  work    = LambdaWork(tick, "tick"),
                  trigger = IntervalTrigger(seconds=2))
        sched.add("once",
                  work    = LambdaWork(lambda c: print("  [once] fired immediately!"), "once"),
                  trigger = ImmediateTrigger())
        time.sleep(args.duration)

    print()
    print(f"  Completed {counter[0]} tick(s).")
    statuses = sched.status()
    for s in statuses:
        print(f"  Task '{s['name']}': {s['run_count']} run(s), last_ok={s['last_ok']}")


def _cmd_schedule_cron(args) -> None:
    import datetime
    from maestro.schedule import CronTrigger
    try:
        trigger = CronTrigger(args.expression)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    _print_header(f"Cron: {args.expression!r}")
    now  = datetime.datetime.now()
    last = None
    for i in range(args.count):
        nxt  = trigger.next_fire_time(last, now)
        print(f"  {i+1:>2}.  {nxt.strftime('%Y-%m-%d %H:%M')}")
        last = nxt


# ════════════════════════════════════════════════════════════════════════════
#  fsm
# ════════════════════════════════════════════════════════════════════════════

def _add_fsm(sub) -> None:
    p = sub.add_parser("fsm", help="Finite state machine utilities")
    fs = p.add_subparsers(title="commands")

    pd = fs.add_parser("dot", help="Generate Graphviz DOT from a YAML FSM description")
    pd.add_argument("file", help="Path to FSM YAML file")
    pd.set_defaults(func=_cmd_fsm_dot)

    pr = fs.add_parser("run", help="Fire events against a YAML-defined FSM interactively")
    pr.add_argument("file", help="Path to FSM YAML file")
    pr.add_argument("--events", nargs="+", help="Event names to fire (space-separated)")
    pr.set_defaults(func=_cmd_fsm_run)

    p.set_defaults(func=lambda a: (p.print_help(), 0))


def _cmd_fsm_dot(args) -> None:
    """
    YAML FSM format::

        name: turnstile
        initial: locked
        states: [locked, unlocked]
        transitions:
          - name: unlock
            from: locked
            event: CoinEvent
            to: unlocked
          - name: lock
            from: unlocked
            event: PushEvent
            to: locked
    """
    try:
        import yaml
        cfg = yaml.safe_load(open(args.file))
    except ImportError:
        print("Error: PyYAML required — pip install maestro-sdk[yaml]", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    name    = cfg.get("name", "FSM")
    initial = cfg.get("initial", "")
    trans   = cfg.get("transitions", [])

    lines = [f'digraph "{name}" {{', '  rankdir=LR;']
    lines.append(f'  "{initial}" [shape=doublecircle];')
    for t in trans:
        label = t.get("name", t.get("event", "?"))
        event = t.get("event", "?")
        lines.append(f'  "{t["from"]}" -> "{t["to"]}" [label="{label}\\n({event})"];')
    lines.append("}")

    _print_header(f"FSM DOT: {name}")
    print("\n".join(lines))
    print()
    print("  Paste into: https://dreampuf.github.io/GraphvizOnline/")


def _cmd_fsm_run(args) -> None:
    try:
        import yaml
        cfg = yaml.safe_load(open(args.file))
    except ImportError:
        print("Error: PyYAML required", file=sys.stderr); return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr); return 1

    from maestro.states import (
        State, Event, FiniteStateMachineBuilder, TransitionBuilder,
        LambdaEventHandler, NoSuchTransitionException,
    )

    state_names = cfg.get("states", [])
    initial_n   = cfg.get("initial", state_names[0] if state_names else "start")
    states      = {n: State(n) for n in state_names}
    initial     = states[initial_n]
    events      = {t["event"]: type(t["event"], (Event,), {}) for t in cfg.get("transitions", [])}

    builder = FiniteStateMachineBuilder(states=set(states.values()), initial_state=initial)
    for t in cfg.get("transitions", []):
        ecls = events[t["event"]]
        builder.register_transition(
            TransitionBuilder()
            .name(t.get("name", f"{t['from']}->{t['to']}"))
            .source_state(states[t["from"]])
            .event_type(ecls)
            .event_handler(LambdaEventHandler(
                lambda e, f=t["from"], to=t["to"], n=t.get("name",""):
                    print(f"  Transition: {f} → {to}" + (f" ({n})" if n else ""))
            ))
            .target_state(states[t["to"]])
            .build()
        )
    fsm = builder.build()

    event_list = args.events or []
    _print_header(f"FSM: {cfg.get('name', 'FSM')}")
    print(f"  Initial state: {fsm.current_state}")
    print()

    if not event_list:
        print("  No events specified. Use --events EventName1 EventName2 ...")
        print(f"  Available events: {list(events.keys())}")
        return

    for ev_name in event_list:
        if ev_name not in events:
            print(f"  Unknown event: {ev_name!r}", file=sys.stderr)
            continue
        try:
            new_state = fsm.fire(events[ev_name]())
            print(f"  After {ev_name}: {new_state}")
        except NoSuchTransitionException as e:
            print(f"  No transition: {e}", file=sys.stderr)


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _print_header(title: str) -> None:
    print(f"\n{'─' * 54}")
    print(f"  {title}")
    print(f"{'─' * 54}")


def _print_job_report(report) -> None:
    m = report.metrics
    print(f"\n  Job Report: {report.status.value}")
    print(f"  ─────────────────────────────")
    print(f"  Total   : {m.total_count}")
    print(f"  Written : {m.written_count}")
    print(f"  Filtered: {m.filtered_count}")
    print(f"  Skipped : {m.skipped_count}")
    print(f"  Failed  : {m.failed_count}")
    print(f"  Duration: {m.duration_seconds:.3f}s")


__all__ = ["main"]
