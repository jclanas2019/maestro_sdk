"""
examples/p3_features.py — Priority 3 features: saga, schedule, cli.

Run: python examples/p3_features.py
"""
import sys, os, time, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import maestro
from maestro.saga import SagaBuilder, SagaStatus, SagaListener, saga_step, LoggingSagaListener
from maestro.schedule import (
    Scheduler, CronTrigger, IntervalTrigger, ImmediateTrigger, OnceTrigger,
)
from maestro.cli import main as cli_main

SEP = "═" * 62


# ════════════════════════════════════════════════════════════════════
#  1. maestro.saga — distributed saga pattern
# ════════════════════════════════════════════════════════════════════
print(SEP); print("1. maestro.saga"); print(SEP)

# Shared helpers
def _ok(name, log=None):
    def fn(ctx):
        if log is not None: log.append(name)
        return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)
    return maestro.LambdaWork(fn, name=name)

def _fail(name):
    return maestro.LambdaWork(
        lambda ctx: maestro.DefaultWorkReport(
            maestro.WorkStatus.FAILED, ctx, error=Exception(f"{name} failed")), name=name)

# ── 1a. Happy path ────────────────────────────────────────────────
print("  1a. Happy path — all steps succeed:")
log1a = []
saga1a = (SagaBuilder()
          .named("book-trip")
          .quiet()
          .step("book-flight",   _ok("book-flight",   log1a), _ok("cancel-flight", log1a))
          .step("book-hotel",    _ok("book-hotel",    log1a), _ok("cancel-hotel",  log1a))
          .step("charge-card",   _ok("charge-card",   log1a), _ok("refund-card",   log1a))
          .step("send-email",    _ok("send-email",    log1a))
          .build())
r1a = saga1a.execute(maestro.WorkContext())
print(f"  Status: {r1a.saga_status.value}")
print(f"  Steps completed: {r1a.succeeded_steps}")
print(str(r1a))

# ── 1b. Failure + compensation ────────────────────────────────────
print("\n  1b. Failure at 'charge-card' → compensate in reverse:")
log1b = []
saga1b = (SagaBuilder()
          .named("book-trip-fail")
          .quiet()
          .step("book-flight",   _ok("book-flight",   log1b), _ok("cancel-flight", log1b))
          .step("book-hotel",    _ok("book-hotel",    log1b), _ok("cancel-hotel",  log1b))
          .step("charge-card",   _fail("charge-card"),        _ok("refund-card",   log1b))
          .step("send-email",    _ok("send-email",    log1b))
          .build())
r1b = saga1b.execute(maestro.WorkContext())
print(f"  Status: {r1b.saga_status.value}")
print(f"  Failed at: {r1b.failed_step!r}")
print(f"  Compensated: {r1b.compensated_steps}")
print(str(r1b))

# ── 1c. Context propagation through steps ─────────────────────────
print("\n  1c. WorkContext flows through steps and compensations:")
saga1c = (SagaBuilder()
          .named("with-context")
          .quiet()
          .step("reserve",
                work=maestro.LambdaWork(
                    lambda c: c.put("reservation_id", "RSV-001"),
                    "reserve"),
                compensation=maestro.LambdaWork(
                    lambda c: print(f"    cancelling reservation: {c.get('reservation_id')}"),
                    "cancel"))
          .step("charge",
                work=maestro.LambdaWork(
                    lambda c: c.put("payment_id", "PAY-42"),
                    "charge"),
                compensation=maestro.LambdaWork(
                    lambda c: print(f"    refunding payment: {c.get('payment_id')}"),
                    "refund"))
          .step("fail-here",  _fail("fail-here"))
          .build())
saga1c.execute(maestro.WorkContext())

# ── 1d. Saga as a Work in a SequentialFlow ───────────────────────
print("\n  1d. Saga embedded as a Work inside a SequentialFlow:")
log1d = []
saga1d = (SagaBuilder()
          .named("order-saga")
          .quiet()
          .step("validate",   _ok("validate",   log1d))
          .step("reserve",    _ok("reserve",    log1d), _ok("release", log1d))
          .step("ship",       _ok("ship",       log1d))
          .build())

flow1d = (maestro.aNewSequentialFlow()
          .named("order-flow")
          .execute(saga1d)
          .then(maestro.LambdaWork(lambda c: log1d.append("notify"), "notify"))
          .build())
r1d = maestro.WorkFlowEngine().run(flow1d, maestro.WorkContext())
print(f"  Flow status: {r1d.status.value}")
print(f"  Execution order: {log1d}")

# ── 1e. saga_step() convenience + custom listener ─────────────────
print("\n  1e. saga_step() helper + custom SagaListener:")

class AuditListener(SagaListener):
    def on_step_started(self, step, ctx):
        print(f"    ▶ {step.name}")
    def on_compensation_started(self, step, ctx):
        print(f"    ↩ compensating {step.name}")
    def on_saga_compensated(self, report):
        print(f"    ✗ saga ended: {report.saga_status.value}")

saga1e = (SagaBuilder()
          .named("audited-saga")
          .quiet()
          .add_listener(AuditListener())
          .step("step-A",
                work=_ok("step-A"),
                compensation=_ok("undo-A"))
          .step("step-B", _fail("step-B"), _ok("undo-B"))
          .build())
saga1e.execute(maestro.WorkContext())


# ════════════════════════════════════════════════════════════════════
#  2. maestro.schedule — cron-style scheduler
# ════════════════════════════════════════════════════════════════════
print(); print(SEP); print("2. maestro.schedule"); print(SEP)

# ── 2a. Cron expression preview ───────────────────────────────────
print("  2a. Next fire times for common cron expressions:")
expressions = [
    ("*/5 * * * *",  "every 5 minutes"),
    ("0 9-17 * * 1-5","weekdays 9-17 on the hour"),
    ("0 0 1 * *",    "1st of every month at midnight"),
    ("30 6 * * 0",   "Sundays at 06:30"),
]
now = datetime.datetime.now()
for expr, label in expressions:
    trigger = CronTrigger(expr)
    nxt     = trigger.next_fire_time(None, now)
    print(f"  {expr:<22}  ({label})")
    print(f"    next: {nxt.strftime('%Y-%m-%d %H:%M')}")

# ── 2b. IntervalTrigger ───────────────────────────────────────────
print("\n  2b. IntervalTrigger — fire every 0.15s for 0.5s:")
tick_log = []
counter = [0]

class CounterListener(maestro.ScheduleListener):
    def on_task_succeeded(self, task, run):
        tick_log.append(f"tick #{task.run_count} at {run.finished_at.strftime('%S.%f')[:5]}s")

with Scheduler(tick_seconds=0.05, listeners=[CounterListener()]) as sched:
    sched.add("ticker",
              work    = maestro.LambdaWork(lambda c: None, "tick"),
              trigger = IntervalTrigger(seconds=0.15))
    time.sleep(0.5)

for entry in tick_log:
    print(f"    {entry}")
print(f"  Total ticks: {len(tick_log)}")

# ── 2c. Multiple tasks with different triggers ─────────────────────
print("\n  2c. Mixed triggers — slow task, fast task, one-shot:")
mixed_log = []
with Scheduler(tick_seconds=0.05) as sched:
    sched.add("slow",
              work    = maestro.LambdaWork(lambda c: mixed_log.append("slow"), "slow"),
              trigger = IntervalTrigger(seconds=0.3))
    sched.add("fast",
              work    = maestro.LambdaWork(lambda c: mixed_log.append("fast"), "fast"),
              trigger = IntervalTrigger(seconds=0.1))
    sched.add("once",
              work    = maestro.LambdaWork(lambda c: mixed_log.append("ONCE"), "once"),
              trigger = ImmediateTrigger())
    time.sleep(0.5)
    status = sched.status()

print(f"  Events: {mixed_log}")
for s in status:
    print(f"  Task '{s['name']}': {s['run_count']} run(s), state={s['state']}")

# ── 2d. Pause/resume ──────────────────────────────────────────────
print("\n  2d. Pause and resume a task:")
pause_log = []
with Scheduler(tick_seconds=0.05) as sched:
    sched.add("p",
              work    = maestro.LambdaWork(lambda c: pause_log.append("●"), "p"),
              trigger = IntervalTrigger(seconds=0.1))
    time.sleep(0.25)
    sched.pause("p")
    n1 = len(pause_log)
    print(f"  Before pause: {n1} fires")
    time.sleep(0.25)
    n2 = len(pause_log)
    print(f"  While paused: {n2 - n1} fires (should be 0)")
    sched.resume("p")
    time.sleep(0.25)
    n3 = len(pause_log)
    print(f"  After resume: {n3 - n2} more fires")

# ── 2e. Scheduling a batch Job ────────────────────────────────────
print("\n  2e. Scheduling a batch Job:")
batch_sink = []
batch_job = (maestro.JobBuilder()
             .named("scheduled-etl")
             .reader(maestro.IterableRecordReader(["a", "b", "c"]))
             .mapper(maestro.LambdaRecordMapper(str.upper))
             .writer(maestro.CollectionRecordWriter(batch_sink))
             .build())

with Scheduler(tick_seconds=0.05) as sched:
    sched.add("etl", job=batch_job, trigger=ImmediateTrigger())
    time.sleep(0.3)

print(f"  Batch output: {batch_sink}")


# ════════════════════════════════════════════════════════════════════
#  3. maestro.cli — command-line interface
# ════════════════════════════════════════════════════════════════════
print(); print(SEP); print("3. maestro.cli"); print(SEP)

import io, contextlib

def run_cli(*args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try: cli_main(list(args))
        except SystemExit: pass
    return buf.getvalue()

# ── 3a. info ──────────────────────────────────────────────────────
print("  3a. maestro info:")
out = run_cli("info")
for line in out.split('\n')[:12]:
    if line.strip(): print(f"    {line.rstrip()}")

# ── 3b. rules validate ────────────────────────────────────────────
import tempfile, pathlib
print("\n  3b. maestro rules validate <yaml>:")
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    f.write("name: umbrella\ncondition: rain == True\nactions:\n  - \"print('take umbrella')\"\n")
    yaml_path = f.name

out = run_cli("rules", "validate", yaml_path)
for line in out.split('\n'):
    if line.strip(): print(f"    {line.rstrip()}")

# ── 3c. rules fire ────────────────────────────────────────────────
print("\n  3c. maestro rules fire <yaml> --facts '{\"rain\": true}':")
out = run_cli("rules", "fire", yaml_path, "--facts", '{"rain": true}')
for line in out.split('\n'):
    if line.strip(): print(f"    {line.rstrip()}")
pathlib.Path(yaml_path).unlink(missing_ok=True)

# ── 3d. schedule cron ─────────────────────────────────────────────
print("\n  3d. maestro schedule cron '0 9 * * 1-5' --count 4:")
out = run_cli("schedule", "cron", "0 9 * * 1-5", "--count", "4")
for line in out.split('\n'):
    if line.strip(): print(f"    {line.rstrip()}")

# ── 3e. saga describe ─────────────────────────────────────────────
print("\n  3e. maestro saga describe <yaml>:")
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    f.write(
        "name: book-trip\nsteps:\n"
        "  - name: book-flight\n    compensation: cancel-flight\n"
        "  - name: book-hotel\n    compensation: cancel-hotel\n"
        "  - name: charge-card\n    compensation: refund-card\n"
        "  - name: send-email\n"
    )
    saga_path = f.name

out = run_cli("saga", "describe", saga_path)
for line in out.split('\n'):
    if line.strip(): print(f"    {line.rstrip()}")
pathlib.Path(saga_path).unlink(missing_ok=True)

# ── 3f. fsm dot ───────────────────────────────────────────────────
print("\n  3f. maestro fsm dot <yaml>:")
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    f.write(
        "name: turnstile\ninitial: locked\nstates: [locked, unlocked]\n"
        "transitions:\n"
        "  - name: unlock\n    from: locked\n    event: CoinEvent\n    to: unlocked\n"
        "  - name: lock\n    from: unlocked\n    event: PushEvent\n    to: locked\n"
    )
    fsm_path = f.name

out = run_cli("fsm", "dot", fsm_path)
for line in out.split('\n'):
    if line.strip(): print(f"    {line.rstrip()}")
pathlib.Path(fsm_path).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════
#  4. All three P3 together
# ════════════════════════════════════════════════════════════════════
print(); print(SEP); print("4. All three P3 together"); print(SEP)
print("  Pattern: Scheduler fires a Saga on interval → Saga publishes results\n")

from maestro.events import EventBus, Topic

bus    = EventBus()
saga_results = []
bus.subscribe_fn("saga.completed",   lambda m: saga_results.append(("ok",   m.payload)))
bus.subscribe_fn("saga.compensated", lambda m: saga_results.append(("comp", m.payload)))

order_counter = [0]

def build_order_saga():
    order_counter[0] += 1
    order_id = f"ORD-{order_counter[0]:03d}"

    def reserve(ctx):
        ctx.put("order_id", order_id)
        print(f"    [{order_id}] reserve inventory")
    def release(ctx):
        print(f"    [{order_id}] release inventory")
    def charge(ctx):
        if order_counter[0] % 2 == 0:
            raise Exception("card declined")
        print(f"    [{order_id}] charge card")
    def refund(ctx):
        print(f"    [{order_id}] refund card")
    def notify(ctx):
        bus.publish("saga.completed", {"order_id": ctx.get("order_id")})
        print(f"    [{order_id}] customer notified")

    return (SagaBuilder()
            .named(f"order-saga-{order_id}")
            .quiet()
            .step("reserve", maestro.LambdaWork(reserve, "reserve"),
                             maestro.LambdaWork(release, "release"))
            .step("charge",  maestro.LambdaWork(charge, "charge"),
                             maestro.LambdaWork(refund, "refund"))
            .step("notify",  maestro.LambdaWork(notify, "notify"))
            .build())

# Each scheduler tick runs a fresh Saga
def run_saga(ctx):
    saga = build_order_saga()
    report = saga.execute(ctx)
    if report.saga_status != SagaStatus.COMPLETED:
        bus.publish("saga.compensated", {"saga": saga.get_name(),
                                          "failed_at": report.failed_step})

with Scheduler(tick_seconds=0.05) as sched:
    sched.add("order-processor",
              work    = maestro.LambdaWork(run_saga, "run-saga"),
              trigger = IntervalTrigger(seconds=0.2))
    time.sleep(0.7)

print(f"\n  Results: {len(saga_results)} saga(s) ran")
for kind, payload in saga_results:
    if kind == "ok":
        print(f"    ✓ {payload['order_id']} completed")
    else:
        print(f"    ✗ {payload['saga']} compensated (failed at {payload.get('failed_at')})")

print(); print(SEP); print("All P3 feature examples completed."); print(SEP)
