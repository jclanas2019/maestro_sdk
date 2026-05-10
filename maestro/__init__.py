"""
Maestro SDK — The simple, unified automation SDK for Python.

Four modules, one SDK:

    maestro.rules    → declarative rules engine
    maestro.batch    → record-oriented ETL batch processing
    maestro.flows    → composable workflow orchestration
    maestro.states   → deterministic finite state machines
    maestro.integration → cross-module bridges

Install::

    pip install maestro-sdk          # core (no extra deps)
    pip install "maestro-sdk[yaml]"  # + YAML rule files

Quick start::

    import maestro

    # Rules
    facts  = maestro.Facts(rain=True)
    rules  = maestro.Rules()
    engine = maestro.DefaultRulesEngine()

    # Batch
    job = (maestro.JobBuilder()
           .reader(maestro.StringRecordReader("a\\nb\\nc"))
           .writer(maestro.StandardOutputRecordWriter())
           .build())
    maestro.JobExecutor().execute(job)

    # Flows
    flow = (maestro.SequentialFlow.Builder()
            .execute(maestro.LambdaWork(lambda ctx: print("step 1")))
            .then(maestro.LambdaWork(lambda ctx: print("step 2")))
            .build())
    maestro.WorkFlowEngine().run(flow, maestro.WorkContext())

    # States
    locked, unlocked = maestro.State("locked"), maestro.State("unlocked")
    class CoinEvent(maestro.Event): pass
    fsm = (maestro.FiniteStateMachineBuilder(states={locked, unlocked}, initial_state=locked)
           .register_transition(
               maestro.TransitionBuilder().source_state(locked).event_type(CoinEvent).target_state(unlocked).build()
           ).build())
    fsm.fire(CoinEvent())

Version: 1.0.0
"""
__version__     = "1.0.0"
__author__      = "Maestro SDK"
__description__ = "The simple, unified automation SDK for Python"

# ── rules ────────────────────────────────────────────────────────────────── #
from maestro.rules import (
    Facts, Rule, BasicRule, AnnotatedRule, wrap_rule,
    rule, condition, action, Rules, RuleBuilder,
    RuleListener, RulesEngineListener, RulesEngineParameters,
    DefaultRulesEngine, InferenceRulesEngine, FirstApplicableRulesEngine,
    CompositeRule, UnitRuleGroup, ConditionalRuleGroup, ActivationRuleGroup,
    ExpressionRule, YamlRuleFactory,
)

# ── batch ─────────────────────────────────────────────────────────────────── #
from maestro.batch import (
    Header, Record, Batch,
    RecordReader, IterableRecordReader, InMemoryRecordReader,
    FlatFileRecordReader, StringRecordReader, CsvDictRecordReader, JsonLinesRecordReader,
    RecordFilter, HeaderRecordFilter, PredicateRecordFilter,
    PoisonRecordFilter, RecordNumberRangeFilter,
    RecordMapper, PassThroughRecordMapper, LambdaRecordMapper,
    DelimitedRecordMapper, FieldSetMapper,
    RecordProcessingException, RecordProcessor,
    LambdaRecordProcessor, CompositeRecordProcessor, FilteringRecordProcessor,
    RecordMarshaller, ToStringMarshaller, LambdaMarshaller, JsonMarshaller, CsvMarshaller,
    RecordWriter, StandardOutputRecordWriter, FileRecordWriter,
    CollectionRecordWriter, StringIORecordWriter, DevNullRecordWriter,
    JobListener, BatchListener, RecordReaderListener, PipelineListener,
    JobParameters, JobStatus, JobMetrics, JobReport, Job, JobBuilder, JobExecutor,
)

# ── flows ─────────────────────────────────────────────────────────────────── #
from maestro.flows import (
    WorkStatus, WorkContext, WorkReport, DefaultWorkReport,
    Work, NoOpWork, LambdaWork, WorkReportPredicate,
    SequentialFlow, ConditionalFlow, ParallelFlow, ParallelFlowReport, RepeatFlow,
    WorkFlowEngine,
    aNewSequentialFlow, aNewConditionalFlow, aNewParallelFlow,
    aNewRepeatFlow, aNewWorkFlowEngine,
)

# ── states ────────────────────────────────────────────────────────────────── #
from maestro.states import (
    State, Event, AbstractEvent, EventHandler, LambdaEventHandler,
    NoSuchTransitionException, FSMException,
    Transition, TransitionBuilder,
    TransitionListener,
    FiniteStateMachine, FiniteStateMachineBuilder,
)

# ── integration ───────────────────────────────────────────────────────────── #
from maestro.integration import (
    RuleSetWork, BatchWork,
    FSMGuardWork, FSMTransitionWork,
    RuleBasedFilter, RuleBasedProcessor,
)

__all__ = [
    # rules
    "Facts", "Rule", "BasicRule", "AnnotatedRule", "wrap_rule",
    "rule", "condition", "action", "Rules", "RuleBuilder",
    "RuleListener", "RulesEngineListener", "RulesEngineParameters",
    "DefaultRulesEngine", "InferenceRulesEngine", "FirstApplicableRulesEngine",
    "CompositeRule", "UnitRuleGroup", "ConditionalRuleGroup", "ActivationRuleGroup",
    "ExpressionRule", "YamlRuleFactory",
    # batch
    "Header", "Record", "Batch",
    "RecordReader", "IterableRecordReader", "InMemoryRecordReader",
    "FlatFileRecordReader", "StringRecordReader", "CsvDictRecordReader", "JsonLinesRecordReader",
    "RecordFilter", "HeaderRecordFilter", "PredicateRecordFilter",
    "PoisonRecordFilter", "RecordNumberRangeFilter",
    "RecordMapper", "PassThroughRecordMapper", "LambdaRecordMapper",
    "DelimitedRecordMapper", "FieldSetMapper",
    "RecordProcessingException", "RecordProcessor",
    "LambdaRecordProcessor", "CompositeRecordProcessor", "FilteringRecordProcessor",
    "RecordMarshaller", "ToStringMarshaller", "LambdaMarshaller", "JsonMarshaller", "CsvMarshaller",
    "RecordWriter", "StandardOutputRecordWriter", "FileRecordWriter",
    "CollectionRecordWriter", "StringIORecordWriter", "DevNullRecordWriter",
    "JobListener", "BatchListener", "RecordReaderListener", "PipelineListener",
    "JobParameters", "JobStatus", "JobMetrics", "JobReport", "Job", "JobBuilder", "JobExecutor",
    # flows
    "WorkStatus", "WorkContext", "WorkReport", "DefaultWorkReport",
    "Work", "NoOpWork", "LambdaWork", "WorkReportPredicate",
    "SequentialFlow", "ConditionalFlow", "ParallelFlow", "ParallelFlowReport", "RepeatFlow",
    "WorkFlowEngine",
    "aNewSequentialFlow", "aNewConditionalFlow", "aNewParallelFlow",
    "aNewRepeatFlow", "aNewWorkFlowEngine",
    # states
    "State", "Event", "AbstractEvent", "EventHandler", "LambdaEventHandler",
    "NoSuchTransitionException", "FSMException",
    "Transition", "TransitionBuilder", "TransitionListener",
    "FiniteStateMachine", "FiniteStateMachineBuilder",
    # integration
    "RuleSetWork", "BatchWork",
    "FSMGuardWork", "FSMTransitionWork",
    "RuleBasedFilter", "RuleBasedProcessor",
]

# ── retry ─────────────────────────────────────────────────────────────────── #
from maestro.retry import (
    BackoffStrategy, NoBackoff, ConstantBackoff, LinearBackoff,
    ExponentialBackoff, JitteredBackoff,
    CircuitBreaker, CircuitState, CircuitOpenError,
    RetryPolicy, RetryWork, RetryableReader,
    MaxAttemptsExceeded,
    retry, retryable, execute_with_retry,
)

# ── observe ───────────────────────────────────────────────────────────────── #
from maestro.observe import (
    MetricEvent,
    Observer, LoggingObserver, PrintObserver,
    CompositeObserver, InMemoryObserver,
    MaestroObserver, timed,
)

# ── validate ──────────────────────────────────────────────────────────────── #
from maestro.validate import (
    ValidationError, ValidationResult, SchemaViolation,
    Validator, Required, Range, Length, Pattern, OneOf, Custom, NotEmpty,
    FieldSchema, field, Schema,
    ValidatedFacts, SchemaFilter, SchemaProcessor, ValidatedWork,
)

# ── events ────────────────────────────────────────────────────────────────── #
from maestro.events import (
    Message, Subscriber, FunctionSubscriber, Subscription,
    EventBus, AsyncEventBus, Topic,
    EventPublisherWork, EventSubscriberWork,
    FSMEventBridge, RuleEventRouter, BusRecordReader,
)

# ── graph ─────────────────────────────────────────────────────────────────── #
from maestro.graph import (
    CyclicDependencyError, UnknownDependencyError,
    NodeReport, GraphReport, GraphFlow, GraphBuilder,
)

# ── async_ ────────────────────────────────────────────────────────────────── #
from maestro.async_ import (
    AsyncWork, AsyncNoOpWork, AsyncLambdaWork,
    AsyncSequentialFlow, AsyncConditionalFlow,
    AsyncParallelFlow, AsyncRepeatFlow,
    AsyncWorkFlowEngine,
    AsyncRecordReader, AsyncIterableReader,
    AsyncRecordWriter, AsyncCollectionWriter, AsyncDevNullWriter,
    AsyncJob, AsyncJobBuilder,
    sync_to_async, async_to_sync,
)

# ── saga ──────────────────────────────────────────────────────────────────── #
from maestro.saga import (
    SagaStatus, SagaStep, StepResult, CompensationResult,
    SagaReport, SagaListener, LoggingSagaListener,
    Saga, SagaBuilder, saga_step,
)

# ── schedule ──────────────────────────────────────────────────────────────── #
from maestro.schedule import (
    Trigger, CronTrigger, IntervalTrigger, OnceTrigger, ImmediateTrigger,
    TaskState, TaskRun, ScheduledTask, ScheduleListener, Scheduler,
)

# ── config ────────────────────────────────────────────────────────────────── #
from maestro.config import (
    load_env, get_config, MaestroConfig,
    global_observer, reset_global_observer,
    configure_logging,
    make_anthropic_from_config, make_openai_from_config,
)
