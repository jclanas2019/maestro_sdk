"""maestro.flows — composable workflow engine."""
from maestro.flows._work       import (WorkStatus, WorkContext, WorkReport, DefaultWorkReport,
                                        Work, NoOpWork, LambdaWork)
from maestro.flows._predicate  import WorkReportPredicate
from maestro.flows._sequential import SequentialFlow
from maestro.flows._conditional import ConditionalFlow
from maestro.flows._parallel   import ParallelFlow, ParallelFlowReport
from maestro.flows._repeat     import RepeatFlow
from maestro.flows._engine     import WorkFlowEngine


def aNewSequentialFlow()  -> SequentialFlow.Builder:  return SequentialFlow.Builder()
def aNewConditionalFlow() -> ConditionalFlow.Builder: return ConditionalFlow.Builder()
def aNewParallelFlow()    -> ParallelFlow.Builder:    return ParallelFlow.Builder()
def aNewRepeatFlow()      -> RepeatFlow.Builder:      return RepeatFlow.Builder()
def aNewWorkFlowEngine()  -> WorkFlowEngine.Builder:  return WorkFlowEngine.Builder()


__all__ = [
    "WorkStatus", "WorkContext", "WorkReport", "DefaultWorkReport",
    "Work", "NoOpWork", "LambdaWork", "WorkReportPredicate",
    "SequentialFlow", "ConditionalFlow", "ParallelFlow", "ParallelFlowReport", "RepeatFlow",
    "WorkFlowEngine",
    "aNewSequentialFlow", "aNewConditionalFlow", "aNewParallelFlow",
    "aNewRepeatFlow", "aNewWorkFlowEngine",
]
