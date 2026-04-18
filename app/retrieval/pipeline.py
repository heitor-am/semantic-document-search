from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import structlog

from app.retrieval.context import Context

logger = structlog.get_logger()


class StageError(Exception):
    """Raised by a `Stage` when it can't produce output.

    Carries the stage name and the underlying cause so the pipeline can log
    the failure consistently and the endpoint can surface which stage
    tripped (useful when rerank is off but dense still works, etc).
    """

    def __init__(self, stage_name: str, cause: BaseException | None = None) -> None:
        self.stage_name = stage_name
        self.cause = cause
        detail = f": {cause}" if cause is not None else ""
        super().__init__(f"stage {stage_name!r} failed{detail}")


class Stage(ABC):
    """One step of the retrieval pipeline.

    Subclasses override `name` (descriptive, used in logs) and `optional`
    (whether a failure should abort the whole pipeline or just be noted).
    Optional stages let query rewriting and reranking fall back gracefully
    — if OpenRouter is slow or a reranker model errors, the user still
    gets hybrid-search results instead of a 500.

    `run` is async so stages can call out to Qdrant / OpenRouter without
    blocking the event loop.
    """

    name: str = "stage"
    optional: bool = False

    @abstractmethod
    async def run(self, ctx: Context) -> Context: ...


class Pipeline:
    """Composes a sequence of `Stage`s over a `Context`.

    The pipeline is a plain sequence — no branching, no concurrency — so the
    flow is easy to reason about and to read in logs. Parallelism (running
    dense+sparse simultaneously) lives inside a single "fan-out" stage, not
    here, so the top-level graph stays linear.

    On each stage failure:
        - if `stage.optional` is True, the StageError is recorded on the
          context and the pipeline continues with the next stage;
        - otherwise the StageError is re-raised, halting the pipeline.

    Unhandled exceptions inside a stage are wrapped in StageError (same
    behaviour) so stages don't have to do their own wrapping.
    """

    def __init__(self, stages: Sequence[Stage]) -> None:
        self._stages: tuple[Stage, ...] = tuple(stages)

    @property
    def stages(self) -> tuple[Stage, ...]:
        return self._stages

    async def run(self, ctx: Context) -> Context:
        for stage in self._stages:
            log = logger.bind(stage=stage.name, optional=stage.optional)
            try:
                ctx = await stage.run(ctx)
            except StageError as exc:
                if stage.optional:
                    log.warning("retrieval.stage.failed_optional", error=str(exc))
                    ctx.errors.append(exc)
                    continue
                log.error("retrieval.stage.failed", error=str(exc))
                raise
            except Exception as exc:
                wrapped = StageError(stage.name, cause=exc)
                if stage.optional:
                    log.warning("retrieval.stage.failed_optional", error=str(wrapped))
                    ctx.errors.append(wrapped)
                    continue
                log.error("retrieval.stage.failed", error=str(wrapped))
                raise wrapped from exc
        return ctx
