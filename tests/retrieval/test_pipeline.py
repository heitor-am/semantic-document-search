from __future__ import annotations

import pytest

from app.retrieval.context import Candidate, Context
from app.retrieval.pipeline import Pipeline, Stage, StageError


class RecordingStage(Stage):
    """Test double that appends its name to a shared log when it runs."""

    def __init__(self, name: str, log: list[str], optional: bool = False) -> None:
        self.name = name
        self.optional = optional
        self._log = log

    async def run(self, ctx: Context) -> Context:
        self._log.append(self.name)
        ctx.state[self.name] = "ran"
        return ctx


class RaisingStage(Stage):
    """Test double that always raises on run."""

    def __init__(self, name: str, *, optional: bool, error: Exception | None = None) -> None:
        self.name = name
        self.optional = optional
        self._error = error or RuntimeError(f"{name} blew up")

    async def run(self, ctx: Context) -> Context:
        raise self._error


class StageErrorRaising(Stage):
    """Test double that raises a pre-wrapped StageError."""

    def __init__(self, name: str, *, optional: bool) -> None:
        self.name = name
        self.optional = optional

    async def run(self, ctx: Context) -> Context:
        raise StageError(self.name, cause=ValueError("no candidates"))


class TerminalStage(Stage):
    """Populates `ctx.results` so we can assert on the pipeline output."""

    def __init__(self, name: str = "terminal") -> None:
        self.name = name
        self.optional = False

    async def run(self, ctx: Context) -> Context:
        ctx.results = [
            Candidate(chunk_id="c1", score=0.9, payload={"content": "one"}),
            Candidate(chunk_id="c2", score=0.7, payload={"content": "two"}),
        ]
        return ctx


class TestCandidate:
    def test_content_property_reads_from_payload(self) -> None:
        c = Candidate(chunk_id="x", score=1.0, payload={"content": "hello"})
        assert c.content == "hello"

    def test_missing_content_falls_back_to_empty(self) -> None:
        c = Candidate(chunk_id="x", score=1.0, payload={})
        assert c.content == ""

    def test_parent_chunk_id_is_none_when_absent(self) -> None:
        c = Candidate(chunk_id="x", score=1.0, payload={})
        assert c.parent_chunk_id is None

    def test_parent_chunk_id_surfaces_when_present(self) -> None:
        c = Candidate(chunk_id="x", score=1.0, payload={"parent_chunk_id": "p1"})
        assert c.parent_chunk_id == "p1"

    def test_is_frozen(self) -> None:
        c = Candidate(chunk_id="x", score=1.0, payload={})
        with pytest.raises((AttributeError, TypeError)):
            c.chunk_id = "y"  # type: ignore[misc]


class TestContext:
    def test_defaults_are_empty(self) -> None:
        ctx = Context(query="hi")
        assert ctx.top_k == 10
        assert ctx.min_score == 0.0
        assert ctx.state == {}
        assert ctx.results == []
        assert ctx.errors == []

    def test_state_is_per_instance(self) -> None:
        # Mutable default-factory regression guard: two contexts must not
        # share the same state dict.
        a = Context(query="a")
        b = Context(query="b")
        a.state["k"] = "v"
        assert "k" not in b.state


class TestPipeline:
    async def test_empty_pipeline_returns_context_unchanged(self) -> None:
        ctx = Context(query="q")
        result = await Pipeline([]).run(ctx)
        assert result is ctx
        assert result.errors == []

    async def test_runs_stages_in_order(self) -> None:
        log: list[str] = []
        pipe = Pipeline(
            [
                RecordingStage("a", log),
                RecordingStage("b", log),
                RecordingStage("c", log),
            ]
        )
        await pipe.run(Context(query="q"))
        assert log == ["a", "b", "c"]

    async def test_state_propagates_between_stages(self) -> None:
        log: list[str] = []
        ctx = await Pipeline([RecordingStage("a", log), RecordingStage("b", log)]).run(
            Context(query="q")
        )
        assert ctx.state == {"a": "ran", "b": "ran"}

    async def test_terminal_stage_populates_results(self) -> None:
        ctx = await Pipeline([TerminalStage()]).run(Context(query="q"))
        assert [c.chunk_id for c in ctx.results] == ["c1", "c2"]

    async def test_required_stage_failure_aborts_and_wraps(self) -> None:
        log: list[str] = []
        pipe = Pipeline(
            [
                RecordingStage("a", log),
                RaisingStage("bad", optional=False),
                RecordingStage("c", log),  # never reached
            ]
        )
        with pytest.raises(StageError) as exc_info:
            await pipe.run(Context(query="q"))
        assert exc_info.value.stage_name == "bad"
        assert isinstance(exc_info.value.cause, RuntimeError)
        assert log == ["a"]  # "c" never ran

    async def test_required_stage_error_propagates_without_rewrapping(self) -> None:
        # A stage that already raised StageError shouldn't get double-wrapped.
        pipe = Pipeline([StageErrorRaising("dense", optional=False)])
        with pytest.raises(StageError) as exc_info:
            await pipe.run(Context(query="q"))
        assert exc_info.value.stage_name == "dense"
        assert isinstance(exc_info.value.cause, ValueError)

    async def test_optional_stage_failure_is_captured_and_pipeline_continues(self) -> None:
        log: list[str] = []
        pipe = Pipeline(
            [
                RecordingStage("a", log),
                RaisingStage("rerank", optional=True),
                RecordingStage("c", log),
            ]
        )
        ctx = await pipe.run(Context(query="q"))
        assert log == ["a", "c"]
        assert len(ctx.errors) == 1
        assert ctx.errors[0].stage_name == "rerank"

    async def test_stages_accessor_exposes_tuple_of_registered_stages(self) -> None:
        stages = [RecordingStage("a", []), RecordingStage("b", [])]
        pipe = Pipeline(stages)
        assert len(pipe.stages) == 2
        assert pipe.stages[0].name == "a"

    async def test_multiple_optional_failures_all_captured(self) -> None:
        log: list[str] = []
        pipe = Pipeline(
            [
                RaisingStage("query_rewriter", optional=True),
                RecordingStage("dense", log),
                RaisingStage("rerank", optional=True),
            ]
        )
        ctx = await pipe.run(Context(query="q"))
        assert log == ["dense"]
        assert [e.stage_name for e in ctx.errors] == ["query_rewriter", "rerank"]


class TestStageError:
    def test_formats_with_cause(self) -> None:
        err = StageError("dense", cause=ValueError("bad query"))
        assert "dense" in str(err)
        assert "bad query" in str(err)

    def test_formats_without_cause(self) -> None:
        err = StageError("sparse")
        assert "sparse" in str(err)
