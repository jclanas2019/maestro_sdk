"""tests/test_p1_validate.py — validation module tests"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
import pytest
import maestro
pytestmark = pytest.mark.core

from maestro.validate import (
    ValidationError, ValidationResult, SchemaViolation,
    Required, Range, Length, Pattern, OneOf, Custom, NotEmpty,
    FieldSchema, field, Schema,
    ValidatedFacts, SchemaFilter, SchemaProcessor, ValidatedWork,
)


# ── Validators ────────────────────────────────────────────────────────────── #

class TestValidators:
    def _r(self, name, value, *validators):
        fs = FieldSchema(None, *validators)
        return fs.validate_value(name, value)

    def test_required_passes_on_value(self):
        r = Required().validate("x", 42)
        assert r.ok

    def test_required_fails_on_none(self):
        r = Required().validate("x", None)
        assert not r.ok

    def test_range_min(self):
        assert not Range(min=0).validate("v", -1).ok
        assert Range(min=0).validate("v", 0).ok
        assert Range(min=0).validate("v", 5).ok

    def test_range_max(self):
        assert not Range(max=10).validate("v", 11).ok
        assert Range(max=10).validate("v", 10).ok

    def test_range_both(self):
        b = Range(min=1, max=5)
        assert b.validate("v", 0).ok is False
        assert b.validate("v", 3).ok is True
        assert b.validate("v", 6).ok is False

    def test_range_non_numeric(self):
        r = Range(min=0).validate("v", "hello")
        assert not r.ok

    def test_length_min(self):
        assert not Length(min=3).validate("s", "ab").ok
        assert Length(min=3).validate("s", "abc").ok

    def test_length_max(self):
        assert not Length(max=5).validate("s", "toolong").ok
        assert Length(max=5).validate("s", "ok").ok

    def test_pattern_match(self):
        assert Pattern(r"^\d+$").validate("n", "123").ok
        assert not Pattern(r"^\d+$").validate("n", "abc").ok

    def test_pattern_non_string(self):
        r = Pattern(r"^\d+$").validate("n", 123)
        assert not r.ok

    def test_one_of_pass(self):
        assert OneOf("a", "b", "c").validate("s", "a").ok
        assert OneOf("a", "b", "c").validate("s", "b").ok

    def test_one_of_fail(self):
        assert not OneOf("a", "b").validate("s", "c").ok

    def test_custom_bool(self):
        assert Custom(lambda v: v > 0).validate("v", 5).ok
        assert not Custom(lambda v: v > 0, "must be positive").validate("v", -1).ok

    def test_custom_string_message(self):
        def check(v): return "too short" if len(v) < 3 else True
        r = Custom(check).validate("s", "ab")
        assert not r.ok
        assert "too short" in r.errors[0].message

    def test_not_empty_string(self):
        assert not NotEmpty().validate("s", "").ok
        assert not NotEmpty().validate("s", "   ").ok
        assert NotEmpty().validate("s", "hello").ok

    def test_not_empty_list(self):
        assert not NotEmpty().validate("l", []).ok
        assert NotEmpty().validate("l", [1]).ok


# ── FieldSchema ───────────────────────────────────────────────────────────── #

class TestFieldSchema:
    def test_type_check_pass(self):
        fs = FieldSchema(int)
        assert fs.validate_value("n", 42).ok

    def test_type_check_fail(self):
        fs = FieldSchema(int)
        r = fs.validate_value("n", "hello")
        assert not r.ok

    def test_optional_none_is_ok(self):
        fs = FieldSchema(int, required=False)
        assert fs.validate_value("n", None).ok

    def test_required_none_fails(self):
        fs = FieldSchema(int, required=True)
        assert not fs.validate_value("n", None).ok

    def test_coerce_converts_type(self):
        fs = FieldSchema(int, coerce=True)
        r = fs.validate_value("n", "42")
        assert r.ok

    def test_coerce_fails_on_bad_value(self):
        fs = FieldSchema(int, coerce=True)
        r = fs.validate_value("n", "hello")
        assert not r.ok

    def test_multiple_validators_all_run(self):
        fs = FieldSchema(int, Range(min=0, max=10), required=True)
        assert fs.validate_value("n", 5).ok
        assert not fs.validate_value("n", -1).ok
        assert not fs.validate_value("n", 11).ok

    def test_union_types(self):
        fs = FieldSchema((int, float))
        assert fs.validate_value("n", 1).ok
        assert fs.validate_value("n", 1.5).ok
        assert not fs.validate_value("n", "one").ok


# ── Schema ────────────────────────────────────────────────────────────────── #

class TestSchema:
    def _order_schema(self):
        return Schema(
            id     = field(int,   Required()),
            status = field(str,   OneOf("pending", "paid", "shipped")),
            total  = field(float, Range(min=0), coerce=True),
            note   = field(str,   required=False, default=""),
        )

    def test_valid_dict(self):
        result = self._order_schema().validate(
            {"id": 1, "status": "paid", "total": 99.9}
        )
        assert result.ok

    def test_missing_required_field(self):
        result = self._order_schema().validate({"status": "paid", "total": 50})
        assert not result.ok
        assert any("id" in e.field for e in result.errors)

    def test_wrong_type(self):
        result = self._order_schema().validate(
            {"id": "not-an-int", "status": "paid", "total": 50}
        )
        assert not result.ok

    def test_one_of_violation(self):
        result = self._order_schema().validate(
            {"id": 1, "status": "unknown", "total": 50}
        )
        assert not result.ok

    def test_default_applied_for_optional_field(self):
        schema = Schema(name=field(str, required=False, default="anonymous"))
        result = schema.validate({})
        assert result.ok

    def test_coerce_converts_string_to_float(self):
        schema = Schema(total=field(float, Range(min=0), coerce=True))
        result = schema.coerce({"total": "99.5"})
        assert result == {"total": 99.5}

    def test_coerce_raises_on_failure(self):
        schema = Schema(total=field(float, Range(min=0), coerce=True))
        with pytest.raises(SchemaViolation):
            schema.coerce({"total": "not-a-number"})

    def test_assert_valid_raises(self):
        schema = Schema(n=field(int, Required()))
        with pytest.raises(SchemaViolation):
            schema.assert_valid({"n": None})

    def test_validate_dataclass(self):
        @dataclass
        class Order:
            id: int
            status: str
            total: float

        schema = Schema(
            id     = field(int),
            status = field(str, OneOf("paid", "pending")),
            total  = field(float, Range(min=0)),
        )
        result = schema.validate(Order(id=1, status="paid", total=50.0))
        assert result.ok

    def test_validate_facts(self):
        schema = Schema(rain=field(bool), temp=field((int, float)))
        facts  = maestro.Facts(rain=True, temp=22.5)
        assert schema.validate(facts).ok

    def test_multiple_errors_collected(self):
        schema = Schema(
            a = field(int,   Required()),
            b = field(str,   Required()),
        )
        result = schema.validate({"a": None, "b": None})
        assert len(result.errors) == 2


# ── ValidationResult ──────────────────────────────────────────────────────── #

class TestValidationResult:
    def test_ok_when_no_errors(self):
        assert ValidationResult().ok

    def test_not_ok_with_error(self):
        r = ValidationResult()
        r.add("x", "bad value")
        assert not r.ok

    def test_bool_conversion(self):
        assert bool(ValidationResult()) is True
        r = ValidationResult()
        r.add("x", "fail")
        assert bool(r) is False

    def test_merge(self):
        r1 = ValidationResult()
        r1.add("a", "err a")
        r2 = ValidationResult()
        r2.add("b", "err b")
        r1.merge(r2)
        assert len(r1.errors) == 2


# ── ValidatedFacts ────────────────────────────────────────────────────────── #

class TestValidatedFacts:
    def test_valid_initial_kwargs(self):
        schema = Schema(age=field(int, Range(min=0)))
        vf = ValidatedFacts(schema, age=25)
        assert vf.get("age") == 25

    def test_put_invalid_warns_not_raises_by_default(self):
        schema = Schema(age=field(int, Range(min=0)))
        vf = ValidatedFacts(schema, age=25)
        vf.put("age", -1)  # should warn but not raise
        assert vf.get("age") == -1  # value is stored

    def test_put_invalid_raises_in_strict_mode(self):
        schema = Schema(age=field(int, Range(min=0)))
        vf = ValidatedFacts(schema, strict=True, age=25)
        with pytest.raises(SchemaViolation):
            vf.put("age", -1)

    def test_validate_method(self):
        schema = Schema(age=field(int, Range(min=0)))
        vf = ValidatedFacts(schema, age=25)
        assert vf.validate().ok


# ── SchemaFilter ──────────────────────────────────────────────────────────── #

class TestSchemaFilter:
    def _record(self, payload, n=1):
        from maestro.batch._record import Header, Record
        return Record(Header(n, "test"), payload)

    def test_accepts_valid_record(self):
        schema = Schema(age=field(int, Range(min=0)))
        filt   = SchemaFilter(schema)
        assert filt.accept(self._record({"age": 25})) is True

    def test_rejects_invalid_record(self):
        schema = Schema(age=field(int, Range(min=0)))
        filt   = SchemaFilter(schema)
        assert filt.accept(self._record({"age": -1})) is False

    def test_in_batch_pipeline(self):
        schema = Schema(
            age  = field(int,   Range(min=18)),
            name = field(str,   Required()),
        )
        data = [
            {"name": "Alice", "age": 25},
            {"name": "Bob",   "age": 15},  # filtered: underage
            {"name": "Carol", "age": 30},
        ]
        sink = []
        (maestro.JobBuilder()
         .named("filter-test")
         .reader(maestro.IterableRecordReader(data))
         .filter(SchemaFilter(schema))
         .writer(maestro.CollectionRecordWriter(sink))
         .build()).call()
        assert len(sink) == 2
        assert all(r["age"] >= 18 for r in sink)


# ── SchemaProcessor ───────────────────────────────────────────────────────── #

class TestSchemaProcessor:
    def _record(self, payload, n=1):
        from maestro.batch._record import Header, Record
        return Record(Header(n, "test"), payload)

    def test_passes_valid_record(self):
        schema = Schema(total=field(float, Range(min=0)))
        proc   = SchemaProcessor(schema)
        r = self._record({"total": 99.0})
        result = proc.process_record(r)
        assert result.payload["total"] == 99.0

    def test_rejects_invalid_record(self):
        schema = Schema(total=field(float, Range(min=0)))
        proc   = SchemaProcessor(schema)
        with pytest.raises(maestro.RecordProcessingException):
            proc.process_record(self._record({"total": -1.0}))

    def test_coerce_converts_types(self):
        schema = Schema(
            id    = field(int,   coerce=True),
            total = field(float, Range(min=0), coerce=True),
        )
        proc = SchemaProcessor(schema, coerce=True)
        r = self._record({"id": "1", "total": "99.5"})
        result = proc.process_record(r)
        assert isinstance(result.payload["id"], int)
        assert isinstance(result.payload["total"], float)

    def test_in_batch_pipeline_with_coercion(self):
        schema = Schema(
            id    = field(int,   Required(), coerce=True),
            total = field(float, Range(min=0), coerce=True),
        )
        raw = [{"id": "1", "total": "100"}, {"id": "2", "total": "bad"}]
        sink = []
        report = (maestro.JobBuilder()
                  .named("coerce-test")
                  .reader(maestro.IterableRecordReader(raw))
                  .processor(SchemaProcessor(schema, coerce=True))
                  .writer(maestro.CollectionRecordWriter(sink))
                  .build()).call()
        assert len(sink) == 1
        assert sink[0]["id"] == 1
        assert report.metrics.skipped_count == 1


# ── ValidatedWork ─────────────────────────────────────────────────────────── #

class TestValidatedWork:
    def test_passes_when_pre_schema_satisfied(self):
        schema = Schema(order_id=field(str, Required()))
        inner  = maestro.NoOpWork()
        work   = ValidatedWork(inner, pre_schema=schema)
        ctx    = maestro.WorkContext(order_id="ord-123")
        report = work.execute(ctx)
        assert report.status == maestro.WorkStatus.COMPLETED

    def test_fails_when_pre_schema_violated(self):
        schema = Schema(order_id=field(str, Required()))
        inner  = maestro.NoOpWork()
        work   = ValidatedWork(inner, pre_schema=schema)
        ctx    = maestro.WorkContext()  # missing order_id
        report = work.execute(ctx)
        assert report.status == maestro.WorkStatus.FAILED

    def test_post_schema_checked_after_execution(self):
        post_schema = Schema(receipt_id=field(str, Required()))

        def produce_receipt(ctx):
            ctx.put("receipt_id", "rec-001")
        inner = maestro.LambdaWork(produce_receipt, "producer")
        work  = ValidatedWork(inner, post_schema=post_schema)

        report = work.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.COMPLETED

    def test_fails_when_post_schema_violated(self):
        post_schema = Schema(receipt_id=field(str, Required()))
        inner       = maestro.NoOpWork()  # doesn't set receipt_id
        work        = ValidatedWork(inner, post_schema=post_schema)
        report      = work.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.FAILED

    def test_strict_raises_on_violation(self):
        schema = Schema(order_id=field(str, Required()))
        work   = ValidatedWork(maestro.NoOpWork(), pre_schema=schema, strict=True)
        with pytest.raises(SchemaViolation):
            work.execute(maestro.WorkContext())

    def test_inside_sequential_flow(self):
        pre  = Schema(x=field(int, Range(min=0)))
        post = Schema(x=field(int), doubled=field(int, Required()))

        def double(ctx): ctx.put("doubled", ctx.get("x", 0) * 2)
        inner = maestro.LambdaWork(double, "double")
        work  = ValidatedWork(inner, pre_schema=pre, post_schema=post)

        flow   = maestro.SequentialFlow.Builder().execute(work).build()
        report = maestro.WorkFlowEngine().run(flow, maestro.WorkContext(x=5))
        assert report.status == maestro.WorkStatus.COMPLETED
