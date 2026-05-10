"""
maestro.validate — schema validation for Facts, Records and WorkContext.

Catches type mismatches, missing fields and invalid values at pipeline
boundaries — before they become silent bugs downstream.

    from maestro.validate import Schema, field, Required, Range, OneOf

    order_schema = Schema(
        id       = field(int,   Required()),
        status   = field(str,   OneOf("pending","paid","shipped")),
        total    = field(float, Range(min=0)),
        email    = field(str,   Pattern(r"^[^@]+@[^@]+$"), required=False),
    )

    # Validate a dict
    result = order_schema.validate({"id": 1, "status": "paid", "total": 99.9})
    if not result.ok:
        print(result.errors)

    # Use with Maestro modules
    validated_facts = ValidatedFacts(order_schema, id=1, status="pending", total=50.0)

    # Use with batch pipelines
    job = (JobBuilder()
           .filter(SchemaFilter(order_schema))          # drop invalid records
           .processor(SchemaProcessor(order_schema))    # coerce types + validate
           .build())
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Optional, Type, Union


# ════════════════════════════════════════════════════════════════════════════
#  ValidationError + ValidationResult
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ValidationError:
    field:   str
    message: str

    def __str__(self) -> str:
        return f"[{self.field}] {self.message}"


@dataclass
class ValidationResult:
    errors: list[ValidationError] = dc_field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def add(self, field: str, message: str) -> "ValidationResult":
        self.errors.append(ValidationError(field, message)); return self

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        self.errors.extend(other.errors); return self

    def __bool__(self) -> bool: return self.ok

    def __repr__(self) -> str:
        if self.ok: return "ValidationResult(ok)"
        errs = "; ".join(str(e) for e in self.errors)
        return f"ValidationResult(errors=[{errs}])"


class SchemaViolation(Exception):
    """Raised when strict validation fails."""
    def __init__(self, result: ValidationResult):
        super().__init__(str(result))
        self.result = result


# ════════════════════════════════════════════════════════════════════════════
#  Validator building blocks
# ════════════════════════════════════════════════════════════════════════════

class Validator(ABC):
    """Abstract validator applied to a single field value."""

    @abstractmethod
    def validate(self, field_name: str, value: Any) -> ValidationResult:
        """Validate *value* for *field_name*. Return a ValidationResult."""


class Required(Validator):
    """Field must be present and not None."""
    def validate(self, field_name: str, value: Any) -> ValidationResult:
        r = ValidationResult()
        if value is None:
            r.add(field_name, "field is required but got None")
        return r


class Range(Validator):
    """Numeric value must be within [min, max] (inclusive, both optional)."""
    def __init__(self, min: Optional[float] = None, max: Optional[float] = None):
        self._min = min; self._max = max

    def validate(self, field_name: str, value: Any) -> ValidationResult:
        r = ValidationResult()
        try: v = float(value)
        except (TypeError, ValueError):
            return r.add(field_name, f"expected a number, got {type(value).__name__}")
        if self._min is not None and v < self._min:
            r.add(field_name, f"value {v} is below minimum {self._min}")
        if self._max is not None and v > self._max:
            r.add(field_name, f"value {v} exceeds maximum {self._max}")
        return r


class Length(Validator):
    """String/sequence length must be within [min, max] (both optional)."""
    def __init__(self, min: Optional[int] = None, max: Optional[int] = None):
        self._min = min; self._max = max

    def validate(self, field_name: str, value: Any) -> ValidationResult:
        r = ValidationResult()
        try: n = len(value)
        except TypeError:
            return r.add(field_name, f"value has no length: {type(value).__name__}")
        if self._min is not None and n < self._min:
            r.add(field_name, f"length {n} is below minimum {self._min}")
        if self._max is not None and n > self._max:
            r.add(field_name, f"length {n} exceeds maximum {self._max}")
        return r


class Pattern(Validator):
    """String must match a regular expression."""
    def __init__(self, regex: str, flags: int = 0):
        self._pattern = re.compile(regex, flags)
        self._regex   = regex

    def validate(self, field_name: str, value: Any) -> ValidationResult:
        r = ValidationResult()
        if not isinstance(value, str):
            return r.add(field_name, f"expected str for regex match, got {type(value).__name__}")
        if not self._pattern.search(value):
            r.add(field_name, f"value {value!r} does not match pattern {self._regex!r}")
        return r


class OneOf(Validator):
    """Value must be one of the allowed choices."""
    def __init__(self, *choices: Any):
        self._choices = set(choices)
        self._display = sorted(str(c) for c in choices)

    def validate(self, field_name: str, value: Any) -> ValidationResult:
        r = ValidationResult()
        if value not in self._choices:
            r.add(field_name, f"value {value!r} must be one of {self._display}")
        return r


class Custom(Validator):
    """User-supplied validation function ``(value) → bool | str``."""
    def __init__(self, fn: Callable[[Any], Union[bool, str]],
                 message: str = "custom validation failed"):
        self._fn = fn; self._msg = message

    def validate(self, field_name: str, value: Any) -> ValidationResult:
        r = ValidationResult()
        result = self._fn(value)
        if result is False:
            r.add(field_name, self._msg)
        elif isinstance(result, str):
            r.add(field_name, result)
        return r


class NotEmpty(Validator):
    """String/sequence must be non-empty after stripping whitespace."""
    def validate(self, field_name: str, value: Any) -> ValidationResult:
        r = ValidationResult()
        if isinstance(value, str) and not value.strip():
            r.add(field_name, "value must not be blank")
        elif hasattr(value, "__len__") and len(value) == 0:
            r.add(field_name, "value must not be empty")
        return r


# ════════════════════════════════════════════════════════════════════════════
#  FieldSchema
# ════════════════════════════════════════════════════════════════════════════

class FieldSchema:
    """
    Schema definition for one field.

    Args:
        type_:      Expected Python type(s). ``None`` disables type checking.
        required:   If True, field must be present and not None.
        default:    Value used when field is absent (only when not required).
        coerce:     If True, attempt to cast to *type_* before validation.
        validators: Additional :class:`Validator` objects applied after type check.
    """

    def __init__(
        self,
        type_:      Optional[Union[Type, tuple[Type, ...]]] = None,
        *validators: Validator,
        required: bool  = True,
        default:  Any   = None,
        coerce:   bool  = False,
    ) -> None:
        self.type_      = type_
        self.required   = required
        self.default    = default
        self.coerce     = coerce
        self.validators = list(validators)

    def validate_value(self, field_name: str, value: Any) -> ValidationResult:
        result = ValidationResult()

        # Presence check
        if value is None:
            if self.required:
                result.add(field_name, "required field is missing or None")
                return result
            return result  # optional and absent — ok

        # Type coercion
        if self.coerce and self.type_ is not None:
            target = self.type_[0] if isinstance(self.type_, tuple) else self.type_
            try:
                value = target(value)
            except (TypeError, ValueError) as e:
                result.add(field_name, f"cannot coerce {value!r} to {target.__name__}: {e}")
                return result

        # Type check
        if self.type_ is not None and not isinstance(value, self.type_):
            types = (self.type_,) if not isinstance(self.type_, tuple) else self.type_
            names = "/".join(t.__name__ for t in types)
            result.add(field_name,
                       f"expected {names}, got {type(value).__name__} ({value!r})")

        # Additional validators
        for v in self.validators:
            result.merge(v.validate(field_name, value))

        return result


def field(
    type_:    Optional[Union[Type, tuple[Type, ...]]] = None,
    *validators: Validator,
    required: bool = True,
    default:  Any  = None,
    coerce:   bool = False,
) -> FieldSchema:
    """Shorthand for constructing a :class:`FieldSchema`."""
    return FieldSchema(type_, *validators, required=required,
                       default=default, coerce=coerce)


# ════════════════════════════════════════════════════════════════════════════
#  Schema
# ════════════════════════════════════════════════════════════════════════════

class Schema:
    """
    A collection of :class:`FieldSchema` definitions that can validate
    dictionaries, dataclass instances or :class:`~maestro.rules.Facts`.

    Example::

        order_schema = Schema(
            id     = field(int,   required=True),
            status = field(str,   OneOf("pending","paid"), required=True),
            total  = field(float, Range(min=0),            coerce=True),
            note   = field(str,   required=False, default=""),
        )

        result = order_schema.validate({"id": 1, "status": "paid", "total": "99.5"})
    """

    def __init__(self, **fields: FieldSchema) -> None:
        self._fields: dict[str, FieldSchema] = fields

    def validate(self, data: Any) -> ValidationResult:
        """Validate *data* (dict, dataclass, Facts, or object with ``__dict__``)."""
        d      = self._to_dict(data)
        result = ValidationResult()

        for fname, fschema in self._fields.items():
            value = d.get(fname)
            if value is None and not fschema.required and fschema.default is not None:
                value = fschema.default
            result.merge(fschema.validate_value(fname, value))

        return result

    def coerce(self, data: dict) -> dict:
        """
        Return a new dict with values coerced to their declared types.
        Raises ``SchemaViolation`` if coercion + validation fails.
        """
        result   = ValidationResult()
        coerced  = dict(data)

        for fname, fschema in self._fields.items():
            value = coerced.get(fname)
            if value is None:
                if not fschema.required and fschema.default is not None:
                    coerced[fname] = fschema.default
                continue
            if fschema.coerce and fschema.type_ is not None:
                target = fschema.type_[0] if isinstance(fschema.type_, tuple) else fschema.type_
                try:
                    coerced[fname] = target(value)
                except (TypeError, ValueError) as e:
                    result.add(fname, f"cannot coerce {value!r} to {target.__name__}: {e}")
                    continue
            r = fschema.validate_value(fname, coerced[fname])
            result.merge(r)

        if not result.ok:
            raise SchemaViolation(result)
        return coerced

    def assert_valid(self, data: Any) -> None:
        """Validate and raise :exc:`SchemaViolation` on failure."""
        result = self.validate(data)
        if not result.ok:
            raise SchemaViolation(result)

    def add_field(self, name: str, schema: FieldSchema) -> "Schema":
        """Add or replace a field definition. Returns self for chaining."""
        self._fields[name] = schema; return self

    def _to_dict(self, data: Any) -> dict:
        if isinstance(data, dict): return data
        # Facts
        if hasattr(data, "as_map"): return data.as_map()
        # dataclass / arbitrary object
        if hasattr(data, "__dict__"):
            return {k: v for k, v in vars(data).items() if not k.startswith("_")}
        raise TypeError(f"Cannot convert {type(data).__name__!r} to dict for validation")

    @property
    def fields(self) -> dict[str, FieldSchema]:
        return dict(self._fields)


# ════════════════════════════════════════════════════════════════════════════
#  ValidatedFacts — Facts with schema enforcement
# ════════════════════════════════════════════════════════════════════════════

from maestro.rules import Facts  # noqa: E402


class ValidatedFacts(Facts):
    """
    A :class:`~maestro.rules.Facts` subclass that validates every ``put``
    call against a :class:`Schema`.

    Args:
        schema: The schema to enforce.
        strict: If True, raises :exc:`SchemaViolation` on any violation.
                If False (default), violations are logged as warnings.
        **kwargs: Initial facts (validated on construction).

    Example::

        schema = Schema(age=field(int, Range(min=0, max=150)))
        facts  = ValidatedFacts(schema, age=25)
        facts.put("age", -1)   # raises SchemaViolation or logs warning
    """

    def __init__(self, schema: Schema, strict: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._schema = schema
        self._strict = strict
        # Validate initial values
        self._validate_all()

    def put(self, name: str, value: Any) -> "ValidatedFacts":
        if name in self._schema.fields:
            field_schema = self._schema.fields[name]
            result = field_schema.validate_value(name, value)
            if not result.ok:
                msg = f"ValidatedFacts violation on put('{name}'): {result}"
                if self._strict: raise SchemaViolation(result)
                import logging; logging.getLogger(__name__).warning(msg)
        super().put(name, value)
        return self

    def validate(self) -> ValidationResult:
        """Validate all current facts against the schema."""
        return self._schema.validate(self._data)

    def _validate_all(self) -> None:
        result = self._schema.validate(self._data)
        if not result.ok:
            msg = f"ValidatedFacts initial validation failed: {result}"
            if self._strict: raise SchemaViolation(result)
            import logging; logging.getLogger(__name__).warning(msg)


# ════════════════════════════════════════════════════════════════════════════
#  Batch integrations
# ════════════════════════════════════════════════════════════════════════════

from maestro.batch._filter    import RecordFilter
from maestro.batch._processor import RecordProcessingException, RecordProcessor
from maestro.batch._record    import Record


class SchemaFilter(RecordFilter):
    """
    Drop batch records whose payload fails schema validation.

    Compatible with dict payloads, dataclasses and objects with ``__dict__``.

    Example::

        job = (JobBuilder()
               .filter(SchemaFilter(order_schema))
               .build())
    """

    def __init__(self, schema: Schema) -> None:
        self._schema = schema

    def accept(self, record: Record) -> bool:
        try:
            result = self._schema.validate(record.payload)
            return result.ok
        except Exception:
            return False


class SchemaProcessor(RecordProcessor):
    """
    Validate and optionally coerce a record's payload against a schema.

    Raises :exc:`~maestro.batch.RecordProcessingException` when validation
    fails so the record is counted as skipped (not written).

    Args:
        schema: The schema to apply.
        coerce: If True, type-coerce values before validation (mutates payload).

    Example::

        job = (JobBuilder()
               .processor(SchemaProcessor(order_schema, coerce=True))
               .build())
    """

    def __init__(self, schema: Schema, coerce: bool = False) -> None:
        self._schema = schema
        self._coerce = coerce

    def process_record(self, record: Record) -> Record:
        if self._coerce and isinstance(record.payload, dict):
            try:
                record.payload = self._schema.coerce(record.payload)
                return record
            except SchemaViolation as e:
                raise RecordProcessingException(
                    f"Record #{record.header.number} schema violation: {e}"
                ) from e

        result = self._schema.validate(record.payload)
        if not result.ok:
            raise RecordProcessingException(
                f"Record #{record.header.number} schema violation: {result}"
            )
        return record


# ════════════════════════════════════════════════════════════════════════════
#  Flow integration
# ════════════════════════════════════════════════════════════════════════════

from maestro.flows._work import DefaultWorkReport, Work, WorkContext, WorkReport, WorkStatus


class ValidatedWork(Work):
    """
    Wrap a :class:`~maestro.flows.Work` with pre- and post-execution
    ``WorkContext`` validation.

    Args:
        work:          The work to wrap.
        pre_schema:    Schema validated against WorkContext *before* execution.
        post_schema:   Schema validated against WorkContext *after* execution.
        strict:        If True, raise on violation; if False, return FAILED report.

    Example::

        step = ValidatedWork(
            work        = process_order_work,
            pre_schema  = Schema(order_id=field(str, Required()), total=field(float)),
            post_schema = Schema(receipt_id=field(str, Required())),
        )
    """

    def __init__(self, work: Work,
                 pre_schema:  Optional[Schema] = None,
                 post_schema: Optional[Schema] = None,
                 strict:      bool = False) -> None:
        self._work       = work
        self._pre        = pre_schema
        self._post       = post_schema
        self._strict     = strict

    def get_name(self) -> str:
        return f"validated({self._work.get_name()})"

    def execute(self, work_context: WorkContext) -> WorkReport:
        # Pre-validation
        if self._pre:
            result = self._pre.validate(work_context.as_map())
            if not result.ok:
                err = SchemaViolation(result)
                if self._strict: raise err
                return DefaultWorkReport(WorkStatus.FAILED, work_context, error=err)

        report = self._work.execute(work_context)

        # Post-validation
        if self._post and report.status == WorkStatus.COMPLETED:
            result = self._post.validate(work_context.as_map())
            if not result.ok:
                err = SchemaViolation(result)
                if self._strict: raise err
                return DefaultWorkReport(WorkStatus.FAILED, work_context, error=err)

        return report


__all__ = [
    "ValidationError", "ValidationResult", "SchemaViolation",
    "Validator", "Required", "Range", "Length", "Pattern",
    "OneOf", "Custom", "NotEmpty",
    "FieldSchema", "field", "Schema",
    "ValidatedFacts",
    "SchemaFilter", "SchemaProcessor",
    "ValidatedWork",
]
