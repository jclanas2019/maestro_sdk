"""
tests/test_easy_batch.py — comprehensive unit tests for the easy_batch port.

Run:  python -m pytest tests/test_easy_batch.py -v
"""
import sys, os, io


import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
pytestmark = pytest.mark.core


from maestro.batch import (
    Header, Record, Batch,
    IterableRecordReader, StringRecordReader, FlatFileRecordReader, CsvDictRecordReader,
    HeaderRecordFilter, PredicateRecordFilter, RecordNumberRangeFilter,
    DelimitedRecordMapper, FieldSetMapper, LambdaRecordMapper, PassThroughRecordMapper,
    LambdaRecordProcessor, FilteringRecordProcessor, CompositeRecordProcessor,
    RecordProcessingException, ToStringMarshaller, JsonMarshaller, CsvMarshaller,
    CollectionRecordWriter, DevNullRecordWriter, FileRecordWriter, StringIORecordWriter,
    JobBuilder, JobExecutor, JobStatus, JobParameters,
    JobListener, BatchListener, PipelineListener,
)


# ─────────────────────────────────────────────────────────────────── #
#  Helpers                                                            #
# ─────────────────────────────────────────────────────────────────── #

def make_record(payload, number=1):
    return Record(Header(number=number, source="test"), payload)


@dataclass
class Tweet:
    id: int
    user: str
    message: str


# ─────────────────────────────────────────────────────────────────── #
#  Header / Record / Batch                                            #
# ─────────────────────────────────────────────────────────────────── #

class TestRecord:
    def test_header_fields(self):
        h = Header(number=5, source="file.csv")
        assert h.number == 5
        assert h.source == "file.csv"

    def test_record_payload(self):
        r = make_record("hello")
        assert r.payload == "hello"
        r.payload = "world"
        assert r.payload == "world"

    def test_batch_add_len(self):
        b = Batch()
        b.add(make_record(1))
        b.add(make_record(2))
        assert len(b) == 2
        assert b.payloads() == [1, 2]

    def test_batch_clear(self):
        b = Batch()
        b.add(make_record(1))
        b.clear()
        assert b.is_empty()

    def test_batch_iter(self):
        b = Batch()
        r1, r2 = make_record("a"), make_record("b", 2)
        b.add(r1); b.add(r2)
        assert list(b) == [r1, r2]


# ─────────────────────────────────────────────────────────────────── #
#  Readers                                                            #
# ─────────────────────────────────────────────────────────────────── #

class TestIterableRecordReader:
    def test_reads_all(self):
        records = list(IterableRecordReader([1, 2, 3]))
        assert [r.payload for r in records] == [1, 2, 3]

    def test_numbers_start_at_1(self):
        records = list(IterableRecordReader(["a", "b"]))
        assert records[0].header.number == 1
        assert records[1].header.number == 2

    def test_empty(self):
        assert list(IterableRecordReader([])) == []


class TestStringRecordReader:
    def test_reads_lines(self):
        records = list(StringRecordReader("foo\nbar\nbaz"))
        assert [r.payload for r in records] == ["foo", "bar", "baz"]

    def test_skips_empty_lines(self):
        records = list(StringRecordReader("a\n\nb"))
        assert len(records) == 2


class TestFlatFileRecordReader:
    def test_reads_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("line1\nline2\nline3\n")
        records = list(FlatFileRecordReader(f))
        assert [r.payload for r in records] == ["line1", "line2", "line3"]


class TestCsvDictRecordReader:
    def test_reads_dict_rows(self):
        stream = io.StringIO("id,user\n1,alice\n2,bob")
        records = list(CsvDictRecordReader(stream))
        assert records[0].payload == {"id": "1", "user": "alice"}
        assert records[1].payload == {"id": "2", "user": "bob"}


# ─────────────────────────────────────────────────────────────────── #
#  Filters                                                            #
# ─────────────────────────────────────────────────────────────────── #

class TestFilters:
    def test_header_filter_discards_first(self):
        r1 = make_record("header", 1)
        r2 = make_record("data", 2)
        f = HeaderRecordFilter()
        assert f.accept(r1) is False
        assert f.accept(r2) is True

    def test_predicate_filter(self):
        f = PredicateRecordFilter(lambda r: r.payload < 0)
        assert f.accept(make_record(5)) is True
        assert f.accept(make_record(-1)) is False

    def test_range_filter(self):
        f = RecordNumberRangeFilter(start=3, end=5)
        assert f.accept(make_record("x", 2)) is False
        assert f.accept(make_record("x", 3)) is True
        assert f.accept(make_record("x", 5)) is True
        assert f.accept(make_record("x", 6)) is False


# ─────────────────────────────────────────────────────────────────── #
#  Mappers                                                            #
# ─────────────────────────────────────────────────────────────────── #

class TestMappers:
    def test_pass_through(self):
        r = make_record("hello")
        result = PassThroughRecordMapper().map_record(r)
        assert result.payload == "hello"

    def test_lambda_mapper(self):
        r = make_record("hello")
        result = LambdaRecordMapper(str.upper).map_record(r)
        assert result.payload == "HELLO"

    def test_delimited_to_dict(self):
        r = make_record("1,alice,hi")
        mapper = DelimitedRecordMapper(field_names=["id", "user", "msg"])
        result = mapper.map_record(r)
        assert result.payload == {"id": "1", "user": "alice", "msg": "hi"}

    def test_delimited_to_dataclass(self):
        r = make_record("1,alice,hello")
        mapper = DelimitedRecordMapper(
            Tweet,
            field_names=["id", "user", "message"],
            type_converters={"id": int},
        )
        result = mapper.map_record(r)
        assert isinstance(result.payload, Tweet)
        assert result.payload.id == 1
        assert result.payload.user == "alice"

    def test_delimited_wrong_field_count(self):
        r = make_record("a,b")
        mapper = DelimitedRecordMapper(field_names=["x", "y", "z"])
        with pytest.raises(ValueError):
            mapper.map_record(r)

    def test_field_set_mapper(self):
        r = make_record({"id": "5", "user": "dave", "message": "yo"})
        mapper = FieldSetMapper(Tweet, type_converters={"id": int})
        result = mapper.map_record(r)
        assert result.payload == Tweet(id=5, user="dave", message="yo")


# ─────────────────────────────────────────────────────────────────── #
#  Processors                                                         #
# ─────────────────────────────────────────────────────────────────── #

class TestProcessors:
    def test_lambda_processor(self):
        r = make_record(3)
        result = LambdaRecordProcessor(lambda x: x * 2).process_record(r)
        assert result.payload == 6

    def test_filtering_processor_passes(self):
        r = make_record(10)
        result = FilteringRecordProcessor(lambda p: p < 0).process_record(r)
        assert result.payload == 10

    def test_filtering_processor_rejects(self):
        r = make_record(-1)
        with pytest.raises(RecordProcessingException):
            FilteringRecordProcessor(lambda p: p < 0).process_record(r)

    def test_composite_processor(self):
        r = make_record(1)
        p = CompositeRecordProcessor([
            LambdaRecordProcessor(lambda x: x + 1),
            LambdaRecordProcessor(lambda x: x * 10),
        ])
        result = p.process_record(r)
        assert result.payload == 20


# ─────────────────────────────────────────────────────────────────── #
#  Marshallers                                                        #
# ─────────────────────────────────────────────────────────────────── #

class TestMarshallers:
    def test_to_string(self):
        r = make_record(42)
        result = ToStringMarshaller().marshal_record(r)
        assert result.payload == "42"

    def test_json_dict(self):
        import json
        r = make_record({"a": 1, "b": 2})
        result = JsonMarshaller().marshal_record(r)
        assert json.loads(result.payload) == {"a": 1, "b": 2}

    def test_json_dataclass(self):
        import json
        r = make_record(Tweet(id=1, user="alice", message="hi"))
        result = JsonMarshaller().marshal_record(r)
        data = json.loads(result.payload)
        assert data == {"id": 1, "user": "alice", "message": "hi"}

    def test_csv_marshaller_dict(self):
        r = make_record({"id": 1, "user": "bob", "message": "hey"})
        result = CsvMarshaller(field_names=["id", "user", "message"]).marshal_record(r)
        assert result.payload == "1,bob,hey"

    def test_csv_marshaller_dataclass(self):
        r = make_record(Tweet(id=2, user="carol", message="wow"))
        result = CsvMarshaller(field_names=["id", "user", "message"]).marshal_record(r)
        assert result.payload == "2,carol,wow"


# ─────────────────────────────────────────────────────────────────── #
#  Writers                                                            #
# ─────────────────────────────────────────────────────────────────── #

class TestWriters:
    def test_collection_writer(self):
        sink = []
        w = CollectionRecordWriter(sink)
        b = Batch()
        b.add(make_record("x"))
        b.add(make_record("y"))
        w.write_records(b)
        assert sink == ["x", "y"]

    def test_dev_null_writer(self):
        b = Batch()
        b.add(make_record(1))
        DevNullRecordWriter().write_records(b)  # no error

    def test_string_io_writer(self):
        w = StringIORecordWriter()
        b = Batch()
        b.add(make_record("hello"))
        w.write_records(b)
        assert "hello" in w.getvalue()

    def test_file_writer(self, tmp_path):
        f = tmp_path / "out.txt"
        w = FileRecordWriter(f)
        w.open()
        b = Batch()
        b.add(make_record("line1"))
        b.add(make_record("line2"))
        w.write_records(b)
        w.close()
        assert f.read_text() == "line1\nline2\n"


# ─────────────────────────────────────────────────────────────────── #
#  Job / JobBuilder / JobExecutor                                     #
# ─────────────────────────────────────────────────────────────────── #

class TestJob:
    def _simple_job(self, data, sink, **kwargs):
        return (
            JobBuilder()
            .named("test")
            .reader(IterableRecordReader(data))
            .writer(CollectionRecordWriter(sink))
            .build()
        )

    def test_basic_pipeline(self):
        sink = []
        job = self._simple_job([1, 2, 3], sink)
        report = JobExecutor().execute(job)
        assert report.status == JobStatus.COMPLETED
        assert sink == [1, 2, 3]
        assert report.metrics.written_count == 3
        assert report.metrics.total_count == 3

    def test_filter_counts(self):
        sink = []
        job = (
            JobBuilder()
            .named("filter-test")
            .reader(IterableRecordReader([1, 2, 3, 4]))
            .filter(PredicateRecordFilter(lambda r: r.payload % 2 == 0))
            .writer(CollectionRecordWriter(sink))
            .build()
        )
        report = JobExecutor().execute(job)
        assert sink == [1, 3]
        assert report.metrics.filtered_count == 2

    def test_mapper_transforms(self):
        sink = []
        job = (
            JobBuilder()
            .named("map-test")
            .reader(IterableRecordReader(["hello", "world"]))
            .mapper(LambdaRecordMapper(str.upper))
            .writer(CollectionRecordWriter(sink))
            .build()
        )
        JobExecutor().execute(job)
        assert sink == ["HELLO", "WORLD"]

    def test_processor_skip(self):
        sink = []
        job = (
            JobBuilder()
            .named("skip-test")
            .reader(IterableRecordReader([1, 2, 3, 4]))
            .processor(FilteringRecordProcessor(lambda p: p % 2 == 0))
            .writer(CollectionRecordWriter(sink))
            .build()
        )
        report = JobExecutor().execute(job)
        assert sink == [1, 3]
        assert report.metrics.skipped_count == 2

    def test_batch_size_respected(self):
        batches = []

        class RecordingBatchListener(BatchListener):
            def before_batch_writing(self, batch):
                batches.append(len(batch))

        sink = []
        job = (
            JobBuilder()
            .named("batch-test")
            .reader(IterableRecordReader(range(10)))
            .batch_listener(RecordingBatchListener())
            .writer(CollectionRecordWriter(sink))
            .batch_size(3)
            .build()
        )
        JobExecutor().execute(job)
        assert len(sink) == 10
        # First three full batches of 3, then a remainder of 1
        assert batches[:3] == [3, 3, 3]
        assert batches[3] == 1

    def test_error_threshold_aborts(self):
        def explode(p):
            raise RuntimeError("boom")

        sink = []
        job = (
            JobBuilder()
            .named("abort-test")
            .reader(IterableRecordReader(range(20)))
            .processor(LambdaRecordProcessor(explode))
            .error_threshold(2)
            .writer(CollectionRecordWriter(sink))
            .build()
        )
        report = JobExecutor().execute(job)
        assert report.status == JobStatus.ABORTED
        assert report.metrics.failed_count > 2

    def test_csv_end_to_end(self):
        csv_data = "id,user,message\n1,alice,hello\n2,bob,world"
        sink = []
        job = (
            JobBuilder()
            .named("e2e-csv")
            .reader(StringRecordReader(csv_data))
            .filter(HeaderRecordFilter())
            .mapper(DelimitedRecordMapper(Tweet, field_names=["id","user","message"],
                                          type_converters={"id": int}))
            .marshaller(JsonMarshaller())
            .writer(CollectionRecordWriter(sink))
            .build()
        )
        report = JobExecutor().execute(job)
        import json
        assert json.loads(sink[0])["user"] == "alice"
        assert report.metrics.written_count == 2

    def test_parallel_execution(self):
        sink_a, sink_b = [], []
        job_a = (
            JobBuilder().named("A").reader(IterableRecordReader([1,2])).writer(CollectionRecordWriter(sink_a)).build()
        )
        job_b = (
            JobBuilder().named("B").reader(IterableRecordReader([3,4])).writer(CollectionRecordWriter(sink_b)).build()
        )
        executor = JobExecutor()
        reports = executor.execute_all([job_a, job_b])
        executor.shutdown()
        assert all(r.status == JobStatus.COMPLETED for r in reports)
        assert sorted(sink_a) == [1, 2]
        assert sorted(sink_b) == [3, 4]

    def test_job_listener_called(self):
        events = []

        class L(JobListener):
            def before_job_start(self, params): events.append("start")
            def after_job_end(self, report): events.append("end")

        job = (
            JobBuilder().named("listener-test")
            .reader(IterableRecordReader([1]))
            .job_listener(L())
            .writer(DevNullRecordWriter())
            .build()
        )
        JobExecutor().execute(job)
        assert events == ["start", "end"]

    def test_pipeline_listener_called(self):
        events = []

        class L(PipelineListener):
            def before_record_processing(self, record): events.append("before")
            def after_record_processing(self, record): events.append("after")

        job = (
            JobBuilder().named("pl-test")
            .reader(IterableRecordReader([1, 2]))
            .pipeline_listener(L())
            .writer(DevNullRecordWriter())
            .build()
        )
        JobExecutor().execute(job)
        assert events == ["before", "after", "before", "after"]

    def test_missing_reader_raises(self):
        with pytest.raises(ValueError):
            JobBuilder().named("no-reader").writer(DevNullRecordWriter()).build()
