from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@dataclass
class Otel:
    meter_provider: MeterProvider
    reader: InMemoryMetricReader
    tracer_provider: TracerProvider
    spans: InMemorySpanExporter

    def collect(self) -> dict[str, list[tuple[Any, dict[str, Any]]]]:
        """metric name -> [(value, attributes)]; histograms report (count, sum)."""
        out: dict[str, list[tuple[Any, dict[str, Any]]]] = defaultdict(list)
        data = self.reader.get_metrics_data()
        for rm in data.resource_metrics if data else []:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    for point in metric.data.data_points:
                        value = (point.count, point.sum) if hasattr(point, "bucket_counts") else point.value
                        out[metric.name].append((value, dict(point.attributes)))
        return out


@pytest.fixture
def otel() -> Otel:
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    spans = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(spans))
    return Otel(meter_provider, reader, tracer_provider, spans)
