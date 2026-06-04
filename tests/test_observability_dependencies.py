def test_opentelemetry_dependencies_import():
    import opentelemetry.trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider

    assert opentelemetry.trace is not None
    assert OTLPSpanExporter is not None
    assert FastAPIInstrumentor is not None
    assert TracerProvider is not None
