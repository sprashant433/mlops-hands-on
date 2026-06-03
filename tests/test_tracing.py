from fastapi import FastAPI

from mlops_lr.tracing import configure_tracing


def test_configure_tracing():
    app = FastAPI()

    configure_tracing(app)

    assert app is not None
