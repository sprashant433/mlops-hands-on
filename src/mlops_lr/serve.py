import uvicorn

from mlops_lr.config import load_config


def serve() -> None:
    config = load_config()

    uvicorn.run(
        "mlops_lr.api:app",
        host=config.serving.host,
        port=config.serving.port,
        reload=False,
    )


if __name__ == "__main__":
    serve()
