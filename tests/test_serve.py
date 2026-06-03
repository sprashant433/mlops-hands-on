from mlops_lr.serve import serve


def test_serve_imports():
    assert callable(serve)
