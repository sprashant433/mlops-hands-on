from importlib.metadata import version


def test_evidently_dependency_installed():
    installed_version = version("evidently")

    assert installed_version == "0.7.11"
