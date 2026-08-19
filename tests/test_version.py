from src.__version__ import __version__


def test_application_version_is_1_0() -> None:
    assert __version__ == "1.0"
