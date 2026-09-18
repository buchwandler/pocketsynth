import pytest

from pocketsynth._onnxvoice import normalize_pocket_ref, normalize_provider_request
from pocketsynth.errors import BundleNotFoundError


def test_normalize_pocket_ref():
    assert normalize_pocket_ref("english_2026-04") == "pocket:english_2026-04"
    assert normalize_pocket_ref("pocket:german") == "pocket:german"
    with pytest.raises(BundleNotFoundError):
        normalize_pocket_ref("piper:en_US-lessac-medium")


def test_provider_default_is_cpu():
    providers, options = normalize_provider_request(None, None)
    assert providers == "CPUExecutionProvider"
    assert options is None
