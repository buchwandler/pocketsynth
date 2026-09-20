import pytest

from pocketsynth._onnxvoice import _call, normalize_pocket_ref, normalize_provider_request
from pocketsynth.errors import BundleNotFoundError, ModelInferenceError


def test_normalize_pocket_ref():
    assert normalize_pocket_ref("english_2026-04") == "pocket:english_2026-04"
    assert normalize_pocket_ref("pocket:german") == "pocket:german"
    with pytest.raises(BundleNotFoundError):
        normalize_pocket_ref("piper:en_US-lessac-medium")


def test_provider_default_is_cpu():
    providers, options = normalize_provider_request(None, None)
    assert providers == "CPUExecutionProvider"
    assert options is None

def test_runtime_error_mapping_preserves_cause():
    class InferenceFailure(Exception):
        pass

    def fail():
        raise InferenceFailure("missing graph output 'audio'")

    with pytest.raises(ModelInferenceError, match="missing graph output 'audio'") as caught:
        _call("infer", fail)
    assert isinstance(caught.value.__cause__, InferenceFailure)
