class PocketSynthError(Exception):
    """Base error for pocketsynth."""


class OptionalDependencyError(PocketSynthError):
    pass


class BundleError(PocketSynthError):
    pass


class BundleNotFoundError(BundleError):
    pass


class UnsupportedBundleError(BundleError):
    pass


class AssetError(PocketSynthError):
    pass


class AssetDownloadError(AssetError):
    pass


class AssetAccessError(AssetDownloadError):
    """A remote Pocket asset requires credentials or repository access."""


class AssetCacheError(AssetError):
    pass


class CatalogUnavailableError(AssetError):
    pass


class OfflineAssetError(AssetError):
    pass


class SessionCreationError(PocketSynthError):
    pass


class BundleLanguageError(PocketSynthError):
    """The requested language is incompatible with the active Pocket bundle."""


class ModelInferenceError(PocketSynthError):
    pass


class RuntimeCapabilityError(PocketSynthError):
    pass


class RuntimeClosedError(PocketSynthError):
    pass


class VoicePromptError(PocketSynthError):
    pass


class SynthesisError(PocketSynthError):
    """Base class for request-level synthesis failures."""


class InvalidRequestError(SynthesisError, ValueError):
    """Base class for invalid synthesis request values."""


class EmptyTextError(InvalidRequestError):
    """Raised when a synthesis request contains no speakable text."""


class InvalidLanguageError(InvalidRequestError):
    """Raised when a synthesis request specifies an invalid language."""


class InvalidVoiceError(InvalidRequestError):
    """Raised when a prepared voice is incompatible with the runtime."""


class InvalidGenerationConfigError(InvalidRequestError):
    """Raised when generation controls contain invalid values."""


class SynthesisInputTooLongError(SynthesisError, ValueError):
    """The encoded full request exceeds the bundle's model token limit."""

    def __init__(
        self, *, text_length: int | None, token_count: int, max_tokens: int, bundle_id: str
    ) -> None:
        self.text_length = text_length
        self.token_count = token_count
        self.max_tokens = max_tokens
        self.bundle_id = bundle_id
        source_length = (
            f"{text_length} characters"
            if text_length is not None
            else "an unknown source-text length"
        )
        super().__init__(
            f"Request text of {source_length} encoded to {token_count} tokens; "
            f"bundle {bundle_id!r} allows at most {max_tokens}."
        )


class UnsupportedFeatureError(SynthesisError):
    """Raised when a request uses a feature unsupported by PocketSynth."""

    def __init__(self, *, feature: str) -> None:
        self.feature = feature
        super().__init__(f"PocketSynth does not support {feature}.")
