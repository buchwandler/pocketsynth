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


class AssetCacheError(AssetError):
    pass


class CatalogUnavailableError(AssetError):
    pass


class OfflineAssetError(AssetError):
    pass


class SessionCreationError(PocketSynthError):
    pass


class ModelInferenceError(PocketSynthError):
    pass


class RuntimeCapabilityError(PocketSynthError):
    pass


class VoicePromptError(PocketSynthError):
    pass


class VoiceBindingError(PocketSynthError):
    pass


class UnsupportedPlanDirectiveError(PocketSynthError):
    pass


class UnsupportedPlanLanguageError(PocketSynthError):
    pass


class PipelineClosedError(PocketSynthError):
    pass
