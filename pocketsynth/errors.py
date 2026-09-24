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
