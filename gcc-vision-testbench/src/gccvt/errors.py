"""Error types. Every error raised on purpose carries a message telling the user what to fix."""


class GccvtError(Exception):
    """Base class; caught by the CLI wrapper and printed without a traceback."""


class ConfigError(GccvtError):
    pass


class DownloadError(GccvtError):
    pass


class ChecksumError(DownloadError):
    pass


class ToolMissingError(GccvtError):
    pass


class MediaError(GccvtError):
    pass
