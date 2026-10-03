"""Audio helpers shared by the speech engines."""


class AudioDecodeError(ValueError):
    """The request carried audio that cannot be decoded: a client error, not a model failure."""
