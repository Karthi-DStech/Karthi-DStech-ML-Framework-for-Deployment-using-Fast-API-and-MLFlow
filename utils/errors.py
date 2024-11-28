class CustomError(Exception):
    """
    Base class for custom errors in the application.
    """

    def __init__(self, message: str):
        super().__init__(message)


class DataLoadingError(CustomError):
    """
    Raised when there is an error loading data.
    """

    pass
