import traceback

class CustomException(Exception):
    def __init__(self, message, errors=None):
        self.errors = errors
        self.traceback = traceback.format_exc()
        super().__init__(message)