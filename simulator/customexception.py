class CustomException(Exception):
    def __init__(self, ex):
        self.type = ex['type']
        self.traceback = ex['traceback']
        super().__init__(ex['message'])