import logging
import io
import sys

class CapturingHandler(logging.Handler):
    def __init__(self, logger, handler):
        super().__init__()
        self.logger = logger
        self.handler = handler
        self.captured_output = io.StringIO()

    def emit(self, record):
        self.handler.emit(record)
        self.captured_output.write(self.format(record) + '\n')

    def __enter__(self):
        self.logger.addHandler(self.handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.removeHandler(self.handler)
