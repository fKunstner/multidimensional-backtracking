import logging
import time
from typing import Optional


class RateLimitedLogger:
    """A logger that rate-limits to one message every X seconds or Y call."""

    def __init__(self, time_interval: int = 5, call_interval: Optional[int] = None):
        """Filter calls to ``log`` based on the time and the number of calls.

        The logger only allows one message to be logged every ``time_interval``
        (measured in seconds). If ``call_interval`` is given,

        Args:
            time_interval: Limit messages to one every time_interval
            call_interval: Limit messages to one every call_interval
        """
        self.last_log = None
        self.ignored_calls = 0
        self.time_interval = time_interval
        self.call_interval = call_interval

    def _should_log(self) -> bool:
        def _never_logged():
            return self.last_log is None

        def _should_log_time():
            return time.perf_counter() - self.last_log > self.time_interval

        def _should_log_count():
            return (
                self.call_interval is not None
                and self.ignored_calls >= self.call_interval
            )

        should_log = [_never_logged, _should_log_count, _should_log_time]
        return any(cond() for cond in should_log)

    def log(self, *args, force=False, **kwargs):
        """Might pass the arguments to ``getlogger(__name__).log``."""
        if force:
            logging.getLogger(__name__).info(*args, **kwargs)
        elif self._should_log():
            logging.getLogger(__name__).info(*args, **kwargs)
            self.last_log = time.perf_counter()
            self.ignored_calls = 0
        else:
            self.ignored_calls += 1
