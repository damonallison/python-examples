import logging


class TestLogging:
    def test_logging(self, caplog) -> None:
        """Pytest's caplog fixture captures all LogRecord[s] logged during a test."""

        caplog.set_level(logging.INFO)

        logging.info("hello, info")
        logging.warning("hello, warn")

        # all logs sent to the logger during the test are available on the
        # fixture. this can be used to verify logs were written.

        assert len(caplog.records) == 2

        assert caplog.records[0].message == "hello, info"
        assert caplog.records[0].levelno == logging.INFO

        assert caplog.records[1].message == "hello, warn"
        assert caplog.records[1].levelname == logging.getLevelName(logging.WARNING)
