import logging


class TestLogging:
    def test_default_logging(self) -> None:
        logging.info("here we are")

    def test_pytest_caplog(self, caplog) -> None:
        """Pytest's caplog fixture captures all LogRecord[s] logged during a test.

        caplog  allows you to inspect / verify logs were written.
        """

        def log_me(x: str) -> None:
            logging.info(f"i'm logging {x}")

        caplog.set_level(logging.INFO)

        logging.info("hello, info")
        logging.warning("hello, warn")

        log_me("damon")

        assert len(caplog.records) == 3

        assert caplog.records[0].message == "hello, info"
        assert caplog.records[0].levelno == logging.INFO

        assert caplog.records[1].message == "hello, warn"
        assert caplog.records[1].levelname == logging.getLevelName(logging.WARNING)

        assert caplog.records[2].message == "i'm logging damon"
