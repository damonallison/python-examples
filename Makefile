# Phony targets are targets are *not* meant to be names of files, rather they
# are names for targets to be executed. They will *always* be executed, even
# if a file or directory with the same name exists.
.PHONY: all lint test

all: lint test

lint:
	poetry run mypy --strict tests/**/*.py

test:
	# Skip ML tests for performance reasons
	poetry run pytest -s tests -m "not ml and not plot"
