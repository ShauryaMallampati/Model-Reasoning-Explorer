SHELL := /bin/bash

.PHONY: dev lint format test backend-install frontend-install

backend-install:
	python3 -m pip install -U pip
	python3 -m pip install -e backend[dev]

frontend-install:
	cd frontend && npm install

dev:
	./scripts/dev.sh

lint:
	./scripts/lint.sh

format:
	./scripts/format.sh

test:
	./scripts/test.sh
