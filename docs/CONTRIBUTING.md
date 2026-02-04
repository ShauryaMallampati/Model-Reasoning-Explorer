# Contributing

Thanks for your interest in MRE.

## Development Setup
- Python 3.11+
- Node 20+

```bash
make backend-install
make frontend-install
make dev
```

## Quality
- `make lint` to run linting
- `make test` to run tests
- `make format` to format code

## Guidelines
- Keep run artifacts deterministic and reproducible
- Avoid heavy UI frameworks or unnecessary dependencies
- Prefer explicit typing and clear docstrings
- Be transparent about interpretability limitations
