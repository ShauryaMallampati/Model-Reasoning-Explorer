# Model Reasoning Explorer (MRE)

Model Reasoning Explorer is a production-grade, open-source DevTools-style debugger for ML model inference. It lets you inspect, compare, and validate model behavior on individual examples and dataset slices with rich internal signals such as activations, attention, gradients, and logit lens views.

## Highlights
- Single-example debugging for text and vision
- Comparative debugging between runs or models
- Counterfactual sandbox for minimal-change tests
- Dataset slice explorer with clustering and exportable reports
- Portable run artifacts and static HTML reports

## Quickstart
1. Install backend dependencies

```bash
make backend-install
```

2. Install frontend dependencies

```bash
make frontend-install
```

3. (Optional) Download demo models for offline use

```bash
python3 scripts/download_models.py
```

4. Start dev mode (backend + frontend)

```bash
make dev
```

5. Open the UI

- Frontend: http://localhost:5173
- Backend: http://localhost:8000

## Demos
- Text demo:

```bash
./scripts/demo_text.sh
```

- Vision demo:

```bash
./scripts/demo_vision.sh
```

## Configuration
MRE reads configuration from `backend/mre_config.toml`. You can override values with environment variables.

- `MRE_ALLOW_GPU=1` to enable GPU if available (CPU is default)
- `MRE_CONFIG=/path/to/mre_config.toml` to load a custom config file

## Docs
- Architecture: `docs/ARCHITECTURE.md`
- Roadmap: `docs/ROADMAP.md`
- Contributing: `docs/CONTRIBUTING.md`
- Security: `docs/SECURITY.md`

## License
MIT
