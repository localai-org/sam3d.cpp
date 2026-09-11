# Optional Python tooling

This nested project contains the pinned dependencies for safe tensor conversion,
reference-data validation and optional Hugging Face publication. It is not the
sam3d.cpp inference runtime. Native build/install and the Go demo do not use it.
PyTorch reference execution remains in the separately reviewed containers.

From the repository root:

```sh
uv sync --project reference/python --frozen
uv run --project reference/python --frozen python -m unittest discover -s tests -v
```

Add `--extra download` to both commands when using HF tooling. The environment
is created here as `.venv`; `--project` does not change the command's working
directory. Thus `scripts/...`, `tests/...` and model paths are root-relative.

See [development](../../docs/DEVELOPMENT.md),
[reference setup](../README.md) and [publishing](../../distribution/README.md).
