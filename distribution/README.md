# Publishing the Body GGUF bundle

Prepared target: **`LocalAI-io/sam-3d-body-dinov3-GGUF`**. `LocalAI-io` is the
Hugging Face namespace; it is not the GitHub organization name. No repository
creation or upload is implied by adding these files.

[The model card](body/README.md) describes the Body pose branch, known limits,
source revisions and component-specific licenses. [artifacts.json](body/artifacts.json)
pins the three approved F32/F32-I32 GGUFs by size and SHA-256. These are not
low-bit quants; BF16 runs from the same archives. No Objects release is included.
The card links `base_model` to Meta's original HF repository. No finetune or
quantized relation is asserted for this F32 format conversion.

## Local preflight (recommended first)

Python 3.11+ and Git suffice. No HF package, credentials or network is needed:

```sh
python3 scripts/publish_gguf.py \
  --backbone /path/to/body-dinov3-f32.gguf \
  --branch /path/to/body-pose-branch-f32.gguf \
  --mhr /path/to/mhr-lod1-f32.gguf
```

This streams the full file hashes using bounded memory and prints the exact
repository, file names, sizes, SHA-256 values and source commit/dirty status.
It does not scan directories, copy model files, deserialize checkpoints, or
send anything to HF. Inputs must be trusted regular files, not symlinks, and
remain immutable for the entire run. Hash/size mismatches fail closed; do not
update the pins merely to make a mismatched model pass. A newly converted or
quantized artifact requires independent conversion/parity review first.

## Upload — a separate explicit action

First review [LICENSING.md](../docs/LICENSING.md) and all model license texts. Commit
the release source and ensure the checkout/submodule is clean. Install the
optional pinned HF tooling and authenticate without putting tokens in scripts:

```sh
uv sync --project reference/python --frozen --extra download
uv run --project reference/python --frozen --extra download hf auth login
uv run --project reference/python --frozen --extra download python scripts/publish_gguf.py \
  --backbone /path/to/body-dinov3-f32.gguf \
  --branch /path/to/body-pose-branch-f32.gguf \
  --mhr /path/to/mhr-lod1-f32.gguf \
  --upload --accept-model-licenses
```

This requires an existing accessible model repository. Add **`--create-repo`**
only when authorized to create the public repository. `--repo-id namespace/name`
selects another destination and updates the card's download command. Existing
repository visibility is not changed. No remote files are deleted.

The uploader performs HF card validation and sends one commit containing only:

- The three explicitly supplied, approved GGUF components.
- The reviewed model card and component-specific NOTICE.
- SAM, DINOv3, MHR, Momentum and original-documentation Apache license texts.
- `MANIFEST.json` with source revisions and file identities.
- `SHA256SUMS` covering all supplied files and the generated manifest (not itself).

The original source checkout must remain clean. The remote parent commit is
checked to avoid silently overwriting a concurrent repository update. Failure
after an explicitly authorized repository creation may leave an empty repository;
the tool does not delete it automatically. Network errors may require checking
the remote commit before retrying.

Source-repository installation packages retain `LICENSE`, `NOTICE`,
`docs/LICENSING.md` and applicable third-party license texts. The model bundle's root
`LICENSE` is deliberately **SAM**, not the source project's Apache license.

## Verification

```sh
uv run --project reference/python --frozen python -m unittest discover -s tests -p 'test_publication.py' -v
```

Tests use synthetic tiny files and mock HF calls: no weights, downloads, remote
repository creation or GPU required. Before an actual release, repeat a clean
source build/test and model-specific numerical/visual acceptance as appropriate.
