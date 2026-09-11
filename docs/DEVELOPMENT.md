# Development

The library and native tests are C/C++ projects built with CMake. Go is needed
only for the optional demo. Python is **not an inference or native-build
dependency**: its isolated project is in [reference/python](../reference/python).
Run the commands below from the source root.

## Native correctness builds

```sh
cmake --preset debug
cmake --build --preset debug -j2
ctest --preset debug
```

Debug enables ASan/UBSan. `optimized-sanitizers` keeps both with `-O2`, symbols
and assertions. `release` disables sanitizers for performance/deployment.
LeakSanitizer needs execution outside ptrace-based sandboxes; do not disable
it to hide a sandbox limitation.

Some NVIDIA drivers fail Vulkan ICD initialization when ASan is loaded. For
that diagnosed case, `vulkan-ubsan` and `vulkan-optimized` retain UBSan with
ASan disabled. Keep the CPU sanitizer build as well. A successful AMD test is
not validation of the NVIDIA path.

Full-model jobs should run serially under hard RAM caps with host headroom.
The optional Linux/systemd [bounded runner](../reference/MEMORY_SAFETY.md)
implements that guard. Docker references need their own container limits.

## Fuzzing

With Clang and libFuzzer installed:

```sh
cmake --preset fuzz
cmake --build --preset fuzz -j2
./build/fuzz/bin/sam3d-crop-fuzz -runs=100000 -max_len=128
./build/fuzz/bin/sam3d-objects-image-fuzz -runs=100000 -max_len=128
./build/fuzz/bin/sam3d-body-model-fuzz -runs=100000 -max_len=512
./build/fuzz/bin/sam3d-body-result-fuzz -runs=100000 -max_len=512
./build/fuzz/bin/sam3d-hand-crop-fuzz -runs=100000 -max_len=256
./build/fuzz/bin/sam3d-hand-frame-fuzz -runs=100000 -max_len=512
```

Fuzzing exercises valid handle lifetimes and bounded caller-owned buffers;
it does not promise recovery from arbitrary invalid pointers. Model API fuzzing
does not load the full GGUF weights.

## Optional Python tools

```sh
uv sync --project reference/python --frozen
uv run --project reference/python --frozen python -m unittest discover -s tests -v
```

`--project` selects the nested Python environment without changing the working
directory; scripts and input paths still resolve from the source root.
The base project installs only data-validation dependencies. Add `--extra
download` for HF publication/download tools. This creates
`reference/python/.venv`, which is ignored by Git. It does not install PyTorch.

Official PyTorch execution is separate, isolated and hash-pinned. Follow the
[reference guide](../reference/README.md), not arbitrary host checkpoint loading.
The [publication workflow](../distribution/README.md) defaults to an offline dry run.

## Demo tests

```sh
(cd demo && go test -race ./...)
node --test tests/test_tracking_math.mjs tests/test_live_presentation.mjs tests/test_frame_pipeline.mjs
```

Node is used only for browser-logic tests, not to build the demo. Model-backed
headless Chrome QA uses the scripts described in the [demo guide](../demo/README.md)
and the [design](DESIGN.md). The optional skeleton recording browser test is
run through Go with explicit Node/Chromium paths; see
[skeleton export verification](SKELETON-EXPORT.md#verification).
Keep numerical and visual checks distinct.

## Documentation

Update [ROADMAP.md](ROADMAP.md) for current scope and remaining work. Put
experiment-specific commands/results under `reference/`; append historical
context to `reference/HISTORY.md` only when useful. Do not create a parallel
TODO/status checklist. The [design](DESIGN.md) describes the process, not an
agent's active goal or the current implementation status.
