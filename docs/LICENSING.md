# Licensing

Copyright 2026 sam3d.cpp contributors. Original contributions are licensed
under [Apache-2.0](../LICENSE), **except for third-party material and adaptations
identified in file headers and [NOTICE](../NOTICE)**.
Those retain their original terms. This is not an Apache-only distribution.
The Apache grant does not offer an alternative license for Meta derivatives.

| Material | Applicable terms |
| --- | --- |
| Original library/API infrastructure, demo and tooling, excluding identified adaptations | Apache-2.0 |
| Adapted SAM 3D Body/Objects model and preprocessing implementations and associated reference material | [SAM License](../LICENSES/SAM.txt) |
| Adapted DINOv3 embedding, transformer and backbone implementations | [DINOv3 License](../LICENSES/DINOv3.md), plus SAM terms where SAM-derived material is incorporated |
| MHR geometry/constants and adaptations | [Apache-2.0](../LICENSES/MHR-Apache-2.0.txt), with [Momentum MIT](../LICENSES/Momentum.txt) for Momentum-derived implementations |
| GGML submodule and GGML-derived Vulkan patches | [GGML MIT](../ggml/LICENSE) |
| Other copied/adapted algorithms and helpers | Their retained notices in `LICENSES/`; file-by-file provenance in `NOTICE` |
| Bundled Three.js | [Three.js MIT](../demo/web/vendor/THREE-LICENSE.txt) |

The provenance document identifies files, original repositories/revisions,
copyright holders and modifications. Multiple applicable notices must be
preserved for mixed-source files. Synthetic fixtures are not neural checkpoints;
fixtures containing original MHR constants retain the MHR asset terms.
LocalAI names/logos identify this project, not an endorsement by Meta; the
Apache license does not grant third-party trademark rights.

## Converted weights

GGUF is a file format, not a license. Model weights are not included in the
source checkout. SAM-derived conversions retain the SAM License; DINO material
retains DINOv3 terms; the independently released MHR companion retains its
Apache-2.0 asset license and applicable notices. Our source license does not
relicense any of these weights.

SAM and DINO section 1 grants modification and redistribution rights, subject
to distributing derivatives under their respective Agreements with a copy of
the Agreement. Both contain use, legal/trade-control and other restrictions,
including a reverse-engineering/decompilation prohibition. They are not
unrestricted Apache-style licenses. Read their complete terms before use or
redistribution. Converting or quantizing weights does not remove those terms.

The publication tooling bundles the license texts, component-specific notices,
source revisions, conversion descriptions and artifact checksums. It requires
an explicit license-compliance acknowledgement before network writes. That
acknowledgement is not legal advice or a substitute for satisfying the terms.
The current bundle is the F32 Body pose branch, not low-bit quantization, the
full hand-refined estimator, or SAM 3D Objects.

## Distributing source or binaries

Retain `LICENSE`, the consolidated `NOTICE`, this file and applicable
third-party license files. CMake installs the notices under `share/sam3d`,
preserving the `docs/`, `LICENSES/` and bundled dependency license paths.
The optional demo embeds its web assets; include the Three.js license
when distributing its binary. Further modifications should be identified and
must not remove upstream copyright notices or license conditions.

See [publication instructions](../distribution/README.md) for the proposed HF
bundle and its explicit-file uploader. No upstream gate, publisher approval or
license obligation is waived by this project.
