# Repository contents and GitHub preparation

This repository contains TokenLight / Wan training and inference code, physical
tasks, shadow refinement tools, experiment configurations, and documentation.
Run commands from the repository root. Many experiment paths assume `/workspace`;
the Docker workspace mount in the main README matches that layout.

## What belongs in Git

- `model/`, `scripts/`, `utils/`, `tokenlight_dataset/`: project source code.
- `configs/`: experiment parameters and distributed training configuration.
- `tests/`, `docs/`, `docker/`, `patches/`, and root Markdown files: tests,
  documentation, environment definition, and local third-party changes.
- Small intentional example assets may be added alongside documentation. Images
  are not globally ignored; generated results belong under `outputs/`.

`.gitignore` keeps datasets, manifests under `data_train/`, weights, downloaded
tokenizers under `model/Wan-AI/`, checkpoints, results, Conda installations,
package caches, third-party checkouts, credentials, and temporary files local.
These rules do not delete files. Model binaries, NumPy caches, archives, and
generated videos are excluded by extension as well.

`.dockerignore` limits the build context to the Dockerfile and its dependency
file. The image installs dependencies; project source and local data are supplied
by the runtime workspace mount.

## Local dependencies

Install the main Python dependencies from `docker/requirements.txt`, or build the
documented CUDA image. Download Wan weights separately to
`weights/Wan2.2-TI2V-5B/`. Dataset metadata and cache paths in each experiment
configuration must point to your own prepared data.

The following optional comparison / preprocessing checkouts were present when
the workspace was prepared. Revisions describe the local checkouts, not a claim
that every experiment has been validated with them.

| Local path | Upstream | Revision |
| --- | --- | --- |
| `repos/LiveLight` | https://github.com/mayuelala/LiveLight | `f17e981f3f7afe2de70345a9547fd6534ff64ef8` |
| `repos/genlit` | https://github.com/sbharadwajj/genlit | `a1a8a811c917443fa41ccc910523aea555d01e38` |
| `external/AdapterShadow` | https://github.com/LeipingJie/AdapterShadow | `7171c6929f4f07117d73c10b821cffade8d8b38c` |
| `external/FOCUS` | https://github.com/geshang777/FOCUS | `18b5c39e0905b5bc984057ff18c8104e1ae8e3b4` |
| `external/detectron2` | https://github.com/facebookresearch/detectron2 | `a2f4a8771ab77e8411c26b27f24f9489a28a2453` |

Clone the dependencies needed for your experiment into those paths and check out
the listed revisions. Each dependency and model retains its upstream license.
`external/python_pkgs/` contains local installed packages and is also excluded.
See `scripts/setup_livelight.sh`, `scripts/setup_genlit.sh`, and
[the shadow pipeline documentation](SHADOW_C2F_PIPELINE.md) for setup details.
The GenLit setup script expects its repository and a local Conda installation to
exist already.

LiveLight and FOCUS had local changes to tracked files. These are preserved in
`patches/livelight-local.patch` (per-frame light trajectories) and
`patches/focus-local.patch` (PyTorch C++/CUDA API adjustments). On fresh checkouts
of the revisions above, apply them from the project root:

```bash
git -C repos/LiveLight apply ../../patches/livelight-local.patch
git -C external/FOCUS apply ../../patches/focus-local.patch
```

The current workspace already includes those changes; do not apply them twice.
The patches capture tracked changes relative to HEAD, not installed packages,
untracked files, or a complete environment export.

## Review before committing

```bash
git status --short
git diff --check
git ls-files --others --exclude-standard
git ls-files -ci --exclude-standard
```

The last command should print nothing: ignore rules do not remove files already
tracked by Git. The workspace already contained edits and 43 deleted tracked
files before this cleanup; those changes were preserved. Review them when making
your upload commit. No commit, history rewrite, or push was performed by the
cleanup. Existing commits remain part of a normal push, including assets deleted
from the current working tree.

## Further documentation

- [480px experiment configurations](../configs/train_480/README.md)
- [Physical tasks](../model/physical_tasks/README.md)
- [CoShadow fixed32](COSHADOW_FIXED32.md)
- [Physics-conditioned training](PHYSICS_CONDITIONED_TRAINING.md)
- [Shadow refinement pipeline](SHADOW_C2F_PIPELINE.md)
- [MoGe3 spatial geometry design](TOKENLIGHT_MOGE3_SPATIAL_GEOMETRY_DESIGN.md)
- [Model notes](../model_info.md)
- [Model settings summary](../model_settings_summary.md)
