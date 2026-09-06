# GenLit and LiveLight baselines

- `repos/genlit/`, `repos/LiveLight/`: source snapshots tracked by the main Git repository, including local changes and upstream licenses.
- `scripts/`: 20 comparison setup, download, verification, trajectory, inference, and configuration files.
- Model weights remain at `/workspace/weights/genlit/` and `/workspace/weights/livelight/`, outside this directory and excluded from Git.

Run from `/workspace`:

```bash
# GenLit
bash baseline/scripts/setup_genlit.sh
bash baseline/scripts/download_genlit_weights.sh
bash baseline/scripts/run_genlit_scene_002583.sh

# GenLit multi-object weights
bash baseline/scripts/download_genlit_multi_weights.sh

# LiveLight: setup also downloads and verifies weights
bash baseline/scripts/setup_livelight.sh
bash baseline/scripts/run_livelight_scene2583.sh
```

Scripts assume `/workspace`, `/workspace/miniconda3`, and the existing environments
under `/workspace/conda_envs/`. Adjust these paths for another host. Run GenLit
setup again after this move to refresh its editable package installation.
Scene data, trajectories, prepared depth, and weights must exist before inference.
The upstream repositories also contain their original examples; use the launchers
and configurations in `baseline/scripts/` for this workspace's shared weights.

Source revisions and patch notes are in [repository notes](../docs/REPOSITORY.md).
The existing LiveLight patch is already applied. Nested Git metadata is preserved
only in the ignored `local_archive/baseline_git_metadata/` directory.
