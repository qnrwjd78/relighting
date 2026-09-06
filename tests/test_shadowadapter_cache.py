from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_shadowadapter_cache.py"
SPEC = importlib.util.spec_from_file_location("run_shadowadapter_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_image(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:19, :23]
    array = np.empty((19, 23, 3), dtype=np.uint8)
    for channel, base in enumerate(rgb):
        array[..., channel] = (base + xx * (channel + 1) + yy) % 256
    Image.fromarray(array, "RGB").save(path)


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    base = root / "dataset"
    baseline = root / "baseline"
    source = base / "scene_000001" / "source.png"
    _write_image(source, (90, 100, 110))
    _write_image(baseline / "scene_000001_light_003.png", (45, 70, 95))
    _write_image(baseline / "scene_000001_light_004.png", (130, 120, 100))
    manifest = root / "manifest.jsonl"
    rows = [
        {
            "scene_id": "scene_000001",
            "light_id": light,
            "input_image": "scene_000001/source.png",
        }
        for light in (3, 4)
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return base, baseline, manifest


def _args(root: Path, extra: list[str] | None = None):
    base, baseline, manifest = _fixture(root)
    argv = [
        "--manifest",
        str(manifest),
        "--base-path",
        str(base),
        "--baseline-dir",
        str(baseline),
        "--output-dir",
        str(root / "cache"),
        "--backend",
        "mock",
        "--device",
        "cpu",
        "--batch-size",
        "2",
    ]
    if extra:
        argv.extend(extra)
    return MODULE.create_parser().parse_args(argv)


class ShadowAdapterCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_mock_backend_writes_source_target_delta_and_index(self) -> None:
        args = _args(self.root)
        summary = MODULE.run(args)
        output = self.root / "cache"

        self.assertEqual(
            summary,
            {
                "rows": 2,
                "unique_sources": 1,
                "source_caches_written": 1,
                "target_caches_written": 2,
                "delta_caches_written": 2,
            },
        )
        source_files = list((output / "sources").glob("*.npz"))
        target_files = sorted((output / "targets").glob("*.npz"))
        delta_files = sorted((output / "deltas").glob("*.npz"))
        self.assertEqual(len(source_files), 1)
        self.assertEqual(len(target_files), 2)
        self.assertEqual(len(delta_files), 2)

        with np.load(source_files[0], allow_pickle=False) as source, np.load(
            target_files[0], allow_pickle=False
        ) as target, np.load(delta_files[0], allow_pickle=False) as delta:
            self.assertEqual(str(source["schema"].item()), MODULE.IMAGE_SCHEMA)
            self.assertEqual(int(source["output_size"].item()), 480)
            self.assertEqual(str(source["cache_id"].item()), "mock-v1-seed-0")
            for key in MODULE.IMAGE_CACHE_KEYS:
                self.assertEqual(source[key].shape, (480, 480))
                self.assertEqual(source[key].dtype, np.float16)
            self.assertTrue(
                np.all((source["final_prob"] >= 0) & (source["final_prob"] <= 1))
            )
            expected = np.maximum(
                target["final_prob"].astype(np.float32)
                - source["final_prob"].astype(np.float32),
                0.0,
            ).astype(np.float16)
            np.testing.assert_array_equal(
                delta["positive_delta_final_prob"], expected
            )
            self.assertEqual(
                delta["positive_delta_coarse_prob"].shape, (480, 480)
            )

        records = [
            json.loads(line)
            for line in (output / "index.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["source_cache"], records[1]["source_cache"])
        self.assertEqual(
            records[0]["adapter_source_cache"], records[0]["source_cache"]
        )
        self.assertEqual(
            records[0]["adapter_target_cache"], records[0]["target_cache"]
        )
        self.assertEqual(
            records[0]["adapter_delta_cache"],
            records[0]["positive_delta_cache"],
        )
        self.assertTrue(
            records[0]["baseline_prediction"].endswith(
                "scene_000001_light_003.png"
            )
        )
        self.assertEqual(records[0]["output_size"], 480)

    def test_skip_existing_is_resumable_and_mock_is_deterministic(self) -> None:
        args = _args(self.root, ["--mock-seed", "17"])
        MODULE.run(args)
        output = self.root / "cache"
        cache_files = sorted(output.glob("*/*.npz"))
        before = {path: path.stat().st_mtime_ns for path in cache_files}
        first_target = output / "targets/scene_000001_light_003.npz"
        with np.load(first_target, allow_pickle=False) as archive:
            target_before = archive["final_logit"].copy()

        summary = MODULE.run(args)
        after = {path: path.stat().st_mtime_ns for path in cache_files}
        self.assertEqual(summary["source_caches_written"], 0)
        self.assertEqual(summary["target_caches_written"], 0)
        self.assertEqual(summary["delta_caches_written"], 0)
        self.assertEqual(before, after)

        second_root = self.root / "second"
        rerun = _args(second_root, ["--mock-seed", "17", "--limit", "1"])
        MODULE.run(rerun)
        second_target = next((second_root / "cache/targets").glob("*.npz"))
        with np.load(second_target, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["final_logit"], target_before)

    def test_changed_backend_fingerprint_recomputes_existing_cache(self) -> None:
        first = _args(self.root, ["--mock-seed", "1"])
        MODULE.run(first)
        second = MODULE.create_parser().parse_args(
            [
                "--manifest",
                first.manifest,
                "--base-path",
                first.base_path,
                "--baseline-dir",
                first.baseline_dir,
                "--output-dir",
                first.output_dir,
                "--backend",
                "mock",
                "--device",
                "cpu",
                "--batch-size",
                "2",
                "--mock-seed",
                "2",
            ]
        )
        summary = MODULE.run(second)
        self.assertEqual(summary["source_caches_written"], 1)
        self.assertEqual(summary["target_caches_written"], 2)
        self.assertEqual(summary["delta_caches_written"], 2)
        target = self.root / "cache/targets/scene_000001_light_003.npz"
        with np.load(target, allow_pickle=False) as archive:
            self.assertEqual(str(archive["cache_id"].item()), "mock-v1-seed-2")

    def test_manifest_preflight_reports_missing_baseline_before_inference(
        self,
    ) -> None:
        args = _args(self.root)
        missing = self.root / "baseline/scene_000001_light_004.png"
        missing.unlink()
        with self.assertRaisesRegex(MODULE.CacheError, "baseline prediction") as error:
            MODULE.run(args)
        self.assertIn(str(missing.resolve()), str(error.exception))
        self.assertFalse((self.root / "cache").exists())

    def test_official_asset_validation_is_lazy_and_actionable(self) -> None:
        root = self.root / "AdapterShadow"
        (root / "models_exp/sam").mkdir(parents=True)
        (root / "models_exp/sam/build_sam.py").write_text("# marker\n")
        (
            root / "efficientnet/hub/rwightman_gen-efficientnet-pytorch_master"
        ).mkdir(parents=True)
        (
            root
            / "efficientnet/hub/rwightman_gen-efficientnet-pytorch_master/hubconf.py"
        ).write_text("# marker\n")
        assets = MODULE.OfficialAssets(
            root=root,
            checkpoint=root / "checkpoint/sbu.ckpt",
            sam_checkpoint=root / "checkpoint/sam/sam_vit_b_01ec64.pth",
            efficientnet_checkpoint=(
                root
                / "efficientnet/hub/checkpoints/tf_efficientnet_b1_ap-44ef0a3d.pth"
            ),
        )

        with self.assertRaises(MODULE.CacheError) as error:
            MODULE.validate_official_assets(assets)
        message = str(error.exception)
        self.assertIn("AdapterShadow task checkpoint", message)
        self.assertIn("SAM ViT-B checkpoint", message)
        self.assertIn("EfficientNet-B1 checkpoint", message)
        self.assertIn("No model was loaded", message)

    def test_explicit_baseline_key_uses_baseline_directory(self) -> None:
        base = self.root / "dataset"
        baseline = self.root / "predictions"
        _write_image(base / "src.png", (20, 30, 40))
        _write_image(baseline / "nested/custom.png", (80, 90, 100))
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "scene_id": "scene_x",
                    "input_image": "src.png",
                    "prediction": "nested/custom.png",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        args = MODULE.create_parser().parse_args(
            [
                "--manifest",
                str(manifest),
                "--base-path",
                str(base),
                "--baseline-dir",
                str(baseline),
                "--baseline-key",
                "prediction",
                "--output-dir",
                str(self.root / "out"),
                "--backend",
                "mock",
                "--device",
                "cpu",
            ]
        )
        MODULE.run(args)
        record = json.loads(
            (self.root / "out/index.jsonl").read_text(encoding="utf-8").strip()
        )
        self.assertEqual(
            record["baseline_prediction"],
            str((baseline / "nested/custom.png").resolve()),
        )


if __name__ == "__main__":
    unittest.main()
