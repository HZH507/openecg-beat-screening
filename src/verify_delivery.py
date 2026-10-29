"""Verify published frozen materials with the Python standard library only.

This never trains a model, performs inference, or rewrites experiment records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root, manifest):
    root = Path(root).resolve()
    entries = manifest["entries"]
    seen = set()
    for entry in entries:
        relative = entry["path"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or relative in seen:
            raise AssertionError("Unsafe or duplicate manifest path: " + relative)
        seen.add(relative)
        if not path.is_file():
            raise AssertionError("Missing published material: " + relative)
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise AssertionError("Byte/hash mismatch: " + relative)
    cv = root / "results/workpoint_cv_v2"
    frozen = read(cv / "frozen_workpoints.json")
    evaluation = read(cv / "ds2_evaluation.json")
    summary = read(cv / "ds2_summary.json")
    if digest(cv / "final_binary.pt") != frozen["final_model_sha256"]:
        raise AssertionError("Frozen binary checkpoint mismatch")
    if digest(cv / "ds2_predictions.npz") != summary["cached_predictions_sha256"]:
        raise AssertionError("Original DS2 prediction cache mismatch")
    for name, expected in frozen["source_sha256"].items():
        if digest(root / "src" / name) != expected:
            raise AssertionError("Frozen source byte mismatch: " + name)
    if evaluation["status"] != "complete" or evaluation["inference_passes"] != 1:
        raise AssertionError("Original clean DS2 audit changed")
    required = {
        "results/workpoint_cv_v2/oof_predictions.npz",
        "results/workpoint_cv_v2/ds2_predictions.npz",
        "results/workpoint_cv_v2/ds2_threshold_curve_exploratory.csv",
        "results/report_final/classification_predictions.npz",
        "results/report_final/noise_predictions.npz",
        "results/report_final/figures/fig06b_binary_roc.png",
        "results/sil/sil_predictions.npz",
        "results/openecg_demo/openecg_demo.json",
        "results/openecg_demo/openecg_sim.png",
    }
    if not required <= seen:
        raise AssertionError("Manifest omits required material: " + str(sorted(required - seen)))
    return {"status": "pass", "files_verified": len(entries),
            "frozen_source_bytes_preserved": True,
            "clean_ds2_inference_passes": evaluation["inference_passes"],
            "inference_performed_by_this_check": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    manifest = read(args.manifest or root / "PACKAGE_MANIFEST.json")
    print(json.dumps(verify(root, manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
