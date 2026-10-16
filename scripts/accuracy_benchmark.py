"""Reproducible full-chain pixel-accuracy and runtime verification.

The pixel gate is a development target, not physical instrument certification.
Repeated timing uses the same image; accuracy varies independently by seed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'src', help='directory containing centerline package')
    parser.add_argument('--output', type=Path, default=ROOT / 'result/v5_benchmark/summary.json')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 37, 67, 97, 127])
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--rmse-limit', type=float, default=.1)
    parser.add_argument('--min-coverage', type=float, default=.95)
    parser.add_argument('--require-pass', action='store_true', help='exit nonzero if any joint development gate fails')
    parser.add_argument('--save-images', action='store_true', help='save lossless input PNGs beside report')
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.rmse_limit <= 0 or not 0 < args.min_coverage <= 1:
        parser.error('invalid repeats, warmup, RMSE limit or coverage')
    source = args.source.resolve()
    sys.path[:0] = [str(source), str(ROOT / 'scripts')]
    # Import first: legacy generator scripts also insert src into sys.path.
    from centerline.pipeline import RealtimePipelineConfig, run_realtime_centerline_pipeline
    from centerline.normal_guided_steger import StegerConfig, extract_normal_guided_steger
    from centerline.normal_guided_centroid import errors_against_truth_normal_distance
    from realtime_scenarios import build_scenarios
    from single_factor_scenarios import PARAMETERS
    image_folder = args.output.parent / 'test_images'
    if args.save_images:
        from PIL import Image
        image_folder.mkdir(parents=True, exist_ok=True)
    modern = 'refine_with_steger' in RealtimePipelineConfig.__dataclass_fields__
    config = RealtimePipelineConfig(roi=(0,60,512,462), y_min=60, y_max=450)

    def run(image):
        start = time.perf_counter()
        result = run_realtime_centerline_pipeline(image, config)
        if modern:
            final = result['selected']
        else:
            p = result['preprocess']
            final = extract_normal_guided_steger(p.image, guided=result['guided'], steger_config=StegerConfig(), roi_mask=p.roi_mask, valid_mask=p.valid_mask)
        elapsed = (time.perf_counter() - start) * 1000
        return final, elapsed

    rows = []
    for seed in args.seeds:
        for scenario in build_scenarios(seed):
            image_file = None
            if args.save_images:
                image_file = image_folder / f'{scenario.name}_seed{seed}.png'
                Image.fromarray(scenario.image).save(image_file)
                with Image.open(image_file) as saved:
                    assert saved.tobytes() == scenario.image.tobytes(), 'Saved pixels changed'
            for _ in range(args.warmup):
                run(scenario.image)
            times = []
            for _ in range(args.repeats):
                final, elapsed = run(scenario.image)
                times.append(elapsed)
            errors = errors_against_truth_normal_distance(final['estimated_x'], final['estimated_y'], scenario.truth_y)
            mask = final['valid'] & np.isfinite(errors)
            errors[~mask] = np.nan
            observed = errors[mask].astype(np.float64)
            n = int(mask.sum())
            rmse = float(np.sqrt(np.mean(observed**2))) if n else None
            coverage = n / errors.size
            misses = int(np.count_nonzero(np.asarray(times) > config.frame_budget_ms))
            row = dict(seed=seed, scenario=scenario.name, image_sha256=hashlib.sha256(scenario.image.tobytes()).hexdigest(),
                       valid_points=n, expected_points=int(errors.size), coverage=coverage,
                       rmse_px=rmse, mean_abs_px=float(np.mean(np.abs(observed))) if n else None,
                       bias_px=float(np.mean(observed)) if n else None,
                       p95_abs_px=float(np.percentile(np.abs(observed),95)) if n else None,
                       max_abs_px=float(np.max(np.abs(observed))) if n else None,
                       steger_refined=int(final['steger_refined'].sum()), steger_fallback=int(final['steger_fallback'].sum()),
                       mean_ms=float(np.mean(times)), p95_ms=float(np.percentile(times,95)), max_ms=max(times),
                       deadline_misses=misses, timing_ms=times,
                       error_by_column=[float(e) if np.isfinite(e) else None for e in errors])
            row['accuracy_met'] = rmse is not None and rmse <= args.rmse_limit
            row['coverage_met'] = coverage >= args.min_coverage
            row['deadline_met'] = misses == 0
            row['development_pass'] = row['accuracy_met'] and row['coverage_met'] and row['deadline_met']
            row['input_image'] = str(image_file.resolve()) if image_file else None
            row['saturated_pixels'] = int(np.count_nonzero(scenario.image == 255))
            rows.append(row)
            print(f"seed={seed} {scenario.name}: RMSE={rmse:.5f}px valid={n}/{errors.size} mean={row['mean_ms']:.1f}ms max={row['max_ms']:.1f}ms joint={'PASS' if row['development_pass'] else 'FAIL'}", flush=True)
    report = dict(source=str(source), source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((source/'centerline').glob('*.py'))},
                  protocol_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT/'scripts/realtime_scenarios.py', ROOT/'scripts/line_grayscale_centroid_demo.py', ROOT/'scripts/single_factor_scenarios.py']},
                  isolated_scenarios=['curved','large_curvature','weak_light','noise','reflection','saturation','double_peak','broken'],
                  single_factor_parameters=PARAMETERS,
                  input_note='All eight factors share one straight control, background and normal width. Same seed pairs the noise field; only the noise scene increases its strength. No cross-input algorithm improvement claim.',
                  python=platform.python_version(), numpy=np.__version__, machine=platform.machine(), processor=platform.processor(),
                  seeds=args.seeds, warmup=args.warmup, repeats=args.repeats, config=asdict(config),
                  baseline_steger_config=asdict(StegerConfig()) if not modern else None,
                  frame_budget_ms=config.frame_budget_ms, rmse_limit_px=args.rmse_limit, min_coverage=args.min_coverage,
                  metric='vertical residual projected onto interpolated truth normal, evaluated at estimated x; not exact shortest curve distance',
                  scope='512x512 synthetic line/curve images; excludes acquisition, polar unwrap, calibration, bore fitting and report; no physical acceptance claim',
                  development_pass=all(r['development_pass'] for r in rows), rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Report: {args.output.resolve()}', flush=True)
    if args.require_pass and not report['development_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
