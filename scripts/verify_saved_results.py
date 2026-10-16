"""Verify final files, both image sets and stored point-to-metric consistency."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def read_json(path): return json.loads(path.read_text(encoding='utf-8'))
def main():
    line=ROOT/'result/v5_benchmark'; ring=ROOT/'result/v5_ring_measurement'
    frozen=read_json(line/'final_manifest.json')
    for row in frozen['files']:
        path=ROOT/row['path']
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path']
    old=read_json(line/'current_summary.json')
    for name,digest in old['source_sha256'].items():
        assert hashlib.sha256((ROOT/'src/centerline'/name).read_bytes()).hexdigest()==digest,name
    for folder in [line,ring]:
        manifest=read_json(folder/'image_manifest.json')
        assert len(manifest['rows'])==45
        with np.load(folder/'ground_truth.npz') as truths:
            for row in manifest['rows']:
                pixels=np.asarray(Image.open(ROOT/row['image']))
                assert hashlib.sha256(pixels.tobytes()).hexdigest()==row['image_sha256']
                assert row['truth_key'] in truths.files
    report=read_json(ring/'summary.json')
    for relative,digest in report['source_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    for row in report['rows']:
        path=ring/'points'/f"{row['scenario']}_seed{row['seed']}.npz"
        with np.load(path) as points:
            valid=points['valid']
            assert len(valid)==row['angular_samples'] and int(valid.sum())==row['valid_points']
            assert np.isnan(points['points_px'][~valid]).all()
            assert np.isfinite(points['points_px'][valid]).all()
            measured=float(np.sqrt(np.mean(points['normal_error_px'][valid]**2)))
            assert abs(measured-row['normal_rmse_px'])<1e-12
            for units in ['px','mm']:
                assert not points['robust_circle_inlier_'+units][~valid].any()
    print(f"Verified {len(frozen['files'])} file hashes, unchanged V5 core, 90 dataset images and 45 saved ring point/RMSE records.")

if __name__=='__main__': main()
