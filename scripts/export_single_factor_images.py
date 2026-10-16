"""Export the active nine scenarios and their paired-noise audit without timing."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from realtime_scenarios import build_scenarios
from single_factor_scenarios import PARAMETERS

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'normal_straight':'普通直线（基准）','curved':'曲线：只改变形状',
         'large_curvature':'大曲率：只改变形状','weak_light':'弱光：只降低条纹幅度',
         'noise':'强噪声：只提高噪声强度','reflection':'反光：只增加局部亮斑',
         'saturation':'饱和：只提高幅度至截断','double_peak':'单侧干扰：副条纹 15%',
         'broken':'断裂：只移除局部条纹'}

def main():
    destination = ROOT/'result/v5_benchmark'
    images = destination/'test_images'
    images.mkdir(parents=True, exist_ok=True)
    manifest_path = destination/'image_manifest.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
    previous_hashes = {(r['scenario'],r['seed']):r['image_sha256'] for r in previous['rows']} if previous else {}
    truths = {s.name:s.truth_y for s in build_scenarios(7)}
    truth_path = destination/'ground_truth.npz'
    np.savez_compressed(truth_path,**truths)
    with np.load(truth_path) as saved_truths:
        assert all(np.array_equal(saved_truths[name],truth) for name,truth in truths.items())
    rows = []
    seeds = [7,37,67,97,127]
    for seed in seeds:
        scenarios = build_scenarios(seed)
        control = scenarios[0]
        for scene in scenarios:
            path = images/f'{scene.name}_seed{seed}.png'
            Image.fromarray(scene.image).save(path)
            restored = np.asarray(Image.open(path))
            assert np.array_equal(scene.image,restored)
            digest = hashlib.sha256(restored.tobytes()).hexdigest()
            if (scene.name,seed) in previous_hashes:
                assert digest == previous_hashes[scene.name,seed], 'Saved dataset changed'
            assert np.array_equal(scene.truth_y,truths[scene.name])
            saturated = int(np.count_nonzero(restored==255))
            if scene.name != 'saturation':
                assert saturated == 0, (seed,scene.name,saturated)
            if scene.name not in {'curved','large_curvature'}:
                assert np.array_equal(scene.truth_y,control.truth_y)
            if scene.name != 'noise':
                assert np.array_equal(scene.image[:60],control.image[:60])
            rows.append(dict(scenario=scene.name,seed=seed,image=str(path.relative_to(ROOT)),
                             image_sha256=digest,
                             truth_file=str(truth_path.relative_to(ROOT)),truth_key=scene.name,
                             saturated_pixels=saturated,factor=NAMES[scene.name],
                             noise_sigma=18.0 if scene.name=='noise' else 5.0))
    manifest = dict(seeds=seeds,parameters=PARAMETERS,rows=rows,
                    truth_format='NPZ: scenario keys, arrays of 512 true center y values indexed by integer x; geometry is identical across noise seeds',
                    audit_passed=True,note='One control and eight isolated factors. Base Gaussian noise remains in every scene; the noise factor changes its strength. Uint8 clipping/quantization is shared. Audit is image validation, not 45-image algorithm testing.')
    (destination/'image_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    sheet = Image.new('RGB',(3*512,3*554),'#eeeeee')
    draw = ImageDraw.Draw(sheet)
    for i,scene in enumerate(build_scenarios(7)):
        x,y = (i%3)*512,(i//3)*554
        draw.text((x+12,y+8),NAMES[scene.name],font=font,fill='#222222')
        sheet.paste(Image.fromarray(scene.image).convert('RGB'),(x,y+42))
    sheet.save(destination/'single_factor_overview.png')
    print('45 active PNGs and nine ground-truth arrays exported and audited; existing dataset pixels preserved.')

if __name__ == '__main__':
    main()
