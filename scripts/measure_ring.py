"""Process an external ring image; calibration JSON is optional for pixel mode."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ring_measurement.polar import PolarGrid
from ring_measurement.pipeline import RingMeasurementConfig,run_ring_measurement
from ring_measurement.calibration import calibration_from_dict
from ring_measurement.io import save_measurement

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--center',nargs=2,type=float,metavar=('CX','CY'))
    parser.add_argument('--radius-range',nargs=2,type=float,metavar=('MIN','MAX'))
    parser.add_argument('--angles',type=int,default=3600)
    parser.add_argument('--radial-step',type=float,default=1)
    parser.add_argument('--nominal-diameter-mm',type=float)
    parser.add_argument('--output',type=Path,default=ROOT/'result/v5_ring_measurement/external')
    args=parser.parse_args()
    if (args.center is None)!=(args.radius_range is None): parser.error('center and radius-range must be supplied together')
    calibration=calibration_from_dict(json.loads(args.calibration.read_text(encoding='utf-8'))) if args.calibration else None
    with Image.open(args.image) as source: image=np.asarray(source.convert('L' if source.mode in {'L','I','F','I;16'} else 'RGB'))
    grid=PolarGrid(tuple(args.center),*args.radius_range,args.angles,args.radial_step) if args.center else None
    config=RingMeasurementConfig(grid=grid,angular_samples=args.angles,radial_step=args.radial_step,nominal_diameter_mm=args.nominal_diameter_mm)
    result=run_ring_measurement(image,config,calibration)
    save_measurement(result,args.output,image)
    units='mm' if calibration else 'px'; metrics=result['metrics_mm'] if calibration else result['metrics_px']
    print(f"valid={result['valid_points']}/{result['angular_samples']} quality={result['measurement_valid']} total={result['timings_ms']['total_ms']:.1f}ms")
    if metrics: print(f"diameter={metrics['diameter']:.6f}{units} roundness_LSC={metrics['roundness_lsc']:.6f}{units}")
    print(f'Saved: {args.output.resolve()}')
    if not result['measurement_valid']: raise SystemExit(2)

if __name__=='__main__': main()
