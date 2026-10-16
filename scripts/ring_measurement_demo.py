"""Reproducible 3600-direction ring and 6 m axial simulation verification."""
import argparse,hashlib,time,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ring_measurement.pipeline import RingMeasurementConfig,run_ring_measurement
from ring_measurement.geometry import contour_metrics
from ring_measurement.sweep import analyze_sweep
from ring_measurement.io import compact_measurement,save_measurement,write_json,fit_diagnostic_masks
from ring_scenarios import CENTER,RADIUS,NAMES,simulation_calibration,build_ring_scenarios,radius_truth

def run_axial_simulation(folder,config,calibration):
    y,x=np.mgrid[:512,:512]
    noise=np.random.default_rng(7).normal(0,5,(512,512))
    sections=[]; clouds=[]
    for z in [0,3000,6000]:
        t=z/6000; cx=CENTER[0]+.2*t; cy=CENTER[1]
        radius=RADIUS+.01*t
        image=np.clip(20+160*np.exp(-.5*((np.hypot(x-cx,y-cy)-radius)/2.6)**2)+noise,0,255).astype(np.uint8)
        result=run_ring_measurement(image,config,calibration)
        Image.fromarray(image).save(folder/'test_images'/f'sweep_z{z}.png')
        summary=compact_measurement(result); summary['position_mm']=z; sections.append(summary)
        world=result['points_world_mm'][result['valid']].copy(); world[:,2]+=z
        clouds.append(world)
    sweep=analyze_sweep(sections,reference_origin_mm=[0,0,400],reference_direction=[0,0,1],nominal_diameter_mm=RADIUS)
    sweep['truth']=dict(taper_mm_over_span=.01,coaxiality_diameter_mm=.2,straightness_ls_mm=0,
                        description='Known fixed fixture axis; no temperature or mechanical compensation is simulated.')
    sweep['sections']=sections
    write_json(folder/'sweep_summary.json',sweep)
    np.savez_compressed(folder/'wall_cloud.npz',points_world_mm=np.vstack(clouds),position_mm=np.array([0,3000,6000]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,37,67,97,127])
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--angles',type=int,default=3600)
    parser.add_argument('--output',type=Path,default=ROOT/'result/v5_ring_measurement')
    args=parser.parse_args()
    if args.repeats<1 or args.warmup<0: parser.error('invalid timing repetitions')
    folder=args.output; folder.mkdir(parents=True,exist_ok=True)
    for name in ['test_images','points']: (folder/name).mkdir(exist_ok=True)
    calibration=simulation_calibration(); write_json(folder/'simulation_calibration.json',calibration.to_dict())
    config=RingMeasurementConfig(angular_samples=args.angles,nominal_diameter_mm=RADIUS)
    angles=np.arange(args.angles)*2*np.pi/args.angles
    truths={}; truth_metrics={}
    for name in NAMES:
        radius,_=radius_truth(name,angles)
        xy=np.column_stack([CENTER[0]+radius*np.cos(angles),CENTER[1]+radius*np.sin(angles)])
        physical,_=calibration.to_plane(xy)
        truths[name]=xy; truth_metrics[name]=contour_metrics(physical,nominal_diameter=RADIUS)
    np.savez_compressed(folder/'ground_truth.npz',angles_rad=angles,**truths)
    rows=[]; datasets=[]; selected_examples={}
    for seed in args.seeds:
        for scenario in build_ring_scenarios(seed):
            name,image=scenario['name'],scenario['image']
            image_path=folder/'test_images'/f'{name}_seed{seed}.png'
            Image.fromarray(image).save(image_path)
            pixel_hash=hashlib.sha256(image.tobytes()).hexdigest()
            assert np.asarray(Image.open(image_path)).tobytes()==image.tobytes()
            if name!='saturation': assert not np.any(image==255)
            datasets.append(dict(scenario=name,seed=seed,image=str(image_path.relative_to(ROOT)),image_sha256=pixel_hash,
                                 truth_file=str((folder/'ground_truth.npz').relative_to(ROOT)),truth_key=name))
            for _ in range(args.warmup): run_ring_measurement(image,config,calibration)
            times=[]
            for _ in range(args.repeats):
                start=time.perf_counter(); result=run_ring_measurement(image,config,calibration)
                times.append((time.perf_counter()-start)*1000)
            xy=result['points_px']; valid=result['valid']
            delta=xy-np.asarray(CENTER)
            theta=np.mod(np.arctan2(delta[:,1],delta[:,0]),2*np.pi)
            expected,derivative=radius_truth(name,theta)
            radial_error=np.linalg.norm(delta,axis=1)-expected
            normal_error=radial_error/np.sqrt(1+(derivative/np.maximum(expected,1))**2)
            normal_rmse=float(np.sqrt(np.mean(normal_error[valid]**2))) if valid.any() else None
            metrics=result['metrics_mm']; truth=truth_metrics[name]
            diameter_error=float(metrics['diameter']-truth['diameter']) if metrics else None
            roundness_error=float(metrics['roundness_lsc']-truth['roundness_lsc']) if metrics else None
            row=compact_measurement(result)
            row.update(scenario=name,seed=seed,image_sha256=pixel_hash,input_image=str(image_path.relative_to(ROOT)),
                       normal_rmse_px=normal_rmse,radial_rmse_px=float(np.sqrt(np.mean(radial_error[valid]**2))) if valid.any() else None,
                       max_abs_normal_px=float(np.max(np.abs(normal_error[valid]))) if valid.any() else None,
                       truth_diameter_mm=truth['diameter'],diameter_error_mm=diameter_error,
                       truth_roundness_lsc_mm=truth['roundness_lsc'],roundness_error_mm=roundness_error,
                       mean_ms=float(np.mean(times)),max_ms=float(np.max(times)),timing_ms=times,
                       pixel_accuracy_met=normal_rmse is not None and normal_rmse<=.1,
                       diameter_accuracy_met=diameter_error is not None and abs(diameter_error)<=.01,
                       roundness_accuracy_met=roundness_error is not None and abs(roundness_error)<=.005,
                       full_chain_deadline_met=max(times)<=result['frame_budget_ms'])
            row['development_pass']=row['pixel_accuracy_met'] and result['measurement_valid'] and row['full_chain_deadline_met']
            rows.append(row)
            np.savez_compressed(folder/'points'/f'{name}_seed{seed}.npz',
                                points_px=xy,undistorted_points_px=result['undistorted_points_px'],
                                polar_points=result['polar_points'],points_mm=result['points_mm'],
                                points_world_mm=result['points_world_mm'],valid=valid,normal_error_px=normal_error,**fit_diagnostic_masks(result))
            if seed==args.seeds[0]: selected_examples[name]=(image,result)
            print(f"seed={seed} {name}: RMSE={normal_rmse:.5f}px diameter_error={diameter_error:.5f}mm roundness_error={roundness_error:.5f}mm coverage={result['coverage']:.1%} max={max(times):.1f}ms",flush=True)
    baseline=[r for r in rows if r['scenario']=='circle']
    repetition=float(np.ptp([r['metrics_mm']['diameter'] for r in baseline]))
    sources=list((ROOT/'src/centerline').glob('*.py'))+list((ROOT/'src/ring_measurement').glob('*.py'))
    sources += [Path(__file__),ROOT/'scripts/ring_scenarios.py']
    report=dict(version='V5 annular integration',seeds=args.seeds,repeats=args.repeats,warmup=args.warmup,
                config=asdict(config),calibration=calibration.to_dict(),rows=rows,
                source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                circle_diameter_repeatability_range_mm=repetition,repeatability_met=repetition<=.005,
                metric='Original-image radial residual projected on the known ring normal; exactly radial error for circle, local normal projection for ellipse/lobed profiles.',
                timing_scope='Outer software call includes localization, polar sampling, V5, inverse coordinates, physical calibration and geometry. Excludes acquisition, generating images and saving outputs.',
                roundness_method='All observed points relative to the geometric least-squares circle; robust fit is separately marked diagnostic.',
                physical_acceptance_claim=False,
                limitation='Synthetic known calibration, 512x512 sensor, 0.5 mm/px. Software flow verification is not system acceptance; 3600 angular samples are correlated image samples, not 3600 independent sensor pixels.')
    write_json(folder/'summary.json',report)
    write_json(folder/'image_manifest.json',dict(rows=datasets,center_px=CENTER,base_radius_px=RADIUS,
                  base_background=20,base_amplitude=160,base_normal_sigma_px=2.6,base_noise_sigma=5,noise_scene_sigma=18,
                  weak_amplitude=67.2,secondary_amplitude=24,secondary_offset_px=8,secondary_sigma_px=2,saturation_amplitude=280,
                  truth_note='circle exact signed radial distance; ellipse and lobed render via first-order normal distance of the analytic radial level set. Truth centers are analytic curves, independent of algorithm.',
                  single_factor_note='Each sensor-image scene changes only the named shape, amplitude, noise strength, glare, secondary stripe or signal-gap factor; no non-saturation image contains white saturation.'))
    if 'circle' in selected_examples:
        image,result=selected_examples['circle']; save_measurement(result,folder/'example',image)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    sheet=Image.new('RGB',(1536,1662),'#eeeeee'); draw=ImageDraw.Draw(sheet)
    labels=['圆环基准','椭圆','大曲率','弱光','强噪声','反光','饱和','单侧干扰 15%','断裂']
    for i,name in enumerate(NAMES):
        image,result=selected_examples[name]; x,y=(i%3)*512,(i//3)*554
        draw.text((x+12,y+9),labels[i],font=font,fill='#222222'); sheet.paste(Image.fromarray(image).convert('RGB'),(x,y+42))
    sheet.save(folder/'dataset_overview.png')
    run_axial_simulation(folder,config,calibration)
    print(f"Saved complete ring validation: {folder.resolve()}; {len(rows)} images, repeatability range={repetition:.6f}mm",flush=True)

if __name__=='__main__': main()
