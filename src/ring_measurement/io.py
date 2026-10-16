"""Lossless points and readable summaries for both simulation and real images."""
from dataclasses import asdict,is_dataclass
from pathlib import Path
import csv,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def json_safe(value):
    if is_dataclass(value): return json_safe(asdict(value))
    if isinstance(value,np.ndarray): return json_safe(value.tolist())
    if isinstance(value,np.generic): return json_safe(value.item())
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value): return None
    return value

def compact_measurement(result):
    keep=['grid','metrics_px','metrics_mm','calibration','center_world_mm','section_normal_world',
          'coverage','valid_points','angular_samples','max_gap_deg','measurement_valid','quality_reasons',
          'steger_refined','timings_ms','frame_budget_ms','deadline_met']
    summary={key:result[key] for key in keep}
    for name in ['metrics_px','metrics_mm']:
        if summary[name] is not None:
            summary[name]={k:v for k,v in summary[name].items() if k!='fit_inliers'}
    return json_safe(summary)

def write_json(path,value):
    Path(path).write_text(json.dumps(json_safe(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def save_measurement(result,destination,input_image=None):
    folder=Path(destination); folder.mkdir(parents=True,exist_ok=True)
    values={key:result[key] for key in ['polar_points','points_px','undistorted_points_px','valid','points_mm','points_world_mm'] if result[key] is not None}
    values.update(fit_diagnostic_masks(result))
    np.savez_compressed(folder/'points.npz',**values)
    write_json(folder/'measurement.json',compact_measurement(result))
    Image.fromarray(np.clip(result['unwrapped'],0,255).astype(np.uint8)).save(folder/'unwrapped.png')
    with (folder/'points.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['sample_index','valid','angle_rad','radius_px','image_x_px','image_y_px','plane_x_mm','plane_y_mm','robust_circle_inlier_px','robust_circle_inlier_mm'])
        grid=result['grid']
        for i,(uv,xy,valid) in enumerate(zip(result['polar_points'],result['points_px'],result['valid'])):
            angle=grid.angle_start_rad+uv[0]*2*np.pi/grid.angular_samples if valid else ''
            radius=grid.radius_min+uv[1]*grid.radial_step if valid else ''
            physical=result['points_mm'][i] if result['points_mm'] is not None and valid else ['', '']
            flags=[int(values['robust_circle_inlier_px'][i]),int(values['robust_circle_inlier_mm'][i])]
            writer.writerow([i,int(valid),angle,radius,*(xy if valid else ['', '']),*physical,*flags])
    if input_image is not None:
        Image.fromarray(np.clip(input_image,0,255).astype(np.uint8)).save(folder/'input.png')
        make_overview(input_image,result,folder/'flow_overview.png')

def fit_diagnostic_masks(result):
    flags={}
    for units in ['px','mm']:
        mask=np.zeros(len(result['valid']),dtype=bool)
        metrics=result['metrics_'+units]
        if metrics is not None: mask[result['valid']]=metrics['fit_inliers']
        flags['robust_circle_inlier_'+units]=mask
    return flags

def make_overview(input_image,result,path):
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(font_path),20) if font_path.exists() else ImageFont.load_default()
    original=Image.fromarray(np.clip(input_image,0,255).astype(np.uint8)).convert('RGB')
    overlay=original.copy(); draw=ImageDraw.Draw(overlay)
    points=result['points_px'][result['valid']]
    for x,y in points:
        draw.ellipse((x-1,y-1,x+1,y+1),fill='#24ec84')
    canvas=Image.new('RGB',(1024,900),'#eeeeee'); d=ImageDraw.Draw(canvas)
    d.text((12,9),'原始环形图',font=font,fill='#222222')
    d.text((524,9),'反变换后的中心线（绿色）',font=font,fill='#222222')
    canvas.paste(original.resize((512,512)),(0,42)); canvas.paste(overlay.resize((512,512)),(512,42))
    d.text((12,565),'极坐标展开图：横轴角度，纵轴半径',font=font,fill='#222222')
    polar=Image.fromarray(np.clip(result['unwrapped'],0,255).astype(np.uint8)).convert('RGB').resize((1024,210))
    polar_draw=ImageDraw.Draw(polar)
    for uv in result['polar_points'][result['valid']]:
        x=uv[0]*1024/result['grid'].angular_samples; y=uv[1]*210/result['unwrapped'].shape[0]
        polar_draw.ellipse((x-1,y-1,x+1,y+1),fill='#24ec84')
    canvas.paste(polar,(0,604))
    diameter=result['metrics_mm']['diameter'] if result['metrics_mm'] is not None else None
    text=f"有效点 {result['valid_points']}/{result['angular_samples']}   覆盖率 {result['coverage']:.1%}"
    if diameter is not None: text+=f'   仿真内径 {diameter:.5f} mm' if result['calibration']['is_simulated'] else f'   内径 {diameter:.5f} mm'
    d.text((12,830),text,font=font,fill='#222222')
    d.text((12,862),'圆度采用最小二乘圆基准；缺失点不补为测量点',font=font,fill='#222222')
    canvas.save(path)
