"""Third-person + foot close-up of recorded SONIC/MuJoCo physics, with warning audio."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import wave

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCENE='/home/user/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output-name',default='review-final')
    p.add_argument('--demo-side',choices=['left','right'],help='For a recorded_token physics experiment')
    a=p.parse_args();run=a.run.resolve();out=run/a.output_name;out.mkdir(exist_ok=False)
    shutil.copy2(__file__,out/'g1_render_heel.py')
    result=json.loads((run/'result.json').read_text())
    events=[json.loads(x) for x in (run/'events.jsonl').read_text().splitlines()]
    demo=result['case']=='recorded_token'
    candidate=json.loads((run/'recorded_episode.json').read_text()).get('candidate') if demo else None
    if demo:
        if not a.demo_side:p.error('recorded_token rendering requires --demo-side')
        side=a.demo_side
        phase_events=[x for x in events if x['event']=='demo_phase']
        motion_start=phase_events[0]['monotonic']
        completed=next(x['monotonic'] for x in events if x['event']=='completed')
        details=dict(side=side,mode='recorded_token',phases=[dict(name=x['phase'],
            start_s=x['monotonic']-motion_start,
            end_s=(phase_events[i+1]['monotonic'] if i+1<len(phase_events) else completed)-motion_start)
            for i,x in enumerate(phase_events)])
    else:
        details=json.loads((run/'heel_plan.json').read_text());side=details['side']
        motion_start=next(x['monotonic'] for x in events if x['event']=='heel_started')
    audio_start=next(x['monotonic'] for x in events if x['event']==('demo_audio_started' if demo else 'heel_audio_started'))
    audio_end=next(x['monotonic'] for x in events if x['event']==('demo_audio_completed' if demo else 'heel_audio_completed'))
    audio_file='demo_warning.wav' if demo else 'ankle_warning.wav'
    if audio_end>motion_start:raise ValueError('Audio must finish before heel trial')
    start=audio_start-1.;finish=motion_start+details['phases'][-1]['end_s']+2
    rows=[]
    with (run/'scene/trajectory.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if start<=r['monotonic']<=finish:rows.append(r)
    times=np.array([r['monotonic'] for r in rows]);start=times[0]
    model=mujoco.MjModel.from_xml_path(SCENE);data=mujoco.MjData(model)
    model.mat_reflectance[:]=0  # Render only: avoid floor reflections obscuring heel clearance.
    model.vis.global_.offwidth=640;model.vis.global_.offheight=640
    renderer=mujoco.Renderer(model,height=640,width=640)
    full_option=mujoco.MjvOption();full_option.sitegroup[:]=0
    contact_option=mujoco.MjvOption();contact_option.sitegroup[:]=0;contact_option.geomgroup[1]=0
    cameras=[]
    for distance,lookat,elevation in [(2.45,[0,0,.78],-12),(.72,[.025,0,.12],-12)]:
        cam=mujoco.MjvCamera();cam.distance=distance;cam.lookat[:]=lookat
        cam.azimuth=-75 if side=='left' else 75;cam.elevation=elevation;cameras.append(cam)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    fps=25;keys=[audio_start+.5,motion_start+5.5,motion_start+10.5,motion_start+21];stills=[]
    silent=out/'silent.mp4'
    with imageio.get_writer(str(silent),fps=fps,codec='libx264',quality=8,ffmpeg_log_level='error') as writer:
        for t in np.arange(times[0],times[-1],1/fps):
            i=int(np.argmin(abs(times-t)));r=rows[i]
            data.qpos[:]=r['qpos'];data.qvel[:]=r['qvel'];mujoco.mj_forward(model,data)
            canvas=Image.new('RGB',(1280,720),(20,25,33))
            for j,cam in enumerate(cameras):
                renderer.update_scene(data,camera=cam,scene_option=full_option if j==0 else contact_option)
                if j==1:
                    body=data.body(side+'_ankle_roll_link');rot=body.xmat.reshape(3,3)
                    # Visual markers follow actual recorded foot geometry only.
                    for x,color in [(-.05,[1.,.55,.05,1.]),(.12,[.2,1.,.3,1.])]:
                        index=renderer.scene.ngeom
                        mujoco.mjv_initGeom(renderer.scene.geoms[index],mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([.004,.004,.004]),body.xpos+rot@np.array([x,.031 if side=='left' else -.031,-.035]),
                            np.eye(3).ravel(),np.array(color,dtype=np.float32))
                        renderer.scene.ngeom+=1
                canvas.paste(Image.fromarray(renderer.render()),(j*640,80))
            draw=ImageDraw.Draw(canvas)
            elapsed=t-motion_start
            phase=(('ANNOUNCING DELTA MOTION TEST' if candidate else 'ANNOUNCING RECORDED MOTION REPLAY' if demo else 'ANNOUNCING SIMULATED HIGH TEMPERATURE') if t<audio_end and t>=audio_start else
                   'waiting for announcement' if elapsed<0 else
                   next((x['name'] for x in details['phases'] if x['start_s']<=elapsed<x['end_s']),'settled'))
            draw.text((12,6),f'{side.upper()} ANKLE | {phase} | video {t-start:.1f}s',font=font,fill='white')
            h=r['ankle_telemetry'][side];other='right' if side=='left' else 'left'
            draw.text((12,35),f"heel {h['heel_min_z_m']*1000:.1f} mm | toe force {h['toe_contact_force_n']:.0f} N | "
                f"foot loads L/R {r['foot_ground_normal_force_n']['left']:.0f}/{r['foot_ground_normal_force_n']['right']:.0f} N",font=small,fill='white')
            condition=(f'CoMotion leg delta x{candidate["gain"]:g}; synthetic entry/return' if candidate else 'original Pico tokens; synthetic entry/return' if demo else 'WHOLE-FOOT UNLOAD: '+('PASS' if result['completed'] else 'FAIL')
                       if details.get('mode')=='whole_foot' else 'normal contacts, no support force')
            draw.text((12,58),f'RECORDED PHYSICS | {condition} | synthetic alert, no thermal model',font=small,fill=(230,200,120))
            draw.text((15,92),'Third-person view',font=small,fill='white')
            draw.text((655,92),'Contact geometry | orange: heel, green: toe',font=small,fill='white')
            writer.append_data(np.asarray(canvas))
            if len(stills)<len(keys) and t>=keys[len(stills)]:stills.append(canvas.copy())
    renderer.close()
    offset=audio_start-start
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(silent),'-i',str(run/audio_file),
        '-filter_complex',f'[1:a]adelay={round(offset*1000)}:all=1,apad[a]',
        '-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac','-shortest','-movflags','+faststart',
        str(out/'announced_heel.mp4')],check=True)
    sheet=Image.new('RGB',(1280,720*len(stills)))
    for i,frame in enumerate(stills):sheet.paste(frame,(0,i*720))
    sheet.save(out/'contact_sheet.jpg',quality=90)
    with wave.open(str(run/audio_file)) as wav:duration=wav.getnframes()/wav.getframerate()
    (out/'manifest.json').write_text(json.dumps(dict(physics_advanced=False,source='scene/trajectory.jsonl',
        audio_source=audio_file,audio_offset_s=offset,audio_duration_s=duration,
        motion_trial_offset_s=motion_start-start,heel_rise_start_s=None if demo else motion_start-start+6,
        audio_playback_completed_before_motion=True,synthetic_alert=True,fps=fps,
        view='third person plus simultaneous foot collision geometry close-up',side=side,
        visual_mesh_hidden_in_closeup=True,physics_collision_unchanged=True),indent=2)+'\n')
    silent.unlink();print(out/'announced_heel.mp4')


if __name__=='__main__':main()
