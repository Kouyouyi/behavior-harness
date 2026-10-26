"""Third-person H.264 video from saved native-SONIC/MuJoCo physical states."""
import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np


def rows(path):
    with path.open() as file:return [json.loads(line) for line in file if line.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--fps',type=int,default=30)
    a=p.parse_args();run=a.run.resolve();out=run/'third-person';out.mkdir(exist_ok=False)
    if not 1<=a.fps<=60:raise ValueError('fps must be 1..60')
    events=rows(run/'events.jsonl');samples=rows(run/'samples.jsonl')
    start=next(e['monotonic'] for e in events if e['event']=='locomotion_started')
    scene=json.loads((run/'scene/scene.json').read_text())
    records=[r for r in rows(run/'scene/trajectory.jsonl') if r['monotonic']>=start-.5]
    scene_path=Path(scene.get('scene_xml',scene.get('scene')))
    if not scene_path.is_absolute():scene_path=Path('/home/user/GR00T-WholeBodyControl')/scene_path
    model=mujoco.MjModel.from_xml_path(str(scene_path));data=mujoco.MjData(model)
    model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
    camera=mujoco.MjvCamera();camera.distance=2.8;camera.azimuth=135;camera.elevation=-18
    camera.lookat[:]=[0,0,.8]
    output=out/'locomotion-upper-hold.mp4';index=sample_index=event_index=0;frames=0;captures=[]
    direction='standing';seen=set()
    with mujoco.Renderer(model,height=720,width=1280) as renderer, imageio.get_writer(
            output,fps=a.fps,codec='libx264',ffmpeg_log_level='error',quality=8) as writer:
        for t in np.arange(records[0]['monotonic'],records[-1]['monotonic'],1/a.fps):
            while index+1<len(records) and records[index+1]['monotonic']<=t:index+=1
            while event_index+1<len(events) and events[event_index+1]['monotonic']<=t:event_index+=1
            while sample_index+1<len(samples) and start+samples[sample_index+1]['elapsed_s']<=t:sample_index+=1
            row=records[index];sample=samples[sample_index]
            # Last move_started persists through its settling/holding phases.
            started=[e for e in events[:event_index+1] if e['event']=='move_started']
            if started:direction=started[-1]['direction']
            data.qpos[:]=row['qpos'];data.qvel[:]=row['qvel'];mujoco.mj_forward(model,data)
            camera.lookat[:]=[row['root_qpos'][0],row['root_qpos'][1],.8]
            renderer.update_scene(data,camera=camera);frame=renderer.render().copy()
            cv2.rectangle(frame,(0,0),(1280,130),(20,25,33),-1)
            lines=[f'{run.name} | {direction} | {sample.get("move_phase","idle")} | t={t-start:.1f}s',
                   'Recorded MuJoCo physics | waist + arms reference held | normal gravity / contacts',
                   f'Upper actual max error {np.rad2deg(sample["upper_measured_error_rad"]):.2f} deg | encoder error {sample["upper_encoder_error_rad"]:.2g} rad',
                   'Third-person replay | stock Dex3, no carried payload | NOT hardware validation']
            for i,line in enumerate(lines):
                cv2.putText(frame,line,(18,28+i*28),cv2.FONT_HERSHEY_SIMPLEX,.63,(245,245,245),1,cv2.LINE_AA)
            writer.append_data(frame);frames+=1
            if direction not in seen and sample.get('move_phase')=='running' and started and t-started[-1]['monotonic']>1.2:
                captures.append(frame.copy());seen.add(direction)
                cv2.imwrite(str(out/(direction+'.jpg')),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    cap=cv2.VideoCapture(str(output));decoded=0
    while cap.read()[0]:decoded+=1
    cap.release()
    if decoded!=frames:raise RuntimeError('Video full decode check failed')
    if captures:
        sheet=np.zeros((((len(captures)+2)//3)*360,1920,3),np.uint8)
        for i,frame in enumerate(captures):sheet[(i//3)*360:(i//3+1)*360,(i%3)*640:(i%3+1)*640]=cv2.resize(frame,(640,360))
        cv2.imwrite(str(out/'contact-sheet.jpg'),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
    metadata=dict(video=str(output),fps=a.fps,frames=frames,decoded_frames=decoded,duration_s=frames/a.fps,
                  source=str(run/'scene/trajectory.jsonl'),render_advances_physics=False,
                  camera=dict(azimuth=135,elevation=-18,distance=2.8,tracking='root xy'),scene=scene)
    (out/'video.json').write_text(json.dumps(metadata,indent=2)+'\n');print(output)


if __name__=='__main__':main()
