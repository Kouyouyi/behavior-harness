"""Render archived simulated states; rendering does not advance physics."""
import argparse
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output-name', default='review')
    a = p.parse_args()
    run = a.run.resolve()
    out = run/a.output_name; out.mkdir(exist_ok=False)
    manifest = json.loads((run/'manifest.json').read_text())
    contact_label = ('DIAGNOSTIC: upper-limb collisions disabled' if manifest.get('arm_collision_ablation') else
                     'DIAGNOSTIC: finger collisions disabled' if manifest.get('hand_collision_ablation') else
                     'stock G1/Dex3 contacts')
    shutil.copy2(__file__, out/'g1_render_sim.py')
    events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    locomotion=manifest.get('case') in ('gait','walk')
    start = next(e['monotonic'] for e in events if e['event'] in ('gait_started','walk_started') if locomotion) if locomotion else next(e['monotonic'] for e in events if e['event']=='job_started')
    reach = manifest.get('case') == 'reach'
    finish = next((e['monotonic'] for e in events if e['event'] == 'completed'), float('inf')) if reach or locomotion else start+17
    records = []
    with (run/'scene/trajectory.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if start-1 <= r['monotonic'] <= finish:
                records.append(r)
    times = np.array([r['monotonic']-start for r in records])
    scene = Path('/home/user/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml')
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    model.vis.global_.offwidth = 960; model.vis.global_.offheight = 640
    renderer = mujoco.Renderer(model, height=640, width=960)
    camera = mujoco.MjvCamera(); camera.distance=2.35; camera.azimuth=135; camera.elevation=-12
    camera.lookat[:] = [0, 0, .85]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    frames = []
    key_times = [-.5, 3.5, 7., 16.]
    if locomotion:key_times=[-.5,3.,7.4,float(times[-1]-.1)]
    if reach:
        key_times = [-.5]+[min(e['monotonic']-start-.1, times[-1]-.1) for e in events if e['event'] == 'stage_settled']
        key_times = key_times[:4]
    fps = 20
    with imageio.get_writer(str(out/('locomotion.mp4' if locomotion else 'roundtrip.mp4')), fps=fps, codec='libx264', ffmpeg_log_level='error') as writer:
        for t in np.arange(times[0], times[-1], 1/fps):
            i = int(np.argmin(abs(times-t))); row = records[i]
            data.qpos[:] = row['qpos']; data.qvel[:] = row['qvel']; mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render()); draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 960, 60), fill=(20, 25, 33))
            draw.text((12, 5), f'{run.name} | RECORDED MUJOCO PHYSICS', font=font, fill='white')
            phase = next((e.get('phase', '') for e in reversed(events)
                          if e['event'] == 'stage_started' and e['monotonic'] <= start+t), '') if reach else ''
            draw.text((12, 32), f't={times[i]:.2f}s {phase} | {contact_label}', font=font, fill='white')
            writer.append_data(np.asarray(frame))
            if len(frames) < len(key_times) and t >= key_times[len(frames)]:
                frames.append(frame.copy())
    renderer.close()
    sheet = Image.new('RGB', (960*2, 640*2))
    for i, frame in enumerate(frames):
        sheet.paste(frame, ((i % 2)*960, (i // 2)*640))
    sheet.save(out/'contact_sheet.jpg', quality=90)
    (out/'manifest.json').write_text(json.dumps(dict(physics_advanced=False, frame_rate=fps,
        source='scene/trajectory.jsonl', time_axis='host monotonic time relative to test; nearest recorded state',
        contact_condition=contact_label,
        camera_azimuth=135, keyframe_requested_times_s=key_times), indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
