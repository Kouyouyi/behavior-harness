"""Offline review of measured Pico demonstrations. No control/network imports.

Export with the data-collection Python; review with the teleop Python.
MuJoCo is used only for forward kinematics/rendering, never mj_step.
Root translation and ground contact were not recorded: floor placement in the
video is inferred from the lowest sole corner and is not measured clearance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

SCENE = Path('/home/user/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def export(dataset, output, episodes):
    import pyarrow.parquet as pq
    output.mkdir(parents=True, exist_ok=False)
    info = json.loads((dataset/'meta/info.json').read_text())
    for ep in episodes:
        source = dataset/info['data_path'].format(episode_index=ep, episode_chunk=ep//info['chunks_size'])
        arrays = {k:np.asarray(v) for k,v in pq.read_table(source).to_pydict().items()}
        with (output/f'episode_{ep:06d}.npz').open('xb') as f:
            np.savez_compressed(f, **arrays)
        dump(output/f'episode_{ep:06d}.json', dict(episode=ep, source=str(source.resolve()),
             sha256=sha(source), frames=len(arrays['timestamp']), fps=info['fps'],
             joint_names=info['features']['observation.state']['names'], features=list(arrays),
             thermal_fields=[k for k in arrays if 'temp' in k.lower()],
             source_info_sha256=sha(dataset/'meta/info.json'), script_config=info['script_config']))
    shutil.copy2(__file__, output/Path(__file__).name)


def review(source, render):
    import mujoco
    from scipy.spatial.transform import Rotation
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    meta = json.loads(source.with_suffix('.json').read_text())
    if sha(meta['source']) != meta['sha256']:
        raise ValueError('Source parquet changed')
    with np.load(source, allow_pickle=False) as f:
        a = {k:f[k] for k in f.files}
    out = source.parent/(source.stem+'_review'); out.mkdir(exist_ok=False)
    shutil.copy2(__file__, out/Path(__file__).name)
    q = a['observation.state']; quat = a['observation.root_orientation']
    t = a['timestamp'].reshape(-1); n = len(t)
    if q.shape != (n,29) or quat.shape != (n,4) or not np.all(np.diff(t)>0):
        raise ValueError('Invalid measured state shape/time')
    if not np.all(a['observation.body_valid']==1) or not np.all(a['teleop.stream_mode']==1):
        raise ValueError('Invalid body samples or non-SMPL source')
    for k,v in a.items():
        if v.dtype.kind == 'f' and not np.isfinite(v).all():
            raise ValueError('Nonfinite '+k)
    norms = np.linalg.norm(quat,axis=1)
    if np.max(abs(norms-1))>.01:
        raise ValueError('Invalid measured IMU quaternion')
    quat = quat/norms[:,None]
    model=mujoco.MjModel.from_xml_path(str(SCENE)); data=mujoco.MjData(model)
    qids=model.jnt_qposadr[[model.joint(name).id for name in meta['joint_names']]]
    # Remove constant initial yaw for readable views; retain measured tilt/yaw changes.
    rotations=Rotation.from_quat(quat[:,[1,2,3,0]])
    yaw0=rotations[0].as_euler('xyz')[2]
    display_quat=(Rotation.from_euler('z',-yaw0)*rotations).as_quat()[:,[3,0,1,2]]
    corners=np.array([[x,y,-.035] for x in (-.05,.12) for y in (-.03,.03)])
    feet={s:[] for s in ('left','right')}; ankle={s:[] for s in feet}; qpos=[]
    for i in range(n):
        data.qpos[:3]=[0,0,0]; data.qpos[3:7]=display_quat[i]; data.qpos[qids]=q[i]
        mujoco.mj_forward(model,data)
        for s in feet:
            b=data.body(s+'_ankle_roll_link')
            feet[s].append(b.xpos+corners@b.xmat.reshape(3,3).T)
            ankle[s].append(b.xpos.copy())
        # Visualization assumption only: whichever sole corner is lowest is at z=0.
        data.qpos[2]=-min(np.min(feet[s][-1][:,2]) for s in feet)
        qpos.append(data.qpos.copy())
    feet={s:np.asarray(v) for s,v in feet.items()}
    ankle={s:np.asarray(v) for s,v in ankle.items()}
    sole={s:feet[s][:,:,2].min(axis=1) for s in feet}
    heeltoe={s:feet[s][:,:2,2].min(axis=1)-feet[s][:,2:,2].min(axis=1) for s in feet}
    height_difference=sole['left']-sole['right']
    tilt=np.arccos(np.clip(1-2*(quat[:,1]**2+quat[:,2]**2),-1,1))
    features={}
    baseline=t<=.2
    for s in feet:
        other='right' if s=='left' else 'left'
        relative=sole[s]-sole[other]
        peak=int(np.argmax(relative))
        idx=[meta['joint_names'].index(s+'_'+x+'_joint') for x in
             ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        features[s]=dict(peak_sole_above_other_m=float(relative[peak]),peak_sole_time_s=float(t[peak]),
            start_sole_difference_m=float(np.median(relative[baseline])),
            end_sole_difference_m=float(np.median(relative[t>=t[-1]-.2])),
            max_heel_minus_toe_m=float(heeltoe[s].max()),
            min_heel_minus_toe_m=float(heeltoe[s].min()),
            joint_min_rad=q[:,idx].min(axis=0).tolist(),joint_max_rad=q[:,idx].max(axis=0).tolist(),
            joint_order=[meta['joint_names'][j] for j in idx])
    body_ns=a['observation.body_monotonic_ns'].reshape(-1)
    report=dict(episode=meta['episode'],source=meta['source'],source_sha256=meta['sha256'],
        fixture_sha256=sha(source),frames=n,nominal_duration_s=float(t[-1]-t[0]),
        body_clock_span_s=float((body_ns[-1]-body_ns[0])/1e9),
        unique_body_cycles=int(len(np.unique(a['observation.control_cycle_index']))),
        unique_smpl_frames=int(len(np.unique(a['teleop.smpl_frame_index']))),
        max_body_source_age_s=float(np.max(a['observation.body_source_age_s'])),
        max_body_capture_age_s=float(np.max(a['observation.body_capture_age_s'])),
        max_joint_speed_rad_s=float(np.max(abs(a['observation.joint_velocity']))),
        max_tilt_rad=float(tilt.max()),sides=features,
        start_to_end_joint_max_change_rad=float(np.max(abs(np.median(q[-10:],axis=0)-np.median(q[:10],axis=0)))),
        thermal_fields=meta['thermal_fields'],hardware_commands=False,physics_advanced=False,
        scene=str(SCENE),scene_xml_sha256=sha(SCENE),
        limitation='FK from measured joints and base IMU; no measured root translation, foot force, temperature or floor clearance. Stock feet/Dex3 visuals; BrainCo not reconstructed.',
        replay_source='Preserve recorded action.motion_token or original SMPL inputs; action.wbc is queued low-level motor target, not a SONIC reference.')
    dump(out/'analysis.json',report)
    np.savez_compressed(out/'kinematics.npz',timestamp=t,qpos=np.asarray(qpos),
                        sole_left_minus_right=height_difference,
                        left_heel_minus_toe=heeltoe['left'],right_heel_minus_toe=heeltoe['right'])
    fig,axes=plt.subplots(4,1,figsize=(10,9),sharex=True)
    axes[0].plot(t,1000*height_difference,label='Left sole - right sole');axes[0].axhline(0,color='grey',lw=.7)
    axes[0].set_ylabel('FK difference (mm)');axes[0].legend()
    for s,color in [('left','tab:blue'),('right','tab:orange')]:
        axes[1].plot(t,1000*heeltoe[s],label=s,color=color)
        for part,style in [('knee','-'),('ankle_pitch','--'),('hip_roll',':')]:
            j=meta['joint_names'].index(s+'_'+part+'_joint')
            axes[2].plot(t,q[:,j],style,label=s+' '+part,color=color)
    axes[1].set_ylabel('Heel - toe (mm)');axes[1].legend()
    axes[2].set_ylabel('Measured q (rad)');axes[2].legend(ncol=3,fontsize=8)
    axes[3].plot(t,np.degrees(tilt),label='Base IMU tilt');axes[3].set_ylabel('Tilt (deg)');axes[3].legend()
    axes[3].set_xlabel('Recorded dataset time (s)')
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle(f'Episode {meta["episode"]}: measured-state kinematics (not contact/thermal measurement)')
    fig.tight_layout();fig.savefig(out/'motion_analysis.png',dpi=160);plt.close(fig)
    if render:
        render_video(model,data,np.asarray(qpos),t,report,out,height_difference,heeltoe)
    print(json.dumps(report,indent=2))


def render_video(model,data,qpos,t,report,out,difference,heeltoe):
    import mujoco
    from PIL import Image,ImageDraw,ImageFont
    model.mat_reflectance[:]=0
    model.vis.global_.offwidth=640;model.vis.global_.offheight=640
    renderer=mujoco.Renderer(model,height=640,width=640)
    option=mujoco.MjvOption();option.sitegroup[:]=0
    cams=[]
    for azi in (-75,75):
        c=mujoco.MjvCamera();c.distance=2.3;c.lookat[:]=[0,0,.76];c.azimuth=azi;c.elevation=-13;cams.append(c)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    fps=50;video=out/'measured_state_reconstruction.mp4'
    cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),
         '-i','pipe:0','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    frames=[];keys={0,len(t)-1,int(np.argmax(difference)),int(np.argmin(difference))}
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
        try:
            for i in range(len(t)):
                data.qpos[:]=qpos[i];data.qvel[:]=0;mujoco.mj_forward(model,data)
                canvas=Image.new('RGB',(1280,720),(22,26,33));draw=ImageDraw.Draw(canvas)
                for j,c in enumerate(cams):
                    renderer.update_scene(data,camera=c,scene_option=option)
                    canvas.paste(Image.fromarray(renderer.render()),(j*640,80))
                draw.text((12,6),f'Episode {report["episode"]} | RECORDED JOINTS + IMU | t={t[i]:.2f}s',font=font,fill='white')
                draw.text((12,31),'KINEMATIC RECONSTRUCTION: inferred floor/root height; no dynamics or measured contact',font=font,fill=(255,210,110))
                draw.text((12,56),f'FK sole L-R {difference[i]*1000:+.1f} mm | heel-toe L/R {heeltoe["left"][i]*1000:+.1f}/{heeltoe["right"][i]*1000:+.1f} mm',font=font,fill='white')
                proc.stdin.write(np.asarray(canvas).tobytes())
                if i in keys:frames.append(canvas.copy())
        finally:
            proc.stdin.close();renderer.close()
        if proc.wait()!=0:raise RuntimeError('ffmpeg render failed')
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
    sheet=Image.new('RGB',(1280,len(frames)*720))
    for i,f in enumerate(frames):sheet.paste(f,(0,i*720))
    sheet.save(out/'keyframes.jpg',quality=90)
    dump(out/'video_manifest.json',dict(video=str(video),sha256=sha(video),frames=len(t),fps=fps,
         duration_s=len(t)/fps,decoded_successfully=True,audio=False,physics_advanced=False,
         root_xy='fixed for visualization; not odometry',root_z='inferred lowest sole corner at floor each frame',
         source='measured joint positions and IMU quaternion; no model-generated action',
         note='Video duration uses dataset 50 Hz timestamps; see analysis.json for body-clock span.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('--dataset',type=Path,required=True)
    e.add_argument('--output',type=Path,required=True);e.add_argument('--episodes',type=int,nargs='+',default=[13,15])
    r=sub.add_parser('review');r.add_argument('--source',type=Path,required=True);r.add_argument('--render',action='store_true')
    args=p.parse_args()
    if args.command=='export':export(args.dataset,args.output,args.episodes)
    else:review(args.source,args.render)
