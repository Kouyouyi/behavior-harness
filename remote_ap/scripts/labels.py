"""Matched numerical / intent labels from the same annotated motion segment."""
import re,numpy as np
VERBS={'move to':'接近','turn to':'转向','open door':'打开门','open drawer':'拉开抽屉','open lid':'揭开盖子','close door':'关上门','close drawer':'关上抽屉','close lid':'合上盖子','pick up from':'拿起','place in':'放入','place on':'放到表面','place under':'放到下方','place in next to':'放入并靠近','place on next to':'放到旁边的表面','hand over':'换手递交','hold':'保持握持','release':'松开','attach':'安装','chop':'切开','hang':'挂起','ignite':'点燃','insert':'插入','pour':'倾倒','press':'按下','push to':'推到','spray':'喷洒','sweep surface':'清扫表面','tip over':'倾倒容器','turn off switch':'关闭开关','turn on switch':'打开开关','wipe hard':'擦拭'}
OBJECTS={'fridge':'冰箱','coffee_table':'茶几','cabinet':'柜子','bottom_cabinet':'下柜','oven':'烤箱','stove':'炉灶','countertop':'操作台','drawer':'抽屉','radio':'收音机','sink':'水槽','table':'桌子','beer_bottle':'瓶子','water_bottle':'水瓶','microwave':'微波炉','dishwasher':'洗碗机','trash_can':'垃圾桶','bowl':'碗','plate':'盘子','cup':'杯子','door':'门','lid':'盖子','right':'右手','left':'左手'}
def clean_object(s):
 s=str(s);s=re.sub(r'_[a-z]{6}_\d+$','',s);s=re.sub(r'_\d+$','',s);return OBJECTS.get(s,s.replace('_',' '))
def describe(skill):
 desc=skill.get('skill_description',[]);objects=skill.get('object_id',[]);parts=[]
 for i,verb in enumerate(desc):
  obj=objects[i] if i<len(objects) else [];obj=obj if isinstance(obj,list) else [obj]
  names=[clean_object(x) for x in obj if isinstance(x,str)]
  if verb=='move to':text='接近'+('、'.join(names) or '目标区域')+'，调整观察位置'
  elif verb=='turn to':text='转向'+('、'.join(names) or '目标区域')+'，调整观察方向'
  elif verb.startswith('open '):text={'open door':'打开','open drawer':'拉开','open lid':'揭开'}[verb]+('、'.join(names) or '目标容器')
  elif verb=='pick up from' and len(names)>=2:text=f'从{names[1]}拿起{names[0]}'
  elif verb in ['place in','place on','place under'] and len(names)>=2:text=f'将{names[0]}'+{'place in':'放入','place on':'放到','place under':'放到下方：'}[verb]+names[1]
  else:text=VERBS.get(verb,verb)+' '+('、'.join(names) or '目标物体')
  parts.append(text.strip())
 return '；'.join(parts)+'。' if parts else '未标注动作。'

def relative_camera_targets(poses,end):
 R=poses[:,:3,:3];p=poses[:,:3,3];endR=R[end];endp=p[end]
 rotation=np.einsum('nji,njk->nik',R,endR);d=np.einsum('nji,nj->ni',R,endp-p);view=rotation[:,:,2]
 yaw=np.degrees(np.arctan2(view[:,0],view[:,2]));pitch=np.degrees(np.arctan2(-view[:,1],np.hypot(view[:,0],view[:,2])))
 # Optical yaw/pitch followed by camera-axis roll.
 cy,sy=np.cos(np.radians(yaw)),np.sin(np.radians(yaw));cp,sp=np.cos(np.radians(pitch)),np.sin(np.radians(pitch))
 xbasis=np.stack([cy,np.zeros(len(cy)),-sy],axis=1);ybasis=np.stack([sy*sp,cp,cy*sp],axis=1);right=rotation[:,:,0]
 roll=np.degrees(np.arctan2(np.einsum('ni,ni->n',ybasis,right),np.einsum('ni,ni->n',xbasis,right)))
 packed=np.c_[d,rotation[:,:,0],rotation[:,:,1]].astype(np.float32)
 return packed,np.c_[yaw,pitch,roll,d[:,0],-d[:,1],d[:,2]]

MOTIONS=[('向右转','向左转','度',1.,0),('抬高视线','压低视线','度',1.,0),('顺时针倾转视角','逆时针倾转视角','度',1.,0),('向右移动视点','向左移动视点','米',.02,2),('向上移动视点','向下移动视点','米',.02,2),('向前移动视点','向后移动视点','米',.02,2)]
def motion_text(values,numerical):
 parts=[]
 for value,(positive,negative,unit,threshold,precision) in zip(values,MOTIONS):
  if abs(value)<threshold:continue
  verb=positive if value>0 else negative
  parts.append(verb+(f'约{abs(value):.{precision}f}{unit}' if numerical else ''))
 text='，'.join(parts) if parts else '保持当前观察姿态'
 return ('以当前相机坐标为参考，' if numerical and parts else '')+text+'。'

def normalize_skills(annotation):
 expanded=[];issues=[]
 def intervals(value):
  if isinstance(value,list) and len(value)==2 and all(isinstance(x,(int,float)) for x in value):return [value]
  if isinstance(value,list):
   result=[]
   for child in value:result.extend(intervals(child))
   return result
  issues.append({'problem':'malformed_frame_range','value':value});return []
 for skill in annotation.get('skill_annotation',[]):
  for begin,end in intervals(skill.get('frame_duration',[])):
   if end<=begin:
    issues.append({'problem':'nonpositive_frame_range','skill_idx':skill.get('skill_idx'),'frame_duration':[begin,end]});continue
   copy=dict(skill);copy['frame_duration']=[int(begin),int(end)];expanded.append(copy)
 return expanded,issues
