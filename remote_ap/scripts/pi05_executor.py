from pathlib import Path
import json,cv2,numpy as np,torch
from transformers import AutoTokenizer
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from training_data import STATE_IDX,IMAGE_KEYS
class Pi05Executor:
 def __init__(self,checkpoint,device='cuda'):
  self.device=device;checkpoint=Path(checkpoint);self.stats=json.loads((checkpoint/'norm_stats.json').read_text());cfg=PreTrainedConfig.from_pretrained(checkpoint);cfg.device=device;cfg.gradient_checkpointing=False
  self.policy=PI05Policy.from_pretrained(checkpoint,config=cfg,local_files_only=True,strict=True).eval();self.tokenizer=AutoTokenizer.from_pretrained('/cs/models/paligemma-3b-pt-224',local_files_only=True)
 def predict(self,observation,instruction):
  state=np.asarray(observation['state'],np.float32)
  if state.shape!=(61,) or not np.isfinite(state).all():raise ValueError('Expected original R1Pro 61D state')
  if len(observation['image_paths'])!=3:raise ValueError('Expected head, left wrist, right wrist images')
  s=self.stats['state'];lo=np.asarray(s['q01'],np.float32);hi=np.asarray(s['q99'],np.float32);state=2*(state[STATE_IDX]-lo)/np.maximum(hi-lo,1e-6)-1
  bins=np.digitize(np.clip(state,-1,1),np.linspace(-1,1,257)[:-1])-1;text='Task: '+instruction+', State: '+' '.join(map(str,bins))+';\nAction: '
  tokens=self.tokenizer(text,add_special_tokens=True,max_length=200,padding='max_length',truncation=True,return_tensors='pt');batch={'observation.state':torch.from_numpy(state.astype(np.float32))[None].to(self.device),'observation.language.tokens':tokens['input_ids'].to(self.device),'observation.language.attention_mask':tokens['attention_mask'].bool().to(self.device)}
  for key,path in zip(IMAGE_KEYS,observation['image_paths']):
   im=cv2.imread(str(path));assert im is not None,path;im=cv2.cvtColor(cv2.resize(im,(224,224)),cv2.COLOR_BGR2RGB);batch[key]=torch.from_numpy(im.copy()).permute(2,0,1)[None].float().to(self.device)/255
  with torch.inference_mode(),torch.autocast(torch.device(self.device).type,dtype=torch.bfloat16,enabled=torch.device(self.device).type=='cuda',cache_enabled=False):out=self.policy.predict_action_chunk(batch)[0].float().cpu().numpy()
  s=self.stats['action'];lo=np.asarray(s['q01'],np.float32);hi=np.asarray(s['q99'],np.float32);out=(out+1)/2*np.maximum(hi-lo,1e-6)+lo
  if not np.isfinite(out).all() or out.shape!=(32,23):raise ValueError('Invalid predicted action')
  return out.astype(np.float32)
