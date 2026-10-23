"""Local pi05 finetuning with explicit temporal loss masks and train-only statistics."""
from pathlib import Path
import argparse,json,time,os,random,math,shutil,itertools
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from transformers import AutoTokenizer
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.configs import PolicyFeature,FeatureType
from training_data import APDataset,IMAGE_KEYS,STATE_IDX
from training_version import dataset_contract,verify_resume_contract
OUT=Path(os.environ.get('AP_PROJECT_ROOT','/cs/kyy/behavior_ap_experiment'))
def masked_loss(losses,mask):
 losses=losses[:,:,:23];weights=mask.to(losses.dtype).unsqueeze(-1)
 return (losses*weights).sum()/(weights.sum()*23).clamp_min(1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=OUT/'training');p.add_argument('--steps',type=int,default=2);p.add_argument('--resume',default=None);p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32');p.add_argument('--memory-fraction',type=float,default=.82);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--accumulation',type=int,default=1);p.add_argument('--run-name',default='smoke');p.add_argument('--expert-only',action='store_true');p.add_argument('--unbalanced',action='store_true');p.add_argument('--workers',type=int,default=2);args=p.parse_args()
 contract=dataset_contract(args.data_root,STATE_IDX,IMAGE_KEYS)
 if args.resume:verify_resume_contract(args.resume,contract)
 torch.manual_seed(20261016);np.random.seed(20261016);random.seed(20261016);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(args.memory_fraction,0)
 run=OUT/'runs'/args.run_name;run.mkdir(parents=True,exist_ok=True)
 if ((run/'checkpoint').exists() or (run/'metrics.jsonl').exists()) and not args.resume:raise RuntimeError('Run already has checkpoint; choose a new name or implement explicit resume')
 index=pd.read_parquet(args.data_root/'train.parquet');stats=json.loads((args.data_root/'norm_stats.json').read_text());tokenizer=AutoTokenizer.from_pretrained(os.environ.get('AP_TOKENIZER_PATH','/cs/models/paligemma-3b-pt-224'),local_files_only=True)
 (run/'dataset_contract.json').write_text(json.dumps(contract,indent=2))
 dataset=APDataset(index,stats,tokenizer)
 counts=index.motion_class.value_counts();weight=index.motion_class.map(lambda x:1/np.sqrt(counts[x])).to_numpy();weight=np.minimum(weight,10*np.median(weight)) if not args.unbalanced else np.ones(len(index))
 sampler=WeightedRandomSampler(torch.from_numpy(weight),args.steps*args.batch_size*args.accumulation,replacement=True,generator=torch.Generator().manual_seed(20261016))
 resume_state=None
 if args.resume:
  resume=Path(args.resume);assert (resume/'READY.json').exists(),'Checkpoint has no completion marker'
  resume_state=torch.load(resume/'training_state.pt',map_location='cpu',weights_only=False)
  sampler=list(itertools.islice(iter(sampler),resume_state['step']*args.batch_size*args.accumulation,None))
 loader=DataLoader(dataset,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,pin_memory=True)
 cfg=PI05Config(device='cuda',dtype=args.dtype,chunk_size=32,n_action_steps=8,gradient_checkpointing=True,train_expert_only=args.expert_only,freeze_vision_encoder=False,input_features={**{k:PolicyFeature(type=FeatureType.VISUAL,shape=(3,224,224)) for k in IMAGE_KEYS},'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(25,))},output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(23,))},push_to_hub=False)
 policy=PI05Policy.from_pretrained(args.resume or os.environ.get('AP_BASE_MODEL_PATH','/cs/models/pi05_base'),config=cfg,local_files_only=True,strict=True);policy.train()
 params=[p for p in policy.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=2.5e-5,betas=(.9,.95),weight_decay=.01,foreach=False)
 record={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'train_examples':len(index),'trainable_parameters':sum(p.numel() for p in params),'base_weights':'/cs/models/pi05_base','dtype':args.dtype,'temporal_loss_mask':True,'state_dim':25,'action_dim':23,'status':'running'};(run/'config.json').write_text(json.dumps(record,indent=2));optimizer.zero_grad(set_to_none=True);started=time.time();step=0;last_loss=0.
 if resume_state:
  optimizer.load_state_dict(resume_state['optimizer']);step=resume_state['step'];torch.set_rng_state(resume_state['torch_rng']);torch.cuda.set_rng_state_all(resume_state['cuda_rng'])
 for micro,batch in enumerate(loader,1):
  batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
  with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
   images,img_masks=policy._preprocess_images(batch);actions=policy.prepare_action(batch);noise=policy.model.sample_noise(actions.shape,actions.device);times=policy.model.sample_time(actions.shape[0],actions.device)
   losses=policy.model.forward(images,img_masks,batch['observation.language.tokens'],batch['observation.language.attention_mask'],actions,noise,times)
   loss=masked_loss(losses,batch['action_loss_mask'])
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
  (loss/args.accumulation).backward();last_loss+=float(loss.detach())/args.accumulation
  if micro%args.accumulation:continue
  grad=torch.nn.utils.clip_grad_norm_(params,1.0)
  if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
  warmup=min(1000,max(1,args.steps//20));progress=max(0.,(step-warmup)/max(1,args.steps-warmup));lr=2.5e-5*(min(1.,(step+1)/warmup) if step<warmup else .1+.9*.5*(1+math.cos(math.pi*progress)))
  for group in optimizer.param_groups:group['lr']=lr
  optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
  status={'step':step,'lr':lr,'loss':last_loss,'grad_norm':float(grad),'elapsed_s':time.time()-started,'max_memory_gib':torch.cuda.max_memory_allocated()/2**30};print(json.dumps(status),flush=True)
  with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
  (run/'status.json').write_text(json.dumps({'status':'running',**status}));last_loss=0.
  if step%1000==0 or step==args.steps:
   if shutil.disk_usage(run).free<100*1024**3:raise RuntimeError('Less than 100 GiB storage reserve before checkpoint')
   dest=run/('checkpoint' if step==args.steps else f'checkpoint_{step:06d}');policy.save_pretrained(dest);(dest/'dataset_contract.json').write_text(json.dumps(contract,indent=2));(dest/'norm_stats.json').write_text(json.dumps(stats));torch.save({'optimizer':optimizer.state_dict(),'step':step,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},dest/'training_state.pt');(dest/'READY.json').write_text(json.dumps({'step':step,'complete':True}))
  if step>=args.steps:break
 (run/'status.json').write_text(json.dumps({'status':'complete','steps':step,'checkpoint':str(run/'checkpoint'),'scope':'training only; closed-loop success not evaluated'}))
if __name__=='__main__':main()
