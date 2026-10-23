import sys,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer
from training_data import APDataset
from train_pi05 import masked_loss
root=Path('/cs/kyy/behavior_ap_experiment')
# A padded action must have exactly zero gradient, regardless of its value.
x=torch.randn(2,32,32,requires_grad=True);mask=torch.zeros(2,32,dtype=torch.bool);mask[0,:10]=True;mask[1,:20]=True
masked_loss(x,mask).backward();assert torch.count_nonzero(x.grad[~mask])==0;assert torch.count_nonzero(x.grad[:,:,23:])==0;assert torch.count_nonzero(x.grad[:,:,:23][mask])==30*23
stats=json.loads((root/'training/norm_stats.json').read_text());index=pd.read_parquet(root/'training/train.parquet')
tokenizer=AutoTokenizer.from_pretrained('/cs/models/paligemma-3b-pt-224',local_files_only=True)
d=APDataset(index,stats,tokenizer);checks=[]
for i in [0,len(d)//2,len(d)-1]:
 s=d[i];assert s['action'].shape==(32,23);assert s['observation.state'].shape==(25,)
 for k,v in s.items():assert torch.isfinite(v).all(),k
 assert s['action_loss_mask'].sum()>0
 checks.append({'sample_id':index.iloc[i].sample_id,'instruction':index.iloc[i].instruction,'valid_action_steps':int(s['action_loss_mask'].sum())})
report={'passed':True,'masked_gradient_verified':True,'padding_dimensions_excluded':True,'actual_decoded_samples':checks,'training_examples':len(d)};(root/'reports/training_contract_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False),flush=True)
