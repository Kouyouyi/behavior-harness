"""Portable preprocessing acceptance check; optionally run actual checkpoint inference."""
from pathlib import Path
import argparse,json
import numpy as np
import cv2
from transformers import AutoTokenizer
from training_data import STATE_IDX,IMAGE_KEYS

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument("--checkpoint",type=Path)
args=p.parse_args()
data=json.loads((root/"fixture/input.json").read_text())
meta=args.checkpoint or root/"checkpoint_metadata"
contract=json.loads((meta/"dataset_contract.json").read_text())
assert STATE_IDX==contract["state_indices"] and IMAGE_KEYS==contract["image_keys"]
stats=json.loads((meta/"norm_stats.json").read_text())["state"]
state=np.array(data["state"],np.float32)[STATE_IDX]
lo=np.array(stats["q01"],np.float32);hi=np.array(stats["q99"],np.float32)
state=2*(state-lo)/np.maximum(hi-lo,1e-6)-1
bins=np.digitize(np.clip(state,-1,1),np.linspace(-1,1,257)[:-1])-1
text="Task: "+data["instruction"]+", State: "+" ".join(map(str,bins))+";\nAction: "
assert text==(root/"fixture/prompt.txt").read_text()
tok=AutoTokenizer.from_pretrained(root/"tokenizer",local_files_only=True)
tokens=tok(text,add_special_tokens=True,max_length=200,padding="max_length",truncation=True,return_tensors="np")
expected=np.load(root/"fixture/expected_preprocessing.npz")
np.testing.assert_allclose(state,expected["normalized_state"],atol=1e-6,rtol=1e-6)
for name,actual in [("state_bins",bins),("input_ids",tokens["input_ids"]),("attention_mask",tokens["attention_mask"])]:
 np.testing.assert_array_equal(actual,expected[name])
paths=[root/"fixture"/x for x in data["image_paths"]]
for role,path in zip(("head","left_wrist","right_wrist"),paths):
 im=cv2.cvtColor(cv2.resize(cv2.imread(str(path)),(224,224)),cv2.COLOR_BGR2RGB)
 np.testing.assert_allclose(im.transpose(2,0,1).astype(np.float32)/255,expected[role],atol=1e-6,rtol=1e-6)
report={"preprocessing_passed":True,"sample_id":data["sample_id"],"gpu_inference_run":False}
if args.checkpoint:
 assert json.loads((meta/"READY.json").read_text())["complete"]
 import torch
 from pi05_executor import Pi05Executor
 torch.manual_seed(20261017);torch.cuda.manual_seed_all(20261017)
 executor=Pi05Executor(meta)
 actions=executor.predict({"state":data["state"],"image_paths":paths},data["instruction"])
 assert actions.shape==(32,23) and np.isfinite(actions).all()
 report.update(gpu_inference_run=True,output_shape=list(actions.shape),finite=True)
print(json.dumps(report))
