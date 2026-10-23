"""Bind checkpoints to immutable dataset bytes and preprocessing conventions."""
import hashlib,json
from pathlib import Path

def sha256(path):
 digest=hashlib.sha256()
 with Path(path).open("rb") as source:
  for block in iter(lambda:source.read(8*1024*1024),b""):digest.update(block)
 return digest.hexdigest()

def dataset_contract(data_root,state_indices,image_keys):
 root=Path(data_root)
 return {"schema_version":1,"train_parquet_sha256":sha256(root/"train.parquet"),"normalization_sha256":sha256(root/"norm_stats.json"),"state_indices":list(state_indices),"image_keys":list(image_keys),"action_dim":23,"prediction_steps":32,"execution_steps":8,"language":"qualitative_direction_and_perception_goal","seed":20261016}

def verify_resume_contract(checkpoint,expected):
 path=Path(checkpoint)/"dataset_contract.json"
 if not path.exists():raise ValueError("Checkpoint has no dataset contract; cannot verify a safe resume")
 actual=json.loads(path.read_text())
 if actual!=expected:
  mismatch=[key for key in set(actual)|set(expected) if actual.get(key)!=expected.get(key)]
  raise ValueError("Resume dataset/preprocessing mismatch: "+", ".join(sorted(mismatch)))
