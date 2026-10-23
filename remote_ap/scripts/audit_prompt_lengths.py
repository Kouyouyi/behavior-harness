from pathlib import Path
import json
import pandas as pd
from transformers import AutoTokenizer
ROOT=Path("/cs/kyy/behavior_ap_experiment");data=ROOT/"training_horizon_v2"
index=pd.read_parquet(data/"train.parquet",columns=["instruction"]);counts=index.instruction.value_counts();prompts=list(counts.index)
tok=AutoTokenizer.from_pretrained("/cs/models/paligemma-3b-pt-224",local_files_only=True)
# Exhaustively test the encoded lengths of all 256 possible state-bin values.
lengths={str(i):len(tok.encode(str(i),add_special_tokens=False)) for i in range(256)}
longest=max(lengths,key=lengths.get);shortest=min(lengths,key=lengths.get)
report={"training_rows":len(index),"unique_instructions":len(prompts),"bin_token_lengths":sorted(set(lengths.values())),"worst_bin":longest,"best_bin":shortest,"max_tokens":200}
for name,token in (("longest_bins",longest),("shortest_bins",shortest)):
 texts=["Task: "+p+", State: "+" ".join([token]*25)+";\nAction: " for p in prompts]
 encoded=tok(texts,add_special_tokens=True,truncation=False,padding=False)
 sizes=[len(ids) for ids in encoded["input_ids"]]
 bad=[{"instruction":p,"tokens":n,"training_rows":int(counts[p])} for p,n in zip(prompts,sizes) if n>200]
 report[name]={"maximum_tokens":max(sizes),"overflow_unique_instructions":len(bad),"overflow_training_rows":sum(x["training_rows"] for x in bad),"examples":sorted(bad,key=lambda x:x["tokens"],reverse=True)[:10]}
report["scope"]="Every distinct training instruction, with 25 repetitions of longest/shortest encoded state-bin tokens. Contextual numeric tokenization may vary; any overflow requires exact state-derived follow-up."
(data/"prompt_length_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
