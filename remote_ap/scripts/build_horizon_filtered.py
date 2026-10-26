import os
"""Build a separate temporal-alignment dataset after the full audit completes."""
from pathlib import Path
import json,shutil,time
import pandas as pd
ROOT=Path(os.environ.get('AP_PROJECT_ROOT',str(Path.cwd()/'ap_work')));AUDIT=ROOT/"evaluation/horizon_audit";OUT=ROOT/"training_horizon_v2"
status=json.loads((AUDIT/"status.json").read_text());assert status["status"]=="complete",status
assert not OUT.exists(),"Use a new version; never overwrite a dataset"
audit=pd.concat([pd.read_parquet(p) for p in AUDIT.glob("*.parquet")],ignore_index=True)
assert not audit.sample_id.duplicated().any()
OUT.mkdir();summary={};episodes={}
for split in ("train","validation","test","radio_task_holdout"):
 source=pd.read_parquet(ROOT/f"training/{split}.parquet")
 merged=source.merge(audit.drop(columns=["split","motion_class"]),on="sample_id",how="left",validate="one_to_one")
 assert merged.base_direction_applicable.notna().all()
 base=merged.base_direction_applicable.astype(bool)
 horizon=merged.frame_end_exclusive-merged.frame_start
 keep=(horizon>=8)&((~base)|(merged.h8_all_aligned.fillna(False)&merged.h32_all_aligned.fillna(False)))
 filtered=merged[keep].copy();filtered["temporal_alignment_rule"]="base: first8 and first32 mean commands align all requested axes at deadband 0.02; non-base: not assessed by this rule; minimum8 valid steps"
 filtered.to_parquet(OUT/f"{split}.parquet",index=False)
 episodes[split]=set(filtered.episode_index)
 summary[split]={"before":len(source),"after":len(filtered),"base_kept":int(filtered.base_direction_applicable.sum()),"non_base_unassessed":int((~filtered.base_direction_applicable.astype(bool)).sum()),"before_classes":source.motion_class.value_counts().to_dict(),"after_classes":filtered.motion_class.value_counts().to_dict(),"lost_classes":sorted(set(source.motion_class)-set(filtered.motion_class))}
for split in ("validation","test","radio_task_holdout"):assert not episodes["train"]&episodes[split]
shutil.copy2(ROOT/"training/norm_stats.json",OUT/"norm_stats.json")
manifest={"status":"complete","version":"horizon_filtered_v2","created_at":time.time(),"summary":summary,"normalization":"Reuse fixed train-only statistics from original training pool; no held-out states or actions used. Statistics are not recomputed on filtered distribution.","limitations":["Filtering favors immediately moving examples and can reduce stopped-state coverage.","Non-base opening and view-height labels are not validated by base command sign checks.","Measured command alignment is not realized motion or visually verified information gain.","Original fixed evaluation suite is retained unchanged; do not interpret filtered validation improvements as performance on all original samples."]}
(OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps({k:{a:b for a,b in v.items() if not a.endswith("classes")} for k,v in summary.items()},ensure_ascii=False))
