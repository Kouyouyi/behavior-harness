"""Local wire-input audit using the packaged tokenizer and quantile statistics."""
import json
import numpy as np
from .paths import HANDOFF

def prepare_text(state, instruction):
    from tokenizers import Tokenizer
    contract = json.loads((HANDOFF/'checkpoint_metadata/dataset_contract.json').read_text())
    stats = json.loads((HANDOFF/'checkpoint_metadata/norm_stats.json').read_text())['state']
    values = np.asarray(state, dtype=np.float32)
    if values.shape != (61,) or not np.isfinite(values).all():
        raise ValueError('Expected raw finite 61D state')
    lo, hi = np.asarray(stats['q01'], np.float32), np.asarray(stats['q99'], np.float32)
    normalized = 2*(values[contract['state_indices']]-lo)/np.maximum(hi-lo,1e-6)-1
    bins = np.digitize(np.clip(normalized,-1,1),np.linspace(-1,1,257)[:-1])-1
    prompt = 'Task: '+instruction+', State: '+' '.join(map(str,bins))+';\nAction: '
    tokenizer = Tokenizer.from_file(str(HANDOFF/'tokenizer/tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    encoded = tokenizer.encode(prompt, add_special_tokens=True)
    if len(encoded.ids)>200:
        raise ValueError(f'Instruction would truncate state tokens: {len(encoded.ids)} > 200')
    return normalized, bins, prompt, encoded
