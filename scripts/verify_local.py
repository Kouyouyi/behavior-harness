import json
import numpy as np
import cv2
from ap_harness.paths import HANDOFF, ROOT
from ap_harness.preprocessing import prepare_text
from ap_harness.control import save_json

fixture = json.loads((HANDOFF/'fixture/input.json').read_text())
expected = np.load(HANDOFF/'fixture/expected_preprocessing.npz')
normalized,bins,prompt,encoding = prepare_text(fixture['state'],fixture['instruction'])
np.testing.assert_allclose(normalized,expected['normalized_state'],rtol=1e-6,atol=1e-6)
np.testing.assert_array_equal(bins,expected['state_bins'])
assert prompt == (HANDOFF/'fixture/prompt.txt').read_text()
ids = [0]*(200-len(encoding.ids)) + encoding.ids
mask = [0]*(200-len(encoding.ids))+[1]*len(encoding.ids)
np.testing.assert_array_equal([ids], expected['input_ids'])
np.testing.assert_array_equal([mask], expected['attention_mask'])
for role,path in zip(['head','left_wrist','right_wrist'],fixture['image_paths']):
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(HANDOFF/'fixture'/path)),(224,224)),cv2.COLOR_BGR2RGB)
    np.testing.assert_allclose(rgb.transpose(2,0,1).astype(np.float32)/255,expected[role],rtol=1e-6,atol=1e-6)
result = {'preprocessing_passed':True,'tokenizer_backend':'tokenizers, frozen packaged tokenizer.json',
          'tokens_before_padding':len(encoding.ids),'gpu_inference_run':False}
save_json(ROOT/'outputs/local_preprocessing.json', result)
print(json.dumps(result))
