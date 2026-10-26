# Third-party provenance

- SeekAct's Agent/tool framework is adapted from
  [jiachengliu3/Symbiosis-Agent](https://github.com/jiachengliu3/Symbiosis-Agent),
  commit `05c51709bdf7862a26aa0a513d77d5ce15b10564`, including its RAX-derived
  framework and original attributions. No top-level LICENSE was present in that
  local snapshot; this repository does not invent a blanket license for it.
  Product/runtime naming is changed to SeekAct; source attribution is retained.
- The AP pipeline is the existing behavior-harness export. Its LeRobot dependency
  and tokenizer retain their upstream licensing. `remote_ap/vendor`,
  `remote_ap/serving_runtime`, and `remote_ap/tokenizer` are relative links into
  the already-published `pi05_harness_20261017` tree, not separate new copies.
- `third_party/champion` comes from
  [IliaLarchenko/behavior-1k-solution](https://github.com/IliaLarchenko/behavior-1k-solution)
  at `ca556f74a455cef7987a2be4537b5ac85cc56dd7`. Apache-2.0 LICENSE is preserved.
  The actual deployed compatibility adapters are included; tracked changes are
  recorded in `LOCAL_ADAPTERS.patch`, with the added `eval_utils_2025.py` also
  listed in the provenance JSON. These adapters do not claim a new champion model.
- The included OpenPI source is from `wensi-ai/openpi` at
  `01177e0242a1c7e8fad2547caa0e987def614cda`, with the local inference-compatible
  `b1k_policy.py` adapter. Its Apache-2.0 license and original attributions remain.
  A small openpi-client package present in the installed runtime is included.
- BEHAVIOR/Isaac assets, datasets and model weights are not redistributed here.
  Obtain them under their original licenses. The separate simulator dependency
  is `Kouyouyi/behavior-ap@f99653f78f94fcb0547f9575c52019dc47fa8f6b`.

The integration tests validate code/control contracts, not third-party license
permissions, newly trained weights or successful closed-loop robot behavior.
