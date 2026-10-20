The copied source hashes are preserved in provenance.json. The runtime is
unchanged from that copy. An attempted startup segmentation-modality patch was
reverted after an Isaac native synthetic-data crash. Live rollout uses RGB.
Optional segmentation is isolated in initialize-only audit processes by the
ap_harness adapter. Each run records the runtime source hashes.
