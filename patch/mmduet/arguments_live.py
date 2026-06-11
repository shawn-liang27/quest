--- a/models/arguments_live.py
+++ b/models/arguments_live.py
@@ LiveTestArguments, add these fields:

+    dataset: str = 'default'                     # 'default', 'roma', 'omnipro', or 'streamingbench'
+    hf_name: str = 'EurekaTian/ROMA_proactive'   # HF dataset repo (used when dataset=roma)
+    hf_split: str = 'alert'                      # HF dataset split (used when dataset=roma)
+    probe_attention: bool = False                 # enable attention-mass diagnostic probe
+    probe_sample_every: int = 4                   # probe every Nth transformer layer
