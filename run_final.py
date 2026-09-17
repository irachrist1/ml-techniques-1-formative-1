"""Execute preselected configurations and seeds; never choose a model on test error."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
config=json.loads((ROOT/'configs/final.json').read_text())
areas=json.loads((ROOT/'results/eda_summary.json').read_text())['areas']
for area in areas:
 for seed in [42,43,44]:
  out=ROOT/'results/runs'/f"final_{config['id']}_area{area}_seed{seed}/summary.json"
  if out.exists():continue
  subprocess.run([sys.executable,'experiments.py','--config','configs/final.json','--area',str(area),'--seed',str(seed),'--phase','final'],cwd=ROOT,check=True)
subprocess.run([sys.executable,'summarize_results.py'],cwd=ROOT,check=True)
subprocess.run([sys.executable,'verify_submission.py'],cwd=ROOT,check=True)
