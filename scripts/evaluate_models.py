import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--metrics',default='artifacts/baseline/metrics.json');a=p.parse_args();f=Path(a.metrics)
if not f.exists(): raise SystemExit(f'No actual metrics at {f}; train first. No metrics are fabricated.')
print(json.dumps(json.loads(f.read_text(encoding='utf-8')),indent=2))
