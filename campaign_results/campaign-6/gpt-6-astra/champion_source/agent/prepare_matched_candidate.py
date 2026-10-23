"""Register exact post-fit matched-control jobs without score-dependent tuning."""
import json
from pathlib import Path
current=json.loads(Path('agent/matched_mix.json').read_text())['constant_mix']
historical=json.loads(Path('agent/matched_mix_jul01.json').read_text())['constant_mix']
hist={'table':'seasonality_jul01.json','clock_table':'clock_jul01.json','event_table':'event_multipliers_jul01.json'}
specs=[('matched_bridge_v9_full',{'mix':current},10,['2026-09-19','2026-10-11']),('sep_late_matched_bridge_v9',{'mix':current},8,['2026-10-01']),('historical_matched_bridge_1h',{**hist,'mix':historical},2,['2026-07-22','2026-08-22']),('historical_new_bridge_1h',hist,2,['2026-07-29','2026-08-29']),('historical_new_matched_bridge_1h',{**hist,'mix':historical},2,['2026-07-29','2026-08-29'])]
jobs=[]
for tag,cfg,days,ends in specs:
 jobs.append({'cmd':['uv','run','agent/run_suite.py','--tag',tag,'--model','candidate_v9_research','--config',json.dumps(cfg),'--cached-prompts','--prediction-dtype','float32','--days',str(days),'--ends',*ends,'--assets','BTC','ETH','SOL','XRP','HYPE','--formats','1h','--workers','2'],'log':f'agent/results/{tag}_suite.log'})
Path('agent/queue_matched_candidate_scores.json').write_text(json.dumps(jobs,indent=2))
