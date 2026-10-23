"""Sequential bounded suites queued behind a verified current-process dependency.

The caller must reserve `workers` cores once the predecessor exits. Arguments and
logs are durable. A vanished dependency is not interpreted as a completed experiment:
only scheduling resources are released; run_suite itself checks artifact completeness.
"""
import argparse,os,time,subprocess,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,default=0);p.add_argument('--jobs',required=True);a=p.parse_args()
if a.after_pid:
 while Path(f'/proc/{a.after_pid}').exists():
  stat=Path(f'/proc/{a.after_pid}/stat').read_text().split()
  if stat[2]=='Z':break
  time.sleep(5)
for job in json.loads(Path(a.jobs).read_text()):
 cmd=job['cmd'];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',**job.get('env',{}))
 with Path(job['log']).open('w') as f:
  r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
 print(json.dumps({'cmd':cmd,'exit':r.returncode}),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
