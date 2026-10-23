"""Fetch only immutable public model/code artifacts, with provenance and license."""
import os,json,hashlib,urllib.request,zipfile,io
from pathlib import Path
os.environ['HF_HOME']=str(Path('agent/hf_cache').resolve())
from huggingface_hub import hf_hub_download
revision='0581e2c56cb06feb51cfd98fc2b4005b74f7187b';repo='google/timesfm-1.0-200m-pytorch'
with urllib.request.urlopen('https://pypi.org/pypi/timesfm/1.3.0/json',timeout=30) as r:metadata=json.load(r)
item=next(x for x in metadata['urls'] if x['filename'].endswith('.whl'))
with urllib.request.urlopen(item['url'],timeout=30) as r:payload=r.read()
assert hashlib.sha256(payload).hexdigest()==item['digests']['sha256']=='a99e22b13654bdbfaecb14a48e4a8dc6c990b191a0322a5cacd09eb0df3bfebe'
archive=zipfile.ZipFile(io.BytesIO(payload));vendor=Path('agent/vendor/timesfm_v1');vendor.mkdir(parents=True,exist_ok=True)
(vendor/'__init__.py').write_text('')
(vendor/'pytorch_patched_decoder.py').write_bytes(archive.read('timesfm/pytorch_patched_decoder.py'))
license_name=next(n for n in archive.namelist() if n.rsplit('/',1)[-1].upper()=='LICENSE')
(vendor/'LICENSE').write_bytes(archive.read(license_name))
(vendor/'provenance.json').write_text(json.dumps({'package':'timesfm','version':'1.3.0','wheel_url':item['url'],'wheel_sha256':item['digests']['sha256'],'source_file':'timesfm/pytorch_patched_decoder.py','modified':False},indent=2))
model_dir=Path('agent/models/timesfm-1.0-200m');model_dir.mkdir(parents=True,exist_ok=True)
for filename in ['README.md','torch_model.ckpt']:
 path=hf_hub_download(repo_id=repo,filename=filename,revision=revision,local_dir=str(model_dir))
 print(filename,Path(path).stat().st_size,'bytes',flush=True)
(model_dir/'provenance.json').write_text(json.dumps({'repo':repo,'revision':revision,'metadata_last_modified':'2024-08-29T23:52:11.000Z','license':'Apache-2.0'},indent=2))
