"""Actual trained-model inspection, offline retention, OS kill and HTTP synchronization."""
import argparse,hashlib,json,os,socket,subprocess,sys,tempfile,time
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[1]
def port():
 with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
def launch(db,model,p,central):
 env={**os.environ,'INSPECT_DB':str(db),'INSPECT_MODEL':str(model),'INSPECT_CENTRAL_URL':central,'OMP_NUM_THREADS':'1'}
 proc=subprocess.Popen([sys.executable,'-m','uvicorn','server:app','--host','127.0.0.1','--port',str(p),'--log-level','error'],cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 for _ in range(200):
  if proc.poll() is not None:raise RuntimeError(proc.stderr.read().decode())
  try:
   if httpx.get(f'http://127.0.0.1:{p}/history',timeout=.2).status_code==200:return proc
  except httpx.HTTPError:pass
  time.sleep(.025)
 proc.kill();proc.wait();raise TimeoutError('startup')
def run(model,image,out):
 model=Path(model).resolve();image=Path(image).resolve();edge_port,central_port=port(),port();edge_url=f'http://127.0.0.1:{edge_port}';central_url=f'http://127.0.0.1:{central_port}';headers={'X-Station-Key':'local-inspect-only'};procs=[]
 with tempfile.TemporaryDirectory() as folder:
  folder=Path(folder)
  try:
   edge=launch(folder/'edge.sqlite',model,edge_port,central_url);procs.append(edge)
   ids=[]
   for _ in range(3):
    r=httpx.post(edge_url+'/inspect',files={'file':(image.name,image.read_bytes(),'image/jpeg')},headers=headers,timeout=60);r.raise_for_status();ids.append(r.json()['id'])
   r=httpx.post(edge_url+'/sync',headers=headers);assert r.status_code==503
   before=httpx.get(edge_url+'/history').json();assert len(before)==3 and not any(r['synced'] for r in before)
   edge.kill();edge.wait();assert edge.returncode==-9
   edge=launch(folder/'edge.sqlite',model,edge_port,central_url);procs.append(edge)
   central=launch(folder/'central.sqlite',model,central_port,central_url);procs.append(central)
   r=httpx.post(edge_url+'/sync',headers=headers);r.raise_for_status();assert r.json()['acknowledged']==3
   retry=httpx.post(edge_url+'/sync',headers=headers).json();assert retry['acknowledged']==0
   after=httpx.get(central_url+'/history').json();assert {r['id'] for r in after}==set(ids)
   original={r['id']:{k:v for k,v in r.items() if k!='synced'} for r in before}
   assert all({k:v for k,v in r.items() if k!='synced'}==original[r['id']] for r in after)
   report={'scenario':'Real-image inference, unavailable central, edge SIGKILL, restart, HTTP sync and empty retry','image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'model_mode':before[0]['mode'],'accepted_inspections':3,'unique_central_inspections':len(after),'pending_after_sync':0,'payload_match':True,'limitations':['Three repeated uploads of one real image; not three independent defect cases','One localhost recovery scenario','Kill occurs after committed records, not during SQLite commit']}
   out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
  finally:
   for p in procs:
    if p.poll() is None:p.terminate();p.wait(timeout=10)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--image',required=True);p.add_argument('--output',default='evidence/http_recovery.json');a=p.parse_args();run(a.model,a.image,a.output)
