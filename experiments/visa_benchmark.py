"""Prespecified three-category VisA evaluation with official one-class splits."""
import argparse,csv,hashlib,json,platform,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,joblib
from PIL import Image
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from detector import fit,predict
CATEGORIES=['cashew','capsules','pcb1']
SETTINGS=[(64,256),(128,1000),(256,1000)]
def rgb(path):
 with Image.open(path) as im:return np.asarray(im.convert('RGB').resize((64,64)),dtype=float).mean((0,1))/255

def summary(y,s,threshold):
 y=np.array(y);s=np.array(s);f=s>threshold;rng=np.random.default_rng(42);boot=[]
 for _ in range(300):
  ix=rng.integers(0,len(y),len(y))
  if len(np.unique(y[ix]))==2:boot.append(roc_auc_score(y[ix],s[ix]))
 return {'n':len(y),'normal':int((y==0).sum()),'anomaly':int(y.sum()),'auroc':float(roc_auc_score(y,s)),'average_precision':float(average_precision_score(y,s)),'normal_false_alert_rate':float(f[y==0].mean()),'anomaly_detection_rate':float(f[y==1].mean()),'threshold':float(threshold),'auroc_bootstrap_95':list(map(float,np.quantile(boot,[.025,.975])))}
def run(root,split,out,neural=False):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=list(csv.DictReader(open(split)));reports=[];manifest=[]
 for category in CATEGORIES:
  records=[r for r in rows if r['object']==category]
  train=[root/r['image'] for r in records if r['split']=='train' and r['label']=='normal']
  test=[r for r in records if r['split']=='test']
  if len(records)!=len(train)+len(test):raise ValueError('Unexpected official split')
  train_hash=set();test_hash=set()
  for r in records:
   p=root/r['image'];h=hashlib.sha256(p.read_bytes()).hexdigest();manifest.append({**r,'sha256':h})
   (train_hash if r['split']=='train' else test_hash).add(h)
  if train_hash&test_hash:raise ValueError('Exact train/test overlap')
  y=[int(r['label']!='normal') for r in test]
  for size,memory in ([(128,1000)] if neural else SETTINGS):
   start=time.perf_counter();model=fit(train,'visa',size,memory,42,extractor='resnet18_patch' if neural else 'handcrafted');raw=[]
   for r in test:
    with Image.open(root/r['image']) as im:p=predict(model,im)
    raw.append({'image':r['image'],'label':int(r['label']!='normal'),'score':p['score'],'ms':p['inference_ms']})
   path=out/f'{category}_{size}_{memory}.joblib';joblib.dump(model,path)
   result={'category':category,'method':'frozen_resnet18_patch_nearest_neighbor' if neural else 'handcrafted_patch_nearest_neighbor','size':size,'memory':memory,'fit_images':len(model['train_files']),'validation_images':len(model['validation_files']),'metrics':summary(y,[r['score'] for r in raw],model['threshold']),'median_ms':float(np.median([r['ms'] for r in raw])),'p95_ms':float(np.quantile([r['ms'] for r in raw],.95)),'model_bytes':path.stat().st_size,'seconds':time.perf_counter()-start,'raw':raw}
   model['report']={k:v for k,v in result.items() if k!='raw'};joblib.dump(model,path)
   reports.append(result);print(json.dumps({k:v for k,v in result.items() if k!='raw'}),flush=True)
   # At 128, compare global-color reference using exactly the same fit/validation.
   if size==128 and not neural:
    center=np.mean([rgb(p) for p in model['train_files']],axis=0)
    th=np.quantile([np.linalg.norm(rgb(p)-center) for p in model['validation_files']],.95)
    sc=[float(np.linalg.norm(rgb(root/r['image'])-center)) for r in test]
    reports.append({'category':category,'method':'global_RGB_distance_reference','metrics':summary(y,sc,th),'raw':[{'image':r['image'],'label':label,'score':s} for r,label,s in zip(test,y,sc)]})
 report={'dataset':'VisA_20220922','source':'https://github.com/amazon-science/spot-diff','split_sha256':hashlib.sha256(Path(split).read_bytes()).hexdigest(),'seed':42,'platform':platform.platform(),'cpu_threads':1,'neural':neural,'reports':reports,'limitations':['Handcrafted baseline, not neural PatchCore','Three categories only','No pixel metric or factory validation','Single-host timings','Bootstrap images may not be independent acquisitions']}
 (out/'evaluation.json').write_text(json.dumps(report,indent=2));(out/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--neural',action='store_true');p.add_argument('--root',required=True);p.add_argument('--split',required=True);p.add_argument('--output',default='artifacts/visa');a=p.parse_args()
 if a.neural:
  import torch
  torch.set_num_threads(1)
 with threadpool_limits(limits=1):run(a.root,a.split,a.output,a.neural)
