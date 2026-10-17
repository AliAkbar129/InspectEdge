"""Normal-only handcrafted patch baseline. No neural weights or claimed novelty."""
import argparse,json,time,hashlib
from pathlib import Path
import joblib,numpy as np
from PIL import Image,ImageFilter
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score,average_precision_score

def patches(image,size=128):
    a=np.asarray(image.convert('RGB').resize((size,size)),dtype=float)/255
    grid=size//16;desc=[]
    for r in range(grid):
        for c in range(grid):
            patch=a[r*16:(r+1)*16,c*16:(c+1)*16]
            dy,dx=np.gradient(patch.mean(2));desc.append(np.r_[patch.mean((0,1)),patch.std((0,1)),np.abs(dx).mean(),np.abs(dy).mean()])
    return np.asarray(desc)

def fit(paths,mode='synthetic_fixture',size=128,memory=1000,seed=42):
    if size not in [64,128,256] or memory<1:raise ValueError('Invalid size or memory')
    if len(paths)<10:raise ValueError('At least 10 distinct normal images required')
    paths=sorted(map(Path,paths));hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    if len(set(hashes))!=len(hashes):raise ValueError('Duplicate normal images would contaminate validation')
    rng=np.random.default_rng(seed);order=rng.permutation(len(paths));cut=max(2,int(.2*len(paths)))
    val=[paths[i] for i in order[:cut]];train=[paths[i] for i in order[cut:]]
    def read(p):
        with Image.open(p) as im:return patches(im,size)
    x=np.concatenate([read(p) for p in train]);scaler=StandardScaler().fit(x);x=scaler.transform(x)
    memory_bank=x[rng.choice(len(x),min(memory,len(x)),replace=False)]
    artifact={'schema_version':1,'scaler':scaler,'memory':memory_bank,'size':size,'mode':mode,'threshold':None,'seed':seed,'normal_image_hashes':hashes,
              'train_files':[str(p) for p in train],'validation_files':[str(p) for p in val]}
    scores=[]
    for p in val:
        with Image.open(p) as im:scores.append(predict(artifact,im)['score'])
    artifact['threshold']=float(np.quantile(scores,.95))
    return artifact

def predict(artifact,image):
    start=time.perf_counter();size=artifact['size'];x=artifact['scaler'].transform(patches(image,size))
    nn=NearestNeighbors(n_neighbors=1,algorithm='brute',n_jobs=1).fit(artifact['memory'])
    distances=nn.kneighbors(x,return_distance=True)[0][:,0];score=float(distances.max());threshold=artifact['threshold']
    grid=size//16;heat=distances.reshape(grid,grid)
    denom=threshold if threshold is not None and threshold>0 else max(float(heat.max()),1e-9)
    intensity=np.clip(heat/denom,0,1)
    overlay=np.zeros((grid,grid,3),dtype=np.uint8);overlay[:,:,0]=(intensity*255).astype('uint8');overlay[:,:,2]=((1-intensity)*100).astype('uint8')
    heat_image=Image.fromarray(overlay).resize((size,size),Image.Resampling.NEAREST)
    return {'score':score,'threshold':threshold,'anomaly':score>threshold if threshold is not None else None,
            'inference_ms':(time.perf_counter()-start)*1000,'heatmap':heat_image,'mode':artifact['mode']}

def evaluate(artifact,category):
    rows=[];root=Path(category)
    for path in sorted((root/'test').glob('*/*.png')):
        with Image.open(path) as im:r=predict(artifact,im)
        rows.append({'file':str(path.relative_to(root)),'label':int(path.parent.name!='good'),'score':r['score'],'anomaly':r['anomaly'],'ms':r['inference_ms']})
    if not rows or len(set(r['label'] for r in rows))<2:raise ValueError('Both normal and anomaly test images required')
    y=np.array([r['label'] for r in rows]);s=np.array([r['score'] for r in rows]);flag=np.array([r['anomaly'] for r in rows])
    return {'mode':artifact['mode'],'n':len(rows),'image_auroc':float(roc_auc_score(y,s)), 'average_precision':float(average_precision_score(y,s)),
            'normal_false_alert_rate':float(flag[y==0].mean()),'anomaly_detection_rate':float(flag[y==1].mean()),
            'median_ms':float(np.median([r['ms'] for r in rows])),'p95_ms':float(np.quantile([r['ms'] for r in rows],.95)),
            'memory_rows':len(artifact['memory']),'threshold':artifact['threshold'],'size':artifact['size'],'raw':rows,
            'limitations':['Handcrafted baseline, not PatchCore','No pixel-level metric','Single host CPU','No external factory validation']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--category',required=True);p.add_argument('--output',default='artifacts');p.add_argument('--mode',choices=['synthetic_fixture','mvtec'],required=True);p.add_argument('--size',type=int,default=128);p.add_argument('--memory',type=int,default=1000);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);model=fit(list((Path(a.category)/'train/good').glob('*.png')),a.mode,a.size,a.memory)
    report=evaluate(model,a.category);model['report']=report;joblib.dump(model,out/'detector.joblib');(out/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='raw'}))
