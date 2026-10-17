import io,base64,hashlib,json,os,sqlite3,time,uuid
from pathlib import Path
import joblib,httpx
from PIL import Image,UnidentifiedImageError
from fastapi import FastAPI,HTTPException,UploadFile,File,Header
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field,ConfigDict
from detector import predict
ROOT=Path(__file__).parent;app=FastAPI(title='InspectEdge research station');Image.MAX_IMAGE_PIXELS=16_000_000

def db():
    p=Path(os.environ.get('INSPECT_DB',str(ROOT/'artifacts/edge.sqlite')));p.parent.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(p,timeout=10);c.row_factory=sqlite3.Row;c.execute('PRAGMA journal_mode=WAL');c.execute('PRAGMA synchronous=FULL')
    c.execute('CREATE TABLE IF NOT EXISTS inspections(id TEXT PRIMARY KEY,payload TEXT NOT NULL,synced INTEGER NOT NULL DEFAULT 0)');c.commit();return c

def artifact():
    p=Path(os.environ.get('INSPECT_MODEL',str(ROOT/'artifacts/detector.joblib')))
    if not p.is_file():raise HTTPException(503,'Fit a normal-image baseline first; no simulated detector output is served.')
    return joblib.load(p),hashlib.sha256(p.read_bytes()).hexdigest()

def key(value):
    if value!=os.environ.get('INSPECT_KEY','local-inspect-only'):raise HTTPException(403,'Local station key required')

@app.get('/')
def home():return FileResponse(ROOT/'index.html')
@app.get('/status')
def status():
    try:a,h=artifact();return {'ready':True,'mode':a['mode'],'sha256':h,'report':a.get('report')}
    except HTTPException as e:return {'ready':False,'reason':e.detail}
@app.post('/inspect')
async def inspect(file:UploadFile=File(...),x_station_key:str=Header(default='')):
    key(x_station_key);a,h=artifact();raw=await file.read(5*1024*1024+1)
    if len(raw)>5*1024*1024:raise HTTPException(413,'Maximum 5 MiB')
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in {'PNG','JPEG'} or im.width*im.height>16_000_000:raise ValueError('Invalid image')
            result=predict(a,im.convert('RGB'))
    except (ValueError,OSError,UnidentifiedImageError,Image.DecompressionBombError):raise HTTPException(422,'Invalid JPEG/PNG')
    buf=io.BytesIO();result.pop('heatmap').save(buf,format='PNG');result['heatmap']=base64.b64encode(buf.getvalue()).decode()
    result.update({'id':str(uuid.uuid4()),'model_sha256':h,'captured_at':time.time(),'image_sha256':hashlib.sha256(raw).hexdigest()})
    c=db()
    try:c.execute('INSERT INTO inspections(id,payload) VALUES (?,?)',(result['id'],json.dumps(result,sort_keys=True)));c.commit()
    finally:c.close()
    return result
@app.get('/history')
def history():
    c=db()
    try:return [{**json.loads(r['payload']),'synced':bool(r['synced'])} for r in c.execute('SELECT * FROM inspections ORDER BY rowid DESC LIMIT 100')]
    finally:c.close()
class Record(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    id:str=Field(min_length=1,max_length=100)
    score:float=Field(ge=0)
    threshold:float=Field(ge=0)
    anomaly:bool
    inference_ms:float=Field(ge=0)
    heatmap:str=Field(max_length=1_000_000)
    mode:str=Field(pattern='^(mvtec|visa|synthetic_fixture)$')
    model_sha256:str=Field(pattern='^[0-9a-f]{64}$')
    image_sha256:str=Field(pattern='^[0-9a-f]{64}$')
    captured_at:float=Field(ge=0)
class Batch(BaseModel):records:list[Record]=Field(min_length=1,max_length=20)
@app.post('/ingest')
def ingest(batch:Batch,x_station_key:str=Header(default='')):
    key(x_station_key);c=db();accepted=[]
    try:
        c.execute('BEGIN IMMEDIATE')
        for record in batch.records:
            r=record.model_dump();payload=json.dumps(r,sort_keys=True);old=c.execute('SELECT payload FROM inspections WHERE id=?',(r['id'],)).fetchone()
            if old and old['payload']!=payload:raise HTTPException(409,'ID/content conflict')
            c.execute('INSERT OR IGNORE INTO inspections VALUES (?,?,1)',(r['id'],payload));accepted.append(r['id'])
        c.commit();return {'accepted':accepted}
    finally:c.close()
@app.post('/sync')
def sync(x_station_key:str=Header(default='')):
    key(x_station_key);url=os.environ.get('INSPECT_CENTRAL_URL')
    if not url:raise HTTPException(503,'INSPECT_CENTRAL_URL is not configured; inspections remain durable locally.')
    c=db()
    try:
        records=[json.loads(r['payload']) for r in c.execute('SELECT payload FROM inspections WHERE synced=0 LIMIT 20')]
        if not records:return {'acknowledged':0}
        try:
            r=httpx.post(url+'/ingest',json={'records':records},headers={'X-Station-Key':x_station_key},timeout=5);r.raise_for_status()
        except httpx.HTTPError:raise HTTPException(503,'Central unavailable; pending inspections are retained.')
        ids=r.json()['accepted'];expected={x['id'] for x in records}
        if not set(ids)<=expected:raise HTTPException(502,'Invalid central acknowledgement')
        c.executemany('UPDATE inspections SET synced=1 WHERE id=?',[(i,) for i in ids]);c.commit();return {'acknowledged':len(ids)}
    finally:c.close()
