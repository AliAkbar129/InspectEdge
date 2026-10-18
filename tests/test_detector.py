import numpy as np,pytest
from PIL import Image
from detector import fit,predict
from fastapi.testclient import TestClient
from server import app

def test_normal_fit_is_disjoint_and_repeatable(tmp_path):
    rng=np.random.default_rng(42);paths=[]
    for i in range(15):
        p=tmp_path/f'{i}.png';Image.fromarray(rng.integers(110,140,(64,64,3),dtype=np.uint8)).save(p);paths.append(p)
    a=fit(paths);b=fit(paths)
    assert set(a['train_files']).isdisjoint(a['validation_files'])
    assert np.allclose(a['memory'],b['memory']) and a['threshold']==b['threshold']
    r=predict(a,Image.new('RGB',(64,64),'red'))
    assert r['anomaly'] and r['mode']=='synthetic_fixture'

def test_duplicate_normal_images_rejected(tmp_path):
    paths=[]
    for i in range(10):
        p=tmp_path/f'{i}.png';Image.new('RGB',(16,16)).save(p);paths.append(p)
    with pytest.raises(ValueError):fit(paths)

def test_missing_detector_never_returns_anomaly(tmp_path,monkeypatch):
    monkeypatch.setenv('INSPECT_MODEL',str(tmp_path/'missing'));c=TestClient(app)
    assert not c.get('/status').json()['ready']
    assert c.post('/inspect',files={'file':('x.png',b'bad')},headers={'X-Station-Key':'local-inspect-only'}).status_code==503

def test_ingestion_idempotency_and_conflicting_retry(tmp_path,monkeypatch):
    monkeypatch.setenv('INSPECT_DB',str(tmp_path/'db'));c=TestClient(app);h={'X-Station-Key':'local-inspect-only'}
    record={'id':'a','score':1.,'threshold':2.,'anomaly':False,'inference_ms':1.,'heatmap':'','mode':'synthetic_fixture','model_sha256':'0'*64,'image_sha256':'0'*64,'captured_at':1.}
    assert c.post('/ingest',json={'records':[record]},headers=h).status_code==200
    assert c.post('/ingest',json={'records':[record]},headers=h).status_code==200
    assert len(c.get('/history').json())==1
    assert c.post('/ingest',json={'records':[{**record,'score':3.}]},headers=h).status_code==409


@pytest.mark.parametrize('body', [{}, [], {'accepted': 'bad'}, {'accepted': [None]}, {'accepted': ['a','a']}, {'accepted':['unknown']}])
def test_malformed_ack_preserves_inspection(tmp_path,monkeypatch,body):
    import server,json
    monkeypatch.setenv('INSPECT_DB',str(tmp_path/'db'))
    monkeypatch.setenv('INSPECT_CENTRAL_URL','http://central')
    con=server.db();con.execute('INSERT INTO inspections VALUES (?,?,0)',('a',json.dumps({'id':'a'})));con.commit();con.close()
    class Reply:
        def raise_for_status(self): pass
        def json(self): return body
    monkeypatch.setattr(server.httpx,'post',lambda *args,**kw:Reply())
    response=TestClient(app).post('/sync',headers={'X-Station-Key':'local-inspect-only'})
    assert response.status_code==502
    con=server.db()
    try:assert con.execute('SELECT synced FROM inspections').fetchone()[0]==0
    finally:con.close()
