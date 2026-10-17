import argparse,hashlib,json,zipfile,tarfile
from pathlib import Path
import httpx

def download(url,path,checksum=None,algorithm='sha256'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temporary=path.with_suffix(path.suffix+'.partial')
        with httpx.stream('GET',url,follow_redirects=True,timeout=120) as response:
            response.raise_for_status()
            with temporary.open('wb') as f:
                for chunk in response.iter_bytes():f.write(chunk)
        temporary.replace(path)
    with path.open('rb') as f:digest=hashlib.file_digest(f,algorithm).hexdigest()
    if checksum and checksum!=digest:raise ValueError(f'Checksum mismatch: {path}; remove and retry')
    print(path,digest,flush=True)
    return digest

p=argparse.ArgumentParser();p.add_argument('--output',default='data/visa');a=p.parse_args();root=Path(a.output)
# Zou et al. ECCV 2022; VisA is CC BY 4.0.
download('https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar',root/'visa.tar','2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362')
download('https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv',root/'1cls.csv','a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995')
with tarfile.open(root/'visa.tar') as archive:
    members=[m for m in archive.getmembers() if m.isfile() and any(m.name.startswith(c+'/') for c in ['cashew','capsules','pcb1'])]
    archive.extractall(root/'images',members=members,filter='data')
