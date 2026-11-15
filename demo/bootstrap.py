"""Download only the licensed public demo, pinned and SHA-256 checked."""
import hashlib,json,os,shutil,tarfile,urllib.request
from pathlib import Path
from filelock import FileLock

def ensure_data(here):
    if os.environ.get('SVDL_DEMO_DATA'):return Path(os.environ['SVDL_DEMO_DATA'])
    cfg=json.loads((here/'dataset_config.json').read_text());cache=Path(os.environ.get('SVDL_DEMO_CACHE','/tmp/singing-voice-data-lab-public'));cache.mkdir(exist_ok=True,parents=True)
    with FileLock(str(cache/'download.lock')):
        stamp=cache/'READY.json'
        if stamp.exists() and json.loads(stamp.read_text())==cfg:return cache/'benchmark'
        archive=cache/'demo.tar'
        def valid(path):
            if not path.exists() or path.stat().st_size!=cfg['bytes']:return False
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
            return h.hexdigest()==cfg['sha256']
        if not valid(archive):
            url='https://huggingface.co/datasets/'+cfg['repo']+'/resolve/'+cfg['revision']+'/'+cfg['file'];tmp=archive.with_suffix('.partial')
            with urllib.request.urlopen(url,timeout=180) as response,tmp.open('wb') as out:shutil.copyfileobj(response,out,length=8<<20)
            if not valid(tmp):raise RuntimeError('Public demo checksum mismatch')
            tmp.replace(archive)
        target=cache/'benchmark';target.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            for member in tar:
                dest=(target/member.name).resolve()
                if not dest.is_relative_to(target.resolve()) or member.issym() or member.islnk():raise RuntimeError('Unsafe demo TAR member')
                if member.isfile():
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with dest.open('wb') as out:shutil.copyfileobj(tar.extractfile(member),out)
        stamp.write_text(json.dumps(cfg));return target
