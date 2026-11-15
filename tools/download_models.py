"""Fetch pinned external checkpoints; bundled separator weights need no download."""
import argparse,hashlib,json,os,tarfile,urllib.request,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];MODELS=ROOT/'models'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def download(url,path,digest=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and (not digest or sha(path)==digest):return
    tmp=path.with_suffix(path.suffix+'.partial')
    with urllib.request.urlopen(url,timeout=180) as response,tmp.open('wb') as out:
        while chunk:=response.read(8<<20):out.write(chunk)
    if digest and sha(tmp)!=digest:tmp.unlink();raise RuntimeError('Checkpoint checksum mismatch')
    tmp.replace(path)

def unpack(archive,target):
    target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            dest=(target/entry.filename).resolve()
            if not dest.is_relative_to(target.resolve()) or (entry.external_attr>>16)&0o170000==0o120000:raise RuntimeError('Unsafe archive member')
            z.extract(entry,target)

def source(name,repo,commit):
    dest=ROOT/'third_party'/name
    if dest.exists():return
    archive=MODELS/'cache'/(name+'-source.tar.gz');download('https://codeload.github.com/'+repo+'/tar.gz/'+commit,archive)
    dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as t:
        for m in t:
            parts=Path(m.name).parts[1:]
            if not parts:continue
            path=(dest/Path(*parts)).resolve()
            if not path.is_relative_to(dest.resolve()) or m.issym() or m.islnk():raise RuntimeError('Unsafe source archive')
            if m.isfile():
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('wb') as out:out.write(t.extractfile(m).read())

def main():
    p=argparse.ArgumentParser();p.add_argument('--extras',nargs='*',default=[],choices=['game','tifa','stars','parakeet','nemotron','lfm3','moss_audio','muscriptor_large','muscriptor_medium','music_whisper']);a=p.parse_args()
    pins=json.loads((MODELS/'upstream_sources.json').read_text());checks={Path(x['file']).name:x['sha256'] for x in json.loads((MODELS/'download_checksums.json').read_text())}
    archive=MODELS/'cache/rosvot.zip'
    download('https://drive.usercontent.google.com/download?id=1JNtNT37KiLq9uFQqHk7JFs-3trxd3bRh&export=download&confirm=t',archive,checks['rosvot.zip'])
    unpack(archive,MODELS/'rosvot')
    from huggingface_hub import hf_hub_download,snapshot_download
    spec=pins['huggingface']['stars']
    path=hf_hub_download(spec['repo'],filename='rmvpe/model.pt',revision=spec['revision'])
    dest=MODELS/'stars/rmvpe/model.pt';dest.parent.mkdir(parents=True,exist_ok=True)
    import shutil
    shutil.copy2(path,dest)
    for extra in a.extras:
        if extra in ['game','tifa']:
            item=pins['official_release_archives'][extra];repo,tag=item['release'].split('@');name=item['asset'];target=MODELS/'cache'/name
            download(f'https://github.com/{repo}/releases/download/{tag}/{name}',target,checks[extra+'.zip']);unpack(target,MODELS/extra)
        elif extra.startswith('muscriptor_'):
            size=extra.split('_',1)[1];lock=json.loads((MODELS/'muscriptor_lock.json').read_text());spec=lock['models'][size]
            source('MuScriptor','muscriptor/muscriptor',lock['implementation']['commit'])
            snapshot_download(spec['repo'],revision=spec['revision'],local_dir=MODELS/extra)
        else:
            spec=(dict(repo='laion/music-whisper',revision='23dfa367927599130a7be9cfc33d1205d6e4b427') if extra=='music_whisper' else pins['huggingface'][extra])
            snapshot_download(spec['repo'],revision=spec['revision'],local_dir=MODELS/extra)
            if extra=='moss_audio':
                item=pins['github']['MOSS-Audio'];source('MOSS-Audio',item['repo'],item['commit'])
    print('Pinned core checkpoints ready. Extra checkpoints:',','.join(a.extras) or 'none')

if __name__=='__main__':main()
