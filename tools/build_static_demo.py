"""Build a static player index; keep audio in one public TAR with byte ranges."""
import argparse,hashlib,json,tarfile
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=json.loads(a.config.read_text());a.output.mkdir(exist_ok=True,parents=True);(a.output/'clips').mkdir(exist_ok=True)
    members={}
    with tarfile.open(a.archive) as tar:
        for m in tar:
            if m.isfile():members[m.name]=dict(start=m.offset_data,size=m.size,sha256=hashlib.sha256((a.data/m.name).read_bytes()).hexdigest())
    manifest=json.loads((a.data/'manifest.json').read_text())
    labels={'rosvot_procedural_segmented':'Recommended · smoothed splits + gaps','rosvot':'ROSVOT alone','procedural':'Procedural baseline','rosvot_procedural_ensemble':'Conservative pitch vote','rosvot_procedural_gapfill':'ROSVOT + gaps','rosvot_procedural_vote_gapfill':'Pitch vote + gaps'}
    for meta in manifest:
        sid=meta['sample_id'];root=a.data/sid;methods={}
        for pipeline in labels:
            methods[pipeline]=dict(notes=json.loads((root/pipeline/'notes.json').read_text())['notes'],tokens=(root/pipeline/'tokens.txt').read_text())
        with np.load(root/'f0.npz') as f:
            time,hz=f['time_seconds'],f['f0_hz'];step=max(1,len(hz)//2000);pitch=[round(float(69+12*np.log2(x/440)),3) if x>0 else None for x in hz[::step]];times=[round(float(x),3) for x in time[::step]]
        data=dict(metadata=meta,methods=methods,pitch=dict(times=times,values=pitch),members={name:info for name,info in members.items() if name.startswith(sid+'/')})
        (a.output/'clips'/(sid+'.json')).write_text(json.dumps(data,separators=(',',':'),allow_nan=False))
    (a.output/'manifest.json').write_text(json.dumps(dict(recordings=manifest,methods=labels,default='rosvot_procedural_segmented',
        archive_url='https://huggingface.co/datasets/'+config['repo']+'/resolve/'+config['revision']+'/'+config['file'],archive_sha256=config['sha256'],archive_bytes=config['bytes']),separators=(',',':')))
    print('STATIC_INDEX_BUILT',len(manifest),len(members),flush=True)

if __name__=='__main__':main()
