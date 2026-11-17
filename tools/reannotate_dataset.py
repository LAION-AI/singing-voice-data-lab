"""Reannotate released TARs without unpacking individual recordings."""
import argparse,io,json,sys,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'code'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--limit-shards',type=int);a=p.parse_args()
    from bulk_annotations import Annotator
    from packed_dataset import process_task
    assert 0<=a.rank<a.world and a.world>=1
    model=Annotator(HERE/'third_party/ROSVOT',HERE/'models',batch_size=a.batch_size)
    paths=sorted((a.data/'data').rglob('*.tar'))
    if a.limit_shards:paths=paths[:a.limit_shards]
    if not paths:raise ValueError('No data/**/*.tar archives; download the public singing dataset first')
    for i,path in enumerate(paths):
        if i%a.world!=a.rank:continue
        rows=[]
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile() or member.name.count('.')!=1 or not member.name.endswith('.json'):continue
                record=json.load(archive.extractfile(member));uid=record['id']
                if record.get('schema')!='synthetic-singing-with-notes-1.0':continue
                rows.append(dict(uid=uid,audio_tar=str(path),audio_member=uid+'.flac',metadata_member=uid+'.json',
                    q12_member=uid+'.codes.npy',q32_member=uid+'.codes32.npy',
                    acoustic_member=uid+'.acoustic.json.gz',source_metadata_index=record['original_index'],
                    source_repository=record['source_repository'],source_revision=record['source_revision'],
                    source_shard=record['source_shard'],modality=record['modality'],language=record['language'],
                    alias=record['speaker_alias'],plain=record['plain_text'],instruction=record['prompt_original'],
                    training_member=record['training_selected_for_experimental_r128']))
        rel=path.relative_to(a.data/'data')
        receipt=process_task(rows,model,a.output/'note_annotations'/rel,a.output/'data'/rel)
        print(json.dumps(dict(shard=str(rel),**receipt)),flush=True)
if __name__=='__main__':main()
