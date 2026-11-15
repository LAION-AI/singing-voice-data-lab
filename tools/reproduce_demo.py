"""Reproduce all four public-demo fusion results on CPU, with exact token checks."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import lab

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,help='Unpacked public demo benchmark folder; downloaded when omitted.');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.data:data=a.data.resolve()
    else:
        sys.path.insert(0,str(ROOT/'demo'));from bootstrap import ensure_data
        data=ensure_data(ROOT/'demo')
    work=a.output.resolve();lab.prepare(work)
    if (work/'benchmark/manifest.json').exists():raise RuntimeError('Use a fresh output folder')
    manifest=[]
    for source in json.loads((data/'manifest.json').read_text()):
        sid=source['sample_id'];src=data/sid;dest=work/'benchmark'/sid
        for folder in ['note_events','midi','flattened_midi','pitch','status']:(dest/folder).mkdir(parents=True,exist_ok=True)
        for pipeline in ['rosvot','procedural']:shutil.copy2(src/pipeline/'notes.json',dest/'note_events'/(pipeline+'.json'))
        (dest/'pitch/f0.npz').symlink_to(src/'f0.npz')
        metadata={**source,'source_audio':'original.flac','group':'C'}
        (dest/'processing_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');manifest.append(metadata)
    (work/'benchmark/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lab.stage(work,'worker_ensemble.py',['--pipeline',lab.DEFAULT])
    checked=0
    for item in manifest:
        sid=item['sample_id']
        for pipeline in ['rosvot_procedural_ensemble','rosvot_procedural_gapfill','rosvot_procedural_vote_gapfill','rosvot_procedural_segmented']:
            assert (work/'benchmark'/sid/'flattened_midi'/(pipeline+'.txt')).read_bytes()==(data/sid/pipeline/'tokens.txt').read_bytes()
            assert (work/'benchmark'/sid/'midi'/(pipeline+'.mid')).read_bytes()==(data/sid/pipeline/'notes.mid').read_bytes()
            checked+=1
    proof=dict(passed=True,recordings=len(manifest),exact_token_and_midi_pairs=checked,neural_inference=False)
    (work/'reproduction_verified.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('PUBLIC_DEMO_REPRODUCED',json.dumps(proof),flush=True)

if __name__=='__main__':main()
