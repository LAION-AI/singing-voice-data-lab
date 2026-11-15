"""Apply the frozen published fusion recipes to any prepared recording set."""
import argparse,copy
from common import ROOT,BENCH,read,write,sha,samples
from ensemble_notes import combine,RULE
from search_ensemble import context,candidate_notes
from midi_tokens import roundtrip,save_midi,encode,name

RULES={
 'rosvot_procedural_ensemble':RULE,
 'rosvot_procedural_gapfill':dict(kind='gap_only',gap_fill=True),
 'rosvot_procedural_vote_gapfill':dict(kind='note_vote',support=.06,coverage=.25,modal=.5,gap_fill=True),
 'rosvot_procedural_segmented':dict(kind='segmented',support=.09,min_fragment=.05,gap_fill=True),
}
DEFAULT='rosvot_procedural_segmented'

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');p.add_argument('--pipeline',choices=['rosvot','procedural',*RULES],default=DEFAULT);a=p.parse_args()
    for item in samples(a.ids):
        sid=item['sample_id'];d=BENCH/sid;c=context(sid);duration=c['ros']['track_duration_seconds']
        for pipeline,rule in RULES.items():
            notes=(combine(c['ros']['notes'],c['proc']['notes'],c['times']) if pipeline=='rosvot_procedural_ensemble' else candidate_notes(c,rule))
            proof=roundtrip(notes,duration)
            write(d/'note_events'/f'{pipeline}.json',dict(schema='singing-note-events-1.0',pipeline=pipeline,
                track_duration_seconds=duration,notes=notes,roundtrip=proof,
                provenance=dict(rule=rule,source_sha256={str((d/'note_events'/f'{source}.json').relative_to(BENCH)):sha(d/'note_events'/f'{source}.json') for source in ['rosvot','procedural']},
                    evaluation_caveat='ROSVOT and Procedural share RMVPE; automatic agreement is not independent accuracy.')))
            save_midi(notes,d/'midi'/f'{pipeline}.mid');(d/'flattened_midi'/f'{pipeline}.txt').write_text(encode(notes,duration))
            write(d/'status'/f'{pipeline}.json',dict(status='complete',notes=len(notes),roundtrip=proof))
        selected=read(d/'note_events'/f'{a.pipeline}.json')
        write(d/'annotation.json',selected);save_midi(selected['notes'],d/'melody.mid')
        (d/'melody.tokens.txt').write_text(encode(selected['notes'],duration))
        print('ANNOTATED',sid,a.pipeline,len(selected['notes']),flush=True)

if __name__=='__main__':main()
