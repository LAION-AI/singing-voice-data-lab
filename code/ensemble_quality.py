"""Frame diagnostics including unmatched activity; still a shared, noisy reference."""
import numpy as np
from common import ROOT,BENCH,read,write,samples
from ensemble_notes import EVIDENCE,raster


def main():
    result=read(EVIDENCE/'ensemble_results.json')
    selection=set(result['search_split']['selection_ids']);report=set(result['search_split']['report_ids'])
    pipelines=['rosvot','procedural']+[x['pipeline'] for x in result['summary'] if x['pipeline'].startswith('rosvot_procedural_')]
    rows=[]
    for item in samples():
        sid=item['sample_id'];d=BENCH/sid
        with np.load(d/'pitch/f0.npz') as f:t,hz=f['time_seconds'],f['f0_hz']
        voiced=hz>0;ref=np.full(len(t),np.nan);ref[voiced]=69+12*np.log2(hz[voiced]/440)
        for pipeline in pipelines:
            notes=read(d/'note_events'/f'{pipeline}.json')['notes'];seq,count=raster(notes,t)
            predicted=count>0;used=voiced&np.isfinite(seq)
            close=int((np.abs(seq[used]-ref[used])<=.5).sum())
            durations=[x['offset']-x['onset'] for x in notes]
            rows.append(dict(sample=sid,pipeline=pipeline,
                close50_f1=2*close/max(1,int(voiced.sum()+predicted.sum())),
                unvoiced_predicted_share=float((predicted&~voiced).sum()/max(1,int(predicted.sum()))),
                overlapping_predicted_share=float((count>1).sum()/max(1,int(predicted.sum()))),
                notes=len(notes),shorter_than_50ms=sum(x<.05-1e-9 for x in durations),
                median_note_duration=float(np.median(durations)),
                coverage=float((voiced&predicted).sum()/max(1,int(voiced.sum()))),
                within50=close/max(1,int(voiced.sum()))))
    summary=[]
    for pipeline in pipelines:
        for cohort,ids in [('all75',None),('selection15',selection),('report60',report)]:
            z=[x for x in rows if x['pipeline']==pipeline and (ids is None or x['sample'] in ids)]
            summary.append(dict(pipeline=pipeline,cohort=cohort,tracks=len(z),
                total_notes=sum(x['notes'] for x in z),shorter_than_50ms=sum(x['shorter_than_50ms'] for x in z),
                **{key:float(np.median([x[key] for x in z])) for key in ['close50_f1','unvoiced_predicted_share','overlapping_predicted_share','median_note_duration','coverage','within50']}))
    write(EVIDENCE/'quality_diagnostics.json',dict(summary=summary,tracks=rows,
        definition='C=close single-note frames; V=reference-voiced frames; N=any predicted-note frames. Frame F1 proxy=2C/(V+N). Reference-unvoiced activity is not verified false-positive audio.',
        caveat='Internal exploratory comparison against shared RMVPE, not human note ground truth.'))
    print('ENSEMBLE_QUALITY_DIAGNOSTICS_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
