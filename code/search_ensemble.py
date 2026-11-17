"""Explore predefined fusion rules on existing outputs, with an internal split.

Automatic RMVPE agreement is a biased proxy. The recordings were examined
previously: this is an internal exploratory split, not independent validation.
"""
import copy
import hashlib
import json
import time
import numpy as np
from common import ROOT, BENCH, read, write, samples
from ensemble_notes import EVIDENCE, PIPELINE, raster, metrics
from midi_tokens import roundtrip, save_midi, encode, name

GAP='rosvot_procedural_gapfill'
VOTE='rosvot_procedural_vote_gapfill'
SPLIT='rosvot_procedural_segmented'


def gap_notes(ros,proc,duration):
    union=[]
    for note in sorted(ros,key=lambda n:n['onset']):
        if union and note['onset']<=union[-1][1]:union[-1][1]=max(union[-1][1],note['offset'])
        else:union.append([note['onset'],note['offset']])
    gaps=[];last=0.
    for a,b in union:
        if a>last:gaps.append((last,a))
        last=max(last,b)
    if last<duration:gaps.append((last,duration))
    output=[]
    for note in proc:
        for a,b in gaps:
            if a>=note['offset']:break
            lo,hi=max(a,note['onset']),min(b,note['offset'])
            if hi-lo<.09-1e-9:continue
            output.append(dict(pitch=note['pitch'],pitch_name=name(note['pitch']),onset=lo,offset=hi,
                duration=hi-lo,confidence=None,ensemble=dict(added=True,changed=False,
                    original_rosvot_pitch=None,decision='add Procedural fragment in a ROSVOT gap',
                    minimum_fragment_seconds=.09)))
    return output


def context(sid):
    d=BENCH/sid;ros=read(d/'note_events/rosvot.json');proc=read(d/'note_events/procedural.json')
    f=np.load(d/'pitch/f0.npz')
    return context_arrays(ros,proc,f,sid)


def context_arrays(ros,proc,f,sid='packed-recording'):
    """The same frozen rules, without extracting per-recording files."""
    times=f['time_seconds'];p,pc=raster(proc['notes'],times)
    assert not (pc>1).any(), 'Unexpected Procedural overlap; do not add ambiguous gap notes'
    rr,rc=raster(ros['notes'],times);statistics=[]
    for note in ros['notes']:
        a,b=np.searchsorted(times,[note['onset'],note['offset']]);v=p[a:b];v=v[np.isfinite(v)]
        pitches,counts=np.unique(v.astype(int),return_counts=True)
        j=int(counts.argmax()) if len(counts) else None
        statistics.append(dict(a=int(a),b=int(b),support=len(v),coverage=len(v)/max(1,b-a),
            modal=int(pitches[j]) if j is not None else None,
            share=float(counts[j]/len(v)) if j is not None else 0.,
            unique=bool((counts==counts[j]).sum()==1) if j is not None else False))
    gaps=gap_notes(ros['notes'],proc['notes'],ros['track_duration_seconds'])
    gr,gc=raster(gaps,times)
    return dict(sid=sid,ros=ros,proc=proc,f=f,times=times,p=p,rr=rr,rc=rc,
                stats=statistics,gaps=gaps,gr=gr,gc=gc)


def candidate_notes(c,rule):
    output=[]
    for index,(original,s) in enumerate(zip(c['ros']['notes'],c['stats'])):
        if rule['kind']=='note_vote':
            note=copy.deepcopy(original)
            eligible=(s['support']*.01>=rule['support']-1e-9 and s['coverage']>=rule['coverage'] and
                      s['share']>=rule['modal'] and s['unique'])
            pitch=s['modal'] if eligible else note['pitch'];note['pitch']=pitch;note['pitch_name']=name(pitch)
            note['ensemble']=dict(original_rosvot_pitch=original['pitch'],original_note_index=index,
                changed=pitch!=original['pitch'],added=False,support_seconds=s['support']*.01,
                note_coverage=s['coverage'],modal_share=s['share'],procedural_modal_pitch=s['modal'],
                decision='replace pitch' if pitch!=original['pitch'] else 'keep ROSVOT pitch')
            output.append(note)
        elif rule['kind']=='segmented':
            a,b=s['a'],s['b'];seq=np.full(b-a,original['pitch'],dtype=int);local=c['p'][a:b]
            if not len(seq):
                # A sub-grid endpoint note can lie after the final F0 sample.
                # No Procedural evidence means retain the original ROSVOT note.
                note=copy.deepcopy(original)
                note['ensemble']=dict(original_rosvot_pitch=original['pitch'],original_note_index=index,
                    changed=False,added=False,split_parent=False,decision='keep ROSVOT note: no aligned Procedural frames')
                output.append(note);continue
            start=0
            while start<len(local):
                end=start+1
                while end<len(local) and ((np.isnan(local[start]) and np.isnan(local[end])) or local[end]==local[start]):end+=1
                if np.isfinite(local[start]) and (end-start)*.01>=rule['support']-1e-9:seq[start:end]=int(local[start])
                start=end
            minimum=rule.get('min_fragment',0.)
            # Absorb very short fragments into the nearest-pitch neighbour;
            # break ties by neighbour duration, then temporal order. Compare
            # real boundaries, including the original outer float times.
            while minimum:
                boundaries=[0]+[j for j in range(1,len(seq)) if seq[j]!=seq[j-1]]+[len(seq)]
                if len(boundaries)<=2:break
                regions=[]
                for k,end in zip(boundaries[:-1],boundaries[1:]):
                    lo=original['onset'] if k==0 else float(c['times'][a+k])
                    hi=original['offset'] if end==len(seq) else float(c['times'][a+end])
                    regions.append((k,end,hi-lo))
                short=[j for j,(_,_,duration) in enumerate(regions) if duration<minimum-1e-9]
                if not short:break
                j=min(short,key=lambda j:(regions[j][2],j));k,end,_=regions[j]
                neighbours=[p for p in [j-1,j+1] if 0<=p<len(regions)]
                target=min(neighbours,key=lambda p:(abs(int(seq[k])-int(seq[regions[p][0]])),-regions[p][2],p))
                seq[k:end]=seq[regions[target][0]]
            starts=[0]+[j for j in range(1,len(seq)) if seq[j]!=seq[j-1]]
            for j,k in enumerate(starts):
                end=starts[j+1] if j+1<len(starts) else len(seq)
                lo=original['onset'] if j==0 else float(c['times'][a+k])
                hi=original['offset'] if j+1==len(starts) else float(c['times'][a+end])
                if hi<=lo:continue
                note=copy.deepcopy(original);note.update(onset=lo,offset=hi,duration=hi-lo,pitch=int(seq[k]),pitch_name=name(seq[k]))
                # Original associations cover the unsplit parent, so do not
                # present them as a new sub-note word/phoneme alignment.
                note.pop('aligned_words',None);note.pop('aligned_phonemes',None)
                note.pop('alignment_association_provenance',None)
                note['ensemble']=dict(original_rosvot_pitch=original['pitch'],original_note_index=index,
                    changed=int(seq[k])!=original['pitch'],added=False,split_parent=len(starts)>1,
                    decision='stable Procedural segment' if int(seq[k])!=original['pitch'] else 'keep ROSVOT segment')
                output.append(note)
        else:
            note=copy.deepcopy(original);note['ensemble']=dict(original_rosvot_pitch=note['pitch'],
                original_note_index=index,changed=False,added=False,decision='keep original ROSVOT note')
            output.append(note)
    if rule['gap_fill']:output.extend(copy.deepcopy(c['gaps']))
    return sorted(output,key=lambda n:(n['onset'],n['pitch'],n['offset']))


def frame_scores(c,rule):
    seq=c['rr'].copy();count=c['rc'].copy()
    if rule['kind']=='note_vote':
        for s in c['stats']:
            if s['support']*.01>=rule['support']-1e-9 and s['coverage']>=rule['coverage'] and s['share']>=rule['modal'] and s['unique']:
                a,b=s['a'],s['b'];sl=seq[a:b];sl[c['rc'][a:b]==1]=s['modal']
    elif rule['kind']=='segmented':
        # Use the actual exported segmentation, rather than an optimistic
        # frame substitution that ignores short fragments/boundaries.
        seq,count=raster(candidate_notes(c,{**rule,'gap_fill':False}),c['times'])
    if rule['gap_fill']:
        use=(count==0)&np.isfinite(c['gr']);seq[use]=c['gr'][use];count[use]=1
    hz=c['f']['f0_hz'];v=hz>0;ref=np.full(len(hz),np.nan);ref[v]=69+12*np.log2(hz[v]/440)
    used=v&np.isfinite(seq);err=np.abs(seq[used]-ref[used]);den=max(1,int(v.sum()))
    predicted=count>0;close=int((err<=.5).sum());precision=close/max(1,int(predicted.sum()));recall=close/den
    return dict(sample=c['sid'],coverage=float((v&predicted).sum()/den),
        within50=float((err<=.5).sum()/den),within100=float((err<=1).sum()/den),
        median_cents=float(np.median(err)*100) if len(err) else None,
        unvoiced_predicted_share=float((predicted&~v).sum()/max(1,int(predicted.sum()))),
        close50_f1=2*precision*recall/(precision+recall) if precision+recall else 0.)


def summarize(rows,ids):
    z=[x for x in rows if x['sample'] in ids]
    return dict(tracks=len(z),**{key:float(np.median([x[key] for x in z if x[key] is not None]))
        for key in ['coverage','within50','within100','median_cents','unvoiced_predicted_share','close50_f1']},
        mean_within50=float(np.mean([x['within50'] for x in z])),mean_coverage=float(np.mean([x['coverage'] for x in z])))


def main():
    started=time.perf_counter();items=samples();ids=[x['sample_id'] for x in items]
    dev=[]
    for group in 'ABC':
        members=[x for x in ids if x.startswith(group)]
        dev.extend(sorted(members,key=lambda sid:hashlib.sha256(('ensemble-v4-selection-v1:'+sid).encode()).hexdigest())[:5])
    held=[x for x in ids if x not in dev]
    split=dict(selection_ids=dev,report_ids=held,scope='Internal exploratory split. These recordings were already examined; not independent validation.',
        selection='Maximize mean within-50-cents fraction on 15 recordings; deterministic rule-name tie break. Report remaining 60 separately.')
    write(EVIDENCE/'search_split.json',split)
    rules=[dict(id='gap_only',kind='gap_only',gap_fill=True)]
    for support in [.03,.06,.12]:
        for coverage in [.25,.5,.6]:
            for modal in [.5,.7]:
                for gap in [False,True]:
                    rules.append(dict(id=f'vote_{support}_{coverage}_{modal}_gap{int(gap)}',kind='note_vote',support=support,coverage=coverage,modal=modal,gap_fill=gap))
    for support in [.09,.12,.18,.24]:
        for minimum in [0.,.05,.08,.1]:
            for gap in [False,True]:rules.append(dict(id=f'segment_{support}_min{minimum}_gap{int(gap)}',kind='segmented',support=support,min_fragment=minimum,gap_fill=gap))
    rows={rule['id']:[] for rule in rules}
    contexts=[]
    for item in items:
        c=context(item['sample_id']);contexts.append(c)
        for rule in rules:rows[rule['id']].append(frame_scores(c,rule))
        print('SEARCH_TRACK',c['sid'],flush=True)
    summaries=[dict(rule=rule,selection=summarize(rows[rule['id']],dev),report=summarize(rows[rule['id']],held),all75=summarize(rows[rule['id']],ids)) for rule in rules]
    def best(kind):
        options=[x for x in summaries if x['rule']['kind']==kind and (kind!='segmented' or x['rule']['min_fragment']>=.05)]
        return sorted(options,key=lambda x:(-x['selection']['mean_within50'],x['rule']['id']))[0]['rule']
    selected={GAP:rules[0],VOTE:best('note_vote'),SPLIT:best('segmented')}
    write(EVIDENCE/'search_results.json',dict(split=split,candidates=summaries,tracks=rows,
        selected=selected,cpu_wall_seconds=time.perf_counter()-started,neural_inference=False,
        export_status='candidate scoring complete; exporting selected MIDI'))
    result=read(EVIDENCE/'ensemble_results.json');base=[x for x in result['summary'] if x['pipeline'] not in selected]
    tracks=[x for x in result['tracks'] if x['pipeline'] not in selected];details={}
    for pipeline,rule in selected.items():
        new_rows=[];note_counts=[]
        for c in contexts:
            notes=candidate_notes(c,rule);sid=c['sid'];d=BENCH/sid;duration=c['ros']['track_duration_seconds']
            report=roundtrip(notes,duration)
            write(d/'note_events'/f'{pipeline}.json',dict(schema='singing-note-events-1.0',pipeline=pipeline,
                track_duration_seconds=duration,notes=notes,roundtrip=report,
                provenance=dict(rule=rule,selection_split=split,source_sha256=result['ensemble']['source_sha256'],
                    boundary_policy='Original ROSVOT notes retained; only stable Procedural fragments of >=90ms added in uncovered gaps.' if rule['kind']=='gap_only' else
                    ('Original ROSVOT outer boundaries retained; internal pitch splits allowed; gap fill is separately declared.' if rule['kind']=='segmented' else
                     'Original ROSVOT note boundaries retained; per-note pitch voting; gap fill is separately declared.'),
                    evaluation_caveat='Selected using shared-RMVPE reference agreement; not independent quality validation.')))
            save_midi(notes,d/'midi'/f'{pipeline}.mid');(d/'flattened_midi'/f'{pipeline}.txt').write_text(encode(notes,duration))
            write(d/'status'/f'{pipeline}.json',dict(status='complete',notes=len(notes),rule=rule))
            m=metrics(sid,notes,c['f']);m.update(pipeline=pipeline,roundtrip=report);new_rows.append(m)
            note_counts.append(dict(sample=sid,notes=len(notes),changed=sum(n['ensemble']['changed'] for n in notes),added=sum(n['ensemble'].get('added',False) for n in notes),shorter_than_50ms=sum(n['offset']-n['onset']<.05-1e-9 for n in notes)))
        base.append(dict(pipeline=pipeline,tracks=len(new_rows),**{key:float(np.median([x[key] for x in new_rows if x[key] is not None]))
            for key in ['pearson','coverage','within_50_cents_of_all_voiced','within_100_cents_of_all_voiced','median_abs_cents']}))
        tracks.extend(new_rows);details[pipeline]=dict(rule=rule,counts=note_counts,
            selection=summarize(rows[rule['id']],dev),report=summarize(rows[rule['id']],held),all75=summarize(rows[rule['id']],ids))
    result.update(summary=base,tracks=tracks,combinations=details,search_split=split)
    write(EVIDENCE/'ensemble_results.json',result)
    write(EVIDENCE/'search_results.json',dict(split=split,candidates=summaries,tracks=rows,
        selected=selected,cpu_wall_seconds=time.perf_counter()-started,neural_inference=False))
    print('ENSEMBLE_SEARCH_COMPLETE',json.dumps(dict(selected=selected,summary=base[-3:])),flush=True)


if __name__=='__main__':main()
