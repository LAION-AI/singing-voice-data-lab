"""Keep every ROSVOT boundary; use strong procedural support to replace pitch.

No neural inference, pitch-reference tuning or gap filling. The fixed rule is
the one used in the initial offline proposal accepted by the user.
"""
import copy
import json
import time
from pathlib import Path
import numpy as np
from common import ROOT, BENCH, read, write, sha, samples
from midi_tokens import roundtrip, save_midi, encode, name

PIPELINE = 'rosvot_procedural_ensemble'
EVIDENCE = ROOT / 'evidence/v4_ensemble'
RULE = dict(version=1,minimum_support_seconds=.12,minimum_note_coverage=.6,
            minimum_modal_share=.7,frame_seconds=.01,
            boundary_source='rosvot',pitch_support_source='procedural',
            missing_or_ambiguous_support='keep original ROSVOT pitch',
            gap_filling=False,pitch_reference_used_for_selection=False)


def raster(notes, times):
    seq = np.full(len(times), np.nan)
    count = np.zeros(len(times), dtype=int)
    for note in notes:
        mask = (times >= note['onset']) & (times < note['offset'])
        seq[mask] = note['pitch']
        count[mask] += 1
    seq[count > 1] = np.nan
    return seq, count


def combine(notes, procedural, times):
    support, _ = raster(procedural, times)
    output = copy.deepcopy(notes)
    for note in output:
        mask = (times >= note['onset']) & (times < note['offset'])
        values = support[mask]
        values = values[np.isfinite(values)]
        evidence = dict(original_rosvot_pitch=note['pitch'],changed=False,
                        support_seconds=len(values) * RULE['frame_seconds'],
                        note_coverage=len(values) / max(1, int(mask.sum())),
                        procedural_modal_pitch=None,modal_share=None,
                        decision='keep: insufficient unambiguous procedural coverage')
        if len(values) >= 12 and evidence['note_coverage'] >= RULE['minimum_note_coverage']:
            pitches, counts = np.unique(values.astype(int), return_counts=True)
            index = int(counts.argmax())
            modal, share = int(pitches[index]), float(counts[index] / len(values))
            evidence.update(procedural_modal_pitch=modal,modal_share=share)
            if share >= RULE['minimum_modal_share']:
                evidence['changed'] = modal != note['pitch']
                evidence['decision'] = 'replace pitch' if evidence['changed'] else 'keep: supported agreement'
                note['pitch'] = modal
            else:
                evidence['decision'] = 'keep: procedural pitches disagree within this note'
        note['pitch_name'] = name(note['pitch'])
        note['ensemble'] = evidence
    assert len(output) == len(notes)
    assert all((a['onset'],a['offset']) == (b['onset'],b['offset']) for a,b in zip(notes,output))
    return output


def metrics(sid, notes, f0):
    times, hz = f0['time_seconds'], f0['f0_hz']
    voiced = hz > 0
    reference = np.full(len(times), np.nan)
    reference[voiced] = 69 + 12 * np.log2(hz[voiced] / 440)
    seq, count = raster(notes, times)
    used = voiced & np.isfinite(seq)
    error = np.abs(seq[used] - reference[used])
    denominator = max(1, int(voiced.sum()))
    correlation = (float(np.corrcoef(seq[used],reference[used])[0,1])
                   if used.sum()>2 and np.std(seq[used])>1e-8 and np.std(reference[used])>1e-8 else None)
    return dict(sample=sid,pipeline=PIPELINE,pearson=correlation,
                coverage=float((voiced & (count>0)).sum()/denominator),
                within_50_cents_of_all_voiced=float((error<=.5).sum()/denominator),
                within_100_cents_of_all_voiced=float((error<=1).sum()/denominator),
                median_abs_cents=float(np.median(error)*100) if len(error) else None,
                note_count=len(notes))


def main():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    rows, changes, source_hashes = [], [], {}
    started=time.perf_counter()
    for item in samples():
        sid=item['sample_id'];directory=BENCH/sid
        ros_path=directory/'note_events/rosvot.json'
        proc_path=directory/'note_events/procedural.json'
        original=read(ros_path);procedural=read(proc_path)
        f0=np.load(directory/'pitch/f0.npz')
        notes=combine(original['notes'],procedural['notes'],f0['time_seconds'])
        report=roundtrip(notes,original['track_duration_seconds'])
        # The unchanged boundaries are exact floats, not the quantized MIDI grid.
        changes.append(dict(sample=sid,notes=len(notes),changed=sum(n['ensemble']['changed'] for n in notes)))
        sources={str(p.relative_to(BENCH)):sha(p) for p in [ros_path,proc_path]}
        source_hashes.update(sources)
        write(directory/'note_events'/f'{PIPELINE}.json',dict(
            schema='singing-note-events-1.0',pipeline=PIPELINE,
            track_duration_seconds=original['track_duration_seconds'],notes=notes,
            roundtrip=report,provenance=dict(rule=RULE,source_sha256=sources,
                original_boundaries_and_note_count_preserved=True,
                confidence='Heuristic support statistics are not calibrated probabilities.',
                evaluation_caveat='Procedural uses the same RMVPE pitch as the automatic evaluation reference.')))
        save_midi(notes,directory/'midi'/f'{PIPELINE}.mid')
        (directory/'flattened_midi'/f'{PIPELINE}.txt').write_text(encode(notes,original['track_duration_seconds']))
        write(directory/'status'/f'{PIPELINE}.json',dict(status='complete',notes=len(notes),changed_notes=changes[-1]['changed'],rule=RULE))
        row=metrics(sid,notes,f0);row['roundtrip']=report;rows.append(row)
    keys=['pearson','coverage','within_50_cents_of_all_voiced','within_100_cents_of_all_voiced','median_abs_cents']
    summary=dict(pipeline=PIPELINE,tracks=len(rows),**{
        key:float(np.median([row[key] for row in rows if row[key] is not None])) for key in keys})
    prior=read(ROOT/'evidence/v3/extension_results.json')
    result={**prior,'created_unix':time.time(),'summary':prior['summary']+[summary],
            'tracks':prior['tracks']+rows,'ensemble':dict(rule=RULE,per_track=changes,
                changed_notes=sum(x['changed'] for x in changes),total_notes=sum(x['notes'] for x in changes),
                build_cpu_wall_seconds=time.perf_counter()-started,
                source_sha256=source_hashes,listening_score=None,
                original_six_pipeline_results_unchanged=True)}
    write(EVIDENCE/'ensemble_results.json',result)
    assert result['ensemble']['changed_notes']==891 and result['ensemble']['total_notes']==22399
    assert all(sha(BENCH/path)==value for path,value in source_hashes.items())
    print('ENSEMBLE_NOTES_COMPLETE',json.dumps(dict(summary=summary,changed=result['ensemble']['changed_notes'],total=result['ensemble']['total_notes'])),flush=True)


if __name__=='__main__':
    main()
