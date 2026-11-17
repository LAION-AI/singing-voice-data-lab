"""SVAMIDI v1.0: reversible 10-ms event grid, velocity fixed at 90.

Vocabulary: BOS, EOS, NOTE_ON_0..127, NOTE_OFF_0..127,
TIME_SHIFT_1..100. Off events precede on events at the same tick.
Terminal time shifts retain the full recording's trailing rest.
Original floating-point boundaries are retained in separate note JSON.
"""
import json, math, re
from pathlib import Path

RESOLUTION=.01
VERSION='svamidi-1.0'
VOCAB=['<BOS>','<EOS>']+[f'<NOTE_{kind}_{p}>' for kind in ['ON','OFF'] for p in range(128)]+[f'<TIME_SHIFT_{n}>' for n in range(1,101)]
def q(t):return int(math.floor(float(t)/RESOLUTION+.5))
def encode(notes,duration):
    ev=[]
    for n in notes:
        a=q(n['onset']);b=max(a+1,q(n['offset']));p=int(n['pitch'])
        assert 0<=p<=127
        ev.extend([(a,1,p),(b,0,p)])
    tokens=['<BOS>'];now=0
    for tick,kind,p in sorted(ev):
        while tick>now:
            delta=min(100,tick-now);tokens.append(f'<TIME_SHIFT_{delta}>');now+=delta
        tokens.append(f'<NOTE_{"ON" if kind else "OFF"}_{p}>')
    end=max(now,q(duration))
    while end>now:
        delta=min(100,end-now);tokens.append(f'<TIME_SHIFT_{delta}>');now+=delta
    return '\n'.join(tokens+['<EOS>'])+'\n'
def decode(text):
    now=0;active={};notes=[]
    for tok in text.split():
        if tok in ['<BOS>','<EOS>']:continue
        m=re.fullmatch(r'<TIME_SHIFT_(\d+)>',tok)
        if m:now+=int(m[1]);continue
        m=re.fullmatch(r'<NOTE_(ON|OFF)_(\d+)>',tok)
        if not m:raise ValueError('Unknown token '+tok)
        kind,p=m[1],int(m[2])
        if kind=='ON':active.setdefault(p,[]).append(now)
        else:
            if not active.get(p):raise ValueError('Unpaired note off')
            start=active[p].pop(0);notes.append(dict(pitch=p,onset=start*RESOLUTION,offset=now*RESOLUTION))
    if any(active.values()):raise ValueError('Unclosed notes')
    return sorted(notes,key=lambda n:(n['onset'],n['pitch'],n['offset'])),now*RESOLUTION
def roundtrip(notes,duration):
    decoded,end=decode(encode(notes,duration))
    # Distinct sub-grid onsets can collapse to the same 10-ms tick. Compare
    # notes in the serialized grid's order, not their former float order.
    order=lambda n:(q(n['onset']),n['pitch'],max(q(n['onset'])+1,q(n['offset'])))
    decoded=sorted(decoded,key=order);expected=sorted(notes,key=order)
    assert len(decoded)==len(expected)
    error=0.
    for a,b in zip(expected,decoded):
        assert a['pitch']==b['pitch']
        error=max(error,abs(a['onset']-b['onset']),abs(a['offset']-b['offset']))
    assert error<=.010000001,(error,notes)
    return dict(passed=True,notes=len(notes),max_boundary_error_seconds=error,quantization_seconds=RESOLUTION,terminal_time=end)
def midi_bytes(notes):
    """Serialize standard MIDI in memory, for direct packed-dataset writes."""
    import io,mido
    mid=mido.MidiFile(type=0,ticks_per_beat=500);tr=mido.MidiTrack();mid.tracks.append(tr)
    tr.append(mido.MetaMessage('set_tempo',tempo=500000,time=0));ev=[]
    for n in notes:
        a=round(n['onset']*1000);b=max(a+1,round(n['offset']*1000))
        ev.extend([(a,1,n['pitch']),(b,0,n['pitch'])])
    now=0
    for t,on,p in sorted(ev):
        tr.append(mido.Message('note_on' if on else 'note_off',note=int(p),velocity=90 if on else 0,time=t-now));now=t
    stream=io.BytesIO();mid.save(file=stream);return stream.getvalue()
def save_midi(notes,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(midi_bytes(notes))
def load_midi(path):
    import mido
    now=0.;active={};notes=[]
    for msg in mido.MidiFile(path):
        now+=msg.time
        if msg.type=='note_on' and msg.velocity>0:active.setdefault(msg.note,[]).append(now)
        elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
            if active.get(msg.note):
                a=active[msg.note].pop(0)
                if now>a:notes.append(dict(pitch=msg.note,onset=a,offset=now,confidence=None))
    return sorted(notes,key=lambda n:(n['onset'],n['pitch']))
def name(p):return ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][int(p)%12]+str(int(p)//12-1)

