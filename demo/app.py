"""Public, precomputed listening lab. No model inference and no API key."""
import html,json,os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote
import gradio as gr
import numpy as np
import plotly.graph_objects as go
from bootstrap import ensure_data

HERE=Path(__file__).resolve().parent;DATA=ensure_data(HERE)
DEFAULT='rosvot_procedural_segmented'
LABELS={DEFAULT:'Recommended: smoothed pitch splits + gap filling',
 'rosvot':'ROSVOT alone','procedural':'Procedural pitch-to-note baseline',
 'rosvot_procedural_ensemble':'Conservative pitch voting',
 'rosvot_procedural_gapfill':'ROSVOT + missing-note filling',
 'rosvot_procedural_vote_gapfill':'Pitch voting + missing-note filling'}

def read(p):return json.loads(p.read_text())
def url(p):return '/gradio_api/file='+quote(str(p.resolve()),safe='/')
def player(p,label):return '<div class="player"><b>'+html.escape(label)+'</b><audio controls preload="none" src="'+html.escape(url(p),quote=True)+'"></audio></div>'

@lru_cache(maxsize=400)
def load(sid,pipeline,full):
    d=DATA/sid;m=read(d/'metadata.json');n=read(d/pipeline/'notes.json');notes=n['notes'];suffix='.mp3' if full else '_preview.mp3'
    info=f"### {sid} · {m.get('genre','')}\n\nLanguage: **{m.get('language','unknown')}** · full recording: **{m['duration_seconds']:.1f} s** · source clip: `{m['source_id']}`. Every note table and token sequence covers the full recording. Audio uses the same excerpt across all methods unless full audio is enabled."
    audio=[player(d/('original.flac' if full else 'original_preview.mp3'),'Original LeVo2 recording'),player(d/('lead.mp3' if full else 'lead_preview.mp3'),'Extracted lead vocal · matched listening level')]
    for kind,label in [('epiano','Melody as electric piano'),('choir','Melody as synthetic choir'),('vocal_epiano','Lead vocal + piano'),('vocal_choir','Lead vocal + choir')]:audio.append(player(d/pipeline/(kind+suffix),label))
    figure=go.Figure();x=[];y=[];cx=[];cy=[];ox=[];oy=[];gx=[];gy=[]
    for note in notes:
        tx=[note['onset'],note['offset'],None];x+=tx;y+=[note['pitch'],note['pitch'],None];e=note.get('ensemble',{})
        if e.get('changed'):cx+=tx;cy+=[note['pitch'],note['pitch'],None];ox+=tx;oy+=[e['original_rosvot_pitch'],e['original_rosvot_pitch'],None]
        if e.get('added'):gx+=tx;gy+=[note['pitch'],note['pitch'],None]
    figure.add_trace(go.Scattergl(x=x,y=y,mode='lines',name='Predicted notes',line=dict(width=6,color='#6765d7')))
    figure.add_trace(go.Scattergl(x=ox,y=oy,mode='lines',name='Original ROSVOT pitch',line=dict(width=2,color='#77828e',dash='dash')))
    figure.add_trace(go.Scattergl(x=cx,y=cy,mode='lines',name='Replaced pitch',line=dict(width=6,color='#c64363')))
    figure.add_trace(go.Scattergl(x=gx,y=gy,mode='lines',name='Added gap note',line=dict(width=6,color='#208264')))
    with np.load(d/'f0.npz') as f:
        t=f['time_seconds'];hz=f['f0_hz'];pitch=np.full(len(hz),np.nan);v=hz>0;pitch[v]=69+12*np.log2(hz[v]/440);step=max(1,len(hz)//4000)
        figure.add_trace(go.Scattergl(x=t[::step],y=pitch[::step],mode='lines',name='Automatic continuous pitch (RMVPE)',line=dict(width=1,color='#d89529')))
    figure.update_layout(template='plotly_white',height=370,xaxis_title='Seconds from start of full recording',yaxis_title='MIDI pitch',margin=dict(l=40,r=15,t=25,b=35))
    table=[[z['pitch'],z.get('pitch_name'),round(z['onset'],4),round(z['offset'],4),round(z['offset']-z['onset'],4)] for z in notes]
    paths=[d/pipeline/'notes.mid',d/pipeline/'tokens.txt',d/pipeline/'notes.json',d/'f0.npz',d/'metadata.json',d/'original.flac',d/'lead_vocal.flac']
    links='<div class="downloads">'+''.join('<a download href="'+html.escape(url(p),quote=True)+'">'+html.escape(p.name)+'</a> &nbsp; ' for p in paths)+'</div>'
    changed=sum(bool(z.get('ensemble',{}).get('changed')) for z in notes);added=sum(bool(z.get('ensemble',{}).get('added')) for z in notes)
    explanation=f'**{len(notes)} notes · {changed} replaced-pitch notes/segments · {added} added gap notes.** Red marks replaced pitches, grey their original ROSVOT pitch, green added notes. Smoothing removes fragments below 50 ms inside each parent note; it does not prove that every remaining boundary is musically correct.'
    return [info,*audio,figure,table,(d/pipeline/'tokens.txt').read_text(),links,explanation]

CSS='.gradio-container{max-width:1280px!important;margin:auto}.player{padding:14px;background:#f2f4fa;border-radius:12px;margin-bottom:8px}audio{width:100%;margin-top:8px}.downloads{line-height:2}footer{display:none!important}'
JS='''() => {function a(root=document){let out=[...root.querySelectorAll('audio')];root.querySelectorAll('*').forEach(e=>{if(e.shadowRoot)out.push(...a(e.shadowRoot))});return out}const seen=new WeakSet();function bind(){a().forEach(x=>{if(!seen.has(x)){seen.add(x);x.addEventListener('play',()=>a().forEach(y=>{if(x!==y)y.pause()}))}})}new MutationObserver(bind).observe(document.documentElement,{childList:true,subtree:true});setInterval(bind,1000);bind()}'''
manifest=read(DATA/'manifest.json')
with gr.Blocks(title='Singing Voice Data Lab',theme=gr.themes.Soft(primary_hue='indigo'),css=CSS,js=JS) as demo:
    gr.Markdown('# Singing Voice Data Lab\n### From a song to a lead voice, melody notes and MIDI tokens.\n[Source code & reproducible pipeline](https://github.com/LAION-AI/singing-voice-data-lab) · [Public demo data](https://huggingface.co/datasets/laion/singing-voice-data-lab-demo) · [Source: LAION Singing Voice 1K](https://huggingface.co/datasets/laion/singing-voice-1k)')
    gr.Markdown('**Start with the selected smoothed ROSVOT + Procedural ensemble.** In the 75-recording exploratory benchmark it reached **98.9% voiced coverage**, **67.2% within half a semitone**, and **25.4 cents conditional pitch difference**. It combines ROSVOT note timing with stable Procedural pitch segments and gap filling. These numbers use a shared automatic RMVPE reference, not human-verified notes; they do not establish a universal quality winner.\n\nThis public player uses **25 licensed LeVo2 recordings** from Singing Voice 1K. Pick a recording, listen to the extracted voice and piano, then compare methods. The original broader benchmark includes access-controlled sources; its aggregate results are in the GitHub documentation.')
    with gr.Row():
        recording=gr.Dropdown(choices=[(x['sample_id']+' · '+str(x.get('genre','')),x['sample_id']) for x in manifest],value=manifest[0]['sample_id'],label='Recording')
        pipeline=gr.Dropdown(choices=[(label,key) for key,label in LABELS.items()],value=DEFAULT,label='Melody method')
        full=gr.Checkbox(value=False,label='Play full-length audio')
    info=gr.Markdown()
    with gr.Row():original=gr.HTML();lead=gr.HTML()
    with gr.Row():piano=gr.HTML();choir=gr.HTML()
    with gr.Row():mixp=gr.HTML();mixc=gr.HTML()
    explanation=gr.Markdown();plot=gr.Plot()
    with gr.Accordion('Inspect all notes, flattened MIDI tokens and downloads',open=False):
        notes=gr.Dataframe(headers=['MIDI pitch','Pitch name','Onset (s)','Offset (s)','Duration (s)'],interactive=False)
        tokens=gr.Code(label='Complete reversible MIDI tokens · 10-ms grid',language=None,lines=14,max_lines=24)
        downloads=gr.HTML()
    with gr.Accordion('How to read the comparison',open=False):
        gr.Markdown('**Coverage** asks whether a predicted note exists during reference-voiced time; a wrong note still counts. **Within 50 cents** additionally requires one unambiguous note within half a semitone of the reference, with missing notes earning no credit. **Pitch difference** is conditional on jointly covered, unambiguous frames, so it excludes missing material. **RMVPE** estimates continuous pitch automatically and is also used by the source pipelines; this creates evaluation bias.\n\nThe selected split rule needs 90 ms of stable Procedural pitch, merges sub-50-ms fragments within each ROSVOT parent note, and fills uncovered gaps with Procedural fragments of at least 90 ms. It can create more note boundaries. Compare the overlays to decide whether those boundaries sound useful. No human or machine listening score is claimed for the ensembles. [Full methods, alternatives and limitations](https://github.com/LAION-AI/singing-voice-data-lab#results-and-limits).')
    gr.Markdown('Data: CC BY 4.0, attributed to LAION Singing Voice 1K; changes include extraction, transcription and listening-level normalization. Application code: MIT. The SoundFont and upstream model licences retain their own terms. All audio is precomputed; selecting a method runs no neural inference.')
    outputs=[info,original,lead,piano,choir,mixp,mixc,plot,notes,tokens,downloads,explanation]
    demo.load(load,[recording,pipeline,full],outputs)
    recording.change(load,[recording,pipeline,full],outputs,api_name='sample',trigger_mode='always_last')
    pipeline.change(load,[recording,pipeline,full],outputs,api_name=False,trigger_mode='always_last')
    full.change(load,[recording,pipeline,full],outputs,api_name=False,trigger_mode='always_last')

if __name__=='__main__':
    demo.queue(default_concurrency_limit=4).launch(server_name='0.0.0.0',server_port=int(os.environ.get('PORT','7860')),allowed_paths=[str(DATA.resolve())])
