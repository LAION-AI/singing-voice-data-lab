"""Windowing and padding regression with an identity predictor, not neural inference."""
import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import worker_stems

class Identity:
    def __call__(self,x):return x[:,None]

class TestOverlapAdd(unittest.TestCase):
    def test_short_tail_and_window_batches_preserve_stereo(self):
        sr=44100;t=np.arange(23*sr)/sr
        audio=np.stack([.1*np.sin(2*np.pi*220*t),.07*np.sin(2*np.pi*331*t)],axis=1).astype(np.float32)
        outputs=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for batch in [1,2]:
                bench=root/str(batch);d=bench/'example';d.mkdir(parents=True);(d/'preview').mkdir()
                sf.write(d/'source.wav',audio,sr,subtype='FLOAT')
                item=dict(sample_id='example',source_audio='source.wav')
                with patch.object(sys,'argv',['worker_stems.py','--device','cpu','--batch-size',str(batch)]),\
                     patch.multiple(worker_stems,ROOT=root,BENCH=bench,samples=lambda ids:[dict(item)],
                         load_model=lambda device:(Identity(),dict(source_checkpoint_sha256='identity-test')),
                         complete=lambda *args,**kwargs:None,metric=lambda *args,**kwargs:None):
                    worker_stems.main()
                decoded,rate=sf.read(d/'lead_vocal.flac',dtype='float32');self.assertEqual(rate,sr)
                self.assertEqual(decoded.shape,audio.shape)
                self.assertLess(float(np.max(np.abs(decoded-audio))),5e-7);outputs.append(decoded)
        self.assertLess(float(np.max(np.abs(outputs[0]-outputs[1]))),5e-7)

if __name__=='__main__':unittest.main()
