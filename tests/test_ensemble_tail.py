"""Regression: a short last note must survive a missing aligned pitch frame."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from search_ensemble import context_arrays,candidate_notes
from midi_tokens import roundtrip

class TestUnalignedTail(unittest.TestCase):
    def test_subgrid_final_note_retains_original_pitch_and_boundaries(self):
        original=dict(pitch=67,onset=.995,offset=.999,confidence=None)
        ros=dict(track_duration_seconds=1.,notes=[original])
        f=dict(time_seconds=np.arange(100)*.01,f0_hz=np.zeros(100))
        c=context_arrays(ros,dict(notes=[]),f)
        notes=candidate_notes(c,dict(kind='segmented',support=.09,min_fragment=.05,gap_fill=True))
        self.assertEqual(len(notes),1)
        self.assertEqual((notes[0]['pitch'],notes[0]['onset'],notes[0]['offset']),(67,.995,.999))
        self.assertFalse(notes[0]['ensemble']['changed'])
        self.assertIn('no aligned Procedural frames',notes[0]['ensemble']['decision'])
        proof=roundtrip(notes,1.)
        self.assertEqual(proof['subgrid_notes'],1)
        self.assertAlmostEqual(proof['max_boundary_error_seconds'],.011)

if __name__=='__main__':unittest.main()
