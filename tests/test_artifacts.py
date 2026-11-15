import hashlib,json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from midi_tokens import roundtrip,encode,decode

class TestArtifacts(unittest.TestCase):
    def test_complete_timeline_and_colliding_grid_onsets(self):
        notes=[dict(pitch=72,onset=.001,offset=.2),dict(pitch=60,onset=.004,offset=.1)]
        self.assertTrue(roundtrip(notes,123.45)['passed'])
        decoded,end=decode(encode(notes,123.45));self.assertEqual(len(decoded),2);self.assertAlmostEqual(end,123.45)
    def test_all_bundled_shards(self):
        folder=ROOT/'models/mega53_lead_v1';manifest=json.loads((folder/'manifest.json').read_text())
        self.assertTrue(manifest['tensor_values_equal_to_source']);self.assertTrue(manifest['strict_model_load_verified'])
        for shard in manifest['shards']:
            self.assertLess(shard['bytes'],50*1024**2)
            self.assertEqual(hashlib.sha256((folder/shard['file']).read_bytes()).hexdigest(),shard['sha256'])
    def test_cli_is_available(self):
        result=subprocess.run([sys.executable,str(ROOT/'lab.py'),'--help'],capture_output=True,text=True,check=True)
        self.assertIn('extract',result.stdout);self.assertIn('annotate-existing',result.stdout)

if __name__=='__main__':unittest.main()
