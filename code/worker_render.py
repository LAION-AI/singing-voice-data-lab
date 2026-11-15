"""Create the same matched-level full and preview listening files as the lab."""
import argparse
from common import BENCH,samples
from render_ensemble import normalize,track

def main():
    p=argparse.ArgumentParser();p.add_argument('--pipeline',default='rosvot_procedural_segmented');a=p.parse_args()
    for item in samples():
        sid=item['sample_id'];normalize(sid,'lead',[BENCH/sid/'lead_vocal.flac'])
        track(sid,pipeline=a.pipeline,retain_raw=True)

if __name__=='__main__':main()
