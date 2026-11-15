"""Independent MIDI parsing and token edge cases required by the benchmark."""
import tempfile
from pathlib import Path
from midi_tokens import *

def main():
    cases=[([],12.5),([dict(pitch=60,onset=2.123,offset=2.911),dict(pitch=60,onset=2.911,offset=3.081),dict(pitch=127,onset=130.011,offset=131.004)],150.08),([dict(pitch=0,onset=0.,offset=.03)],.06)]
    results=[]
    for notes,duration in cases:
        report=roundtrip(notes,duration)
        assert abs(decode(encode(notes,duration))[1]-duration)<=.005001
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'test.mid';save_midi(notes,path);other=load_midi(path)
            assert len(other)==len(notes)
            for x,y in zip(notes,other):
                assert x['pitch']==y['pitch']
                assert abs(x['onset']-y['onset'])<=.000501
                assert abs(x['offset']-y['offset'])<=.000501
        results.append(report)
    return dict(passed=True,cases=results,midi_time_tolerance_seconds=.000501,token_time_tolerance_seconds=.010001,vocabulary_size=len(VOCAB),version=VERSION)

if __name__=='__main__':
    import json
    print(json.dumps(main(),indent=2))
