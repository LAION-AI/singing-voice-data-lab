# The pipeline, step by step

## 1. Extract the lead voice

`code/worker_stems.py` loads the bundled Mega53 shared backbone and lead head through `models/mega53_lead_v1/manifest.json`. It verifies every shard's SHA-256, reconstructs the state and loads it strictly. This is the published head 27 remapped to slot 0, not a newly trained separator.

Decode stereo 44.1-kHz audio, process 20-second windows with a 10-second hop, pad the final window with zeros, and combine outputs with a normalized Hann window whose edge weight is at least 1e-5. Save the complete lead as PCM24 FLAC. GPU inference uses float16 autocast with the unchanged checkpoint values. The benchmark used batch size 1; `--separator-batch-size` optionally batches windows. There is no pitch correction, time stretching or vocal alignment.

For separation alone, use `python lab.py extract song.wav --output runs/lead-only`. This also accepts `--device cpu` for a slower CPU-only extraction path. The normal full annotation pipeline uses CUDA for RMVPE and ROSVOT.

## 2. Obtain two complementary symbolic views

`worker_f0.py` runs the pinned RMVPE model at 16 kHz, with 30-second cores and 0.5 seconds of context. It retains the complete continuous pitch and 10-ms frame times. A procedural segmenter uses a 90-ms median window, sustained pitch-change hysteresis and a 90-ms minimum note. It attempts to avoid interpreting every vibrato oscillation as a new note.

`worker_notes.py --models rosvot` runs the pinned original ROSVOT implementation on complete 30-second chunks, in batches up to four, and adds each chunk's original offset to produce a full-track timeline. ROSVOT uses its own word-boundary predictor and RMVPE; intended lyrics are not supplied as verified text. Missing model outputs remain missing rather than being filled and relabelled as raw ROSVOT.

## 3. Apply the frozen combination rules

`worker_ensemble.py` writes all four alternatives. The selected/default files are also copied to `annotation.json`, `melody.mid` and `melody.tokens.txt`. `--pipeline` can select `rosvot`, `procedural`, or any hybrid's full identifier. The fixed production rules do not rerun the parameter search on every new dataset.

See [the recipes](ENSEMBLES.md) for exact support thresholds, gap handling and smoothing. Source note files and their hashes are retained beside every combined result.

## 4. Flatten MIDI without losing the full timeline

The actual codec is `code/midi_tokens.py`. It uses 358 symbols: BOS/EOS, NOTE_ON and NOTE_OFF for each of 128 absolute pitches, and TIME_SHIFT_1…100. One time-shift unit is 10 ms. Long pauses use multiple shifts; terminal shifts retain the track tail. Off events precede on events at one tick. Velocity is fixed at 90. Original float boundaries remain in note JSON.

Example: `<BOS> <TIME_SHIFT_12> <NOTE_ON_60> <TIME_SHIFT_45> <NOTE_OFF_60> <EOS>` describes MIDI 60 from 0.12 to 0.57 seconds. Encode/decode round trips check event pairing, pitch and a maximum 10-ms timing tolerance. The ordinary MIDI file uses 1-ms ticks at a fixed 120-BPM serialization tempo; that tempo does not change audio timing.

```bash
python lab.py flatten runs/example/benchmark/song_00001/annotation.json --output melody.tokens.txt
```

The decoder is available as `decode()` in `code/midi_tokens.py` and returns both notes and the terminal time. It rejects unpaired note events. Distinct float onsets can round to the same grid tick, so integrity checks compare serialized-grid ordering.

## 5. Render for listening

`--render` uses TinySoundFont and the bundled GeneralUser GS v2.0.3 bank. Bank 0 programs 4 and 52 produce electric piano and synthetic choir. Events are scheduled at the original note times, with no latency compensation or warping. Full and shared-excerpt MP3s receive the same measured loudness treatment as the original lab: -18 LUFS target, peak-limited normalization and correction of residual deviations. These are listening copies, not a source for measuring original dynamics.

Compare the isolated voice with its piano overlay, then inspect octave errors, missing phrases, repeated notes and excessive fragmentation. The default's higher automatic agreement does not certify those properties.

## Resume or run one advanced stage

The workspace retains input hashes, status JSON, raw predictions and intermediate files. `run` rejects an existing manifest so inputs cannot be silently replaced. Run a specific worker against the same workspace to resume it:

```bash
# Put --workspace before the worker name; everything after -- is passed through.
python lab.py stage --workspace runs/example worker_notes.py -- --models rosvot
python lab.py annotate-existing --workspace runs/example

# Optional ASR after downloading its pinned weights and dependencies.
python lab.py stage --workspace runs/example worker_asr.py -- parakeet
```

The normal `run` stages use one visible CUDA device. Set `CUDA_VISIBLE_DEVICES` or your scheduler allocation before starting. The separator model is loaded once for the input manifest; ROSVOT batches chunks rather than submitting a job per song. No cluster-specific path or credential is embedded in the pipeline.
