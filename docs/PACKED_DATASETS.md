# Packed singing datasets and training annotation format

[Public synthetic singing corpus](https://huggingface.co/datasets/laion/synthetic-singing-voice)
· [Experimental singing checkpoint](https://huggingface.co/laion/Humaneness-Voice-Large-1.7-Singing-Experimental)

The corpus combines 49,881 lyric songs in English, German, Dutch, French and
Spanish, 9,988 nonlexical songs, and 146,640 LeVo singing recordings. It retains
the available original metadata, audio and MOSS-v2 q12/q32 arrays, and adds the
same **rosvot_procedural_segmented** note recipe used by this laboratory.
The dataset card reports actual release progress; planned counts are not a claim
that every annotation has already been uploaded. All recordings are synthetic.

These inputs are already intended to be isolated solo voices. This bulk path
**skips separation**, preserving original audio/token alignment. Use the ordinary
`lab.py run` path for mixed songs that need lead extraction. Residual instruments,
backing vocals and annotation failures can still occur and require filtering.

## Reproduce without extracting many files

```bash
pip install -e '.[inference]'
python lab.py download-models
hf download laion/synthetic-singing-voice --repo-type dataset --local-dir singing-data
python tools/reannotate_dataset.py --data singing-data --output rerun --limit-shards 1
```

For a larger run, start one process per CUDA GPU with distinct `--rank` values
from zero through `--world - 1`. Use the cluster's normal GPU visibility binding.
Each process keeps ROSVOT, RWBD and RMVPE resident. ROSVOT batches up to eight
30-second chunks, while Procedural retains the original 30-second cores, 0.5-s
context margin and RMVPE threshold 0.03. Whole source TARs are read sequentially;
MIDI and note exports are written in memory, directly into output TARs.

The input PCM is decoded with SoundFile, converted to mono by channel mean, and
resampled with SciPy `resample_poly` to 24 kHz for ROSVOT and 16 kHz for
Procedural. This bulk preprocessing differs from the original per-file FFmpeg
wrapper. The frozen fusion thresholds and 10-ms representation are unchanged;
floating-point/batching/resampling differences can change new predictions.
ROSVOT interval endpoints are clipped at the available chunk end, without time
warping. The original source recording and codec arrays remain byte-identical.

## Files for one recording

| Suffix | Meaning |
| --- | --- |
| `.flac` | Original synthetic vocal audio |
| `.json` | Dictionary: source metadata, labels, original prompt, new prompt, joins and review flags |
| `.notes.json` | Complete selected note-event dictionary, rule and round-trip receipt |
| `.rosvot.json`, `.procedural.json` | Both uncombined baseline note predictions |
| `.mid` | Standard MIDI, with note times in seconds converted to MIDI ticks |
| `.midi.txt` | Complete reversible SVAMIDI 1.0 token stream |
| `.pitch.npz` | Continuous 100-Hz F0, times and voiced mask |
| `.prompt.txt` | GENERAL, MIDI Annotation, SCRIPT, when an observed script exists |
| `.codes.npy`, `.codes32.npy` | Existing MOSS-v2 acoustic code arrays |
| `.acoustic.json.gz` | Existing acoustic annotations when stored in the same source TAR |

Earlier LeVo ASR/EARS annotations also remain in their pinned original dataset,
joined by source ID and the saved acoustic annotation locator. A note annotation
does not turn an unreliable or missing ASR transcript into verified lyrics.

## Conditioning text

```text
GENERAL: One solo singer, an original unaccompanied melody.
MIDI Annotation:
<BOS>
<NOTE_ON_60>
<TIME_SHIFT_50>
<NOTE_OFF_60>
<EOS>
SCRIPT:
(sung, sustained, gentle) The original words or performance directions.
```

The example illustrates the format; it is not a transcription of a released
recording. The dictionary stores `prompt_original`, `general`, `midi_annotation`,
`script` and `prompt_with_midi` separately. Nothing silently replaces the old
GENERAL/SCRIPT instructions. Missing observed scripts stay missing and are
explicitly excluded from the corresponding text-conditioned training selection.

Each note dictionary contains integer MIDI pitch (60 = middle C), floating-point
onset/offset in seconds, duration, pitch name and heuristic fusion provenance.
The flat stream uses NOTE_ON/NOTE_OFF events and TIME_SHIFT units of 10 ms.
A TIME_SHIFT_50 means 0.5 s. Terminal shifts retain trailing silence. The JSON
retains the original float boundaries; serialization is quantized to 10 ms.
MIDI-token round trips are checked before a shard is committed.

**MIDI tokens are separate from acoustic codec tokens.** MOSS-v2 has 12.5
80-ms frames/s: 12 codebook indices/frame means 150 indices/s; 32 means 400
indices/s. A frame contains all codebooks at once. Store arrays as
`[frames, codebooks]`; do not treat codebooks as extra elapsed time.

The experimental checkpoint was trained before this MIDI extension. Its
release demonstrates learning to sing alongside TTS; it does not demonstrate
following a MIDI score. The new prompt field prepares data for a future
conditioning experiment.

## Validation and limits

The in-memory fusion and standard-MIDI serializer reproduce all 75 frozen lab
outputs exactly at the note/timing level. A separate GPU pilot tests the packed
path across languages, nonlexical singing and LeVo. Shared RMVPE agreement,
coverage, training loss and successful serialization are not human judgments of
musical correctness. Data remain raw and need further filtering/annotation.

Dataset audio and annotations: CC BY 4.0. This repository's processing code:
MIT. Upstream ROSVOT/RMVPE/RWBD terms remain separate. See the dataset card for
actual counts, source revisions and archive checksums.
