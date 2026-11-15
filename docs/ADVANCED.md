# Optional comparisons, ASR and captions

The default pipeline needs Mega53 lead extraction, RMVPE, ROSVOT and the frozen Procedural enhancements. The optional original workers are included for broader comparisons; they do not run during a normal default call.

| Worker | Purpose | Optional download |
|---|---|---|
| `worker_notes.py` | GAME, ROSVOT, STARS, TIFA | `--extras game stars tifa` |
| `worker_muscriptor.py --model large/medium` | Native voice-only MuScriptor transcription | `--extras muscriptor_large muscriptor_medium` |
| `worker_asr.py parakeet/nemotron` | Complete ASR and native word timing | `--extras parakeet nemotron` |
| `worker_caption.py lfm3/moss` | Audio captions, blind or original-prompt-aware | `--extras lfm3 moss_audio` |
| `worker_whisper_caption.py --model music_whisper` | Native timestamped 30-second-window captions | `--extras music_whisper` |

For example, after installing the appropriate environment and weights:

```bash
python lab.py download-models --extras game
python lab.py stage --workspace runs/example worker_notes.py -- --models game
```

The source snapshots and revision locks retain the executed upstream interfaces. MuScriptor runs its voice mask with greedy defaults and zero added onset delay; overlaps are preserved rather than silently corrected. Its weights are gated CC BY-NC 4.0. STARS needs usable transcript/phoneme conditioning, and TIFA is forced alignment of an ASR hypothesis, not a guarantee that the sung words are correct. Missing chunks or unsupported symbols remain explicit diagnostics.

The optional models can require different Transformers versions and much more VRAM. The recorded LFM/Nemotron path used Transformers source revision `109a7d19bf667168837f32929e00b791e195d14d`; MOSS used a separate environment with Transformers 4.57.1 and Tokenizers 0.22.2. Keep those environments separate. `requirements-observed.txt` lists the observed benchmark packages, rather than promising one universal package solution for every operating system.

Parakeet uses the local Moondream Photon interface (`moondream` 2.4.0 in the recorded environment). The exact revision is carried into workspace provenance. The caption workers receive the same full isolated lead; prompt-aware variants require an actual original prompt and remain unavailable if it is absent. No prompt, private dataset or model access token is invented to fill a missing output. The requested Vocal Music Whisper checkpoint was unavailable in the recorded experiment and is not required by this repository.

Use an authorized `HF_TOKEN` environment variable or the Hugging Face CLI's credential store when a gated model requires it. Never add tokens to source or commit downloaded private model caches. There are no paid generative API calls in the default pipeline or public demo.
