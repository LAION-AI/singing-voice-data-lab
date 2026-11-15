"""Native Whisper caption inference on complete lead vocals in explicit 30s windows.

This is an additional audio-only captioning experiment, not an instruction-following
replication of the four LFM3/MOSS variants. No original prompt, tags or lyrics are
given to either Whisper captioner. Run neural work only in an allocated GPU job.
"""
import argparse
import datetime
import hashlib
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

from common import BENCH, ROOT, complete, decode, failure, hardware, metric, read, samples, sha, sync, write

MODELS = {
    "music_whisper": {
        "model": "laion/music-whisper",
        "revision": "23dfa367927599130a7be9cfc33d1205d6e4b427",
        "license": "apache-2.0",
        "private": False,
    },
}
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 30 * SAMPLE_RATE


def windows(n):
    """Disjoint, complete coverage; last window is padded only by the processor."""
    return [(i, min(i + WINDOW_SAMPLES, n)) for i in range(0, n, WINDOW_SAMPLES)]


def word_count(text):
    return len(re.findall(r"\b[\w'-]+\b", text))


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def valid_chunk(path, audio_sha, revision, start, stop, max_tokens):
    if not path.exists():
        return None
    row = read(path)
    if (row.get("status") == "complete" and row.get("input_audio_sha256") == audio_sha
            and row.get("revision") == revision and row.get("onset_sample") == start
            and row.get("offset_sample") == stop
            and row.get("generation", {}).get("max_new_tokens") == max_tokens):
        return row
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--ids", nargs="*")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=440)
    p.add_argument("--model-dir", type=Path,
                   help="Local traceable checkpoint; requires --source-provenance for overrides.")
    p.add_argument("--source-provenance", type=Path,
                   help="JSON model/revision/license/source details for an explicitly documented local checkpoint.")
    a = p.parse_args()
    if a.batch_size < 1 or not 1 <= a.max_new_tokens <= 440:
        p.error("batch-size must be positive; max-new-tokens must be in [1,440]")
    spec = dict(MODELS[a.model])
    if a.model_dir and not a.source_provenance:
        p.error("--model-dir requires --source-provenance; do not silently substitute checkpoints")
    if a.source_provenance:
        spec.update(read(a.source_provenance))
        if not spec.get("model") or not spec.get("revision"):
            p.error("source provenance must identify model and revision")
    model_dir = a.model_dir or ROOT / "models" / a.model
    targets = samples(a.ids)
    available = (model_dir / "config.json").exists() and any(
        (model_dir / name).exists() for name in ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"])
    if not available:
        reason = ("The requested pinned repository contains no usable checkpoint: "
                  "no model weights/configuration are available. No substitute model was used.")
        for item in targets:
            d = BENCH / item["sample_id"]
            out = dict(status="unavailable", reason=reason, condition="audio_only_native_caption",
                       prompt=None, caption=None, source_metadata_provided=False,
                       original_generation_prompt=None, timestamp=utc_now(), **spec)
            write(d / "captions" / (a.model + ".json"), out)
            write(d / "status" / (a.model + ".json"), out)
        print("UNAVAILABLE", a.model, len(targets), reason, flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Caption inference must run inside an allocated GPU job")
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(model_dir), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    # These are the native English Whisper decoder task tokens, not lyric text.
    # The fine-tuned checkpoint interprets this as its learned caption task.
    gen = dict(max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=1,
               language="en", task="transcribe", return_timestamps=False,
               use_cache=True, return_dict_in_generate=True)
    sync()
    metric(a.model + "_load", None, time.perf_counter() - started,
           hardware=hardware(), parameter_count=sum(x.numel() for x in model.parameters()), **spec)
    import transformers
    model_generation = model.generation_config.to_dict()
    failures = []
    for item in targets:
        sid = item["sample_id"]
        d = BENCH / sid
        try:
            track_start = time.perf_counter()
            audio_path = d / "lead_vocal.flac"
            source_sha = sha(audio_path)
            audio = decode(audio_path, SAMPLE_RATE, 1)[:, 0]
            if not len(audio):
                raise ValueError("Empty isolated vocal")
            if not np.isfinite(audio).all():
                raise ValueError("Non-finite isolated vocal samples")
            decoded_sha = hashlib.sha256(np.asarray(audio, dtype="<f4").tobytes()).hexdigest()
            bounds = windows(len(audio))
            chunk_root = d / "raw" / a.model
            chunk_root.mkdir(parents=True, exist_ok=True)
            chunks = []
            missing = []
            for i, (start, stop) in enumerate(bounds):
                path = chunk_root / f"chunk-{i:04d}.json"
                saved = valid_chunk(path, source_sha, spec["revision"], start, stop, a.max_new_tokens)
                chunks.append(saved)
                if saved is None:
                    missing.append(i)
            generated_audio_seconds = 0.0
            generated_wall_seconds = 0.0
            generated_windows = 0
            for offset in range(0, len(missing), a.batch_size):
                indexes = missing[offset:offset + a.batch_size]
                waveforms = [audio[bounds[i][0]:bounds[i][1]] for i in indexes]
                sync()
                batch_start = time.perf_counter()
                features = processor(waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                     padding="max_length", max_length=WINDOW_SAMPLES,
                                     truncation=True, return_attention_mask=True)
                inputs = {k: (v.to("cuda", dtype=model.dtype) if k == "input_features" else v.to("cuda"))
                          for k, v in features.items()}
                with torch.inference_mode():
                    result = model.generate(**inputs, **gen)
                sync()
                elapsed = time.perf_counter() - batch_start
                sequences = result.sequences.cpu()
                texts = processor.batch_decode(sequences, skip_special_tokens=True)
                consumed = sum(len(w) for w in waveforms) / SAMPLE_RATE
                generated_audio_seconds += consumed
                generated_wall_seconds += elapsed
                generated_windows += len(indexes)
                metric(a.model + "_batch", sid, elapsed, consumed, batch_size=len(indexes),
                       precision="bfloat16", window_indexes=indexes,
                       output_token_slots=int(sequences.numel()), throughput_audio_seconds_per_second=consumed / elapsed)
                for i, seq, text in zip(indexes, sequences.tolist(), texts):
                    start, stop = bounds[i]
                    # Keep native full token IDs, including task prefix/end markers.
                    end = next((j for j, token in enumerate(seq) if j > 0 and token == processor.tokenizer.eos_token_id), len(seq))
                    tokens = seq[:end + 1]
                    # Whisper uses start-of-transcript + en + transcribe + no-timestamps.
                    initial_decoder_tokens = 4
                    generated_tokens = max(0, len(tokens) - initial_decoder_tokens)
                    row = dict(status="complete", caption=text, word_count=word_count(text),
                               onset_sample=start, offset_sample=stop, onset_seconds=start / SAMPLE_RATE,
                               offset_seconds=stop / SAMPLE_RATE, input_duration_seconds=(stop - start) / SAMPLE_RATE,
                               input_audio_sha256=source_sha, decoded_full_audio_sha256=decoded_sha,
                               original_generation_prompt=None, source_metadata_provided=False, prompt=None,
                               decoder_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
                               token_ids=tokens, generation=dict(gen, return_dict_in_generate=True,
                                   precision="bfloat16", output_tokens=generated_tokens,
                                   hit_token_limit=generated_tokens >= a.max_new_tokens),
                               batch_size=len(indexes), batch_wall_seconds=elapsed,
                               batch_time_allocation="Shared batch timing; do not sum batch_wall_seconds across chunks",
                               timestamp=utc_now(), **spec)
                    chunks[i] = row
                    write(chunk_root / f"chunk-{i:04d}.json", row)
            assert all(c is not None for c in chunks)
            caption = "\n\n".join(f"[{c['onset_seconds']:.2f}–{c['offset_seconds']:.2f} s]\n{c['caption']}" for c in chunks)
            meta = read(d / "processing_metadata.json")
            preview_start = float(meta.get("preview_start_seconds", 0))
            preview_end = preview_start + float(meta.get("preview_duration_seconds", 35))
            preview_chunks = [c for c in chunks if c["offset_seconds"] > preview_start and c["onset_seconds"] < preview_end]
            out = dict(status="complete", condition="audio_only_native_caption", caption=caption,
                       caption_format="Verbatim per-window captions with explicit absolute time labels; not an LLM summary",
                       preview_caption="\n\n".join(f"[{c['onset_seconds']:.2f}–{c['offset_seconds']:.2f} s]\n{c['caption']}" for c in preview_chunks),
                       word_count=sum(c["word_count"] for c in chunks), within_requested_word_range=None,
                       input_audio="complete lead_vocal.flac decoded mono 16000 Hz float32; consecutive nonoverlapping <=30s windows",
                       input_audio_sha256=source_sha, decoded_full_audio_sha256=decoded_sha,
                       input_duration_seconds=len(audio) / SAMPLE_RATE,
                       window_count=len(chunks), full_track_coverage=True, chunks=chunks,
                       prompt=None, source_metadata_provided=False, original_generation_prompt=None,
                       generation=dict(gen, precision="bfloat16", batch_size=a.batch_size,
                                       hit_token_limit=any(c["generation"]["hit_token_limit"] for c in chunks)),
                       decoding_note="Native learned caption output. The Whisper task token is named transcribe but its outputs are music captions, not ASR lyrics. No 50–80 word instruction is supported.",
                       generation_config_from_checkpoint=model_generation,
                       processor_config=processor.feature_extractor.to_dict(),
                       environment=dict(transformers=transformers.__version__, **hardware()),
                       generated_windows_this_invocation=generated_windows,
                       generated_wall_seconds_this_invocation=generated_wall_seconds,
                       generated_audio_seconds_this_invocation=generated_audio_seconds,
                       timestamp=utc_now(), **spec)
            write(d / "captions" / (a.model + ".json"), out)
            complete(a.model, sid, window_count=len(chunks), full_track_coverage=True)
            metric(a.model + "_track", sid, time.perf_counter() - track_start, len(audio) / SAMPLE_RATE,
                   generated_windows=generated_windows, reused_windows=len(chunks) - generated_windows,
                   inference_wall_seconds=generated_wall_seconds, precision="bfloat16", batch_size=a.batch_size)
        except Exception as exc:
            failure(a.model, sid, exc)
            failures.append(sid)
    if failures:
        raise RuntimeError(f"{a.model} failed for {len(failures)} track(s): {', '.join(failures)}; see per-track status JSON")


if __name__ == "__main__":
    main()
