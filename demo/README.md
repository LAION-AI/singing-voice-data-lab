---
title: Singing Voice Data Lab
emoji: 🎶
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.49.1
python_version: '3.13'
app_file: app.py
license: mit
---
# Singing Voice Data Lab

[Complete pipeline and bundled lead model](https://github.com/LAION-AI/singing-voice-data-lab) · [Public demo data and attribution](https://huggingface.co/datasets/laion/singing-voice-data-lab-demo)

Public listening comparison of 25 licensed LeVo2 examples, with six melody methods. The default is ROSVOT + smoothed Procedural pitch splits and gap filling. All predictions and audio are precomputed; no API key, account login or model inference is needed to listen.

The player includes original/lead audio, piano, choir, vocal overlays, a pitch plot, complete notes and reversible MIDI downloads. The data card records CC BY 4.0 source attribution and changes. Application code is MIT. The method comparisons use an automatic, shared RMVPE reference and are not verified transcription accuracy.
