---
title: Singing Voice Data Lab
emoji: 🎶
colorFrom: indigo
colorTo: pink
sdk: static
app_file: index.html
license: mit
---
# Singing Voice Data Lab

[Complete code and bundled lead-vocal model](https://github.com/LAION-AI/singing-voice-data-lab) · [Public demo data](https://huggingface.co/datasets/laion/singing-voice-data-lab-demo)

Public listening comparison of 25 licensed LeVo2 recordings. The selected default is ROSVOT + smoothed Procedural pitch splits and gap filling. Six baseline/fusion methods, full audio, notes, complete MIDI tokens and downloads are available. Audio is precomputed and remains in a pinned TAR: browser byte-range requests load only the selected member, checked by SHA-256. No account login, API key or runtime model inference is needed.

Application code is MIT. The public recording/annotation subset is CC BY 4.0 with source attribution and changes recorded in its data card. Static clip metadata is derived from that licensed subset and retains CC BY 4.0 terms. Models and SoundFont retain their separate licences. Automatic reference agreement is not human-verified accuracy.
