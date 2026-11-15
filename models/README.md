# Models and provenance

## Included in this repository

- `mega53_lead_v1/`: about 77.5 MB of unchanged Safetensors weights, the JSON architecture config, individual shard checksums, original checkpoint hash, head mapping and MIT licence. This contains the shared Mega53 backbone plus original lead-vocal head 27 only. Every retained value/dtype was checked against the official v1.0.21 checkpoint; strict loading passed. No quantization or retraining was used.
- `generaluser.sf2`: GeneralUser GS v2.0.3, SHA-256 `9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe`, with the complete separate licence. It is used for the piano/choir listening reconstructions.

The original Mega53 checkpoint is from [this official release](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21), SHA-256 `c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f`. The author [confirmed MIT licensing of these weights](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/245#issuecomment-5834608838). `tools/prepare_lead_checkpoint.py` reproduces the derivation from that checkpoint and its YAML. The implementation is vendored under `third_party/MEGA53`; it computes only requested mask heads.

## Fetched separately

`python lab.py download-models` obtains the pinned official ROSVOT archive (including ROSVOT, RWBD and its RMVPE checkpoint) and the separate RMVPE checkpoint used by the Procedural baseline from the pinned STARS snapshot. The archive's SHA-256 is checked. Immutable upstream revisions and official URLs are in `upstream_sources.json`; recorded archive checksums are in `download_checksums.json`.

The core source revisions are ROSVOT `3c8332bf43adae35f6e4d64971862f2f6139b310` and STARS `f0e43e96cfe953f71a6cf9efd8b908b2c9d7e167`. Vendored copyright notices and licences remain intact. The RMVPE checkpoint is fetched from `verstar/STARS` revision `744a7ad02e1d788452293cd903ea6a933f7862c4`.

Larger comparison/caption/ASR weights are opt-in through `--extras`, with pinned versions and their own access/licensing requirements. MuScriptor weights require accepted upstream access and carry CC BY-NC 4.0; they are not included. A model licence does not automatically settle rights to source music or training audio.
