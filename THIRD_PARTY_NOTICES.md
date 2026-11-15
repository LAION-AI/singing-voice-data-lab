# Third-party notices

Our orchestration, post-processing and application code are MIT. This does not replace the original terms below or grant rights to input music.

| Component | Attribution / terms |
|---|---|
| MVSEP Mega53 code and weights | Roman Solovyev (ZFTurbo), MVSep; MIT. Original licence retained in `third_party/MEGA53/LICENSE` and the bundled model directory. Weight permission is confirmed in the linked upstream issue. |
| ROSVOT | Ruiqi Li and contributors; MIT, retained in `third_party/ROSVOT/LICENSE`. Original model files are downloaded from the official archive. |
| STARS / RMVPE implementation snapshot | Yu Zhang and contributors; MIT, retained in `third_party/STARS/LICENSE.txt`; original subcomponent notices remain in source. |
| GAME / TIFA | openvpi and contributors; original notices/licences remain in their vendored directories. Optional weights retain upstream terms. |
| TinySoundFont / tinysoundfont | MIT implementations, installed as an optional rendering dependency. |
| GeneralUser GS v2.0.3 | S. Christian Collins; separate GeneralUser GS licence v2.0, retained as `models/generaluser_LICENSE.txt`. It permits music creation and software use subject to its stated terms. It is not relicensed by our MIT code tag. |
| MuScriptor | Kyutai × Mirelo; optional weight access/CC BY-NC 4.0 terms. No weights are bundled. |
| Public demo recordings | LAION Singing Voice 1K / LeVo2; CC BY 4.0. Dataset card records extraction, transcription and normalization changes. |

Vendored source revisions and checkpoint provenance are in `models/upstream_sources.json` and the bundled separator manifest. Only the smaller lead-head derivative and SoundFont are stored in Git. Other model downloads and source datasets require their respective permissions and licences.
