# Four ROSVOT + Procedural recipes

All recipes preserve the raw inputs and record decisions. They use integer MIDI pitches; they do not average two pitches into an arbitrary intermediate note.

| Identifier | Frozen rule |
|---|---|
| `rosvot_procedural_ensemble` | Keep each ROSVOT note's exact boundaries. Replace its pitch only with at least 120 ms of unambiguous Procedural support, 60% note-time coverage, and a modal pitch share of at least 70%. |
| `rosvot_procedural_gapfill` | Keep all ROSVOT notes and pitches. Add Procedural fragments of at least 90 ms only inside the complement of the union of ROSVOT note intervals. |
| `rosvot_procedural_vote_gapfill` | Keep original ROSVOT parent boundaries. A unique modal pitch needs at least 60 ms support, 25% note-time coverage and 50% modal share. Add the same gap fragments. |
| `rosvot_procedural_segmented` | Within each ROSVOT parent note, use stable Procedural pitch runs of at least 90 ms. Absorb fragments shorter than 50 ms into the nearest-pitch neighbour; ties choose the longer neighbour, then temporal order. Retain original outer boundaries and add the same gap fragments. |

Ambiguous overlapping Procedural frames provide no pitch-vote support. Missing or insufficient support keeps ROSVOT's pitch. Smoothing stays within one parent note and consults no reference pitch value. The note JSON contains the original ROSVOT pitch, changed/added flags and the relevant support/decision metadata.

The default split recipe creates more note boundaries. On the recorded 75 examples it produced 27,755 notes, versus 22,399 for ROSVOT. It eliminated the 1,689 sub-50-ms fragments created by the unsmoothed split trial. Notes longer than 50 ms can still be musically unnecessary; listen to the overlays.

## How the recipe was selected

`code/search_ensemble.py` preserves the exploratory search implementation: 69 predefined combinations, with 15 recordings chosen by a fixed SHA-256 order (five per source group) for selection and 60 for separate internal reporting. Vote/split settings maximize mean within-50-cents reference agreement; the selected split family must also have a fragment floor of at least 50 ms. The released default is fixed and does not reselect a setting per input.

The benchmark had been examined previously. This is an internal exploratory comparison, not an untouched external validation set. The source recordings span different styles, but this is not evidence of arbitrary-language or arbitrary-genre note accuracy.

## Read the metrics together

- **Coverage:** fraction of automatic reference-voiced time with any predicted note. A wrong note counts as covered.
- **Within 50/100 cents:** fraction of all reference-voiced frames with exactly one predicted pitch close to the reference. Missing/overlapping notes earn no credit.
- **Conditional pitch difference:** median absolute error on jointly covered, unambiguous frames. Missing material is excluded.
- **Contour correlation:** whether the melody rises/falls similarly. An octave shift can retain high correlation.
- **Frame F1 proxy:** let C be correctly matched single-pitch frames within 50 cents, V reference-voiced frames, and N any predicted-note frames. The proxy is 2C/(V+N); it penalizes unmatched activity as well as missed singing. It is not note-level F1 against human labels.

ROSVOT and Procedural both use RMVPE, and Procedural is constructed from the same F0 used as reference. Agreement is therefore biased. Procedural alone has lower coverage and stronger Frame F1 in this comparison; the default offers a useful different trade-off. There is no universal winner, verified note ground truth, or new ensemble listening score.

The public demo contains the 25 CC BY 4.0 LeVo examples. The README table is the original aggregate 75-recording result, not a claim that those percentages are specific to the public subset. The public baseline data lets anyone reproduce the four symbolic combinations on CPU with exact token/MIDI checks.
