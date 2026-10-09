# CBIO064T "Automated 3D Cell Visualization" (2025, Chhabra and Boccara) - preregistration
Written and committed WITH src/run064t.py before any Cellpose output is scored. Two stacks were run once for timing only (Emb1_t301 test stack, 3D 36 s, 2D+stitch 7 s, object counts only, not compared with ground truth). Disclosure: before this commit I also ran the metric code on GT-vs-GT and z-flipped GT (self-test) and ran the Otsu+watershed baseline once on test stack index 10 (Emb1_t501) as a code-path check; it scored Dice 0.85, F1 0.77. No gate or parameter was changed after seeing it, and that stack is in the final 44.

## What the abstract claims, and what this can test
The students use Cellpose 2.0 to outline cells automatically on stacks of 2D images, then filter outlines and render in 3D. Their images, code and ground truth are not available. We test ONE piece: does pretrained Cellpose (v2.3.2, "nuclei" weights, no training, CPU) agree with expert manual nucleus annotation on public 3D fluorescence stacks? Data: BBBC050 (Broad Bioimage Benchmark, early mouse embryo nuclei, Tokuoka et al. 2020), https://bbbc.broadinstitute.org/BBBC050 .
LIMITS stated up front: (1) nuclei in mouse embryos, not the students' cells or tissue; (2) the ground truth we score against is manual nucleus annotation (GroundTruth_QCANet: one integer label per nucleus), so this measures segmentation agreement on nuclei, not the students' filtering step, their 3D rendering, or usability; (3) 44 test stacks come from only 4 embryos, so confidence intervals are wide and "underpowered" is not "no effect"; (4) the BBBC050 authors trained their own 3D network on this data, we do not compare with it.

## Data
test split: 44 stacks (Emb1-4, 11 timepoints t001..t501, CV1000 microscope, mCherry-H2B), shape 51x114x112 uint16, GT = test/GroundTruth_QCANet. Device-comparison set: train-split stacks of Emb01-Emb04 (44 stacks, IX71, mRFP1-H2B, shape 51x124x127), GT = train/GroundTruth_QCANet. Nuclei counts 1-50 per stack.

## Pipeline (fixed before scoring)
Cellpose 2.3.2 models.Cellpose(model_type='nuclei', gpu=False), channels [0,0], z_axis=0, default normalisation, diameter 17 (the model's default nuclei diameter in pixels), no tuning on this data. PRIMARY = do_3D=True. Other mode = do_3D=False with stitch_threshold=0.5 (2D masks stitched across z).
Metrics per stack: (a) foreground Dice (pred>0 vs GT>0); (b) instance F1 at 3D IoU 0.5 (greedy one-to-one match on IoU>=0.5; TP/FP/FN); (c) count error = predicted objects - GT objects. Summaries: mean over stacks, sample sd (ddof=1) over the 4 embryos, bootstrap 95% CI over stacks (2000 resamples, seed 0).
Baseline: gaussian smoothing sigma 1, Otsu threshold per stack, distance transform, peak_local_max(min_distance=4) markers, watershed.

## The 10 directions (all run; gates fixed now, not re-fitted after a miss)
D1 Data audit: counts, shapes, GT nuclei per stack, class balance. No gate.
D2 Headline: primary pipeline on the 44 test stacks. Gate "agreement reproduced": median foreground Dice >= 0.80 AND mean F1@0.5 >= 0.70.
D3 Non-deep baseline (Otsu+watershed). Gate "deep model adds value": mean F1 of Cellpose primary minus baseline >= 0.05.
D4 Diameter sensitivity: diameters 8, 12, 17, 24 in 2D+stitch mode. Gate "diameter-sensitive": range of mean F1 across the four > 0.10.
D5 Mode: primary 3D vs 2D+stitch (d17). Gate "modes differ": |difference in mean F1| >= 0.05.
D6 Stage (density) strata by GT nuclei per stack: <=4, 5-16, >16 (primary). Gate "works at high density": F1 >= 0.50 in the >16 stratum.
D7 Per-embryo: mean F1 and Dice per embryo, sd (ddof=1), bootstrap CI. Gate: none; report spread and say underpowered.
D8 Cross-device: 2D+stitch d17 on the 44 train-device stacks vs the 44 test-device stacks (matched embryo/timepoint grid). Gate "device-dependent": |difference in mean F1| >= 0.10.
D9 Robustness and metric sanity: (a) Gaussian noise sd = 0.10 x (p99-p1) of each stack, seed 0, 2D+stitch d17; (b) every 2nd z-slice kept (stack and GT); (c) CONTROL: foreground Dice of the primary prediction against the z-flipped GT. Gates: "fragile" if F1 falls >= 0.10 under (a) or (b) vs clean 2D+stitch; "metric non-trivial" if (c) is at least 0.20 below the primary Dice.
D10 Outline filtering (descriptive, NOT the students' filter): remove predicted objects < 100 voxels from the primary prediction; report F1 and count error before and after. No gate.

## Wording rule
README first line and RESULTS state the verdict from D2/D3 gates and then name any of D4-D9 that fired. If D2 passes but D3 shows the baseline is within 0.05, the claim is "agreement with manual nuclei annotation, no demonstrated advantage over thresholding". No statement about the students' own software, cells, or 3D visualisation quality.
