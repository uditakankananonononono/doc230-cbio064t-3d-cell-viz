partial: pretrained Cellpose (nuclei, default diameter) did NOT reproduce agreement with manual annotation on public 3D mouse-embryo nuclei (BBBC050): median Dice 0.69, F1@0.5 0.24, below the preregistered gates and below a simple Otsu+watershed baseline (F1 0.74); no claim about the students' own cells, filtering or 3D rendering.

Parent CBIO064T "Automated 3D Cell Visualization" (2025). 10 directions (D1-D10), PREREG.md + src/run064t.py committed together (18c4447) before scoring. Results and wording: RESULTS.md. Raw: results/.
Reproduce: download Images.zip and GroundTruth.zip from https://data.broadinstitute.org/bbbc/BBBC050/ and unzip into data/img and data/gt, then python src/run064t.py and python src/analyze064t.py (python 3.10, cellpose 2.3.2, CPU torch; about 2.5 hours on 2 CPU).
Constitution: real public data, CPU only, no stubs.
