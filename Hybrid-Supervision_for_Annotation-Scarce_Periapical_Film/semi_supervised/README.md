# Adaptive Teacher-Student Semi-Supervised Training

This folder contains the confidence-adaptive teacher-student training entry point used after supervised Mask2Former initialization.

The implementation uses:

- a student Mask2Former updated by backpropagation;
- an EMA teacher updated from the student;
- teacher-generated semantic pseudo-labels for unlabeled images;
- a sigmoid confidence weight centered at `SEMI_SUPERVISED.CONFIDENCE_CENTER`;
- ignored low-confidence pixels controlled by `SEMI_SUPERVISED.MIN_PSEUDO_WEIGHT`;
- a weighted supervised:pseudo objective, defaulting to `0.85:0.15`.

No unlabeled images, pseudo-labels, or model weights are distributed.
