# Hybrid-Supervision-Periapical-Film

This repository provides the reproducibility code for our hybrid self-supervised and semi-supervised learning framework for multi-class periapical radiograph segmentation.

The framework contains two main components:

1. **IGLC-MAE pre-training**
   A domain-adaptive masked autoencoder pre-training strategy guided by intensity, gradient, and local contrast.

2. **Hybrid-supervised downstream segmentation**
   A Mask2Former-based segmentation framework with supervised fine-tuning, static pseudo-labeling, and confidence-adaptive semi-supervised learning.


## License

This repository is released for academic and research use. Please refer to the license file for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact the corresponding author.
