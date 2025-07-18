The work has been improved as a new version. Please refer to the most up to date repository with PyTorch Geometric and HuggingFace dataset support!
[CrysMTM](https://github.com/KurbanIntelligenceLab/CrysMTM)
---

## TDCM25: A Multi-modal Multi-task Benchmark for Temperature-dependent Crystalline Materials 
#### by Can Polat, Hasan Kurban, Erchin Serpedin, and Mustafa Kurban, ICLR 2025, Singapore.

TDCM25 is a comprehensive, multi-modal benchmark dataset designed to advance machine learning research in materials science. It focuses on modeling temperature-dependent properties of titanium dioxide (TiO₂) across its three crystalline phases—anatase, brookite, and rutile—by combining structural, visual, and textual modalities. [Paper Link](https://openreview.net/forum?id=bNB5SQTqKL)

##### The experiment codes are currently undergoing further improvements. For any questions or assistance with implementing them into your own models, feel free to contact can.polat@tamu.edu. Please include [TDCM25 Request] in the subject line.


## Overview

- **Dataset Size:** ~100,000 entries  
- **Material:** Titanium Dioxide (TiO₂)  
- **Phases:** Anatase, Brookite, Rutile  
- **Temperature Range:** 0K to 1000K (sampled in 50K increments)  
- **Modalities:**
  - **3D Atomic Coordinates:** Provided as XYZ files
  - **Molecular Images:** High-resolution RGB images generated per configuration
  - **Textual Metadata:** Detailed descriptions (e.g., Ti:O ratios, temperature, dimensions, and rotation angles)

### Data

You can download the data for compounds from the following link: [TDCM25 Data](https://drive.google.com/drive/folders/1gcTNtTMUI-ws8v2uaLw9REs4vx7qR6ke?usp=sharing)
## Dataset File Structure

```
TDCM25/
├── Phases/
│   ├──Temperatures/
│      ├── xyz/                # 3D atomic coordinates (.xyz files)
│      ├── images/             # Molecular images (e.g., .png files)
│      └── text/               # Textual metadata (.json or .txt files)
├── all_labels.txt             # Labels for prediction tasks
```
### Overall Dataset and Tasks

Below is the overall dataset figure illustrating the data structure and multi-modal representations:

![Alt text](figs/overal_fig.jpg?raw=true "TDCM25")

## Key Features

- **Multi-modal Data:** Integrates 3D geometry, visual representations, and textual descriptions to support diverse AI tasks.
- **Rotational Diversity:** Each configuration is sampled across ~526 distinct orientations using a quaternion-based method, ensuring robust rotational invariance.
- **Benchmark Tasks:** Designed for:
  - **Phase Classification:** Identify the crystalline phase of each TiO₂ sample.
  - **Property Prediction:** Predict key electronic structure properties (e.g., ground-state energy, LUMO/HOMO energies, Fermi Energy) from the multi-modal inputs.
  - **Explainability:** Generate human-readable explanations that capture structural and thermal characteristics using the provided data.

## Dataset Generation

TDCM25 was generated using density functional tight binding (DFTB) simulations (via DFTB+ with the tiorg-0-1 parameters) to compute electronic and structural properties across a range of temperatures.


## How to Use

1. **Download the Dataset:**  
   Clone the repository or download the dataset files from the [releases page](#).

2. **Setup Environment:**  
   Install the necessary libraries (e.g., Python 3.8+, NumPy, PyTorch, Matplotlib) as specified in `requirements.txt` (if provided).

   ```bash
   pip install -r requirements.txt
   ```

3. **Run Benchmark Tasks:**  
   - **Phase Classification:** Navigate to the `benchmarks/phase_classification` folder and run the training script:
     ```bash
     python train_phase_classifier.py --data_path ../data
     ```
   - **Property Prediction:** Use the scripts in `benchmarks/property_prediction` to train regression models.
   - **Explainability:** Explore the `benchmarks/explainability` directory for scripts that fine-tune language models on the textual metadata.


## Citation

If you use TDCM25 in your research, please cite our work as follows:

```bibtex
@inproceedings{polat2025tdcm,
  title     = {{TDCM}25: A Multi-Modal Multi-Task Benchmark for Temperature-Dependent Crystalline Materials},
  author    = {Can Polat and Hasan Kurban and Erchin Serpedin and Mustafa Kurban},
  booktitle = {AI for Accelerated Materials Design - ICLR 2025},
  year      = {2025},
  url       = {https://openreview.net/forum?id=bNB5SQTqKL}
}
```

## Contact

For questions or suggestions, please open an issue on this repository or contact the CalciumNitrade.

---
