This is the official repository for the TDCM25: A Multi-Modal Multi-Task Benchmark for Temperature-Dependent Crystalline Materials by Can Polat, Hasan Kurban, Erchin Serpedin, and Mustafa Kurban, ICLR 2025.

Below is an updated README file that includes an embedded view of your overall dataset figure.

---

# TDCM25: A Multi-modal Multi-task Benchmark for Temperature-dependent Crystalline Materials 
## by Can Polat, Hasan Kurban, Erchin Serpedin, and Mustafa Kurban. ICLR 2025

TDCM25 is a comprehensive, multi-modal benchmark dataset designed to advance machine learning research in materials science. It focuses on modeling temperature-dependent properties of titanium dioxide (TiO₂) across its three crystalline phases—anatase, brookite, and rutile—by combining structural, visual, and textual modalities.

## Overview

- **Dataset Size:** ~100,000 entries  
- **Material:** Titanium Dioxide (TiO₂)  
- **Phases:** Anatase, Brookite, Rutile  
- **Temperature Range:** 0K to 1000K (sampled in 50K increments)  
- **Modalities:**
  - **3D Atomic Coordinates:** Provided as XYZ files
  - **Molecular Images:** High-resolution RGB images generated per configuration
  - **Textual Metadata:** Detailed descriptions (e.g., Ti:O ratios, temperature, dimensions, and rotation angles)

### Overall Dataset and Tasks

Below is the overall dataset figure illustrating the data structure and multi-modal representations:

<div align="center">
  <embed src="figs/overal_fig.pdf" type="application/pdf" width="80%" height="600px" />
  <br>
  <em>Figure: Overall structure and modalities of the TDCM25 dataset</em>
</div>

*If the embedded figure does not display correctly, please [click here](figs/overal_fig.pdf) to view the figure.*

## Key Features

- **Multi-modal Data:** Integrates 3D geometry, visual representations, and textual descriptions to support diverse AI tasks.
- **Rotational Diversity:** Each configuration is sampled across ~526 distinct orientations using a quaternion-based method, ensuring robust rotational invariance.
- **Benchmark Tasks:** Designed for:
  - **Phase Classification:** Identify the crystalline phase of each TiO₂ sample.
  - **Property Prediction:** Predict key electronic structure properties (e.g., ground-state energy, LUMO/HOMO energies) from the multi-modal inputs.
  - **Explainability:** Generate human-readable explanations that capture structural and thermal characteristics using the provided data.

## Dataset Generation

TDCM25 was generated using density functional tight binding (DFTB) simulations (via DFTB+ with the tiorg-0-1 parameters) to compute electronic and structural properties across a range of temperatures. The simulation outputs were post-processed to generate:
- **XYZ Files:** Capturing the 3D atomic coordinates.
- **Molecular Images:** Created with Matplotlib to visually represent each configuration.
- **Textual Metadata:** Concise descriptions that include elemental composition, spatial dimensions, and rotation angles.

## File Structure

A typical repository layout might look like:

```
TDCM25/
├── data/
|   ├── Phases    
│       ├── xyz/                # 3D atomic coordinates (.xyz files)
│       ├── images/             # Molecular images (e.g., .png files)
│       └── metadata/           # Textual metadata (.json or .txt files)
├── benchmarks/
│   ├── phase_classification/  # Scripts and splits for phase classification task
│   ├── property_prediction/   # Scripts for regression tasks on electronic properties
│   └── explainability/        # Code for LLM-based explainability experiments
├── figs/
│   └── overal_fig.pdf      # Overall dataset figure
├── docs/
│   └── TDCM25_paper.pdf    # (Optional) Link to the paper under review
├── README.md
└── LICENSE
```

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

4. **Explore Data:**  
   Visualize sample images, inspect corresponding XYZ files, and review the metadata to better understand the dataset's structure.

## Citation

If you use TDCM25 in your research, please cite our work as follows:

> Anonymous authors. *TDCM25: A Multi-modal Multi-task Benchmark for Temperature-dependent Crystalline Materials*. Under review at ICLR 2025.

Additional citation details and BibTeX entries can be found in the [docs](docs/TDCM25_citation.bib).

## License

This dataset is provided under the [INSERT LICENSE NAME] license. Please see the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please open an issue on this repository or contact the CalciumNitrade.

---
