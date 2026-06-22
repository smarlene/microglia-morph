# microglia-morph

This repository is part of the [TU Delft 2025-2026 Research Project](https://github.com/TU-Delft-CSE/Research-Project) and contains the notebooks and python scripts utilized in microglia segmentation and classification.

## Installation

We recommend that a package/environment manager such as conda is used for running this project. To create an environment and install the requirements run the following command from the project root with Anaconda or conda installed on your system:

```bash
conda env create -f environment.yml
conda activate microglia-env
```

### Checkpoints
Model checkpoints can be downloaded from HuggingFace and should be placed in the directory `cell_localization/checkpoints/`. Checkpoints can be downloaded from: [HuggingFace](https://huggingface.co/smarleneschultz/microglia-localization/tree/main).

## Repository Organization

This repository is organized into subdirectories for different stages of analysis. The subdirectories consist of `cell_localization` for operations done to tile and localize individual microglia cells, `cell_analysis` for feature extraction, `cell_classification` for GMM classification and statistical tests, and `scripts` for functions used throughout the project.

```
Microglia-morph/
├── cell_analysis/
|     ├── sandbox.ipynb                # notebook for generating visualizations of cell feature extraction and debugging
|     └── extract_features.py          # methods used to skeletonize and binarize microglia and extract relevant features used for classification and comparison between experimental and control group
├── cell_classification/
|     ├── annotate.ipynb     # notebook for parsing annotations for NDPView2
|     ├── brain_region_comparison.ipynb       # notebook for comparing features by different annotated brain regions
|     ├── calculate_features.ipynb       # notebook for calculating and saving cell features from localized cells
|     ├── cell_density.ipynb       # notebook for comparing cell density by different annotated brain regions and by brain slice
|     ├── stats_analysis.ipynb       # notebook for carrying out statistical tests comparing TPO and EV brains
|     └── visualize_feats.ipynb            # notebook for creating cell analysis and density visualizations
├── cell_localization/
|     ├── bboxes/                   # bounding boxes calculated during cell localization
|     ├── checkpoints/              # trained models for heatmap localization
|     ├── evaluation/               # ground truth annotations and training/test split of images for localization
|     ├── ilastik_files/            # ilastik model files for localization
|     ├── evaluate.ipynb            # notebook to sample and split training and test set for localization evaluatoin
|     ├── soma_handcrafted_localization.ipynb     # notebook for handcrafted localization pipeline
|     ├── soma_heatmap_unet_localization.ipynb       # notebook for heatmap localization pipeline
|     ├── soma_ilastik_localization.ipynb            # notebook for ilastik localization pipeline
|     └── visualize_feats.ipynb            # notebook for creating cell localization visualizations

├── scripts/
|     ├── evaluate.py               # evaluation functions to calculate bulk scores based on predictions and ground truth
|     ├── filters.py        # filter functions for working with image tiles
|     └── utils.py          # miscellaneous utility functions used throughout cell localization
├── environment.yml         # Python dependencies
└── README.md               # Project overview
```
