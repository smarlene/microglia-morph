# microglia-morph

This repository contains the notebooks and python scripts utilized in microglia segmentation and classification. 

## Installation

It is recommended that a package/environemnt manager such as conda is used for running this project. To create an environment and install the requirements run the following command from the project root with Anaconda or conda installed on your system:

```bash
conda env create -f environment.yml
conda activate microglia-env
```

### Checkpoints
Model checkpoints can be downloaded from HuggingFace and should be placed in the directory `cell_localization/checkpoints/`. Checkpoints can be downloaded from: [HuggingFace](https://huggingface.co/smarlene/microglia-morph/tree/main).

## Repository Organization

This repository is organized into subdirectories for different stages of analysis. The subdirectories consist of `cell_localization` for operations done to tile and localize individual microglia cells, `cell_classification` for training and running the model used to classify microglia cells, and `scripts` for functions used throughout the project.

```
Microglia-morph/
├── cell_classification/
├── cell_localization/
|     ├── bboxes/                   # bounding boxes calculated during cell localization
|     ├── checkpoints/              # trained models for heatmap localization
|     ├── evaluation/               # ground truth annotations and training/test split of images for localization
|     ├── ilastik_files/            # ilastik model files for localization
|     ├── evaluate.ipynb            # notebook to sample and split training and test set for localization evaluatoin
|     ├── evaluate.py               # evaluation functions to calculate bulk scores based on predictions and ground truth
|     ├── model_utils.py            # functions used by heatmap localization
|     ├── soma_automated_localization.ipynb     # notebook for handcrafted localization pipeline
|     ├── soma_heatmap_localization.ipynb       # notebook for heatmap localization pipeline
|     └── soma_ml_localization.ipynb            # notebook for ilastik localization pipeline
├── scripts/
|     ├── filters.py        # filter functions for working with image tiles
|     ├── preprocess.py     # binarization functions for image tiles
|     └── utils.py          # miscellaneous utility functions used throughout cell localization
├── environment.yml         # Python dependencies
└── README.md               # Project overview
```