# microglia-morph

This repository contains the notebooks and python scripts utilized in {title}. 

## Installation

Clone this repository and install the requirements using `pip install -r requirements.txt`.

## Repository Organization

This repository is organized into subdirectories for different stages of analysis. The subdirectories consist of `preprocessing` for operations done to tile and initially binarize the images, `localization` for localizing individual microglia cells, and `analysis` for analysis carried out on these individually localized cells.

```
Microglia-morph/
├── preprocessing/
|     ├── automated/
|     ├── filters.py
|     ├── image_loader.ipynb
|     └── prprocess.py
├── localization/
├── analysis/
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```