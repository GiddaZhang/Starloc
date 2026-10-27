# Starloc

This repository reproduces the main claim of Starloc: approximately 0.7 deg AoA estimation error and less than 5 km orbital height estimation error. It also reproduces the primary result figures (Fig. 12 and Fig. 13) from the paper.

## Project Structure

- `Starloc_generate_results.py`: Main script for generating experiment results.
- `plot_final_results.ipynb`: Notebook for plotting and final result visualization.
- `Starloc.ipynb`: Notebook for interactive analysis (optional).
- `utils/`: Utility functions used by scripts and notebooks.
- `data/`: Input data directory.
- `results/`: Output directory for generated results.
- `environment.yml`: Conda environment definition.

## Setup

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate starloc
```

If your environment name differs, use the name defined in `environment.yml`.

## Usage

Generate results with:

```bash
python Starloc_generate_results.py
```

Then use notebook for analysis and plotting:

- `plot_final_results.ipynb`

## Notes

- Place required input files in `data/`.
- Generated outputs are written to `results/`.

## Contact
Code author: Jida Zhang (email: jidaz [at] stanford.edu), Ishani Janveja (email: janveja2 [at] illinois.edu)
