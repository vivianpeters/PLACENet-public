# PLACENet-public
PLACENet - Passive Localization of Active materials in shielded Containers Enabled by neural Networks

CNN-based model for multi-source detection and 3D bounding box regression from spectral data, with "boxes + confidence"-style output and optional hybrid loss (smooth L1 + CIoU) with greedy or Jonker–Volgenant matching.

The source code and sample data are archived on Zenodo: [![DOI](https://zenodo.org/badge/1164606150.svg)](https://doi.org/10.5281/zenodo.18743642)

## Repository structure

```
├── run_PLACENet.py          # Entry script: configure and run training + evaluation
├── README.md
├── requirements.txt
├── LICENSE
├── PLACENet/                # Python package
│   ├── __init__.py
│   ├── PLACENet_prep.py     # Data loading and preparation
│   ├── PLACENet_model.py    # Model and training
│   └── PLACENet_evaluate.py # Evaluation and plotting
└── sample_data/         # Optional: example data to try the pipeline

```

## Requirements

- Python 3.8 or higher (tested with 3.10)
- Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies: `numpy`, `tensorflow`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`.

## Data format

Place your data in a single directory. The pipeline expects **paired files** per dataset:

- **Energy/spectra:** CSV files whose names contain `detector_energy_data` and `_gamma` (e.g. `MySet_gamma_detector_energy_data.csv`).
- **Positions:** CSV files whose names contain `position_arrays` and the same set prefix (e.g. `MySet_gamma_position_arrays.csv`).

Each dataset is identified by the prefix before `_gamma`. See `PLACENet/PLACENet_prep.py` for the exact discovery logic.

If you use the provided **sample_dataset/** folder, point `data_root` to it (or to its parent). Otherwise use your own data directory.

## Quick start

1. **Set the data path** in `run_PLACENet.py`:

   ```python
   data_root = Path("/path/to/your/data/directory")
   ```

   Use the path to the folder containing your `*_gamma_detector_energy_data*` and `*_gamma_position_arrays*` files, or the path to `sample_data` if you use the included sample.

2. **Run from the repository root** (the directory that contains `run_PLACENet.py` and the `PLACENet` folder):

   ```bash
   python run_PLACENet.py
   ```

   Do not run from inside `PLACENet/`; the script must see the `PLACENet` package as a subfolder.

3. The script will:
   - Load and prepare datasets from `data_root`.
   - Train a model (concatenated datasets, 5-fold by default) and save checkpoints and loss curves to a `Results_<run_name>` directory.
   - Run evaluation on the held-out test fold and print R², IoU, and related metrics.
   - Optionally save a comparison plot (e.g. `compare_histo`).

## Configuration

- **Data prep:** `PLACEPrepConfig` in `run_PLACENet.py` — `data_dir`, `dets`, `chunk_size`, `max_sources`, `cluster_eps`, `randomize_slot_assignment`, etc.
- **Training:** `PLACENetConfig` — `loss_type` (`"smooth_l1"`, `"ciou"`, `"hybrid"`), `matching_strategy` (`"greedy"` or `"jv"`), `epochs`, `batch_size`, `folds`, `run_name`, `max_sources`, loss weights, augmentation options.

Evaluation uses the same `max_sources` as the trainer config; the script passes `trainer.config.max_sources` into `evaluate_run()`.

## Outputs

- **Results directory** `Results_<run_name>/`: trained model weights per fold, test data/labels, loss curves, and evaluation logs.
- **Evaluation:** R² (with and without slot matching), IoU, source-count metrics, and plots (R² scatter, confidence, etc.). Summary metrics are printed to the console and written to `r2_scores.log`.

## Citation

If you use this code in your research, please cite the accompanying paper.

