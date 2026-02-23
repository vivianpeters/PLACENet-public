"""
Example script showing how to run the PLACENet pipeline.
"""

import glob
import re
from pathlib import Path

import numpy as np

from PLACENet import (
    PLACENet,
    PLACENetConfig,
    PLACENetPlot,
    PLACENetPrep,
    PLACEPlotConfig,
    PLACEPrepConfig,
)
from PLACENet.PLACENet_evaluate import evaluate_run

# ---------------------------------------------------------------------------
# 1) Configure and load datasets
# ---------------------------------------------------------------------------
data_root = Path("/path/to/data/directory")

prep = PLACENetPrep( # Prepare the datasets
    PLACEPrepConfig(
        data_dir=str(data_root),
        dets=16, # Number of detectors
        chunk_size=16, # Number of spectra to group together
        pad_value=-1,
        max_sources=3, # Maximum number of sources to classify
        cluster_eps=2.0,
        randomize_slot_assignment=True, # If True, randomize the slot assignment
    )
)

datasets = prep.load_datasets() # Load the datasets
for dataset in datasets:
    print(dataset.summary())

# ---------------------------------------------------------------------------
# 2) Train a model (concatenated example shown; per-dataset also available)
# ---------------------------------------------------------------------------

trainer = PLACENet( # Train the model
    PLACENetConfig(
        pad_value=-1,
        l2_reg=1e-2, # L2 regularization strength
        learning_rate=1e-4,
        loss_type="hybrid",  # "smooth_l1", "ciou", or "hybrid"
        matching_strategy="greedy",  # "greedy" or "jv"
        epochs=3000,
        batch_size=32,
        folds=5,
        run_name="run_name_of_choice",
        max_sources=3,
        delta=1.0, # for smooth_l1 loss
        ciou_weight=1.0, # weight for CIoU cost in hybrid box matching cost
        smooth_weight=0.1, # weight for smooth L1 cost in hybrid box matching cost
        confidence_weight=1.5, # scale confidence loss compared to box loss
        noobj_weight=0.1, # scale class imbalance for confidence loss
        enable_augmentation=True,
        augmentation_multiplier=5,
    )
)

result = trainer.train_concatenated(datasets) # Train the model and save the results
print(f"Finished training model ({result.run_label})")
print(result.metrics[-1])

# Save results to a directory
res_dir = Path(f"Results_{trainer.config.run_name}") if trainer.config.run_name else Path("Results")
PLACENetPlot.save_loss_curves(result.histories, res_dir / "loss_curves.pdf")
file_label = trainer.config.run_name or result.run_label

# ---------------------------------------------------------------------------
# 3) Evaluate model on held-out test set
# ---------------------------------------------------------------------------
evaluate_run(trainer.config.max_sources, res_dir, file_label)

# ---------------------------------------------------------------------------
# 4) Optional plotting example
# ---------------------------------------------------------------------------

plotter = PLACENetPlot(PLACEPlotConfig(binx=32, biny=32, binz=72))
plotter.compare_histo(res_dir, file_label, model_suffix="", sample_idx=42, binx=102, biny=102, binz=72)
