"""Evaluation and plotting for PLACENet."""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from cycler import cycler
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

def compute_iou_per_slot_stats(y_true, y_pred, pad_value=-1):
    """
    Compute 3D IoU per valid slot from labels and predictions (NumPy only).
    Box format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre].
    Returns (mean_iou, std_iou, count) over valid (non-padded) slots.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape[-1] == 7:
        y_true = y_true[..., :6]
    if y_pred.shape[-1] == 7:
        y_pred = y_pred[..., :6]
    # Valid slot = all 6 dims non-pad
    valid = np.all(y_true != pad_value, axis=-1)
    # Centre/width -> min/max
    xw_t, xc_t, yw_t, yc_t, zw_t, zc_t = y_true[..., 0], y_true[..., 1], y_true[..., 2], y_true[..., 3], y_true[..., 4], y_true[..., 5]
    xw_p, xc_p, yw_p, yc_p, zw_p, zc_p = y_pred[..., 0], y_pred[..., 1], y_pred[..., 2], y_pred[..., 3], y_pred[..., 4], y_pred[..., 5]
    x_min_t = xc_t - np.abs(xw_t) / 2.0
    x_max_t = xc_t + np.abs(xw_t) / 2.0
    y_min_t = yc_t - np.abs(yw_t) / 2.0
    y_max_t = yc_t + np.abs(yw_t) / 2.0
    z_min_t = zc_t - np.abs(zw_t) / 2.0
    z_max_t = zc_t + np.abs(zw_t) / 2.0
    x_min_p = xc_p - np.abs(xw_p) / 2.0
    x_max_p = xc_p + np.abs(xw_p) / 2.0
    y_min_p = yc_p - np.abs(yw_p) / 2.0
    y_max_p = yc_p + np.abs(yw_p) / 2.0
    z_min_p = zc_p - np.abs(zw_p) / 2.0
    z_max_p = zc_p + np.abs(zw_p) / 2.0
    xi_min = np.maximum(x_min_t, x_min_p)
    xi_max = np.minimum(x_max_t, x_max_p)
    yi_min = np.maximum(y_min_t, y_min_p)
    yi_max = np.minimum(y_max_t, y_max_p)
    zi_min = np.maximum(z_min_t, z_min_p)
    zi_max = np.minimum(z_max_t, z_max_p)
    inter = np.maximum(xi_max - xi_min, 0) * np.maximum(yi_max - yi_min, 0) * np.maximum(zi_max - zi_min, 0)
    vol_t = np.abs(xw_t) * np.abs(yw_t) * np.abs(zw_t)
    vol_p = np.abs(xw_p) * np.abs(yw_p) * np.abs(zw_p)
    union = vol_t + vol_p - inter + 1e-7
    iou = inter / union
    iou_valid = np.where(valid, iou, np.nan)
    flat = iou_valid.ravel()
    flat = flat[~np.isnan(flat)] # remove nan values (~ is reverse of is)
    count = len(flat) # number of valid slots
    if count == 0:
        return np.nan, np.nan, 0
    mean_iou = float(np.mean(flat)) # mean IoU
    std_iou = float(np.std(flat, ddof=1)) if count > 1 else 0.0
    return mean_iou, std_iou, count # return mean IoU, std IoU, and number of valid slots


@dataclass
class PLACEPlotConfig:
    """Configuration for plotting."""
    # number of bins in each direction
    # default values based on current dataset
    binx: int = 32
    biny: int = 32
    binz: int = 72


# R² scatter plots: max points per subplot for downsampling. Set to None to plot all points.
MAX_SCATTER_POINTS = None


# --------------------------------
# Extract boxes from YOLO-style output
# --------------------------------
def extract_boxes_from_yolo_output(y_pred, confidence_threshold=0.5, pad_value=-1):
    """
    Extract boxes from YOLO-style output and filter by confidence.
    
    Args:
        y_pred: (batch, max_sources, 7) where last dim is [boxes(6), confidence(1)]
        confidence_threshold: Minimum confidence to keep a prediction
        pad_value: Value to use for filtered-out slots
        
    Returns:
        boxes: (batch, max_sources, 6) - boxes with low-confidence slots set to pad_value
        confidences: (batch, max_sources) - confidence scores
    """
    if y_pred.shape[-1] == 7:
        # YOLO-style output
        boxes = y_pred[..., :6]  # (batch, max_sources, 6)
        confidences = y_pred[..., 6]  # (batch, max_sources)
        
        # Filter by confidence: set boxes to pad_value if confidence < threshold
        low_conf_mask = confidences < confidence_threshold
        boxes_filtered = boxes.copy()
        boxes_filtered[low_conf_mask] = pad_value
        
        return boxes_filtered, confidences
    else:
        # Original output (no confidence)
        return y_pred, None


# --------------------------------
# Helpers for matching (valid boxes + assignment)
# --------------------------------
def _get_valid_boxes(y_true_b: np.ndarray, y_pred_b: np.ndarray, pad_value: float = -1):
    """
    Get valid (non-padded) box rows for one batch item.
    Returns (true_valid_mask, pred_valid_mask, true_valid, pred_valid).
    true_valid has shape (n_true, 6), pred_valid has shape (n_pred, 6).
    """
    true_valid_mask = ~np.all(y_true_b == pad_value, axis=-1)
    pred_valid_mask = ~np.all(y_pred_b == pad_value, axis=-1)
    true_valid = y_true_b[true_valid_mask]
    pred_valid = y_pred_b[pred_valid_mask]
    return true_valid_mask, pred_valid_mask, true_valid, pred_valid


def _compute_box_assignment(
    true_valid: np.ndarray, pred_valid: np.ndarray, matching_type: str = "greedy"
) -> List[tuple]:
    """
    Compute assignment of predicted boxes to ground truth boxes.
    Cost = L2 distance between box vectors. Returns list of (pred_idx, true_idx).
    """
    n_true = len(true_valid)
    n_pred = len(pred_valid)
    if n_true == 0 or n_pred == 0:
        return []

    cost_matrix = np.zeros((n_pred, n_true))
    for i in range(n_pred):
        for j in range(n_true):
            cost_matrix[i, j] = np.linalg.norm(pred_valid[i] - true_valid[j])

    if matching_type == "jv":
        # Square matrix for linear_sum_assignment
        size = max(n_true, n_pred)
        cost_square = np.full((size, size), 1e10)
        cost_square[:n_pred, :n_true] = cost_matrix
        row_ind, col_ind = linear_sum_assignment(cost_square)
        return [(i, j) for i, j in zip(row_ind, col_ind) if i < n_pred and j < n_true]

    # greedy
    cost = cost_matrix.copy()
    inf = 1e10
    matched_pairs = []
    used_pred = set()
    used_true = set()
    for _ in range(min(n_pred, n_true)):
        min_val = np.inf
        min_i, min_j = -1, -1
        for i in range(n_pred):
            if i in used_pred:
                continue
            for j in range(n_true):
                if j in used_true:
                    continue
                if cost[i, j] < min_val:
                    min_val = cost[i, j]
                    min_i, min_j = i, j
        if min_i == -1 or min_j == -1 or min_val >= inf:
            break
        matched_pairs.append((min_i, min_j))
        used_pred.add(min_i)
        used_true.add(min_j)
        cost[min_i, :] = inf
        cost[:, min_j] = inf
    return matched_pairs


def _save_r2_scatter_plot(
    output_path: Path,
    overall_truth: np.ndarray,
    overall_pred: np.ndarray,
    widths_truth: np.ndarray,
    widths_pred: np.ndarray,
    positions_truth: np.ndarray,
    positions_pred: np.ndarray,
    r2_overall: float,
    r2_widths: float,
    r2_positions: float,
    suptitle: str,
    verbose: bool = True,
) -> None:
    """
    Create and save the 3-panel R² scatter (overall, widths, positions).
    Skips saving if all data arrays are empty.
    """
    if (overall_truth.size == 0 and widths_truth.size == 0 and positions_truth.size == 0):
        return

    def _plot_subset(ax, x, y, color, title, r2_value):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0 or y.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
            ax.set_axis_off()
            return
        n = x.size
        if MAX_SCATTER_POINTS is not None and n > MAX_SCATTER_POINTS:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=MAX_SCATTER_POINTS, replace=False)
            x, y = x[idx], y[idx]
        ax.scatter(x, y, s=8, alpha=0.4, color=color, edgecolors="none", rasterized=True)
        vmin = min(x.min(), y.min())
        vmax = max(x.max(), y.max())
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.0, label="y = x")
        ax.set_xlabel("True (cm)", fontsize=16)
        ax.set_ylabel("Predicted (cm)", fontsize=16)
        if r2_value is not None and not np.isnan(r2_value):
            ax.set_title(f"{title}\nR² = {r2_value:.3f}", fontsize=16, pad=20)
        else:
            ax.set_title(f"{title}\nR² = n/a", fontsize=16, pad=20)
        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=14)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _plot_subset(axes[0], overall_truth, overall_pred, "#1f77b4", "Overall (all components)", r2_overall)
    _plot_subset(axes[1], widths_truth, widths_pred, "#ff7f0e", "Widths (xwidth, ywidth, zwidth)", r2_widths)
    _plot_subset(axes[2], positions_truth, positions_pred, "#2ca02c", "Positions (xcentre, ycentre, zcentre)", r2_positions)
    fig.suptitle(suptitle, fontsize=20, fontweight="bold", y=0.98)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.82, wspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"Saved R² scatter plot to {output_path} and .png")


# --------------------------------
# Apply greedy matching
# --------------------------------
def apply_greedy_matching(y_true, y_pred, pad_value=-1, confidence_threshold=0.5):
    """
    Apply greedy matching to predictions before evaluation.
    Reorders predicted boxes to match ground truth box order using greedy algorithm.
    Greedy matching is a simple algorithm that matches the predicted box to the closest true box.
    This is the same algorithm used in the training loop when greedy=True.
    
    Args:
        y_true: Ground truth boxes (batch, max_sources, 6) or (batch, max_sources, 7) if YOLO
        y_pred: Predicted boxes (batch, max_sources, 6) or (batch, max_sources, 7) if YOLO
        pad_value: Value used for padding (rows with all pad_value are ignored)
        confidence_threshold: If y_pred has 7 dims, filter by confidence
    
    Returns:
        y_pred_matched: Predictions reordered to match ground truth (batch, max_sources, 6)
    """
    # Handle YOLO-style output (extract boxes and filter by confidence)
    if y_pred.shape[-1] == 7:
        y_pred, _ = extract_boxes_from_yolo_output(y_pred, confidence_threshold, pad_value)
    
    # Handle YOLO-style ground truth (extract boxes only)
    if y_true.shape[-1] == 7:
        y_true = y_true[..., :6]
    
    batch_size = y_true.shape[0]
    max_sources = y_true.shape[1]
    y_pred_matched = np.full_like(y_pred, pad_value)
    
    for b in range(batch_size):
        true_valid_mask, pred_valid_mask, true_valid, pred_valid = _get_valid_boxes(
            y_true[b], y_pred[b], pad_value
        )
        if len(true_valid) == 0:
            y_pred_matched[b] = y_pred[b]
            continue
        if len(pred_valid) == 0:
            continue
        matched_pairs = _compute_box_assignment(true_valid, pred_valid, "greedy")
        matched_pred = np.full((max_sources, 6), pad_value, dtype=y_pred.dtype)
        for pred_idx, true_idx in matched_pairs:
            true_pos = np.where(true_valid_mask)[0][true_idx]
            matched_pred[true_pos] = pred_valid[pred_idx]
        y_pred_matched[b] = matched_pred
    
    return y_pred_matched


# --------------------------------
# Apply JV matching
# --------------------------------
def apply_jv_matching(y_true, y_pred, pad_value=-1, confidence_threshold=0.5):
    """
    Apply Jonker-Volgenant (JV) matching to predictions before evaluation.
    Reorders predicted boxes to match ground truth box order.
    JV matching is a variant of the Hungarian algorithm that ensures a unique assignment.

    The algorithm works by building a cost matrix between all predicted and true boxes,
    and then finding the minimum cost assignment.
    The algorithm is guaranteed to find a unique assignment if it exists.
    If there is no unique assignment, the algorithm will find the best assignment that minimizes the total cost.
    
    Args:
        y_true: Ground truth boxes (batch, max_sources, 6) or (batch, max_sources, 7) if YOLO
        y_pred: Predicted boxes (batch, max_sources, 6) or (batch, max_sources, 7) if YOLO
        pad_value: Value used for padding (rows with all pad_value are ignored)
        confidence_threshold: If y_pred has 7 dims, filter by confidence
    
    Returns:
        y_pred_matched: Predictions reordered to match ground truth (batch, max_sources, 6)
    """
    # Handle YOLO-style output (extract boxes and filter by confidence)
    if y_pred.shape[-1] == 7:
        y_pred, _ = extract_boxes_from_yolo_output(y_pred, confidence_threshold, pad_value)
    
    # Handle YOLO-style ground truth (extract boxes only)
    if y_true.shape[-1] == 7:
        y_true = y_true[..., :6]
    
    batch_size = y_true.shape[0]
    max_sources = y_true.shape[1]
    y_pred_matched = np.full_like(y_pred, pad_value)
    
    for b in range(batch_size):
        true_valid_mask, pred_valid_mask, true_valid, pred_valid = _get_valid_boxes(
            y_true[b], y_pred[b], pad_value
        )
        if len(true_valid) == 0:
            y_pred_matched[b] = y_pred[b]
            continue
        if len(pred_valid) == 0:
            continue
        matched_pairs = _compute_box_assignment(true_valid, pred_valid, "jv")
        matched_pred = np.full((max_sources, 6), pad_value, dtype=y_pred.dtype)
        for pred_idx, true_idx in matched_pairs:
            true_pos = np.where(true_valid_mask)[0][true_idx]
            matched_pred[true_pos] = pred_valid[pred_idx]
        y_pred_matched[b] = matched_pred
    
    return y_pred_matched


# --------------------------------
# Match confidence scores
# --------------------------------
def match_confidence_scores(y_true, y_pred_boxes, pred_confidences, pad_value=-1, matching_type="greedy"):
    """
    Match confidence scores to ground truth slots using the same matching as boxes.
    NB: This is not the same as matching the confidence scores to the ground truth boxes.
    Essentially, this is a copy of the box matching algorithm, but for the confidence scores.
    If the confidence scores are not matched to the correct ground truth box, the loss will be incorrect.
    Ensure that the same assignment is used for the confidence scores as for the boxes.
    
    Args:
        y_true: Ground truth boxes (batch, max_sources, 6)
        y_pred_boxes: Predicted boxes (batch, max_sources, 6) - should be UNMATCHED (raw predictions)
        pred_confidences: Predicted confidence scores (batch, max_sources)
        pad_value: Value used for padding
        matching_type: "greedy" or "jv"
    
    Returns:
        matched_confidences: Confidence scores reordered to match ground truth (batch, max_sources)
    """
    if pred_confidences is None:
        return None
    
    batch_size = y_true.shape[0]
    max_sources = y_true.shape[1]
    matched_confidences = np.zeros_like(pred_confidences)
    
    for b in range(batch_size):
        true_valid_mask, pred_valid_mask, true_valid, pred_valid = _get_valid_boxes(
            y_true[b], y_pred_boxes[b], pad_value
        )
        pred_conf_valid = pred_confidences[b][pred_valid_mask]
        if len(true_valid) == 0:
            matched_confidences[b] = pred_confidences[b]
            continue
        if len(pred_valid) == 0:
            continue
        matched_pairs = _compute_box_assignment(true_valid, pred_valid, matching_type)
        matched_conf = np.zeros(max_sources, dtype=pred_confidences.dtype)
        for pred_idx, true_idx in matched_pairs:
            true_pos = np.where(true_valid_mask)[0][true_idx]
            matched_conf[true_pos] = pred_conf_valid[pred_idx]
        matched_confidences[b] = matched_conf
    
    return matched_confidences


# --------------------------------
# Evaluate single fold
# --------------------------------
def evaluate_single_fold(
    max_sources: int,
    res_dir: Path,
    file_label: str,
    kfold_str: str,
    model_suffix: str = "",
    pad_value: int = -1,
    verbose: bool = True,
):
    """Evaluate a single fold and return metrics."""
    # Load test data
    npz_path = res_dir / f"data_labels_test_{file_label}_kf{kfold_str}.npz"
    if not npz_path.exists():
        if verbose:
            print(f"Test data file not found: {npz_path}")
        return None

    # Load model
    model_path = res_dir / f"model_{file_label}{model_suffix}_kf{kfold_str}.keras"
    if not model_path.exists():
        if verbose:
            print(f"Model not found: {model_path}")
        return None

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating fold {kfold_str}")
        print(f"{'='*60}")
        print(f"Loading test data from: {npz_path}")
        print(f"Loading model from: {model_path}")

    with np.load(npz_path) as npz_file:
        data_test = npz_file["data"]
        labels_test_raw = npz_file["labels"]

    if verbose:
        print(f"Test data shape: {data_test.shape}")
        print(f"Test labels shape (raw): {labels_test_raw.shape}")

    # Extract boxes from labels if they have confidence dimension (7 dims -> 6 dims)
    if labels_test_raw.shape[-1] == 7:
        labels_test = labels_test_raw[..., :6]  # Extract boxes only
        if verbose:
            print(f"Labels have confidence dimension, extracting boxes: {labels_test.shape}")
    else:
        labels_test = labels_test_raw

    if verbose:
        print(f"Test labels shape (after extraction): {labels_test.shape}")

    # Load model without compiling
    model = tf.keras.models.load_model(model_path, compile=False)

    if verbose:
        print("Making predictions on test set...")
    # Make predictions on test set
    pred_raw = model.predict(data_test, verbose=0)
    
    if verbose:
        print(f"Raw predictions shape: {pred_raw.shape}")
        print("\nExtracting boxes from predictions (YOLO-style if applicable)...")
    
    # Extract boxes from YOLO-style output if needed (7 dims -> 6 dims)
    pred, pred_confidences = extract_boxes_from_yolo_output(pred_raw, confidence_threshold=0.5, pad_value=pad_value)
    
    if verbose:
        print(f"Boxes shape after extraction: {pred.shape}")
        if pred_confidences is not None:
            print(f"Confidence scores available: min={pred_confidences.min():.3f}, max={pred_confidences.max():.3f}, mean={pred_confidences.mean():.3f}")
        print("\nApplying matching to predictions...")
    
    # Apply matching to predictions (it does both JV and Greedy)
    pred_matched_jv = apply_jv_matching(labels_test, pred, pad_value=pad_value)
    pred_matched_greedy = apply_greedy_matching(labels_test, pred, pad_value=pad_value)
    
    # Match confidence scores to ground truth slots using the same matching as boxes
    # Set to greedy, ensure it matches the box matching
    pred_confidences_matched = None
    if pred_confidences is not None:
        pred_confidences_matched = match_confidence_scores(
            labels_test, pred, pred_confidences, pad_value=pad_value, matching_type="greedy"
        )

    # --------------------------------
    # IoU from predictions vs labels
    # --------------------------------
    iou_no_matching, iou_std_nm, iou_count_nm = compute_iou_per_slot_stats(labels_test, pred, pad_value=pad_value)
    iou_with_jv, iou_std_jv, iou_count_jv = compute_iou_per_slot_stats(labels_test, pred_matched_jv, pad_value=pad_value)
    iou_with_greedy, iou_std_gr, iou_count_greedy = compute_iou_per_slot_stats(labels_test, pred_matched_greedy, pad_value=pad_value)

    # --------------------------------
    # Calculate R² scores
    # --------------------------------
    truth_flat = labels_test.reshape(labels_test.shape[0], -1)
    pred_flat = pred.reshape(pred.shape[0], -1)
    pred_matched_jv_flat = pred_matched_jv.reshape(pred_matched_jv.shape[0], -1)
    pred_matched_greedy_flat = pred_matched_greedy.reshape(pred_matched_greedy.shape[0], -1)

    mask_no_pad = (truth_flat != pad_value) & (pred_flat != pad_value)
    mask_jv = (truth_flat != pad_value) & (pred_matched_jv_flat != pad_value)
    mask_greedy = (truth_flat != pad_value) & (pred_matched_greedy_flat != pad_value)
    
    # R² on valid slots only
    r2_no_matching = r2_score(truth_flat[mask_no_pad], pred_flat[mask_no_pad]) if np.any(mask_no_pad) else np.nan
    r2_with_jv = r2_score(truth_flat[mask_jv], pred_matched_jv_flat[mask_jv]) if np.any(mask_jv) else np.nan
    r2_with_greedy = r2_score(truth_flat[mask_greedy], pred_matched_greedy_flat[mask_greedy]) if np.any(mask_greedy) else np.nan
    
    # R² on all slots (including padded slots)
    r2_all_slots_no_matching = r2_score(truth_flat.flatten(), pred_flat.flatten())
    r2_all_slots_jv = r2_score(truth_flat.flatten(), pred_matched_jv_flat.flatten())
    r2_all_slots_greedy = r2_score(truth_flat.flatten(), pred_matched_greedy_flat.flatten())

    # Per-dimension R² (Greedy)
    per_dim_results = []
    for i in range(min(max_sources * 6, truth_flat.shape[1])):
        mask = (truth_flat[:, i] != pad_value) & (pred_matched_greedy_flat[:, i] != pad_value)
        if np.sum(mask) > 0:
            r2_dim = r2_score(truth_flat[mask, i], pred_matched_greedy_flat[mask, i])
        else:
            r2_dim = np.nan
        per_dim_results.append(r2_dim)

    # Per-dimension R² (JV)
    per_dim_results_jv = []
    for i in range(min(max_sources * 6, truth_flat.shape[1])):
        mask = (truth_flat[:, i] != pad_value) & (pred_matched_jv_flat[:, i] != pad_value)
        if np.sum(mask) > 0:
            r2_dim = r2_score(truth_flat[mask, i], pred_matched_jv_flat[mask, i])
        else:
            r2_dim = np.nan
        per_dim_results_jv.append(r2_dim)

    # Calculate R² for widths and positions separately
    # Format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
    width_indices = [i for i in range(min(max_sources * 6, truth_flat.shape[1])) if i % 6 in [0, 2, 4]]
    position_indices = [i for i in range(min(max_sources * 6, truth_flat.shape[1])) if i % 6 in [1, 3, 5]]  # centres
    
    # For widths: combine all width dimensions and calculate R²
    widths_truth_greedy = []
    widths_pred_greedy = []
    widths_truth_jv = []
    widths_pred_jv = []
    
    for idx in width_indices:
        mask = (truth_flat[:, idx] != pad_value) & (pred_matched_greedy_flat[:, idx] != pad_value)
        if np.any(mask):
            widths_truth_greedy.extend(truth_flat[mask, idx])
            widths_pred_greedy.extend(pred_matched_greedy_flat[mask, idx])
        
        mask_j = (truth_flat[:, idx] != pad_value) & (pred_matched_jv_flat[:, idx] != pad_value)
        if np.any(mask_j):
            widths_truth_jv.extend(truth_flat[mask_j, idx])
            widths_pred_jv.extend(pred_matched_jv_flat[mask_j, idx])
    
    r2_widths_greedy = r2_score(widths_truth_greedy, widths_pred_greedy) if len(widths_truth_greedy) > 0 else np.nan
    r2_widths_jv = r2_score(widths_truth_jv, widths_pred_jv) if len(widths_truth_jv) > 0 else np.nan
    
    # For positions: combine all position dimensions and calculate R²
    positions_truth_greedy = []
    positions_pred_greedy = []
    positions_truth_jv = []
    positions_pred_jv = []
    
    for idx in position_indices:
        mask = (truth_flat[:, idx] != pad_value) & (pred_matched_greedy_flat[:, idx] != pad_value)
        if np.any(mask):
            positions_truth_greedy.extend(truth_flat[mask, idx])
            positions_pred_greedy.extend(pred_matched_greedy_flat[mask, idx])
        
        mask_j = (truth_flat[:, idx] != pad_value) & (pred_matched_jv_flat[:, idx] != pad_value)
        if np.any(mask_j):
            positions_truth_jv.extend(truth_flat[mask_j, idx])
            positions_pred_jv.extend(pred_matched_jv_flat[mask_j, idx])
    
    r2_positions_greedy = r2_score(positions_truth_greedy, positions_pred_greedy) if len(positions_truth_greedy) > 0 else np.nan
    r2_positions_jv = r2_score(positions_truth_jv, positions_pred_jv) if len(positions_truth_jv) > 0 else np.nan

    # ------------------------------------------------------------------
    # R² scatter plots (truth vs prediction) for this fold (Greedy)
    # ------------------------------------------------------------------
    try:
        overall_truth = truth_flat[mask_greedy] if np.any(mask_greedy) else np.array([])
        overall_pred = pred_matched_greedy_flat[mask_greedy] if np.any(mask_greedy) else np.array([])
        widths_truth_arr = np.asarray(widths_truth_greedy) if len(widths_truth_greedy) > 0 else np.array([])
        widths_pred_arr = np.asarray(widths_pred_greedy) if len(widths_pred_greedy) > 0 else np.array([])
        positions_truth_arr = np.asarray(positions_truth_greedy) if len(positions_truth_greedy) > 0 else np.array([])
        positions_pred_arr = np.asarray(positions_pred_greedy) if len(positions_pred_greedy) > 0 else np.array([])
        scatter_path = res_dir / f"r2_scatter_{file_label}_kf{kfold_str}_greedy.pdf"
        _save_r2_scatter_plot(
            scatter_path,
            overall_truth, overall_pred,
            widths_truth_arr, widths_pred_arr,
            positions_truth_arr, positions_pred_arr,
            r2_with_greedy, r2_widths_greedy, r2_positions_greedy,
            f"Truth vs Prediction (Greedy matching) - Fold {kfold_str}",
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"Warning: Could not generate R² scatter plot for fold {kfold_str}: {e}")

    # --------------------------------
    # Source count metrics
    # --------------------------------
    # Calculate source count metrics
    def count_valid_sources(boxes, pad_value=-1):
        """Count number of valid (non-padded) sources in each sample."""
        batch_size = boxes.shape[0]
        counts = []
        for b in range(batch_size):
            valid_mask = ~np.all(boxes[b] == pad_value, axis=-1)
            counts.append(np.sum(valid_mask))
        return np.array(counts)
    
    true_source_counts = count_valid_sources(labels_test, pad_value)
    pred_source_counts_no_matching = count_valid_sources(pred, pad_value)
    pred_source_counts_jv = count_valid_sources(pred_matched_jv, pad_value)
    pred_source_counts_greedy = count_valid_sources(pred_matched_greedy, pad_value)
    
    # Source count accuracy (exact match)
    source_count_accuracy_no_matching = np.mean(true_source_counts == pred_source_counts_no_matching)
    source_count_accuracy_jv = np.mean(true_source_counts == pred_source_counts_jv)
    source_count_accuracy_greedy = np.mean(true_source_counts == pred_source_counts_greedy)
    
    # Source count MAE
    source_count_mae_no_matching = np.mean(np.abs(true_source_counts - pred_source_counts_no_matching))
    source_count_mae_jv = np.mean(np.abs(true_source_counts - pred_source_counts_jv))
    source_count_mae_greedy = np.mean(np.abs(true_source_counts - pred_source_counts_greedy))
    
    # Source count mean error
    source_count_mean_error_no_matching = np.mean(pred_source_counts_no_matching - true_source_counts)
    source_count_mean_error_jv = np.mean(pred_source_counts_jv - true_source_counts)
    source_count_mean_error_greedy = np.mean(pred_source_counts_greedy - true_source_counts)
    
    # Source count classification metrics (precision, recall, F1)
    # Get all possible source counts (1 to max_sources) - exclude 0 as it doesn't exist
    all_labels = np.arange(1, max_sources + 1)
    
    # Calculate classification metrics for each matching strategy
    def calc_classification_metrics(y_true, y_pred, labels, strategy_name):
        """Calculate precision, recall, F1 (both macro and weighted), and confusion matrix."""
        # Ensure predictions are within valid range
        y_pred_clipped = np.clip(y_pred, 0, max_sources)
        
        # Calculate metrics with average='weighted' for multi-class
        precision_weighted = precision_score(y_true, y_pred_clipped, labels=labels, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred_clipped, labels=labels, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred_clipped, labels=labels, average='weighted', zero_division=0)
        
        # Calculate metrics with average='macro' for multi-class
        precision_macro = precision_score(y_true, y_pred_clipped, labels=labels, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred_clipped, labels=labels, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred_clipped, labels=labels, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred_clipped, labels=labels, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred_clipped, labels=labels, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred_clipped, labels=labels, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_clipped, labels=labels)
        
        return {
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
        }
    
    source_count_metrics_no_matching = calc_classification_metrics(
        true_source_counts, pred_source_counts_no_matching, all_labels, 'no_matching'
    )
    source_count_metrics_jv = calc_classification_metrics(
        true_source_counts, pred_source_counts_jv, all_labels, 'jv'
    )
    source_count_metrics_greedy = calc_classification_metrics(
        true_source_counts, pred_source_counts_greedy, all_labels, 'greedy'
    )
    
    if verbose:
        print(f"\nSource Count Classification Metrics (Greedy):")
        print(f"  Precision (weighted): {source_count_metrics_greedy['precision_weighted']:.4f}")
        print(f"  Precision (macro): {source_count_metrics_greedy['precision_macro']:.4f}")
        print(f"  Recall (weighted): {source_count_metrics_greedy['recall_weighted']:.4f}")
        print(f"  Recall (macro): {source_count_metrics_greedy['recall_macro']:.4f}")
        print(f"  F1 (weighted): {source_count_metrics_greedy['f1_weighted']:.4f}")
        print(f"  F1 (macro): {source_count_metrics_greedy['f1_macro']:.4f}")
        print(f"  Per-class Precision: {[f'{x:.3f}' for x in source_count_metrics_greedy['precision_per_class']]}")
        print(f"  Per-class Recall: {[f'{x:.3f}' for x in source_count_metrics_greedy['recall_per_class']]}")
        print(f"  Per-class F1: {[f'{x:.3f}' for x in source_count_metrics_greedy['f1_per_class']]}")
        
        # Print confusion matrix for source count
        cm = np.array(source_count_metrics_greedy['confusion_matrix'])
        print(f"\nSource Count Confusion Matrix (Greedy) - rows=true, cols=predicted:")
        print(f"        Predicted")
        print(f"        ", end="")
        for i in range(len(all_labels)):
            print(f"{all_labels[i]:6d}", end="")
        print()
        for i in range(len(all_labels)):
            print(f"True {all_labels[i]:2d}  ", end="")
            for j in range(len(all_labels)):
                print(f"{cm[i,j]:6d}", end="")
            print()
        
        # Print class distribution
        print(f"\n  Class distribution:")
        for i, label in enumerate(all_labels):
            true_count = np.sum(true_source_counts == label)
            pred_count = np.sum(pred_source_counts_greedy == label)
            print(f"    Class {label}: True={true_count:5d}, Predicted={pred_count:5d}")
        
        # Explain macro vs weighted
        print(f"\n  Note: Macro metrics average equally across all classes.")
        print(f"        Weighted metrics weight by class frequency.")
        print(f"        If some classes are rare, macro can be lower than weighted.")
    
    # Empty slot prediction accuracy
    def empty_slot_accuracy(true_boxes, pred_boxes, pad_value=-1):
        """Calculate accuracy of predicting pad_value for empty slots."""
        batch_size = true_boxes.shape[0]
        empty_slot_correct = []
        for b in range(batch_size):
            true_empty_mask = np.all(true_boxes[b] == pad_value, axis=-1)
            if np.any(true_empty_mask):
                pred_empty_slots = pred_boxes[b][true_empty_mask]
                pred_is_pad = np.all(pred_empty_slots == pad_value, axis=-1)
                empty_slot_correct.append(np.mean(pred_is_pad))
        return np.mean(empty_slot_correct) if empty_slot_correct else np.nan
    
    empty_slot_acc_no_matching = empty_slot_accuracy(labels_test, pred, pad_value)
    empty_slot_acc_jv = empty_slot_accuracy(labels_test, pred_matched_jv, pad_value)
    empty_slot_acc_greedy = empty_slot_accuracy(labels_test, pred_matched_greedy, pad_value)
    
    # --------------------------------
    # Absolute errors for overall metrics
    # --------------------------------
    # Calculate absolute errors for overall metrics
    abs_errors_no_matching = np.abs(truth_flat[mask_no_pad] - pred_flat[mask_no_pad]) if np.any(mask_no_pad) else np.array([])
    abs_errors_jv = np.abs(truth_flat[mask_jv] - pred_matched_jv_flat[mask_jv]) if np.any(mask_jv) else np.array([])
    abs_errors_greedy = np.abs(truth_flat[mask_greedy] - pred_matched_greedy_flat[mask_greedy]) if np.any(mask_greedy) else np.array([])

    # Overall error metrics (Greedy)
    mae_greedy = np.mean(abs_errors_greedy) if len(abs_errors_greedy) > 0 else np.nan
    median_ae_greedy = np.median(abs_errors_greedy) if len(abs_errors_greedy) > 0 else np.nan
    min_ae_greedy = np.min(abs_errors_greedy) if len(abs_errors_greedy) > 0 else np.nan
    max_ae_greedy = np.max(abs_errors_greedy) if len(abs_errors_greedy) > 0 else np.nan

    # Overall error metrics (JV)
    mae_jv = np.mean(abs_errors_jv) if len(abs_errors_jv) > 0 else np.nan
    median_ae_jv = np.median(abs_errors_jv) if len(abs_errors_jv) > 0 else np.nan
    min_ae_jv = np.min(abs_errors_jv) if len(abs_errors_jv) > 0 else np.nan
    max_ae_jv = np.max(abs_errors_jv) if len(abs_errors_jv) > 0 else np.nan

    # Overall error metrics (No matching)
    mae_no_matching = np.mean(abs_errors_no_matching) if len(abs_errors_no_matching) > 0 else np.nan
    median_ae_no_matching = np.median(abs_errors_no_matching) if len(abs_errors_no_matching) > 0 else np.nan
    min_ae_no_matching = np.min(abs_errors_no_matching) if len(abs_errors_no_matching) > 0 else np.nan
    max_ae_no_matching = np.max(abs_errors_no_matching) if len(abs_errors_no_matching) > 0 else np.nan

    # Per-dimension error metrics (Greedy)
    per_dim_mae_greedy = []
    per_dim_median_ae_greedy = []
    per_dim_min_ae_greedy = []
    per_dim_max_ae_greedy = []
    for i in range(min(max_sources * 6, truth_flat.shape[1])):
        mask = (truth_flat[:, i] != pad_value) & (pred_matched_greedy_flat[:, i] != pad_value)
        if np.sum(mask) > 0:
            abs_errors = np.abs(truth_flat[mask, i] - pred_matched_greedy_flat[mask, i])
            per_dim_mae_greedy.append(np.mean(abs_errors))
            per_dim_median_ae_greedy.append(np.median(abs_errors))
            per_dim_min_ae_greedy.append(np.min(abs_errors))
            per_dim_max_ae_greedy.append(np.max(abs_errors))
        else:
            per_dim_mae_greedy.append(np.nan)
            per_dim_median_ae_greedy.append(np.nan)
            per_dim_min_ae_greedy.append(np.nan)
            per_dim_max_ae_greedy.append(np.nan)

    # Per-dimension error metrics (JV)
    per_dim_mae_jv = []
    per_dim_median_ae_jv = []
    per_dim_min_ae_jv = []
    per_dim_max_ae_jv = []
    for i in range(min(max_sources * 6, truth_flat.shape[1])):
        mask = (truth_flat[:, i] != pad_value) & (pred_matched_jv_flat[:, i] != pad_value)
        if np.sum(mask) > 0:
            abs_errors = np.abs(truth_flat[mask, i] - pred_matched_jv_flat[mask, i])
            per_dim_mae_jv.append(np.mean(abs_errors))
            per_dim_median_ae_jv.append(np.median(abs_errors))
            per_dim_min_ae_jv.append(np.min(abs_errors))
            per_dim_max_ae_jv.append(np.max(abs_errors))
        else:
            per_dim_mae_jv.append(np.nan)
            per_dim_median_ae_jv.append(np.nan)
            per_dim_min_ae_jv.append(np.nan)
            per_dim_max_ae_jv.append(np.nan)

    # Confidence metrics (if available)
    confidence_metrics = {}
    if pred_confidences_matched is not None:
        # Create ground truth confidence: 1.0 for valid slots, 0.0 for empty slots
        true_confidences = np.zeros((labels_test.shape[0], labels_test.shape[1]), dtype=np.float32)
        for b in range(labels_test.shape[0]):
            for s in range(labels_test.shape[1]):
                is_valid = not np.all(labels_test[b, s] == pad_value)
                true_confidences[b, s] = 1.0 if is_valid else 0.0
        
        # Binary classification metrics for confidence
        # Use MATCHED confidence scores (aligned with ground truth slots)
        
        # Flatten for sklearn metrics
        true_conf_flat = true_confidences.flatten()
        pred_conf_flat = pred_confidences_matched.flatten()
        pred_conf_binary = (pred_conf_flat >= 0.5).astype(int)
        true_conf_binary = true_conf_flat.astype(int)
        
        # Confusion matrix for confidence
        conf_cm = confusion_matrix(true_conf_binary, pred_conf_binary, labels=[0, 1])
        
        confidence_metrics = {
            'confidence_accuracy': accuracy_score(true_conf_binary, pred_conf_binary),
            'confidence_precision': precision_score(true_conf_binary, pred_conf_binary, zero_division=0),
            'confidence_recall': recall_score(true_conf_binary, pred_conf_binary, zero_division=0),
            'confidence_f1': f1_score(true_conf_binary, pred_conf_binary, zero_division=0),
            'confidence_mean_pred': float(pred_conf_flat.mean()),
            'confidence_mean_true': float(true_conf_flat.mean()),
            'confidence_auc': roc_auc_score(true_conf_binary, pred_conf_flat) if len(np.unique(true_conf_binary)) > 1 else np.nan,
            'confidence_confusion_matrix': conf_cm.tolist(),
        }
        
        
        if verbose:
            print(f"\nConfidence Prediction Metrics:")
            print(f"  Accuracy: {confidence_metrics['confidence_accuracy']:.4f}")
            print(f"  Precision: {confidence_metrics['confidence_precision']:.4f}")
            print(f"  Recall: {confidence_metrics['confidence_recall']:.4f}")
            print(f"  F1: {confidence_metrics['confidence_f1']:.4f}")
            print(f"  AUC-ROC: {confidence_metrics['confidence_auc']:.4f}")
            print(f"  Mean predicted confidence: {confidence_metrics['confidence_mean_pred']:.4f}")
            print(f"  Mean true confidence: {confidence_metrics['confidence_mean_true']:.4f}")
            print(f"\nConfidence Confusion Matrix (rows=true, cols=predicted):")
            print(f"                    Predicted")
            print(f"                  Empty  Valid")
            print(f"  True Empty    {conf_cm[0,0]:6d}  {conf_cm[0,1]:6d}")
            print(f"  True Valid    {conf_cm[1,0]:6d}  {conf_cm[1,1]:6d}")
            print(f"\n  Total samples: {len(true_conf_binary)}")
            print(f"  True empty slots: {np.sum(true_conf_binary == 0)}")
            print(f"  True valid slots: {np.sum(true_conf_binary == 1)}")
            print(f"  Predicted empty: {np.sum(pred_conf_binary == 0)}")
            print(f"  Predicted valid: {np.sum(pred_conf_binary == 1)}")
            
            # Verify metrics from confusion matrix
            tn, fp, fn, tp = conf_cm[0,0], conf_cm[0,1], conf_cm[1,0], conf_cm[1,1]
            if tp + fp > 0:
                precision_from_cm = tp / (tp + fp)
            else:
                precision_from_cm = 0.0
            if tp + fn > 0:
                recall_from_cm = tp / (tp + fn)
            else:
                recall_from_cm = 0.0
            if precision_from_cm + recall_from_cm > 0:
                f1_from_cm = 2 * (precision_from_cm * recall_from_cm) / (precision_from_cm + recall_from_cm)
            else:
                f1_from_cm = 0.0
            accuracy_from_cm = (tn + tp) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0.0
            
            print(f"\n  Metrics calculated from confusion matrix:")
            print(f"    Precision: {precision_from_cm:.4f} (TP={tp} / (TP={tp} + FP={fp}))")
            print(f"    Recall: {recall_from_cm:.4f} (TP={tp} / (TP={tp} + FN={fn}))")
            print(f"    F1: {f1_from_cm:.4f}")
            print(f"    Accuracy: {accuracy_from_cm:.4f} ((TN={tn} + TP={tp}) / Total={tn+fp+fn+tp})")
        
        # Plot confidence distribution
        try:
            confidence_plot_path = res_dir / f"confidence_distribution_{file_label}_kf{kfold_str}.pdf"
            PLACENetPlot.plot_confidence_distribution(
                pred_confidences=pred_confidences_matched,
                true_confidences=true_confidences,
                output_path=confidence_plot_path,
                title=f"Confidence Score Distribution - Fold {kfold_str}"
            )
            
            # Also create separate "Distribution by Slot Type" plot with mean/median lines
            if pred_confidences_matched is not None and true_confidences is not None:
                slot_type_plot_path = res_dir / f"confidence_by_slot_type_{file_label}_kf{kfold_str}.pdf"
                PLACENetPlot.plot_confidence_by_slot_type(
                    pred_confidences=pred_confidences_matched,
                    true_confidences=true_confidences,
                    output_path=slot_type_plot_path,
                    title=f"Confidence Distribution by Slot Type - Fold {kfold_str}"
                )
                
                # Create diagnostic plots (ROC, PR, threshold analysis)
                diagnostics_plot_path = res_dir / f"confidence_diagnostics_{file_label}_kf{kfold_str}.pdf"
                PLACENetPlot.plot_confidence_diagnostics(
                    pred_confidences=pred_confidences_matched,
                    true_confidences=true_confidences,
                    output_path=diagnostics_plot_path,
                    title=f"Confidence Classification Diagnostics - Fold {kfold_str}",
                    threshold=0.5
                )
        except Exception as e:
            if verbose:
                print(f"Warning: Could not generate confidence distribution plot: {e}")

    if verbose:
        print(f"\nFold {kfold_str} Results:")
        print(f"  R² (No matching): {r2_no_matching:.4f}")
        print(f"  R² (JV matching): {r2_with_jv:.4f}")
        print(f"  R² (Greedy matching): {r2_with_greedy:.4f}")
        print(f"  R² - All Slots (No matching): {r2_all_slots_no_matching:.4f}")
        print(f"  R² - All Slots (JV matching): {r2_all_slots_jv:.4f}")
        print(f"  R² - All Slots (Greedy matching): {r2_all_slots_greedy:.4f}")
        print(f"  IoU (No matching): {iou_no_matching:.4f}")
        print(f"  IoU (JV matching): {iou_with_jv:.4f}")
        print(f"  IoU (Greedy matching): {iou_with_greedy:.4f}")

    return {
        'fold': kfold_str,
        'npz_path': str(npz_path),
        'model_path': str(model_path),
        'r2_no_matching': r2_no_matching,
        'r2_with_jv': r2_with_jv,
        'r2_with_greedy': r2_with_greedy,
        'r2_all_slots_no_matching': r2_all_slots_no_matching,
        'r2_all_slots_jv': r2_all_slots_jv,
        'r2_all_slots_greedy': r2_all_slots_greedy,
        'r2_widths_greedy': r2_widths_greedy,
        'r2_widths_jv': r2_widths_jv,
        'r2_positions_greedy': r2_positions_greedy,
        'r2_positions_jv': r2_positions_jv,
        'iou_no_matching': iou_no_matching,
        'iou_with_jv': iou_with_jv,
        'iou_with_greedy': iou_with_greedy,
        'iou_std_no_matching': iou_std_nm,
        'iou_count_no_matching': iou_count_nm,
        'iou_std_jv': iou_std_jv,
        'iou_count_jv': iou_count_jv,
        'iou_std_greedy': iou_std_gr,
        'iou_count_greedy': iou_count_greedy,
        'labels_test': labels_test,
        'pred_greedy': pred_matched_greedy,
        'pred_jv': pred_matched_jv,
        'pred_no_matching': pred,
        'per_dim_r2_greedy': per_dim_results,
        'per_dim_r2_jv': per_dim_results_jv,
        'mae_no_matching': mae_no_matching,
        'mae_jv': mae_jv,
        'mae_greedy': mae_greedy,
        'median_ae_no_matching': median_ae_no_matching,
        'median_ae_jv': median_ae_jv,
        'median_ae_greedy': median_ae_greedy,
        'min_ae_no_matching': min_ae_no_matching,
        'min_ae_jv': min_ae_jv,
        'min_ae_greedy': min_ae_greedy,
        'max_ae_no_matching': max_ae_no_matching,
        'max_ae_jv': max_ae_jv,
        'max_ae_greedy': max_ae_greedy,
        'per_dim_mae_greedy': per_dim_mae_greedy,
        'per_dim_median_ae_greedy': per_dim_median_ae_greedy,
        'per_dim_min_ae_greedy': per_dim_min_ae_greedy,
        'per_dim_max_ae_greedy': per_dim_max_ae_greedy,
        'per_dim_mae_jv': per_dim_mae_jv,
        'per_dim_median_ae_jv': per_dim_median_ae_jv,
        'per_dim_min_ae_jv': per_dim_min_ae_jv,
        'per_dim_max_ae_jv': per_dim_max_ae_jv,
        'source_count_accuracy_no_matching': source_count_accuracy_no_matching,
        'source_count_accuracy_jv': source_count_accuracy_jv,
        'source_count_accuracy_greedy': source_count_accuracy_greedy,
        'source_count_mae_no_matching': source_count_mae_no_matching,
        'source_count_mae_jv': source_count_mae_jv,
        'source_count_mae_greedy': source_count_mae_greedy,
        'source_count_mean_error_no_matching': source_count_mean_error_no_matching,
        'source_count_mean_error_jv': source_count_mean_error_jv,
        'source_count_mean_error_greedy': source_count_mean_error_greedy,
        'source_count_precision_weighted_no_matching': source_count_metrics_no_matching['precision_weighted'],
        'source_count_precision_weighted_jv': source_count_metrics_jv['precision_weighted'],
        'source_count_precision_weighted_greedy': source_count_metrics_greedy['precision_weighted'],
        'source_count_precision_macro_no_matching': source_count_metrics_no_matching['precision_macro'],
        'source_count_precision_macro_jv': source_count_metrics_jv['precision_macro'],
        'source_count_precision_macro_greedy': source_count_metrics_greedy['precision_macro'],
        'source_count_recall_weighted_no_matching': source_count_metrics_no_matching['recall_weighted'],
        'source_count_recall_weighted_jv': source_count_metrics_jv['recall_weighted'],
        'source_count_recall_weighted_greedy': source_count_metrics_greedy['recall_weighted'],
        'source_count_recall_macro_no_matching': source_count_metrics_no_matching['recall_macro'],
        'source_count_recall_macro_jv': source_count_metrics_jv['recall_macro'],
        'source_count_recall_macro_greedy': source_count_metrics_greedy['recall_macro'],
        'source_count_f1_weighted_no_matching': source_count_metrics_no_matching['f1_weighted'],
        'source_count_f1_weighted_jv': source_count_metrics_jv['f1_weighted'],
        'source_count_f1_weighted_greedy': source_count_metrics_greedy['f1_weighted'],
        'source_count_f1_macro_no_matching': source_count_metrics_no_matching['f1_macro'],
        'source_count_f1_macro_jv': source_count_metrics_jv['f1_macro'],
        'source_count_f1_macro_greedy': source_count_metrics_greedy['f1_macro'],
        'source_count_precision_per_class_no_matching': source_count_metrics_no_matching['precision_per_class'],
        'source_count_precision_per_class_jv': source_count_metrics_jv['precision_per_class'],
        'source_count_precision_per_class_greedy': source_count_metrics_greedy['precision_per_class'],
        'source_count_recall_per_class_no_matching': source_count_metrics_no_matching['recall_per_class'],
        'source_count_recall_per_class_jv': source_count_metrics_jv['recall_per_class'],
        'source_count_recall_per_class_greedy': source_count_metrics_greedy['recall_per_class'],
        'source_count_f1_per_class_no_matching': source_count_metrics_no_matching['f1_per_class'],
        'source_count_f1_per_class_jv': source_count_metrics_jv['f1_per_class'],
        'source_count_f1_per_class_greedy': source_count_metrics_greedy['f1_per_class'],
        'true_mean_source_count': float(np.mean(true_source_counts)),
        'true_std_source_count': float(np.std(true_source_counts)),
        'pred_mean_source_count_no_matching': float(np.mean(pred_source_counts_no_matching)),
        'pred_mean_source_count_jv': float(np.mean(pred_source_counts_jv)),
        'pred_mean_source_count_greedy': float(np.mean(pred_source_counts_greedy)),
        'empty_slot_accuracy_no_matching': empty_slot_acc_no_matching,
        'empty_slot_accuracy_jv': empty_slot_acc_jv,
        'empty_slot_accuracy_greedy': empty_slot_acc_greedy,
        # Return raw source counts for combining across folds
        'true_source_counts': true_source_counts.tolist(),
        'pred_source_counts_no_matching': pred_source_counts_no_matching.tolist(),
        'pred_source_counts_jv': pred_source_counts_jv.tolist(),
        'pred_source_counts_greedy': pred_source_counts_greedy.tolist(),
        **confidence_metrics,
    }

# --------------------------------
# Evaluate run for all folds
# --------------------------------
def evaluate_run(
    max_sources: int,
    res_dir: Path,
    file_label: str,
    model_suffix: str = "",
    log_name: str = "r2_scores.log",
):
    """Evaluate saved models on their corresponding test splits and print metrics for all folds."""
    suffix_display = model_suffix or ""
    print(50 * "-")
    print(f"Testing model{suffix_display} (R² score on test set - all folds)")
    print(50 * "-")

    # Find all test data files
    npz_candidates = sorted(
        glob.glob(str(res_dir / f"data_labels_test_{file_label}_kf*.npz")),
    )

    if not npz_candidates:
        print(f"No test data files found in {res_dir}. Train the model first.")
        return

    # Extract fold numbers
    fold_numbers = []
    for npz_path in npz_candidates:
        m = re.search(r"_kf(\d+)\.npz$", npz_path)
        if m:
            fold_numbers.append(m.group(1))
    
    if not fold_numbers:
        print(f"Could not extract fold numbers from test files.")
        return

    fold_numbers = sorted(fold_numbers, key=int)
    print(f"Found {len(fold_numbers)} folds: {fold_numbers}")

    pad_value = -1
    all_results = []

    # Evaluate each fold
    for kfold_str in fold_numbers:
        result = evaluate_single_fold(
            max_sources=max_sources,
            res_dir=res_dir,
            file_label=file_label,
            kfold_str=kfold_str,
            model_suffix=model_suffix,
            pad_value=pad_value,
            verbose=True,
        )
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("No valid results from any fold.")
        return

    # IoU over all folds in one go: concatenate labels and predictions, then one mean/std/SEM
    _pad = -1
    iou_combined_mean_greedy = iou_combined_mean_jv = iou_combined_mean_no_matching = np.nan
    iou_combined_std_greedy = iou_combined_std_jv = iou_combined_std_no_matching = np.nan
    iou_combined_sem_greedy = iou_combined_sem_jv = iou_combined_sem_no_matching = np.nan
    iou_combined_N_greedy = iou_combined_N_jv = iou_combined_N_no_matching = 0
    if all(r.get('labels_test') is not None for r in all_results):
        all_labels = np.concatenate([r['labels_test'] for r in all_results], axis=0)
        all_pred_greedy = np.concatenate([r['pred_greedy'] for r in all_results], axis=0)
        all_pred_jv = np.concatenate([r['pred_jv'] for r in all_results], axis=0)
        all_pred_nm = np.concatenate([r['pred_no_matching'] for r in all_results], axis=0)
        m_gr, s_gr, n_gr = compute_iou_per_slot_stats(all_labels, all_pred_greedy, pad_value=_pad)
        m_jv, s_jv, n_jv = compute_iou_per_slot_stats(all_labels, all_pred_jv, pad_value=_pad)
        m_nm, s_nm, n_nm = compute_iou_per_slot_stats(all_labels, all_pred_nm, pad_value=_pad)
        iou_combined_mean_greedy, iou_combined_std_greedy = m_gr, s_gr
        iou_combined_N_greedy = n_gr
        iou_combined_sem_greedy = s_gr / np.sqrt(n_gr) if n_gr > 0 else np.nan
        iou_combined_mean_jv, iou_combined_std_jv = m_jv, s_jv
        iou_combined_N_jv = n_jv
        iou_combined_sem_jv = s_jv / np.sqrt(n_jv) if n_jv > 0 else np.nan
        iou_combined_mean_no_matching, iou_combined_std_no_matching = m_nm, s_nm
        iou_combined_N_no_matching = n_nm
        iou_combined_sem_no_matching = s_nm / np.sqrt(n_nm) if n_nm > 0 else np.nan

    # Aggregate statistics across folds
    print("\n" + "="*60)
    print("AGGREGATE RESULTS ACROSS ALL FOLDS")
    print("="*60)

    r2_greedy_all = [r['r2_with_greedy'] for r in all_results if not np.isnan(r['r2_with_greedy'])]
    r2_jv_all = [r['r2_with_jv'] for r in all_results if not np.isnan(r['r2_with_jv'])]
    r2_no_matching_all = [r['r2_no_matching'] for r in all_results if not np.isnan(r['r2_no_matching'])]
    r2_all_slots_greedy_all = [r['r2_all_slots_greedy'] for r in all_results if not np.isnan(r['r2_all_slots_greedy'])]
    r2_all_slots_jv_all = [r['r2_all_slots_jv'] for r in all_results if not np.isnan(r['r2_all_slots_jv'])]
    r2_all_slots_no_matching_all = [r['r2_all_slots_no_matching'] for r in all_results if not np.isnan(r['r2_all_slots_no_matching'])]
    iou_greedy_all = [r['iou_with_greedy'] for r in all_results if not np.isnan(r['iou_with_greedy'])]
    iou_jv_all = [r['iou_with_jv'] for r in all_results if not np.isnan(r['iou_with_jv'])]
    iou_no_matching_all = [r['iou_no_matching'] for r in all_results if not np.isnan(r['iou_no_matching'])]

    # Pooled IoU: propagate per-fold (mean, std, count) to overall mean, std, and SEM
    def pooled_iou_stats(results, mean_key, std_key, count_key):
        # Pooled mean = Σ(n_k μ_k) / N; pooled var = (1/(N-1)) Σ[(n_k-1)σ_k² + n_k(μ_k - μ_pooled)²]
        # SEM = σ_pooled / √N (use for "mean ± SEM" so the interval stays sensible; IoU ∈ [0,1])
        total_sum = 0.0
        total_count = 0.0
        total_sum_sq = 0.0
        for r in results:
            n = r.get(count_key, 0)
            if n <= 0:
                continue
            mu = r.get(mean_key, np.nan)
            sig = r.get(std_key, 0.0)
            if np.isnan(mu):
                continue
            total_sum += n * mu
            total_count += n
            total_sum_sq += (n - 1) * (sig**2) + n * (mu**2)
        if total_count <= 0:
            return np.nan, np.nan, np.nan, 0
        pooled_mean = total_sum / total_count
        var = (total_sum_sq - total_count * pooled_mean**2) / max(1, total_count - 1)
        pooled_std = np.sqrt(max(0.0, var))
        sem = pooled_std / np.sqrt(total_count) if total_count > 0 else np.nan
        return pooled_mean, pooled_std, sem, total_count

    iou_pooled_mean_greedy, iou_pooled_std_greedy, iou_pooled_sem_greedy, iou_N_greedy = pooled_iou_stats(
        all_results, 'iou_with_greedy', 'iou_std_greedy', 'iou_count_greedy')
    iou_pooled_mean_jv, iou_pooled_std_jv, iou_pooled_sem_jv, iou_N_jv = pooled_iou_stats(
        all_results, 'iou_with_jv', 'iou_std_jv', 'iou_count_jv')
    iou_pooled_mean_no_matching, iou_pooled_std_no_matching, iou_pooled_sem_no_matching, iou_N_no_matching = pooled_iou_stats(
        all_results, 'iou_no_matching', 'iou_std_no_matching', 'iou_count_no_matching')

    print(f"\nOverall R² (Greedy matching):")
    if r2_greedy_all:
        print(f"  Mean: {np.mean(r2_greedy_all):.4f}")
        print(f"  Std:  {np.std(r2_greedy_all):.4f}")
        print(f"  Min:  {np.min(r2_greedy_all):.4f}")
        print(f"  Max:  {np.max(r2_greedy_all):.4f}")

    print(f"\nOverall R² (JV matching):")
    if r2_jv_all:
        print(f"  Mean: {np.mean(r2_jv_all):.4f}")
        print(f"  Std:  {np.std(r2_jv_all):.4f}")
        print(f"  Min:  {np.min(r2_jv_all):.4f}")
        print(f"  Max:  {np.max(r2_jv_all):.4f}")

    print(f"\nOverall R² (No matching):")
    if r2_no_matching_all:
        print(f"  Mean: {np.mean(r2_no_matching_all):.4f}")
        print(f"  Std:  {np.std(r2_no_matching_all):.4f}")
        print(f"  Min:  {np.min(r2_no_matching_all):.4f}")
        print(f"  Max:  {np.max(r2_no_matching_all):.4f}")

    print(f"\nOverall R² - All Slots (Greedy matching):")
    if r2_all_slots_greedy_all:
        print(f"  Mean: {np.mean(r2_all_slots_greedy_all):.4f}")
        print(f"  Std:  {np.std(r2_all_slots_greedy_all):.4f}")
        print(f"  Min:  {np.min(r2_all_slots_greedy_all):.4f}")
        print(f"  Max:  {np.max(r2_all_slots_greedy_all):.4f}")

    print(f"\nOverall R² - All Slots (JV matching):")
    if r2_all_slots_jv_all:
        print(f"  Mean: {np.mean(r2_all_slots_jv_all):.4f}")
        print(f"  Std:  {np.std(r2_all_slots_jv_all):.4f}")
        print(f"  Min:  {np.min(r2_all_slots_jv_all):.4f}")
        print(f"  Max:  {np.max(r2_all_slots_jv_all):.4f}")

    print(f"\nOverall R² - All Slots (No matching):")
    if r2_all_slots_no_matching_all:
        print(f"  Mean: {np.mean(r2_all_slots_no_matching_all):.4f}")
        print(f"  Std:  {np.std(r2_all_slots_no_matching_all):.4f}")
        print(f"  Min:  {np.min(r2_all_slots_no_matching_all):.4f}")
        print(f"  Max:  {np.max(r2_all_slots_no_matching_all):.4f}")

    # Aggregate widths and positions R²
    r2_widths_greedy_all = [r['r2_widths_greedy'] for r in all_results if not np.isnan(r['r2_widths_greedy'])]
    r2_widths_jv_all = [r['r2_widths_jv'] for r in all_results if not np.isnan(r['r2_widths_jv'])]
    r2_positions_greedy_all = [r['r2_positions_greedy'] for r in all_results if not np.isnan(r['r2_positions_greedy'])]
    r2_positions_jv_all = [r['r2_positions_jv'] for r in all_results if not np.isnan(r['r2_positions_jv'])]

    print(f"\nR² for All Widths (xwidth, ywidth, zwidth) - Greedy matching:")
    if r2_widths_greedy_all:
        print(f"  Mean: {np.mean(r2_widths_greedy_all):.4f}")
        print(f"  Std:  {np.std(r2_widths_greedy_all):.4f}")
        print(f"  Min:  {np.min(r2_widths_greedy_all):.4f}")
        print(f"  Max:  {np.max(r2_widths_greedy_all):.4f}")

    print(f"\nR² for All Widths (xwidth, ywidth, zwidth) - JV matching:")
    if r2_widths_jv_all:
        print(f"  Mean: {np.mean(r2_widths_jv_all):.4f}")
        print(f"  Std:  {np.std(r2_widths_jv_all):.4f}")
        print(f"  Min:  {np.min(r2_widths_jv_all):.4f}")
        print(f"  Max:  {np.max(r2_widths_jv_all):.4f}")

    print(f"\nR² for All Positions (xcentre, ycentre, zcentre) - Greedy matching:")
    if r2_positions_greedy_all:
        print(f"  Mean: {np.mean(r2_positions_greedy_all):.4f}")
        print(f"  Std:  {np.std(r2_positions_greedy_all):.4f}")
        print(f"  Min:  {np.min(r2_positions_greedy_all):.4f}")
        print(f"  Max:  {np.max(r2_positions_greedy_all):.4f}")

    print(f"\nR² for All Positions (xcentre, ycentre, zcentre) - JV matching:")
    if r2_positions_jv_all:
        print(f"  Mean: {np.mean(r2_positions_jv_all):.4f}")
        print(f"  Std:  {np.std(r2_positions_jv_all):.4f}")
        print(f"  Min:  {np.min(r2_positions_jv_all):.4f}")
        print(f"  Max:  {np.max(r2_positions_jv_all):.4f}")

    print(f"\nIoU (all folds combined, one run):")
    if iou_combined_N_greedy > 0:
        print(f"  Greedy matching:    Mean ± std = {iou_combined_mean_greedy:.4f} ± {iou_combined_std_greedy:.4f}  (N={int(iou_combined_N_greedy)})")
    if iou_combined_N_jv > 0:
        print(f"  JV matching:        Mean ± std = {iou_combined_mean_jv:.4f} ± {iou_combined_std_jv:.4f}  (N={int(iou_combined_N_jv)})")
    if iou_combined_N_no_matching > 0:
        print(f"  No matching:        Mean ± std = {iou_combined_mean_no_matching:.4f} ± {iou_combined_std_no_matching:.4f}  (N={int(iou_combined_N_no_matching)})")
    if iou_combined_N_greedy == 0 and iou_greedy_all:
        # Fallback to pooled (e.g. old results without labels/preds in return dict)
        print(f"  Greedy (from fold means): Mean = {np.mean(iou_greedy_all):.4f}, Std(across folds) = {np.std(iou_greedy_all):.4f}")
        print(f"  JV (from fold means):    Mean = {np.mean(iou_jv_all):.4f}, Std(across folds) = {np.std(iou_jv_all):.4f}")
        print(f"  No matching (from fold means): Mean = {np.mean(iou_no_matching_all):.4f}, Std(across folds) = {np.std(iou_no_matching_all):.4f}")

    # Aggregate source count metrics
    source_count_accuracy_greedy_all = [r['source_count_accuracy_greedy'] for r in all_results if not np.isnan(r['source_count_accuracy_greedy'])]
    source_count_accuracy_jv_all = [r['source_count_accuracy_jv'] for r in all_results if not np.isnan(r['source_count_accuracy_jv'])]
    source_count_accuracy_no_matching_all = [r['source_count_accuracy_no_matching'] for r in all_results if not np.isnan(r['source_count_accuracy_no_matching'])]
    
    source_count_mae_greedy_all = [r['source_count_mae_greedy'] for r in all_results if not np.isnan(r['source_count_mae_greedy'])]
    source_count_mae_jv_all = [r['source_count_mae_jv'] for r in all_results if not np.isnan(r['source_count_mae_jv'])]
    source_count_mae_no_matching_all = [r['source_count_mae_no_matching'] for r in all_results if not np.isnan(r['source_count_mae_no_matching'])]
    
    source_count_mean_error_greedy_all = [r['source_count_mean_error_greedy'] for r in all_results if not np.isnan(r['source_count_mean_error_greedy'])]
    source_count_mean_error_jv_all = [r['source_count_mean_error_jv'] for r in all_results if not np.isnan(r['source_count_mean_error_jv'])]
    source_count_mean_error_no_matching_all = [r['source_count_mean_error_no_matching'] for r in all_results if not np.isnan(r['source_count_mean_error_no_matching'])]
    
    empty_slot_acc_greedy_all = [r['empty_slot_accuracy_greedy'] for r in all_results if not np.isnan(r['empty_slot_accuracy_greedy'])]
    empty_slot_acc_jv_all = [r['empty_slot_accuracy_jv'] for r in all_results if not np.isnan(r['empty_slot_accuracy_jv'])]
    empty_slot_acc_no_matching_all = [r['empty_slot_accuracy_no_matching'] for r in all_results if not np.isnan(r['empty_slot_accuracy_no_matching'])]
    
    true_mean_source_count_all = [r['true_mean_source_count'] for r in all_results if not np.isnan(r['true_mean_source_count'])]
    pred_mean_source_count_greedy_all = [r['pred_mean_source_count_greedy'] for r in all_results if not np.isnan(r['pred_mean_source_count_greedy'])]
    pred_mean_source_count_jv_all = [r['pred_mean_source_count_jv'] for r in all_results if not np.isnan(r['pred_mean_source_count_jv'])]
    pred_mean_source_count_no_matching_all = [r['pred_mean_source_count_no_matching'] for r in all_results if not np.isnan(r['pred_mean_source_count_no_matching'])]

    print(f"\nSource Count Accuracy (Greedy matching):")
    if source_count_accuracy_greedy_all:
        print(f"  Mean: {np.mean(source_count_accuracy_greedy_all):.4f} ({np.mean(source_count_accuracy_greedy_all)*100:.2f}%)")
        print(f"  Std:  {np.std(source_count_accuracy_greedy_all):.4f}")
        print(f"  Min:  {np.min(source_count_accuracy_greedy_all):.4f} ({np.min(source_count_accuracy_greedy_all)*100:.2f}%)")
        print(f"  Max:  {np.max(source_count_accuracy_greedy_all):.4f} ({np.max(source_count_accuracy_greedy_all)*100:.2f}%)")

    print(f"\nSource Count Accuracy (JV matching):")
    if source_count_accuracy_jv_all:
        print(f"  Mean: {np.mean(source_count_accuracy_jv_all):.4f} ({np.mean(source_count_accuracy_jv_all)*100:.2f}%)")
        print(f"  Std:  {np.std(source_count_accuracy_jv_all):.4f}")
        print(f"  Min:  {np.min(source_count_accuracy_jv_all):.4f} ({np.min(source_count_accuracy_jv_all)*100:.2f}%)")
        print(f"  Max:  {np.max(source_count_accuracy_jv_all):.4f} ({np.max(source_count_accuracy_jv_all)*100:.2f}%)")

    print(f"\nSource Count Accuracy (No matching):")
    if source_count_accuracy_no_matching_all:
        print(f"  Mean: {np.mean(source_count_accuracy_no_matching_all):.4f} ({np.mean(source_count_accuracy_no_matching_all)*100:.2f}%)")
        print(f"  Std:  {np.std(source_count_accuracy_no_matching_all):.4f}")
        print(f"  Min:  {np.min(source_count_accuracy_no_matching_all):.4f} ({np.min(source_count_accuracy_no_matching_all)*100:.2f}%)")
        print(f"  Max:  {np.max(source_count_accuracy_no_matching_all):.4f} ({np.max(source_count_accuracy_no_matching_all)*100:.2f}%)")

    print(f"\nSource Count MAE (Greedy matching):")
    if source_count_mae_greedy_all:
        print(f"  Mean: {np.mean(source_count_mae_greedy_all):.4f}")
        print(f"  Std:  {np.std(source_count_mae_greedy_all):.4f}")
        print(f"  Min:  {np.min(source_count_mae_greedy_all):.4f}")
        print(f"  Max:  {np.max(source_count_mae_greedy_all):.4f}")

    print(f"\nSource Count MAE (JV matching):")
    if source_count_mae_jv_all:
        print(f"  Mean: {np.mean(source_count_mae_jv_all):.4f}")
        print(f"  Std:  {np.std(source_count_mae_jv_all):.4f}")
        print(f"  Min:  {np.min(source_count_mae_jv_all):.4f}")
        print(f"  Max:  {np.max(source_count_mae_jv_all):.4f}")

    print(f"\nSource Count MAE (No matching):")
    if source_count_mae_no_matching_all:
        print(f"  Mean: {np.mean(source_count_mae_no_matching_all):.4f}")
        print(f"  Std:  {np.std(source_count_mae_no_matching_all):.4f}")
        print(f"  Min:  {np.min(source_count_mae_no_matching_all):.4f}")
        print(f"  Max:  {np.max(source_count_mae_no_matching_all):.4f}")

    print(f"\nSource Count Mean Error (Greedy matching):")
    if source_count_mean_error_greedy_all:
        print(f"  Mean: {np.mean(source_count_mean_error_greedy_all):+.4f}")
        print(f"  Std:  {np.std(source_count_mean_error_greedy_all):.4f}")
        print(f"  Min:  {np.min(source_count_mean_error_greedy_all):+.4f}")
        print(f"  Max:  {np.max(source_count_mean_error_greedy_all):+.4f}")

    print(f"\nSource Count Mean Error (JV matching):")
    if source_count_mean_error_jv_all:
        print(f"  Mean: {np.mean(source_count_mean_error_jv_all):+.4f}")
        print(f"  Std:  {np.std(source_count_mean_error_jv_all):.4f}")
        print(f"  Min:  {np.min(source_count_mean_error_jv_all):+.4f}")
        print(f"  Max:  {np.max(source_count_mean_error_jv_all):+.4f}")

    print(f"\nSource Count Mean Error (No matching):")
    if source_count_mean_error_no_matching_all:
        print(f"  Mean: {np.mean(source_count_mean_error_no_matching_all):+.4f}")
        print(f"  Std:  {np.std(source_count_mean_error_no_matching_all):.4f}")
        print(f"  Min:  {np.min(source_count_mean_error_no_matching_all):+.4f}")
        print(f"  Max:  {np.max(source_count_mean_error_no_matching_all):+.4f}")

    print(f"\nTrue Mean Source Count:")
    if true_mean_source_count_all:
        print(f"  Mean: {np.mean(true_mean_source_count_all):.4f}")
        print(f"  Std:  {np.std(true_mean_source_count_all):.4f}")

    print(f"\nPredicted Mean Source Count (Greedy matching):")
    if pred_mean_source_count_greedy_all:
        print(f"  Mean: {np.mean(pred_mean_source_count_greedy_all):.4f}")
        print(f"  Std:  {np.std(pred_mean_source_count_greedy_all):.4f}")

    print(f"\nPredicted Mean Source Count (JV matching):")
    if pred_mean_source_count_jv_all:
        print(f"  Mean: {np.mean(pred_mean_source_count_jv_all):.4f}")
        print(f"  Std:  {np.std(pred_mean_source_count_jv_all):.4f}")

    print(f"\nPredicted Mean Source Count (No matching):")
    if pred_mean_source_count_no_matching_all:
        print(f"  Mean: {np.mean(pred_mean_source_count_no_matching_all):.4f}")
        print(f"  Std:  {np.std(pred_mean_source_count_no_matching_all):.4f}")

    print(f"\nEmpty Slot Accuracy (Greedy matching):")
    if empty_slot_acc_greedy_all:
        print(f"  Mean: {np.mean(empty_slot_acc_greedy_all):.4f} ({np.mean(empty_slot_acc_greedy_all)*100:.2f}%)")
        print(f"  Std:  {np.std(empty_slot_acc_greedy_all):.4f}")

    print(f"\nEmpty Slot Accuracy (JV matching):")
    if empty_slot_acc_jv_all:
        print(f"  Mean: {np.mean(empty_slot_acc_jv_all):.4f} ({np.mean(empty_slot_acc_jv_all)*100:.2f}%)")
        print(f"  Std:  {np.std(empty_slot_acc_jv_all):.4f}")

    print(f"\nEmpty Slot Accuracy (No matching):")
    if empty_slot_acc_no_matching_all:
        print(f"  Mean: {np.mean(empty_slot_acc_no_matching_all):.4f} ({np.mean(empty_slot_acc_no_matching_all)*100:.2f}%)")
        print(f"  Std:  {np.std(empty_slot_acc_no_matching_all):.4f}")

    # Aggregate error metrics
    mae_greedy_all = [r['mae_greedy'] for r in all_results if not np.isnan(r['mae_greedy'])]
    mae_jv_all = [r['mae_jv'] for r in all_results if not np.isnan(r['mae_jv'])]
    mae_no_matching_all = [r['mae_no_matching'] for r in all_results if not np.isnan(r['mae_no_matching'])]
    
    median_ae_greedy_all = [r['median_ae_greedy'] for r in all_results if not np.isnan(r['median_ae_greedy'])]
    median_ae_jv_all = [r['median_ae_jv'] for r in all_results if not np.isnan(r['median_ae_jv'])]
    median_ae_no_matching_all = [r['median_ae_no_matching'] for r in all_results if not np.isnan(r['median_ae_no_matching'])]
    
    min_ae_greedy_all = [r['min_ae_greedy'] for r in all_results if not np.isnan(r['min_ae_greedy'])]
    min_ae_jv_all = [r['min_ae_jv'] for r in all_results if not np.isnan(r['min_ae_jv'])]
    min_ae_no_matching_all = [r['min_ae_no_matching'] for r in all_results if not np.isnan(r['min_ae_no_matching'])]
    
    max_ae_greedy_all = [r['max_ae_greedy'] for r in all_results if not np.isnan(r['max_ae_greedy'])]
    max_ae_jv_all = [r['max_ae_jv'] for r in all_results if not np.isnan(r['max_ae_jv'])]
    max_ae_no_matching_all = [r['max_ae_no_matching'] for r in all_results if not np.isnan(r['max_ae_no_matching'])]

    print(f"\nMAE (Greedy matching):")
    if mae_greedy_all:
        print(f"  Mean: {np.mean(mae_greedy_all):.4f}")
        print(f"  Std:  {np.std(mae_greedy_all):.4f}")
        print(f"  Min:  {np.min(mae_greedy_all):.4f}")
        print(f"  Max:  {np.max(mae_greedy_all):.4f}")

    print(f"\nMAE (JV matching):")
    if mae_jv_all:
        print(f"  Mean: {np.mean(mae_jv_all):.4f}")
        print(f"  Std:  {np.std(mae_jv_all):.4f}")
        print(f"  Min:  {np.min(mae_jv_all):.4f}")
        print(f"  Max:  {np.max(mae_jv_all):.4f}")

    print(f"\nMAE (No matching):")
    if mae_no_matching_all:
        print(f"  Mean: {np.mean(mae_no_matching_all):.4f}")
        print(f"  Std:  {np.std(mae_no_matching_all):.4f}")
        print(f"  Min:  {np.min(mae_no_matching_all):.4f}")
        print(f"  Max:  {np.max(mae_no_matching_all):.4f}")

    print(f"\nMedian AE (Greedy matching):")
    if median_ae_greedy_all:
        print(f"  Mean: {np.mean(median_ae_greedy_all):.4f}")
        print(f"  Std:  {np.std(median_ae_greedy_all):.4f}")
        print(f"  Min:  {np.min(median_ae_greedy_all):.4f}")
        print(f"  Max:  {np.max(median_ae_greedy_all):.4f}")

    print(f"\nMedian AE (JV matching):")
    if median_ae_jv_all:
        print(f"  Mean: {np.mean(median_ae_jv_all):.4f}")
        print(f"  Std:  {np.std(median_ae_jv_all):.4f}")
        print(f"  Min:  {np.min(median_ae_jv_all):.4f}")
        print(f"  Max:  {np.max(median_ae_jv_all):.4f}")

    print(f"\nMedian AE (No matching):")
    if median_ae_no_matching_all:
        print(f"  Mean: {np.mean(median_ae_no_matching_all):.4f}")
        print(f"  Std:  {np.std(median_ae_no_matching_all):.4f}")
        print(f"  Min:  {np.min(median_ae_no_matching_all):.4f}")
        print(f"  Max:  {np.max(median_ae_no_matching_all):.4f}")

    print(f"\nMin AE (Greedy matching):")
    if min_ae_greedy_all:
        print(f"  Mean: {np.mean(min_ae_greedy_all):.4f}")
        print(f"  Std:  {np.std(min_ae_greedy_all):.4f}")
        print(f"  Min:  {np.min(min_ae_greedy_all):.4f}")
        print(f"  Max:  {np.max(min_ae_greedy_all):.4f}")

    print(f"\nMin AE (JV matching):")
    if min_ae_jv_all:
        print(f"  Mean: {np.mean(min_ae_jv_all):.4f}")
        print(f"  Std:  {np.std(min_ae_jv_all):.4f}")
        print(f"  Min:  {np.min(min_ae_jv_all):.4f}")
        print(f"  Max:  {np.max(min_ae_jv_all):.4f}")

    print(f"\nMin AE (No matching):")
    if min_ae_no_matching_all:
        print(f"  Mean: {np.mean(min_ae_no_matching_all):.4f}")
        print(f"  Std:  {np.std(min_ae_no_matching_all):.4f}")
        print(f"  Min:  {np.min(min_ae_no_matching_all):.4f}")
        print(f"  Max:  {np.max(min_ae_no_matching_all):.4f}")

    print(f"\nMax AE (Greedy matching):")
    if max_ae_greedy_all:
        print(f"  Mean: {np.mean(max_ae_greedy_all):.4f}")
        print(f"  Std:  {np.std(max_ae_greedy_all):.4f}")
        print(f"  Min:  {np.min(max_ae_greedy_all):.4f}")
        print(f"  Max:  {np.max(max_ae_greedy_all):.4f}")

    print(f"\nMax AE (JV matching):")
    if max_ae_jv_all:
        print(f"  Mean: {np.mean(max_ae_jv_all):.4f}")
        print(f"  Std:  {np.std(max_ae_jv_all):.4f}")
        print(f"  Min:  {np.min(max_ae_jv_all):.4f}")
        print(f"  Max:  {np.max(max_ae_jv_all):.4f}")

    print(f"\nMax AE (No matching):")
    if max_ae_no_matching_all:
        print(f"  Mean: {np.mean(max_ae_no_matching_all):.4f}")
        print(f"  Std:  {np.std(max_ae_no_matching_all):.4f}")
        print(f"  Min:  {np.min(max_ae_no_matching_all):.4f}")
        print(f"  Max:  {np.max(max_ae_no_matching_all):.4f}")

    # Per-dimension aggregate statistics
    # Format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
    dim_names = ["xwidth", "xcentre", "ywidth", "ycentre", "zwidth", "zcentre"]
    
    print(f"\nPer-dimension R² (Greedy matching) - Mean ± Std across folds:")
    for i in range(min(max_sources * 6, len(all_results[0]['per_dim_r2_greedy']))):
        dim_values = [r['per_dim_r2_greedy'][i] for r in all_results if not np.isnan(r['per_dim_r2_greedy'][i])]
        if dim_values:
            dim_name = dim_names[i % 6]
            source_num = i // 6 + 1
            mean_val = np.mean(dim_values)
            std_val = np.std(dim_values, ddof=1)  # Sample std
            print(f"  Source {source_num} - {dim_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(dim_values)})")

    print(f"\nPer-dimension R² (JV matching) - Mean ± Std across folds:")
    for i in range(min(max_sources * 6, len(all_results[0]['per_dim_r2_jv']))):
        dim_values = [r['per_dim_r2_jv'][i] for r in all_results if not np.isnan(r['per_dim_r2_jv'][i])]
        if dim_values:
            dim_name = dim_names[i % 6]
            source_num = i // 6 + 1
            mean_val = np.mean(dim_values)
            std_val = np.std(dim_values, ddof=1)  # Sample std
            print(f"  Source {source_num} - {dim_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(dim_values)})")

    # Per-dimension error metrics (Greedy)
    print(f"\nPer-dimension MAE (Greedy matching) - Mean ± Std across folds:")
    for i in range(min(max_sources * 6, len(all_results[0]['per_dim_mae_greedy']))):
        dim_values = [r['per_dim_mae_greedy'][i] for r in all_results if not np.isnan(r['per_dim_mae_greedy'][i])]
        if dim_values:
            dim_name = dim_names[i % 6]
            source_num = i // 6 + 1
            print(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}")

    print(f"\nPer-dimension Median AE (Greedy matching) - Mean ± Std across folds:")
    for i in range(min(max_sources * 6, len(all_results[0]['per_dim_median_ae_greedy']))):
        dim_values = [r['per_dim_median_ae_greedy'][i] for r in all_results if not np.isnan(r['per_dim_median_ae_greedy'][i])]
        if dim_values:
            dim_name = dim_names[i % 6]
            source_num = i // 6 + 1
            print(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}")

    print(f"\nPer-dimension Min AE (Greedy matching) - Mean ± Std across folds:")
    for i in range(min(max_sources * 6, len(all_results[0]['per_dim_min_ae_greedy']))):
        dim_values = [r['per_dim_min_ae_greedy'][i] for r in all_results if not np.isnan(r['per_dim_min_ae_greedy'][i])]
        if dim_values:
            dim_name = dim_names[i % 6]
            source_num = i // 6 + 1
            print(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}")

    print(f"\nPer-dimension Max AE (Greedy matching) - Mean ± Std across folds:")
    for i in range(min(max_sources * 6, len(all_results[0]['per_dim_max_ae_greedy']))):
        dim_values = [r['per_dim_max_ae_greedy'][i] for r in all_results if not np.isnan(r['per_dim_max_ae_greedy'][i])]
        if dim_values:
            dim_name = dim_names[i % 6]
            source_num = i // 6 + 1
            print(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}")

    # Per-dimension error metrics (JV)
    print(f"\nPer-dimension MAE (JV matching) - Mean ± Std across folds:")
    for i in range(min(max_sources * 6, len(all_results[0]['per_dim_mae_jv']))):
        dim_values = [r['per_dim_mae_jv'][i] for r in all_results if not np.isnan(r['per_dim_mae_jv'][i])]
        if dim_values:
            dim_name = dim_names[i % 6]
            source_num = i // 6 + 1
            print(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}")

    print(f"\nPer-dimension Median AE (JV matching) - Mean ± Std across folds:")
    for i in range(min(max_sources * 6, len(all_results[0]['per_dim_median_ae_jv']))):
        dim_values = [r['per_dim_median_ae_jv'][i] for r in all_results if not np.isnan(r['per_dim_median_ae_jv'][i])]
        if dim_values:
            dim_name = dim_names[i % 6]
            source_num = i // 6 + 1
            print(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}")

    print(f"\nPer-dimension Min AE (JV matching) - Mean ± Std across folds:")
    for i in range(min(max_sources * 6, len(all_results[0]['per_dim_min_ae_jv']))):
        dim_values = [r['per_dim_min_ae_jv'][i] for r in all_results if not np.isnan(r['per_dim_min_ae_jv'][i])]
        if dim_values:
            dim_name = dim_names[i % 6]
            source_num = i // 6 + 1
            print(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}")

    print(f"\nPer-dimension Max AE (JV matching) - Mean ± Std across folds:")
    for i in range(min(max_sources * 6, len(all_results[0]['per_dim_max_ae_jv']))):
        dim_values = [r['per_dim_max_ae_jv'][i] for r in all_results if not np.isnan(r['per_dim_max_ae_jv'][i])]
        if dim_values:
            dim_name = dim_names[i % 6]
            source_num = i // 6 + 1
            print(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}")

    # Write aggregate results to log file
    r2_log_path = res_dir / log_name
    with open(r2_log_path, "a", encoding="utf-8") as r2_file:
        r2_file.write(f"\n{'='*60}\n")
        r2_file.write(f"Test results for run '{file_label}{model_suffix}' (aggregated across {len(all_results)} folds)\n")
        r2_file.write(f"{'='*60}\n\n")
        
        # Overall R² statistics (valid slots only)
        r2_file.write("Overall R² Scores (Valid Slots Only):\n")
        if r2_greedy_all:
            r2_file.write(f"  Greedy matching:    Mean={np.mean(r2_greedy_all):.4f}, Std={np.std(r2_greedy_all):.4f}, Min={np.min(r2_greedy_all):.4f}, Max={np.max(r2_greedy_all):.4f}\n")
        if r2_jv_all:
            r2_file.write(f"  JV matching:        Mean={np.mean(r2_jv_all):.4f}, Std={np.std(r2_jv_all):.4f}, Min={np.min(r2_jv_all):.4f}, Max={np.max(r2_jv_all):.4f}\n")
        if r2_no_matching_all:
            r2_file.write(f"  No matching:        Mean={np.mean(r2_no_matching_all):.4f}, Std={np.std(r2_no_matching_all):.4f}, Min={np.min(r2_no_matching_all):.4f}, Max={np.max(r2_no_matching_all):.4f}\n")
        
        # Overall R² statistics (all slots including padded)
        r2_file.write("\nOverall R² Scores (All Slots Including Padded):\n")
        if r2_all_slots_greedy_all:
            r2_file.write(f"  Greedy matching:    Mean={np.mean(r2_all_slots_greedy_all):.4f}, Std={np.std(r2_all_slots_greedy_all):.4f}, Min={np.min(r2_all_slots_greedy_all):.4f}, Max={np.max(r2_all_slots_greedy_all):.4f}\n")
        if r2_all_slots_jv_all:
            r2_file.write(f"  JV matching:        Mean={np.mean(r2_all_slots_jv_all):.4f}, Std={np.std(r2_all_slots_jv_all):.4f}, Min={np.min(r2_all_slots_jv_all):.4f}, Max={np.max(r2_all_slots_jv_all):.4f}\n")
        if r2_all_slots_no_matching_all:
            r2_file.write(f"  No matching:        Mean={np.mean(r2_all_slots_no_matching_all):.4f}, Std={np.std(r2_all_slots_no_matching_all):.4f}, Min={np.min(r2_all_slots_no_matching_all):.4f}, Max={np.max(r2_all_slots_no_matching_all):.4f}\n")
        
        # Widths and Positions R² statistics
        r2_file.write("\nR² for All Widths (xwidth, ywidth, zwidth):\n")
        if r2_widths_greedy_all:
            r2_file.write(f"  Greedy matching:    Mean={np.mean(r2_widths_greedy_all):.4f}, Std={np.std(r2_widths_greedy_all):.4f}, Min={np.min(r2_widths_greedy_all):.4f}, Max={np.max(r2_widths_greedy_all):.4f}\n")
        if r2_widths_jv_all:
            r2_file.write(f"  JV matching:        Mean={np.mean(r2_widths_jv_all):.4f}, Std={np.std(r2_widths_jv_all):.4f}, Min={np.min(r2_widths_jv_all):.4f}, Max={np.max(r2_widths_jv_all):.4f}\n")
        
        r2_file.write("\nR² for All Positions (xcentre, ycentre, zcentre):\n")
        if r2_positions_greedy_all:
            r2_file.write(f"  Greedy matching:    Mean={np.mean(r2_positions_greedy_all):.4f}, Std={np.std(r2_positions_greedy_all):.4f}, Min={np.min(r2_positions_greedy_all):.4f}, Max={np.max(r2_positions_greedy_all):.4f}\n")
        if r2_positions_jv_all:
            r2_file.write(f"  JV matching:        Mean={np.mean(r2_positions_jv_all):.4f}, Std={np.std(r2_positions_jv_all):.4f}, Min={np.min(r2_positions_jv_all):.4f}, Max={np.max(r2_positions_jv_all):.4f}\n")
        
        # IoU statistics (pooled = over all predictions; across folds = of fold-level means)
        r2_file.write("\nIoU Metrics (IoU in [0,1]; std = spread, so mean ± std may extend outside [0,1]):\n")
        if iou_combined_N_greedy > 0:
            r2_file.write("  (all folds combined, one run)\n")
            r2_file.write(f"  Greedy matching:    Mean ± std = {iou_combined_mean_greedy:.4f} ± {iou_combined_std_greedy:.4f} (N={int(iou_combined_N_greedy)})\n")
            r2_file.write(f"  JV matching:        Mean ± std = {iou_combined_mean_jv:.4f} ± {iou_combined_std_jv:.4f} (N={int(iou_combined_N_jv)})\n")
            r2_file.write(f"  No matching:        Mean ± std = {iou_combined_mean_no_matching:.4f} ± {iou_combined_std_no_matching:.4f} (N={int(iou_combined_N_no_matching)})\n")
        elif iou_greedy_all:
            r2_file.write("  (from fold means)\n")
            r2_file.write(f"  Greedy matching:    Mean={np.mean(iou_greedy_all):.4f}, Std(across folds)={np.std(iou_greedy_all):.4f}\n")
            r2_file.write(f"  JV matching:        Mean={np.mean(iou_jv_all):.4f}, Std(across folds)={np.std(iou_jv_all):.4f}\n")
            r2_file.write(f"  No matching:        Mean={np.mean(iou_no_matching_all):.4f}, Std(across folds)={np.std(iou_no_matching_all):.4f}\n")
        
        # Source count metrics
        r2_file.write("\nSource Count Accuracy:\n")
        if source_count_accuracy_greedy_all:
            r2_file.write(f"  Greedy matching:    Mean={np.mean(source_count_accuracy_greedy_all):.4f} ({np.mean(source_count_accuracy_greedy_all)*100:.2f}%), Std={np.std(source_count_accuracy_greedy_all):.4f}, Min={np.min(source_count_accuracy_greedy_all):.4f}, Max={np.max(source_count_accuracy_greedy_all):.4f}\n")
        if source_count_accuracy_jv_all:
            r2_file.write(f"  JV matching:        Mean={np.mean(source_count_accuracy_jv_all):.4f} ({np.mean(source_count_accuracy_jv_all)*100:.2f}%), Std={np.std(source_count_accuracy_jv_all):.4f}, Min={np.min(source_count_accuracy_jv_all):.4f}, Max={np.max(source_count_accuracy_jv_all):.4f}\n")
        if source_count_accuracy_no_matching_all:
            r2_file.write(f"  No matching:        Mean={np.mean(source_count_accuracy_no_matching_all):.4f} ({np.mean(source_count_accuracy_no_matching_all)*100:.2f}%), Std={np.std(source_count_accuracy_no_matching_all):.4f}, Min={np.min(source_count_accuracy_no_matching_all):.4f}, Max={np.max(source_count_accuracy_no_matching_all):.4f}\n")
        
        r2_file.write("\nSource Count MAE:\n")
        if source_count_mae_greedy_all:
            r2_file.write(f"  Greedy matching:    Mean={np.mean(source_count_mae_greedy_all):.4f}, Std={np.std(source_count_mae_greedy_all):.4f}, Min={np.min(source_count_mae_greedy_all):.4f}, Max={np.max(source_count_mae_greedy_all):.4f}\n")
        if source_count_mae_jv_all:
            r2_file.write(f"  JV matching:        Mean={np.mean(source_count_mae_jv_all):.4f}, Std={np.std(source_count_mae_jv_all):.4f}, Min={np.min(source_count_mae_jv_all):.4f}, Max={np.max(source_count_mae_jv_all):.4f}\n")
        if source_count_mae_no_matching_all:
            r2_file.write(f"  No matching:        Mean={np.mean(source_count_mae_no_matching_all):.4f}, Std={np.std(source_count_mae_no_matching_all):.4f}, Min={np.min(source_count_mae_no_matching_all):.4f}, Max={np.max(source_count_mae_no_matching_all):.4f}\n")
        
        r2_file.write("\nSource Count Mean Error (predicted - true):\n")
        if source_count_mean_error_greedy_all:
            r2_file.write(f"  Greedy matching:    Mean={np.mean(source_count_mean_error_greedy_all):+.4f}, Std={np.std(source_count_mean_error_greedy_all):.4f}, Min={np.min(source_count_mean_error_greedy_all):+.4f}, Max={np.max(source_count_mean_error_greedy_all):+.4f}\n")
        if source_count_mean_error_jv_all:
            r2_file.write(f"  JV matching:        Mean={np.mean(source_count_mean_error_jv_all):+.4f}, Std={np.std(source_count_mean_error_jv_all):.4f}, Min={np.min(source_count_mean_error_jv_all):+.4f}, Max={np.max(source_count_mean_error_jv_all):+.4f}\n")
        if source_count_mean_error_no_matching_all:
            r2_file.write(f"  No matching:        Mean={np.mean(source_count_mean_error_no_matching_all):+.4f}, Std={np.std(source_count_mean_error_no_matching_all):.4f}, Min={np.min(source_count_mean_error_no_matching_all):+.4f}, Max={np.max(source_count_mean_error_no_matching_all):+.4f}\n")
        
        r2_file.write("\nEmpty Slot Prediction Accuracy (how well model predicts pad_value for empty slots):\n")
        if empty_slot_acc_greedy_all:
            r2_file.write(f"  Greedy matching:    Mean={np.mean(empty_slot_acc_greedy_all):.4f} ({np.mean(empty_slot_acc_greedy_all)*100:.2f}%), Std={np.std(empty_slot_acc_greedy_all):.4f}, Min={np.min(empty_slot_acc_greedy_all):.4f}, Max={np.max(empty_slot_acc_greedy_all):.4f}\n")
        if empty_slot_acc_jv_all:
            r2_file.write(f"  JV matching:        Mean={np.mean(empty_slot_acc_jv_all):.4f} ({np.mean(empty_slot_acc_jv_all)*100:.2f}%), Std={np.std(empty_slot_acc_jv_all):.4f}, Min={np.min(empty_slot_acc_jv_all):.4f}, Max={np.max(empty_slot_acc_jv_all):.4f}\n")
        if empty_slot_acc_no_matching_all:
            r2_file.write(f"  No matching:        Mean={np.mean(empty_slot_acc_no_matching_all):.4f} ({np.mean(empty_slot_acc_no_matching_all)*100:.2f}%), Std={np.std(empty_slot_acc_no_matching_all):.4f}, Min={np.min(empty_slot_acc_no_matching_all):.4f}, Max={np.max(empty_slot_acc_no_matching_all):.4f}\n")
        
        r2_file.write("\nTrue Mean Source Count:\n")
        if true_mean_source_count_all:
            r2_file.write(f"  Mean: {np.mean(true_mean_source_count_all):.2f}, Std: {np.std(true_mean_source_count_all):.2f}, Min: {np.min(true_mean_source_count_all):.2f}, Max: {np.max(true_mean_source_count_all):.2f}\n")
        
        r2_file.write("\nPredicted Mean Source Count:\n")
        if pred_mean_source_count_greedy_all:
            r2_file.write(f"  Greedy matching:    Mean={np.mean(pred_mean_source_count_greedy_all):.2f}, Std={np.std(pred_mean_source_count_greedy_all):.2f}, Min={np.min(pred_mean_source_count_greedy_all):.2f}, Max={np.max(pred_mean_source_count_greedy_all):.2f}\n")
        if pred_mean_source_count_jv_all:
            r2_file.write(f"  JV matching:        Mean={np.mean(pred_mean_source_count_jv_all):.2f}, Std={np.std(pred_mean_source_count_jv_all):.2f}, Min={np.min(pred_mean_source_count_jv_all):.2f}, Max={np.max(pred_mean_source_count_jv_all):.2f}\n")
        if pred_mean_source_count_no_matching_all:
            r2_file.write(f"  No matching:        Mean={np.mean(pred_mean_source_count_no_matching_all):.2f}, Std={np.std(pred_mean_source_count_no_matching_all):.2f}, Min={np.min(pred_mean_source_count_no_matching_all):.2f}, Max={np.max(pred_mean_source_count_no_matching_all):.2f}\n")
        
        # Source Count Classification Metrics (from individual folds - mean across folds)
        r2_file.write("\nSource Count Classification Metrics (Mean across folds):\n")
        
        # Collect classification metrics from all folds
        prec_w_greedy_all = [r['source_count_precision_weighted_greedy'] for r in all_results if 'source_count_precision_weighted_greedy' in r and not np.isnan(r['source_count_precision_weighted_greedy'])]
        prec_m_greedy_all = [r['source_count_precision_macro_greedy'] for r in all_results if 'source_count_precision_macro_greedy' in r and not np.isnan(r['source_count_precision_macro_greedy'])]
        rec_w_greedy_all = [r['source_count_recall_weighted_greedy'] for r in all_results if 'source_count_recall_weighted_greedy' in r and not np.isnan(r['source_count_recall_weighted_greedy'])]
        rec_m_greedy_all = [r['source_count_recall_macro_greedy'] for r in all_results if 'source_count_recall_macro_greedy' in r and not np.isnan(r['source_count_recall_macro_greedy'])]
        f1_w_greedy_all = [r['source_count_f1_weighted_greedy'] for r in all_results if 'source_count_f1_weighted_greedy' in r and not np.isnan(r['source_count_f1_weighted_greedy'])]
        f1_m_greedy_all = [r['source_count_f1_macro_greedy'] for r in all_results if 'source_count_f1_macro_greedy' in r and not np.isnan(r['source_count_f1_macro_greedy'])]
        
        prec_w_jv_all = [r['source_count_precision_weighted_jv'] for r in all_results if 'source_count_precision_weighted_jv' in r and not np.isnan(r['source_count_precision_weighted_jv'])]
        prec_m_jv_all = [r['source_count_precision_macro_jv'] for r in all_results if 'source_count_precision_macro_jv' in r and not np.isnan(r['source_count_precision_macro_jv'])]
        rec_w_jv_all = [r['source_count_recall_weighted_jv'] for r in all_results if 'source_count_recall_weighted_jv' in r and not np.isnan(r['source_count_recall_weighted_jv'])]
        rec_m_jv_all = [r['source_count_recall_macro_jv'] for r in all_results if 'source_count_recall_macro_jv' in r and not np.isnan(r['source_count_recall_macro_jv'])]
        f1_w_jv_all = [r['source_count_f1_weighted_jv'] for r in all_results if 'source_count_f1_weighted_jv' in r and not np.isnan(r['source_count_f1_weighted_jv'])]
        f1_m_jv_all = [r['source_count_f1_macro_jv'] for r in all_results if 'source_count_f1_macro_jv' in r and not np.isnan(r['source_count_f1_macro_jv'])]
        
        prec_w_no_matching_all = [r['source_count_precision_weighted_no_matching'] for r in all_results if 'source_count_precision_weighted_no_matching' in r and not np.isnan(r['source_count_precision_weighted_no_matching'])]
        prec_m_no_matching_all = [r['source_count_precision_macro_no_matching'] for r in all_results if 'source_count_precision_macro_no_matching' in r and not np.isnan(r['source_count_precision_macro_no_matching'])]
        rec_w_no_matching_all = [r['source_count_recall_weighted_no_matching'] for r in all_results if 'source_count_recall_weighted_no_matching' in r and not np.isnan(r['source_count_recall_weighted_no_matching'])]
        rec_m_no_matching_all = [r['source_count_recall_macro_no_matching'] for r in all_results if 'source_count_recall_macro_no_matching' in r and not np.isnan(r['source_count_recall_macro_no_matching'])]
        f1_w_no_matching_all = [r['source_count_f1_weighted_no_matching'] for r in all_results if 'source_count_f1_weighted_no_matching' in r and not np.isnan(r['source_count_f1_weighted_no_matching'])]
        f1_m_no_matching_all = [r['source_count_f1_macro_no_matching'] for r in all_results if 'source_count_f1_macro_no_matching' in r and not np.isnan(r['source_count_f1_macro_no_matching'])]
        
        r2_file.write("\n  Greedy matching:\n")
        if prec_w_greedy_all:
            r2_file.write(f"    Precision (weighted): Mean={np.mean(prec_w_greedy_all):.4f}, Std={np.std(prec_w_greedy_all):.4f}\n")
        if prec_m_greedy_all:
            r2_file.write(f"    Precision (macro):    Mean={np.mean(prec_m_greedy_all):.4f}, Std={np.std(prec_m_greedy_all):.4f}\n")
        if rec_w_greedy_all:
            r2_file.write(f"    Recall (weighted):    Mean={np.mean(rec_w_greedy_all):.4f}, Std={np.std(rec_w_greedy_all):.4f}\n")
        if rec_m_greedy_all:
            r2_file.write(f"    Recall (macro):       Mean={np.mean(rec_m_greedy_all):.4f}, Std={np.std(rec_m_greedy_all):.4f}\n")
        if f1_w_greedy_all:
            r2_file.write(f"    F1 (weighted):        Mean={np.mean(f1_w_greedy_all):.4f}, Std={np.std(f1_w_greedy_all):.4f}\n")
        if f1_m_greedy_all:
            r2_file.write(f"    F1 (macro):           Mean={np.mean(f1_m_greedy_all):.4f}, Std={np.std(f1_m_greedy_all):.4f}\n")
        
        r2_file.write("\n  JV matching:\n")
        if prec_w_jv_all:
            r2_file.write(f"    Precision (weighted): Mean={np.mean(prec_w_jv_all):.4f}, Std={np.std(prec_w_jv_all):.4f}\n")
        if prec_m_jv_all:
            r2_file.write(f"    Precision (macro):    Mean={np.mean(prec_m_jv_all):.4f}, Std={np.std(prec_m_jv_all):.4f}\n")
        if rec_w_jv_all:
            r2_file.write(f"    Recall (weighted):    Mean={np.mean(rec_w_jv_all):.4f}, Std={np.std(rec_w_jv_all):.4f}\n")
        if rec_m_jv_all:
            r2_file.write(f"    Recall (macro):       Mean={np.mean(rec_m_jv_all):.4f}, Std={np.std(rec_m_jv_all):.4f}\n")
        if f1_w_jv_all:
            r2_file.write(f"    F1 (weighted):        Mean={np.mean(f1_w_jv_all):.4f}, Std={np.std(f1_w_jv_all):.4f}\n")
        if f1_m_jv_all:
            r2_file.write(f"    F1 (macro):           Mean={np.mean(f1_m_jv_all):.4f}, Std={np.std(f1_m_jv_all):.4f}\n")
        
        r2_file.write("\n  No matching:\n")
        if prec_w_no_matching_all:
            r2_file.write(f"    Precision (weighted): Mean={np.mean(prec_w_no_matching_all):.4f}, Std={np.std(prec_w_no_matching_all):.4f}\n")
        if prec_m_no_matching_all:
            r2_file.write(f"    Precision (macro):    Mean={np.mean(prec_m_no_matching_all):.4f}, Std={np.std(prec_m_no_matching_all):.4f}\n")
        if rec_w_no_matching_all:
            r2_file.write(f"    Recall (weighted):    Mean={np.mean(rec_w_no_matching_all):.4f}, Std={np.std(rec_w_no_matching_all):.4f}\n")
        if rec_m_no_matching_all:
            r2_file.write(f"    Recall (macro):       Mean={np.mean(rec_m_no_matching_all):.4f}, Std={np.std(rec_m_no_matching_all):.4f}\n")
        if f1_w_no_matching_all:
            r2_file.write(f"    F1 (weighted):        Mean={np.mean(f1_w_no_matching_all):.4f}, Std={np.std(f1_w_no_matching_all):.4f}\n")
        if f1_m_no_matching_all:
            r2_file.write(f"    F1 (macro):           Mean={np.mean(f1_m_no_matching_all):.4f}, Std={np.std(f1_m_no_matching_all):.4f}\n")
        
        # Overall error statistics
        r2_file.write("\nOverall MAE:\n")
        if mae_greedy_all:
            r2_file.write(f"  Greedy matching:    Mean={np.mean(mae_greedy_all):.4f}, Std={np.std(mae_greedy_all):.4f}, Min={np.min(mae_greedy_all):.4f}, Max={np.max(mae_greedy_all):.4f}\n")
        if mae_jv_all:
            r2_file.write(f"  JV matching:        Mean={np.mean(mae_jv_all):.4f}, Std={np.std(mae_jv_all):.4f}, Min={np.min(mae_jv_all):.4f}, Max={np.max(mae_jv_all):.4f}\n")
        if mae_no_matching_all:
            r2_file.write(f"  No matching:        Mean={np.mean(mae_no_matching_all):.4f}, Std={np.std(mae_no_matching_all):.4f}, Min={np.min(mae_no_matching_all):.4f}, Max={np.max(mae_no_matching_all):.4f}\n")
        
        r2_file.write("\nOverall Median AE:\n")
        if median_ae_greedy_all:
            r2_file.write(f"  Greedy matching:    Mean={np.mean(median_ae_greedy_all):.4f}, Std={np.std(median_ae_greedy_all):.4f}, Min={np.min(median_ae_greedy_all):.4f}, Max={np.max(median_ae_greedy_all):.4f}\n")
        if median_ae_jv_all:
            r2_file.write(f"  JV matching:        Mean={np.mean(median_ae_jv_all):.4f}, Std={np.std(median_ae_jv_all):.4f}, Min={np.min(median_ae_jv_all):.4f}, Max={np.max(median_ae_jv_all):.4f}\n")
        if median_ae_no_matching_all:
            r2_file.write(f"  No matching:        Mean={np.mean(median_ae_no_matching_all):.4f}, Std={np.std(median_ae_no_matching_all):.4f}, Min={np.min(median_ae_no_matching_all):.4f}, Max={np.max(median_ae_no_matching_all):.4f}\n")
        
        r2_file.write("\nOverall Min AE:\n")
        if min_ae_greedy_all:
            r2_file.write(f"  Greedy matching:    Mean={np.mean(min_ae_greedy_all):.4f}, Std={np.std(min_ae_greedy_all):.4f}, Min={np.min(min_ae_greedy_all):.4f}, Max={np.max(min_ae_greedy_all):.4f}\n")
        if min_ae_jv_all:
            r2_file.write(f"  JV matching:        Mean={np.mean(min_ae_jv_all):.4f}, Std={np.std(min_ae_jv_all):.4f}, Min={np.min(min_ae_jv_all):.4f}, Max={np.max(min_ae_jv_all):.4f}\n")
        if min_ae_no_matching_all:
            r2_file.write(f"  No matching:        Mean={np.mean(min_ae_no_matching_all):.4f}, Std={np.std(min_ae_no_matching_all):.4f}, Min={np.min(min_ae_no_matching_all):.4f}, Max={np.max(min_ae_no_matching_all):.4f}\n")
        
        r2_file.write("\nOverall Max AE:\n")
        if max_ae_greedy_all:
            r2_file.write(f"  Greedy matching:    Mean={np.mean(max_ae_greedy_all):.4f}, Std={np.std(max_ae_greedy_all):.4f}, Min={np.min(max_ae_greedy_all):.4f}, Max={np.max(max_ae_greedy_all):.4f}\n")
        if max_ae_jv_all:
            r2_file.write(f"  JV matching:        Mean={np.mean(max_ae_jv_all):.4f}, Std={np.std(max_ae_jv_all):.4f}, Min={np.min(max_ae_jv_all):.4f}, Max={np.max(max_ae_jv_all):.4f}\n")
        if max_ae_no_matching_all:
            r2_file.write(f"  No matching:        Mean={np.mean(max_ae_no_matching_all):.4f}, Std={np.std(max_ae_no_matching_all):.4f}, Min={np.min(max_ae_no_matching_all):.4f}, Max={np.max(max_ae_no_matching_all):.4f}\n")
        
        # Per-dimension R² statistics (Greedy)
        r2_file.write("\nPer-dimension R² (Greedy matching) - Mean ± Std:\n")
        for i in range(min(max_sources * 6, len(all_results[0]['per_dim_r2_greedy']))):
            dim_values = [r['per_dim_r2_greedy'][i] for r in all_results if not np.isnan(r['per_dim_r2_greedy'][i])]
            if dim_values:
                dim_name = dim_names[i % 6]
                source_num = i // 6 + 1
                mean_val = np.mean(dim_values)
                std_val = np.std(dim_values, ddof=1)  # Sample std
                r2_file.write(f"  Source {source_num} - {dim_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(dim_values)})\n")
        
        # Per-dimension R² statistics (JV)
        r2_file.write("\nPer-dimension R² (JV matching) - Mean ± Std:\n")
        for i in range(min(max_sources * 6, len(all_results[0]['per_dim_r2_jv']))):
            dim_values = [r['per_dim_r2_jv'][i] for r in all_results if not np.isnan(r['per_dim_r2_jv'][i])]
            if dim_values:
                dim_name = dim_names[i % 6]
                source_num = i // 6 + 1
                mean_val = np.mean(dim_values)
                std_val = np.std(dim_values, ddof=1)  # Sample std
                r2_file.write(f"  Source {source_num} - {dim_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(dim_values)})\n")
        
        # Per-dimension error statistics (Greedy)
        r2_file.write("\nPer-dimension MAE (Greedy matching) - Mean ± Std:\n")
        for i in range(min(max_sources * 6, len(all_results[0]['per_dim_mae_greedy']))):
            dim_values = [r['per_dim_mae_greedy'][i] for r in all_results if not np.isnan(r['per_dim_mae_greedy'][i])]
            if dim_values:
                dim_name = dim_names[i % 6]
                source_num = i // 6 + 1
                r2_file.write(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}\n")
        
        r2_file.write("\nPer-dimension Median AE (Greedy matching) - Mean ± Std:\n")
        for i in range(min(max_sources * 6, len(all_results[0]['per_dim_median_ae_greedy']))):
            dim_values = [r['per_dim_median_ae_greedy'][i] for r in all_results if not np.isnan(r['per_dim_median_ae_greedy'][i])]
            if dim_values:
                dim_name = dim_names[i % 6]
                source_num = i // 6 + 1
                r2_file.write(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}\n")
        
        r2_file.write("\nPer-dimension Min AE (Greedy matching) - Mean ± Std:\n")
        for i in range(min(max_sources * 6, len(all_results[0]['per_dim_min_ae_greedy']))):
            dim_values = [r['per_dim_min_ae_greedy'][i] for r in all_results if not np.isnan(r['per_dim_min_ae_greedy'][i])]
            if dim_values:
                dim_name = dim_names[i % 6]
                source_num = i // 6 + 1
                r2_file.write(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}\n")
        
        r2_file.write("\nPer-dimension Max AE (Greedy matching) - Mean ± Std:\n")
        for i in range(min(max_sources * 6, len(all_results[0]['per_dim_max_ae_greedy']))):
            dim_values = [r['per_dim_max_ae_greedy'][i] for r in all_results if not np.isnan(r['per_dim_max_ae_greedy'][i])]
            if dim_values:
                dim_name = dim_names[i % 6]
                source_num = i // 6 + 1
                r2_file.write(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}\n")
        
        # Per-dimension error statistics (JV)
        r2_file.write("\nPer-dimension MAE (JV matching) - Mean ± Std:\n")
        for i in range(min(max_sources * 6, len(all_results[0]['per_dim_mae_jv']))):
            dim_values = [r['per_dim_mae_jv'][i] for r in all_results if not np.isnan(r['per_dim_mae_jv'][i])]
            if dim_values:
                dim_name = dim_names[i % 6]
                source_num = i // 6 + 1
                r2_file.write(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}\n")
        
        r2_file.write("\nPer-dimension Median AE (JV matching) - Mean ± Std:\n")
        for i in range(min(max_sources * 6, len(all_results[0]['per_dim_median_ae_jv']))):
            dim_values = [r['per_dim_median_ae_jv'][i] for r in all_results if not np.isnan(r['per_dim_median_ae_jv'][i])]
            if dim_values:
                dim_name = dim_names[i % 6]
                source_num = i // 6 + 1
                r2_file.write(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}\n")
        
        r2_file.write("\nPer-dimension Min AE (JV matching) - Mean ± Std:\n")
        for i in range(min(max_sources * 6, len(all_results[0]['per_dim_min_ae_jv']))):
            dim_values = [r['per_dim_min_ae_jv'][i] for r in all_results if not np.isnan(r['per_dim_min_ae_jv'][i])]
            if dim_values:
                dim_name = dim_names[i % 6]
                source_num = i // 6 + 1
                r2_file.write(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}\n")
        
        r2_file.write("\nPer-dimension Max AE (JV matching) - Mean ± Std:\n")
        for i in range(min(max_sources * 6, len(all_results[0]['per_dim_max_ae_jv']))):
            dim_values = [r['per_dim_max_ae_jv'][i] for r in all_results if not np.isnan(r['per_dim_max_ae_jv'][i])]
            if dim_values:
                dim_name = dim_names[i % 6]
                source_num = i // 6 + 1
                r2_file.write(f"  Source {source_num} - {dim_name}: {np.mean(dim_values):.4f} ± {np.std(dim_values):.4f}\n")
        
        r2_file.write("\n" + "="*60 + "\n")

    print(f"\nR² scores appended to {r2_log_path}")
    
    # Generate R² plots
    r2_plot_path_greedy = res_dir / f"r2_scores_{file_label}_greedy.pdf"
    r2_plot_path_jv = res_dir / f"r2_scores_{file_label}_jv.pdf"
    
    try:
        PLACENetPlot.plot_r2_scores(all_results, r2_plot_path_greedy, matching_type="greedy")
        PLACENetPlot.plot_r2_scores(all_results, r2_plot_path_jv, matching_type="jv")
    except Exception as e:
        print(f"Warning: Could not generate R² plots: {e}")
    
    # Generate combined R² scatter plots across all folds
    try:
        print("\nGenerating combined R² scatter plots across all folds...")
        
        # Collect all data from all folds
        all_overall_truth_greedy = []
        all_overall_pred_greedy = []
        all_widths_truth_greedy = []
        all_widths_pred_greedy = []
        all_positions_truth_greedy = []
        all_positions_pred_greedy = []
        
        all_overall_truth_jv = []
        all_overall_pred_jv = []
        all_widths_truth_jv = []
        all_widths_pred_jv = []
        all_positions_truth_jv = []
        all_positions_pred_jv = []
        
        # Collect confidence scores across all folds
        all_pred_confidences = []
        all_true_confidences = []
        
        for kfold_str in fold_numbers:
            npz_path = res_dir / f"data_labels_test_{file_label}_kf{kfold_str}.npz"
            if not npz_path.exists():
                continue
            
            # Load data from npz file (need to load before context closes)
            with np.load(npz_path) as npz_file:
                data_test = npz_file["data"]
                labels_test_raw = npz_file["labels"]
            
            # Extract boxes if they have confidence dimension
            if labels_test_raw.shape[-1] == 7:
                labels_test = labels_test_raw[..., :6]
            else:
                labels_test = labels_test_raw
            
            model_path = res_dir / f"model_{file_label}{model_suffix}_kf{kfold_str}.keras"
            if not model_path.exists():
                continue
            
            model = tf.keras.models.load_model(model_path, compile=False)
            pred_raw = model.predict(data_test, verbose=0)
            
            # Extract boxes from YOLO-style output if needed
            pred, pred_conf = extract_boxes_from_yolo_output(pred_raw, confidence_threshold=0.5, pad_value=pad_value)
            
            # Apply matching
            pred_matched_jv = apply_jv_matching(labels_test, pred, pad_value=pad_value)
            pred_matched_greedy = apply_greedy_matching(labels_test, pred, pad_value=pad_value)
            
            # Match confidence scores to ground truth slots (use greedy matching for consistency)
            pred_conf_matched = None
            if pred_conf is not None:
                pred_conf_matched = match_confidence_scores(
                    labels_test, pred, pred_conf, pad_value=pad_value, matching_type="greedy"
                )
            
            # Collect MATCHED confidence scores if available
            if pred_conf_matched is not None:
                all_pred_confidences.extend(pred_conf_matched.flatten())
                # Create true confidence labels (1.0 for valid slots, 0.0 for empty slots)
                true_conf = np.zeros_like(pred_conf_matched, dtype=np.float32)
                for b in range(labels_test.shape[0]):
                    for s in range(labels_test.shape[1]):
                        is_valid = not np.all(labels_test[b, s] == pad_value)
                        true_conf[b, s] = 1.0 if is_valid else 0.0
                all_true_confidences.extend(true_conf.flatten())
            
            # Flatten for analysis
            truth_flat = labels_test.reshape(labels_test.shape[0], -1)
            pred_matched_greedy_flat = pred_matched_greedy.reshape(pred_matched_greedy.shape[0], -1)
            pred_matched_jv_flat = pred_matched_jv.reshape(pred_matched_jv.shape[0], -1)
            
            # Masks for valid data
            mask_greedy = (truth_flat != pad_value) & (pred_matched_greedy_flat != pad_value)
            mask_jv = (truth_flat != pad_value) & (pred_matched_jv_flat != pad_value)
            
            # Overall data (Greedy)
            overall_truth_greedy = truth_flat[mask_greedy] if np.any(mask_greedy) else np.array([])
            overall_pred_greedy = pred_matched_greedy_flat[mask_greedy] if np.any(mask_greedy) else np.array([])
            all_overall_truth_greedy.extend(overall_truth_greedy.flatten())
            all_overall_pred_greedy.extend(overall_pred_greedy.flatten())
            
            # Overall data (JV)
            overall_truth_jv = truth_flat[mask_jv] if np.any(mask_jv) else np.array([])
            overall_pred_jv = pred_matched_jv_flat[mask_jv] if np.any(mask_jv) else np.array([])
            all_overall_truth_jv.extend(overall_truth_jv.flatten())
            all_overall_pred_jv.extend(overall_pred_jv.flatten())
            
            # Widths and positions (Greedy)
            width_indices = [i for i in range(min(max_sources * 6, truth_flat.shape[1])) if i % 6 in [0, 2, 4]]
            position_indices = [i for i in range(min(max_sources * 6, truth_flat.shape[1])) if i % 6 in [1, 3, 5]]
            
            for idx in width_indices:
                mask = (truth_flat[:, idx] != pad_value) & (pred_matched_greedy_flat[:, idx] != pad_value)
                if np.any(mask):
                    all_widths_truth_greedy.extend(truth_flat[mask, idx])
                    all_widths_pred_greedy.extend(pred_matched_greedy_flat[mask, idx])
            
            for idx in position_indices:
                mask = (truth_flat[:, idx] != pad_value) & (pred_matched_greedy_flat[:, idx] != pad_value)
                if np.any(mask):
                    all_positions_truth_greedy.extend(truth_flat[mask, idx])
                    all_positions_pred_greedy.extend(pred_matched_greedy_flat[mask, idx])
            
            # Widths and positions (JV)
            for idx in width_indices:
                mask = (truth_flat[:, idx] != pad_value) & (pred_matched_jv_flat[:, idx] != pad_value)
                if np.any(mask):
                    all_widths_truth_jv.extend(truth_flat[mask, idx])
                    all_widths_pred_jv.extend(pred_matched_jv_flat[mask, idx])
            
            for idx in position_indices:
                mask = (truth_flat[:, idx] != pad_value) & (pred_matched_jv_flat[:, idx] != pad_value)
                if np.any(mask):
                    all_positions_truth_jv.extend(truth_flat[mask, idx])
                    all_positions_pred_jv.extend(pred_matched_jv_flat[mask, idx])
            
            # Clean up
            del model, pred_raw, pred, pred_matched_jv, pred_matched_greedy
            tf.keras.backend.clear_session()
        
        # Convert to arrays
        all_overall_truth_greedy = np.array(all_overall_truth_greedy)
        all_overall_pred_greedy = np.array(all_overall_pred_greedy)
        all_widths_truth_greedy = np.array(all_widths_truth_greedy)
        all_widths_pred_greedy = np.array(all_widths_pred_greedy)
        all_positions_truth_greedy = np.array(all_positions_truth_greedy)
        all_positions_pred_greedy = np.array(all_positions_pred_greedy)
        
        all_overall_truth_jv = np.array(all_overall_truth_jv)
        all_overall_pred_jv = np.array(all_overall_pred_jv)
        all_widths_truth_jv = np.array(all_widths_truth_jv)
        all_widths_pred_jv = np.array(all_widths_pred_jv)
        all_positions_truth_jv = np.array(all_positions_truth_jv)
        all_positions_pred_jv = np.array(all_positions_pred_jv)
        
        # Calculate combined R² scores
        r2_overall_greedy = r2_score(all_overall_truth_greedy, all_overall_pred_greedy) if len(all_overall_truth_greedy) > 0 else np.nan
        r2_widths_greedy = r2_score(all_widths_truth_greedy, all_widths_pred_greedy) if len(all_widths_truth_greedy) > 0 else np.nan
        r2_positions_greedy = r2_score(all_positions_truth_greedy, all_positions_pred_greedy) if len(all_positions_truth_greedy) > 0 else np.nan
        
        r2_overall_jv = r2_score(all_overall_truth_jv, all_overall_pred_jv) if len(all_overall_truth_jv) > 0 else np.nan
        r2_widths_jv = r2_score(all_widths_truth_jv, all_widths_pred_jv) if len(all_widths_truth_jv) > 0 else np.nan
        r2_positions_jv = r2_score(all_positions_truth_jv, all_positions_pred_jv) if len(all_positions_truth_jv) > 0 else np.nan
        
        # Plot combined scatter plots (Greedy and JV) via shared helper
        for matching_type in ["greedy", "jv"]:
            if matching_type == "greedy":
                o_truth, o_pred = all_overall_truth_greedy, all_overall_pred_greedy
                w_truth, w_pred = all_widths_truth_greedy, all_widths_pred_greedy
                p_truth, p_pred = all_positions_truth_greedy, all_positions_pred_greedy
                r2_o, r2_w, r2_p = r2_overall_greedy, r2_widths_greedy, r2_positions_greedy
                suptitle = "Truth vs Prediction - All Folds Combined"
            else:
                o_truth, o_pred = all_overall_truth_jv, all_overall_pred_jv
                w_truth, w_pred = all_widths_truth_jv, all_widths_pred_jv
                p_truth, p_pred = all_positions_truth_jv, all_positions_pred_jv
                r2_o, r2_w, r2_p = r2_overall_jv, r2_widths_jv, r2_positions_jv
                suptitle = "Truth vs Prediction (JV matching) - All Folds Combined"
            scatter_path = res_dir / f"r2_scatter_{file_label}_all_folds_{matching_type}.pdf"
            _save_r2_scatter_plot(
                scatter_path, o_truth, o_pred, w_truth, w_pred, p_truth, p_pred,
                r2_o, r2_w, r2_p, suptitle, verbose=True,
            )
        
        # Plot combined confidence distribution across all folds
        if len(all_pred_confidences) > 0:
            try:
                confidence_plot_path = res_dir / f"confidence_distribution_{file_label}_all_folds.pdf"
                all_pred_conf_array = np.array(all_pred_confidences)
                all_true_conf_array = np.array(all_true_confidences) if len(all_true_confidences) > 0 else None
                PLACENetPlot.plot_confidence_distribution(
                    pred_confidences=all_pred_conf_array,
                    true_confidences=all_true_conf_array,
                    output_path=confidence_plot_path,
                    title=f"Confidence Score Distribution - All Folds Combined"
                )
                
                # Also create separate "Distribution by Slot Type" plot with mean/median lines
                if all_true_conf_array is not None:
                    slot_type_plot_path = res_dir / f"confidence_by_slot_type_{file_label}_all_folds.pdf"
                    PLACENetPlot.plot_confidence_by_slot_type(
                        pred_confidences=all_pred_conf_array,
                        true_confidences=all_true_conf_array,
                        output_path=slot_type_plot_path,
                        title=f"Confidence Distribution by Slot Type - All Folds Combined"
                    )
                    
                    # Create diagnostic plots (ROC, PR, threshold analysis) for all folds
                    diagnostics_plot_path = res_dir / f"confidence_diagnostics_{file_label}_all_folds.pdf"
                    PLACENetPlot.plot_confidence_diagnostics(
                        pred_confidences=all_pred_conf_array,
                        true_confidences=all_true_conf_array,
                        output_path=diagnostics_plot_path,
                        title=f"Confidence Classification Diagnostics - All Folds Combined",
                        threshold=0.5
                    )
            except Exception as e:
                print(f"Warning: Could not generate combined confidence distribution plot: {e}")
        
    except Exception as e:
        print(f"Warning: Could not generate combined R² scatter plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate combined confusion matrices across all folds
    try:
        # Collect all source counts from all folds
        all_true_counts = []
        all_pred_counts_no_matching = []
        all_pred_counts_jv = []
        all_pred_counts_greedy = []
        
        for result in all_results:
            if 'true_source_counts' in result:
                all_true_counts.extend(result['true_source_counts'])
                all_pred_counts_no_matching.extend(result['pred_source_counts_no_matching'])
                all_pred_counts_jv.extend(result['pred_source_counts_jv'])
                all_pred_counts_greedy.extend(result['pred_source_counts_greedy'])
        
        if all_true_counts:
            all_true_counts = np.array(all_true_counts)
            all_pred_counts_no_matching = np.array(all_pred_counts_no_matching)
            all_pred_counts_jv = np.array(all_pred_counts_jv)
            all_pred_counts_greedy = np.array(all_pred_counts_greedy)
            
            # Get all possible source counts (1..max_sources) for combined metrics - exclude 0
            all_labels = np.arange(1, max_sources + 1)
            
            # Calculate combined classification metrics
            def calc_combined_metrics(y_true, y_pred, labels):
                """Calculate combined classification metrics (both macro and weighted)."""
                y_pred_clipped = np.clip(y_pred, 0, max_sources)
                precision_weighted = precision_score(y_true, y_pred_clipped, labels=labels, average='weighted', zero_division=0)
                recall_weighted = recall_score(y_true, y_pred_clipped, labels=labels, average='weighted', zero_division=0)
                f1_weighted = f1_score(y_true, y_pred_clipped, labels=labels, average='weighted', zero_division=0)
                precision_macro = precision_score(y_true, y_pred_clipped, labels=labels, average='macro', zero_division=0)
                recall_macro = recall_score(y_true, y_pred_clipped, labels=labels, average='macro', zero_division=0)
                f1_macro = f1_score(y_true, y_pred_clipped, labels=labels, average='macro', zero_division=0)
                cm = confusion_matrix(y_true, y_pred_clipped, labels=labels)
                return precision_weighted, recall_weighted, f1_weighted, precision_macro, recall_macro, f1_macro, cm
            
            # Calculate metrics for each matching strategy
            prec_w_no_matching, rec_w_no_matching, f1_w_no_matching, prec_m_no_matching, rec_m_no_matching, f1_m_no_matching, cm_no_matching = calc_combined_metrics(
                all_true_counts, all_pred_counts_no_matching, all_labels
            )
            prec_w_jv, rec_w_jv, f1_w_jv, prec_m_jv, rec_m_jv, f1_m_jv, cm_jv = calc_combined_metrics(
                all_true_counts, all_pred_counts_jv, all_labels
            )
            prec_w_greedy, rec_w_greedy, f1_w_greedy, prec_m_greedy, rec_m_greedy, f1_m_greedy, cm_greedy = calc_combined_metrics(
                all_true_counts, all_pred_counts_greedy, all_labels
            )
            
            # Print combined metrics
            print(f"\n" + "="*60)
            print("COMBINED CLASSIFICATION METRICS (ALL FOLDS)")
            print("="*60)
            print(f"\nSource Count Classification Metrics (Greedy Matching):")
            print(f"  Precision (weighted): {prec_w_greedy:.4f}")
            print(f"  Precision (macro): {prec_m_greedy:.4f}")
            print(f"  Recall (weighted): {rec_w_greedy:.4f}")
            print(f"  Recall (macro): {rec_m_greedy:.4f}")
            print(f"  F1 (weighted): {f1_w_greedy:.4f}")
            print(f"  F1 (macro): {f1_m_greedy:.4f}")
            print(f"\nSource Count Classification Metrics (JV Matching):")
            print(f"  Precision (weighted): {prec_w_jv:.4f}")
            print(f"  Precision (macro): {prec_m_jv:.4f}")
            print(f"  Recall (weighted): {rec_w_jv:.4f}")
            print(f"  Recall (macro): {rec_m_jv:.4f}")
            print(f"  F1 (weighted): {f1_w_jv:.4f}")
            print(f"  F1 (macro): {f1_m_jv:.4f}")
            print(f"\nSource Count Classification Metrics (No Matching):")
            print(f"  Precision (weighted): {prec_w_no_matching:.4f}")
            print(f"  Precision (macro): {prec_m_no_matching:.4f}")
            print(f"  Recall (weighted): {rec_w_no_matching:.4f}")
            print(f"  Recall (macro): {rec_m_no_matching:.4f}")
            print(f"  F1 (weighted): {f1_w_no_matching:.4f}")
            print(f"  F1 (macro): {f1_m_no_matching:.4f}")
            
            # Write combined classification metrics to log file
            r2_log_path = res_dir / log_name
            with open(r2_log_path, 'a') as r2_file:
                r2_file.write("\n" + "="*60 + "\n")
                r2_file.write("COMBINED CLASSIFICATION METRICS (ALL FOLDS COMBINED)\n")
                r2_file.write("="*60 + "\n")
                r2_file.write("\nSource Count Classification Metrics (Greedy Matching):\n")
                r2_file.write(f"  Precision (weighted): {prec_w_greedy:.4f}\n")
                r2_file.write(f"  Precision (macro):    {prec_m_greedy:.4f}\n")
                r2_file.write(f"  Recall (weighted):    {rec_w_greedy:.4f}\n")
                r2_file.write(f"  Recall (macro):       {rec_m_greedy:.4f}\n")
                r2_file.write(f"  F1 (weighted):        {f1_w_greedy:.4f}\n")
                r2_file.write(f"  F1 (macro):           {f1_m_greedy:.4f}\n")
                r2_file.write("\nSource Count Classification Metrics (JV Matching):\n")
                r2_file.write(f"  Precision (weighted): {prec_w_jv:.4f}\n")
                r2_file.write(f"  Precision (macro):    {prec_m_jv:.4f}\n")
                r2_file.write(f"  Recall (weighted):    {rec_w_jv:.4f}\n")
                r2_file.write(f"  Recall (macro):       {rec_m_jv:.4f}\n")
                r2_file.write(f"  F1 (weighted):        {f1_w_jv:.4f}\n")
                r2_file.write(f"  F1 (macro):           {f1_m_jv:.4f}\n")
                r2_file.write("\nSource Count Classification Metrics (No Matching):\n")
                r2_file.write(f"  Precision (weighted): {prec_w_no_matching:.4f}\n")
                r2_file.write(f"  Precision (macro):    {prec_m_no_matching:.4f}\n")
                r2_file.write(f"  Recall (weighted):    {rec_w_no_matching:.4f}\n")
                r2_file.write(f"  Recall (macro):       {rec_m_no_matching:.4f}\n")
                r2_file.write(f"  F1 (weighted):        {f1_w_no_matching:.4f}\n")
                r2_file.write(f"  F1 (macro):           {f1_m_no_matching:.4f}\n")
                r2_file.write("\n" + "="*60 + "\n\n")
            
            # Plot combined confusion matrices
            def plot_combined_confusion_matrix(cm, labels, title, output_path):
                """Plot and save combined confusion matrix."""
                cm = np.array(cm)
                fig, ax = plt.subplots(figsize=(8, 7))
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                
                ax.set(xticks=np.arange(cm.shape[1]),
                       yticks=np.arange(cm.shape[0]),
                       xticklabels=labels, yticklabels=labels,
                       title=title,
                       ylabel='True Source Count',
                       xlabel='Predicted Source Count')
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(int(cm[i, j]), 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black",
                               fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"Saved combined confusion matrix to {output_path}")
            
            # Save combined confusion matrices
            cm_path_no_matching = res_dir / f"confusion_matrix_source_count_{file_label}_all_folds_no_matching.pdf"
            plot_combined_confusion_matrix(
                cm_no_matching,
                all_labels,
                f'Source Count Confusion Matrix (No Matching) - All Folds Combined',
                cm_path_no_matching
            )
            
            cm_path_jv = res_dir / f"confusion_matrix_source_count_{file_label}_all_folds_jv.pdf"
            plot_combined_confusion_matrix(
                cm_jv,
                all_labels,
                f'Source Count Confusion Matrix (JV Matching) - All Folds Combined',
                cm_path_jv
            )
            
            cm_path_greedy = res_dir / f"confusion_matrix_source_count_{file_label}_all_folds_greedy.pdf"
            plot_combined_confusion_matrix(
                cm_greedy,
                all_labels,
                f'Source Count Confusion Matrix (Greedy Matching) - All Folds Combined',
                cm_path_greedy
            )
    except Exception as e:
        print(f"Warning: Could not generate combined confusion matrices: {e}")
        import traceback
        traceback.print_exc()


class PLACENetPlot:
    """Plotting utilities for PLACENet."""

    def __init__(self, config: Optional[PLACEPlotConfig] = None):
        cfg = config or PLACEPlotConfig()
        self.binx = cfg.binx
        self.biny = cfg.biny
        self.binz = cfg.binz

    def compare_histo(
        self,
        res_dir: Path,
        file_label: str,
        model_suffix: str = "",
        sample_idx: int = 0,
        binx: Optional[int] = None,
        biny: Optional[int] = None,
        binz: Optional[int] = None,
    ):
        """
        Compare predicted and actual histograms using box format (xwidth, xcentre, ywidth, ycentre, zwidth, zcentre).
        
        Args:
            res_dir: Directory containing test data and model files
            file_label: Label used in file names (e.g., "smooth_l1_jv")
            model_suffix: Optional suffix for model file (e.g., "_ciou")
            sample_idx: Index of sample to visualize (default: 0)
            binx, biny, binz: Histogram bin dimensions
        """
        binx = binx or self.binx
        biny = biny or self.biny
        binz = binz or self.binz

        # Load test data
        npz_candidates = sorted(
            glob.glob(str(res_dir / f"data_labels_test_{file_label}_kf*.npz")),
            key=os.path.getmtime,
            reverse=True,
        )

        if not npz_candidates:
            raise FileNotFoundError(f"No test data files found in {res_dir}")

        npz_path = npz_candidates[0]
        m = re.search(r"_kf(\d+)\.npz$", npz_path)
        kfold_str = m.group(1) if m else "0"

        with np.load(npz_path) as npz_file:
            data_test = npz_file["data"]
            labels_test = npz_file["labels"]

        if sample_idx >= len(data_test):
            raise ValueError(f"sample_idx {sample_idx} exceeds test set size {len(data_test)}")

        # Load model
        model_path = res_dir / f"model_{file_label}{model_suffix}_kf{kfold_str}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = tf.keras.models.load_model(model_path, compile=False)

        # Make predictions
        pred = model.predict(data_test[sample_idx : sample_idx + 1], verbose=0)
        
        # Get single sample
        y_true = labels_test[sample_idx : sample_idx + 1]  # (1, max_sources, 6)
        y_pred = pred[0:1]  # (1, max_sources, 6)

        # Extract boxes if YOLO-style
        if y_pred.shape[-1] == 7:
            y_pred, _ = extract_boxes_from_yolo_output(y_pred, confidence_threshold=0.5, pad_value=-1)
        if y_true.shape[-1] == 7:
            y_true = y_true[..., :6]

        # Apply JV matching
        pad_value = -1
        y_pred_matched = apply_jv_matching(y_true, y_pred, pad_value=pad_value)

        # Convert box format to voxel histograms
        prediction_hist = np.zeros((binx, biny, binz))
        actual_hist = np.zeros((binx, biny, binz))

        # Process predicted sources
        for source_idx in range(y_pred_matched.shape[1]):
            source_bbox = y_pred_matched[0, source_idx]
            
            # Skip padded sources (all -1)
            if np.all(source_bbox == pad_value):
                continue
            
            # Format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
            # Box extends from centre ± width/2
            xwidth, xcentre, ywidth, ycentre, zwidth, zcentre = source_bbox
            xmin = xcentre - xwidth / 2.0
            xmax = xcentre + xwidth / 2.0
            ymin = ycentre - ywidth / 2.0
            ymax = ycentre + ywidth / 2.0
            zmin = zcentre - zwidth / 2.0
            zmax = zcentre + zwidth / 2.0

            # Map box boundaries to voxel indices
            x_indices = np.arange(int(xmin), int(xmax) + 1)
            y_indices = np.arange(int(ymin), int(ymax) + 1)
            z_indices = np.arange(int(zmin), int(zmax) + 1)

            # Fill the voxels within the bounding box
            for i in x_indices:
                for j in y_indices:
                    for k in z_indices:
                        if 0 <= i < binx and 0 <= j < biny and 0 <= k < binz:
                            prediction_hist[i, j, k] = 1

        # Process actual sources
        for source_idx in range(y_true.shape[1]):
            source_bbox = y_true[0, source_idx]
            
            # Skip padded sources (all -1)
            if np.all(source_bbox == pad_value):
                continue
            
            # Format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
            # Box extends from centre ± width/2
            xwidth, xcentre, ywidth, ycentre, zwidth, zcentre = source_bbox
            xmin = xcentre - xwidth / 2.0
            xmax = xcentre + xwidth / 2.0
            ymin = ycentre - ywidth / 2.0
            ymax = ycentre + ywidth / 2.0
            zmin = zcentre - zwidth / 2.0
            zmax = zcentre + zwidth / 2.0

            # Map box boundaries to voxel indices
            x_indices = np.arange(int(xmin), int(xmax) + 1)
            y_indices = np.arange(int(ymin), int(ymax) + 1)
            z_indices = np.arange(int(zmin), int(zmax) + 1)

            # Fill the voxels within the bounding box
            for i in x_indices:
                for j in y_indices:
                    for k in z_indices:
                        if 0 <= i < binx and 0 <= j < biny and 0 <= k < binz:
                            actual_hist[i, j, k] = 1

        # Create cylindrical surfaces with fixed physical dimensions centered in plot space
        center_x = binx / 2.0
        center_y = biny / 2.0
        
        radius_inner = 16.0
        radius_outer = 21.0
        
        z_min_inner = 5.0
        z_max_inner = 77.0
        
        z_min_outer = 0.0
        z_max_outer = 82.0
        
        # Generate inner drum surface
        zg = np.linspace(z_min_inner, z_max_inner, 50)
        theta = np.linspace(0, 2 * np.pi, 50)
        theta_grid, zcyl = np.meshgrid(theta, zg)
        xcyl = radius_inner * np.cos(theta_grid) + center_x
        ycyl = radius_inner * np.sin(theta_grid) + center_y

        # Generate outer drum surface
        zg1 = np.linspace(z_min_outer, z_max_outer, 50)
        theta1 = np.linspace(0, 2 * np.pi, 50)
        theta_grid1, zcyl1 = np.meshgrid(theta1, zg1)
        xcyl1 = radius_outer * np.cos(theta_grid1) + center_x
        ycyl1 = radius_outer * np.sin(theta_grid1) + center_y

        # Calculate figure size
        base_width = 6
        base_height = 5
        z_scale_factor = binz / max(binx, biny)
        fig_height = base_height * max(1.0, z_scale_factor * 0.5)
        
        # Plot comparison
        fig = plt.figure(figsize=(base_width, fig_height), layout="constrained")
        ax1 = fig.add_subplot(111, projection="3d")
        
        # Plot cylindrical surfaces
        ax1.plot_surface(xcyl, ycyl, zcyl, color="y", alpha=0.2)
        ax1.plot_surface(xcyl1, ycyl1, zcyl1, color="y", alpha=0.2)
        
        # Plot voxels
        ax1.voxels(prediction_hist, facecolor="b", alpha=0.5)
        ax1.voxels(actual_hist, facecolor="r", alpha=1)

        legend_elements = [
            Patch(facecolor="r", edgecolor="r", label="Actual boxes"),
            Patch(facecolor="b", edgecolor="b", label="Predicted boxes"),
        ]
        ax1.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=2,
            fontsize=14,
        )

        ax1.set_xlim([0, binx])
        ax1.set_ylim([0, biny])
        ax1.set_zlim([0, binz])
        
        max_xy = max(binx, biny)
        ax1.set_box_aspect([max_xy, max_xy, binz * 0.6])

        ax1.tick_params(labelsize=14)
        ax1.set_xlabel("x (cm)", fontsize=14)
        ax1.set_ylabel("y (cm)", fontsize=14)
        ax1.set_zlabel("z (cm)", fontsize=14)

        output_path = res_dir / f"comparison_sample{sample_idx}.pdf"
        plt.savefig(output_path)
        print(f"Saved comparison plot to {output_path}")
        return fig

    @staticmethod
    def save_loss_curves(
        histories: Sequence,
        output_path: Path,
    ) -> None:
        """Save training vs validation loss curves to disk."""
        if not histories:
            print("No training histories available to plot.")
            return

        plt.figure(figsize=(8, 5))
        plotted = False
        for idx, history in enumerate(histories):
            loss = history.history.get("loss")
            val_loss = history.history.get("val_loss")
            if loss is None or val_loss is None:
                continue
            epochs = range(1, len(loss) + 1)
            plt.plot(epochs, loss, label=f"Fold {idx+1} - Train", alpha=0.7)
            plt.plot(epochs, val_loss, label=f"Fold {idx+1} - Val", linestyle="--", alpha=0.7)
            plotted = True

        if not plotted:
            print("No loss curves found in histories.")
            plt.close()
            return

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved loss curves to {output_path}")

    @staticmethod
    def plot_r2_scores(
        all_results: List[Dict],
        output_path: Path,
        matching_type: str = "greedy",  # "greedy" or "jv"
    ) -> None:
        """
        Plot R² scores showing overall, widths, and positions with error bars.
        
        Args:
            all_results: List of result dictionaries from evaluate_single_fold
            output_path: Path to save the plot
            matching_type: "greedy" or "jv" matching strategy to plot
        """
        if not all_results:
            print("No results available to plot R² scores.")
            return

        # Extract R² values
        if matching_type == "greedy":
            r2_overall = [r['r2_with_greedy'] for r in all_results if not np.isnan(r.get('r2_with_greedy', np.nan))]
            r2_widths = [r['r2_widths_greedy'] for r in all_results if not np.isnan(r.get('r2_widths_greedy', np.nan))]
            r2_positions = [r['r2_positions_greedy'] for r in all_results if not np.isnan(r.get('r2_positions_greedy', np.nan))]
        else:  # jv
            r2_overall = [r['r2_with_jv'] for r in all_results if not np.isnan(r.get('r2_with_jv', np.nan))]
            r2_widths = [r['r2_widths_jv'] for r in all_results if not np.isnan(r.get('r2_widths_jv', np.nan))]
            r2_positions = [r['r2_positions_jv'] for r in all_results if not np.isnan(r.get('r2_positions_jv', np.nan))]

        if not (r2_overall or r2_widths or r2_positions):
            print(f"No valid R² data found for {matching_type} matching.")
            return

        # Calculate statistics
        def calc_stats(values):
            if not values:
                return None, None
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            return mean_val, std_val

        overall_mean, overall_std = calc_stats(r2_overall)
        widths_mean, widths_std = calc_stats(r2_widths)
        positions_mean, positions_std = calc_stats(r2_positions)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_pos = np.arange(3)
        means = []
        stds = []
        labels = []
        colors = []
        markers = []
        
        if overall_mean is not None:
            means.append(overall_mean)
            stds.append(overall_std)
            labels.append("Overall R²")
            colors.append('#1f77b4')  # Blue
            markers.append('o')
        
        if widths_mean is not None:
            means.append(widths_mean)
            stds.append(widths_std)
            labels.append("Widths R²")
            colors.append('#ff7f0e')  # Orange
            markers.append('s')
        
        if positions_mean is not None:
            means.append(positions_mean)
            stds.append(positions_std)
            labels.append("Positions R²")
            colors.append('#2ca02c')  # Green
            markers.append('^')

        if not means:
            plt.close()
            return

        x_plot = np.arange(len(means))
        
        # Plot with error bars (using Std across folds)
        for i, (mean, std, color, marker, label) in enumerate(zip(means, stds, colors, markers, labels)):
            ax.errorbar(i, mean, yerr=std, fmt=marker, color=color, markersize=10, 
                       capsize=5, capthick=2, elinewidth=2, label=label, alpha=0.8)

        ax.set_xlim(-0.5, len(means) - 0.5)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x_plot)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel("R² Score", fontsize=12)
        ax.set_title(f"R² Scores Comparison ({matching_type.upper()} matching)", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.3, axis='y')
        ax.legend(loc='best', fontsize=10)
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved R² plot to {output_path}")

    @staticmethod
    def plot_confidence_distribution(
        pred_confidences: np.ndarray,
        true_confidences: Optional[np.ndarray] = None,
        output_path: Path = None,
        title: Optional[str] = None,
    ):
        """
        Plot the distribution of predicted confidence scores.
        
        Args:
            pred_confidences: Array of predicted confidence scores (can be any shape, will be flattened)
            true_confidences: Optional array of true confidence scores (1.0 for valid slots, 0.0 for empty)
            output_path: Path to save the plot (if None, plot is not saved)
            title: Optional title for the plot
        """
        if pred_confidences is None or pred_confidences.size == 0:
            print("Warning: No confidence scores available for plotting.")
            return
        
        # Flatten confidence arrays
        pred_conf_flat = pred_confidences.flatten()
        
        # Create figure with subplots
        if true_confidences is not None:
            true_conf_flat = true_confidences.flatten()
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Histogram of all predicted confidences
            ax1 = axes[0, 0]
            ax1.hist(pred_conf_flat, bins=50, alpha=0.7, color='#1f77b4', edgecolor='black', linewidth=0.5)
            ax1.axvline(pred_conf_flat.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {pred_conf_flat.mean():.3f}')
            ax1.axvline(np.median(pred_conf_flat), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(pred_conf_flat):.3f}')
            ax1.set_xlabel('Predicted Confidence Score', fontsize=11)
            ax1.set_ylabel('Frequency', fontsize=11)
            ax1.set_title('Distribution of All Predicted Confidence Scores', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            # Plot 2: Histogram separated by true confidence (valid vs empty slots)
            ax2 = axes[0, 1]
            valid_mask = true_conf_flat == 1.0
            empty_mask = true_conf_flat == 0.0
            
            if np.any(valid_mask):
                ax2.hist(pred_conf_flat[valid_mask], bins=50, alpha=0.6, color='#2ca02c', 
                        label=f'Valid slots (n={np.sum(valid_mask)})', edgecolor='black', linewidth=0.5)
            if np.any(empty_mask):
                ax2.hist(pred_conf_flat[empty_mask], bins=50, alpha=0.6, color='#ff7f0e', 
                        label=f'Empty slots (n={np.sum(empty_mask)})', edgecolor='black', linewidth=0.5)
            ax2.set_xlabel('Predicted Confidence Score', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title('Distribution by Slot Type', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.3)
            
            # Plot 3: Box plot comparing valid vs empty slots
            ax3 = axes[1, 0]
            box_data = []
            box_labels = []
            if np.any(valid_mask):
                box_data.append(pred_conf_flat[valid_mask])
                box_labels.append('Valid\nslots')
            if np.any(empty_mask):
                box_data.append(pred_conf_flat[empty_mask])
                box_labels.append('Empty\nslots')
            
            if box_data:
                bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
                colors = ['#2ca02c', '#ff7f0e']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                ax3.set_ylabel('Predicted Confidence Score', fontsize=11)
                ax3.set_title('Confidence Distribution Comparison', fontsize=12, fontweight='bold')
                ax3.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            # Plot 4: Statistics summary
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            stats_text = "Confidence Score Statistics\n" + "="*30 + "\n\n"
            stats_text += f"All Predictions:\n"
            stats_text += f"  Mean: {pred_conf_flat.mean():.4f}\n"
            stats_text += f"  Median: {np.median(pred_conf_flat):.4f}\n"
            stats_text += f"  Std: {pred_conf_flat.std():.4f}\n"
            stats_text += f"  Min: {pred_conf_flat.min():.4f}\n"
            stats_text += f"  Max: {pred_conf_flat.max():.4f}\n"
            stats_text += f"  Total: {len(pred_conf_flat)}\n\n"
            
            if np.any(valid_mask):
                valid_conf = pred_conf_flat[valid_mask]
                stats_text += f"Valid Slots:\n"
                stats_text += f"  Mean: {valid_conf.mean():.4f}\n"
                stats_text += f"  Median: {np.median(valid_conf):.4f}\n"
                stats_text += f"  Std: {valid_conf.std():.4f}\n"
                stats_text += f"  Count: {len(valid_conf)}\n\n"
            
            if np.any(empty_mask):
                empty_conf = pred_conf_flat[empty_mask]
                stats_text += f"Empty Slots:\n"
                stats_text += f"  Mean: {empty_conf.mean():.4f}\n"
                stats_text += f"  Median: {np.median(empty_conf):.4f}\n"
                stats_text += f"  Std: {empty_conf.std():.4f}\n"
                stats_text += f"  Count: {len(empty_conf)}\n"
            
            ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', 
                    verticalalignment='center', horizontalalignment='left')
            
        else:
            # Simpler plot if no true confidences available
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            ax1 = axes[0]
            ax1.hist(pred_conf_flat, bins=50, alpha=0.7, color='#1f77b4', edgecolor='black', linewidth=0.5)
            ax1.axvline(pred_conf_flat.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {pred_conf_flat.mean():.3f}')
            ax1.axvline(np.median(pred_conf_flat), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(pred_conf_flat):.3f}')
            ax1.set_xlabel('Predicted Confidence Score', fontsize=11)
            ax1.set_ylabel('Frequency', fontsize=11)
            ax1.set_title('Distribution of Predicted Confidence Scores', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            # Statistics
            ax2 = axes[1]
            ax2.axis('off')
            stats_text = "Confidence Score Statistics\n" + "="*30 + "\n\n"
            stats_text += f"Mean: {pred_conf_flat.mean():.4f}\n"
            stats_text += f"Median: {np.median(pred_conf_flat):.4f}\n"
            stats_text += f"Std: {pred_conf_flat.std():.4f}\n"
            stats_text += f"Min: {pred_conf_flat.min():.4f}\n"
            stats_text += f"Max: {pred_conf_flat.max():.4f}\n"
            stats_text += f"Total: {len(pred_conf_flat)}\n"
            ax2.text(0.1, 0.5, stats_text, fontsize=12, family='monospace', 
                    verticalalignment='center', horizontalalignment='left')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved confidence distribution plot to {output_path}")
        else:
            plt.show()
            plt.close()

    @staticmethod
    def plot_confidence_by_slot_type(
        pred_confidences: np.ndarray,
        true_confidences: np.ndarray,
        output_path: Path = None,
        title: Optional[str] = None,
    ):
        """
        Plot the distribution of predicted confidence scores separated by slot type (valid vs empty),
        with mean and median lines for each group.
        
        Args:
            pred_confidences: Array of predicted confidence scores (can be any shape, will be flattened)
            true_confidences: Array of true confidence scores (1.0 for valid slots, 0.0 for empty)
            output_path: Path to save the plot (if None, plot is not saved)
            title: Optional title for the plot
        """
        if pred_confidences is None or pred_confidences.size == 0:
            print("Warning: No confidence scores available for plotting.")
            return
        
        if true_confidences is None or true_confidences.size == 0:
            print("Warning: No true confidence labels available for plotting.")
            return
        
        # Flatten confidence arrays
        pred_conf_flat = pred_confidences.flatten()
        true_conf_flat = true_confidences.flatten()
        
        # Create masks
        valid_mask = true_conf_flat == 1.0
        empty_mask = true_conf_flat == 0.0
        
        if not np.any(valid_mask) and not np.any(empty_mask):
            print("Warning: No valid or empty slots found.")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        if np.any(valid_mask):
            valid_conf = pred_conf_flat[valid_mask]
            ax.hist(valid_conf, bins=50, alpha=0.6, color='#2ca02c', 
                   label=f'Valid slots (n={np.sum(valid_mask)})', edgecolor='black', linewidth=0.5)
            # Add mean and median lines for valid slots
            valid_mean = valid_conf.mean()
            valid_median = np.median(valid_conf)
            ax.axvline(valid_mean, color='darkgreen', linestyle='--', linewidth=2.5, 
                      label=f'Valid mean: {valid_mean:.3f}')
            ax.axvline(valid_median, color='green', linestyle=':', linewidth=2.5, 
                      label=f'Valid median: {valid_median:.3f}')
        
        if np.any(empty_mask):
            empty_conf = pred_conf_flat[empty_mask]
            ax.hist(empty_conf, bins=50, alpha=0.6, color='#ff7f0e', 
                   label=f'Empty slots (n={np.sum(empty_mask)})', edgecolor='black', linewidth=0.5)
            # Add mean and median lines for empty slots
            empty_mean = empty_conf.mean()
            empty_median = np.median(empty_conf)
            ax.axvline(empty_mean, color='darkorange', linestyle='--', linewidth=2.5, 
                      label=f'Empty mean: {empty_mean:.3f}')
            ax.axvline(empty_median, color='orange', linestyle=':', linewidth=2.5, 
                      label=f'Empty median: {empty_median:.3f}')
        
        ax.set_xlabel('Predicted Confidence Score', fontsize=16)
        ax.set_ylabel('Frequency', fontsize=16)
        if title:
            ax.set_title(title, fontsize=18, fontweight='bold')
        else:
            ax.set_title('Distribution by Slot Type', fontsize=18, fontweight='bold')
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        
        # Increase tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Set x-axis limits to start at 0 (after legend is placed)
        x_min = -0.05
        x_max = max(1.0, pred_conf_flat.max() + 0.05)
        ax.set_xlim(x_min, x_max)
        
        # Adjust layout to make room for legend on the right
        plt.subplots_adjust(right=0.75)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved confidence by slot type plot to {output_path}")
        else:
            plt.show()
            plt.close()

    @staticmethod
    def plot_confidence_diagnostics(
        pred_confidences: np.ndarray,
        true_confidences: np.ndarray,
        output_path: Path = None,
        title: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """
        Plot diagnostic curves for confidence score classification:
        - ROC curve
        - Precision-Recall curve
        - Threshold analysis showing overlap region
        
        Args:
            pred_confidences: Array of predicted confidence scores (can be any shape, will be flattened)
            true_confidences: Array of true confidence scores (1.0 for valid slots, 0.0 for empty)
            output_path: Path to save the plot (if None, plot is not saved)
            title: Optional title for the plot
            threshold: Threshold used for binary classification (default 0.5)
        """
        if pred_confidences is None or pred_confidences.size == 0:
            print("Warning: No confidence scores available for plotting.")
            return
        
        if true_confidences is None or true_confidences.size == 0:
            print("Warning: No true confidence labels available for plotting.")
            return
        
        # Flatten confidence arrays
        pred_conf_flat = pred_confidences.flatten()
        true_conf_flat = true_confidences.flatten()
        
        # Create masks
        valid_mask = true_conf_flat == 1.0
        empty_mask = true_conf_flat == 0.0
        
        if not np.any(valid_mask) or not np.any(empty_mask):
            print("Warning: Need both valid and empty slots for diagnostic plots.")
            return
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(true_conf_flat, pred_conf_flat)
        roc_auc = roc_auc_score(true_conf_flat, pred_conf_flat)
        
        # Calculate Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(true_conf_flat, pred_conf_flat)
        pr_auc = np.trapz(precision, recall)
        
        # Calculate metrics at different thresholds
        thresholds_to_test = np.linspace(0.0, 1.0, 101)
        f1_scores = []
        precisions = []
        recalls = []
        
        for thresh in thresholds_to_test:
            pred_binary = (pred_conf_flat >= thresh).astype(int)
            if len(np.unique(pred_binary)) > 1:  # Need both classes
                prec = precision_score(true_conf_flat, pred_binary, zero_division=0)
                rec = recall_score(true_conf_flat, pred_binary, zero_division=0)
                f1 = f1_score(true_conf_flat, pred_binary, zero_division=0)
            else:
                prec = rec = f1 = 0.0
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)
        
        # Find optimal threshold (max F1)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds_to_test[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 5))
        
        # Plot 1: ROC curve
        ax1 = plt.subplot(1, 3, 1)
        ax1.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random classifier')
        ax1.scatter([fpr[np.argmin(np.abs(roc_thresholds - threshold))]], 
                   [tpr[np.argmin(np.abs(roc_thresholds - threshold))]], 
                   color='red', s=100, zorder=5, label=f'Threshold = {threshold:.2f}')
        ax1.set_xlabel('False Positive Rate', fontsize=14)
        ax1.set_ylabel('True Positive Rate', fontsize=14)
        ax1.set_title('ROC Curve', fontsize=16, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Plot 2: Precision-Recall curve
        ax2 = plt.subplot(1, 3, 2)
        ax2.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        baseline = np.sum(true_conf_flat == 1.0) / len(true_conf_flat)
        ax2.axhline(y=baseline, color='gray', lw=1, linestyle='--', label=f'Baseline = {baseline:.3f}')
        pr_thresh_idx = np.argmin(np.abs(pr_thresholds - threshold))
        ax2.scatter([recall[pr_thresh_idx]], [precision[pr_thresh_idx]], 
                   color='red', s=100, zorder=5, label=f'Threshold = {threshold:.2f}')
        ax2.set_xlabel('Recall', fontsize=14)
        ax2.set_ylabel('Precision', fontsize=14)
        ax2.set_title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        ax2.legend(loc='lower left', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        
        # Plot 3: Metrics vs Threshold + Overlap region
        ax3 = plt.subplot(1, 3, 3)
        ax3.plot(thresholds_to_test, f1_scores, label='F1 Score', lw=2, color='blue')
        ax3.plot(thresholds_to_test, precisions, label='Precision', lw=2, color='green')
        ax3.plot(thresholds_to_test, recalls, label='Recall', lw=2, color='orange')
        ax3.axvline(x=threshold, color='red', linestyle='--', lw=2, label=f'Current threshold = {threshold:.2f}')
        ax3.axvline(x=optimal_threshold, color='purple', linestyle=':', lw=2, 
                   label=f'Optimal threshold = {optimal_threshold:.2f} (F1={optimal_f1:.3f})')
        
        # Highlight overlap region (where both distributions have significant density)
        valid_conf = pred_conf_flat[valid_mask]
        empty_conf = pred_conf_flat[empty_mask]
        
        # Find overlap region (where both distributions have >5% of their density)
        overlap_min = max(valid_conf.min(), empty_conf.min())
        overlap_max = min(valid_conf.max(), empty_conf.max())
        
        # Calculate density in overlap region
        if overlap_min < overlap_max:
            valid_in_overlap = np.sum((valid_conf >= overlap_min) & (valid_conf <= overlap_max)) / len(valid_conf)
            empty_in_overlap = np.sum((empty_conf >= overlap_min) & (empty_conf <= overlap_max)) / len(empty_conf)
            
            if valid_in_overlap > 0.05 or empty_in_overlap > 0.05:
                ax3.axvspan(overlap_min, overlap_max, alpha=0.2, color='yellow', 
                          label=f'Overlap region ({overlap_min:.2f}-{overlap_max:.2f})')
        
        ax3.set_xlabel('Threshold', fontsize=14)
        ax3.set_ylabel('Score', fontsize=14)
        ax3.set_title('Metrics vs Threshold', fontsize=16, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1])
        
        # Increase tick label sizes
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        if title:
            fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved confidence diagnostics plot to {output_path}")
            print(f"  Optimal threshold: {optimal_threshold:.3f} (F1={optimal_f1:.3f})")
            print(f"  Current threshold: {threshold:.3f} (F1={f1_scores[np.argmin(np.abs(thresholds_to_test - threshold))]:.3f})")
        else:
            plt.show()
            plt.close()

