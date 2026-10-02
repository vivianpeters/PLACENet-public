# PLACENet - Simplified CNN-based model with bounding box output and confidence output
# Supports smooth_l1, ciou, and hybrid loss types
# Supports greedy or JV (Jonker-Volgenant) matching

from __future__ import annotations
import gc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import json
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import register_keras_serializable

from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from .PLACENet_prep import PLACEDataset


@dataclass
class PLACENetConfig:
    """Configuration for PLACENet model."""
    pad_value: int = -1 # Value to pad the empty slots with
    l2_reg: float = 0.01 # L2 regularization parameter
    learning_rate: float = 1e-4 # Learning rate (fixed for now)
    loss_type: str = "hybrid"  # "smooth_l1", "ciou", or "hybrid"
    matching_strategy: str = "jv"  # "greedy" or "jv" (Jonker-Volgenant)
    epochs: int = 3000 # Number of epochs to train for
    batch_size: int = 32 # Batch size
    folds: int = 5 # Number of folds for cross-validation
    run_name: Optional[str] = None
    max_sources: int = 1 # Maximum number of sources to classify
    delta: float = 1.0  # For smooth_l1
    ciou_weight: float = 1.0 # Weight for the CIoU loss
    smooth_weight: float = 1.0 # Weight for the smooth L1 loss
    confidence_weight: float = 1.5 # Weight for the confidence loss
    noobj_weight: float = 0.1 # Weight for the no object loss
    enable_augmentation: bool = False # If True, enable data augmentation
    augmentation_multiplier: int = 0 # How many augmented copies to create per original sample
    augmenter_kwargs: Dict[str, Any] = field(default_factory=dict) # Keyword arguments for the augmenter


@dataclass
class PLACENetTrainingResult:
    """Result of training the PLACENet model."""
    run_label: str # Name of the run
    metrics: List[Dict[str, float]] # Metrics for the run
    histories: List[Any] # Histories for the run
    dataset_name: Optional[str] = None # Name of the dataset


# ============================================================================
# Pairwise cost computation functions
# ============================================================================
_EPS = 1e-7 # Epsilon for numerical stability


def _ensure_positive_widths(boxes):
    """Ensure positive widths for predicted boxes.
    Input format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
    Output format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
    """
    boxes = tf.cast(boxes, tf.float32) # Cast the boxes to float32
    xw = tf.nn.softplus(boxes[..., 0]) + 1e-6
    xcentre = boxes[..., 1] # x centre
    yw = tf.nn.softplus(boxes[..., 2]) + 1e-6
    ycentre = boxes[..., 3] # y centre
    zw = tf.nn.softplus(boxes[..., 4]) + 1e-6
    zcentre = boxes[..., 5] # z centre
    return tf.stack([xw, xcentre, yw, ycentre, zw, zcentre], axis=-1)


def _abs_gt_widths(boxes):
    """Use absolute widths for ground truth boxes.
    Input format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
    Output format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
    """
    boxes = tf.cast(boxes, tf.float32)
    xw = tf.abs(boxes[..., 0]) + 1e-7
    xcentre = boxes[..., 1]
    yw = tf.abs(boxes[..., 2]) + 1e-7
    ycentre = boxes[..., 3]
    zw = tf.abs(boxes[..., 4]) + 1e-7
    zcentre = boxes[..., 5]
    return tf.stack([xw, xcentre, yw, ycentre, zw, zcentre], axis=-1)


def _pairwise_smooth_l1_batch(y_true, y_pred, delta=1.0):
    """Compute pairwise smooth L1 costs for batch."""
    # y_true, y_pred: (B, S, 6)
    yt = tf.expand_dims(y_true, 2)  # (B, S, 1, 6)
    yp = tf.expand_dims(y_pred, 1)  # (B, 1, S, 6)
    diff = yt - yp  # (B, S, S, 6)
    absdiff = tf.abs(diff)
    mask_quad = tf.less_equal(absdiff, delta)
    quad = 0.5 * tf.square(absdiff)
    linear = delta * (absdiff - 0.5 * delta)
    per_elem = tf.where(mask_quad, quad, linear)
    per_box = tf.reduce_sum(per_elem, axis=-1)  # (B, S, S)
    return per_box


# ============================================================================
# Spectrum augmentation functions
# ============================================================================
class PLACESpectrumAugmenter:
    """Augmenter for spectrum data."""
    def __init__(
        self,
        poisson_noise: bool = True, # If True, add Poisson noise
        energy_shift: bool = True, # If True, shift the energy
        intensity_scale: bool = True, # If True, scale the intensity
        detector_dropout: bool = True, # If True, dropout the detectors
        energy_shift_range: float = 0.02, # Range of the energy shift
        intensity_scale_range: Tuple[float, float] = (0.8, 1.2), # Range of the intensity scale
        detector_dropout_prob: float = 0.1, # Probability of dropping a detector
        poisson_scale: float = 500.0, # Scale for the Poisson noise
    ):
        self.poisson_noise = poisson_noise
        self.energy_shift = energy_shift
        self.intensity_scale = intensity_scale
        self.detector_dropout = detector_dropout
        self.energy_shift_range = energy_shift_range
        self.intensity_scale_range = intensity_scale_range
        self.detector_dropout_prob = detector_dropout_prob
        self.poisson_scale = poisson_scale

    def augment_batch(self, batch: np.ndarray) -> np.ndarray:
        """Augment a batch of spectra."""
        if batch.size == 0:
            return batch
        return np.asarray(
            [self._augment_single(sample) for sample in batch],
            dtype=batch.dtype,
        )

    def _augment_single(self, spectrum: np.ndarray) -> np.ndarray:
        """Augment a single spectrum."""
        augmented = spectrum.copy()
        if self.intensity_scale and np.random.random() < 0.5:
            scale = np.random.uniform(*self.intensity_scale_range)
            augmented *= scale
        if self.energy_shift and np.random.random() < 0.5:
            shift_fraction = np.random.uniform(-self.energy_shift_range, self.energy_shift_range)
            augmented = self._shift_spectrum(augmented, shift_fraction)
        if self.poisson_noise:
            augmented = self._add_poisson_noise(augmented)
        if self.detector_dropout and np.random.random() < 0.3:
            augmented = self._dropout_detectors(augmented)
        return augmented

    def _shift_spectrum(self, spectrum: np.ndarray, fraction: float) -> np.ndarray:
        """Shift the spectrum by a fraction of the bins."""
        n_detectors, n_bins, _ = spectrum.shape
        shift_bins = int(fraction * n_bins)
        if shift_bins == 0:
            return spectrum
        shifted = np.zeros_like(spectrum)
        if shift_bins > 0:
            shifted[:, shift_bins:, :] = spectrum[:, :-shift_bins, :]
        else:
            shifted[:, :shift_bins, :] = spectrum[:, -shift_bins:, :]
        return shifted

    def _add_poisson_noise(self, spectrum: np.ndarray) -> np.ndarray:
        """Add Poisson noise to the spectrum."""
        counts = np.maximum(spectrum * self.poisson_scale, 0)
        noisy = np.random.poisson(counts)
        return noisy / self.poisson_scale

    def _dropout_detectors(self, spectrum: np.ndarray) -> np.ndarray:
        """Dropout the detectors from the spectrum."""
        drop_mask = np.random.rand(spectrum.shape[0]) < self.detector_dropout_prob
        spectrum[drop_mask] = 0
        return spectrum

# ============================================================================
# Custom loss functions and metrics
# ============================================================================
class PLACENetCustomLoss:
    """Custom loss functions and metrics for PLACENet."""
    
    def __init__(self, pad_value: int):
        self.PAD_VALUE = pad_value

    def masked_iou_metric(self, y_true, y_pred):
        """Compute masked IoU metric."""
        return 1.0 - self.masked_iou_loss(y_true, y_pred)

def _pairwise_ciou3d_batch(y_true, y_pred, pad_value=-1, eps=_EPS):
    """Compute pairwise CIoU3D *costs* for batch.
    Input format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
    Padded boxes (all values = pad_value) are masked out to avoid invalid CIoU computation.
    """
    pad_val = tf.cast(pad_value, y_true.dtype)
    
    # Check if boxes are padded (all 6 dimensions equal pad_value)
    true_is_padded = tf.reduce_all(tf.abs(y_true - pad_val) < 1e-3, axis=-1)  # (B, S)
    pred_is_padded = tf.reduce_all(tf.abs(y_pred - pad_val) < 1e-3, axis=-1)  # (B, S)
    
    T = _abs_gt_widths(y_true)
    P = _ensure_positive_widths(y_pred)
    # broadcast to (B, S, S, 6)
    T_exp = tf.expand_dims(T, 2)
    P_exp = tf.expand_dims(P, 1)

    xw_t = T_exp[..., 0]; xcentre_t = T_exp[..., 1]
    yw_t = T_exp[..., 2]; ycentre_t = T_exp[..., 3]
    zw_t = T_exp[..., 4]; zcentre_t = T_exp[..., 5]

    xw_p = P_exp[..., 0]; xcentre_p = P_exp[..., 1]
    yw_p = P_exp[..., 2]; ycentre_p = P_exp[..., 3]
    zw_p = P_exp[..., 4]; zcentre_p = P_exp[..., 5]

    # Convert from centre-based to min/max for IoU calculation
    xmin_t = xcentre_t - xw_t / 2.0
    xmax_t = xcentre_t + xw_t / 2.0
    ymin_t = ycentre_t - yw_t / 2.0
    ymax_t = ycentre_t + yw_t / 2.0
    zmin_t = zcentre_t - zw_t / 2.0
    zmax_t = zcentre_t + zw_t / 2.0

    xmin_p = xcentre_p - xw_p / 2.0
    xmax_p = xcentre_p + xw_p / 2.0
    ymin_p = ycentre_p - yw_p / 2.0
    ymax_p = ycentre_p + yw_p / 2.0
    zmin_p = zcentre_p - zw_p / 2.0
    zmax_p = zcentre_p + zw_p / 2.0

    # Ensure min <= max for robustness
    xmin_t_actual = tf.minimum(xmin_t, xmax_t)
    xmax_t_actual = tf.maximum(xmin_t, xmax_t)
    ymin_t_actual = tf.minimum(ymin_t, ymax_t)
    ymax_t_actual = tf.maximum(ymin_t, ymax_t)
    zmin_t_actual = tf.minimum(zmin_t, zmax_t)
    zmax_t_actual = tf.maximum(zmin_t, zmax_t)
    
    xmin_p_actual = tf.minimum(xmin_p, xmax_p)
    xmax_p_actual = tf.maximum(xmin_p, xmax_p)
    ymin_p_actual = tf.minimum(ymin_p, ymax_p)
    ymax_p_actual = tf.maximum(ymin_p, ymax_p)
    zmin_p_actual = tf.minimum(zmin_p, zmax_p)
    zmax_p_actual = tf.maximum(zmin_p, zmax_p)

    ix_min = tf.maximum(xmin_t_actual, xmin_p_actual)
    iy_min = tf.maximum(ymin_t_actual, ymin_p_actual)
    iz_min = tf.maximum(zmin_t_actual, zmin_p_actual)
    ix_max = tf.minimum(xmax_t_actual, xmax_p_actual)
    iy_max = tf.minimum(ymax_t_actual, ymax_p_actual)
    iz_max = tf.minimum(zmax_t_actual, zmax_p_actual)

    iw = tf.maximum(ix_max - ix_min, 0.0)
    ih = tf.maximum(iy_max - iy_min, 0.0)
    idp = tf.maximum(iz_max - iz_min, 0.0)
    inter = iw * ih * idp

    vol_t = xw_t * yw_t * zw_t
    vol_p = xw_p * yw_p * zw_p
    union = vol_t + vol_p - inter
    union = tf.maximum(union, eps)
    iou = inter / union
    iou = tf.clip_by_value(iou, 0.0, 1.0)

    # Calculate centers (already have them, but recalculate from actual min/max for consistency)
    cx_t = 0.5 * (xmin_t_actual + xmax_t_actual)
    cy_t = 0.5 * (ymin_t_actual + ymax_t_actual)
    cz_t = 0.5 * (zmin_t_actual + zmax_t_actual)
    cx_p = 0.5 * (xmin_p_actual + xmax_p_actual)
    cy_p = 0.5 * (ymin_p_actual + ymax_p_actual)
    cz_p = 0.5 * (zmin_p_actual + zmax_p_actual)

    dx = cx_t - cx_p
    dy = cy_t - cy_p
    dz = cz_t - cz_p
    rho2 = dx*dx + dy*dy + dz*dz

    # Enclosing box
    encl_min_x = tf.minimum(xmin_t_actual, xmin_p_actual)
    encl_min_y = tf.minimum(ymin_t_actual, ymin_p_actual)
    encl_min_z = tf.minimum(zmin_t_actual, zmin_p_actual)
    encl_max_x = tf.maximum(xmax_t_actual, xmax_p_actual)
    encl_max_y = tf.maximum(ymax_t_actual, ymax_p_actual)
    encl_max_z = tf.maximum(zmax_t_actual, zmax_p_actual)
    diag2 = tf.square(encl_max_x - encl_min_x) + tf.square(encl_max_y - encl_min_y) + tf.square(encl_max_z - encl_min_z)
    diag2 = tf.maximum(diag2, eps)

    # Compute aspect ratio consistency term v
    # Add numerical stability: avoid division by very small numbers
    width_sum_sq = xw_t**2 + yw_t**2 + zw_t**2
    width_sum_sq = tf.maximum(width_sum_sq, eps)  # Ensure >= eps
    v = ((xw_t - xw_p)**2 + (yw_t - yw_p)**2 + (zw_t - zw_p)**2) / width_sum_sq
    
    # Clip v to reasonable range to prevent extreme values while preserving relative ordering
    # This prevents alpha*v from becoming unbounded
    v = tf.clip_by_value(v, 0.0, 10.0)
    
    # Compute alpha with numerical stability
    alpha_denom = 1.0 - iou + v + eps
    alpha_denom = tf.maximum(alpha_denom, eps)  # Ensure >= eps
    alpha = v / alpha_denom
    
    # Clip alpha to reasonable range to avoid extreme values
    alpha = tf.clip_by_value(alpha, 0.0, 1.0)

    ciou = iou - (rho2 / diag2) - alpha * v
    # Only clip upper bound to 1.0 (CIoU should never exceed perfect match)
    # Allow lower bound to be more negative to properly penalize very bad matches
    # With v clipped to [0, 10] and alpha to [0, 1], worst case is approximately -11
    ciou = tf.clip_by_value(ciou, -11.0, 1.0 - eps)
    cost = 1.0 - ciou
    
    # Mask out costs for padded boxes (set to large value so they're ignored in matching)
    # Broadcast masks: (B, S) -> (B, S, S)
    true_padded_exp = tf.expand_dims(true_is_padded, 2)  # (B, S, 1)
    pred_padded_exp = tf.expand_dims(pred_is_padded, 1)   # (B, 1, S)
    either_padded = tf.logical_or(true_padded_exp, pred_padded_exp)  # (B, S, S)
    
    # Replace costs for padded pairs with large value (will be masked later)
    cost = tf.where(either_padded, tf.constant(1e10, dtype=cost.dtype), cost)
    
    # Ensure no NaN or Inf values (replace with large cost)
    cost = tf.where(tf.math.is_finite(cost), cost, tf.constant(1e10, dtype=cost.dtype))
    
    return cost


# ============================================================================
# YOLO-style loss wrapper with matching
# ============================================================================

@tf.keras.utils.register_keras_serializable(package="PLACENet")
class YOLOStyleMatchingLoss(tf.keras.losses.Loss):
    """
    YOLO-style loss that combines box regression with confidence prediction.
    Output format: (max_sources, 7) where last dimension is [boxes(6), confidence(1)]
    Uses matching (greedy or JV) for box assignment.
    """
    def __init__(
        self,
        base: str = "hybrid", # "smooth_l1", "ciou", or "hybrid"
        pad_value: float | int = -1,
        max_sources: int = 5, # Maximum number of sources to classify
        matching_strategy: str = "jv",  # "greedy" or "jv"
        delta: float = 1.0, # Delta for the smooth L1 loss
        ciou_weight: float = 1.0, # Weight for the CIoU loss
        smooth_weight: float = 1.0, # Weight for the smooth L1 loss
        confidence_weight: float = 1.5, # Weight for the confidence loss
        noobj_weight: float = 0.1, # Weight for the no object loss
        name: str = "yolo_style_matching_loss", # Name of the loss since Keras needs a name
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, 
        **kwargs,
    ):
        super().__init__(name=name, reduction=reduction, **kwargs)
        assert base in ("smooth_l1", "ciou", "hybrid")
        assert matching_strategy in ("greedy", "jv")
        self.base = base
        self.pad_value = pad_value
        self.max_sources = int(max_sources)
        self.matching_strategy = matching_strategy
        self.delta = float(delta)
        self.ciou_weight = float(ciou_weight)
        self.smooth_weight = float(smooth_weight)
        self.confidence_weight = float(confidence_weight)
        self.noobj_weight = float(noobj_weight)
        matching_type = "GREEDY" if matching_strategy == "greedy" else "JV (Jonker-Volgenant)"
        print(f"[YOLOStyleMatchingLoss] Initialized with matching: {matching_type}, base={base}, confidence_weight={confidence_weight}, noobj_weight={noobj_weight}")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "base": self.base,
            "pad_value": self.pad_value,
            "max_sources": self.max_sources,
            "matching_strategy": self.matching_strategy,
            "delta": self.delta,
            "ciou_weight": self.ciou_weight,
            "smooth_weight": self.smooth_weight,
            "confidence_weight": self.confidence_weight,
            "noobj_weight": self.noobj_weight,
        })
        return cfg

    def _split_outputs(self, y_pred):
        """Split predictions into boxes and confidence."""
        # y_pred: (B, S, 7) where last dim is [boxes(6), confidence(1)]
        boxes = y_pred[..., :6]  # (B, S, 6)
        confidence = y_pred[..., 6:7]  # (B, S, 1) - keep dim for consistency
        confidence = tf.squeeze(confidence, axis=-1)  # (B, S)
        return boxes, confidence

    def _confidence_loss(self, conf_true, conf_pred):
        """Compute YOLO-style confidence loss using binary cross-entropy."""
        # Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
        conf_pred = tf.clip_by_value(conf_pred, 1e-7, 1.0 - 1e-7)
        
        # Object confidence loss (when conf_true=1.0)
        obj_loss = -conf_true * tf.math.log(conf_pred)
        
        # No-object confidence loss (when conf_true=0.0)
        noobj_loss = -(1.0 - conf_true) * tf.math.log(1.0 - conf_pred)
        
        # Combine with different weights
        confidence_loss = obj_loss + self.noobj_weight * noobj_loss
        
        return tf.reduce_mean(confidence_loss)

    def _mask_costs(self, cost, true_mask, pred_mask):
        """Mask invalid entries in cost matrix."""
        inf = tf.constant(1e10, dtype=cost.dtype)
        tm = tf.logical_not(true_mask)
        pm = tf.logical_not(pred_mask)
        row_mask = tf.cast(tm, cost.dtype) * inf
        col_mask = tf.cast(pm, cost.dtype) * inf
        cost = cost + tf.expand_dims(row_mask, 2) + tf.expand_dims(col_mask, 1)
        return cost

    def _jv_numpy_batch_indices(self, cost_batch_np):
        """
        Batch-wise JV (Jonker-Volgenant) matching that returns INDICES.
        This matches the evaluation implementation using linear_sum_assignment.
        
        Args:
            cost_batch_np: (B, S, S) numpy array of costs
        Returns:
            indices: (B, S, 3) array where indices[b, k, :] = [b, row, col] for k-th match in batch b
            counts: (B,) array of number of valid matches per sample
        """
        B, S, _ = cost_batch_np.shape
        indices = np.full((B, S, 3), -1, dtype=np.int32)  # [batch_idx, row, col]
        counts = np.zeros((B,), dtype=np.int32)
        
        for b in range(B):
            cm = cost_batch_np[b]  # (S, S)
            
            # If all entries are large (invalid), skip this sample
            if np.isfinite(cm).sum() == 0 or cm.size == 0:
                continue
            
            try:
                # Use linear_sum_assignment (Jonker-Volgenant algorithm)
                row_ind, col_ind = linear_sum_assignment(cm)
            except Exception as e:
                print(f"Error in linear_sum_assignment for batch {b}: {e}")
                # Fallback to greedy in numpy
                row_ind = []
                col_ind = []
                cm_copy = cm.copy()
                K = min(cm.shape)
                for _ in range(K):
                    idx = np.unravel_index(np.argmin(cm_copy, axis=None), cm_copy.shape)
                    row_ind.append(idx[0])
                    col_ind.append(idx[1])
                    cm_copy[idx[0], :] = 1e10
                    cm_copy[:, idx[1]] = 1e10
                row_ind = np.array(row_ind, dtype=np.int32)
                col_ind = np.array(col_ind, dtype=np.int32)
            
            # Filter out masked assignments (where cost is very large)
            if len(row_ind) > 0:
                valid_mask = np.isfinite(cm[row_ind, col_ind]) & (cm[row_ind, col_ind] < 1e9)
                row_ind = row_ind[valid_mask]
                col_ind = col_ind[valid_mask]
                
                n_matches = len(row_ind)
                if n_matches > 0:
                    indices[b, :n_matches, 0] = b
                    indices[b, :n_matches, 1] = row_ind
                    indices[b, :n_matches, 2] = col_ind
                    counts[b] = n_matches
        
        return indices, counts

    def _jv_tf_batch(self, cost_batch, match_cost=None):
        """
        Batch-wise JV matching that maintains gradient flow.
        Uses numpy for matching but gathers from TensorFlow tensor for gradients.
        """
        B = tf.shape(cost_batch)[0]
        S = self.max_sources
        
        if match_cost is None:
            match_cost = cost_batch
            
        # Get indices from numpy JV (non-differentiable)
        indices_np, counts_np = tf.numpy_function(
            self._jv_numpy_batch_indices,
            [match_cost],
            [tf.int32, tf.int32]
        )
        indices_np.set_shape((None, S, 3))
        counts_np.set_shape((None,))
        
        def per_sample_gather(args):
            """Gather matched costs for one sample"""
            sample_idx, sample_indices, count = args
            
            valid_indices = sample_indices[:count]
            
            def no_matches():
                return tf.constant(0.0, dtype=tf.float32)
            
            def has_matches():
                rows = valid_indices[:, 1]
                cols = valid_indices[:, 2]
                gather_indices = tf.stack([rows, cols], axis=1)
                costs = tf.gather_nd(cost_batch[sample_idx], gather_indices)
                costs = tf.where(tf.math.is_finite(costs), costs, tf.constant(0.0, dtype=costs.dtype))
                return tf.reduce_mean(costs)
            
            return tf.cond(tf.equal(count, 0), no_matches, has_matches)
        
        sample_indices = tf.range(B, dtype=tf.int32)
        mean_costs = tf.map_fn(
            per_sample_gather,
            (sample_indices, indices_np, counts_np),
            fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32)
        )
        # Return indices and counts alongside the mean cost to allow confidence target assignment
        return mean_costs, indices_np, counts_np
    
    def _greedy_tf_batch(self, cost_batch, match_cost=None):
        """Batch-wise greedy matching algorithm."""
        B = tf.shape(cost_batch)[0]
        S = self.max_sources
        inf = tf.constant(1e10, dtype=cost_batch.dtype)
        
        if match_cost is None:
            match_cost = cost_batch
            
        cost_for_matching = tf.identity(match_cost)
        total = tf.zeros((B,), dtype=cost_batch.dtype)
        counts = tf.zeros((B,), dtype=tf.int32)
        indices = tf.zeros((B, S, 3), dtype=tf.int32) # tracks per-step (batch, pred, gt) assignment indices

        def cond(step, cost, total, counts, indices):
            return tf.less(step, S)

        def body(step, cost_for_matching, total, counts, indices):
            flat = tf.reshape(cost_for_matching, (B, -1))
            argmin = tf.argmin(flat, axis=1, output_type=tf.int32)
            i = argmin // S
            j = argmin % S
            idx = tf.stack([tf.range(B, dtype=tf.int32), i, j], axis=1)
            
            # Gather from original cost_batch for the loss, but from cost_for_matching to check validity
            chosen = tf.gather_nd(cost_batch, idx)
            chosen_matching_cost = tf.gather_nd(cost_for_matching, idx)
            
            chosen = tf.where(tf.math.is_finite(chosen_matching_cost) & (chosen_matching_cost < 1e9), chosen, 0.0)
            total = total + chosen
            counts = counts + tf.where(tf.math.is_finite(chosen_matching_cost) & (chosen_matching_cost < 1e9), 1, 0)
            
            row_mask = tf.tensor_scatter_nd_update(
                tf.zeros((B, S), dtype=cost_for_matching.dtype),
                tf.stack([tf.range(B, dtype=tf.int32), i], axis=1),
                tf.ones((B,), dtype=cost_for_matching.dtype) * inf
            )
            col_mask = tf.tensor_scatter_nd_update(
                tf.zeros((B, S), dtype=cost_for_matching.dtype),
                tf.stack([tf.range(B, dtype=tf.int32), j], axis=1),
                tf.ones((B,), dtype=cost_for_matching.dtype) * inf
            )
            cost_for_matching = cost_for_matching + tf.expand_dims(row_mask, 2) + tf.expand_dims(col_mask, 1)

            # Record the (batch, pred_slot, gt_slot) assignment for this step
            indices = tf.tensor_scatter_nd_update(
                indices,
                tf.stack([tf.range(B, dtype=tf.int32), tf.fill((B,), step)], axis=1),
                tf.stack([tf.range(B, dtype=tf.int32), i, j], axis=1)
            )
            return step + 1, cost_for_matching, total, counts, indices

        step0 = tf.constant(0, dtype=tf.int32)
        _, cost_fin, total_fin, counts_fin, indices_fin = tf.while_loop(
            cond, body,
            loop_vars=[step0, cost_for_matching, total, counts, indices],
            maximum_iterations=S
        )
        counts_float = tf.cast(counts_fin, cost_fin.dtype) # counts_fin is int, must use float for division below
        mean_costs = tf.where(counts_float > 0, total_fin / counts_float, tf.zeros_like(total_fin))
        return mean_costs, indices_fin, counts_fin

    # Assign GT confidence targets to prediction slots based on the bipartite matching.
    def _match_conf_tf(self, conf_true, indices, counts):
        """
        Construct matched confidence targets (B,S) based on dynamic matching indices.
        Prediction slot k matched to GT row j inherits conf_true[b,j].
        Unmatched prediction slots remain 0.
        Args:
            conf_true: (B,S) batch of confidence targets
            indices_np: (B,S,3) numpy array of matching indices
            counts_np: (B,) array of number of valid matches per sample
        Returns:
            matched_conf: (B,S) tensor of matched confidence targets
        """
        B = tf.shape(conf_true)[0]
        S = self.max_sources

        def per_sample_target(args):
            """Build matched confidence targets for one sample"""
            sample_idx, sample_matches, count = args # sample_matches (renamed from sample_indices) contains [batch_idx, row, col], count is the number of valid matches
            valid_indices = sample_matches[:count] # take the first 'count' valid matches
            
            def no_matches():
                return tf.zeros((S,), dtype=tf.float32) # if no matches, return zeros
            
            def has_matches():
                gt_rows = valid_indices[:, 1] # GT indices
                pred_cols = valid_indices[:, 2] # pred indices
                gt_conf_vals = tf.gather(conf_true[sample_idx], gt_rows) # take the GT confidence values
                scatter_idx = tf.expand_dims(pred_cols, axis=-1) # expand the prediction indices to be used for scatter_nd
                return tf.scatter_nd(scatter_idx, gt_conf_vals, shape=(S,)) # scatter the GT confidence values into the prediction slots
            
            return tf.cond(tf.equal(count, 0), no_matches, has_matches) # if count is 0, return zeros, otherwise return the matched confidence values
        
        batch_indices = tf.range(B, dtype=tf.int32) # renamed to avoid confusion with sample_indices
        matched_conf = tf.map_fn(
            per_sample_target,
            (batch_indices, indices, counts),
            fn_output_signature=tf.TensorSpec(shape=(S,), dtype=tf.float32)
        ) # map over each sample to get the matched confidence values

        return matched_conf

    def call(self, y_true, y_pred):
        """
        Compute YOLO-style loss with matching.
        
        Args:
            y_true: (B, S, 7) where last dim is [boxes(6), confidence(1)]
            y_pred: (B, S, 7) where last dim is [boxes(6), confidence(1)]
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        pad_val = tf.cast(self.pad_value, y_true.dtype)
        B = tf.shape(y_true)[0]
        S = self.max_sources

        # Split into boxes and confidence
        boxes_true, conf_true = self._split_outputs(y_true)
        boxes_pred, conf_pred = self._split_outputs(y_pred)

        # Compute valid masks (with tolerance)
        true_mask = tf.logical_not(tf.reduce_all(tf.abs(boxes_true - pad_val) < 1e-3, axis=-1))
        pred_mask = tf.logical_not(tf.reduce_all(tf.abs(boxes_pred - pad_val) < 1e-3, axis=-1))

        boxes_t_clip = boxes_true[:, :S, :]
        boxes_p_clip = boxes_pred[:, :S, :]

        # Compute pairwise costs for box matching
        if self.base in ("smooth_l1", "hybrid"): # If the base is smooth_l1 or hybrid, compute the smooth L1 cost
            cost_smooth = _pairwise_smooth_l1_batch(boxes_t_clip, boxes_p_clip, delta=self.delta)
        else:
            cost_smooth = tf.zeros((B, S, S), dtype=tf.float32)

        if self.base in ("ciou", "hybrid"): # If the base is ciou or hybrid, compute the CIoU cost
            cost_ciou = _pairwise_ciou3d_batch(boxes_t_clip, boxes_p_clip, pad_value=pad_val)
        else:
            cost_ciou = tf.zeros((B, S, S), dtype=tf.float32)

        cost_pure = self.smooth_weight * cost_smooth + self.ciou_weight * cost_ciou 
        
        # Include predicted confidence in the matching cost (Higher confidence = Lower cost)
        cost_scale = tf.stop_gradient(tf.reduce_mean(cost_pure) + 1.0)
        conf_penalty = tf.expand_dims(conf_pred, axis=1) # (B, 1, S)
        true_mask_float = tf.cast(tf.expand_dims(true_mask, axis=2), tf.float32) # (B, S, 1)
        match_cost = cost_pure - (cost_scale * self.confidence_weight * conf_penalty * true_mask_float)
        
        match_cost = self._mask_costs(match_cost, true_mask, pred_mask)

        # Check for valid pairs
        any_true = tf.reduce_any(true_mask, axis=1)
        any_pred = tf.reduce_any(pred_mask, axis=1)
        any_pair = tf.logical_and(any_true, any_pred)

        # Run bipartite matching; retrieve both indices and counts for downstream confidence target assignment
        # Box matching loss
        if self.matching_strategy == "greedy":
            box_loss_per_sample, match_indices, match_counts = self._greedy_tf_batch(cost_pure, match_cost)
        else:  # jv
            box_loss_per_sample, match_indices, match_counts = self._jv_tf_batch(cost_pure, match_cost)

        box_loss_per_sample = tf.where(any_pair, box_loss_per_sample, tf.zeros_like(box_loss_per_sample))

        # Confidence loss
        # Assign GT confidence targets to the prediction slots determined by the box matching
        matched_conf = self._match_conf_tf(conf_true, match_indices, match_counts)
        confidence_loss = self._confidence_loss(matched_conf, conf_pred)
        box_loss_mean = tf.reduce_mean(box_loss_per_sample)

        # Combine losses
        total_loss = tf.reduce_mean(box_loss_per_sample) + self.confidence_weight * confidence_loss
        
        total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, tf.constant(0.0, dtype=total_loss.dtype))
        return total_loss



# ============================================================================
# Model creation and training
# ============================================================================

class PLACENetCore:
    """Core model creation and training logic."""
    
    def __init__(self, train_labels: np.ndarray, pad_value: int):
        self.PAD_VALUE = pad_value
        self.train_labels = train_labels

    @staticmethod
    def add_confidence_to_labels(labels: np.ndarray, pad_value: int = -1) -> np.ndarray:
        """
        Add confidence dimension to labels for YOLO-style training.
        
        Args:
            labels: (N, max_sources, 6) array of boxes
            pad_value: Value used for padding empty slots
            
        Returns:
            labels_with_conf: (N, max_sources, 7) where last dim is [boxes(6), confidence(1)]
            Confidence is 1.0 for valid slots, 0.0 for empty slots
        """
        N, S, _ = labels.shape
        labels_with_conf = np.zeros((N, S, 7), dtype=labels.dtype)
        
        # Copy boxes
        labels_with_conf[:, :, :6] = labels
        
        # Add confidence: 1.0 for valid slots, 0.0 for empty slots
        for i in range(N):
            for j in range(S):
                is_valid = not np.all(np.isclose(labels[i, j], pad_value, atol=1e-3))
                labels_with_conf[i, j, 6] = 1.0 if is_valid else 0.0
        
        return labels_with_conf

    def make_CNN_model(
        self,
        train_labels: np.ndarray,
        config: PLACENetConfig,
        input_shape: Optional[Tuple[int, ...]] = None,
    ):
        """
        Create CNN model with YOLO-style output (boxes + confidence).
        
        Args:
            train_labels: Training labels (N, max_sources, 7) with confidence
            config: PLACENetConfig instance
            input_shape: Input shape tuple, inferred from data if None
        """
        # YOLO-style: output boxes + confidence
        if input_shape is None:
            input_shape = (16, 152, 1)  # Default shape
        
        input_layer = tf.keras.layers.Input(shape=input_shape)
        
        # Shared backbone
        x = tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg)
        )(input_layer)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg)
        )(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            128, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        shared_features = tf.keras.layers.Dense(
            64, activation="softplus",
            kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg)
        )(x)
        shared_features = tf.keras.layers.BatchNormalization()(shared_features)
        
        # Split into two heads
        # Box head: (max_sources, 6)
        box_output = tf.keras.layers.Dense(
            config.max_sources * 6, activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg)
        )(shared_features)
        box_output = tf.keras.layers.Reshape((config.max_sources, 6))(box_output)
        
        # Confidence head: (max_sources, 1) with sigmoid
        conf_output = tf.keras.layers.Dense(
            config.max_sources, activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(config.l2_reg)
        )(shared_features)
        conf_output = tf.keras.layers.Reshape((config.max_sources, 1))(conf_output)
        
        # Concatenate: (max_sources, 7)
        output = tf.keras.layers.Concatenate(axis=-1)([box_output, conf_output])
        
        model = tf.keras.Model(inputs=input_layer, outputs=output)

        # Create loss function
        loss_fn = YOLOStyleMatchingLoss(
            base=config.loss_type,
            pad_value=self.PAD_VALUE,
            max_sources=config.max_sources,
            matching_strategy=config.matching_strategy,
            delta=config.delta,
            smooth_weight=config.smooth_weight,
            ciou_weight=config.ciou_weight,
            confidence_weight=config.confidence_weight,
            noobj_weight=config.noobj_weight,
        )

        print(f"[make_CNN_model] Created YOLO-style CNN with matching={config.matching_strategy}, base={config.loss_type}")

        opt = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        model.compile(optimizer=opt, loss=loss_fn, metrics=[], jit_compile=False)
        return model

    @staticmethod
    def _build_augmenter(config: PLACENetConfig) -> Optional[PLACESpectrumAugmenter]:
        """Create an augmenter from config, or None when disabled."""
        if not config.enable_augmentation or config.augmentation_multiplier <= 0:
            return None
        kwargs = {
            "poisson_noise": True,
            "energy_shift": True,
            "intensity_scale": False,
            "detector_dropout": True,
            "energy_shift_range": 0.02,
            "intensity_scale_range": (0.8, 1.2),
            "detector_dropout_prob": 0.1,
            "poisson_scale": 500.0,
        }
        kwargs.update(config.augmenter_kwargs)
        return PLACESpectrumAugmenter(**kwargs)

    @staticmethod
    def _augment_train_fold(
        train_data: np.ndarray,
        train_labels: np.ndarray,
        config: PLACENetConfig,
        augmenter: Optional[PLACESpectrumAugmenter],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment train-fold only to avoid train/validation leakage."""
        if augmenter is None:
            return train_data, train_labels

        augmented_batches = [train_data]
        augmented_labels = [train_labels]
        original_size = len(train_data)
        for _ in range(config.augmentation_multiplier):
            augmented_batches.append(augmenter.augment_batch(train_data))
            augmented_labels.append(train_labels)

        train_data_aug = np.concatenate(augmented_batches, axis=0)
        train_labels_aug = np.concatenate(augmented_labels, axis=0)
        permutation = np.random.permutation(len(train_data_aug))
        train_data_aug = train_data_aug[permutation]
        train_labels_aug = train_labels_aug[permutation]

        print(
            f"Augmented train-fold size: {len(train_data_aug)} samples "
            f"(from {original_size}, multiplier {config.augmentation_multiplier + 1}x)"
        )
        return train_data_aug, train_labels_aug

    def do_kfold(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        config: PLACENetConfig,
        label: str,
    ) -> Tuple[List[Dict[str, float]], List[Any]]:
        """
        Perform k-fold cross-validation training.
        
        Args:
            data: Input data (N, H, W, C)
            labels: Labels (N, max_sources, 6) - will be converted to (N, max_sources, 7)
            config: PLACENetConfig instance
            label: Run label for file naming
            
        Returns:
            metrics_summary: List of metric dictionaries per fold
            history_list: List of training histories per fold
        """
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
        history_list = []
        metrics_summary = []
        file_label = config.run_name if config.run_name else label
        results_dir = Path(f"Results_{file_label}")
        results_dir.mkdir(parents=True, exist_ok=True)
        augmenter = self._build_augmenter(config)

        for kfold, (train, test) in enumerate(KFold(n_splits=config.folds, shuffle=True, random_state=42).split(data, labels)):
            tf.keras.backend.clear_session()
            gc.collect()
            try:
                tf.config.experimental.reset_memory_stats("GPU:0")
            except Exception:
                pass

            # Set random seeds AFTER clear_session() to ensure reproducible weight initialization
            np.random.seed(42) # Set the random seed to 42 to compare when different models are trained on the same data
            tf.random.set_seed(42) # Set the random seed to 42 to compare when different models are trained on the same data

            train_data_fold = data[train]
            train_labels_fold = labels[train]
            train_data_fold, train_labels_fold = self._augment_train_fold(
                train_data_fold,
                train_labels_fold,
                config,
                augmenter,
            )

            # Prepare labels: add confidence for YOLO-style
            train_labels_prep = self.add_confidence_to_labels(train_labels_fold, pad_value=self.PAD_VALUE)
            test_labels_prep = self.add_confidence_to_labels(labels[test], pad_value=self.PAD_VALUE)
            
            # Infer input shape from data
            input_shape = None
            if len(data) > 0:
                input_shape = data.shape[1:]  # (height, width, channels)
            
            # Create the model
            model = self.make_CNN_model(train_labels_prep, config, input_shape=input_shape)

            print(f"[{file_label}] Fold {kfold+1}/{config.folds}: train={data[train].shape[0]} samples, val={data[test].shape[0]} samples")

            # Save the test data and labels for later evaluation
            np.savez(results_dir / f"data_labels_test_{file_label}_kf{kfold}.npz", data=data[test], labels=test_labels_prep)

            data_test_2d = data[test].reshape(data[test].shape[0], -1)
            labels_test_2d = test_labels_prep.reshape(test_labels_prep.shape[0], -1)
            np.savetxt(results_dir / f"data_test_{file_label}_kf{kfold}.csv", data_test_2d, delimiter=" ", fmt="%.8g")
            np.savetxt(results_dir / f"labels_test_{file_label}_kf{kfold}.csv", labels_test_2d, delimiter=" ", fmt="%.8g")

            log_dir = results_dir / "logs" / f"{file_label}_kf{kfold}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=0)

            # Train the model and save the history
            history = model.fit(
                train_data_fold, train_labels_prep,
                epochs=config.epochs,
                batch_size=config.batch_size,
                callbacks=[callback, tensorboard_callback],
                validation_data=(data[test], test_labels_prep),
                verbose=0
            )

            # Save the metrics for this fold
            final_epoch = history.history
            fold_metrics = {metric: values[-1] for metric, values in final_epoch.items()}
            fold_metrics["fold"] = kfold
            metrics_summary.append(fold_metrics)
            history_list.append(history)

            # Save loss curve for this fold
            try:
                from .PLACENet_evaluate import PLACENetPlot
                PLACENetPlot.save_loss_curves(
                    [history],
                    results_dir / f"loss_curves_{file_label}_kf{kfold}.pdf"
                )
            except Exception as e:
                print(f"Warning: Could not save loss curve for fold {kfold}: {e}")

            model.save(results_dir / f"model_{file_label}_kf{kfold}.keras")

            del model, history
            gc.collect()
            tf.keras.backend.clear_session()
            try:
                tf.config.experimental.reset_memory_stats("GPU:0")
            except Exception:
                pass

        df = pd.DataFrame(metrics_summary) # Save the metrics to a CSV file
        df.to_csv(results_dir / f"metrics_{file_label}.csv", sep=" ", index=False)

        with open(results_dir / f"metrics_{file_label}.json", "w") as f: # Save the metrics to a JSON file
            json.dump(metrics_summary, f, indent=2)

        with open(results_dir / f"metrics_summary_{file_label}.txt", "w") as f:
            f.write(f"Metrics Summary for {file_label}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Summary Statistics:\n")
            f.write(df.describe().to_string())
            f.write("\n\nRaw Data:\n")
            f.write(df.to_string())

        return metrics_summary, history_list


# ============================================================================
# Main PLACENet class
# ============================================================================

class PLACENet:
    """Simplified PLACENet model with CNN architecture and YOLO-style output."""
    
    def __init__(self, config: PLACENetConfig):
        self.config = config

    def train_concatenated(
        self,
        datasets: Sequence["PLACEDataset"],
        run_label: Optional[str] = None
    ) -> PLACENetTrainingResult:
        """Train on concatenated datasets."""
        data, labels = self._concatenate(datasets)
        label = run_label or self.config.run_name or "run"
        metrics, histories = self._run_training(data, labels, label, run_name=self.config.run_name or label)
        return PLACENetTrainingResult(label, metrics, histories)

    def train_per_dataset(
        self,
        datasets: Sequence["PLACEDataset"],
        run_labels: Optional[Sequence[str]] = None
    ) -> List[PLACENetTrainingResult]:
        """Train on each dataset separately."""
        results: List[PLACENetTrainingResult] = []
        for idx, dataset in enumerate(datasets):
            label = (run_labels[idx % len(run_labels)] if run_labels else dataset.name)
            data, labels = self._align(dataset.data, dataset.labels)
            metrics, histories = self._run_training(data, labels, label, run_name=self.config.run_name or label)
            results.append(PLACENetTrainingResult(
                run_label=label,
                metrics=metrics,
                histories=histories,
                dataset_name=dataset.name
            ))
        return results

    def _run_training(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        run_label: str,
        run_name: Optional[str] = None,
    ) -> Tuple[List[Dict[str, float]], List[Any]]:
        """Run k-fold training."""
        runner = PLACENetCore(labels, self.config.pad_value)
        return runner.do_kfold(data, labels, self.config, run_label)

    def _concatenate(
        self,
        datasets: Sequence["PLACEDataset"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate multiple datasets."""
        aligned = [self._align(ds.data, ds.labels) for ds in datasets]
        data_parts = [item[0] for item in aligned if len(item[0])]
        label_parts = [item[1] for item in aligned if len(item[1])]
        if not data_parts or not label_parts:
            raise ValueError("No datasets available for concatenated training")
        return np.concatenate(data_parts), np.concatenate(label_parts)

    @staticmethod
    def _align(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align data and labels to same length."""
        if len(data) == len(labels):
            return data, labels
        length = min(len(data), len(labels))
        return data[:length], labels[:length]

