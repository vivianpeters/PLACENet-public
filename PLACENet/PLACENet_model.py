# PLACENet - Simplified CNN-based model with bounding box output and confidence output
# Supports smooth_l1, ciou, and hybrid loss types
# Supports greedy or JV (Jonker-Volgenant) matching

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
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

    def masked_iou_loss(self, y_true, y_pred):
        """Compute masked IoU *loss* (1 - IoU).
        Input format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # Extract only box dimensions (first 6) if using YOLO-style (7 dims)
        y_true_boxes = y_true[..., :6]
        y_pred_boxes = y_pred[..., :6]
        mask = tf.not_equal(y_true_boxes, self.PAD_VALUE)
        mask = tf.cast(mask, dtype=tf.float32)
        y_true_masked = y_true_boxes * mask
        y_pred_masked = y_pred_boxes * mask
        x_width_true, x_centre_true, y_width_true, y_centre_true, z_width_true, z_centre_true = tf.unstack(y_true_masked, axis=-1)
        x_width_pred, x_centre_pred, y_width_pred, y_centre_pred, z_width_pred, z_centre_pred = tf.unstack(y_pred_masked, axis=-1)
        # Convert from centre-based to min/max for IoU calculation
        x_min_true = x_centre_true - x_width_true / 2.0
        x_max_true = x_centre_true + x_width_true / 2.0
        y_min_true = y_centre_true - y_width_true / 2.0
        y_max_true = y_centre_true + y_width_true / 2.0
        z_min_true = z_centre_true - z_width_true / 2.0
        z_max_true = z_centre_true + z_width_true / 2.0
        x_min_pred = x_centre_pred - x_width_pred / 2.0
        x_max_pred = x_centre_pred + x_width_pred / 2.0
        y_min_pred = y_centre_pred - y_width_pred / 2.0
        y_max_pred = y_centre_pred + y_width_pred / 2.0
        z_min_pred = z_centre_pred - z_width_pred / 2.0
        z_max_pred = z_centre_pred + z_width_pred / 2.0
        x_intersect_min = tf.maximum(x_min_true, x_min_pred)
        x_intersect_max = tf.minimum(x_max_true, x_max_pred)
        y_intersect_min = tf.maximum(y_min_true, y_min_pred)
        y_intersect_max = tf.minimum(y_max_true, y_max_pred)
        z_intersect_min = tf.maximum(z_min_true, z_min_pred)
        z_intersect_max = tf.minimum(z_max_true, z_max_pred)
        intersection = tf.maximum(x_intersect_max - x_intersect_min, 0) * tf.maximum(y_intersect_max - y_intersect_min, 0) * tf.maximum(z_intersect_max - z_intersect_min, 0)
        true_volume = tf.abs(x_width_true) * tf.abs(y_width_true) * tf.abs(z_width_true)
        pred_volume = tf.abs(x_width_pred) * tf.abs(y_width_pred) * tf.abs(z_width_pred)
        union = true_volume + pred_volume - intersection + 1e-7
        iou = intersection / union
        
        # Only average over valid (non-padded) boxes
        # mask has shape (..., max_sources, 6), check if all 6 dims are non-pad
        valid_box_mask = tf.reduce_all(mask > 0.5, axis=-1)  # Shape: (..., max_sources)
        valid_box_mask = tf.cast(valid_box_mask, dtype=tf.float32)
        
        # Compute mean IoU only over valid boxes
        iou_masked = iou * valid_box_mask
        sum_iou = tf.reduce_sum(iou_masked)
        count_valid = tf.reduce_sum(valid_box_mask)
        mean_iou = tf.cond(
            count_valid > 0,
            lambda: sum_iou / count_valid,
            lambda: 0.0
        )
        
        return 1.0 - mean_iou


def _pairwise_ciou3d_batch(y_true, y_pred, pad_value=-1, eps=_EPS):
    """Compute pairwise CIoU3D *costs* for batch.
    Input format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
    Padded boxes (all values = pad_value) are masked out to avoid invalid CIoU computation.
    """
    pad_val = tf.cast(pad_value, y_true.dtype)
    
    # Check if boxes are padded (all 6 dimensions equal pad_value)
    true_is_padded = tf.reduce_all(tf.equal(y_true, pad_val), axis=-1)  # (B, S)
    pred_is_padded = tf.reduce_all(tf.equal(y_pred, pad_val), axis=-1)  # (B, S)
    
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
        **kwargs,
    ):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, **kwargs)
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

    def _jv_tf_batch(self, cost_batch):
        """
        Batch-wise JV matching that maintains gradient flow.
        Uses numpy for matching but gathers from TensorFlow tensor for gradients.
        """
        B = tf.shape(cost_batch)[0]
        S = self.max_sources
        
        # Get indices from numpy JV (non-differentiable)
        indices_np, counts_np = tf.numpy_function(
            self._jv_numpy_batch_indices,
            [cost_batch],
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
        
        return mean_costs
    
    def _greedy_tf_batch(self, cost_batch, n_true, n_pred):
        """Batch-wise greedy matching algorithm."""
        B = tf.shape(cost_batch)[0]
        S = self.max_sources
        inf = tf.constant(1e10, dtype=cost_batch.dtype)
        cost = tf.identity(cost_batch)
        total = tf.zeros((B,), dtype=cost.dtype)
        counts = tf.zeros((B,), dtype=tf.int32)

        def cond(step, cost, total, counts):
            return tf.less(step, S)

        def body(step, cost, total, counts):
            flat = tf.reshape(cost, (B, -1))
            argmin = tf.argmin(flat, axis=1, output_type=tf.int32)
            i = argmin // S
            j = argmin % S
            idx = tf.stack([tf.range(B, dtype=tf.int32), i, j], axis=1)
            chosen = tf.gather_nd(cost, idx)
            chosen = tf.where(tf.math.is_finite(chosen) & (chosen < 1e9), chosen, 0.0)
            total = total + chosen
            counts = counts + tf.where(tf.math.is_finite(chosen) & (chosen < 1e9), 1, 0)
            
            row_mask = tf.tensor_scatter_nd_update(
                tf.zeros((B, S), dtype=cost.dtype),
                tf.stack([tf.range(B, dtype=tf.int32), i], axis=1),
                tf.ones((B,), dtype=cost.dtype) * inf
            )
            col_mask = tf.tensor_scatter_nd_update(
                tf.zeros((B, S), dtype=cost.dtype),
                tf.stack([tf.range(B, dtype=tf.int32), j], axis=1),
                tf.ones((B,), dtype=cost.dtype) * inf
            )
            cost = cost + tf.expand_dims(row_mask, 2) + tf.expand_dims(col_mask, 1)
            return step + 1, cost, total, counts

        step0 = tf.constant(0, dtype=tf.int32)
        _, cost_fin, total_fin, counts_fin = tf.while_loop(
            cond, body, (step0, cost, total, counts), maximum_iterations=S
        )
        counts_fin = tf.cast(counts_fin, cost_fin.dtype)
        mean_vals = tf.where(counts_fin > 0, total_fin / counts_fin, tf.zeros_like(total_fin))
        return mean_vals

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

        # Compute valid masks
        true_mask = tf.logical_not(tf.reduce_all(tf.equal(boxes_true, pad_val), axis=-1))
        pred_mask = tf.logical_not(tf.reduce_all(tf.equal(boxes_pred, pad_val), axis=-1))

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

        cost = self.smooth_weight * cost_smooth + self.ciou_weight * cost_ciou # Combine the smooth L1 and CIoU costs
        cost = self._mask_costs(cost, true_mask, pred_mask) # Mask the costs for the valid pairs

        # Check for valid pairs
        any_true = tf.reduce_any(true_mask, axis=1)
        any_pred = tf.reduce_any(pred_mask, axis=1)
        any_pair = tf.logical_and(any_true, any_pred)

        # Box matching loss
        if self.matching_strategy == "greedy":
            box_loss_per_sample = self._greedy_tf_batch(
                cost,
                tf.reduce_sum(tf.cast(true_mask, tf.int32), axis=1),
                tf.reduce_sum(tf.cast(pred_mask, tf.int32), axis=1)
            )
        else:  # jv
            box_loss_per_sample = self._jv_tf_batch(cost)

        box_loss_per_sample = tf.where(any_pair, box_loss_per_sample, tf.zeros_like(box_loss_per_sample))

        # Confidence loss
        confidence_loss = self._confidence_loss(conf_true, conf_pred)

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
                is_valid = not np.all(labels[i, j] == pad_value)
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
        callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30)
        history_list = []
        metrics_summary = []
        file_label = config.run_name if config.run_name else label
        results_dir = Path(f"Results_{file_label}")
        results_dir.mkdir(parents=True, exist_ok=True)

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

            # Prepare labels: add confidence for YOLO-style
            train_labels_prep = self.add_confidence_to_labels(labels[train], pad_value=self.PAD_VALUE)
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
                data[train], train_labels_prep,
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
                from .PLACENet_plot import PLACENetPlot
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
        self._augmenter = self._build_augmenter()
    
    def _build_augmenter(self) -> Optional[PLACESpectrumAugmenter]:
        if not self.config.enable_augmentation or self.config.augmentation_multiplier <= 0:
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
        kwargs.update(self.config.augmenter_kwargs)
        return PLACESpectrumAugmenter(**kwargs)
    
    def _maybe_augment(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Augment the data and labels if the augmenter is not None."""
        if self._augmenter is None: # If the augmenter is None, return the data and labels
            return data, labels
        augmented_batches = [data]
        augmented_labels = [labels]
        original_size = len(data)
        for _ in range(self.config.augmentation_multiplier):
            augmented_batches.append(self._augmenter.augment_batch(data))
            augmented_labels.append(labels)
        data = np.concatenate(augmented_batches, axis=0)
        labels = np.concatenate(augmented_labels, axis=0)
        permutation = np.random.permutation(len(data))
        data = data[permutation]
        labels = labels[permutation]
        print(f"Augmented dataset size: {len(data)} samples (from {original_size}, multiplier {self.config.augmentation_multiplier + 1}x)")
        return data, labels

    def train_concatenated(
        self,
        datasets: Sequence["PLACEDataset"],
        run_label: Optional[str] = None
    ) -> PLACENetTrainingResult:
        """Train on concatenated datasets."""
        data, labels = self._concatenate(datasets)
        data, labels = self._maybe_augment(data, labels)
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
            data, labels = self._maybe_augment(data, labels)
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

