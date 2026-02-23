"""Data preparation for PLACENet."""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass(frozen=True) # immutable class
class PLACEPrepConfig:
    """Configuration for loading and preparing datasets."""

    data_dir: str  # Path to the data directory
    dets: int = 16  # Number of detectors
    chunk_size: int = 16  # Number of spectra to group together
    pad_value: int = -1  # Value to pad the labels with
    max_sources: int = 1  # Maximum number of sources to classify
    cluster_eps: float = 2.0  # DBSCAN epsilon parameter (voxel distance for clustering)
    randomize_slot_assignment: bool = False  # If True, randomly assign sources to slots instead of sequential


@dataclass(frozen=True) # immutable class
class PLACEFilePair:
    """Container describing the paired files needed for one dataset."""

    setname: str  # Name of the dataset
    energy_path: Path  # Path to the energy data file
    position_path: Path  # Path to the position data file


@dataclass
class PLACEDataset:
    """Structured representation of a prepared dataset."""

    name: str
    data: np.ndarray  # Array of spectra
    labels: np.ndarray  # Array of labels
    positions: List[np.ndarray]  # List of position arrays
    pos_keys: List[str]  # List of filename keys (matching the order of data samples)

    def summary(self) -> str:
        return (
            f"{self.name}: samples={len(self.data)}, "
            f"spectrum_shape={self.data.shape[1:]}, "
            f"label_shape={self.labels.shape[1:]}"
        )


class PLACENetPrep:
    """Data preparation pipeline for datasets."""

    def __init__(self, config: PLACEPrepConfig):
        self.config = config
        self._pairs = self._discover_pairs()

    def _discover_pairs(self) -> Dict[str, PLACEFilePair]:
        """
        Discover the paired files needed for each dataset by searching for _gamma in the file name.
        Hardcoded for the gamma datasets for now.
        Args:
            None
        Returns:
            Dict[str, PLACEFilePair]: A dictionary of dataset names and their paired files.
        """
        root = Path(self.config.data_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Data directory not found: {root}")

        buckets: Dict[str, Dict[str, Path]] = {} # Dictionary to store the paired files for each dataset
        for file in root.iterdir():
            if not file.is_file():
                continue

            name_lower = file.name.lower()
            gamma_idx = name_lower.find("_gamma")
            if gamma_idx == -1:
                continue

            setname = file.name[:gamma_idx]
            slot = buckets.setdefault(setname, {})
            if "detector_energy_data" in name_lower:
                slot["energy"] = file
            elif "position_arrays" in name_lower:
                slot["pos"] = file

        pairs = {
            setname: PLACEFilePair(setname, paths["energy"], paths["pos"])
            for setname, paths in buckets.items()
            if "energy" in paths and "pos" in paths
        }

        if not pairs:
            raise RuntimeError(f"No valid dataset pairs found in {root}")

        return pairs

    def list_sets(self) -> List[str]:
        """Return the discovered dataset identifiers (setnames)."""
        return sorted(self._pairs.keys())

    def load_datasets(self) -> List[PLACEDataset]:
        """Load, normalize, and label every discovered dataset."""
        datasets: List[PLACEDataset] = []

        for setname in self.list_sets():
            pair = self._pairs[setname]
            matrices = self._load_csv_as_matrices(str(pair.energy_path))
            position_filenames, positions = self._load_csv_as_pos(str(pair.position_path))
            
            # Diagnostic: Check file sizes for huge files
            pos_file_size_mb = pair.position_path.stat().st_size / (1024 * 1024)
            energy_file_size_mb = pair.energy_path.stat().st_size / (1024 * 1024)
            total_voxels = sum(len(pos) for pos in positions) if positions else 0
            if pos_file_size_mb > 100 or len(positions) > 1000:
                print(f"[INFO] Dataset '{setname}': Position file size: {pos_file_size_mb:.1f} MB, "
                      f"Energy file: {energy_file_size_mb:.1f} MB. "
                      f"Position entries (histograms): {len(positions)}, "
                      f"Total voxels: {total_voxels:,}")
            
            spectra, pos_keys = self._prep_data(matrices, self.config.chunk_size)
            fixed_spectra = self._fix_spectra_shapes_with_merge(spectra)
            data = self._reshape_spectra(fixed_spectra)
            
            # Align positions with pos_keys (filenames) before building labels
            positions_by_filename = {fname: pos for fname, pos in zip(position_filenames, positions)}
            
            aligned_positions = []
            for filename in pos_keys:
                if filename in positions_by_filename:
                    aligned_positions.append(positions_by_filename[filename])
                else:
                    warnings.warn(
                        f"Dataset '{setname}': Filename '{filename}' found in energy file but not in position file. "
                        f"Using empty position array for this entry.",
                        UserWarning
                    )
                    aligned_positions.append(np.empty((0, 4), dtype=float))
            
            if len(aligned_positions) != len(pos_keys):
                raise ValueError(
                    f"Dataset '{setname}': Alignment failed. Expected {len(pos_keys)} positions, "
                    f"got {len(aligned_positions)}."
                )
            
            labels = self._build_labels(setname, aligned_positions)

            # Verify alignment
            if len(data) != len(pos_keys):
                raise ValueError(
                    f"Dataset '{setname}': data length ({len(data)}) != pos_keys length ({len(pos_keys)}) "
                    f"after merging. This should not happen."
                )
            
            target_len = len(data)
            if len(labels) != target_len:
                if len(labels) < target_len:
                    raise ValueError(
                        f"Dataset '{setname}': labels length ({len(labels)}) < data length ({target_len}). "
                        f"Not enough labels for all data samples. This may indicate an issue with position alignment."
                    )
                else:
                    warnings.warn(
                        f"Dataset '{setname}': labels length ({len(labels)}) > data length ({target_len}). "
                        f"Truncating labels to match data length. This may indicate a bug in the alignment logic.",
                        UserWarning
                    )
                    labels = labels[:target_len]

            aligned_len = len(data)
            datasets.append(
                PLACEDataset(
                    name=setname,
                    data=data,
                    labels=labels,
                    positions=list(aligned_positions)[:aligned_len],
                    pos_keys=list(pos_keys)[:aligned_len],
                )
            )

        return datasets

    def concatenate(self, datasets: Sequence[PLACEDataset]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Concatenate multiple datasets, keeping only aligned samples.
        Handles overlapping spectra by merging them.
        """
        aligned_data = []
        aligned_labels = []

        for dataset in datasets:
            if len(dataset.data) != len(dataset.labels):
                min_len = min(len(dataset.data), len(dataset.labels))
                aligned_data.append(dataset.data[:min_len])
                aligned_labels.append(dataset.labels[:min_len])
            else:
                aligned_data.append(dataset.data)
                aligned_labels.append(dataset.labels)

        if not aligned_data:
            raise ValueError("No datasets available for concatenation")

        return np.concatenate(aligned_data), np.concatenate(aligned_labels)

    def _reshape_spectra(self, spectra: Sequence[np.ndarray]) -> np.ndarray:
        """
        Reshape the spectra to the correct shape.
        If empty, return empty array.
        Ensures the number of detectors is correct and adds a channel dimension.
        """
        if not spectra:
            return np.empty((0, self.config.dets, 0, 1), dtype=np.float32)

        array = np.asarray(spectra, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(f"Expected 3D spectra, received shape {array.shape}")

        n_detectors = array.shape[1]
        if n_detectors != self.config.dets:
            raise ValueError(
                f"Detector count mismatch (expected {self.config.dets}, got {n_detectors})"
            )

        return array.reshape(array.shape[0], array.shape[1], array.shape[2], 1)

    def _build_labels(self, setname: str, positions: List[np.ndarray]) -> np.ndarray:
        """
        Build the labels for the dataset.
        Pads and reshapes label arrays for training, removing count values.
        Calls the _make_simple_labels function to create the labels.
        Args:
            setname: The name of the dataset.
            positions: The positions of the sources in the dataset.
        Returns:
            np.ndarray: The labels for the dataset.
        """
        raw_labels = self._make_simple_labels(
            [positions],
            [setname],
            max_sources=self.config.max_sources,
            cluster_sources=True,
            eps=self.config.cluster_eps,
            pad_value=self.config.pad_value,
            randomize_slots=self.config.randomize_slot_assignment,
        )
        labels = np.asarray(raw_labels[0], dtype=np.int32)
        if labels.ndim == 2:
            labels = labels[:, np.newaxis, :]
        return labels

    @staticmethod
    def _load_csv_as_matrices(file_path: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Load energy data from CSV file.
        
        Returns list of tuples: (filename, source_position, spectrum)
        where source_position is [SrcVoxelX, SrcVoxelY, SrcVoxelZ]
        """
        matrices = []
        with open(file_path, "r") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # Skip header
            for row in reader:
                if len(row) < 9:
                    continue
                try:
                    filename = row[0].strip()
                    position = np.array([float(x) for x in row[1:4]])
                    spectrum = np.array([float(x) for x in row[8:]])
                    matrices.append((filename, position, spectrum))
                except (ValueError, IndexError):
                    continue
        return matrices

    @staticmethod
    def _load_csv_as_pos(file_path: str) -> Tuple[List[str], List[np.ndarray]]:
        """Load the position data from the CSV file (voxel coordinates).
        Returns a tuple of lists: (filenames, position arrays).
        """
        grouped_positions: Dict[str, List[List[float]]] = {}
        # Group the positions by filename (histogram id)
        with open(file_path, "r") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                current_hist = row[0].strip()
                if len(row) < 5:
                    continue
                try:
                    x = float(row[1])
                    y = float(row[2])
                    z = float(row[3])
                    count = float(row[4])
                except ValueError:
                    continue
                grouped_positions.setdefault(current_hist, []).append([x, y, z, count])
        # Turn it into list
        filenames = []
        pos_list = []
        for key in sorted(grouped_positions.keys()):
            entries = grouped_positions[key]
            if entries:
                filenames.append(key) # Append the filename to the list
                pos_list.append(np.array(entries, dtype=float)) # Append the position array to the list
        return filenames, pos_list

    def _prep_data(
        self,
        matrices: Sequence[Tuple[str, np.ndarray, np.ndarray]],
        chunk_size: int,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Group and normalize the spectra by filename, and chunk them into groups of size `chunk_size`.
        Args:
            matrices: The matrices to process.
            chunk_size: The size of the chunks to create.
        Returns:
            Tuple[List[np.ndarray], List[str]]: The spectra and filenames.
        """
        filename_spectra: Dict[str, List[np.ndarray]] = {}
        current_filename: Optional[str] = None
        spectrum_group: List[np.ndarray] = []

        for filename, position, spectrum in matrices:
            if filename != current_filename:
                if current_filename is not None:
                    self._commit_spectrum_group(
                        filename_spectra, current_filename, spectrum_group, chunk_size
                    )
                current_filename = filename
                spectrum_group = [spectrum]
            else:
                spectrum_group.append(spectrum)

        if current_filename is not None:
            self._commit_spectrum_group(
                filename_spectra, current_filename, spectrum_group, chunk_size
            )

        for key in filename_spectra:
            arr = np.array(filename_spectra[key])
            arr = arr / np.max(arr)
            filename_spectra[key] = arr
        # Sort the filenames for consistent ordering (matching position file ordering)
        spectrum_list: List[np.ndarray] = []
        filename_list: List[str] = []
        for key in sorted(filename_spectra.keys()):
            spectrum_list.append(filename_spectra[key]) # Append the spectrum to the list
            filename_list.append(key) # Append the filename to the list

        return spectrum_list, filename_list

    @staticmethod
    def _commit_spectrum_group(
        container: Dict[str, List[np.ndarray]],
        key: str,
        spectra: Sequence[np.ndarray],
        chunk_size: int,
    ) -> None:
        """Commit the spectrum group to the dictionary.
        Args:
            container: The dictionary to store the spectra.
            key: The key to store the spectra.
            spectra: The spectra to store.
            chunk_size: The size of the chunks to create.
        Returns:
            None
        """
        if key not in container:
            container[key] = []
        for i in range(0, len(spectra), chunk_size):
            chunk = np.vstack(spectra[i : i + chunk_size])
            container[key].append(chunk)

    def _fix_spectra_shapes_with_merge(
        self, spectra_list: Sequence[np.ndarray]
    ) -> List[np.ndarray]:
        """Fix the shapes of the spectra (if multisource positions overlap)."""
        fixed: List[np.ndarray] = []
        target = self.config.dets
        for spec_array in spectra_list:
            if len(spec_array.shape) == 3:
                n_chunks, dets, bins = spec_array.shape
                flattened = spec_array.reshape(n_chunks * dets, bins)
                # If the number of spectra is greater than or equal to the target size, merge them
                if len(flattened) >= target:
                    if len(flattened) == target:
                        fixed_chunk = flattened
                    else:
                        fixed_chunk = self._merge_spectra(flattened, target)
                    fixed.append(fixed_chunk) # Append the fixed chunk to the list
                else:
                    padded = np.zeros((target, bins))
                    padded[: len(flattened)] = flattened
                    fixed.append(padded) # Append the padded array to the list
            else:
                fixed.append(spec_array) # Append the spectrum array to the list

        return fixed

    @staticmethod
    def _merge_spectra(spectra_array: np.ndarray, target_size: int) -> np.ndarray:
        """Merge the spectra into the target size if multisource positions overlap."""
        n_spectra, n_bins = spectra_array.shape
        spectra_per_group = n_spectra / target_size

        merged = []
        for i in range(target_size):
            start_idx = int(i * spectra_per_group)
            end_idx = int((i + 1) * spectra_per_group)
            if i == target_size - 1:
                end_idx = n_spectra
            group = spectra_array[start_idx:end_idx]
            merged.append(np.mean(group, axis=0) if len(group) > 0 else np.zeros(n_bins))
        return np.array(merged)

    def _make_simple_labels(
        self,
        position_list: List[List[np.ndarray]],
        setnames: List[str],
        max_sources: int,
        cluster_sources: bool,
        eps: float,
        pad_value: int,
        randomize_slots: bool = False,
    ) -> List[np.ndarray]:
        """
        Create the bounding box labels for the dataset.
        Original data is a set of voxels.
        Get centre and width of the bounding box encompassing the voxels.
        Args:
            position_list: The list of position arrays.
            setnames: The list of setnames.
            max_sources: The maximum number of sources to classify.
            cluster_sources: If True, cluster the voxels into sources.
            eps: The DBSCAN epsilon parameter (voxel distance for clustering).
            pad_value: The value to pad the labels with.
            randomize_slots: If True, randomly assign sources to slots.
        Returns:
            List[np.ndarray]: The labels for the dataset.
        """
        labels: List[np.ndarray] = []
        for i in range(len(setnames)):
            dataset_labels: List[np.ndarray] = []
            for pos in position_list[i]:
                if cluster_sources:
                    sources = self._cluster_voxels_into_sources(pos, eps=eps)
                    source_bboxes = np.full((max_sources, 6), pad_value)
                    
                    n_sources = min(len(sources), max_sources)
                    if n_sources == 0:
                        dataset_labels.append(source_bboxes)
                        continue
                    
                    # Create bounding boxes for each source
                    # Format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
                    # Box extends from centre ± width/2
                    bbox_list = []
                    for src_voxels in sources[:max_sources]:
                        xmin, xmax = np.min(src_voxels[:, 0]), np.max(src_voxels[:, 0])
                        ymin, ymax = np.min(src_voxels[:, 1]), np.max(src_voxels[:, 1])
                        zmin, zmax = np.min(src_voxels[:, 2]), np.max(src_voxels[:, 2])
                        xwidth = xmax - xmin + 1
                        ywidth = ymax - ymin + 1
                        zwidth = zmax - zmin + 1
                        xcentre = (xmin + xmax) / 2.0
                        ycentre = (ymin + ymax) / 2.0
                        zcentre = (zmin + zmax) / 2.0
                        bbox_list.append([
                            xwidth, xcentre,
                            ywidth, ycentre,
                            zwidth, zcentre,
                        ])
                    
                    # Assign sources to slots
                    if randomize_slots:
                        available_slots = np.arange(max_sources)
                        selected_slots = np.random.choice(available_slots, size=n_sources, replace=False)
                        np.random.shuffle(selected_slots)
                        for bbox, slot_idx in zip(bbox_list, selected_slots):
                            source_bboxes[slot_idx] = bbox
                    else:
                        # Sequential assignment
                        for src_idx, bbox in enumerate(bbox_list):
                            source_bboxes[src_idx] = bbox
                    
                    dataset_labels.append(source_bboxes)
                else:
                    # Original behavior: single bounding box for all voxels
                    # Format: [xwidth, xcentre, ywidth, ycentre, zwidth, zcentre]
                    # Box extends from centre ± width/2
                    xmin, xmax = np.min(pos[:, 0]), np.max(pos[:, 0])
                    ymin, ymax = np.min(pos[:, 1]), np.max(pos[:, 1])
                    zmin, zmax = np.min(pos[:, 2]), np.max(pos[:, 2])
                    xwidth = xmax - xmin + 1
                    ywidth = ymax - ymin + 1
                    zwidth = zmax - zmin + 1
                    xcentre = (xmin + xmax) / 2.0
                    ycentre = (ymin + ymax) / 2.0
                    zcentre = (zmin + zmax) / 2.0
                    bbox = [
                        xwidth, xcentre,
                        ywidth, ycentre,
                        zwidth, zcentre,
                    ]
                    dataset_labels.append(np.array(bbox))
            labels.append(np.array(dataset_labels))
        return labels

    @staticmethod
    def _cluster_voxels_into_sources(
        pos: np.ndarray, eps: float = 3.0, min_samples: int = 1
    ) -> List[np.ndarray]:
        """Cluster the voxels into sources using DBSCAN.
        Args:
            pos: The position array.
            eps: The DBSCAN epsilon parameter (voxel distance for clustering).
            min_samples: The minimum number of samples in a neighborhood to form a cluster.
        Returns:
            List[np.ndarray]: The sources.
        """
        if len(pos) == 0:
            return []

        coords = pos[:, :3]
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = clustering.labels_

        sources: List[np.ndarray] = []
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        for label in sorted(unique_labels):
            mask = labels == label
            sources.append(pos[mask])

        return sources

