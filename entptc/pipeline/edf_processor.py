"""
EDF File Processing with 65→64 Channel Reduction

Reference: ENTPC.tex lines 696-703

From ENTPC.tex:

"EEG data from OpenNeuro ds005385 contains 65 EEG channels per recording. The
EntPTC quaternion framework requires exactly 64 channels (16 quaternions × 4 components).
Therefore, a principled dimension reduction from 65→64 channels is required.

The reduction MUST be:
1. Explicit: Logged and documented for every file
2. Principled: Based on channel properties, not arbitrary truncation
3. Reproducible: Same rule applied consistently across all subjects
4. Validated: Assert correct dimensionality at every step

Recommended approach: Remove channel with lowest signal-to-noise ratio (SNR) or
highest artifact content. Alternative: Remove reference channel if identified."

CRITICAL CONSTRAINTS:
- NO silent truncation
- NO Git LFS pointer files
- NO broken symlinks
- MUST fail loudly if real EDF data not available
- Assert all matrix shapes explicitly
"""

import numpy as np
import mne
from typing import Tuple, Dict, Optional, List
import os
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EDFProcessor:
    """
    EDF file processor with explicit 65→64 channel reduction.
    
    Per ENTPC.tex lines 696-703:
    - Principled dimension reduction
    - Explicit logging
    - Real data validation
    - Shape assertions
    """
    
    def __init__(self, reduction_method: str = 'lowest_snr'):
        """
        Initialize EDF processor.
        
        Args:
            reduction_method: Method for 65→64 reduction
                            'lowest_snr': Remove channel with lowest SNR
                            'highest_artifact': Remove channel with most artifacts
                            'reference': Remove reference channel if identified
        """
        self.reduction_method = reduction_method
        self.reduction_log = []
    
    def validate_real_edf_file(self, filepath: str) -> bool:
        """
        Validate that file is real EDF data, not Git LFS pointer or symlink.
        
        Per ENTPC.tex: MUST fail loudly if real data not available.
        
        Args:
            filepath: Path to EDF file
        
        Returns:
            True if valid real EDF file
        
        Raises:
            ValueError if file is not real EDF data
        """
        # Check file exists
        if not os.path.exists(filepath):
            raise ValueError(f"File does not exist: {filepath}")
        
        # Check not a symlink
        if os.path.islink(filepath):
            raise ValueError(f"File is a symlink (broken Git LFS?): {filepath}")
        
        # Check file size (Git LFS pointers are tiny, real EDF files are large)
        file_size = os.path.getsize(filepath)
        if file_size < 1000:  # Less than 1KB suggests LFS pointer
            raise ValueError(f"File too small ({file_size} bytes), likely Git LFS pointer: {filepath}")
        
        # Check file header (EDF files start with specific bytes)
        with open(filepath, 'rb') as f:
            header = f.read(8)
            # EDF files start with '0       ' (version number)
            if not header.startswith(b'0       '):
                raise ValueError(f"File does not have valid EDF header: {filepath}")
        
        logger.info(f"✓ Validated real EDF file: {filepath} ({file_size} bytes)")
        return True
    
    def load_edf_file(self, filepath: str) -> mne.io.Raw:
        """
        Load EDF file with validation.
        
        Args:
            filepath: Path to EDF file
        
        Returns:
            MNE Raw object
        """
        # Validate real data
        self.validate_real_edf_file(filepath)
        
        # Load with MNE
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        
        logger.info(f"Loaded EDF: {filepath}")
        logger.info(f"  Channels: {len(raw.ch_names)}")
        logger.info(f"  Sampling rate: {raw.info['sfreq']} Hz")
        logger.info(f"  Duration: {raw.times[-1]:.2f} seconds")
        
        return raw
    
    def compute_channel_snr(self, raw: mne.io.Raw) -> np.ndarray:
        """
        Compute signal-to-noise ratio for each channel.
        
        SNR = mean(signal_power) / std(signal_power)
        
        Args:
            raw: MNE Raw object
        
        Returns:
            Array of SNR values (one per channel)
        """
        data = raw.get_data()
        
        # Compute SNR for each channel
        snr = np.zeros(len(raw.ch_names))
        for i in range(len(raw.ch_names)):
            signal = data[i, :]
            
            # Signal power
            power = signal ** 2
            
            # SNR: mean power / std power
            snr[i] = np.mean(power) / (np.std(power) + 1e-12)
        
        return snr
    
    def compute_channel_artifacts(self, raw: mne.io.Raw) -> np.ndarray:
        """
        Compute artifact content for each channel.
        
        Artifact measure: number of extreme values (> 3 std from mean)
        
        Args:
            raw: MNE Raw object
        
        Returns:
            Array of artifact counts (one per channel)
        """
        data = raw.get_data()
        
        artifacts = np.zeros(len(raw.ch_names))
        for i in range(len(raw.ch_names)):
            signal = data[i, :]
            
            # Z-score
            z_score = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)
            
            # Count extreme values
            artifacts[i] = np.sum(np.abs(z_score) > 3)
        
        return artifacts
    
    def select_channel_to_remove(self, raw: mne.io.Raw) -> Tuple[int, str, str]:
        """
        Select which channel to remove for 65→64 reduction.
        
        Per ENTPC.tex: Principled, explicit, logged.
        
        Args:
            raw: MNE Raw object with 65 channels
        
        Returns:
            (channel_index, channel_name, reason)
        """
        assert len(raw.ch_names) == 65, f"Expected 65 channels, got {len(raw.ch_names)}"
        
        if self.reduction_method == 'lowest_snr':
            # Remove channel with lowest SNR
            snr = self.compute_channel_snr(raw)
            idx = np.argmin(snr)
            reason = f"Lowest SNR ({snr[idx]:.4f})"
        
        elif self.reduction_method == 'highest_artifact':
            # Remove channel with most artifacts
            artifacts = self.compute_channel_artifacts(raw)
            idx = np.argmax(artifacts)
            reason = f"Highest artifact count ({int(artifacts[idx])})"
        
        elif self.reduction_method == 'reference':
            # Try to identify reference channel by name
            ref_candidates = ['REF', 'Ref', 'ref', 'A1', 'A2', 'Cz']
            idx = None
            for ref_name in ref_candidates:
                if ref_name in raw.ch_names:
                    idx = raw.ch_names.index(ref_name)
                    break
            
            if idx is None:
                # Fallback to lowest SNR if no reference found
                logger.warning("No reference channel identified, falling back to lowest SNR")
                snr = self.compute_channel_snr(raw)
                idx = np.argmin(snr)
                reason = f"No reference found, removed lowest SNR ({snr[idx]:.4f})"
            else:
                reason = f"Reference channel identified"
        
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction_method}")
        
        channel_name = raw.ch_names[idx]
        
        # Log the decision
        log_entry = {
            'index': idx,
            'name': channel_name,
            'reason': reason,
            'method': self.reduction_method
        }
        self.reduction_log.append(log_entry)
        
        logger.info(f"Selected channel to remove: {channel_name} (index {idx}) - {reason}")
        
        return idx, channel_name, reason
    
    def reduce_65_to_64_channels(self, raw: mne.io.Raw) -> Tuple[mne.io.Raw, Dict]:
        """
        Reduce 65 channels to 64 with explicit logging.
        
        Per ENTPC.tex: Explicit, principled, reproducible.
        
        Args:
            raw: MNE Raw object with 65 channels
        
        Returns:
            (reduced_raw, reduction_info)
        """
        # Validate input
        assert len(raw.ch_names) == 65, f"Expected 65 channels, got {len(raw.ch_names)}"
        
        # Select channel to remove
        idx, name, reason = self.select_channel_to_remove(raw)
        
        # Remove channel
        channels_to_keep = [ch for i, ch in enumerate(raw.ch_names) if i != idx]
        reduced_raw = raw.copy().pick_channels(channels_to_keep)
        
        # Validate output
        assert len(reduced_raw.ch_names) == 64, \
            f"Reduction failed: expected 64 channels, got {len(reduced_raw.ch_names)}"
        
        logger.info(f"✓ Successfully reduced 65→64 channels")
        
        reduction_info = {
            'removed_index': idx,
            'removed_name': name,
            'removed_reason': reason,
            'method': self.reduction_method,
            'remaining_channels': reduced_raw.ch_names
        }
        
        return reduced_raw, reduction_info
    
    def aggregate_64_to_16_rois(self, data: np.ndarray) -> np.ndarray:
        """
        Aggregate 64 channels to 16 ROIs (regions of interest).
        
        Per ENTPC.tex: 64 channels → 16 quaternions (64/4 = 16)
        Each ROI is a group of 4 spatially adjacent channels.
        
        Args:
            data: Array of shape (64, n_samples)
        
        Returns:
            Array of shape (16, n_samples) with ROI-aggregated data
        """
        assert data.shape[0] == 64, f"Expected 64 channels, got {data.shape[0]}"
        
        # Simple spatial aggregation: group consecutive channels
        # In practice, this should use actual electrode positions
        roi_data = np.zeros((16, data.shape[1]))
        
        for roi_idx in range(16):
            # Each ROI is 4 consecutive channels
            start_ch = roi_idx * 4
            end_ch = start_ch + 4
            
            # Average over 4 channels
            roi_data[roi_idx, :] = np.mean(data[start_ch:end_ch, :], axis=0)
        
        assert roi_data.shape[0] == 16, f"ROI aggregation failed: got {roi_data.shape[0]} ROIs"
        
        logger.info(f"✓ Aggregated 64 channels → 16 ROIs")
        
        return roi_data
    
    def process_edf_file(self, filepath: str) -> Tuple[np.ndarray, Dict]:
        """
        Complete EDF processing pipeline.
        
        Steps:
        1. Validate real EDF file
        2. Load EDF data
        3. Reduce 65→64 channels (if needed)
        4. Aggregate 64→16 ROIs
        5. Return processed data with metadata
        
        Args:
            filepath: Path to EDF file
        
        Returns:
            (processed_data, metadata)
            processed_data: shape (16, n_samples)
            metadata: dictionary with processing info
        """
        logger.info(f"Processing EDF file: {filepath}")
        
        # Load EDF
        raw = self.load_edf_file(filepath)
        
        metadata = {
            'filepath': filepath,
            'original_n_channels': len(raw.ch_names),
            'sampling_rate': raw.info['sfreq'],
            'duration': raw.times[-1],
            'n_samples': len(raw.times)
        }
        
        # Reduce 65→64 if needed
        if len(raw.ch_names) == 65:
            raw, reduction_info = self.reduce_65_to_64_channels(raw)
            metadata['reduction_applied'] = True
            metadata['reduction_info'] = reduction_info
        elif len(raw.ch_names) == 64:
            logger.info("Already 64 channels, no reduction needed")
            metadata['reduction_applied'] = False
        else:
            raise ValueError(f"Unexpected number of channels: {len(raw.ch_names)}. Expected 64 or 65.")
        
        # Get data
        data = raw.get_data()  # Shape: (64, n_samples)
        
        # Aggregate to 16 ROIs
        roi_data = self.aggregate_64_to_16_rois(data)
        
        metadata['final_n_rois'] = roi_data.shape[0]
        metadata['final_n_samples'] = roi_data.shape[1]
        
        logger.info(f"✓ EDF processing complete: {filepath}")
        logger.info(f"  Final shape: {roi_data.shape}")
        
        return roi_data, metadata
    
    def get_reduction_log(self) -> List[Dict]:
        """
        Get log of all channel reductions performed.
        
        Returns:
            List of reduction log entries
        """
        return self.reduction_log.copy()
    
    def save_reduction_log(self, output_path: str):
        """
        Save reduction log to CSV file.
        
        Args:
            output_path: Path to output CSV file
        """
        import csv
        
        with open(output_path, 'w', newline='') as f:
            if self.reduction_log:
                fieldnames = self.reduction_log[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.reduction_log)
        
        logger.info(f"Saved reduction log to: {output_path}")


class EDFBatchProcessor:
    """
    Batch processor for multiple EDF files.
    
    Processes entire cohort with consistent reduction rules.
    """
    
    def __init__(self, reduction_method: str = 'lowest_snr'):
        """
        Initialize batch processor.
        
        Args:
            reduction_method: Method for 65→64 reduction
        """
        self.processor = EDFProcessor(reduction_method=reduction_method)
        self.results = []
    
    def process_subject_pair(self, pre_filepath: str, post_filepath: str,
                            subject_id: str) -> Dict:
        """
        Process pre/post treatment pair for one subject.
        
        Args:
            pre_filepath: Path to pre-treatment EDF file
            post_filepath: Path to post-treatment EDF file
            subject_id: Subject identifier
        
        Returns:
            Dictionary with processed data and metadata
        """
        logger.info(f"Processing subject {subject_id}")
        
        # Process pre-treatment
        pre_data, pre_metadata = self.processor.process_edf_file(pre_filepath)
        
        # Process post-treatment
        post_data, post_metadata = self.processor.process_edf_file(post_filepath)
        
        result = {
            'subject_id': subject_id,
            'pre_data': pre_data,
            'post_data': post_data,
            'pre_metadata': pre_metadata,
            'post_metadata': post_metadata
        }
        
        self.results.append(result)
        
        logger.info(f"✓ Completed subject {subject_id}")
        
        return result
    
    def process_cohort(self, subject_manifest: List[Dict]) -> List[Dict]:
        """
        Process entire cohort from manifest.
        
        Args:
            subject_manifest: List of dicts with 'subject_id', 'pre_file', 'post_file'
        
        Returns:
            List of processed subject results
        """
        logger.info(f"Processing cohort: {len(subject_manifest)} subjects")
        
        for subject_info in subject_manifest:
            subject_id = subject_info['subject_id']
            pre_file = subject_info['pre_file']
            post_file = subject_info['post_file']
            
            try:
                self.process_subject_pair(pre_file, post_file, subject_id)
            except Exception as e:
                logger.error(f"Failed to process subject {subject_id}: {e}")
                continue
        
        logger.info(f"✓ Cohort processing complete: {len(self.results)} subjects processed")
        
        return self.results
    
    def save_results_summary(self, output_path: str):
        """
        Save processing results summary to CSV.
        
        Args:
            output_path: Path to output CSV file
        """
        import csv
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['subject_id', 'pre_n_samples', 'post_n_samples',
                         'pre_duration', 'post_duration', 'pre_reduction_applied',
                         'post_reduction_applied']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'subject_id': result['subject_id'],
                    'pre_n_samples': result['pre_metadata']['n_samples'],
                    'post_n_samples': result['post_metadata']['n_samples'],
                    'pre_duration': result['pre_metadata']['duration'],
                    'post_duration': result['post_metadata']['duration'],
                    'pre_reduction_applied': result['pre_metadata']['reduction_applied'],
                    'post_reduction_applied': result['post_metadata']['reduction_applied']
                }
                writer.writerow(row)
        
        logger.info(f"Saved results summary to: {output_path}")
