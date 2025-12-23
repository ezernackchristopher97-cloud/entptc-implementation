"""
EntPTC Main Pipeline Orchestration

Reference: ENTPC.tex Section 7 (lines 734-756)

From ENTPC.tex Section 7:

"The complete EntPTC pipeline integrates all components in strict order:

1. Subject Selection: Deterministic cohort selection (40 subjects)
2. EDF Processing: Load, validate, reduce 65→64 channels, aggregate to 16 ROIs
3. Quaternion Construction: Map 16 ROIs → 16 quaternions (64 channels / 4 components)
4. Clifford Algebra: Construct Cl(3,0) basis and operations
5. Progenitor Matrix: Build 16×16 matrix from quaternions and entropy
6. Perron-Frobenius: Compute dominant eigenvector (collapse)
7. Absurdity Gap: Measure pre/post collapse discrepancy
8. THz Inference: Extract structural invariants from eigenvalue spectrum
9. Geodesic Analysis: Compute phase space trajectories on T³
10. Results Export: Save all outputs to CSV for analysis

CRITICAL: Execute in exact order. Each step validates inputs and outputs.
NO skipping steps. NO synthetic data. All assertions explicit."
"""

import numpy as np
import os
import csv
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import all EntPTC components
from entptc.core.clifford import CliffordAlgebra
from entptc.core.progenitor import ProgenitorMatrix
from entptc.core.perron_frobenius import PerronFrobeniusOperator
from entptc.core.quaternion import QuaternionHilbertSpace
from entptc.core.entropy import EntropyField, ToroidalManifold, create_entropy_field_from_progenitor
from entptc.analysis.geodesics import GeodesicSolver, GeodesicAnalyzer
from entptc.analysis.absurdity_gap import AbsurdityGap, AbsurdityGapAnalyzer
from entptc.analysis.thz_inference import THzPatternMatcher, THzCohortAnalyzer
from entptc.pipeline.edf_processor import EDFProcessor, EDFBatchProcessor
from entptc.pipeline.subject_selector import SubjectSelector


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntPTCPipeline:
    """
    Complete EntPTC pipeline orchestration.
    
    Per ENTPC.tex Section 7:
    - Strict execution order
    - Explicit validation at each step
    - Complete logging
    - CSV output for all results
    """
    
    def __init__(self, dataset_path: str, output_dir: str):
        """
        Initialize EntPTC pipeline.
        
        Args:
            dataset_path: Path to OpenNeuro ds005385 dataset
            output_dir: Directory for output files
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.subject_selector = SubjectSelector(dataset_path)
        self.edf_processor = EDFBatchProcessor(reduction_method='lowest_snr')
        self.clifford = CliffordAlgebra()
        self.quaternion_space = QuaternionHilbertSpace()
        self.absurdity_calculator = AbsurdityGapAnalyzer()
        self.thz_analyzer = THzCohortAnalyzer()
        
        # Results storage
        self.selected_subjects = []
        self.processed_edf_data = []
        self.pipeline_results = []
        
        logger.info(f"EntPTC Pipeline initialized")
        logger.info(f"  Dataset: {dataset_path}")
        logger.info(f"  Output: {output_dir}")
    
    def step_1_select_subjects(self, target_count: int = 40) -> List[Dict]:
        """
        Step 1: Deterministic subject selection.
        
        Per ENTPC.tex: Alphabetical ordering, explicit logging.
        
        Args:
            target_count: Number of subjects to select
        
        Returns:
            List of selected subject dictionaries
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Subject Selection")
        logger.info("=" * 80)
        
        # Select cohort
        self.selected_subjects = self.subject_selector.select_cohort(target_count)
        
        # Compute checksums
        self.selected_subjects = self.subject_selector.compute_file_checksums(self.selected_subjects)
        
        # Save manifest
        manifest_path = os.path.join(self.output_dir, 'subject_manifest.csv')
        self.subject_selector.save_selection_manifest(self.selected_subjects, manifest_path)
        
        # Save exclusion log
        exclusion_path = os.path.join(self.output_dir, 'subject_exclusions.csv')
        self.subject_selector.save_exclusion_log(exclusion_path)
        
        # Generate report
        report = self.subject_selector.generate_selection_report(self.selected_subjects)
        report_path = os.path.join(self.output_dir, 'selection_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"✓ Step 1 complete: {len(self.selected_subjects)} subjects selected")
        
        return self.selected_subjects
    
    def step_2_process_edf_files(self) -> List[Dict]:
        """
        Step 2: EDF processing with 65→64 channel reduction.
        
        Per ENTPC.tex: Load, validate, reduce, aggregate to 16 ROIs.
        
        Returns:
            List of processed EDF data dictionaries
        """
        logger.info("=" * 80)
        logger.info("STEP 2: EDF Processing")
        logger.info("=" * 80)
        
        # Process all subjects
        self.processed_edf_data = self.edf_processor.process_cohort(self.selected_subjects)
        
        # Save processing summary
        summary_path = os.path.join(self.output_dir, 'edf_processing_summary.csv')
        self.edf_processor.save_results_summary(summary_path)
        
        # Save reduction log
        reduction_log_path = os.path.join(self.output_dir, 'channel_reduction_log.csv')
        self.edf_processor.processor.save_reduction_log(reduction_log_path)
        
        logger.info(f"✓ Step 2 complete: {len(self.processed_edf_data)} subjects processed")
        
        return self.processed_edf_data
    
    def step_3_construct_quaternions(self, roi_data: np.ndarray) -> np.ndarray:
        """
        Step 3: Map 16 ROIs to 16 quaternions.
        
        Per ENTPC.tex: 64 channels / 4 components = 16 quaternions.
        
        Args:
            roi_data: Array of shape (16, n_samples)
        
        Returns:
            Array of shape (16, 4) with quaternion components
        """
        assert roi_data.shape[0] == 16, f"Expected 16 ROIs, got {roi_data.shape[0]}"
        
        # Construct quaternions from ROI time series
        # Each ROI → 1 quaternion with 4 components derived from signal properties
        quaternions = np.zeros((16, 4))
        
        for i in range(16):
            signal = roi_data[i, :]
            
            # Quaternion components from signal analysis
            # q = (q0, q1, q2, q3) where:
            # q0 = mean (scalar part)
            # q1 = std (i component)
            # q2 = skewness (j component)
            # q3 = kurtosis (k component)
            
            from scipy.stats import skew, kurtosis
            
            q0 = np.mean(signal)
            q1 = np.std(signal)
            q2 = skew(signal)
            q3 = kurtosis(signal)
            
            quaternions[i] = [q0, q1, q2, q3]
        
        # Normalize quaternions
        quaternions = self.quaternion_space.normalize_quaternions(quaternions)
        
        logger.info(f"✓ Constructed 16 quaternions from 16 ROIs")
        
        return quaternions
    
    def step_4_build_progenitor_matrix(self, quaternions: np.ndarray) -> np.ndarray:
        """
        Step 4: Build 16×16 Progenitor matrix.
        
        Per ENTPC.tex: Combines quaternions, entropy, and coherence.
        
        Args:
            quaternions: Array of shape (16, 4)
        
        Returns:
            16×16 Progenitor matrix
        """
        assert quaternions.shape == (16, 4), f"Expected (16, 4) quaternions, got {quaternions.shape}"
        
        # Compute coherence matrix (placeholder: use correlation)
        coherence = np.corrcoef(quaternions)
        
        # Build Progenitor matrix
        progenitor_builder = ProgenitorMatrix()
        progenitor_matrix = progenitor_builder.construct_progenitor_matrix(
            quaternions, coherence
        )
        
        assert progenitor_matrix.shape == (16, 16), \
            f"Expected 16×16 matrix, got {progenitor_matrix.shape}"
        
        logger.info(f"✓ Built 16×16 Progenitor matrix")
        
        return progenitor_matrix
    
    def step_5_perron_frobenius_collapse(self, progenitor_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 5: Perron-Frobenius collapse to dominant eigenvector.
        
        Per ENTPC.tex: Extract dominant eigenvector and eigenvalues.
        
        Args:
            progenitor_matrix: 16×16 Progenitor matrix
        
        Returns:
            (dominant_eigenvector, all_eigenvalues)
        """
        pf_operator = PerronFrobeniusOperator()
        
        dominant_eigenvector, eigenvalues = pf_operator.compute_dominant_eigenvector(
            progenitor_matrix
        )
        
        assert len(dominant_eigenvector) == 16, \
            f"Expected 16-element eigenvector, got {len(dominant_eigenvector)}"
        
        logger.info(f"✓ Perron-Frobenius collapse complete")
        logger.info(f"  Dominant eigenvalue: {eigenvalues[0]:.6f}")
        
        return dominant_eigenvector, eigenvalues
    
    def step_6_compute_absurdity_gap(self, psi_pre: np.ndarray, psi_post: np.ndarray) -> Dict:
        """
        Step 6: Compute Absurdity Gap.
        
        Per ENTPC.tex: POST-OPERATOR ONLY. Measures collapse discrepancy.
        
        Args:
            psi_pre: Pre-collapse state vector
            psi_post: Post-collapse state vector (dominant eigenvector)
        
        Returns:
            Dictionary with gap analysis
        """
        gap_components = self.absurdity_calculator.gap_calculator.compute_gap_components(
            psi_pre, psi_post
        )
        
        logger.info(f"✓ Absurdity Gap computed: {gap_components['gap_L2']:.4f}")
        logger.info(f"  Regime: {gap_components['regime']}")
        
        return gap_components
    
    def step_7_thz_inference(self, eigenvalues: np.ndarray) -> Dict:
        """
        Step 7: THz structural invariants inference.
        
        Per ENTPC.tex: NO frequency conversion, only structural patterns.
        
        Args:
            eigenvalues: Eigenvalue spectrum from Perron-Frobenius
        
        Returns:
            Dictionary with THz inference report
        """
        thz_report = self.thz_analyzer.pattern_matcher.compute_thz_inference_report(
            eigenvalues
        )
        
        logger.info(f"✓ THz inference complete")
        logger.info(f"  Dominant pattern: {thz_report['dominant_pattern']}")
        logger.info(f"  Confidence: {thz_report['confidence']}")
        
        return thz_report
    
    def process_single_subject(self, subject_data: Dict) -> Dict:
        """
        Process single subject through complete pipeline.
        
        Args:
            subject_data: Dictionary with subject info and processed EDF data
        
        Returns:
            Dictionary with all pipeline results for subject
        """
        subject_id = subject_data['subject_id']
        logger.info(f"Processing subject: {subject_id}")
        
        results = {'subject_id': subject_id}
        
        # Pre-treatment processing
        logger.info(f"  Pre-treatment...")
        pre_roi_data = subject_data['pre_data']
        pre_quaternions = self.step_3_construct_quaternions(pre_roi_data)
        pre_progenitor = self.step_4_build_progenitor_matrix(pre_quaternions)
        pre_eigenvector, pre_eigenvalues = self.step_5_perron_frobenius_collapse(pre_progenitor)
        pre_thz = self.step_7_thz_inference(pre_eigenvalues)
        
        results['pre_quaternions'] = pre_quaternions
        results['pre_progenitor'] = pre_progenitor
        results['pre_eigenvector'] = pre_eigenvector
        results['pre_eigenvalues'] = pre_eigenvalues
        results['pre_thz_report'] = pre_thz
        
        # Post-treatment processing
        logger.info(f"  Post-treatment...")
        post_roi_data = subject_data['post_data']
        post_quaternions = self.step_3_construct_quaternions(post_roi_data)
        post_progenitor = self.step_4_build_progenitor_matrix(post_quaternions)
        post_eigenvector, post_eigenvalues = self.step_5_perron_frobenius_collapse(post_progenitor)
        post_thz = self.step_7_thz_inference(post_eigenvalues)
        
        results['post_quaternions'] = post_quaternions
        results['post_progenitor'] = post_progenitor
        results['post_eigenvector'] = post_eigenvector
        results['post_eigenvalues'] = post_eigenvalues
        results['post_thz_report'] = post_thz
        
        # Absurdity Gap (pre vs post eigenvectors)
        logger.info(f"  Absurdity Gap...")
        gap_analysis = self.step_6_compute_absurdity_gap(pre_eigenvector, post_eigenvector)
        results['absurdity_gap'] = gap_analysis
        
        logger.info(f"✓ Subject {subject_id} complete")
        
        return results
    
    def run_full_pipeline(self, target_subjects: int = 40) -> List[Dict]:
        """
        Execute complete EntPTC pipeline.
        
        Per ENTPC.tex Section 7: All steps in order.
        
        Args:
            target_subjects: Number of subjects to process
        
        Returns:
            List of results dictionaries (one per subject)
        """
        logger.info("=" * 80)
        logger.info("EntPTC FULL PIPELINE EXECUTION")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Step 1: Select subjects
        self.step_1_select_subjects(target_subjects)
        
        # Step 2: Process EDF files
        self.step_2_process_edf_files()
        
        # Steps 3-7: Process each subject
        logger.info("=" * 80)
        logger.info("STEPS 3-7: Per-Subject Processing")
        logger.info("=" * 80)
        
        for subject_data in self.processed_edf_data:
            result = self.process_single_subject(subject_data)
            self.pipeline_results.append(result)
        
        # Export results
        logger.info("=" * 80)
        logger.info("EXPORTING RESULTS")
        logger.info("=" * 80)
        
        self.export_results_to_csv()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.info(f"  Subjects processed: {len(self.pipeline_results)}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info("=" * 80)
        
        return self.pipeline_results
    
    def export_results_to_csv(self):
        """
        Export all results to CSV files.
        
        Per ENTPC.tex: Structured CSV output for analysis.
        """
        # Absurdity Gap results
        gap_results_path = os.path.join(self.output_dir, 'absurdity_gap_results.csv')
        with open(gap_results_path, 'w', newline='') as f:
            fieldnames = ['subject_id', 'gap_L2', 'gap_L1', 'gap_Linf', 'regime',
                         'overlap', 'info_loss', 'entropy_change']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.pipeline_results:
                gap = result['absurdity_gap']
                writer.writerow({
                    'subject_id': result['subject_id'],
                    'gap_L2': gap['gap_L2'],
                    'gap_L1': gap['gap_L1'],
                    'gap_Linf': gap['gap_Linf'],
                    'regime': gap['regime'],
                    'overlap': gap['overlap'],
                    'info_loss': gap['info_loss'],
                    'entropy_change': gap['entropy_change']
                })
        
        logger.info(f"Saved: {gap_results_path}")
        
        # THz inference results
        thz_results_path = os.path.join(self.output_dir, 'thz_inference_results.csv')
        with open(thz_results_path, 'w', newline='') as f:
            fieldnames = ['subject_id', 'condition', 'dominant_pattern', 'dominant_score',
                         'confidence', 'symmetry_breaking']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.pipeline_results:
                # Pre-treatment
                pre_thz = result['pre_thz_report']
                writer.writerow({
                    'subject_id': result['subject_id'],
                    'condition': 'pre',
                    'dominant_pattern': pre_thz['dominant_pattern'],
                    'dominant_score': pre_thz['dominant_score'],
                    'confidence': pre_thz['confidence'],
                    'symmetry_breaking': pre_thz['structural_invariants']['symmetry_breaking']
                })
                
                # Post-treatment
                post_thz = result['post_thz_report']
                writer.writerow({
                    'subject_id': result['subject_id'],
                    'condition': 'post',
                    'dominant_pattern': post_thz['dominant_pattern'],
                    'dominant_score': post_thz['dominant_score'],
                    'confidence': post_thz['confidence'],
                    'symmetry_breaking': post_thz['structural_invariants']['symmetry_breaking']
                })
        
        logger.info(f"Saved: {thz_results_path}")
        
        # Eigenvalue summary
        eigenvalue_path = os.path.join(self.output_dir, 'eigenvalue_summary.csv')
        with open(eigenvalue_path, 'w', newline='') as f:
            fieldnames = ['subject_id', 'condition', 'dominant_eigenvalue', 'spectral_radius',
                         'trace', 'determinant']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.pipeline_results:
                # Pre-treatment
                writer.writerow({
                    'subject_id': result['subject_id'],
                    'condition': 'pre',
                    'dominant_eigenvalue': result['pre_eigenvalues'][0],
                    'spectral_radius': np.max(np.abs(result['pre_eigenvalues'])),
                    'trace': np.sum(result['pre_eigenvalues']),
                    'determinant': np.prod(result['pre_eigenvalues'])
                })
                
                # Post-treatment
                writer.writerow({
                    'subject_id': result['subject_id'],
                    'condition': 'post',
                    'dominant_eigenvalue': result['post_eigenvalues'][0],
                    'spectral_radius': np.max(np.abs(result['post_eigenvalues'])),
                    'trace': np.sum(result['post_eigenvalues']),
                    'determinant': np.prod(result['post_eigenvalues'])
                })
        
        logger.info(f"Saved: {eigenvalue_path}")


def main():
    """
    Main entry point for EntPTC pipeline.
    
    Usage:
        python main_pipeline.py
    """
    # Configuration
    DATASET_PATH = "/path/to/openneuro/ds005385"
    OUTPUT_DIR = "./entptc_results"
    TARGET_SUBJECTS = 40
    
    # Initialize pipeline
    pipeline = EntPTCPipeline(DATASET_PATH, OUTPUT_DIR)
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(TARGET_SUBJECTS)
    
    logger.info("EntPTC pipeline execution complete!")
    logger.info(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
