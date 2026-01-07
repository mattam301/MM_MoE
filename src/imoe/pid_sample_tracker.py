"""
PID Sample Tracker - Know exactly which samples you're analyzing
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime


@dataclass
class SamplePIDRecord:
    """Complete PID record for a single sample."""
    
    # Identification
    sample_id: Any                    # Original dataset ID
    batch_idx: int                    # Position in batch
    epoch: int                        # When analyzed
    split: str                        # 'train', 'val', 'test'
    
    # Ground truth
    true_label: int
    true_label_name: Optional[str] = None
    
    # Prediction
    predicted_label: int = -1
    prediction_correct: bool = False
    confidence: float = 0.0
    
    # PID Components (raw magnitudes)
    unique_norm: float = 0.0
    redundant_norm: float = 0.0
    synergy_norm: float = 0.0
    
    # PID Components (percentages)
    unique_pct: float = 0.0
    redundant_pct: float = 0.0
    synergy_pct: float = 0.0
    
    # Dominant component
    dominant_component: str = ""      # 'U', 'R', or 'S'
    
    # Per-modality info (if available)
    modality_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Optional: raw data snippet for reference
    data_snippet: Optional[str] = None


class PIDSampleTracker:
    """
    Track and analyze PID components at the sample level.
    Provides full traceability from analysis back to original data.
    """
    
    def __init__(
        self, 
        save_dir: str = "./pid_analysis",
        class_names: Optional[List[str]] = None,
        modality_names: Optional[List[str]] = None,
        max_samples_per_epoch: int = 500,  # Limit memory usage
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_names = class_names or []
        self.modality_names = modality_names or []
        self.max_samples = max_samples_per_epoch
        
        # Storage
        self.records: List[SamplePIDRecord] = []
        self.epoch_summaries: List[Dict] = []
        
        # Current analysis context
        self.current_epoch = 0
        self.current_split = "unknown"
        
    def set_context(self, epoch: int, split: str = "val"):
        """Set current analysis context."""
        self.current_epoch = epoch
        self.current_split = split
    
    def analyze_batch(
        self,
        sample_ids: torch.Tensor,           # [B] - Original IDs from dataloader
        u: torch.Tensor,                     # [B, D] - Unique components
        r: torch.Tensor,                     # [B, D] - Redundant components
        s: torch.Tensor,                     # [B, D] - Synergy components
        labels: torch.Tensor,                # [B] - True labels
        outputs: torch.Tensor,               # [B, C] - Model outputs
        raw_samples: Optional[Dict] = None,  # Original input data for snippets
        verbose: bool = True,
    ) -> List[SamplePIDRecord]:
        """
        Analyze a batch and create detailed per-sample records.
        
        Args:
            sample_ids: Original sample IDs from the dataset
            u, r, s: PID components
            labels: Ground truth labels
            outputs: Model predictions (logits)
            raw_samples: Optional dict of {modality: tensor} for data snippets
            verbose: Print analysis summary
            
        Returns:
            List of SamplePIDRecord for each sample in batch
        """
        
        batch_size = u.shape[0]
        records = []
        
        # Compute predictions
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        confidences = probs.max(dim=1).values
        
        # Compute norms
        u_norms = u.norm(dim=1)
        r_norms = r.norm(dim=1)
        s_norms = s.norm(dim=1)
        totals = u_norms + r_norms + s_norms + 1e-8
        
        # Determine dominant component
        stacked = torch.stack([u_norms, r_norms, s_norms], dim=1)
        dominant_idx = stacked.argmax(dim=1)
        component_names = ['U', 'R', 'S']
        
        for i in range(min(batch_size, self.max_samples)):
            # Get sample ID
            if isinstance(sample_ids, torch.Tensor):
                sid = sample_ids[i].item() if sample_ids[i].numel() == 1 else str(sample_ids[i].tolist())
            else:
                sid = sample_ids[i]
            
            # Get label name
            label_val = labels[i].item()
            label_name = self.class_names[label_val] if label_val < len(self.class_names) else f"Class_{label_val}"
            
            # Create record
            record = SamplePIDRecord(
                sample_id=sid,
                batch_idx=i,
                epoch=self.current_epoch,
                split=self.current_split,
                
                true_label=label_val,
                true_label_name=label_name,
                
                predicted_label=preds[i].item(),
                prediction_correct=(preds[i] == labels[i]).item(),
                confidence=confidences[i].item(),
                
                unique_norm=u_norms[i].item(),
                redundant_norm=r_norms[i].item(),
                synergy_norm=s_norms[i].item(),
                
                unique_pct=u_norms[i].item() / totals[i].item() * 100,
                redundant_pct=r_norms[i].item() / totals[i].item() * 100,
                synergy_pct=s_norms[i].item() / totals[i].item() * 100,
                
                dominant_component=component_names[dominant_idx[i].item()],
                
                data_snippet=self._create_snippet(raw_samples, i) if raw_samples else None,
            )
            
            records.append(record)
            self.records.append(record)
        
        if verbose:
            self._print_batch_summary(records)
        
        return records
    
    def _create_snippet(self, raw_samples: Dict, idx: int, max_len: int = 100) -> str:
        """Create a brief data snippet for reference."""
        snippets = []
        
        for mod_name, data in raw_samples.items():
            if isinstance(data, torch.Tensor):
                sample = data[idx]
                
                # Handle different data types
                if sample.dtype in [torch.float32, torch.float64]:
                    # Numerical features - show stats
                    snippet = f"{mod_name}: shape={list(sample.shape)}, mean={sample.mean():.3f}"
                elif sample.dtype in [torch.int32, torch.int64, torch.long]:
                    # Token IDs - show first few
                    tokens = sample[:10].tolist()
                    snippet = f"{mod_name}: {tokens}..."
                else:
                    snippet = f"{mod_name}: {sample.shape}"
                    
                snippets.append(snippet)
        
        result = " | ".join(snippets)
        return result[:max_len] + "..." if len(result) > max_len else result
    
    def _print_batch_summary(self, records: List[SamplePIDRecord]):
        """Print informative summary of analyzed batch."""
        
        n = len(records)
        if n == 0:
            return
            
        print(f"\n  {'='*60}")
        print(f"  PID Analysis: Epoch {self.current_epoch}, Split: {self.current_split}")
        print(f"  Samples analyzed: {n}")
        print(f"  {'='*60}")
        
        # Sample ID range
        ids = [r.sample_id for r in records]
        print(f"\n  📋 Sample IDs: {ids[:5]}{'...' if len(ids) > 5 else ''}")
        
        # Overall stats
        u_pcts = [r.unique_pct for r in records]
        r_pcts = [r.redundant_pct for r in records]
        s_pcts = [r.synergy_pct for r in records]
        
        print(f"\n  📊 Component Distribution (mean ± std):")
        print(f"      Unique:    {np.mean(u_pcts):5.1f}% ± {np.std(u_pcts):4.1f}%")
        print(f"      Redundant: {np.mean(r_pcts):5.1f}% ± {np.std(r_pcts):4.1f}%")
        print(f"      Synergy:   {np.mean(s_pcts):5.1f}% ± {np.std(s_pcts):4.1f}%")
        
        # Dominant component breakdown
        dom_counts = {'U': 0, 'R': 0, 'S': 0}
        for r in records:
            dom_counts[r.dominant_component] += 1
        
        print(f"\n  🎯 Dominant Component:")
        for comp, count in dom_counts.items():
            bar = '█' * int(count / n * 20)
            print(f"      {comp}: {bar} {count}/{n} ({count/n*100:.1f}%)")
        
        # Accuracy by dominant component
        print(f"\n  ✅ Accuracy by Dominant Component:")
        for comp in ['U', 'R', 'S']:
            comp_records = [r for r in records if r.dominant_component == comp]
            if comp_records:
                acc = sum(r.prediction_correct for r in comp_records) / len(comp_records) * 100
                print(f"      {comp}: {acc:5.1f}% (n={len(comp_records)})")
        
        # Per-class breakdown
        print(f"\n  📈 Per-Class Analysis:")
        classes = set(r.true_label for r in records)
        for cls in sorted(classes):
            cls_records = [r for r in records if r.true_label == cls]
            cls_name = cls_records[0].true_label_name if cls_records else f"Class_{cls}"
            
            u_mean = np.mean([r.unique_pct for r in cls_records])
            r_mean = np.mean([r.redundant_pct for r in cls_records])
            s_mean = np.mean([r.synergy_pct for r in cls_records])
            acc = sum(r.prediction_correct for r in cls_records) / len(cls_records) * 100
            
            print(f"      {cls_name:15s}: U={u_mean:4.1f}% R={r_mean:4.1f}% S={s_mean:4.1f}% | Acc={acc:5.1f}%")
        
        # Extreme samples
        self._print_extreme_samples(records)
    
    def _print_extreme_samples(self, records: List[SamplePIDRecord], top_k: int = 3):
        """Print most extreme samples for each component."""
        
        print(f"\n  🔍 Extreme Samples (Top {top_k}):")
        
        # Highest unique
        by_unique = sorted(records, key=lambda r: r.unique_pct, reverse=True)[:top_k]
        print(f"\n      Highest UNIQUE (modality-specific info):")
        for r in by_unique:
            status = "✓" if r.prediction_correct else "✗"
            print(f"        {status} ID={r.sample_id}: U={r.unique_pct:.1f}% | "
                  f"Label={r.true_label_name} | Conf={r.confidence:.2f}")
            if r.data_snippet:
                print(f"          └─ {r.data_snippet}")
        
        # Highest redundant
        by_redundant = sorted(records, key=lambda r: r.redundant_pct, reverse=True)[:top_k]
        print(f"\n      Highest REDUNDANT (modalities agree):")
        for r in by_redundant:
            status = "✓" if r.prediction_correct else "✗"
            print(f"        {status} ID={r.sample_id}: R={r.redundant_pct:.1f}% | "
                  f"Label={r.true_label_name} | Conf={r.confidence:.2f}")
            if r.data_snippet:
                print(f"          └─ {r.data_snippet}")
        
        # Highest synergy
        by_synergy = sorted(records, key=lambda r: r.synergy_pct, reverse=True)[:top_k]
        print(f"\n      Highest SYNERGY (cross-modal reasoning):")
        for r in by_synergy:
            status = "✓" if r.prediction_correct else "✗"
            print(f"        {status} ID={r.sample_id}: S={r.synergy_pct:.1f}% | "
                  f"Label={r.true_label_name} | Conf={r.confidence:.2f}")
            if r.data_snippet:
                print(f"          └─ {r.data_snippet}")
        
        # Misclassified with high confidence (interesting errors)
        errors = [r for r in records if not r.prediction_correct and r.confidence > 0.7]
        if errors:
            print(f"\n      ⚠️ Confident Errors (investigate these!):")
            for r in sorted(errors, key=lambda r: r.confidence, reverse=True)[:top_k]:
                print(f"        ID={r.sample_id}: True={r.true_label_name}, "
                      f"Pred=Class_{r.predicted_label}, Conf={r.confidence:.2f}")
                print(f"          PID: U={r.unique_pct:.1f}% R={r.redundant_pct:.1f}% S={r.synergy_pct:.1f}%")
    
    def get_samples_by_criteria(
        self,
        dominant_component: Optional[str] = None,
        correct_only: Optional[bool] = None,
        label: Optional[int] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        epoch: Optional[int] = None,
    ) -> List[SamplePIDRecord]:
        """
        Query samples by various criteria.
        
        Example:
            # Get all synergy-dominant misclassified samples
            tracker.get_samples_by_criteria(dominant_component='S', correct_only=False)
        """
        
        results = self.records
        
        if dominant_component:
            results = [r for r in results if r.dominant_component == dominant_component]
        if correct_only is not None:
            results = [r for r in results if r.prediction_correct == correct_only]
        if label is not None:
            results = [r for r in results if r.true_label == label]
        if min_confidence is not None:
            results = [r for r in results if r.confidence >= min_confidence]
        if max_confidence is not None:
            results = [r for r in results if r.confidence <= max_confidence]
        if epoch is not None:
            results = [r for r in results if r.epoch == epoch]
            
        return results
    
    def save_records(self, filename: Optional[str] = None):
        """Save all records to CSV and JSON."""
        
        if not self.records:
            print("  No records to save.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = filename or f"pid_samples_{timestamp}"
        
        # Save as CSV
        csv_path = self.save_dir / f"{base_name}.csv"
        with open(csv_path, 'w', newline='') as f:
            # Get field names from dataclass
            fieldnames = [field for field in asdict(self.records[0]).keys() 
                         if field != 'modality_contributions']  # Skip complex fields
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                row = {k: v for k, v in asdict(record).items() if k in fieldnames}
                writer.writerow(row)
        
        print(f"  💾 Saved {len(self.records)} records to {csv_path}")
        
        # Save as JSON (includes all fields)
        json_path = self.save_dir / f"{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in self.records], f, indent=2, default=str)
        
        print(f"  💾 Saved detailed records to {json_path}")
        
        return csv_path, json_path
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive analysis report."""
        
        if not self.records:
            return "No records to analyze."
        
        report = []
        report.append("=" * 70)
        report.append("PID SAMPLE ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total samples analyzed: {len(self.records)}")
        report.append(f"Epochs covered: {len(set(r.epoch for r in self.records))}")
        report.append("")
        
        # Overall statistics
        report.append("-" * 70)
        report.append("OVERALL STATISTICS")
        report.append("-" * 70)
        
        u_pcts = [r.unique_pct for r in self.records]
        r_pcts = [r.redundant_pct for r in self.records]
        s_pcts = [r.synergy_pct for r in self.records]
        
        report.append(f"Component Distribution:")
        report.append(f"  Unique:    {np.mean(u_pcts):5.1f}% ± {np.std(u_pcts):4.1f}%  (range: {min(u_pcts):.1f}% - {max(u_pcts):.1f}%)")
        report.append(f"  Redundant: {np.mean(r_pcts):5.1f}% ± {np.std(r_pcts):4.1f}%  (range: {min(r_pcts):.1f}% - {max(r_pcts):.1f}%)")
        report.append(f"  Synergy:   {np.mean(s_pcts):5.1f}% ± {np.std(s_pcts):4.1f}%  (range: {min(s_pcts):.1f}% - {max(s_pcts):.1f}%)")
        report.append("")
        
        # Accuracy analysis
        overall_acc = sum(r.prediction_correct for r in self.records) / len(self.records) * 100
        report.append(f"Overall Accuracy: {overall_acc:.2f}%")
        report.append("")
        
        report.append("Accuracy by Dominant Component:")
        for comp in ['U', 'R', 'S']:
            comp_records = [r for r in self.records if r.dominant_component == comp]
            if comp_records:
                acc = sum(r.prediction_correct for r in comp_records) / len(comp_records) * 100
                report.append(f"  {comp}: {acc:5.1f}% (n={len(comp_records)}, {len(comp_records)/len(self.records)*100:.1f}% of samples)")
        report.append("")
        
        # Per-class analysis
        report.append("-" * 70)
        report.append("PER-CLASS ANALYSIS")
        report.append("-" * 70)
        
        classes = sorted(set(r.true_label for r in self.records))
        for cls in classes:
            cls_records = [r for r in self.records if r.true_label == cls]
            cls_name = cls_records[0].true_label_name
            
            report.append(f"\n{cls_name} (n={len(cls_records)}):")
            report.append(f"  PID: U={np.mean([r.unique_pct for r in cls_records]):.1f}% | "
                         f"R={np.mean([r.redundant_pct for r in cls_records]):.1f}% | "
                         f"S={np.mean([r.synergy_pct for r in cls_records]):.1f}%")
            
            acc = sum(r.prediction_correct for r in cls_records) / len(cls_records) * 100
            report.append(f"  Accuracy: {acc:.1f}%")
            
            dom_dist = {'U': 0, 'R': 0, 'S': 0}
            for r in cls_records:
                dom_dist[r.dominant_component] += 1
            report.append(f"  Dominant: U={dom_dist['U']} R={dom_dist['R']} S={dom_dist['S']}")
        
        # Interesting samples
        report.append("")
        report.append("-" * 70)
        report.append("NOTABLE SAMPLES")
        report.append("-" * 70)
        
        # Extreme unique
        report.append("\nTop 5 Unique-Dominant Samples:")
        for r in sorted(self.records, key=lambda x: x.unique_pct, reverse=True)[:5]:
            report.append(f"  ID={r.sample_id}: U={r.unique_pct:.1f}%, Label={r.true_label_name}, "
                         f"{'✓' if r.prediction_correct else '✗'}")
        
        # Extreme synergy
        report.append("\nTop 5 Synergy-Dominant Samples:")
        for r in sorted(self.records, key=lambda x: x.synergy_pct, reverse=True)[:5]:
            report.append(f"  ID={r.sample_id}: S={r.synergy_pct:.1f}%, Label={r.true_label_name}, "
                         f"{'✓' if r.prediction_correct else '✗'}")
        
        # Confident errors
        errors = [r for r in self.records if not r.prediction_correct]
        if errors:
            report.append(f"\nTop 5 Confident Errors (for investigation):")
            for r in sorted(errors, key=lambda x: x.confidence, reverse=True)[:5]:
                report.append(f"  ID={r.sample_id}: True={r.true_label_name}, "
                             f"Pred=Class_{r.predicted_label}, Conf={r.confidence:.2f}")
                report.append(f"    PID: U={r.unique_pct:.1f}% R={r.redundant_pct:.1f}% S={r.synergy_pct:.1f}%")
        
        report.append("")
        report.append("=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        if save_path:
            path = Path(save_path)
        else:
            path = self.save_dir / f"pid_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(path, 'w') as f:
            f.write(report_text)
        
        print(f"  📄 Report saved to {path}")
        
        return report_text