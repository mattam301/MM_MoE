"""Functions defined for extracting insights from PID"""

def log_per_class_split(u, r, s, labels, class_names=None):
    """Analyze information split per class."""
    unique_labels = labels.unique()
    
    print("  Per-Class Info Split:")
    print("  " + "-" * 50)
    
    for label in unique_labels:
        mask = labels == label
        
        u_contrib = u[mask].norm(dim=1).mean().item()
        r_contrib = r[mask].norm(dim=1).mean().item()
        s_contrib = s[mask].norm(dim=1).mean().item()
        total = u_contrib + r_contrib + s_contrib + 1e-8
        
        name = class_names[label.item()] if class_names else f"Class {label.item()}"
        print(f"    {name}: U={u_contrib/total*100:5.1f}% | R={r_contrib/total*100:5.1f}% | S={s_contrib/total*100:5.1f}%")
        

import torch
def log_confidence_correlation(u, r, s, outputs):
    """Check: Does high synergy correlate with prediction confidence?"""
    
    probs = torch.softmax(outputs, dim=1)
    confidence = probs.max(dim=1).values  # [B]
    
    # Compute per-sample dominant component
    u_norm = u.norm(dim=1)
    r_norm = r.norm(dim=1)
    s_norm = s.norm(dim=1)
    
    # Correlation with confidence
    def corr(x, y):
        x = x - x.mean()
        y = y - y.mean()
        return (x * y).sum() / (x.norm() * y.norm() + 1e-8)
    
    print("  Confidence Correlation:")
    print(f"    U ↔ Conf: {corr(u_norm, confidence).item():+.3f}")
    print(f"    R ↔ Conf: {corr(r_norm, confidence).item():+.3f}")
    print(f"    S ↔ Conf: {corr(s_norm, confidence).item():+.3f}")

def find_extreme_samples(u, r, s, sample_ids, top_k=3):
    """Find samples with extreme U, R, S values for manual inspection."""
    
    u_norm = u.norm(dim=1)
    r_norm = r.norm(dim=1)
    s_norm = s.norm(dim=1)
    
    print("  Extreme Samples (for manual inspection):")
    
    # Highest unique
    top_u = u_norm.topk(top_k).indices
    print(f"    Highest UNIQUE: {[sample_ids[i].item() if hasattr(sample_ids[i], 'item') else sample_ids[i] for i in top_u]}")
    
    # Highest redundant
    top_r = r_norm.topk(top_k).indices
    print(f"    Highest REDUNDANT: {[sample_ids[i].item() if hasattr(sample_ids[i], 'item') else sample_ids[i] for i in top_r]}")
    
    # Highest synergy
    top_s = s_norm.topk(top_k).indices
    print(f"    Highest SYNERGY: {[sample_ids[i].item() if hasattr(sample_ids[i], 'item') else sample_ids[i] for i in top_s]}")

# Usage (need sample IDs from dataloader):
# find_extreme_samples(u, r, s, batch_ids, top_k=3)

def log_dominant_component_distribution(u, r, s):
    """What % of samples are dominated by each component?"""
    
    u_norm = u.norm(dim=1)
    r_norm = r.norm(dim=1)
    s_norm = s.norm(dim=1)
    
    stacked = torch.stack([u_norm, r_norm, s_norm], dim=1)  # [B, 3]
    dominant = stacked.argmax(dim=1)  # [B]
    
    total = len(dominant)
    u_dom = (dominant == 0).sum().item() / total * 100
    r_dom = (dominant == 1).sum().item() / total * 100
    s_dom = (dominant == 2).sum().item() / total * 100
    
    # Visual bar
    def bar(pct, char='█'):
        return char * int(pct / 5)
    
    print("  Dominant Component Distribution:")
    print(f"    U: {bar(u_dom)} {u_dom:.1f}%")
    print(f"    R: {bar(r_dom)} {r_dom:.1f}%")
    print(f"    S: {bar(s_dom)} {s_dom:.1f}%")

def log_accuracy_by_dominant_component(u, r, s, preds, labels):
    """Are synergy-dominant samples harder to classify correctly?"""
    
    u_norm = u.norm(dim=1)
    r_norm = r.norm(dim=1)
    s_norm = s.norm(dim=1)
    
    stacked = torch.stack([u_norm, r_norm, s_norm], dim=1)
    dominant = stacked.argmax(dim=1)
    
    correct = (preds == labels)
    
    names = ['Unique', 'Redundant', 'Synergy']
    print("  Accuracy by Dominant Component:")
    
    for i, name in enumerate(names):
        mask = dominant == i
        if mask.sum() > 0:
            acc = correct[mask].float().mean().item() * 100
            count = mask.sum().item()
            print(f"    {name:10s}: {acc:5.1f}% (n={count})")
        else:
            print(f"    {name:10s}: N/A (n=0)")

import csv
from pathlib import Path

class SimpleComponentTracker:
    """Minimal tracker - saves to CSV for later analysis."""
    
    def __init__(self, save_path="pid_tracking.csv"):
        self.save_path = Path(save_path)
        self.data = []
        
    def log(self, epoch, u, r, s, val_acc=None):
        """Log one epoch."""
        u_mean = u.norm(dim=1).mean().item()
        r_mean = r.norm(dim=1).mean().item()
        s_mean = s.norm(dim=1).mean().item()
        total = u_mean + r_mean + s_mean + 1e-8
        
        self.data.append({
            'epoch': epoch,
            'u_pct': u_mean / total * 100,
            'r_pct': r_mean / total * 100,
            's_pct': s_mean / total * 100,
            'u_norm': u_mean,
            'r_norm': r_mean,
            's_norm': s_mean,
            'val_acc': val_acc or 0,
        })
    
    def save(self):
        """Save to CSV."""
        if not self.data:
            return
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)
        print(f"  Saved PID tracking to {self.save_path}")
    
    def print_summary(self):
        """Print training summary."""
        if len(self.data) < 2:
            return
        
        first = self.data[0]
        last = self.data[-1]
        
        print("\n  PID Evolution Summary:")
        print("  " + "=" * 50)
        print(f"    Component    | Start   →  End     | Change")
        print("  " + "-" * 50)
        print(f"    Unique       | {first['u_pct']:5.1f}%  →  {last['u_pct']:5.1f}%  | {last['u_pct']-first['u_pct']:+5.1f}%")
        print(f"    Redundant    | {first['r_pct']:5.1f}%  →  {last['r_pct']:5.1f}%  | {last['r_pct']-first['r_pct']:+5.1f}%")
        print(f"    Synergy      | {first['s_pct']:5.1f}%  →  {last['s_pct']:5.1f}%  | {last['s_pct']-first['s_pct']:+5.1f}%")
        print("  " + "=" * 50)

# # Usage:
# tracker = SimpleComponentTracker(save_path=f"./results/{args.data}_pid.csv")

# # In epoch loop:
# tracker.log(epoch, u, r, s, val_acc=val_acc)

# # After training:
# tracker.save()
# tracker.print_summary()

def plot_component_histograms(u, r, s, save_path=None):
    """Quick histogram of component magnitudes."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    u_norm = u.norm(dim=1).detach().cpu().numpy()
    r_norm = r.norm(dim=1).detach().cpu().numpy()
    s_norm = s.norm(dim=1).detach().cpu().numpy()
    
    axes[0].hist(u_norm, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_title(f'Unique (μ={u_norm.mean():.2f})')
    axes[0].set_xlabel('Magnitude')
    
    axes[1].hist(r_norm, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_title(f'Redundant (μ={r_norm.mean():.2f})')
    axes[1].set_xlabel('Magnitude')
    
    axes[2].hist(s_norm, bins=30, color='red', alpha=0.7, edgecolor='black')
    axes[2].set_title(f'Synergy (μ={s_norm.mean():.2f})')
    axes[2].set_xlabel('Magnitude')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"  Saved histogram to {save_path}")
    else:
        plt.show()
    plt.close()

# # Usage (every N epochs):
# if epoch % 10 == 0:
#     plot_component_histograms(u, r, s, save_path=f"./results/hist_epoch{epoch}.png")

def print_trend_ascii(tracker_data, width=40):
    """Print ASCII trend of component evolution."""
    
    if len(tracker_data) < 2:
        return
    
    u_vals = [d['u_pct'] for d in tracker_data]
    r_vals = [d['r_pct'] for d in tracker_data]
    s_vals = [d['s_pct'] for d in tracker_data]
    
    def normalize(vals):
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return [0.5] * len(vals)
        return [(v - min_v) / (max_v - min_v) for v in vals]
    
    def make_sparkline(vals, char='█'):
        normalized = normalize(vals)
        # Sample to fit width
        step = max(1, len(vals) // width)
        sampled = [normalized[i] for i in range(0, len(vals), step)][:width]
        
        # Convert to heights (0-7)
        blocks = ' ▁▂▃▄▅▆▇█'
        return ''.join(blocks[int(v * 8)] for v in sampled)
    
    print("\n  Component Trends (over training):")
    print(f"    U: {make_sparkline(u_vals)} ({u_vals[0]:.0f}%→{u_vals[-1]:.0f}%)")
    print(f"    R: {make_sparkline(r_vals)} ({r_vals[0]:.0f}%→{r_vals[-1]:.0f}%)")
    print(f"    S: {make_sparkline(s_vals)} ({s_vals[0]:.0f}%→{s_vals[-1]:.0f}%)")

# Usage (after training):
# print_trend_ascii(tracker.data)