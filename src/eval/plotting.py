"""
Publication-Quality Plotting for Deep Hedging.

Generates all required figures:
- P&L histogram and tail CDF
- Boxplots across strategies
- Cumulative P&L paths
- No-transaction band visualization
- Learning curves
- Hedge vs delta paths (sample trajectories)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

COLORS = {
    'deep_hedging': '#1f77b4',
    'kozyra_rnn': '#ff7f0e',
    'kozyra_lstm': '#2ca02c',
    'bs_delta': '#d62728',
    'leland': '#9467bd',
    'whalley_wilmott': '#8c564b',
    'transformer': '#e377c2',
    'signature': '#7f7f7f',
    'no_hedge': '#bcbd22'
}


def plot_pnl_histogram(
    pnl_dict: Dict[str, np.ndarray],
    title: str = "P&L Distribution",
    save_path: Optional[str] = None,
    show_stats: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot P&L histograms for multiple strategies.
    
    Args:
        pnl_dict: Dictionary mapping strategy names to P&L arrays
        title: Plot title
        save_path: Path to save figure
        show_stats: Whether to show statistics in legend
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (name, pnl) in enumerate(pnl_dict.items()):
        color = list(COLORS.values())[i % len(COLORS)]
        
        label = name
        if show_stats:
            label += f" (μ={np.mean(pnl):.2f}, σ={np.std(pnl):.2f})"
        
        ax.hist(pnl, bins=50, alpha=0.5, label=label, color=color, density=True)
        
        # Add KDE
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(pnl)
            x_range = np.linspace(pnl.min(), pnl.max(), 200)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2)
        except:
            pass
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Zero P&L')
    
    ax.set_xlabel('P&L')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_tail_cdf(
    pnl_dict: Dict[str, np.ndarray],
    title: str = "Tail CDF (Losses)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot tail CDF focusing on losses (left tail of P&L).
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (name, pnl) in enumerate(pnl_dict.items()):
        color = list(COLORS.values())[i % len(COLORS)]
        losses = -pnl
        sorted_losses = np.sort(losses)
        cdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
        
        # Focus on tail (top 10% of losses)
        tail_idx = int(0.9 * len(sorted_losses))
        ax.plot(sorted_losses[tail_idx:], cdf[tail_idx:], 
                label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Loss')
    ax.set_ylabel('CDF')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_pnl_boxplot(
    pnl_dict: Dict[str, np.ndarray],
    title: str = "P&L Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create boxplots comparing P&L across strategies.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(pnl_dict.keys())
    data = [pnl_dict[name] for name in names]
    colors = [list(COLORS.values())[i % len(COLORS)] for i in range(len(names))]
    
    bp = ax.boxplot(data, labels=names, patch_artist=True, 
                    showfliers=True, flierprops={'markersize': 3})
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax.set_ylabel('P&L')
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    
    # Add mean markers
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(names) + 1), means, marker='D', 
               color='red', s=50, zorder=5, label='Mean')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_learning_curves(
    history: Dict[str, List[float]],
    title: str = "Learning Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot training and validation learning curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss curves
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train', color='blue')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation', color='orange')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # P&L metrics
    ax2 = axes[1]
    if 'val_mean_pnl' in history:
        ax2.plot(history['val_mean_pnl'], label='Mean P&L', color='green')
    if 'val_std_pnl' in history:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(history['val_std_pnl'], label='Std P&L', 
                      color='red', linestyle='--')
        ax2_twin.set_ylabel('Std P&L', color='red')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean P&L', color='green')
    ax2.set_title('P&L Metrics')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_delta_paths(
    stock_paths: np.ndarray,
    delta_dict: Dict[str, np.ndarray],
    n_paths: int = 5,
    title: str = "Hedging Strategy Paths",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot sample hedging paths for multiple strategies.
    
    Args:
        stock_paths: Stock price paths (n_paths, n_steps + 1)
        delta_dict: Dictionary mapping strategy names to delta arrays
        n_paths: Number of sample paths to plot
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    n_total = stock_paths.shape[0]
    indices = np.random.choice(n_total, min(n_paths, n_total), replace=False)
    
    # Plot stock paths
    ax1 = axes[0]
    for idx in indices:
        ax1.plot(stock_paths[idx], alpha=0.7, linewidth=1)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Stock Price')
    ax1.set_title('Stock Price Paths')
    ax1.grid(True, alpha=0.3)
    
    # Plot delta paths
    ax2 = axes[1]
    colors = list(COLORS.values())
    
    for i, (name, deltas) in enumerate(delta_dict.items()):
        color = colors[i % len(colors)]
        for j, idx in enumerate(indices):
            label = name if j == 0 else None
            ax2.plot(deltas[idx], color=color, alpha=0.5, 
                    linewidth=1, label=label)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Delta (Hedge Position)')
    ax2.set_title('Hedging Positions')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_cumulative_pnl(
    pnl_dict: Dict[str, np.ndarray],
    n_paths: int = 100,
    title: str = "Cumulative P&L Paths",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot cumulative P&L evolution (for sequential view).
    
    Note: This shows the distribution of terminal P&L as a cumulative sum
    of randomly ordered P&Ls for visualization.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (name, pnl) in enumerate(pnl_dict.items()):
        color = list(COLORS.values())[i % len(COLORS)]
        
        # Sort P&L for cumulative view
        sorted_pnl = np.sort(pnl)
        cumsum = np.cumsum(sorted_pnl) / np.arange(1, len(sorted_pnl) + 1)
        
        ax.plot(cumsum, label=name, color=color, linewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Sample Index (sorted)')
    ax.set_ylabel('Running Average P&L')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_no_transaction_band(
    stock_paths: np.ndarray,
    bs_deltas: np.ndarray,
    ww_deltas: np.ndarray,
    band_width: float,
    n_paths: int = 3,
    title: str = "No-Transaction Band Strategy",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Visualize Whalley-Wilmott no-transaction band.
    
    Args:
        stock_paths: Stock prices
        bs_deltas: Black-Scholes deltas
        ww_deltas: Whalley-Wilmott deltas
        band_width: Half-width of no-transaction band
        n_paths: Number of paths to show
    """
    fig, axes = plt.subplots(n_paths, 1, figsize=figsize, sharex=True)
    
    if n_paths == 1:
        axes = [axes]
    
    n_total = stock_paths.shape[0]
    indices = np.random.choice(n_total, min(n_paths, n_total), replace=False)
    
    for ax, idx in zip(axes, indices):
        n_steps = bs_deltas.shape[1]
        time = np.arange(n_steps)
        
        # Plot BS delta
        ax.plot(time, bs_deltas[idx], 'b-', label='BS Delta', linewidth=2)
        
        # Plot band
        ax.fill_between(time, 
                       bs_deltas[idx] - band_width,
                       bs_deltas[idx] + band_width,
                       alpha=0.2, color='blue', label='No-Trade Band')
        
        # Plot WW strategy
        ax.plot(time, ww_deltas[idx], 'r-', label='WW Delta', 
                linewidth=2, marker='o', markersize=3)
        
        ax.set_ylabel('Delta')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Step')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def plot_metrics_comparison(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = "Strategy Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create bar chart comparing metrics across strategies.
    """
    if metrics is None:
        metrics = ['mean_pnl', 'std_pnl', 'var_95', 'cvar_95', 'entropic_risk']
    
    n_metrics = len(metrics)
    n_strategies = len(results_dict)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    strategies = list(results_dict.keys())
    colors = [list(COLORS.values())[i % len(COLORS)] for i in range(n_strategies)]
    
    for ax, metric in zip(axes, metrics):
        values = [results_dict[s].get(metric, 0) for s in strategies]
        
        bars = ax.bar(range(n_strategies), values, color=colors)
        ax.set_xticks(range(n_strategies))
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


def create_results_table(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    bootstrap_ci: Dict[str, Dict] = None
) -> str:
    """
    Create a LaTeX-ready results table.
    """
    if metrics is None:
        metrics = ['mean_pnl', 'std_pnl', 'var_95', 'cvar_95', 'entropic_risk']
    
    strategies = list(results_dict.keys())
    
    # Header
    header = "Strategy & " + " & ".join([m.replace('_', ' ') for m in metrics]) + " \\\\"
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{l" + "c" * len(metrics) + "}",
        "\\toprule",
        header,
        "\\midrule"
    ]
    
    # Data rows
    for strategy in strategies:
        row = strategy.replace('_', ' ')
        for metric in metrics:
            val = results_dict[strategy].get(metric, float('nan'))
            row += f" & {val:.4f}"
        row += " \\\\"
        lines.append(row)
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Hedging Strategy Comparison}",
        "\\label{tab:results}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def save_all_figures(
    pnl_dict: Dict[str, np.ndarray],
    delta_dict: Dict[str, np.ndarray],
    stock_paths: np.ndarray,
    history: Dict[str, List[float]],
    save_dir: str
) -> None:
    """
    Generate and save all publication figures.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # P&L histogram
    plot_pnl_histogram(pnl_dict, save_path=str(save_path / "pnl_histogram.pdf"))
    
    # Tail CDF
    plot_tail_cdf(pnl_dict, save_path=str(save_path / "tail_cdf.pdf"))
    
    # Boxplot
    plot_pnl_boxplot(pnl_dict, save_path=str(save_path / "pnl_boxplot.pdf"))
    
    # Learning curves
    if history:
        plot_learning_curves(history, save_path=str(save_path / "learning_curves.pdf"))
    
    # Delta paths
    plot_delta_paths(stock_paths, delta_dict, 
                    save_path=str(save_path / "delta_paths.pdf"))
    
    # Cumulative P&L
    plot_cumulative_pnl(pnl_dict, save_path=str(save_path / "cumulative_pnl.pdf"))
    
    print(f"All figures saved to {save_path}")
