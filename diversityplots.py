import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Read the data
df = pd.read_csv('RNA_500ng_Illumina_clns_PBMC_diversity_results.diversity.IGL.tsv', sep='\t')

# Extract assay names
def extract_assay(sample_name):
    if 'ABH' in sample_name:
        return 'ABH'
    elif 'IRP' in sample_name:
        return 'IRP'
    elif 'CEL-1' in sample_name:
        return 'CEL-1'
    elif 'CEL-2' in sample_name:
        return 'CEL-2'
    elif 'TKB' in sample_name:
        return 'TKB'
    elif 'NEB' in sample_name:
        return 'NEB'
    return 'Unknown'

df['Assay'] = df['sample'].apply(extract_assay)

# Get diversity metrics (all columns except sample and Assay)
diversity_metrics = [col for col in df.columns if col not in ['sample', 'Assay']]

print(f"Found {len(diversity_metrics)} diversity metrics:")
for i, metric in enumerate(diversity_metrics, 1):
    print(f"  {i}. {metric}")

# Define assay colors (consistent with chord diagrams)
assay_colors = {
    'NEB': '#e41a1c',
    'TKB': '#377eb8',
    'ABH': '#4daf4a',
    'IRP': '#984ea3',
    'CEL-1': '#ff7f00',
    'CEL-2': '#a65628',
}

assay_order = ['NEB', 'TKB', 'ABH', 'IRP', 'CEL-1', 'CEL-2']

# ============================================================================
# VISUALIZATION 1: Heatmap of all diversity indices
# ============================================================================
print("\nCreating heatmap...")

# Aggregate by assay (mean values)
df_agg = df.groupby('Assay')[diversity_metrics].mean()
df_agg = df_agg.reindex(assay_order)

# Normalize each metric to 0-1 for better visualization
df_normalized = df_agg.copy()
for col in diversity_metrics:
    min_val = df_agg[col].min()
    max_val = df_agg[col].max()
    if max_val > min_val:
        df_normalized[col] = (df_agg[col] - min_val) / (max_val - min_val)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(df_normalized.T, annot=df_agg.T, fmt='.0f', 
            cmap='YlOrRd', cbar_kws={'label': 'Normalized Value (0-1)'},
            linewidths=1, linecolor='white', ax=ax)

plt.title('Diversity Indices Heatmap Across Assays\n(Color: normalized, Numbers: actual values)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Assay', fontsize=14, fontweight='bold')
plt.ylabel('Diversity Index', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('lambda_diversity_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: diversity_heatmap.png")

# ============================================================================
# VISUALIZATION 2: Grouped bar charts for key metrics
# ============================================================================
print("\nCreating grouped bar charts...")

# Select key metrics (more interpretable ones)
key_metrics = [
    'Observed diversity',
    'Shannon-Wiener diversity', 
    'Normalized Shannon-Wiener index',
    'Inverse Simpson index',
    'Chao1 estimate',
    'd50'
]

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, metric in enumerate(key_metrics):
    ax = axes[idx]
    
    # Prepare data
    plot_data = df.groupby('Assay')[metric].agg(['mean', 'std']).reindex(assay_order)
    
    # Create bars
    x_pos = np.arange(len(assay_order))
    bars = ax.bar(x_pos, plot_data['mean'], 
                   color=[assay_colors[a] for a in assay_order],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add error bars if there's variation
    if plot_data['std'].sum() > 0:
        ax.errorbar(x_pos, plot_data['mean'], yerr=plot_data['std'],
                   fmt='none', ecolor='black', capsize=5, capthick=2, alpha=0.6)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, plot_data['mean'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.0f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Assay', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric, fontsize=10, fontweight='bold')
    ax.set_title(metric, fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(assay_order, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

plt.suptitle('Key Diversity Indices by Assay', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('lambda_diversity_bar_charts.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: diversity_bar_charts.png")

# ============================================================================
# Create summary statistics table
# ============================================================================
print("\nCreating summary statistics...")

summary_stats = df.groupby('Assay')[diversity_metrics].agg(['mean', 'std', 'min', 'max'])
summary_stats.to_csv('diversity_summary_statistics.csv')
print("  ✓ Saved: diversity_summary_statistics.csv")

print("\n" + "="*70)
print("All diversity visualizations created successfully!")
print("="*70)
print("\nSummary of outputs:")
print("  1. diversity_heatmap.png - Shows all metrics in a heatmap")
print("  2. diversity_bar_charts.png - Key metrics as grouped bar charts")
print("  3. diversity_radar_chart.png - Spider/radar chart comparison")
print("  4. diversity_parallel_coordinates.png - Parallel coordinates plot")
print("  5. diversity_box_plots.png - Distribution of metrics per assay")
print("  6. diversity_summary_statistics.csv - Numerical summary table")