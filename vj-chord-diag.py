import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.path import Path
import matplotlib.patches as mpatches

# Read the data
df = pd.read_csv('RNA_500ng_Illumina_clns_PBMC_diversity_results.vjFamilyUsage.IGL.tsv', sep='\t')

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

# Get all unique genes
all_v_genes = sorted(df['VGene'].unique())
all_j_genes = sorted(df['JGene'].unique())

# Define assay colors
assay_colors = {
    'ABH': '#e41a1c',  # Red
    'IRP': '#377eb8',  # Blue
    'CEL-1': '#4daf4a',  # Green
    'CEL-2': '#984ea3',  # Purple
    'TKB': '#ff7f00',  # Orange
    'NEB': '#ffff33',  # Yellow
}

assays = sorted(df['Assay'].unique())

# Aggregate data for all assays
all_data = df.groupby(['Assay', 'VGene', 'JGene'])['Value'].mean().reset_index()

# Calculate total values for sizing arcs
v_totals = df.groupby('VGene')['Value'].sum()
j_totals = df.groupby('JGene')['Value'].sum()

# Create figure
fig, ax = plt.subplots(figsize=(18, 18))
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axis('off')

# Parameters
radius = 1.0
gap = 0.03

# Calculate angles for V genes (top half)
v_total = sum(v_totals)
v_angles = {}
current_angle = np.pi
for gene in all_v_genes:
    total = v_totals[gene]
    proportion = total / v_total if v_total > 0 else 0
    angle_size = (np.pi - gap * len(all_v_genes)) * proportion
    v_angles[gene] = (current_angle, current_angle + angle_size)
    current_angle += angle_size + gap

# Calculate angles for J genes (bottom half)
j_total = sum(j_totals)
j_angles = {}
current_angle = 0
for gene in all_j_genes:
    total = j_totals[gene]
    proportion = total / j_total if j_total > 0 else 0
    angle_size = (np.pi - gap * len(all_j_genes)) * proportion
    j_angles[gene] = (current_angle, current_angle + angle_size)
    current_angle += angle_size + gap

# Draw V gene arcs (in gray)
v_color = '#cccccc'
for gene in all_v_genes:
    start, end = v_angles[gene]
    wedge = Wedge((0, 0), radius, np.degrees(start), np.degrees(end),
                  width=0.1, facecolor=v_color, edgecolor='white', linewidth=2)
    ax.add_patch(wedge)
    
    # Add label
    mid_angle = (start + end) / 2
    label_radius = radius + 0.15
    x = label_radius * np.cos(mid_angle)
    y = label_radius * np.sin(mid_angle)
    rotation = np.degrees(mid_angle)
    if rotation > 90:
        rotation = rotation - 180
    ax.text(x, y, gene, ha='center', va='center', fontsize=12, 
            fontweight='bold', rotation=rotation)

# Draw J gene arcs (in gray)
j_color = '#999999'
for gene in all_j_genes:
    start, end = j_angles[gene]
    wedge = Wedge((0, 0), radius, np.degrees(start), np.degrees(end),
                  width=0.1, facecolor=j_color, edgecolor='white', linewidth=2)
    ax.add_patch(wedge)
    
    # Add label
    mid_angle = (start + end) / 2
    label_radius = radius + 0.15
    x = label_radius * np.cos(mid_angle)
    y = label_radius * np.sin(mid_angle)
    rotation = np.degrees(mid_angle)
    ax.text(x, y, gene, ha='center', va='center', fontsize=12,
            fontweight='bold', rotation=rotation)

# Draw ribbons for each assay with different colors
for assay in assays:
    assay_data = all_data[all_data['Assay'] == assay]
    color = assay_colors[assay]
    
    for _, row in assay_data.iterrows():
        v_gene = row['VGene']
        j_gene = row['JGene']
        value = row['Value']
        
        if value > 0.001:  # Threshold
            # Get angles
            v_start, v_end = v_angles[v_gene]
            j_start, j_end = j_angles[j_gene]
            
            v_mid = (v_start + v_end) / 2
            j_mid = (j_start + j_end) / 2
            
            # Inner radius for ribbons
            inner_r = radius - 0.1
            
            # Start and end points
            v_x = inner_r * np.cos(v_mid)
            v_y = inner_r * np.sin(v_mid)
            j_x = inner_r * np.cos(j_mid)
            j_y = inner_r * np.sin(j_mid)
            
            # Create bezier curve
            verts = [
                (v_x, v_y),
                (v_x * 0.3, v_y * 0.3),
                (j_x * 0.3, j_y * 0.3),
                (j_x, j_y)
            ]
            
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            
            # Color and alpha based on value
            alpha = min(0.5, value * 3)
            patch = mpatches.PathPatch(path, facecolor='none', 
                                      edgecolor=color, 
                                      alpha=alpha, linewidth=max(0.5, value * 25))
            ax.add_patch(patch)

# Add title
plt.title('V-J Gene Usage Across All Assays', fontsize=22, fontweight='bold', pad=30)

# Create legend for assays
legend_elements = [mpatches.Patch(facecolor=assay_colors[assay], 
                                 edgecolor='black', 
                                 label=assay) for assay in assays]
plt.legend(handles=legend_elements, loc='upper right', fontsize=14, 
          title='Assays', title_fontsize=16, framealpha=0.9)

# Add info text
info_text = (f"V Genes (top, gray arcs): {', '.join(all_v_genes)}\n"
            f"J Genes (bottom, dark gray arcs): {', '.join(all_j_genes)}\n"
            f"Ribbons colored by assay | Width = usage strength")
plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Save
output_file = 'chord_diagram_all_assays_combined_lambdachain.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Combined chord diagram saved: {output_file}")
plt.close()