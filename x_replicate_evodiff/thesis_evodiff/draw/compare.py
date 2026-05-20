import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import os

# 1. Base Configuration (Academic English Fonts)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "text.usetex": False,
})

# 2. Definitions (Matched to your previous table terminology)
methods = [
    'Baseline\n(DPM-Solver++)', 
    'Variant A\n($\\zeta$-only)', 
    'Variant B\n($\\eta$-only)', 
    'Full EVODiff\n(Ours)'
]
fids = ['FID: 8.71', 'FID: 8.24', 'FID: 7.36', 'FID: 6.92']

# Folder where your images are stored (relative to this script)
image_dir = os.path.join(os.path.dirname(__file__), 'images')

# Image filenames 
image_files = [
    ['b1.png', 'a1.png', 'v1.png', 'f1.png'], # Row 1 images
    ['b2.png', 'a2.png', 'v2.png', 'f2.png']  # Row 2 images
]

# 3. Create 2x4 Figure
fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

for row in range(2):
    for col in range(4):
        ax = axes[row, col]
        
        # Construct full path safely
        img_name = image_files[row][col]
        img_path = os.path.join(image_dir, img_name)
        
        # Read and display image OR show a placeholder if missing
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img)
        else:
            # Placeholder logic so the script doesn't crash
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor='#ecf0f1', edgecolor='#bdc3c7'))
            ax.text(0.5, 0.5, f'Missing:\n{img_name}', ha='center', va='center', color='#7f8c8d')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        ax.axis('off')
        
        # Add Method Titles to Row 0
        if row == 0:
            ax.set_title(methods[col], fontsize=13, fontweight='bold', pad=10)
        
        # Add FID scores below Row 1
        if row == 1:
            # Using a dark red color (#c0392b) which looks more professional in papers
            ax.text(0.5, -0.15, fids[col], transform=ax.transAxes, 
                    ha='center', fontsize=12, color='#c0392b', fontweight='bold')

# 4. Save
output_dir = "thesis_evodiff/results/figures"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'figure_4_4_quality_comparison.png')

plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Figure generated and saved to:\n  {save_path}")