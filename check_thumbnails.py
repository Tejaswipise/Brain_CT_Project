import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

THUMBNAIL_DIR = Path('results/thumbnails')

patients = [
    'CT180-01-2026', 'CT183-01-2026',
    'CT20015788', 'CT20015807', 'CT20015886',
    'CT3225-12-2025', 'CT3259-12-2025',
    'CT3274-12-2025', 'CT3277-12-2025', 'CT3289-12-2025'
]

labels = {
    'CT180-01-2026':  'Normal / Revolution',
    'CT183-01-2026':  'Normal / Revolution',
    'CT20015788':     'Normal / BRIVO',
    'CT20015807':     'Normal / BRIVO',
    'CT20015886':     'Normal / BRIVO',
    'CT3225-12-2025': 'Haemorrhage / Revolution',
    'CT3259-12-2025': 'Haemorrhage / Revolution',
    'CT3274-12-2025': 'Haemorrhage / Revolution',
    'CT3277-12-2025': 'Haemorrhage / Revolution',
    'CT3289-12-2025': 'Haemorrhage / Revolution',
}

# Plot all raw thumbnails in one figure
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, pid in enumerate(patients):
    img_path = THUMBNAIL_DIR / f"{pid}_slice.png"
    if img_path.exists():
        img = plt.imread(str(img_path))
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"{pid}\n{labels[pid]}", fontsize=7)
    else:
        axes[i].set_title(f"MISSING\n{pid}", fontsize=7, color='red')
    axes[i].axis('off')

plt.suptitle('All Patient Thumbnails — Visual Check', fontsize=12)
plt.tight_layout()
plt.savefig('results/thumbnails/visual_check_raw.png',
            dpi=100, bbox_inches='tight')
plt.close()
print("Saved: results/thumbnails/visual_check_raw.png")

# Plot masked vs raw for one Normal and one Haemorrhage patient
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

pairs = [
    ('CT20015788',     'Normal BRIVO'),
    ('CT3225-12-2025', 'Haemorrhage Revolution'),
]

for row, (pid, label) in enumerate(pairs):
    raw_path    = THUMBNAIL_DIR / f"{pid}_slice.png"
    masked_path = THUMBNAIL_DIR / f"{pid}_masked.png"

    if raw_path.exists():
        axes[row, 0].imshow(plt.imread(str(raw_path)), cmap='gray')
        axes[row, 0].set_title(f'{label}\nRaw slice')
    axes[row, 0].axis('off')

    if masked_path.exists():
        axes[row, 1].imshow(plt.imread(str(masked_path)), cmap='gray')
        axes[row, 1].set_title(f'{label}\nMasked slice')
    axes[row, 1].axis('off')

plt.suptitle('Raw vs Masked — Visual Check', fontsize=12)
plt.tight_layout()
plt.savefig('results/thumbnails/visual_check_masked.png',
            dpi=100, bbox_inches='tight')
plt.close()
print("Saved: results/thumbnails/visual_check_masked.png")

# Average normal
avg_path = THUMBNAIL_DIR / 'average_normal.png'
if avg_path.exists():
    print("average_normal.png exists and is readable")
else:
    print("MISSING: average_normal.png")