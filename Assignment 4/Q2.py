import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from skimage import io
from skimage.transform import resize
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2024)

print("Loading image from Berkeley dataset...")
img = io.imread('https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/images/plain/normal/color/108073.jpg')

if img.shape[0] * img.shape[1] > 60000:
    scale = np.sqrt(60000 / (img.shape[0] * img.shape[1]))
    new_h = int(img.shape[0] * scale)
    new_w = int(img.shape[1] * scale)
    img = resize(img, (new_h, new_w), anti_aliasing=True)
    img = (img * 255).astype(np.uint8)
    print(f"Resized image to: {new_h}x{new_w}")

h, w = img.shape[:2]
print(f"Image size: {h}x{w} = {h*w} pixels")

print("\nExtracting 5D features [row, col, R, G, B]...")
features = []
for i in range(h):
    for j in range(w):
        features.append([i, j, img[i, j, 0], img[i, j, 1], img[i, j, 2]])

features = np.array(features, dtype=float)
print(f"Feature matrix shape: {features.shape}")

print("\nNormalizing features to [0, 1]...")
features[:, 0] = features[:, 0] / (h - 1)
features[:, 1] = features[:, 1] / (w - 1)
features[:, 2] = features[:, 2] / 255.0
features[:, 3] = features[:, 3] / 255.0
features[:, 4] = features[:, 4] / 255.0
print(f"Normalized range: [{features.min():.3f}, {features.max():.3f}]")

k_values = [2, 3, 4, 5, 6, 8, 10]
n_folds = 10

print(f"\nPerforming {n_folds}-fold cross-validation...")
print(f"Testing K values: {k_values}")

results = {}

for k in k_values:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(features):
        gmm = GaussianMixture(n_components=k, max_iter=100, random_state=42)
        gmm.fit(features[train_idx])
        scores.append(gmm.score(features[val_idx]))
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    results[k] = (mean_score, std_score)
    print(f"K={k:2d} | Val Log-Likelihood: {mean_score:7.4f} ± {std_score:.4f}")

best_k = max(results, key=lambda x: results[x][0])
best_score = results[best_k][0]

print(f"\nBest K: {best_k} components")
print(f"Best log-likelihood: {best_score:.4f}")

print(f"\nTraining final GMM with K={best_k}...")
gmm_final = GaussianMixture(n_components=best_k, max_iter=200, random_state=42)
gmm_final.fit(features)

print(f"GMM converged: {gmm_final.converged_} (iterations: {gmm_final.n_iter_})")

labels = gmm_final.predict(features).reshape(h, w)

print(f"\nSegmentation complete:")
for i in range(best_k):
    count = np.sum(labels == i)
    pct = 100 * count / len(features)
    print(f"  Segment {i}: {count:6d} pixels ({pct:5.1f}%)")

fig = plt.figure(figsize=(14, 9))

plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
k_list = sorted(results.keys())
means = [results[k][0] for k in k_list]
stds = [results[k][1] for k in k_list]
plt.plot(k_list, means, 'o-', linewidth=2)
plt.fill_between(k_list, np.array(means) - np.array(stds), 
                 np.array(means) + np.array(stds), alpha=0.2)
plt.plot(best_k, best_score, 'r*', markersize=18)
plt.xlabel('K')
plt.ylabel('Log-likelihood')
plt.title('K-Fold CV')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
unique_labels = np.unique(labels)
display = np.zeros_like(labels, dtype=float)
for i, l in enumerate(unique_labels):
    display[labels == l] = i / (len(unique_labels) - 1)
plt.imshow(display, cmap='tab20')
plt.title(f'Segmentation K={best_k}')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img)
plt.title(f'{h}x{w}')
plt.axis('off')

plt.subplot(2, 3, 5)
colors = plt.cm.tab20(np.linspace(0, 1, best_k))
colored = np.zeros((h, w, 3))
for i in range(best_k):
    colored[labels == i] = colors[i, :3]
plt.imshow(colored)
plt.title('Colored Segments')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(display, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

plt.tight_layout()
plt.savefig('q2_results.png', dpi=300, bbox_inches='tight')
print("\nResults saved to q2_results.png")
plt.show()
