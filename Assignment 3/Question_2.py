import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def create_true_gmm():
    
    means = [
        np.array([-2, 0]),      
        np.array([1, 1]),       
        np.array([6, 6]),       
        np.array([9, 2])        
    ]
    
    covariances = [
        np.array([[1.2, 0.3], [0.3, 1.5]]),   
        np.array([[1.5, 0.5], [0.5, 1.3]]),   
        np.array([[0.8, 0.1], [0.1, 0.8]]),   
        np.array([[0.6, -0.2], [-0.2, 0.7]])  
    ]
    
    weights = np.array([0.30, 0.25, 0.25, 0.20])  
    
    return means, covariances, weights

def generate_gmm_data(n_samples, means, covariances, weights):
    
    n_components = len(means)
    
    component_assignments = np.random.choice(n_components, size=n_samples, p=weights)
    
    samples = np.zeros((n_samples, 2))
    for i in range(n_samples):
        component = component_assignments[i]
        samples[i] = np.random.multivariate_normal(means[component], covariances[component])
    
    return samples, component_assignments

def cross_validate_gmm(data, max_components=10, n_folds=10):
    
    n_samples = len(data)
    cv_scores = np.zeros(max_components)
    
    for n_components in range(1, max_components + 1):
        fold_scores = []
        
        actual_folds = min(n_folds, n_samples)
        if actual_folds < 2:
            actual_folds = 2
        
        kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(data):
            train_data = data[train_idx]
            val_data = data[val_idx]
            
            try:
                gmm = GaussianMixture(n_components=n_components, 
                                    covariance_type='full',
                                    max_iter=200,
                                    n_init=5,
                                    random_state=42)
                gmm.fit(train_data)
                
                score = gmm.score(val_data) * len(val_data)  
                fold_scores.append(score)
            except:
                fold_scores.append(-np.inf)
        
        cv_scores[n_components - 1] = np.mean(fold_scores)
    
    best_order = np.argmax(cv_scores) + 1
    
    return cv_scores, best_order

def run_experiments(n_experiments=100, sample_sizes=[10, 100, 1000]):
    
    means, covariances, weights = create_true_gmm()
    true_order = len(means)
    
    results = {size: [] for size in sample_sizes}
    cv_profiles = {size: [] for size in sample_sizes}
    
    for exp in range(n_experiments):
        if (exp + 1) % 10 == 0:
            print(f"Experiment {exp + 1}/{n_experiments}")
        
        for n_samples in sample_sizes:
            data, _ = generate_gmm_data(n_samples, means, covariances, weights)
            
            cv_scores, best_order = cross_validate_gmm(data, max_components=10, n_folds=10)
            
            results[n_samples].append(best_order)
            cv_profiles[n_samples].append(cv_scores)
    
    return results, cv_profiles, means, covariances, weights

def plot_true_gmm(means, covariances, weights, n_samples=1500):
   
    data, labels = generate_gmm_data(n_samples, means, covariances, weights)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red', 'blue', 'green', 'purple']
    labels_text = ['Component 0 (π=0.30)', 'Component 1 (π=0.25)', 
                   'Component 2 (π=0.25)', 'Component 3 (π=0.20)']
    
    for i in range(4):
        mask = labels == i
        ax.scatter(data[mask, 0], data[mask, 1], c=colors[i], 
                  alpha=0.6, s=20, label=labels_text[i])
    
    ax.set_xlabel('X₁', fontsize=12)
    ax.set_ylabel('X₂', fontsize=12)
    ax.set_title('True GMM: 4 Components (Components 0 & 1 Overlap)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('true_gmm.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_cv_profiles(cv_profiles, results, sample_sizes):
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, n_samples in enumerate(sample_sizes):
        ax = axes[idx]
        
        avg_cv_scores = np.mean(cv_profiles[n_samples], axis=0)
        std_cv_scores = np.std(cv_profiles[n_samples], axis=0)
        
        most_selected = max(set(results[n_samples]), key=results[n_samples].count)
        
        x = np.arange(1, 11)
        ax.plot(x, avg_cv_scores, 'o-', color='#1f77b4', linewidth=2, 
               markersize=6, label='CV Score')
        
        ax.plot(most_selected, avg_cv_scores[most_selected - 1], 
               marker='*', markersize=20, color='red', 
               label=f'Selected: C={most_selected}')
        
        ax.axvline(x=4, color='green', linestyle='--', linewidth=2, 
                  label='True Order (C=4)')
        
        ax.text(0.05, 0.95, f'Max C={most_selected}', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               verticalalignment='top')
        
        ax.set_xlabel('Number of GMM Components (C)', fontsize=10)
        ax.set_ylabel('Avg Validation Log-Likelihood', fontsize=10)
        ax.set_title(f'N = {n_samples} samples', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.suptitle('2.4.2 Cross-Validation Score Profiles', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('cv_profiles.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_selection_frequency(results, sample_sizes, n_experiments):
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, n_samples in enumerate(sample_sizes):
        ax = axes[idx]
        
        unique, counts = np.unique(results[n_samples], return_counts=True)
        frequencies = (counts / n_experiments) * 100
        
        most_selected_idx = np.argmax(counts)
        most_selected = unique[most_selected_idx]
        most_selected_freq = frequencies[most_selected_idx]
        
        colors = ['#1f77b4' if c == 1 else '#ff7f0e' if c == 3 else '#d62728' 
                 if c == 4 else '#7f7f7f' for c in unique]
        
        bars = ax.bar(unique, frequencies, color=colors, alpha=0.8, edgecolor='black')
        
        for i, (u, f) in enumerate(zip(unique, frequencies)):
            if f > 3:  
                ax.text(u, f + 2, f'{f:.1f}%', ha='center', fontsize=9)
        
        ax.text(0.98, 0.95, f'Most selected: C={most_selected}\n({most_selected_freq:.1f}%)', 
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               verticalalignment='top', horizontalalignment='right')
        
        ax.set_xlabel('Number of GMM Components', fontsize=10)
        ax.set_ylabel('Selection Frequency (%)', fontsize=10)
        ax.set_title(f'N = {n_samples} samples\n(100 experiments)', fontsize=11, fontweight='bold')
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Figure 4: Selection Frequency Bar Plots (100 experiments per dataset size)', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('selection_frequency.png', dpi=150, bbox_inches='tight')
    plt.show()

def gmm_pdf(x, y, means, covariances, weights):
    
    pdf = np.zeros_like(x)
    
    for i, (mean, cov, weight) in enumerate(zip(means, covariances, weights)):
        diff = np.stack([x - mean[0], y - mean[1]], axis=-1)
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        
        mahalanobis = np.sum(diff @ cov_inv * diff, axis=-1)
        component_pdf = (1 / (2 * np.pi * np.sqrt(cov_det))) * np.exp(-0.5 * mahalanobis)
        
        pdf += weight * component_pdf
    
    return pdf

def plot_gmm_heatmap(means, covariances, weights):
    
    x = np.linspace(-6, 12, 300)
    y = np.linspace(-4, 10, 300)
    X, Y = np.meshgrid(x, y)
    
    Z = gmm_pdf(X, Y, means, covariances, weights)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    im = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
    
    contours = ax.contour(X, Y, Z, levels=10, colors='white', alpha=0.3, linewidths=1)
    
    colors = ['red', 'blue', 'green', 'purple']
    labels = ['Component 0 (π=0.30)', 'Component 1 (π=0.25)', 
             'Component 2 (π=0.25)', 'Component 3 (π=0.20)']
    
    for i, (mean, label) in enumerate(zip(means, labels)):
        ax.plot(mean[0], mean[1], 'o', color=colors[i], markersize=12, 
               markeredgecolor='white', markeredgewidth=2, label=label)
        ax.text(mean[0], mean[1] + 0.5, f'C{i}', fontsize=10, 
               color='white', fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.7))
    
    overlap_x = (means[0][0] + means[1][0]) / 2
    overlap_y = (means[0][1] + means[1][1]) / 2
    ax.plot([means[0][0], means[1][0]], [means[0][1], means[1][1]], 
           'w--', linewidth=2, alpha=0.7)
    ax.annotate('Overlap Region', xy=(overlap_x, overlap_y), 
               xytext=(overlap_x, overlap_y - 2),
               fontsize=10, color='yellow', fontweight='bold',
               ha='center',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.6),
               arrowprops=dict(arrowstyle='->', color='yellow', lw=2))
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability Density', fontsize=11)
    
    ax.set_xlabel('X₁', fontsize=12)
    ax.set_ylabel('X₂', fontsize=12)
    ax.set_title('True GMM Probability Density Heatmap\n4 Components (Components 0 & 1 Overlap)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2, color='white')
    
    plt.tight_layout()
    plt.savefig('gmm_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_selection_heatmap(results, n_experiments):
    
    sample_sizes = [10, 100, 1000]
    max_components = 10
    
    freq_matrix = np.zeros((len(sample_sizes), max_components))
    
    for i, n_samples in enumerate(sample_sizes):
        for c in range(1, max_components + 1):
            count = results[n_samples].count(c)
            freq_matrix[i, c-1] = (count / n_experiments) * 100
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    im = ax.imshow(freq_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    
    for i in range(len(sample_sizes)):
        for j in range(max_components):
            value = freq_matrix[i, j]
            if value > 0:
                text = ax.text(j, i, f'{value:.1f}%',
                             ha="center", va="center", 
                             color="black" if value < 50 else "white",
                             fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Selection Frequency (%)', fontsize=11)
    
    ax.set_xticks(np.arange(max_components))
    ax.set_yticks(np.arange(len(sample_sizes)))
    ax.set_xticklabels([f'{i+1}' for i in range(max_components)])
    ax.set_yticklabels([f'N={n}' for n in sample_sizes])
    
    ax.set_xlabel('Number of GMM Components (C)', fontsize=12)
    ax.set_ylabel('Dataset Size', fontsize=12)
    ax.set_title('Model Order Selection Frequency Heatmap\n(100 Experiments per Dataset Size)', 
                fontsize=14, fontweight='bold')
    
    ax.set_xticks(np.arange(max_components+1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(sample_sizes)+1)-0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    
    rect = plt.Rectangle((3-0.5, -0.5), 1, len(sample_sizes), 
                         fill=False, edgecolor='lime', linewidth=3, linestyle='--')
    ax.add_patch(rect)
    ax.text(3, len(sample_sizes), 'True Order', ha='center', va='bottom',
           fontsize=10, color='lime', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('selection_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting GMM Model Order Selection Experiment...")
    print("=" * 60)
    
    sample_sizes = [10, 100, 1000]
    n_experiments = 100
    
    print(f"\nRunning {n_experiments} experiments for each sample size...")
    results, cv_profiles, means, covariances, weights = run_experiments(
        n_experiments=n_experiments, 
        sample_sizes=sample_sizes
    )
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for n_samples in sample_sizes:
        unique, counts = np.unique(results[n_samples], return_counts=True)
        frequencies = (counts / n_experiments) * 100
        
        print(f"\nN = {n_samples} samples:")
        print("-" * 40)
        for u, f in zip(unique, frequencies):
            print(f"  C={u}: {f:.1f}% ({int(f)} out of {n_experiments})")
        
        most_selected = unique[np.argmax(counts)]
        print(f"  → Most selected: C={most_selected}")
    
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    
    plot_true_gmm(means, covariances, weights)
    plot_gmm_heatmap(means, covariances, weights)  # NEW: GMM density heatmap
    plot_cv_profiles(cv_profiles, results, sample_sizes)
    plot_selection_frequency(results, sample_sizes, n_experiments)
    plot_selection_heatmap(results, n_experiments)  # NEW: Selection frequency heatmap
    
    print("\nAll visualizations saved!")
    print("  - true_gmm.png")
    print("  - gmm_heatmap.png")
    print("  - cv_profiles.png")
    print("  - selection_frequency.png")
    print("  - selection_heatmap.png")
