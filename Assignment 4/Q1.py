import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2024)

def generate_data(n_samples, r_inner=2, r_outer=4, sigma=1):
    n_per_class = n_samples // 2
    
    theta_inner = np.random.uniform(-np.pi, np.pi, n_per_class)
    x_inner = r_inner * np.cos(theta_inner)
    y_inner = r_inner * np.sin(theta_inner)
    noise_inner = np.random.normal(0, sigma, (n_per_class, 2))
    X_inner = np.column_stack([x_inner, y_inner]) + noise_inner
    y_inner = -np.ones(n_per_class)
    
    theta_outer = np.random.uniform(-np.pi, np.pi, n_per_class)
    x_outer = r_outer * np.cos(theta_outer)
    y_outer = r_outer * np.sin(theta_outer)
    noise_outer = np.random.normal(0, sigma, (n_per_class, 2))
    X_outer = np.column_stack([x_outer, y_outer]) + noise_outer
    y_outer = np.ones(n_per_class)
    
    X = np.vstack([X_inner, X_outer])
    y = np.hstack([y_inner, y_outer])
    
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]

X_train, y_train = generate_data(1000)
X_test, y_test = generate_data(10000)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

def svm_kfold_cv(X, y, C_vals, gamma_vals, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = {}
    
    for C in C_vals:
        for gamma in gamma_vals:
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                
                model = SVC(kernel='rbf', C=C, gamma=gamma)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                scores.append(accuracy_score(y_val, pred))
            
            results[(C, gamma)] = (np.mean(scores), np.std(scores))
    
    return results

C_values = [0.1, 1, 10, 100]
gamma_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

print("\nSVM: Running 10-fold cross-validation...")
svm_results = svm_kfold_cv(X_train, y_train, C_values, gamma_values)

best_params = max(svm_results, key=lambda k: svm_results[k][0])
best_C, best_gamma = best_params
best_acc, best_std = svm_results[best_params]

print(f"Best SVM parameters: C={best_C}, gamma={best_gamma}")
print(f"CV accuracy: {best_acc:.4f} +/- {best_std:.4f}")

svm_model = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_err = 1 - svm_acc

print(f"SVM test accuracy: {svm_acc:.4f}")
print(f"SVM test error: {svm_err:.4f}")

def mlp_kfold_cv(X, y, hidden_sizes, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = {}
    
    for h in hidden_sizes:
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = MLPClassifier(hidden_layer_sizes=(h,), activation='tanh',
                                 solver='adam', max_iter=1000, random_state=42)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, pred))
        
        results[h] = (np.mean(scores), np.std(scores))
    
    return results

hidden_sizes = [8, 16, 24, 32, 40, 50, 64]

print("\nMLP: Running 10-fold cross-validation...")
mlp_results = mlp_kfold_cv(X_train, y_train, hidden_sizes)

best_h = max(mlp_results, key=lambda k: mlp_results[k][0])
best_mlp_acc, best_mlp_std = mlp_results[best_h]

print(f"Best MLP hidden size: {best_h}")
print(f"CV accuracy: {best_mlp_acc:.4f} +/- {best_mlp_std:.4f}")

mlp_model = MLPClassifier(hidden_layer_sizes=(best_h,), activation='tanh',
                          solver='adam', max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)

y_pred_mlp = mlp_model.predict(X_test)
mlp_acc = accuracy_score(y_test, y_pred_mlp)
mlp_err = 1 - mlp_acc

print(f"MLP test accuracy: {mlp_acc:.4f}")
print(f"MLP test error: {mlp_err:.4f}")

fig = plt.figure(figsize=(16, 10))

ax1 = plt.subplot(2, 3, 1)
acc_matrix = np.zeros((len(C_values), len(gamma_values)))
for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):
        acc_matrix[i, j] = svm_results[(C, gamma)][0]

im = ax1.imshow(acc_matrix, cmap='viridis', aspect='auto')
ax1.set_xticks(range(len(gamma_values)))
ax1.set_yticks(range(len(C_values)))
ax1.set_xticklabels(gamma_values)
ax1.set_yticklabels(C_values)
ax1.set_xlabel('gamma')
ax1.set_ylabel('C')
ax1.set_title('SVM K-Fold CV Results')
plt.colorbar(im, ax=ax1)

opt_i = C_values.index(best_C)
opt_j = gamma_values.index(best_gamma)
ax1.plot(opt_j, opt_i, 'r*', markersize=20)

ax2 = plt.subplot(2, 3, 2)
h_list = sorted(mlp_results.keys())
means = [mlp_results[h][0] for h in h_list]
stds = [mlp_results[h][1] for h in h_list]
ax2.errorbar(h_list, means, yerr=stds, marker='o', capsize=3)
ax2.plot(best_h, best_mlp_acc, 'r*', markersize=15)
ax2.set_xlabel('Hidden layer size')
ax2.set_ylabel('Accuracy')
ax2.set_title('MLP K-Fold CV Results')
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 3, 3)
ax3.scatter(X_train[y_train==-1, 0], X_train[y_train==-1, 1], 
           c='blue', s=10, alpha=0.5, label='Class -1')
ax3.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
           c='red', s=10, alpha=0.5, label='Class +1')
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_title('Training Data')
ax3.legend()
ax3.axis('equal')

def plot_boundary(model, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    idx = np.random.choice(len(X), 2000, replace=False)
    ax.scatter(X[idx][y[idx]==-1, 0], X[idx][y[idx]==-1, 1],
              c='blue', s=5, alpha=0.5, label='Class -1')
    ax.scatter(X[idx][y[idx]==1, 0], X[idx][y[idx]==1, 1],
              c='red', s=5, alpha=0.5, label='Class +1')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.legend()
    ax.axis('equal')

ax4 = plt.subplot(2, 3, 4)
plot_boundary(svm_model, X_test, y_test, ax4, 
             f'SVM Decision Boundary\nTest Error: {svm_err:.4f}')

ax5 = plt.subplot(2, 3, 5)
plot_boundary(mlp_model, X_test, y_test, ax5,
             f'MLP Decision Boundary\nTest Error: {mlp_err:.4f}')

ax6 = plt.subplot(2, 3, 6)
models = ['SVM', 'MLP']
errors = [svm_err, mlp_err]
bars = ax6.bar(models, errors, color=['steelblue', 'seagreen'], alpha=0.7)
ax6.set_ylabel('Probability of Error')
ax6.set_title('Test Performance')
for bar, err in zip(bars, errors):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{err:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('question1_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults saved to question1_results.png")
