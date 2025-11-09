import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

RNG_SEED = 2025
np.random.seed(RNG_SEED)

class GaussianSampler:
    def __init__(self):
        self.K = 4
        self.prior = np.ones(self.K) / self.K
        self.means = [
            np.array([0.0, 0.0, 0.0]),
            np.array([3.0, 0.5, 0.0]),
            np.array([0.5, 3.0, 0.5]),
            np.array([2.5, 2.5, 2.5]),
        ]
        self.covs = [
            np.array([[1.5, 0.3, 0.2], [0.3, 1.5, 0.3], [0.2, 0.3, 1.5]]),
            np.array([[1.2, -0.2, 0.1], [-0.2, 1.8, 0.2], [0.1, 0.2, 1.2]]),
            np.array([[1.8, 0.4, -0.1], [0.4, 1.3, 0.2], [-0.1, 0.2, 1.6]]),
            np.array([[1.4, 0.2, 0.3], [0.2, 1.4, 0.1], [0.3, 0.1, 1.4]]),
        ]
        self.rv = [multivariate_normal(mean=m, cov=c) for m, c in zip(self.means, self.covs)]

    def sample(self, n_samples):
        counts = np.random.multinomial(n_samples, self.prior)
        Xparts = []
        yparts = []
        for k in range(self.K):
            n = counts[k]
            if n == 0:
                continue
            s = self.rv[k].rvs(size=n)
            if n == 1:
                s = s.reshape(1, -1)
            Xparts.append(s)
            yparts.append(np.full(n, k, dtype=int))
        X = np.vstack(Xparts)
        y = np.hstack(yparts)
        perm = np.random.permutation(len(y))
        return X[perm], y[perm]

    def compute_posteriors(self, X):
        N = X.shape[0]
        probs = np.zeros((N, self.K))
        for k in range(self.K):
            probs[:, k] = self.rv[k].pdf(X) * self.prior[k]
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def map_predict(self, X):
        return np.argmax(self.compute_posteriors(X), axis=1)

def choose_hidden_units(X_train, y_train, options):
    n = X_train.shape[0]
    if n <= 1000:
        cv = 10
    else:
        cv = min(10, max(2, np.min(np.bincount(y_train))))
    results = {}
    for p in options:
        clf = MLPClassifier(
            hidden_layer_sizes=(p,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=0.01,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=12,
            random_state=RNG_SEED,
        )
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        results[p] = 1.0 - scores.mean()
    best = min(results, key=results.get)
    return best, results

def fit_final_network(X_train, y_train, hidden_units, n_restarts=5):
    best_model = None
    best_train_acc = -1.0
    for seed in range(n_restarts):
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_units,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=0.01,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RNG_SEED + seed,
        )
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_model = clf
    return best_model

def produce_plots(summary, bayes_error, out_path="question1_results.png"):
    train_sizes = summary["sizes"]
    test_errors = summary["test_errs"]
    chosen_P = summary["chosen_P"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.semilogx(train_sizes, test_errors, "bo-", linewidth=2.2, markersize=8)
    ax.axhline(y=bayes_error, color="red", linestyle="--", linewidth=1.8)
    for x, y in zip(train_sizes, test_errors):
        ax.annotate(f"{y*100:.1f}%", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlabel("Training set size (N, log scale)")
    ax.set_ylabel("Test error (empirical P[error])")
    ax.set_title("MLP Performance vs Training Size")
    ax.grid(True, ls=":", alpha=0.6)
    ax2 = axes[1]
    ax2.plot(train_sizes, chosen_P, "gs-", linewidth=2.2, markersize=8)
    for x, y in zip(train_sizes, chosen_P):
        ax2.annotate(f"{y}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax2.set_xscale("log")
    ax2.set_xlabel("Training set size (N, log scale)")
    ax2.set_ylabel("Selected hidden units (P)")
    ax2.set_title("Model Complexity chosen by CV")
    ax2.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

def main():
    sampler = GaussianSampler()
    train_sizes = [100, 500, 1000, 5000, 10000]
    test_size = 100000
    X_test, y_test = sampler.sample(test_size)
    y_map = sampler.map_predict(X_test)
    bayes_err = float(np.mean(y_map != y_test))
    options = [6, 12, 30]
    summary = {"sizes": [], "chosen_P": [], "test_errs": []}
    start_time = time.time()
    print("=" * 60)
    print("QUESTION 1 — MLP CLASSIFICATION (refactored)")
    print("=" * 60)
    print(f"Bayes (MAP) empirical test error ≈ {bayes_err:.4f}\n")
    for N in train_sizes:
        X_train, y_train = sampler.sample(N)
        best_p, cv_results = choose_hidden_units(X_train, y_train, options)
        model = fit_final_network(X_train, y_train, best_p, n_restarts=6)
        test_err = 1.0 - model.score(X_test, y_test)
        summary["sizes"].append(N)
        summary["chosen_P"].append(best_p)
        summary["test_errs"].append(test_err)
        print(f"[N={N:6d}] chosen P={best_p:2d}  test P(err)={test_err:.4f}")
    produce_plots(summary, bayes_err, out_path="question1_results_refactored.png")
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Train N':<10s} {'Best P':<8s} {'Test err':<10s} {'Error %':<8s}")
    print("-" * 60)
    for n, p, e in zip(summary["sizes"], summary["chosen_P"], summary["test_errs"]):
        print(f"{n:<10d} {p:<8d} {e:<10.4f} {100*e:<8.2f}")
    print("-" * 60)
    print(f"{'Optimal':<10s} {'-':<8s} {bayes_err:<10.4f} {100*bayes_err:<8.2f}")
    print("=" * 60)
    elapsed = (time.time() - start_time) / 60.0
    print(f"\n✓ Question 1 complete in {elapsed:.2f} minutes\n")

if __name__ == "__main__":
    main()
