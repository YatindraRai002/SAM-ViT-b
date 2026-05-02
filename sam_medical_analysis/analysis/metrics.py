import torch
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def compute_cka(features_x, features_y):
    # Ensure tensors are on the same device and float32
    features_x = features_x.float()
    features_y = features_y.float()
    device = features_x.device
    features_y = features_y.to(device)

    def centering(K):
        n = K.shape[0]
        unit = torch.ones(n, n, device=K.device) / n
        return K - unit @ K - K @ unit + unit @ K @ unit

    def linear_hsic(X, Y):
        K = X @ X.T
        L = Y @ Y.T
        K_cent = centering(K)
        L_cent = centering(L)
        return torch.sum(K_cent * L_cent)

    hsic_xy = linear_hsic(features_x, features_y)
    hsic_xx = linear_hsic(features_x, features_x)
    hsic_yy = linear_hsic(features_y, features_y)

    denom = torch.sqrt(hsic_xx) * torch.sqrt(hsic_yy)
    if denom < 1e-6:
        return 0.0
    return (hsic_xy / denom).item()

def compute_silhouette(features, labels):
    return silhouette_score(features, labels, metric='cosine')

def linear_probe_accuracy(features, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
        
    counts = [np.sum(np.array(labels) == l) for l in unique_labels]
    min_samples = min(counts)
    
    cv_folds = min(5, min_samples)
    if cv_folds < 2:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)

    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, features, labels, cv=cv_folds)
    return scores.mean()

def compute_dice(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
