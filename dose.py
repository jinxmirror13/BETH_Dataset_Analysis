from math import log

import numpy as np
import os
import pickle
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import SGDOneClassSVM
import torch
import tqdm
    
# DEBUG timestamps
from datetime import datetime # DEBUG


def kl_divergence(z, P, Q):
    return P.log_prob(z) - Q.log_prob(z)


def get_summary_stats(data_loader, model, marginal_posterior, mc_samples, iwae_samples, seed, device):
    torch.manual_seed(seed)  # Reset seed (ensure statistics' reproducibility while preserving the logic of the approximation)
    model.eval()
    summary_stats = []
    with torch.no_grad():
        for t in tqdm.trange(1, mc_samples + 1):
            summary_stats_t = []
            for i, (x, y) in enumerate(tqdm.tqdm(data_loader, leave=False)):
                x, y = x.to(device=device, non_blocking=True), y.to(device=device, non_blocking=True)
                observation, posterior, z = model(x)
                T_rate = kl_divergence(z, posterior, marginal_posterior)  # KL divergence between the posterior and marginal (rate): KL[q(Z|X, θ), q(Z)]
                T_xent = -marginal_posterior.log_prob(z)  # Cross-entropy between the posterior and marginal: H[q(Z|X, θ), q(Z)]
                T_ent = -posterior.log_prob(z)  # Entropy of the posterior: H[q(Z|X, θ)]
                T_dist = observation.log_prob(x)  # Expected log-likelihood computed over the posterior (distortion): E[q(X|Z, θ)]
                zs = posterior.sample((iwae_samples, ))
                observations = model.decode(zs.view(iwae_samples * x.size(0), -1))
                T_iwae = torch.mean(observations.log_prob(x.repeat_interleave(iwae_samples, dim=0)).view(iwae_samples, x.size(0)) - kl_divergence(zs, posterior, marginal_posterior), dim=0)  # Estimate of the evidence computed using a 16-sample IWAE: q(X|θ) = E[q(X|Z, θ)q(Z)/q(Z|X, θ)]
                summary_stats_t.append(torch.stack([T_rate, T_xent, T_ent, T_dist, T_iwae], dim=1))
            summary_stats.append(torch.cat(summary_stats_t, dim=0).cpu().numpy())
    return np.mean(np.stack(summary_stats, axis=2), axis=2)


class DoSE_SVM():
    def __init__(self, train_summary_stats):
        # Whiten the training summary statistics using PCA
        self.pca = PCA(whiten=True).fit(train_summary_stats)
        # Fit a one-class SVM to the whitened summary statistics
        self.clf = SGDOneClassSVM().fit(self.pca.transform(train_summary_stats))

    def detect_outliers(self, test_summary_stats):
        # Evaluate DoSE one-class SVM on test dataset
        return self.clf.predict(self.pca.transform(test_summary_stats)) == -1
