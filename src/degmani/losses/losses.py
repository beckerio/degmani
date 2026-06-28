import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def nt_xent_loss(a: torch.Tensor, b: torch.Tensor, tau: float = 0.1):
    """

    Compute the NT-Xent loss.
    see https://github.com/miccunifi/ARNIQA/blob/main/models/simclr.py

    Args:
        a (torch.Tensor): first set of features
        b (torch.Tensor): second set of features
        tau (float): temperature parameter
    """
    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    a_cap = torch.div(a, a_norm)
    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    b_cap = torch.div(b, b_norm)
    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    sim_by_tau = torch.div(sim, tau)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators, denominators)
    neglog_num_by_den = -torch.log(num_by_den)
    return torch.mean(neglog_num_by_den)



# NT-Xent Loss
def nt_xent_loss_stable(a: torch.Tensor, b: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """
    Stable NT-Xent (SimCLR) loss with aligned positives: a[i] <-> b[i].
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    as used in SimCLR/ARNIQA.

    Args:
        a: tensor of shape (N, D) - first view embeddings
        b: tensor of shape (N, D) - second view embeddings
        tau: temperature scalar
    """
    # Normalize
    a = F.normalize(a, dim=1, eps=1e-8)
    b = F.normalize(b, dim=1, eps=1e-8)

    # Combine views
    z = torch.cat([a, b], dim=0)  # (2N, D)
    sim_matrix = torch.matmul(z, z.T) / tau  # cosine similarity since z is normalized [2N, 2N]

    # Mask out self-similarity
    N = a.shape[0]
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    #sim_matrix.masked_fill_(mask, -9e15)  # effectively -inf
    #sim_matrix.masked_fill_(mask, torch.finfo(sim_matrix.dtype).min)
    sim_matrix.masked_fill_(mask, -1e4)

    # Positive pairs: i-th sample in a with i-th in b, and vice versa
    # positives: i in [0..N-1] -> i+N, and i in [N..2N-1] -> i-N
    positives = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)

    loss = F.cross_entropy(sim_matrix, positives)
    return loss


def supervised_contrastive(z, labels, temperature=0.2):
    # z: [N, d], labels: list/hashable length N
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / temperature  # [N, N]
    labels = np.array(labels)
    N = z.size(0)
    mask_pos = torch.zeros((N, N), dtype=torch.bool, device=z.device)
    for i, lab in enumerate(labels):
        pos = np.where(labels == lab)[0]
        pos = [j for j in pos if j != i]
        if pos:
            mask_pos[i, torch.tensor(pos, device=z.device)] = True
    logits = sim - torch.eye(N, device=z.device) * 1e9
    exp = torch.exp(logits)
    denom = exp.sum(dim=1, keepdim=True)
    log_prob = logits - torch.log(denom + 1e-9)
    loss = -(log_prob[mask_pos]).mean() if mask_pos.any() else torch.tensor(0., device=z.device)
    return loss
