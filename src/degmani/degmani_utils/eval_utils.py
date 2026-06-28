import numpy as np


def embedding_distances(a, b, eps=1e-12):
    """
    Compute cosine distance and L2 distance between two embeddings.

    Args:
        a, b: Arrays or lists of embedding length.
        eps: Small value to prevent division by zero.

    Returns:
        dict with 'cosine_distance' and 'l2_distance'.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    if a.shape != b.shape:
        raise ValueError("Embeddings must have the same shape.")

    # L2 (Euclidean) distance
    l2_distance = np.linalg.norm(a - b)

    # Cosine distance (1 - cosine similarity)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na > eps and nb > eps:
        cos_sim = np.dot(a, b) / (na * nb)
        # Clamp to handle numerical issues
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        cosine_distance = 1.0 - cos_sim
    else:
        # If either vector is (near) zero, define cosine distance as maximal (1.0)
        cosine_distance = 1.0

    return {"cosine_distance": cosine_distance, "l2_distance": l2_distance}


def compute_overall_mean(features_list):
    """
    features_list: list of np.ndarray of shape (D,)
    returns: np.ndarray of shape (D,)
    """
    feats = np.stack(features_list, axis=0)  # (N, D)
    return feats.mean(axis=0)

def compute_per_class_means(features_list, labels):
    """
    features_list: list of np.ndarray of shape (D,)
    labels: list/array of ints or strings, length N
    returns: dict[label] -> np.ndarray (D,)
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for f, y in zip(features_list, labels):
        groups[y].append(f)
    return {y: np.mean(np.stack(v, axis=0), axis=0) for y, v in groups.items()}