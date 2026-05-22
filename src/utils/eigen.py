import numpy as np
import torch

def manual_norm(v):
    """Menghitung panjang vektor manual."""
    return np.sqrt(np.sum(v**2))

def manual_eig(A, num_components=50, num_iter=1000):
    """
    Menghitung Nilai Eigen dan Vektor Eigen murni dengan perulangan
    (Metode Power Iteration & Deflation), dilarang pakai np.linalg.eig.
    """
    # Konversi ke numpy jika input adalah torch tensor
    if isinstance(A, torch.Tensor):
        A_np = A.cpu().numpy().astype(float)
        use_torch = True
        device = A.device
    else:
        A_np = A.copy().astype(float)
        use_torch = False

    n = A_np.shape[0]
    eigenvalues  = []
    eigenvectors = []
    A_def        = A_np.copy()
    num_components = min(num_components, n)

    for _ in range(num_components):          # ← loop luar: tiap komponen
        v = np.random.rand(n)
        norm_v = manual_norm(v)
        if norm_v == 0:
            break
        v = v / norm_v

        for _ in range(num_iter):            # ← loop dalam: power iteration
            w      = np.dot(A_def, v)
            norm_w = manual_norm(w)
            if norm_w == 0:
                break
            v_new = w / norm_w

            # Update v DULU, baru cek konvergensi
            converged = (np.allclose(v, v_new, atol=1e-6) or
                         np.allclose(v, -v_new, atol=1e-6))
            v = v_new
            if converged:
                break

        # Hitung eigenvalue & deflation SETELAH power iteration selesai
        eigenvalue = np.dot(v.T, np.dot(A_def, v))
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
        A_def = A_def - eigenvalue * np.outer(v, v)   # deflation

    eigenvalues_np  = np.array(eigenvalues)
    eigenvectors_np = np.array(eigenvectors).T         # shape: (n, num_components)

    # Kembalikan dalam tipe yang sama dengan input
    if use_torch:
        return (
            torch.tensor(eigenvalues_np,  dtype=torch.float32, device=device),
            torch.tensor(eigenvectors_np, dtype=torch.float32, device=device),
        )
    return eigenvalues_np, eigenvectors_np


def manual_euclidean_distance(vec1, vec2):
    """Menghitung jarak Euclidean manual: Akar dari jumlah kuadrat selisih."""
    if isinstance(vec1, torch.Tensor):
        return torch.sqrt(torch.sum((vec1 - vec2) ** 2)).item()
    return np.sqrt(np.sum((vec1 - vec2) ** 2))