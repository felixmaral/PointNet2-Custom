import numpy as np

def sample_fixed_num_points(points, labels, num_points):
    """
    Recorta la nube a num_points si tiene más puntos. Si tiene menos, lanza un error.
    """
    N = points.shape[0]
    if N >= num_points:
        indices = np.random.choice(N, num_points, replace=False)
        return points[indices], labels[indices]
    else:
        raise ValueError(f"Nube con solo {N} puntos, pero se requieren {num_points}.")