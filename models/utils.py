import torch

def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.
    src: (B, N, C)
    dst: (B, M, C)
    Returns:
        dist: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Index point features with a given index tensor.
    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    Returns:
        new_points: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape.append(points.shape[-1])
    # More robust batch_indices creation that adapts to idx's shape
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(B, *[1] * (idx.dim() - 1)).expand_as(idx)
    return points[batch_indices, idx]


def farthest_point_sample(xyz, npoint):
    """
    Sample the farthest points from the point cloud.
    xyz: (B, N, 3)
    Returns:
        centroids: (B, npoint)
    """
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(xyz.device)
    distance = torch.ones(B, N).to(xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long).to(xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Group local regions using ball query.
    xyz: (B, N, 3)
    new_xyz: (B, S, 3)
    Returns:
        group_idx: (B, S, nsample)
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    group_idx[group_idx == N] = group_first[group_idx == N]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Perform FPS and group points within each region.
    xyz: (B, N, 3)
    points: (B, N, D)
    Returns:
        new_xyz: (B, npoint, 3)
        new_points: (B, npoint, nsample, 3+D)
    """
    B, N, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points