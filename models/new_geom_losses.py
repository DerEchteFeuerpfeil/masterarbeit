import torch

def directional_consistency_loss(points):
    # points: Tensor of shape (T, 2)
    p_1 = points[0]  # First point
    p_T = points[-1]  # Last point
    overall_direction = p_T - p_1
    
    segment_directions = points[1:] - points[:-1]
    norm_overall_direction = torch.norm(overall_direction)
    
    cos_similarities = []
    for d_i in segment_directions:
        cos_sim = torch.dot(d_i, overall_direction) / (torch.norm(d_i) * norm_overall_direction)
        cos_similarities.append(cos_sim)
    
    cos_similarities = torch.stack(cos_similarities)
    loss = torch.sum((1 - cos_similarities) ** 2)
    return loss


def consistent_distance_loss(points):
    # points: Tensor of shape (T, 2)
    segment_lengths = torch.sqrt(torch.sum((points[1:] - points[:-1]) ** 2, dim=1))
    mean_length = torch.mean(segment_lengths)
    loss = torch.mean((segment_lengths - mean_length) ** 2)
    return loss
