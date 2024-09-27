from torch import Tensor
import torch
import numpy as np
from utils import shapes_to_drawing, calculate_global_positions, drawing_to_tensor



def _get_closest_idxs2(tensor1: Tensor, tensor2: Tensor, num_segments:int=1):
    """
    returns the indices of the closest points in tensor1 and tensor2 and the distance between them
    """
    # Calculate pairwise distances between points in tensor1 and tensor2
    dist_matrix = torch.cdist(tensor1, tensor2)

    # Find the indices of the minimum distance
    min_dist_index = torch.argmin(dist_matrix)
    min_dist = dist_matrix.view(-1)[min_dist_index]

    # Convert the flat index to two-dimensional indices representing the indices of the closest points
    indices = divmod(min_dist_index.item(), dist_matrix.size(1))

    # multiply index by three because 0*3 = 0 (start point) and 1*3 = 3 (end point)
    idx_0 = indices[0] * 3*num_segments
    idx_1 = indices[1] * 3*num_segments

    return idx_0, idx_1, min_dist.item()

def _get_closest_idxs(tensor1: Tensor, tensor2: Tensor):
    """
    Returns the indices of the closest points in tensor1 and tensor2 and the distance between them.
    Considers only real points, not control points.
    """
    real_points1 = tensor1[::3]  # Only real points (1st, 4th, 7th, etc.)
    real_points2 = tensor2[::3]

    # Calculate pairwise distances between real points in tensor1 and tensor2
    dist_matrix = torch.cdist(real_points1, real_points2)

    # Find the indices of the minimum distance
    min_dist_index = torch.argmin(dist_matrix)
    min_dist = dist_matrix.view(-1)[min_dist_index]

    # Convert the flat index to two-dimensional indices representing the indices of the closest points
    indices = divmod(min_dist_index.item(), dist_matrix.size(1))

    # Multiply index by three to get the actual indices in the original tensor
    idx_0 = indices[0] * 3
    idx_1 = indices[1] * 3

    return idx_0, idx_1, min_dist.item()

def min_dist_fix(global_positions: Tensor, method: str = "min_dist_clip", max_dist: float = None, connect_last: bool = True):
    """
    Fixing method that uses the minimum distance between the points of consecutive strokes.
    
    Args:
    - global_positions: shape (n_strokes, 1 + 3 * n, 2)
    - method: str, either "min_dist_clip" or "min_dist_interpolate"
    - max_dist: float, maximum distance between points to be considered for fixing, None to disable
    - connect_last: bool, whether to connect the last stroke to the first stroke

    Returns:
    - new global_positions: shape (n_strokes + n_interpolations, 1 + 3 * n, 2) or (n_strokes, 1 + 3 * n, 2) depending on the method
    """
    assert method in ["min_dist_clip", "min_dist_interpolate"], 'method must be either "min_dist_clip" or "min_dist_interpolate"'
    interpolated_positions = []
    min_dists = []

    n_strokes = global_positions.size(0)

    for i in range(n_strokes):
        stroke = global_positions[i]
        if i == 0 and connect_last:
            prev_stroke = global_positions[-1]
        elif i == 0 and not connect_last:
            continue
        else:
            prev_stroke = global_positions[i-1]

        closest_idx_prev, closest_idx_curr, min_dist = _get_closest_idxs(prev_stroke, stroke)
        min_dists.append(min_dist)

        if max_dist is not None and min_dist > max_dist:
            continue

        if method == "min_dist_clip":
            global_positions[i][closest_idx_curr] = prev_stroke[closest_idx_prev]
        elif method == "min_dist_interpolate":
            new_start_point = global_positions[i][closest_idx_curr]
            new_end_point = prev_stroke[closest_idx_prev]
            middle_point = (new_start_point + new_end_point) / 2
            interpolated_positions.append(torch.stack([new_end_point, middle_point, middle_point, new_start_point]))

    if method == "min_dist_interpolate":
        if interpolated_positions:
            interpolated_positions = torch.stack(interpolated_positions)
            if interpolated_positions.size(1) != global_positions.size(1):
                num_repeat = global_positions.size(1) - interpolated_positions.size(1)
                pad = interpolated_positions[:, -1, :].unsqueeze(1).repeat(1, num_repeat, 1)
                interpolated_positions = torch.cat([interpolated_positions, pad], dim=1)
            global_positions = torch.cat([global_positions, interpolated_positions], dim=0)

    return global_positions


def min_dist_fix_global(global_positions: torch.Tensor, method: str = "min_dist_clip", max_dist: float = None, **kwargs):
    """
    Global fixing method that uses the minimum distance between the points of all strokes, connecting only to the closest stroke.
    
    Args:
    - global_positions: shape (n_strokes, 1 + 3 * n, 2)
    - method: str, either "min_dist_clip" or "min_dist_interpolate"
    - max_dist: float, maximum distance to be considered for connecting points

    Returns:
    - new global_positions: shape (n_strokes + n_interpolations, 1 + 3 * n, 2) or (n_strokes, 1 + 3 * n, 2) depending on the method
    """
    assert method in ["min_dist_clip", "min_dist_interpolate"], 'method must be either "min_dist_clip" or "min_dist_interpolate"'
    interpolated_positions = []

    n_strokes = global_positions.size(0)
    min_dist_combinations = []

    # Loop to consider each stroke for connection to the closest stroke
    for i in range(n_strokes):
        closest_global_idx = None
        closest_idx_i = None
        closest_idx_j = None
        global_min_dist = float('inf')

        for j in range(n_strokes):
            if i != j:
                stroke_i = global_positions[i].clone()  # Clone to ensure separate memory
                stroke_j = global_positions[j].clone()  # Clone to ensure separate memory
                idx_i, idx_j, min_dist = _get_closest_idxs(stroke_i, stroke_j)

                if min_dist < global_min_dist and min_dist > 0.0:
                    global_min_dist = min_dist
                    closest_global_idx = j
                    closest_idx_i = idx_i
                    closest_idx_j = idx_j

        if max_dist is not None and global_min_dist > max_dist:
            continue

        # Apply the chosen method using the closest global stroke
        if closest_global_idx is not None:
            stroke_j = global_positions[closest_global_idx].clone()  # Clone to ensure separate memory
            if method == "min_dist_clip":
                global_positions[i][closest_idx_i] = stroke_j[closest_idx_j]
            elif method == "min_dist_interpolate":
                new_point_i = global_positions[i][closest_idx_i]
                new_point_j = stroke_j[closest_idx_j]
                middle_point = (new_point_i + new_point_j) / 2
                interpolated_positions.append(torch.stack([new_point_j, middle_point, middle_point, new_point_i]))

    if method == "min_dist_interpolate":
        if interpolated_positions:
            interpolated_positions = torch.stack(interpolated_positions)
            if interpolated_positions.size(1) != global_positions.size(1):
                num_repeat = global_positions.size(1) - interpolated_positions.size(1)
                pad = interpolated_positions[:, -1, :].unsqueeze(1).repeat(1, num_repeat, 1)
                interpolated_positions = torch.cat([interpolated_positions, pad], dim=1)
            print("stacking now")
            print(f"before: {global_positions.shape}")
            global_positions = torch.cat([global_positions, interpolated_positions], dim=0)
            print(f"after: {global_positions.shape}")
            print(f"some interpolated_positions: {interpolated_positions[:5]}")
            return global_positions

    return global_positions

def min_dist_fix_global2(global_positions: torch.Tensor, method: str = "min_dist_clip", max_dist: float = None, **kwargs):
    """
    Global fixing method that uses the minimum distance between the start and end points of all strokes, connecting only to the closest stroke.
    
    Args:
    - global_positions: shape (n_strokes, 1 + 3 * n, 2)
    - method: str, either "min_dist_clip" or "min_dist_interpolate"
    - max_dist: float, maximum distance to be considered for connecting points

    Returns:
    - new global_positions: shape (n_strokes + n_interpolations, 1 + 3 * n, 2) or (n_strokes, 1 + 3 * n, 2) depending on the method
    """
    assert method in ["min_dist_clip", "min_dist_interpolate"], 'method must be either "min_dist_clip" or "min_dist_interpolate"'
    interpolated_positions = []

    n_strokes = global_positions.size(0)

    # Loop to consider each stroke for connection to the closest stroke for both start and end points
    for i in range(n_strokes):
        for point_type in ['start', 'end']:
            closest_global_idx = None
            closest_idx_i = None
            closest_idx_j = None
            global_min_dist = float('inf')

            for j in range(n_strokes):
                if i != j:
                    stroke_i = global_positions[i].clone()
                    stroke_j = global_positions[j].clone()

                    # Get index for start (0) or end (-3) based on point_type
                    idx_i = 0 if point_type == 'start' else -3

                    # Iterate through each real point in stroke_j to find the closest point to stroke_i[idx_i]
                    min_dist = float('inf')
                    for k in range(0, len(stroke_j), 3):
                        dist = torch.norm(stroke_i[idx_i] - stroke_j[k])
                        if dist < min_dist:
                            min_dist = dist
                            temp_closest_idx_j = k

                    # Update global minimum if a new minimum is found that's smaller than the previous global minimum
                    if min_dist < global_min_dist and min_dist > 0.0:
                        global_min_dist = min_dist
                        closest_idx_i = idx_i
                        closest_idx_j = temp_closest_idx_j
                        closest_global_idx = j

            # Check against max_dist outside the inner loop after finding the global minimum for this point
            if max_dist is not None and global_min_dist > max_dist:
                continue

            # Apply the chosen method using the closest global stroke
            if closest_global_idx is not None:
                stroke_j = global_positions[closest_global_idx].clone()
                if method == "min_dist_clip":
                    global_positions[i][closest_idx_i] = stroke_j[closest_idx_j]
                elif method == "min_dist_interpolate":
                    new_point_i = global_positions[i][closest_idx_i]
                    new_point_j = stroke_j[closest_idx_j]
                    middle_point = (new_point_i + new_point_j) / 2
                    interpolated_positions.append(torch.stack([new_point_j, middle_point, middle_point, new_point_i]))


    if method == "min_dist_interpolate":
        if interpolated_positions:
            interpolated_positions = torch.stack(interpolated_positions)
            if interpolated_positions.size(1) != global_positions.size(1):
                num_repeat = global_positions.size(1) - interpolated_positions.size(1)
                pad = interpolated_positions[:, -1, :].unsqueeze(1).repeat(1, num_repeat, 1)
                interpolated_positions = torch.cat([interpolated_positions, pad], dim=1)
            global_positions = torch.cat([global_positions, interpolated_positions], dim=0)

    return global_positions


def path_interpolation(global_positions, connect_last:bool = True, **kwargs):
    """
    descr from strokenuwa:
    PI entails the addition of M commands between each pair of adjacent, yet non-interconnected SVG commands to bridge the discrepancy to force the 
    previous command’s end point to move to the beginning point of the next adjacent command

    Args:
    - global_positions: shape (n_strokes, 4, 2)
    - connect_last: bool, whether to connect the last stroke to the first stroke

    Returns:
    - interpolated_positions: shape (n_strokes + n_interpolations, 4, 2)
    """
    interpolated_positions = []
    for i, pos in enumerate(global_positions):
        if i == 0:
            if connect_last:
                prev_end_point = global_positions[-1][-1]
            else:
                continue
        else:
            prev_end_point = global_positions[i-1][-1]
        curr_start_point = global_positions[i][0]
        middle_point = (prev_end_point + curr_start_point) / 2
        interpolated_positions.append(torch.stack([prev_end_point, middle_point, middle_point, curr_start_point]))
    interpolations = torch.stack(interpolated_positions)
    last_points = interpolations[:, -1, :]
    if last_points.size(1) != global_positions.size(1):
        num_repeat = global_positions.size(1) - interpolations.size(1)
        pad = last_points.unsqueeze(1).repeat(1, num_repeat, 1)
        interpolations = torch.cat([interpolations, pad], dim=1)
    return torch.concat([global_positions, interpolations], dim=0)

def min_dist_fix2(global_positions: Tensor, method: str = "min_dist_clip", max_dist: float = None, connect_last:bool = True):
    """
    Fixing method that uses the minimum distance between the start and end points of two consecutive strokes. 
    Necessary as visual supervision does not oppose a consistent ordering of start and end points of strokes.
    
    Supports two methods:
    - min_dist_clip: modifies the start point of i to be the end points of i-1 (if the distance is smaller than max_dist)
    - min_dist_interpolate: adds a new stroke between the start of i and end points of i-1 (if the distance is smaller than max_dist)

    Args:
    - global_positions: shape (n_strokes, 4, 2)
    - method (default min_dist_clip): str, either "min_dist_clip" or "min_dist_interpolate"
    - max_dist (default: None): float, maximum distance between two strokes to be considered as connected, None to disable
    - connect_last (default: True): bool, whether to connect the last stroke to the first stroke

    Returns:
    - new global_positions: shape (n_strokes + n_interpolations, 4, 2) or (n_strokes, 4, 2) depening on the method

    """
    assert method in ["min_dist_clip", "min_dist_interpolate"], f'method must be in {["min_dist_clip", "min_dist_interpolate"]}'
    interpolated_positions = []
    min_dists = []
    for i, stroke in enumerate(global_positions):
        num_segments = int((stroke.shape[0] - 1) / 3)
        if i == 0:
            if connect_last:
                prev_stroke = global_positions[-1]
            else:
                continue
        else:
            prev_stroke = global_positions[i-1]  # (4, 2)

        # extract only start and end points of a stroke
        prev_start_end_points = torch.stack([prev_stroke[0], prev_stroke[-1]])  # (2, 2)
        curr_start_end_points = torch.stack([stroke[0], stroke[-1]])  # (2, 2)

        closest_idx_prev, closest_idx_curr, min_dist = _get_closest_idxs(prev_start_end_points, curr_start_end_points, num_segments)
        min_dists.append(min_dist)

        if max_dist is not None and min_dist > max_dist:
            # TODO this signals that this might be the beginning of a new disconnected path in another area of the canvas which means that the previous shape can be closed
            continue

        if method == "min_dist_clip":
            global_positions[i][closest_idx_curr] = prev_stroke[closest_idx_prev]
        elif method == "min_dist_interpolate":
            new_start_point = global_positions[i][closest_idx_curr]
            new_end_point = prev_stroke[closest_idx_prev]
            middle_point = (new_end_point + new_start_point) / 2
            interpolated_positions.append(torch.stack([new_start_point, middle_point, middle_point, new_end_point]))  # (4, 2)

    # print(f"Mean distance between strokes: {np.mean(min_dists):.2f}")
    # print(f"Outlier distance: {np.max(min_dists):.2f}")

    if method == "min_dist_interpolate":
        # add the new strokes to the original strokes
        interpolated_positions = torch.stack(interpolated_positions) # (n_interpolations, 4, 2)
        if interpolated_positions.size(1) != global_positions.size(1):
            num_repeat = global_positions.size(1) - interpolated_positions.size(1)
            pad = interpolated_positions[:,-1,:].unsqueeze(1).repeat(1,num_repeat,1)
            interpolated_positions = torch.cat([interpolated_positions, pad], dim=1)
        global_positions = torch.cat([global_positions, interpolated_positions], dim=0)
    return global_positions

def path_clipping(global_positions: Tensor, connect_last:bool=True, **kwargs):
    """
    descr from strokenuwa:
    PC involves the direct substitution of each SVG command’s beginning point with the endpoint of adjacent SVG commands
    
    Args:
    - global_positions: shape (n_strokes, 4, 2)
    - connect_last: bool, whether to connect the last stroke to the first stroke

    Returns:
    - clipped_positions: shape (n_strokes, 4, 2)
    """
    clipped_positions = global_positions.clone()
    for i, pos in enumerate(clipped_positions):
        if i == 0:
            if connect_last:
                prev_end_point = clipped_positions[-1][-1]
            else:
                continue
        else:
            prev_end_point = clipped_positions[i-1][-1]
        clipped_positions[i][0] = prev_end_point

    return clipped_positions

def get_fixed_svg_drawing(bezier_points: Tensor,
                         positions: Tensor,
                         method: str,
                         stroke_width: float = 0.7,
                         padded_individual_max_length: int = 9.5,
                         width: int = 480,
                         max_dist: float = 4.5,
                         num_strokes_to_paint:int=0,
                         max_position_value:int = 128,
                         visual_attribute_dict:dict=None,
                         global_position_fixing:bool=False,
                         **kwargs):
    """
    Takes local points and global positions and returns a fixed SVG drawing using the specified method.
    Use the tokenizer to convert the model output to the points and positions, then feed it into here.

    Want the result as an img tensor instead? Use the `get_fixed_svg_render` convenience function or pass the output to `drawing_to_tensor`.
    """

    assert method in ["clip", "interpolate", "min_dist_clip", "min_dist_interpolate"], f'method must be in {["clipped", "interpolated", "min_dist_clip", "min_dist_interpolate"]}'
    global_shapes = calculate_global_positions(bezier_points, padded_individual_max_length, positions)[:,0]
    if method == "clip":
        fixed_global_shapes = path_clipping(global_shapes, **kwargs)
    elif method == "interpolate":
        fixed_global_shapes = path_interpolation(global_shapes, **kwargs)
    elif method == "min_dist_clip":
        if global_position_fixing:
            if kwargs.get("v2") is not None and kwargs.get("v2"):
                fixed_global_shapes = min_dist_fix_global2(global_shapes, method="min_dist_clip", max_dist=max_dist, **kwargs)
            else:
                fixed_global_shapes = min_dist_fix_global(global_shapes, method="min_dist_clip", max_dist=max_dist, **kwargs)
        else:
            fixed_global_shapes = min_dist_fix(global_shapes, method="min_dist_clip", max_dist=max_dist, **kwargs)
    elif method == "min_dist_interpolate":
        if global_position_fixing:
            if kwargs.get("v2") is not None and kwargs.get("v2"):
                fixed_global_shapes = min_dist_fix_global2(global_shapes, method="min_dist_interpolate", max_dist=max_dist, **kwargs)
            else:
                fixed_global_shapes = min_dist_fix_global(global_shapes, method="min_dist_interpolate", max_dist=max_dist, **kwargs)
        else:
            fixed_global_shapes = min_dist_fix(global_shapes, method="min_dist_interpolate", max_dist=max_dist, **kwargs)
        
    # scale them back to [0,1]
    fixed_global_shapes = fixed_global_shapes / max_position_value
    fixed_drawing = shapes_to_drawing(fixed_global_shapes, stroke_width, width, num_strokes_to_paint=num_strokes_to_paint,visual_attribute_dict=visual_attribute_dict)
    return fixed_drawing


def get_fixed_global_shapes(bezier_points: Tensor,
                         positions: Tensor,
                         method: str,
                         padded_individual_max_length: int = 9.5,
                         max_dist: float = 4.5):
    """
    Takes local points and global positions and returns a fixed SVG drawing using the specified method.
    Use the tokenizer to convert the model output to the points and positions, then feed it into here.

    Want the result as an img tensor instead? Use the `get_fixed_svg_render` convenience function or pass the output to `drawing_to_tensor`.
    """

    assert method in ["clip", "interpolate", "min_dist_clip", "min_dist_interpolate"], f'method must be in {["clipped", "interpolated", "min_dist_clip", "min_dist_interpolate"]}'
    global_shapes = calculate_global_positions(bezier_points, padded_individual_max_length, positions)[:,0]
    if method == "clip":
        fixed_global_shapes = path_clipping(global_shapes)
    elif method == "interpolate":
        fixed_global_shapes = path_interpolation(global_shapes)
    elif method == "min_dist_clip":
        fixed_global_shapes = min_dist_fix(global_shapes, method="min_dist_clip", max_dist=max_dist)
    elif method == "min_dist_interpolate":
        fixed_global_shapes = min_dist_fix(global_shapes, method="min_dist_interpolate", max_dist=max_dist)
        
    return fixed_global_shapes

def get_fixed_svg_render(bezier_points: Tensor,
                         positions: Tensor,
                         method: str = "min_dist_clip",
                         stroke_width: float = 0.7,
                         padded_individual_max_length: int = 9.5,
                         width: int = 480,
                         max_dist: float = 4.5,
                         num_strokes_to_paint:int=0,
                         max_position_value: int = 128,
                         visual_attribute_dict:dict = None,):
    """
    Convenience function to get the fixed SVG drawing as an image tensor. For more info look at `get_fixed_svg_drawing`.
    """
    fixed_drawing = get_fixed_svg_drawing(bezier_points, 
                                          positions, 
                                          method, 
                                          stroke_width, 
                                          padded_individual_max_length, 
                                          width, 
                                          max_dist, 
                                          num_strokes_to_paint=num_strokes_to_paint,
                                          max_position_value=max_position_value,
                                          visual_attribute_dict=visual_attribute_dict)
    return drawing_to_tensor(fixed_drawing)

def get_svg_render(bezier_points: Tensor,
                   positions: Tensor,
                   stroke_width: float = 0.7,
                   padded_individual_max_length: int = 9.5,
                   width: int = 480,
                   num_strokes_to_paint:int=0):
    drawing = shapes_to_drawing(calculate_global_positions(bezier_points, padded_individual_max_length, positions)[:,0], stroke_width, width, num_strokes_to_paint=num_strokes_to_paint)
    return drawing_to_tensor(drawing)
