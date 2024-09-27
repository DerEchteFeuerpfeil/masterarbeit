from typing import List
from torchvision import transforms
from PIL import Image
import wandb
import numpy as np
import torch
from torch import Tensor
import os
from torchvision.utils import make_grid
from torchvision.transforms import Resize
from svgwrite import Drawing
from svgpathtools import disvg, CubicBezier, Line
import cairosvg
from PIL import Image
from io import BytesIO
from torchvision.transforms import ToTensor
import re
import matplotlib.colors as mcolors

def width_pred_to_local_stroke_width(width_pred: Tensor, 
                                     diff_vg_raster_resolution:int,
                                     padded_local_viewbox_width:int):
    """
    this function takes the stroke width prediction of the VSQ and converts it to the stroke width that can be used in local SVG rendering

    this function is required as DiffVG handles stroke width very (stupidly) different than in SVGs. 
    In SVGs the stroke width is the total thickness of the stroke and its apparence is related to the viewbox size.
    In DiffVG, the stroke width is the thickness of the stroke in BOTH directions (not total) in pixels and related to the canvas size in the rasterization process. 
    """
    return width_pred / diff_vg_raster_resolution * padded_local_viewbox_width * 2

def map_wand_config(config):
    if not "wandb_version" in config:
        return config
    new_config = {}
    for k, v in config.items():
        if not "wandb" in k:
            new_config[k] = v["value"]
    return new_config

def interpolate_rows(a, b, n, method='linear'):
    """
    Interpolates rows between a and b in n steps using the specified method. Used in weight interpolation of pyramid loss
    
    Parameters:
    a (numpy array): The first row.
    b (numpy array): The last row.
    n (int): Number of interpolation steps.
    method (str): Interpolation method, either 'linear' or 'exponential'.
    
    Returns:
    numpy array: A 2D array with interpolated rows.
    """
    # Initialize the tensor
    tensor = np.zeros((n, len(a)))
    
    # Fill the tensor with interpolated values
    for i in range(n):
        if method == 'linear':
            tensor[i] = a + i / (n - 1) * (b - a)
        elif method == 'exponential':
            tensor[i] = np.exp(np.log(a) + i / (n - 1) * (np.log(b + 1e-10) - np.log(a)))
        else:
            raise ValueError("Invalid method. Use 'linear' or 'exponential'.")
    
    return tensor

def get_color_gradient(num_colors: int, start_color="red", end_color="blue", return_hex:bool=True, strip_alpha:bool=False):
    gradient = mcolors.LinearSegmentedColormap.from_list('gradient', [start_color, end_color])
    colors = [gradient(i / num_colors) for i in range(num_colors)]
    if return_hex:
        return [mcolors.rgb2hex(color, keep_alpha=(not strip_alpha)) for color in colors]
    else:
        colors = [color[:3] for color in colors] if strip_alpha else colors
        return colors

def get_rendered_svg_with_gradient(svg_path, stroke_width_fraction = 0.7/72,image_size:int=480, alternating_colors:bool=False, indicator:bool=False, return_drawing:bool=False, lseg:float=None):
    base_attribute = {
        "fill": "none",
        "fill-opacity": "1.0",
        "filling": "0",
        "stroke":"black",
        "stroke-width":"1.5",
    }
    indicator_attribute = {
        "fill": "none",
        "fill-opacity": "1.0",
        "filling": "0",
        "stroke":"black",
        "stroke-width":"0.8",
        "stroke-opacity" : "0.5",
        "stroke-dasharray" : "5,5",
    }
    paths, attributes, svg_attributes = svg2paths2(svg_path)

    sw = stroke_width_fraction * float(svg_attributes["viewBox"].split(" ")[-1])

    if indicator:
        indicators = [Line(x[0].start,end=complex(0.,0.)) for x in paths]
    else:
        indicators = []
    # flattened_paths = get_flattened_paths(paths)

    if lseg is not None:
        flattened_paths = get_similar_length_paths(paths, max_length=lseg, min_length=0.001)

    num_paths = len(flattened_paths)
    gradient = get_color_gradient(num_paths, start_color = "red",end_color="black")
    if alternating_colors:
        start = gradient[0]
        end = gradient[-1]
        gradient = [start, end] * (num_paths // 2) + [start] * (num_paths % 2)
    new_attributes=[]
    for i in range(num_paths):
        # Calculate the color for the current path based on the gradient
        color = gradient[i]

        # Create a separate attribute dictionary for the current path
        path_attribute = base_attribute.copy()
        path_attribute['stroke'] = color
        path_attribute["stroke-width"] = f"{sw}"

        # Add the attribute dictionary to the list
        new_attributes.append(path_attribute)

    drawing = disvg(flattened_paths + indicators, attributes=new_attributes + [indicator_attribute]*len(indicators), paths2Drawing=True, viewbox=svg_attributes["viewBox"], dimensions=(image_size, image_size))
    img = drawing_to_tensor(drawing, w=image_size)
    if return_drawing:
        return img, drawing
    return img

def svg_file_path_to_tensor(path, permuted = False, plot=False, stroke_width=0.5, filling:bool=False,image_size:int=224):
    paths, attributes, svg_attributes = svg2paths2(path)
    for i, attr in enumerate(attributes):
        attr["stroke_width"] = f"{stroke_width}"
        attr["fill"] = "black" if filling else "none"

    if "viewbox" in svg_attributes:
        viewbox = svg_attributes["viewbox"]
    else:
        viewbox = None
    return_tensor = raster(disvg(paths, attributes=attributes,paths2Drawing=True, viewbox=viewbox), out_h=image_size, out_w = image_size)

    if permuted:
        return_tensor = return_tensor.permute(1,2,0)
    if plot:
        plt.imshow(return_tensor)
    return return_tensor

def add_points_to_image(all_points:Tensor, image:Tensor, image_scale:int, only_start_end:bool=False, radius:int=2, with_gradient:bool=False):
    """
    inputs:
    - all_points: tensor of shape (batch, n_points, 2)
    - image: tensor of shape (batch, 3, 128, 128)
    - image_scale: 128 if the image is 128x128, 224 if the image is 224x224

    this function should be used to add predicted points to a reconstructed image for better debugging of shape predictions. 
    start/end points are red, bending control points are green.
    """
    all_points = all_points.detach().clone()
    image = image.detach().clone()
    radius = int(radius/128 * image_scale)


    for batch in range(all_points.shape[0]):
        if with_gradient:
            colors = get_color_gradient(len(all_points[batch][0]), start_color = "red",end_color="yellow", return_hex=False, strip_alpha=True)
        else:
            colors = [(1,0,0), (0,1,0), (0,1,0)] * (len(all_points[batch][0]))
        for i, point in enumerate(all_points[batch][0]):
            point = point * image_scale
            point = point.long()
            # this could crash if the point is outside the image or on the border
            # try:
            if i%3 == 0:
                image[batch, 0, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+radius] = colors[i][0]
                image[batch, 1, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+radius] = colors[i][1]
                image[batch, 2, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+radius] = colors[i][2]
            else:
                if not only_start_end:
                    image[batch, 0, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+radius] = colors[i][0]
                    image[batch, 1, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+radius] = colors[i][1]
                    image[batch, 2, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+radius] = colors[i][2]
            # elif i>3:
            #     image[batch, 0, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+radius] = 0
            #     image[batch, 1, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+radius] = 0.5
            #     image[batch, 2, point[1]-radius:point[1]+radius, point[0]-radius:point[0]+radius] = 1

            # except Exception as e:
            #     print("[INFO] couldnt add points to logging image", e)
    return image

def svg_string_to_tensor(svg_string, output_width:int=480):
    # Convert SVG string to PNG bytes
    png_bytes = cairosvg.svg2png(bytestring=svg_string, background_color="white", output_width=output_width, output_height=output_width)
    
    # Convert PNG bytes to PIL Image
    image = Image.open(BytesIO(png_bytes))
    
    # Ensure the image is in RGB mode
    image = image.convert("RGB")
    
    # Convert the PIL Image to a PyTorch tensor with three channels
    tensor = ToTensor()(image)
    
    return tensor

def get_side_by_side_reconstruction(model, dataset, idx, device, w=480, dataset_name:str = "glyphazzn", return_drawing=False, use_model_width_prediction:bool=False, override_global_stroke_width:float=0.04, quantize_grid_size:int=None):
    """
    cannot type hint dataset and model because of circular import

        i know override_global_stroke_width is used as local, but its global, just follow the functions

    """
    if "mnist" in dataset_name.lower():
        gt, label, _, _ = dataset.__getitem__(idx)
        recons_rastered_drawing = model.forward(gt.to(device), only_return_recons=True).cpu()

        num_tiles_per_row = np.sqrt(gt.shape[0]).astype(int)
        if num_tiles_per_row > 1:
            fig, axs = plt.subplots(num_tiles_per_row, num_tiles_per_row*2, figsize=(15, 15))
            # add the gt
            for i in range(0, num_tiles_per_row):
                for j in range(0, num_tiles_per_row):
                    axs[i, j].imshow(gt[i * num_tiles_per_row + j].permute(1, 2, 0).numpy())
                    # axs[i, j].axis('off')
            # add the recons
            for i in range(0, num_tiles_per_row):
                for j in range(num_tiles_per_row, num_tiles_per_row*2):
                    axs[i, j].imshow(recons_rastered_drawing[i * num_tiles_per_row + (j - num_tiles_per_row)].permute(1, 2, 0).cpu().numpy())
                    # axs[i, j].axis('off')
                fig.tight_layout()
            img = fig2img(fig)
            plt.close(fig)

            return img
        else:
            # de-batch the tensors
            if recons_rastered_drawing.dim() == 4:
                recons_rastered_drawing = recons_rastered_drawing[0]
            if gt.dim() == 4:
                gt = gt[0]
    else:
        # print("we glyphn")
        # Get the ground truth SVG drawing
        gt = dataset._get_full_svg_drawing(idx, width=w, as_tensor=True)

        # print("we gettin full")
        # Reconstruct the SVG drawing
        patches, labels, positions, _ = dataset._get_full_item(idx)
        patches = patches.to(device)
        positions = positions.to(device)

        if quantize_grid_size is not None:
            positions = positions * quantize_grid_size
            positions = positions.round()
            positions = positions / quantize_grid_size

        # print("we reconstructing")
        if use_model_width_prediction:
            recons_drawing, _ = model.reconstruct(patches, positions, dataset.individual_max_length +2, None, rendered_w=w)
        else:
            recons_drawing, _ = model.reconstruct(patches, positions, dataset.individual_max_length +2, local_stroke_width=override_global_stroke_width, rendered_w=w)
        recons_rastered_drawing = svg_string_to_tensor(recons_drawing.tostring())
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Plot reconstructed drawing
    axes[1].imshow(recons_rastered_drawing.cpu().permute(1, 2, 0))
    axes[1].set_title('Reconstructed SVG')

    # Plot ground truth drawing
    axes[0].imshow(gt.cpu().permute(1, 2, 0))
    axes[0].set_title('Ground Truth SVG')
    for ax in axes:
        ax.axis('off')
     
    fig.tight_layout()
    img = fig2img(fig)
    plt.close(fig)
    if return_drawing:
        if dataset_name.lower() == "glyphazzn":
            return img, recons_drawing
        else:
            return img, None
    else:
        return img
    
def drawing_to_tensor(drawing: Drawing, w:int=480):
    return svg_string_to_tensor(drawing.tostring(), output_width=w)

def svg_to_tensor(file_path, 
                  new_stroke_width:float = None,
                  new_stroke_width_fraction:float = None,
                  new_stroke_color:str = None,
                  new_fill_color:str = None,
                  output_width:int=None,
                  return_svgs:bool=False):
    if new_stroke_width is None and new_fill_color is None and new_stroke_color is None and new_stroke_width_fraction is None:
        png_data = cairosvg.svg2png(url=file_path, background_color="white", output_width=output_width, output_height=output_width)
        image = Image.open(BytesIO(png_data))
        image = image.convert("RGB")
        tensor = ToTensor()(image)
        drawing = None
    else:
        paths, attributes, svg_attributes = svg2paths2(file_path)
        if svg_attributes.get("viewBox") is None:
            svg_width = svg_attributes.get("width")
            svg_attributes["viewBox"] = f"0 0 {svg_width} {svg_width}"
        for i in range(len(attributes)):
            if new_fill_color is not None:
                attributes[i]["fill"] = new_fill_color
            if new_stroke_color is not None:
                attributes[i]["stroke"] = new_stroke_color
            if new_stroke_width is not None:
                attributes[i]["stroke-width"] = f"{new_stroke_width}"
                if attributes[i].get("stroke") is None:
                    attributes[i]["stroke"] = "black"
            if new_stroke_width_fraction is not None:
                vb_size = float(svg_attributes["viewBox"].split(" ")[-1])
                attributes[i]["stroke-width"] = f"{new_stroke_width_fraction * vb_size}"
                if attributes[i].get("stroke") is None:
                    attributes[i]["stroke"] = "black"

        drawing = disvg(paths, viewbox = svg_attributes.get("viewBox"), dimensions = (output_width, output_width), attributes = attributes, paths2Drawing=True)
        tensor = drawing_to_tensor(drawing)
    if return_svgs:
        return tensor, drawing
    return tensor

def calculate_global_positions(local_positions: Tensor, local_viewbox_width:float, global_center_positions: Tensor):
    """
    Calculates the global positions of svg shapes from the local centered positions.

    local_positions in [0,1]
    
    """
    assert local_positions.max() <= 1.0 and local_positions.min() >= 0.0, f"local_positions must be in [0,1], got max-min of {local_positions.max()} {local_positions.min()}"
    assert global_center_positions.max() > 1.0, f"global_center_positions must NOT be in [0,1], but got max of {global_center_positions.max()}"
    assert global_center_positions.max() <= 72, f"global_center_positions must be in [0,72] to keep the original ratio of full SVG and local viewbox intact, but got max of {global_center_positions.max()}"

    local_points_delta_to_middle = local_positions - 0.5
    scaled_local_points_delta_to_middle = local_points_delta_to_middle * local_viewbox_width
    global_center_positions = global_center_positions.unsqueeze(1).unsqueeze(1).repeat(1, scaled_local_points_delta_to_middle.shape[1], scaled_local_points_delta_to_middle.shape[2], 1)
    global_positions = global_center_positions + scaled_local_points_delta_to_middle
    return global_positions

def tensor_to_complex(my_tensor):
    return complex(my_tensor[0].item(), my_tensor[1].item())

def stroke_points_to_bezier(my_tensor:Tensor):
    """
    expects my_tensor to be in shape (4, 2)
    """
    return CubicBezier(tensor_to_complex(my_tensor[0]), tensor_to_complex(my_tensor[1]), tensor_to_complex(my_tensor[2]), tensor_to_complex(my_tensor[3]))

def stroke_to_path(my_tensor: Tensor):
    """
    expects my_tensor to be in shape (1+3*num_segments, 2)
    """
    num_segments = (my_tensor.shape[0] - 1) // 3
    all_paths = []
    for seg_idx in range(num_segments):
        start_idx = seg_idx * 3
        end_idx = (seg_idx+1) * 3 + 1
        all_paths.append(stroke_points_to_bezier(my_tensor[start_idx:end_idx]))
    return Path(*all_paths)

def rgb_to_hex(r, g, b):
    """
    Convert RGB values between [0,1] to a hex color string suitable for SVG.

    Returns:
    str: Hex color string (e.g., "#4A5699").
    """
    # Ensure the RGB values are within the valid range
    r = min(1, max(0, r))
    g = min(1, max(0, g))
    b = min(1, max(0, b))

    # Convert to 0-255 range
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)

    # Format as a hex string
    return f"#{r:02X}{g:02X}{b:02X}"

def shapes_to_drawing(shapes:Tensor, 
                      stroke_width:float|List|Tensor, 
                      w=128, 
                      num_strokes_to_paint:int = 0, 
                      linecap="round", 
                      linejoin="round",
                      mode="stroke",
                      visual_attribute_dict:dict=None,
                      return_individual_drawings:bool=False) -> Drawing:
    """
    stroke_width here is GLOBAL, not local - if None is provided, use the stroke_widths from the visual_attribute_dict

    expects shapes to be in shape (n, 1+3*num_segments, 2)

    colors in attribute dict must be RBG tensors in [0,1]
    """
    assert linecap in ["round", "butt", "square"], "linecap must be either 'round', 'butt' or 'square'."
    assert linejoin in ["round", "bevel", "miter"], "linejoin must be either 'round', 'bevel' or 'miter'."

    if mode == "stroke":
        base_attribute = {
            "fill": "none",
            "fill-opacity": "1.0",
            "filling": "0",
            "stroke":"black",
            "stroke-width":"1",
            "stroke-linecap":linecap,
            "stroke-linejoin" : linejoin

        }
    else:
        base_attribute = {
            "fill": "none",
            "fill-opacity": "1.0",
            "filling": "0",
            "stroke":"black",
            "stroke-width":"1",
            "stroke-linecap":linecap,
            "stroke-linejoin" : linejoin

        }

    assert shapes.max() <= 1.1, f"shapes must be roughly in [0,1], got max of {shapes.max()}"

    if shapes.ndim > 3:
        shapes = shapes.squeeze()
    
    shapes[shapes>1.0] = 1.0
    viewbox_width = 72
    shapes = shapes * viewbox_width

    all_shapes = []
    for shape in shapes:
        all_shapes.append(stroke_to_path(shape))
    if len(shapes) != len(all_shapes):
        print("MISMATCH in shapes_to_drawing: ", len(shapes), len(all_shapes))

    if num_strokes_to_paint > len(all_shapes):
        num_strokes_to_paint = len(all_shapes)
        print(f"setting num_strokes_to_paint {num_strokes_to_paint} to {len(all_shapes)}")
    
    # if stroke mode and we predicted colors, also use them in the render
    if mode =="stroke" and visual_attribute_dict is None:
        colors = ["red"] * num_strokes_to_paint + ["black"] * (len(all_shapes) - num_strokes_to_paint)
    elif mode == "stroke" and visual_attribute_dict["colors"] is not None:
        assert visual_attribute_dict["colors"].shape[0] == len(all_shapes) + num_strokes_to_paint, f"you provided more colors than shapes to paint, got {visual_attribute_dict['colors'].shape[0]} colors and {len(all_shapes)} shapes"
        stroke_colors = visual_attribute_dict["colors"].cpu()[:len(all_shapes) - num_strokes_to_paint]
        stroke_colors = [color.flatten() for color in stroke_colors]
        stroke_colors = [rgb_to_hex(color[0].item(), color[1].item(), color[2].item()) for color in stroke_colors]
        # colors = ["red"] * num_strokes_to_paint + ["black"] * (len(all_shapes) - num_strokes_to_paint)
        colors = ["red"] * num_strokes_to_paint + stroke_colors
        if len(colors) < len(all_shapes): # this case happens when visual attribute dict does not align with the shapes, e.g. when using min_dist_interpolate post processing
            colors += ["black"] * (len(all_shapes) - len(colors))
    elif mode == "stroke" and visual_attribute_dict["colors"] is None:
        colors = ["red"] * num_strokes_to_paint + ["black"] * (len(all_shapes) - num_strokes_to_paint)
    else:
        colors = visual_attribute_dict["colors"].cpu() # (num_circles, 4)
        colors = [rgb_to_hex(color[0].item(), color[1].item(), color[2].item()) for color in colors] # -> (num_circles)
    
    if isinstance(stroke_width, float):
        stroke_widths = [stroke_width] * len(all_shapes)
    elif isinstance(stroke_width, list):
        stroke_widths = stroke_width
    elif isinstance(stroke_width, Tensor):
        stroke_widths = [sw.mean().item() for sw in stroke_width.cpu()]
    else:
        stroke_widths = visual_attribute_dict["stroke_widths"].cpu() # (num_circles, num_strokes, 1)
        # NOTE future should use all individually or just predict 1 value, use mean of prediction for now
        stroke_widths = [stroke_width.mean().item() for stroke_width in stroke_widths]  # -> (num_circles)
    all_attributes = []
    for i, shape in enumerate(all_shapes):
        attributes = base_attribute.copy()
        attributes["stroke-width"] = f"{stroke_widths[i]}"
        attributes["stroke"] = colors[i]
        if mode != "stroke":
            attributes["fill"] = colors[i]
        all_attributes.append(attributes)

    # for i in range(len(all_shapes)//2):
    #     all_attributes[-i]["stroke"] = "red"

    if return_individual_drawings:
        return [disvg([shape], attributes=[all_attributes[i]], paths2Drawing=True, viewbox=f"0 0 {viewbox_width} {viewbox_width}", dimensions=(w, w)) for i, shape in enumerate(all_shapes)]
    else:
        drawing = disvg(all_shapes, attributes=all_attributes, paths2Drawing=True, viewbox=f"0 0 {viewbox_width} {viewbox_width}", dimensions=(w, w))
        return drawing

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer.buffer_rgba())
    return X[:,:,:3]


def make_tensor(x, grad=False):
    x = torch.tensor(x, dtype=torch.float32)
    x.requires_grad = grad
    return x

def log_all_images(images: List[Tensor], log_key="validation", caption="Captions not set"):
    """
    Logs all images of a list as grids to wandb.

    Args:
        - images (List[Tensor]): List of images to log
        - log_key (str): key for wandb logging
        - captions (str): caption for the images
    """

    assert len(images) > 0, "No images to log"

    common_size = images[0].shape[-2:]
    resizer = Resize(common_size, antialias=True)

    image_result = make_grid(images[0], nrow=4, padding=5, pad_value=0.2)
    for image in images[1:]:
        image_result = torch.concat((image_result, make_grid(resizer(image), nrow=4, padding=5, pad_value=0.2)), dim=-1)

    return log_key, image_result
    # return log_key, wandb.Image(image_result, caption=caption)
    # wandb.log({log_key: wandb.Image(image_result, caption=caption)})

def get_merged_image_for_logging(images: List[Tensor]) -> Tensor:
    """
    resized and merges all images of a list into a single loggable tensor
    """
    common_size = images[0].shape[-2:]
    resizer = Resize(common_size, antialias=True)
    images = [resizer(image) for image in images]

    merged_image = make_grid(images, nrow=math.ceil(np.sqrt(len(images))), padding=5, pad_value=0.2)

    return merged_image


def log_images(recons: Tensor, real_imgs: Tensor, log_key="validation", captions="Captions not set"):

    # if get_rank() != 0:
    #     return

    if recons.shape[-2:] != real_imgs.shape[-2:]:
        common_size = recons.shape[-2:]
        resizer = Resize(common_size, antialias=True)
        real_imgs_resized = resizer(real_imgs)
    else:
        real_imgs_resized = real_imgs

    bs, c, w, h = real_imgs_resized.shape

    if recons.shape[1] > real_imgs_resized.shape[1]:
        real_imgs_resized = torch.cat((real_imgs_resized, torch.ones((bs, 1, w, h), device=real_imgs_resized.device)), dim=1)
    elif recons.shape[1] < real_imgs_resized.shape[1]:
        recons = torch.cat((recons, torch.ones((bs, 1, w, h), device=recons.device)), dim=1)

    image_result = torch.concat((
        make_grid(real_imgs_resized, nrow=4, padding=5, pad_value=0.2),
        make_grid(recons, nrow=4, padding=5, pad_value=0.2)
        ),
        dim=-1
    )
    # return log_key, wandb.Image(image_result, caption=captions)
    # WandbLogger.log_image(key=log_key, images=image_result, caption=captions)
    wandb.log({log_key: wandb.Image(image_result, caption=captions)})


def get_rank() -> int:
    if not torch.distributed.is_available():
        return 0  # Training on CPU
    if not torch.distributed.is_initialized():
        rank = os.environ.get("LOCAL_RANK")  # from pytorch-lightning
        if rank is not None:
            return int(rank)
        else:
            return 0
    else:
        return torch.distributed.get_rank()

def tensor_to_histogram_image(tensor, bins=100):
    # Create a histogram plot
    plt.hist(tensor, bins=bins)
    plt.title('Codebook usage histogram')

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Create a PIL image from the BytesIO object
    image = Image.open(buf).copy()

    # Close the buffer
    buf.close()

    return image

##############################################################################################################
# SVG splitting utils
##############################################################################################################
from svgpathtools import svg2paths, svg2paths2, disvg, Path  # this is used to READ and breakdown SVG
import math
from svgwrite import Drawing
from cairosvg import svg2png
import io
from matplotlib import pyplot as plt
import copy
from torchvision import transforms
def raster(svg_file: Drawing, out_h: int = 128, out_w: int = 128):
    """
    This function simply resizes and rasters a series of Paths
    @param svg_file: Drawing object
    @return: Numpy array of the raster image single-channel
    """
    svg_png_image = svg2png(
        bytestring=svg_file.tostring(),
        output_width=out_w,
        output_height=out_h,
        background_color="white")
    img = Image.open(io.BytesIO(svg_png_image))
    # rgb_image = Image.new("RGB", img.size, (255, 255, 255))
    # rgb_image.paste(img, mask=img.split()[3])
    transform = transforms.ToTensor()
    tensor_image = transform(img)
    return tensor_image

def save_path_as_image(path: Path, out_h: int = 128, out_w: int = 128):
    """
    This function simply resizes and rasters a series of Paths
    @param svg_file: Drawing object
    @return: Numpy array of the raster image single-channel
    """
    svg_file = disvg(path, paths2Drawing=True, stroke_widths=[2.0] * len(path))
    svg_png_image = svg2png(
        bytestring=svg_file.tostring(),
        output_width=out_w,
        output_height=out_h,
        background_color="white")
    img = Image.open(io.BytesIO(svg_png_image))
    img.save("test.png")

def plot_segments(rasterized_segments, title:str="A disassembled tree"):
    assert rasterized_segments.shape[0] > 8, "too few segments to plot"
    nrows = math.ceil(len(rasterized_segments) / 8)
    ncols = 8
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(5*ncols, 5*nrows))
    for i, img in enumerate(rasterized_segments):
        curr_row = i // ncols
        curr_col = i % ncols
        axs[curr_row][curr_col].imshow(img, cmap="gray")
        axs[curr_row][curr_col].axis("off")
    if title is not None:
        axs[0][ncols//2].set_title(title)

def plot_merged_segments(rasterized_segments, title=None):
    plt.imshow(np.array(rasterized_segments).min(axis=0), cmap="gray")

def get_flattened_paths(paths):
    flattened_paths = [segment for path in paths for segment in path._segments]
    return flattened_paths

def get_single_paths(paths, filter_zero_length = True):
    single_paths = [Path(segment) for path in paths for segment in path._segments]
    if filter_zero_length:
        single_paths = [path for path in single_paths if path.length() > 0.]
    return single_paths

def calc_max_diff(single_paths):
    total_max_diff = 0
    for idx in range(len(single_paths)):
        abs_start = single_paths[idx].start #- single_paths[0].end
        abs_end = single_paths[idx].end #- single_paths[0].end
        top_left = complex(min(abs_start.real, abs_end.real), min(abs_start.imag, abs_end.imag))
        bottom_right = complex(max(abs_start.real, abs_end.real), max(abs_start.imag, abs_end.imag))
        diff = bottom_right - top_left
        max_diff = max(diff.real, diff.imag)
        if max_diff > total_max_diff:
            total_max_diff = max_diff
    return total_max_diff

def all_paths_to_max_diff(all_paths, index:int = 1):
    """
    index is the idx of the max_diff you want to get. idx=0 is largest, idx=1 is second largest, etc.
    """
    all_max_diffs = []
    for path in all_paths:
        paths, _, _ = svg2paths2(path)
        single_paths = get_single_paths(paths)
        all_max_diffs.append(calc_max_diff(single_paths))
    all_max_diffs = np.array(all_max_diffs)
    total_max_diff = all_max_diffs[np.argsort(-all_max_diffs)[:index+1]][index]
    return total_max_diff

def all_paths_to_max_diffs(all_paths):
    all_max_diffs = []
    for path in all_paths:
        paths, _, _ = svg2paths2(path)
        single_paths = get_single_paths(paths)
        all_max_diffs.append(calc_max_diff(single_paths))
    return all_max_diffs

def get_viewbox(single_path, total_max_diff, offset: float = 1.0):
    """
    returns viewbox and center of the viewbox as x-y-tuple
    """
    abs_start = single_path.start
    abs_end = single_path.end
    top_left = complex(min(abs_start.real, abs_end.real), min(abs_start.imag, abs_end.imag))
    bottom_right = complex(max(abs_start.real, abs_end.real), max(abs_start.imag, abs_end.imag))
    diff = bottom_right - top_left
    center = top_left + diff / 2
    new_top_left = center - complex(total_max_diff / 2, total_max_diff / 2)
    viewbox = f"{new_top_left.real - offset} {new_top_left.imag - offset} {total_max_diff + offset*2} {total_max_diff + offset*2}"
    return viewbox, [center.real, center.imag]

def old_get_rasterized_segments(single_paths:list, stroke_width:float, total_max_diff: float, svg_attributes, centered = False, height: int = 128, width: int = 128, colors=None) -> List:
    
    if centered:
        single_paths = [my_path for my_path in single_paths if my_path.length() > 0.]
        if len(single_paths) == 0:
            # print("[INFO] tried to rasterize an empty path")
            return [torch.ones((3, height, width)), torch.ones((3, height, width))], [[width/2,height/2], [width/2,height/2]]
        out = [get_viewbox(my_path, total_max_diff) for my_path in single_paths]
        viewboxes = [x[0] for x in out]
        centers = [x[1] for x in out]
        if colors is not None:
            rasterized_segments = [raster(disvg(my_path, paths2Drawing=True, colors=[colors[i]], stroke_widths=[stroke_width] * len(my_path), viewbox=viewboxes[i]), out_h = height, out_w = width) for i, my_path in enumerate(single_paths)]
        else:
            rasterized_segments = [raster(disvg(my_path, paths2Drawing=True, stroke_widths=[stroke_width] * len(my_path), viewbox=viewboxes[i]), out_h = height, out_w = width) for i, my_path in enumerate(single_paths)]
        return rasterized_segments, centers
    else:
        viewbox=svg_attributes["viewBox"]
        rasterized_segments = [raster(disvg(my_path, paths2Drawing=True, stroke_widths=[stroke_width] * len(my_path), viewbox=viewbox), out_h = height, out_w = width) for my_path in single_paths if my_path.length() > 0.]
        centers = [(0,0)] * len(rasterized_segments)
        return rasterized_segments, centers


def rgb_tensor_to_svg_color(rgb_tensor):
    """
    Converts an RGB tensor to a hexadecimal color string for SVG.

    Parameters:
    rgb_tensor (torch.Tensor): A 1D tensor of shape [3] with values in the range [0, 1].

    Returns:
    str: A hexadecimal color string usable in SVG.
    """
    # Ensure the tensor has the right shape
    if rgb_tensor.shape != (3,):
        raise ValueError("Input tensor must be a 1D tensor with 3 elements (R, G, B).")
    
    # Clamp the values to [0, 1] to ensure valid RGB range
    rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)
    
    # Convert to 8-bit values and then to hexadecimal
    hex_color = ''.join([f'{int(c * 255):02X}' for c in rgb_tensor])
    
    # Prepend '#' to form a valid SVG color string
    return f'#{hex_color}'

def get_rasterized_segments(single_paths:list, 
                            stroke_width:float, 
                            total_max_diff: float, 
                            svg_attributes, 
                            centered = False, 
                            height: int = 128, 
                            width: int = 128, 
                            colors=None,
                            fill:bool=False) -> List:
    
    # this is important as it matches the rendering of DiffVG, so reconstructions are more consistent
    base_attribute = {
        "stroke-linecap" : "round",
    }

    # filter out zero lengths
    if centered:
        single_paths = [my_path for my_path in single_paths if my_path.length() > 0.]
    
    # build attribute dict for visual attributes
    all_attributes = []
    for i, path in enumerate(single_paths):
        curr_attribute = base_attribute.copy()
        curr_attribute["stroke-width"] = f"{stroke_width}"
        if colors is not None:
            curr_attribute["stroke"] = colors[i]
            if fill:
                curr_attribute["fill"] = colors[i]
            else:
                curr_attribute["fill"] = "none"
        else:
            curr_attribute["stroke"] = "black"
            curr_attribute["fill"] = "none"
        all_attributes.append(curr_attribute)

    if centered:
        if len(single_paths) == 0:
            print("[INFO] tried to rasterize an empty path")
            return [torch.ones((3, height, width)), torch.ones((3, height, width))], [[width/2,height/2], [width/2,height/2]]
        out = [get_viewbox(my_path, total_max_diff) for my_path in single_paths]
        viewboxes = [x[0] for x in out]
        centers = [x[1] for x in out]

        rasterized_segments = []
        for i, my_path in enumerate(single_paths):
            single_path_drawing = disvg(my_path, attributes=[all_attributes[i]], paths2Drawing=True, viewbox=viewboxes[i])
            rasterized_single_path_drawing = raster(single_path_drawing, out_h = height, out_w = width)
            rasterized_segments.append(rasterized_single_path_drawing)
        
        return rasterized_segments, centers
    else:
        viewbox=svg_attributes["viewBox"]
        rasterized_segments = [raster(disvg(my_path, attributes=[all_attributes[i]], paths2Drawing=True, viewbox=viewbox), out_h = height, out_w = width) for my_path in single_paths if my_path.length() > 0.]
        centers = [(0,0)] * len(rasterized_segments)
        return rasterized_segments, centers


def svg_path_to_segment_image_arrays(svg_path, total_max_diff: float):
    """
    This function takes a path to an SVG file and returns two numpy arrays of the rasterized path segments.

    Inputs:
        svg_path: path to the SVG file
    
    Returns:
        rasterized_segments_centered: numpy array of the rasterized segments, all placed in the middle of the image
        rasterized_segments: numpy array of the rasterized segments, placed on their relative position where they belong
    """
    paths, attributes, svg_attributes = svg2paths2(svg_path)
    single_paths = get_single_paths(paths)

    # everything placed in the middle
    rasterized_segments_centered = get_rasterized_segments(single_paths, stroke_width = 0.5, total_max_diff=total_max_diff, svg_attributes=svg_attributes, centered=True)

    # everything placed where it belongs
    rasterized_segments = get_rasterized_segments(single_paths, stroke_width = 2.0, total_max_diff=total_max_diff, svg_attributes=svg_attributes, centered=False)

    return rasterized_segments_centered, rasterized_segments

def get_positional_array_from_paths(single_paths, svg_attributes):
    viewbox_x, viewbox_y, viewbox_w, viewbox_h = [float(x) for x in svg_attributes["viewBox"].split(" ")]
    assert viewbox_x == 0 and viewbox_y == 0, "you require normalization of viewbox"
    abs_start_points = []
    abs_end_points = []
    rel_start_points = []
    rel_end_points = []
    for i, path in enumerate(single_paths):
        abs_start_points.append([path.start.real, path.start.imag])
        abs_end_points.append([path.end.real, path.end.imag])

        rel_start_x = path.start.real / viewbox_w
        rel_start_y = path.start.imag / viewbox_h

        rel_start_points.append([rel_start_x, rel_start_y])

        rel_end_x = path.end.real / viewbox_w
        rel_end_y = path.end.imag / viewbox_h

        rel_end_points.append([rel_end_x, rel_end_y])
    
    stacked_points = np.stack([abs_start_points, abs_end_points,  rel_start_points,  rel_end_points], axis=1)
    return stacked_points 

# def get_similar_length_paths(queue:list, max_length:float):
#     similar_length_paths = []
#     curr_aggregated_path = Path()
#     while len(queue) > 0:
#         path = queue.pop(0)
#         if curr_aggregated_path.length() + path.length() < max_length and curr_aggregated_path.end == path.start:
#             curr_aggregated_path.extend(path)
#         else:
#             similar_length_paths.append(curr_aggregated_path)
#             curr_aggregated_path = path
#     return similar_length_paths[1:]  # first path is always empty

def crop_path_into_segments(path:Path, length:float = 5.):
    """
    a single input path is cropped into segments of approx length `length`. I say "approx" because we divide the path into same length segments, which will not be exactly `length` long.
    """
    segments = []
    try:
        num_iters = math.ceil(path.length() / length)
        for i in range(num_iters):
            cropped_segment = path.cropped(i/num_iters, (i+1)/num_iters)
            segments.append(cropped_segment)
    except Exception as e:
        pass
    return segments

def get_similar_length_paths(single_paths, max_length: float = 5., min_length:float = None):
        """
        splits all the paths into similar length segments if they're too long
        """

        similar_length_paths = []
        if min_length is not None:
            prev_len = len(single_paths)
            single_paths = [x for x in single_paths if x.length() >= min_length]
            after_len = len(single_paths)
            if after_len >= 0.8 * prev_len:
                print("More than 80% of paths were removed because they were too short. This is likely an error.")
        for path in single_paths:
            if path.length() < min_length:
                similar_length_paths.append(path)
                continue
            try:
                segments = crop_path_into_segments(path, length=max_length)
                similar_length_paths.extend(segments)
            except AssertionError:
                print("Error while cropping path into segments, skipping...")
                continue
        return similar_length_paths

def check_for_continouity(single_paths: list):
    for path in single_paths:
        if not path.iscontinuous():
            return False
    return True