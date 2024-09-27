import pydiffvg
import diffvg
import torch
import skimage
import numpy as np
import os
import pandas as pd
from torch.nn.functional import sigmoid
from tqdm import tqdm
import kornia
from torch import Tensor
import torch.nn.functional as F

def gaussian_pyramid_loss(recons_images: Tensor, gt_images: Tensor, down_sample_steps: int = 3, pyramid_weights:Tensor = None):
        """
        Calculates the gaussian pyramid loss between reconstructed images and ground truth images.

        Args:
            - recons_images (Tensor): Reconstructed images in format (-1, c, w, h)
            - gt_images (Tensor): Ground truth images in format (-1, c, w, h)
            - down_sample_steps (int): Number of downsample steps to calculate the loss for. Default: 3

        Returns:
            - recon_loss (Tensor): The gaussian pyramid loss between reconstructed images and ground truth images.
        """
        dsample = kornia.geometry.transform.pyramid.PyrDown()
        recon_loss = F.mse_loss(recons_images, gt_images, reduction='none')

        recon_loss = recon_loss.mean() * pyramid_weights[0]
        recons_loss_contributions = {
            "pyramid_loss_step_0" : recon_loss.detach().cpu().item()
        }
        for j in range(1, 1 + down_sample_steps):
            if j < len(pyramid_weights):
                weight = pyramid_weights[j]
            else:
                weight = 0.0

            recons_images = dsample(recons_images)
            gt_images = dsample(gt_images)
            loss_images = F.mse_loss(recons_images, gt_images, reduction='none')


            curr_pyramid_loss = loss_images.mean() * weight
            recons_loss_contributions[f"pyramid_loss_step_{j}"] = curr_pyramid_loss.detach().cpu().item()
            recon_loss = recon_loss + curr_pyramid_loss


        return recon_loss, recons_loss_contributions


# loss_df = pd.DataFrame(columns=['pxf', 'loss', "step"])
RADIUS = 0.5  # too low of a radius makes antwar, too high makes blurry
USE_PYRAMID = True

SAVE_PATH = f"results/single_stroke_color_sw2"
iter_path = os.path.join(SAVE_PATH, 'iter')

if not os.path.exists(iter_path):
    os.makedirs(iter_path)

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256, 256
num_control_points = torch.tensor([2])
points = torch.tensor([[0.2,  0.2], # base
                    [1.0,  1.0], # control point
                    [ 0.2, 1.0], # control point
                    [ 0.4, 0.2]]) # base

# s curve
points = torch.tensor([[0.2,  0.8], # base
                    [0.2,  0.3], # control point
                    [ 0.9, 0.9], # control point
                    [ 0.9, 0.2]]) # base

# very sharp s curve
points = torch.tensor([[0.2,  0.8], # base
                    [0.2,  0.3], # control point
                    [ 0.9, 0.9], # control point
                    [ 0.2,  0.3]]) # base

points = points * canvas_width

path = pydiffvg.Path(num_control_points = num_control_points,
                    points = points,
                    is_closed = False,
                    stroke_width = torch.tensor(5.0))
shapes = [path]
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                fill_color = torch.tensor([0.0, 0.0, 0.0, 0.0]),
                                stroke_color = torch.tensor([0.7, 0.1, 0.0, 1.0]))
shape_groups = [path_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, 
                                                    canvas_height, 
                                                    shapes, 
                                                    shape_groups,
                                                    filter=pydiffvg.PixelFilter(type = diffvg.FilterType.box,
                                                                                    radius = torch.tensor(RADIUS)),)

render = pydiffvg.RenderFunction.apply
bg_image = torch.ones((canvas_width, canvas_height, 4))
img = render(256, # width
            256, # height
            2,   # num_samples_x
            2,   # num_samples_y
            0,   # seed
            bg_image, # background_image
            *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), os.path.join(SAVE_PATH, 'target.png'), gamma=2.2)
target = img.clone()

# Move the path to produce initial guess
# normalize points for easier learning rate
points_n = torch.tensor([[100.0/256.0,  40.0/256.0], # base
                        [155.0/256.0,  65.0/256.0], # control point
                        [100.0/256.0, 180.0/256.0], # control point
                        [ 65.0/256.0, 238.0/256.0]], # base
                        requires_grad = True) 
# stroke_color = torch.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)
stroke_color = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
stroke_width_n = torch.tensor(1.0, requires_grad=True)

path.points = points_n * 256
path.stroke_width = sigmoid(stroke_width_n) * 5
stroke_color_with_alpha = torch.cat([stroke_color, torch.tensor([1.0])])
path_group.stroke_color = sigmoid(stroke_color_with_alpha)

scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, 
                                                    canvas_height, 
                                                    shapes, 
                                                    shape_groups,
                                                    filter=pydiffvg.PixelFilter(type = diffvg.FilterType.box,
                                                                                    radius = torch.tensor(RADIUS)),)
img = render(256, # width
            256, # height
            2,   # num_samples_x
            2,   # num_samples_y
            1,   # seed
            bg_image, # background_image
            *scene_args)
pydiffvg.imwrite(img.cpu(), os.path.join(SAVE_PATH, 'init.png'), gamma=2.2)

# Optimize
optimizer = torch.optim.Adam([points_n, stroke_color, stroke_width_n], lr=1e-2)
# optimizer = torch.optim.Adam([points_n], lr=1e-2)
# Run 200 Adam iterations.
losses = []
for t in range(500):
    optimizer.zero_grad()
    # Forward pass: render the image.
    path.points = points_n * 256
    path.stroke_width = sigmoid(stroke_width_n) * 5
    path_group.stroke_color = torch.cat([sigmoid(stroke_color), torch.tensor([1.0])])
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, 
                                                    canvas_height, 
                                                    shapes, 
                                                    shape_groups,
                                                    filter=pydiffvg.PixelFilter(type = diffvg.FilterType.box,
                                                                                    radius = torch.tensor(RADIUS)),)
    img = render(256,   # width
                256,   # height
                2,     # num_samples_x
                2,     # num_samples_y
                t+1,   # seed
                bg_image, # background_image
                *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), os.path.join(iter_path, 'iter_{}.png'.format(t)), gamma=2.2)
    # Compute the loss function. Here it is L2.
    if USE_PYRAMID:
        # print(img.shape)
        loss, _ = gaussian_pyramid_loss(img.permute(2,0,1).unsqueeze(0), target.permute(2,0,1).unsqueeze(0), down_sample_steps=5, pyramid_weights=torch.tensor([1.0, 0.5, 0.5, 0.5, 0.5, 0.5])) # pyramid_weights=torch.tensor([1.0, 0.5, 0.25, 0.125])
    else:
        loss = (img - target).pow(2).mean()
    losses.append(loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    if t % 25 == 0:
        print('iteration:', t)
        print('loss:', loss.item())
        print("stroke_color:", path_group.stroke_color)
        print("stroke_width:", path.stroke_width)
        print('points_n.grad:', points_n.grad)
        print('stroke_color.grad:', stroke_color.grad)
        print('stroke_width.grad:', stroke_width_n.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    # print('points:', path.points)
    # print('stroke_color:', path_group.stroke_color)
    # print('stroke_width:', path.stroke_width)

# loss_df = pd.concat([loss_df, pd.DataFrame({'pxf': [PIXEL_FILTER_SIZE] * len(losses), 'loss': losses, 'step': range(len(losses))})], ignore_index=True)

# Render the final result.
# path.points = points_n * 256
# path.stroke_width = sigmoid(stroke_width_n) * 5
# path_group.stroke_color = sigmoid(stroke_color_with_alpha)
scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, 
                                                    canvas_height, 
                                                    shapes, 
                                                    shape_groups,
                                                    filter=pydiffvg.PixelFilter(type = diffvg.FilterType.box,
                                                                                    radius = torch.tensor(RADIUS)),)
img = render(256,   # width
            256,   # height
            2,     # num_samples_x
            2,     # num_samples_y
            202,    # seed
            bg_image, # background_image
            *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), os.path.join(SAVE_PATH, 'final.png'), gamma=2.2)
scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, 
                                                    canvas_height, 
                                                    shapes, 
                                                    shape_groups,
                                                    filter=pydiffvg.PixelFilter(type = diffvg.FilterType.box,
                                                                                    radius = torch.tensor(RADIUS)),)
img = render(256,   # width
            256,   # height
            20,     # num_samples_x
            20,     # num_samples_y
            202,    # seed
            bg_image, # background_image
            *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), os.path.join(SAVE_PATH, 'final_more_samples.png'))

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-y","-framerate", "24", "-i",
    f"{iter_path}/iter_%d.png", "-vb", "20M",
    f"{SAVE_PATH}/out.mp4"])

# loss_df.to_csv(os.path.join("results/single_stroke", 'loss.csv'), index=False)
import matplotlib.pyplot as plt
# plot all losses in one plot with legend being the pixel filter size, give each loss a gradient color depending on the pixel filter size
def get_color(value, cmap_name='seismic'):
    cmap = plt.cm.get_cmap(cmap_name)  # Get the colormap
    return cmap(value)  #

# loss_df = pd.read_csv(os.path.join("results/single_stroke", 'loss.csv'))
# loss_df = loss_df[loss_df["pxf"] >= 0.1]
# plot all losses in one plot with legend being the pixel filter size
# for pxf in loss_df['pxf'].unique():
#     df = loss_df[loss_df['pxf'] == pxf]
#     plt.plot(df['step'], df['loss'], label=f"pxf: {pxf}", color=get_color((pxf)/4.0))
# plt.legend()
# plt.title("Loss curves for different pixel filter sizes in DiffVG")
# plt.xlabel('step')
# plt.ylabel('loss')
# plt.savefig(os.path.join("results/single_stroke", 'loss.png'), dpi=300)