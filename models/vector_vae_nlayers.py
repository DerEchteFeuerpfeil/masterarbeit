"""
This code is mostly taken from the original Im2Vec implementation and is used to train a model for comparisson.
I have made slight modifications and improvements.
See original files here: https://github.com/preddy5/Im2Vec
"""

import random

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import List
import wandb

from .vector_vae import VectorVAE
from utils import make_tensor
import pydiffvg
import math
import numpy as np
import logging
import os


class VectorVAEnLayers(VectorVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: list = None,
                 loss_fn: str = 'MSE',
                 imsize: int = 128,
                 paths: int = 4,
                 b_w = True,
                 wandb_logging = None,
                 **kwargs) -> None:
        super(VectorVAEnLayers, self).__init__(in_channels,
                                               latent_dim,
                                               hidden_dims,
                                               loss_fn,
                                               imsize,
                                               paths,
                                               wandb_logging,
                                               **kwargs)

        def get_computational_unit(in_channels, out_channels, unit):
            if unit == 'conv':
                return nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular', stride=1,
                                 dilation=1)
            else:
                return nn.Linear(in_channels, out_channels)

        def get_random_color():
            r = np.random.randint(0,255)
            g = np.random.randint(0,255)
            b = np.random.randint(0,255)

            # don't want white-ish
            if(r+g+b > 600):
                r = 0

            return [nn.Parameter(torch.Tensor([r/255]), requires_grad=True), nn.Parameter(torch.Tensor([g/255]), requires_grad=True), nn.Parameter(torch.Tensor([b/255]), requires_grad=True), nn.Parameter(torch.Tensor([1.]), requires_grad=True)]
        # self.colors = [[0, 0, 0, 1], [255/255, 0/255, 0/255, 1],]
        # self.colors = [[0, 0, 0, 1], [255/255, 0/255, 255/255, 1], [0/255, 255/255, 255/255, 1],]
        # self.colors = [[0, 0, 0, 1], [255/255, 165/255, 0/255, 1], [0/255, 0/255, 255/255, 1],]

        # self.colors = [[252 / 255, 194 / 255, 27 / 255, 1], [255 / 255, 0 / 255, 0 / 255, 1],
        #                [0 / 255, 255 / 255, 0 / 255, 1], [0 / 255, 0 / 255, 255 / 255, 1], ]

        # T = number of circles to predict
        T = kwargs['T']
        print(f"using {T} paths")
        if(b_w):
            if(kwargs["optimize_colors"]):
                self.colors = torch.nn.Parameter(torch.Tensor([0.,0.,0.,1.]).unsqueeze(0).repeat(T,1))
            else:
                self.colors = torch.Tensor([[0., 0., 0., 1.] for _ in range(T)])
        else:
            # self.colors = torch.nn.Parameter([get_random_color() for _ in range(T)], requires_grad=kwargs["optimize_colors"])
            self.colors = torch.nn.Parameter(torch.Tensor([0.5,0.5,0.5,1.]).unsqueeze(0).repeat(T,1), requires_grad=kwargs["optimize_colors"])

        
        
        self.rnn = nn.LSTM(latent_dim, latent_dim, 2, bidirectional=True)
        self.composite_fn = self.hard_composite
        if kwargs['composite_fn'] == 'soft':
            print('Using Differential Compositing')
            self.composite_fn = self.soft_composite
        self.divide_shape = nn.Sequential(
            nn.ReLU(),  # bound spatial extent
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # nn.ReLU(),  # bound spatial extent
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # nn.ReLU(),  # bound spatial extent
        )
        self.final_shape_latent = nn.Sequential(
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU(),  # bound spatial extent
        )
        self.z_order = nn.Sequential(
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # nn.ReLU(),  # bound spatial extent
            # get_computational_unit(latent_dim, latent_dim, 'mlp'),
            # nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, 1, 'mlp'),
        )
        layer_id = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.register_buffer('layer_id', layer_id)

        # Logging stuff
        self.wandb_logging = wandb_logging # None if not logging

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        output, control_loss = self.decode_and_composite(z, verbose=False, return_overlap_loss=True, img_size=self.imsize,**kwargs)
        return [output, input, mu, log_var, control_loss]

    def hard_composite(self, **kwargs):
        layers = kwargs['layers']
        n = len(layers)
        alpha = (1 - layers[n - 1][:, 3:4, :, :])
        rgb = layers[n - 1][:, :3] * layers[n - 1][:, 3:4, :, :]
        for i in reversed(range(n-1)):
            rgb = rgb + layers[i][:, :3] * layers[i][:, 3:4, :, :] * alpha
            alpha = (1-layers[i][:, 3:4, :, :]) * alpha
        rgb = rgb + alpha
        return rgb

    def soft_composite(self, **kwargs):
        layers = kwargs['layers']
        z_layers = kwargs['z_layers']
        n = len(layers)

        inv_mask = (1 - layers[0][:, 3:4, :, :])
        for i in range(1, n):
            inv_mask = inv_mask * (1 - layers[i][:, 3:4, :, :])

        sum_alpha = layers[0][:, 3:4, :, :] * z_layers[0]
        for i in range(1, n):
            sum_alpha = sum_alpha + layers[i][:, 3:4, :, :] * z_layers[i]
        sum_alpha = sum_alpha + inv_mask

        inv_mask = inv_mask / sum_alpha

        rgb = layers[0][:, :3] * layers[0][:, 3:4, :, :] * z_layers[0] / sum_alpha
        for i in range(1, n):
            rgb = rgb + layers[i][:, :3] * layers[i][:, 3:4, :, :] * z_layers[i] / sum_alpha
        rgb = rgb * (1 - inv_mask) + inv_mask
        return rgb


    def soft_composite_W_bg(self, **kwargs):
        layers = kwargs['layers']
        z_layers = kwargs['z_layers']
        n = len(layers)

        sum_alpha = layers[0][:, 3:4, :, :] * z_layers[0]
        for i in range(1, n):
            sum_alpha = sum_alpha + layers[i][:, 3:4, :, :] * z_layers[i]
        sum_alpha = sum_alpha + 600

        inv_mask = 600 / sum_alpha

        rgb = layers[0][:, :3] * layers[0][:, 3:4, :, :] * z_layers[0] / sum_alpha
        for i in range(1, n):
            rgb = rgb + layers[i][:, :3] * layers[i][:, 3:4, :, :] * z_layers[i] / sum_alpha
        rgb = rgb + inv_mask
        return rgb
    
    def manhattan_dist(self, triplet:Tensor) -> float:
        """calculates the summed manhattan distance between the three points of a cubic bezier curve.

        Parameters:
        self: self
        triplet (Tensor): contains the coordinates of a cubic bezier curve in the pixel space
        """
        first = abs(triplet[1][0] - triplet[0][0]) + abs(triplet[1][1] - triplet[0][1])
        second = abs(triplet[2][0] - triplet[1][0]) + abs(triplet[2][1] - triplet[1][1])

        return first.item()+second.item()

    def log_path_lengths(self, all_points:Tensor, current_shape_idx:int):
        """
        logs the path lengths of each full shape (consisting of individual bezier curves) in a batch.
        """
        logging_dict = {}
        for batch_idx in range(all_points.shape[0]):
            total_length = 0
            for curve_idx in range(0, self.curves*3, 3):
                try:
                    # TODO calculate the distance to the control point also
                    total_length += self.manhattan_dist(all_points[batch_idx][curve_idx*3:curve_idx*3+3])
                except:
                    pass
            logging_dict[f"shapeidx_{current_shape_idx}_batchidx_{batch_idx}"] = total_length
            break

        wandb.log(logging_dict)

    def decode_and_composite(self, z: Tensor, return_overlap_loss=False,return_points=False, img_size:int=128, **kwargs):
        bs = z.shape[0]
        layers = []
        n = len(self.colors)
        loss = 0
        z_rnn_input = z[None, :, :].repeat(n, 1, 1)  # [len, batch size, emb dim], copies the latent code for each shape of the composition
        outputs, hidden = self.rnn(z_rnn_input)
        outputs = outputs.permute(1, 0, 2)  # [batch size, len, emb dim]
        outputs = outputs[:, :, :self.latent_dim] + outputs[:, :, self.latent_dim:] # aggregate outputs of both RNNs
        z_layers = []
        all_n_points = []
        for i in range(n):
            shape_output = self.divide_shape(outputs[:, i, :]) # [bs, latent_size] - just a RELU
            shape_latent = self.final_shape_latent(shape_output) # [bs, latent_size] - actual MLP
            all_points = self.decode(shape_latent)#, point_predictor=self.point_predictor[i])
            if return_points:
                all_n_points.append(all_points.cpu())
            # if("log_path_length" in kwargs.keys() and self.wandb_logging):
            #     if(kwargs["log_path_length"]):
            #         self.log_path_lengths(all_points*self.imsize, current_shape_idx=i)

            # print(torch.isfinite(all_points).all())
            # import pdb; pdb.set_trace()
            if("verbose" in kwargs):
                layer = self.raster(all_points, self.colors[i], verbose=kwargs['verbose'], white_background=False, imgsize=img_size)
            else:
                layer = self.raster(all_points, self.colors[i], white_background=False,imgsize=img_size)

            z_pred = self.z_order(shape_output)
            layers.append(layer)
            z_layers.append(torch.exp(z_pred[:, :, None, None]))
            if return_overlap_loss:
                loss = loss + self.control_polygon_distance(all_points)
        if(self.wandb_logging):
            wandb.log({"overlap_loss":loss})
        output = self.composite_fn(layers = layers, z_layers=z_layers)
        if return_points:
            return all_n_points
        if return_overlap_loss:
        #     overlap_alpha = layers[1][:, 3:4, :, :] + layers[2][:, 3:4, :, :]
        #     loss = F.relu(overlap_alpha - 1).mean()
            return output, loss
        return output

    def generate(self, x: Tensor, return_points=False, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        output = self.decode_and_composite(z, return_points=return_points,**kwargs) # might want to use verbose keyword here
        return output  # [:, :3]
    
    @torch.no_grad()
    def multishape_sample(self,n:int,return_points=False,device="cpu", **kwargs) -> Tensor:
        """
        Samples n images from the model
        :param n: (int) number of images to sample
        :return: (Tensor) [B x C x H x W]
        """
        z = torch.randn(n, self.latent_dim).to(device)
        output = [self.decode_and_composite(z[i].unsqueeze(0), return_points=return_points, **kwargs) for i in range(len(z))]
        return output

    def interpolate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(mu.shape[0]):
            z = self.interpolate_vectors(mu[2], mu[i], 10)
            output = self.decode_and_composite(z, verbose=kwargs['verbose'])
            all_interpolations.append(output)
        return all_interpolations

    def interpolate_mini(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        z = self.interpolate_vectors(mu[0], mu[1], 10)
        output = self.decode_and_composite(z, verbose=kwargs['verbose'])
        return output

    def interpolate2D(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        y_axis = self.interpolate_vectors(mu[7], mu[6], 10)
        for i in range(10):
            z = self.interpolate_vectors(y_axis[i], mu[3], 10)
            output = self.decode_and_composite(z, verbose=kwargs['verbose'])
            all_interpolations.append(output)
        return all_interpolations

    def naive_vector_interpolate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        bs = mu.shape[0]
        n = len(self.colors)
        for j in range(bs):
            layers = []
            z_rnn_input = mu[None, :, :].repeat(n, 1, 1)  # [len, batch size, emb dim]
            outputs, hidden = self.rnn(z_rnn_input)
            outputs = outputs.permute(1, 0, 2)  # [batch size, len, emb dim]
            outputs = outputs[:, :, :self.latent_dim] + outputs[:, :, self.latent_dim:]
            for i in range(n):
                shape_latent = self.divide_shape(outputs[:, i, :])
                all_points = self.decode(shape_latent)
                all_points_interpolate = self.interpolate_vectors(all_points[2], all_points[j], 10)
                layer = self.raster(all_points_interpolate, self.colors[i], verbose=kwargs['verbose'])
                layers.append(layer)
            # output = (layers[0][:, :3] * layers[0][:, 3:4, :, :] * (1 - layers[1][:, 3:4, :, :]) * (
            #             1 - layers[2][:, 3:4, :, :])) + \
            #          (layers[1][:, :3] * layers[1][:, 3:4, :, :] * (1 - layers[2][:, 3:4, :, :])) + \
            #          (layers[2][:, :3] * layers[2][:, 3:4, :, :]) + \
            #          ((1 - layers[0][:, 3:4, :, :]) * (1 - layers[1][:, 3:4, :, :]) * (1 - layers[2][:, 3:4, :, :]))
            # output = (layers[0][:, :3] * layers[0][:, 3:4, :, :] * (1 - layers[1][:, 3:4, :, :]) * (
            #             1 - layers[2][:, 3:4, :, :])) + \
            #          (layers[1][:, :3] * layers[1][:, 3:4, :, :]) + \
            #          (layers[2][:, :3] * layers[2][:, 3:4, :, :]) + \
            #          ((1 - layers[0][:, 3:4, :, :]) * (1 - layers[1][:, 3:4, :, :]) * (1 - layers[2][:, 3:4, :, :]))

            output = self.composite_fn(layers = layers)
            all_interpolations.append(output)
        return all_interpolations

    def visualize_sampling(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        all_interpolations = []
        for i in range(7, 25):
            self.redo_features(i)
            output = self.decode_and_composite(mu, verbose=kwargs['verbose'])
            all_interpolations.append(output)
        return all_interpolations


    def save(self, all_points, save_dir, name, verbose=False, white_background=True):
        # note that this if for a single shape and bs dimension should have multiple curves
        # print('1:', process.memory_info().rss*1e-6)
        render_size = self.imsize
        bs = all_points.shape[0]
        if verbose:
            render_size = render_size*2
        all_points = all_points*render_size
        num_ctrl_pts = torch.zeros(self.curves, dtype=torch.int32) + 2

        shapes = []
        shape_groups = []
        for k in range(bs):
            # Get point parameters from network
            color = make_tensor(color[k])
            points = all_points[k].cpu().contiguous()#[self.sort_idx[k]]

            if verbose:
                np.random.seed(0)
                colors = np.random.rand(self.curves, 4)
                high = np.array((0.565, 0.392, 0.173, 1))
                low = np.array((0.094, 0.310, 0.635, 1))
                diff = (high-low)/(self.curves)
                colors[:, 3] = 1
                for i in range(self.curves):
                    scale = diff*i
                    color = low + scale
                    color[3] = 1
                    color = torch.tensor(color)
                    num_ctrl_pts = torch.zeros(1, dtype=torch.int32) + 2
                    if i*3 + 4 > self.curves * 3:
                        curve_points = torch.stack([points[i*3], points[i*3+1], points[i*3+2], points[0]])
                    else:
                        curve_points = points[i*3:i*3 + 4]
                    path = pydiffvg.Path(
                        num_control_points=num_ctrl_pts, points=curve_points,
                        is_closed=False, stroke_width=torch.tensor(4))
                    path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([i]),
                        fill_color=None,
                        stroke_color=color)
                    shapes.append(path)
                    shape_groups.append(path_group)
                for i in range(self.curves * 3):
                    scale = diff*(i//3)
                    color = low + scale
                    color[3] = 1
                    color = torch.tensor(color)
                    if i%3==0:
                        # color = torch.tensor(colors[i//3]) #green
                        shape = pydiffvg.Rect(p_min = points[i]-8,
                                             p_max = points[i]+8)
                        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([self.curves+i]),
                                                           fill_color=color)

                    else:
                        # color = torch.tensor(colors[i//3]) #purple
                        shape = pydiffvg.Circle(radius=torch.tensor(8.0),
                                                 center=points[i])
                        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([self.curves+i]),
                                                           fill_color=color)
                    shapes.append(shape)
                    shape_groups.append(group)

            else:

                path = pydiffvg.Path(
                    num_control_points=num_ctrl_pts, points=points,
                    is_closed=True)

                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=color,
                    stroke_color=color)
                shape_groups.append(path_group)
        pydiffvg.save_svg(f"{save_dir}{name}/{name}.svg",
                          self.imsize, self.imsize, shapes, shape_groups)