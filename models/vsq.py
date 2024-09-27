from typing import Tuple, Union
import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import wandb
from utils import log_all_images, tensor_to_histogram_image, calculate_global_positions, shapes_to_drawing, svg_string_to_tensor, width_pred_to_local_stroke_width
from models.vsq_heads import MLPVectorHead, CNNVectorHead
from models.vq_vae import VectorQuantizer
from torchvision.models import ResNet, resnet18
from vector_quantize_pytorch import FSQ
from x_transformers import TransformerWrapper, Decoder
from transformers import BertModel
from svgwrite import Drawing
from einops import rearrange


class DeconvResNet(nn.Module):
    """
    This class only exists for debugging and validation purposes.
    It is used to validate if everything in the VSQ works when paired with a regular pixel-based prediction head.
    """
    def __init__(self):
        super(DeconvResNet, self).__init__()

        # Define layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = F.sigmoid(self.deconv5(x))  # Using sigmoid for the final layer to scale values between 0 and 1

        return x, {}


class VSQ(nn.Module):
    """
    Vector Shape/Stroke Quantizer. 
    Vector quantized pre-training of an autoencoder for SVG primitives.
    
    Input/Output are shape layers (or patches), no positions. Positions are leraned using the Transformer in Stage II.
    """

    def __init__(self,
                 vector_decoder_model: str = "mlp",
                 quantized_dim: int = 256,
                 codebook_size: int = 512,
                 patch_size:int=128,
                 image_loss: str = "pyramid",
                 num_codes_per_shape: int = 1,
                 vq_method:str = "fsq",
                 fsq_levels:list =[8,5,5,5],
                 num_segments:int = 1,
                #  geometric_constraint: str = None,
                 alpha: float = 0.0,
                 pred_color:bool=False,
                 dropout:float=0.1,
                 **kwargs) -> None:
        super(VSQ, self).__init__()

        assert vector_decoder_model in ["mlp", "raster_conv", "cnn"], "vector_decoder_model must be one of ['mlp', 'raster_conv', 'cnn']"
        # assert geometric_constraint in ["inner_distance", None], f"geometric_constraint must be one of ['inner_distance'], but was {geometric_constraint}"

        self.vector_decoder_model = vector_decoder_model
        self.quantized_dim = quantized_dim
        self.image_loss = image_loss
        self.vq_method = vq_method.lower()
        assert self.vq_method == "fsq", "Please use FSQ."
        self.fsq_levels = fsq_levels
        self.num_segments = num_segments
        self.num_codes_per_shape = num_codes_per_shape
        self.pred_color = pred_color
        self.patch_size = patch_size
        self.dropout = dropout

        if alpha > 0.0:
            self.geometric_constraint = "inner_distance"
            self.alpha = alpha
        else:
            self.geometric_constraint = "None"
            self.alpha = 0.0

        if self.vq_method == "fsq":
            self.codebook_size = np.prod(fsq_levels)
        else:
            self.codebook_size = codebook_size

        self.encoder = resnet18(num_classes = self.quantized_dim * self.num_codes_per_shape)

        if self.vq_method == "vqvae":
            self.quantize_layer = VectorQuantizer(num_embeddings = self.codebook_size,
                                                embedding_dim = self.quantized_dim,
                                                beta = 0.25)
        elif self.vq_method == "fsq":
            self.quantize_layer = FSQ(levels=self.fsq_levels,
                                      dim=self.quantized_dim)
        elif self.vq_method == "vqtorch":
            raise NotImplementedError("VQVAE with vqtorch not implemented yet.")
        else:
            raise ValueError(f"vq_method must be one of ['vqvae', 'fsq', 'vqtorch'], but is {self.vq_method}")

        
        self.latent_dim = self.quantized_dim
        
        if self.vector_decoder_model == "mlp":
            self.decoder = MLPVectorHead(latent_dim = self.quantized_dim * self.num_codes_per_shape,
                                              segments = self.num_segments,
                                              imsize = self.patch_size,
                                              max_stroke_width=20.,
                                              pred_color=self.pred_color,
                                              dropout=self.dropout,)
        elif self.vector_decoder_model == "cnn":
            self.decoder = CNNVectorHead(latent_dim = self.quantized_dim * self.num_codes_per_shape,
                                        segments = self.num_segments,
                                        imsize = self.patch_size,
                                        max_stroke_width=20.,
                                        pred_color=self.pred_color,)
        elif self.vector_decoder_model == "raster_conv":
            self.decoder = DeconvResNet()

    def encode(self, input: Tensor, quantize: bool = False):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) latent codes
        """
        result = self.encoder.forward(input)  # output from default resnet pytorch is (bs, self.quantized_dim * self.num_codes_per_shape)
        while result.dim() < 4:
            result = result.unsqueeze(-1)
        if self.num_codes_per_shape > 1:
            result = rearrange(result, 'b (c2 c) h w -> b c2 (c h) w', c2=self.quantized_dim)
        # result = self.mapping_layer(result.view(-1, 512 * 4 * 4))
        if quantize:
            result = self.quantize_layer.forward(result)  # this might change the result return type to list
        return result
    
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result, logging_dict = self.decoder.forward(z)
        # if self.vector_decoder_model == "mlp":
        #     result = result[0]  # extract only the raster image for now
        return result, logging_dict
    
    def decode_from_indices(self, idxs: Tensor) -> Union[Tensor, dict]:
        """
        Maps the given idxs to [reconstructions, input, all_points, vq_loss], all_points are the points of the bezier curves
        :param z: (Tensor) [B x 1]

        """
        if self.vq_method == "fsq":
            codes = self.quantize_layer.indices_to_codes(idxs)
        else:
            raise NotImplementedError("Only FSQ implemented for now.")
        # concat the codes
        if self.num_codes_per_shape > 1:
            b_dim = int(codes.shape[0] / self.num_codes_per_shape)
            codes = codes.view(b_dim, self.num_codes_per_shape, self.quantized_dim).permute(0, 2, 1).unsqueeze(-1)
            codes = rearrange(codes, 'b d (c h) w -> b (d c) h w', c=self.num_codes_per_shape)
            codes = codes.view(b_dim, self.quantized_dim * self.num_codes_per_shape)

        result, logging_dict = self.decode(codes)
        # if self.vector_decoder_model == "mlp":
        #     result = result[0]  # extract only the raster image for now
        return result, logging_dict
    
    
    def forward(self, input: Tensor, logging = False, return_visual_attributes = False, only_return_recons = False, **kwargs):
        """
        visual_attribute_dict = {
            "stroke_widths" : all_widths,
            "alphas" : all_alphas,
            "colors": all_colors
        }
        """
        logging_dict = {}
        encoding = self.encode(input, quantize=False)
        bs = encoding.shape[0]
        vq_logging_dict={}
        if self.vector_decoder_model in ["mlp", "cnn"]:
            # quantize the encoding
            if self.vq_method == "vqvae":
                quantized_inputs, vq_loss, vq_logging_dict = self.quantize_layer.forward(encoding, logging=logging)
            elif self.vq_method == "fsq":
                quantized_inputs, indices = self.quantize_layer.forward(encoding)
                vq_loss = torch.tensor(0.)
                if logging:
                    vq_logging_dict = {"codebook_histogram":wandb.Image(tensor_to_histogram_image(indices.detach().flatten().cpu()), caption="histogram of codebook indices")}
                    
            # flatten it for MLP digestion
            # quantized_inputs = quantized_inputs.permute(0,2,1,3)
            quantized_inputs = rearrange(quantized_inputs, 'b d (c h) w -> b (d c) h w', c=self.num_codes_per_shape)
            quantized_inputs = quantized_inputs.view(bs, self.quantized_dim * self.num_codes_per_shape)
            # print("quantized_inputs: ", quantized_inputs.shape)
        elif self.vector_decoder_model == "raster_conv":
            quantized_inputs, vq_loss = self.quantize_layer(encoding)
        
        # re-merge the quantized codes
        # quantized_inputs = rearrange(quantized_inputs, 'b d (c h) w -> b (d c) h w', c=self.num_codes_per_shape)
        out, decode_logging_dict = self.decode(quantized_inputs)  # for mlp out is [output, scenes, all_points, all_widths]
        reconstructions = out[0]
        all_points = out[2]
        visual_attribute_dict = out[3]
        logging_dict = {**logging_dict, **decode_logging_dict, **vq_logging_dict}
        if only_return_recons:
            return reconstructions
        if return_visual_attributes:
            return [reconstructions, input, all_points, vq_loss, visual_attribute_dict], logging_dict
        else:
            return [reconstructions, input, all_points, vq_loss], logging_dict
    
    def gaussian_pyramid_loss(self, recons_images: Tensor, gt_images: Tensor, down_sample_steps: int = 3, log_loss: bool = False, pyramid_weights:Tensor = None):
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
        timesteps_to_log = 4
        recon_loss = F.mse_loss(recons_images, gt_images, reduction='none')
        if log_loss:
            all_loss_images = []
            all_loss_images.append(self.transform_loss_tensor_to_image(recon_loss[:timesteps_to_log]))
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
            if log_loss:
                all_loss_images.append(self.transform_loss_tensor_to_image(loss_images[:timesteps_to_log]))

            curr_pyramid_loss = loss_images.mean() * weight
            recons_loss_contributions[f"pyramid_loss_step_{j}"] = curr_pyramid_loss.detach().cpu().item()
            recon_loss = recon_loss + curr_pyramid_loss

        if log_loss:
            log_all_images(all_loss_images, log_key="pyramid loss", caption=f"Gaussian Pyramid Loss, {down_sample_steps+1} steps")
            wandb.log(recons_loss_contributions)
        return recon_loss, recons_loss_contributions

    def _get_mean_inner_distance(self,
                        points: Tensor,
                        use_neighbors_only:bool=False) -> Tensor:
        """
        mean inner distance is defined as the distance between start and end point of each segment of the path

        returns batched mean
        """
        inner_dists = []
        for i in range(self.num_segments if use_neighbors_only else self.num_segments+1):
            if use_neighbors_only:
                inner_dist = torch.cdist(points[:,:,i*3,:], points[:,:,(i+1)*3,:], p=2)
                inner_dists.append(inner_dist.mean())
            else:
                inner_dist = None
                for j in range(self.num_segments+1):
                    if i != j:
                        ij_dist = torch.cdist(points[:,:,i*3,:], points[:,:,j*3,:], p=2)
                        inner_dist = inner_dist + ij_dist if inner_dist is not None else ij_dist
                inner_dists.append(inner_dist.squeeze() / self.num_segments)
        return torch.mean(torch.stack(inner_dists, dim=1), dim=1)


    def _get_inner_distance_penalty(self, points:Tensor):
        """
        input: points, Tensor, (bs, 3*num_segments+1, 2)

        inner distance penalty punishes points that are non-equally distributed. 
        it does this by calculating the mean scaled distance between each point and all other points
        """
        inner_penalties = []
        for j in range(self.num_segments + 1):
            inner_dists = []
            for i in range(self.num_segments + 1):
                if i != j:
                    ij_dist = torch.cdist(points[:,:,i*3,:], points[:,:,j*3,:], p=2)
                    # by scaling the distance by the inverse of the nieghborhood distance, we make sure everything is equidistant
                    ij_dist = ij_dist * (1/abs(i-j))
                    inner_dists.append(ij_dist)
            mean_inner_dist = torch.mean(torch.stack(inner_dists, dim=1), dim=1)
            # inner penalty is the deviation from the mean squared
            inner_penalty = torch.mean(torch.square(torch.stack(inner_dists).squeeze() - mean_inner_dist.squeeze()))
            inner_penalties.append(inner_penalty)
        return torch.mean(torch.stack(inner_penalties))


    def loss_function(self,
                      reconstructions: Tensor,
                      gt_images: Tensor,
                      vq_loss: Tensor,
                      points: Tensor,
                      log_loss: bool = False,
                      **kwargs) -> dict:
        if self.image_loss == "mse":
            recons_loss = F.mse_loss(reconstructions, gt_images)
            recons_loss_contributions = {}
        elif self.image_loss == "pyramid":
            recons_loss, recons_loss_contributions = self.gaussian_pyramid_loss(reconstructions, gt_images, down_sample_steps=3, log_loss=log_loss, pyramid_weights=kwargs["pyramid_weights"])
        else:
            raise NotImplementedError("Only mse and pyramid loss implemented for now.")
        if self.geometric_constraint == "inner_distance":
            inner_distance_penalty = self._get_inner_distance_penalty(points)
            scaled_geometric_loss = inner_distance_penalty
            geometric_loss = inner_distance_penalty
        else:
            geometric_loss = 0.0
            scaled_geometric_loss = 0.0
        
        loss = recons_loss + vq_loss + self.alpha * scaled_geometric_loss

        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                "geometric_loss":geometric_loss,
                "frac_black_scaled_geometric_loss" : scaled_geometric_loss,
                self.geometric_constraint+"_loss" : self.alpha * scaled_geometric_loss,
                # "frac_black": (1-gt_images).mean(),
                'VQ_Loss':vq_loss,
                **recons_loss_contributions}
    

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    
    @torch.no_grad()
    def reconstruct(self, 
                    patches: Tensor, 
                    gt_center_positions: Tensor, 
                    padded_individual_max_length: float, 
                    local_stroke_width: float = None,
                    rendered_w = 128., 
                    return_shapes:bool=False,
                    return_local_points:bool = False) -> Union[Drawing, Tensor]:
        """
        Reconstructs the input patches and uses gt positions to assemble them into a full SVG. Can be used to observe quality degradation of the quantization process.
        
        Args:
            - patches (Tensor): Input patches to be reconstructed
            - gt_center_positions (Tensor): Ground truth center positions of the patches
            - padded_individual_max_length (float): Padded individual max length of the patches, usually is individual_max_length+2
            - local_stroke_width (float): effects only reconstructed SVG, override the prediction of the model with a fixed stroke width

        Returns:
            - reconstructed_drawing (Drawing): Reconstructed drawing (use to save svg)
            - rasterized_reconstructions (Tensor): Rasterized reconstructions
        """
        [reconstructions, input, all_points, vq_loss, visual_attribute_dict], logging_dict = self.forward(patches, logging = False, return_visual_attributes=True)
        # these need to be scaled with 72 to keep the original viewbox aspect ratios intact
        if gt_center_positions.max() < 1.0:
            gt_center_positions = gt_center_positions * 72

        global_shapes = calculate_global_positions(all_points, padded_individual_max_length, gt_center_positions)[:,0]

        # scale back into [0,1] range
        if global_shapes.max() > 1.0:
            global_shapes = global_shapes / 72

        if local_stroke_width is not None:
            local_stroke_widths = torch.ones_like(visual_attribute_dict["stroke_widths"]) * local_stroke_width
        else:
            local_stroke_widths = width_pred_to_local_stroke_width(visual_attribute_dict["stroke_widths"],
                                                                    self.patch_size,
                                                                    padded_individual_max_length)
        global_stroke_widths = local_stroke_widths / padded_individual_max_length * 72
        visual_attribute_dict["local_stroke_widths"] = local_stroke_widths
        visual_attribute_dict["global_stroke_widths"] = global_stroke_widths

        try:
            # reconstructed_drawing = shapes_to_drawing(global_shapes, stroke_width=stroke_widths, w=rendered_w)
            # the misconception here is that we need global stroke width, which is WRONG. That would look as thick as the local strokes, which is not desired
            reconstructed_drawing = shapes_to_drawing(global_shapes, stroke_width=local_stroke_widths, visual_attribute_dict=visual_attribute_dict, w=rendered_w)
            # rasterized_reconstructions = svg_string_to_tensor(reconstructed_drawing.tostring())
        except Exception as e:
            print("Error during reconstruction: ", e)
            print(f"Got max of: {global_shapes.max()} Limited shapes to [0,1]")
            global_shapes = torch.clamp(global_shapes, 0, 1)
            reconstructed_drawing = shapes_to_drawing(global_shapes, global_stroke_widths, visual_attribute_dict=visual_attribute_dict, w=rendered_w)
            # rasterized_reconstructions = None
        if return_shapes:
            if return_local_points:
                return reconstructed_drawing, reconstructions, global_shapes, visual_attribute_dict, all_points
            return reconstructed_drawing, reconstructions, global_shapes, visual_attribute_dict
        else:
            return reconstructed_drawing, reconstructions