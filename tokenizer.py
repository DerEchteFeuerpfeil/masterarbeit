"""
This file contains the tokenizers that are required to train the second stage, the auto-regressive transformer.
They utilize the VSQ to turn patches into tokens and contain all the rest of the logic for positions, text conditioning and special tokens.
"""

from typing import Iterable, List, Tuple, Union
from utils import calculate_global_positions, shapes_to_drawing, drawing_to_tensor
from models.vsq import VSQ
from svg_fixing import get_fixed_svg_render, get_fixed_svg_drawing

import numpy as np
import torch
from torch import Tensor
from svgwrite import Drawing
from transformers import BertTokenizer, BertModel,PreTrainedTokenizerBase
from torch import nn

class VQTokenizer(nn.Module):
    """
    Tokenizer for the stroke-based VSQ model. 
    It tokenizes the patches of the rasterized SVGs and their middle positions + some special tokens + text conditioning.

    Args:
        - vq_model (VSQ): VSQ model to use for patch tokenization
        - full_image_res (int): Full resolution of the rasterized SVGs
        - tokens_per_patch (int): Number of tokens per patch
        - text_encoder_str (str): huggingface string of the BERT text encoder to use, default: google/bert_uncased_L-12_H-512_A-8
        - device (str, optional): Device to use. Defaults to "cpu".
        - use_text_encoder_only (bool, optional): Whether to use the text encoder only. Defaults to False. Used to benefit from special token mapping and text tokenization without the need for a VSQ model.
    """

    def __init__(self, 
                 vq_model: VSQ, 
                 full_image_res: int, 
                 tokens_per_patch:int, 
                 text_encoder_str: str = "google/bert_uncased_L-12_H-512_A-8",
                 device = "cpu",
                 use_text_encoder_only: bool = False,
                 codebook_size:int = None,
                 lseg:float = 3.0,
                 max_text_token_length:int=50,
                 **kwargs) -> None:

        super(VQTokenizer, self).__init__()
        self.text_encoder_str = text_encoder_str
        self.full_image_res = full_image_res
        self.tokens_per_patch = tokens_per_patch
        self.max_num_pos_tokens = self.full_image_res ** 2  # for now this is just resolution squared, could be quantized to a smaller number of positions later
        self.device = device
        self.lseg = lseg
        self.use_text_encoder_only = use_text_encoder_only
        self.max_text_token_length = max_text_token_length
        if self.use_text_encoder_only:
            self.vq_model = None
            self.codebook_size = codebook_size
        else:
            self.vq_model = vq_model.to(device)
            self.codebook_size = self.vq_model.codebook_size
        
        self.text_tokenizer: PreTrainedTokenizerBase = BertTokenizer.from_pretrained(self.text_encoder_str)
        assert self.text_tokenizer.vocab_size < 65535, "VQTokenizer only supports 16-bit np.ushort encoded tokens, but the text tokenizer exceeds that."

        self.bert_cls_token = self.text_tokenizer.get_vocab().get("[CLS]")
        self.bert_sep_token = self.text_tokenizer.get_vocab().get("[SEP]")
        self.bert_pad_token = self.text_tokenizer.get_vocab().get("[PAD]")

        # CLS and SEP are handled by the text embedding model
        self.special_token_mapping = {
            "<SOS>": 0,  # start of sequence
            "<BOS>": 1,  # beginning of SVG, separates text tokens from SVG
            "<EOS>": 2,  # end of sequence
            "<PAD>": 3,  # padding
        }

        self.start_of_patch_token_idx = len(self.special_token_mapping)
        self.start_of_pos_token_idx = self.start_of_patch_token_idx + self.codebook_size 
        self.num_tokens = self.start_of_pos_token_idx + self.max_num_pos_tokens
        
    def _is_position(self, token: int) -> bool:
        return token >= self.start_of_pos_token_idx and token <= self.num_tokens

    def _is_patch(self, token: int) -> bool:
        return token >= self.start_of_patch_token_idx and token < self.start_of_pos_token_idx
    
    def _get_patch_idx_range(self) -> Tuple[int, int]:
        return self.start_of_patch_token_idx, self.start_of_pos_token_idx
    
    def _get_pos_idx_range(self) -> Tuple[int, int]:
        return self.start_of_pos_token_idx, self.num_tokens
    
    def tokenize_patches(self, patches: Tensor) -> Tensor:
        """
        Tokenizes the patches of the rasterized SVGs.

        Args:
            patches (Tensor): Tensor of shape (num_patches, channels, patch_res, patch_res)

        Returns:
            Tensor: Tensor of shape (num_patches, self.tokens_per_patch)
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Tokenizing patches is not supported when using the text encoder only.")
        with torch.no_grad():
            _, indices = self.vq_model.encode(patches, quantize=True)
        indices = indices.flatten().to(self.device)
        return indices + self.start_of_patch_token_idx
    
    def tokenize_positions(self, positions: Tensor) -> Tensor:
        """
        Tokenizes the positions of the patches of the rasterized SVGs.

        Args:
            positions (Tensor): Tensor of shape (num_pos, 2)

        Returns:
            Tensor: Tensor of shape (num_pos, 1)
        """
        #NOTE this currently assumes that all positions are scaled in range [0, self.full_image_res]
        if self.use_text_encoder_only:
            raise NotImplementedError("Tokenizing positions is not supported when using the text encoder only.")
        assert positions.mean() > 1., f"Positions should be scaled with the full image resolution already, got mean: {positions.mean()}"
        positions = positions[:, 0].round() + self.full_image_res * positions[:, 1].round()
        return positions + self.start_of_pos_token_idx
    
    def tokenize_text(self, text: str, add_padding=False, return_attention_mask:bool=False) -> Tensor:
        """
        Tokenizes the conditional text.

        Args:
            text (str): Text to tokenize

        Returns:
            Tensor: Tensor of shape (num_tokens) without any padding but with special tokens [CLS] and [SEP]
        """
        tokens = torch.tensor(self.text_tokenizer.encode(text, add_special_tokens=True), device = self.device)
        atten = torch.ones_like(tokens)
        if add_padding and len(tokens) < self.max_text_token_length:
            padding = torch.tensor([self.bert_pad_token] * (self.max_text_token_length - len(tokens)), device=self.device, dtype=torch.int64)
            tokens = torch.cat([tokens, padding])
            atten = torch.cat([atten, torch.zeros_like(padding)])
        
        if return_attention_mask:
            return tokens, atten
        return tokens

    def forward(self):
        pass
    
    def tokenize(self, patches: Tensor, positions: Tensor, text:str, return_np_uint16:bool = False) -> Union[Tensor, Tensor] | Union[np.ndarray, np.ndarray]:
        """
        Tokenizes the patches and positions of the rasterized SVGs. Padding is done in the dataloader dynamically to avoid requiring a fixed context length during pre-tokenization.

        Args:
            - patches (Tensor): Tensor of shape (num_patches, channels, patch_res, patch_res)
            - positions (Tensor): Tensor of shape (num_pos, 2)
            - text (str): conditional text
            - return_np_uint16 (bool, optional): Whether to return the tokens as np.uint16. Defaults to False.
            - batched (bool, optional): Whether the input is batched or not.

        Returns:
            - start_token: [<SOS>], either Tensor or np.ndarray
            - text_tokens: [<CLS>, ...text..., <SEP>], no padding, CLS and SEP come from text tokenizer, either Tensor or np.ndarray
            - vq_tokens: [<BOS>, patch_tokens, pos_token, patch_tokens, pos_token, ...], no padding, either Tensor or np.ndarray
            - end_token: [<EOS>], either Tensor or np.ndarray
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Tokenizing patches/positions is not supported when using the text encoder only.")
        patch_tokens = self.tokenize_patches(patches).cpu()
        pos_tokens = self.tokenize_positions(positions)
        text_tokens = self.tokenize_text(text)
        if self.tokens_per_patch == 1:
            vq_tokens = torch.stack([patch_tokens, pos_tokens], dim=1).reshape(-1).int()
        else:
            result = []
            for i in range(1, len(patch_tokens) + len(pos_tokens)):
                if len(patch_tokens) + len(pos_tokens) == 0:
                    continue
                if i%(self.tokens_per_patch+1) == 0 and i>0 and len(pos_tokens) > 0:
                    result.append(pos_tokens[0])
                    pos_tokens = pos_tokens[1:]
                else:
                    if len(patch_tokens) > 0:
                        result.append(patch_tokens[0])
                        patch_tokens = patch_tokens[1:]
            if len(pos_tokens) == 1:
                result.append(pos_tokens[0])

            vq_tokens = torch.tensor(result).reshape(-1).int()
            # raise NotImplementedError("Merging not implemented for tokens_per_patch > 1")
        
        # NOTE: this is now done manually in the tokenization script as <SOS> needs to be put before the text tokens but I want to keep text and SVG tokens separate
        start_token = (self.special_token_mapping["<SOS>"]) * torch.ones(1).int()
        end_token = (self.special_token_mapping["<EOS>"]) * torch.ones(1).int()
        bos_token = (self.special_token_mapping["<BOS>"]) * torch.ones(1).int()

        vq_tokens = torch.cat([bos_token, vq_tokens], dim=0)

        # final_tokens = torch.cat([start_token, vq_tokens, end_token], dim=0)

        if return_np_uint16:
            vq_tokens = vq_tokens.cpu().numpy().astype(np.ushort)
            text_tokens = text_tokens.cpu().numpy().astype(np.ushort)
            start_token = start_token.cpu().numpy().astype(np.ushort)
            end_token = end_token.cpu().numpy().astype(np.ushort)
        
        return start_token, text_tokens, vq_tokens, end_token
    
    def decode_patches(self, tokens: Tensor, raster:bool = False, return_visual_attribute_dict:bool=False) -> Tensor:
        """
        Decodes the patches from the tokens into bezier points.

        Args:
            tokens (Tensor): Tensor of shape (num_patches, self.tokens_per_patch)
            raster (bool, optional): Whether to return the rasterized patches. Defaults to False.

        Returns:
            Tensor: Tensor of shape (num_patches, channels, patch_res, patch_res)
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Decoding patches is not supported when using the text encoder only.")
        with torch.no_grad():
            out, _ = self.vq_model.decode_from_indices(tokens - self.start_of_patch_token_idx)
        if raster:
            if return_visual_attribute_dict:
                return out[0], out[3]
            return out[0]
        else:
            if return_visual_attribute_dict:
                return out[2], out[3]
            return out[2]
    
    def decode_positions(self, tokens: Tensor) -> Tensor:
        """
        Decodes the positions from the tokens.

        Args:
            tokens (Tensor): Tensor of shape (num_pos, 1)

        Returns:
            Tensor: Tensor of shape (num_pos, 2)
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Decoding positions is not supported when using the text encoder only.")
        assert torch.logical_and(tokens >= self.start_of_pos_token_idx, tokens <= self.num_tokens).all(), f"Position tokens should be in range [{self.start_of_pos_token_idx}, {self.num_tokens}], got {tokens}"
        tokens = tokens - self.start_of_pos_token_idx
        positions = torch.stack([tokens % self.full_image_res, tokens // self.full_image_res], dim=1)
        return positions
    
    def decode_text(self, tokens: Tensor) -> str:
        """
        Decodes the text from the tokens.

        Args:
            tokens (Tensor): Tensor of shape (num_tokens)

        Returns:
            str: Decoded text
        """
        text = self.text_tokenizer.decode(tokens, skip_special_tokens=True)
        return text
    
    def _generate_indices(self, length, count, start=0):
        indices = []
        i = start
        while i + count < length:
            indices.extend(range(i, i + count))  # Select 'count' consecutive numbers
            i += count + 1  # Skip one and move to the next block
        return indices
    
    def decode(self, tokens: Tensor, ignore_special_tokens: bool = False, return_visual_attribute_dict:bool=False):
        """
        Decodes the patches and positions from the tokens.

        Args:
            tokens (Tensor): Tensor of shape (num_tokens)
            ignore_special_tokens (bool, optional): Whether to ignore the required special tokens like BOS and EOS. Defaults to False.

        Returns:
            Tuple[Tensor, Tensor]: Tuple of tensors of shape (num_patches, channels, patch_res, patch_res) and (num_pos, 2)
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Decoding patches/positions is not supported when using the text encoder only.")
        # remove all occurence of <PAD> token
        tokens = tokens[tokens != self.special_token_mapping["<PAD>"]]

        assert tokens.ndim == 1, f"Tokens should be 1D, got shape {tokens.shape}"
        assert tokens.size(0) > 3, f"Tokens should have at least 4 elements, got {tokens.size(0)}"
        if not ignore_special_tokens:
            assert tokens[0] == self.special_token_mapping["<BOS>"], f"First token should be <BOS>, got {tokens[0]}"
            assert tokens[-1] == self.special_token_mapping["<EOS>"] or tokens[-1] == self.special_token_mapping["<PAD>"], f"Last token should be <EOS> or <PAD>, got {tokens[-1]}"
        if tokens[-1] == self.special_token_mapping["<EOS>"]:
            tokens = tokens[:-1]
        if tokens[0] == self.special_token_mapping["<BOS>"]:
            tokens = tokens[1:]
        while self._is_patch(tokens[-1]):
            # print("[INFO] Last token is a patch token, removing it.")
            tokens = tokens[:-1]
        if self.tokens_per_patch == 1:
            assert tokens.size(0) % 2 == 0, f"Number of tokens should be even, got {tokens.size(0)}"
        
        patch_idx = self._generate_indices(len(tokens), self.tokens_per_patch, start=0)
        patch_tokens = tokens[patch_idx]
        pos_tokens = tokens[self.tokens_per_patch::self.tokens_per_patch+1]
        positions = self.decode_positions(pos_tokens)
        if return_visual_attribute_dict:
            bezier_points, visual_attribute_dict = self.decode_patches(patch_tokens, return_visual_attribute_dict=True)
            return bezier_points, positions, visual_attribute_dict
        else:
            bezier_points = self.decode_patches(patch_tokens)
            return bezier_points, positions
    
    def assemble_svg(self, bezier_points: Tensor, center_positions: Tensor, stroke_width: float, w=128., padded_lseg = None, num_strokes_to_paint:int = 0,visual_attribute_dict=None) -> Drawing:
        """
        center_positions: scaled with self.full_image_res
        berzier_points: in [0,1]
        """
        if center_positions.max() < 72 and self.full_image_res > 72:
            print("[WARNING] Center positions are not scaled with the full image resolution, might already be scaled to original 72 viewbox, could yield errors.")
        padded_lseg = padded_lseg or self.lseg + 2
        # this is really dodgy but the issue is that the local viewbox and the original viewbox (72x72 for the simplified svgs) have to be consistent
        # by allowing variable position granularity through full_image_res, we have to do this conversion to properly get the the global positions
        original_center_positions = center_positions / self.full_image_res * 72
        global_shapes = calculate_global_positions(bezier_points, padded_lseg, original_center_positions)[:,0]
        relative_global_shapes = global_shapes / 72
        # manually fix errors that can happen in very early stages of Transformer training where non-compatible shape/pos combinations are sampled
        relative_global_shapes[relative_global_shapes>1.0] = 1.0
        reconstructed_drawing = shapes_to_drawing(relative_global_shapes, stroke_width=stroke_width, w=w, num_strokes_to_paint=num_strokes_to_paint, visual_attribute_dict=visual_attribute_dict)
        return reconstructed_drawing
    
    def _tokens_to_svg_drawing(self, 
                               tokens:Tensor,
                               global_stroke_width:float = 0.7,
                               post_process:bool = True, 
                               num_strokes_to_paint: int = 0, 
                               w=480, 
                               max_dist_frac=0.0625, 
                               method="min_dist_clip", 
                               **kwargs):
        tokens = tokens.clone().detach()
        bezier_points, positions, visual_attribute_dict = self.decode(tokens, ignore_special_tokens=True, return_visual_attribute_dict=True)
        padded_lseg = self.lseg + 2
        # # WRONG
        # stroke_width = self.lseg / 3.0 * 0.4
        if kwargs.get("max_dist") is not None:
            max_dist = kwargs.get("max_dist")
            kwargs.pop("max_dist")
        else:
            max_dist = max_dist_frac * self.full_image_res
        original_center_positions = positions / self.full_image_res * 72
        if post_process:
            drawing = get_fixed_svg_drawing(bezier_points, original_center_positions, method, global_stroke_width, padded_lseg, width=w, max_dist=max_dist, 
                                            num_strokes_to_paint=num_strokes_to_paint, max_position_value=72,
                                            visual_attribute_dict=visual_attribute_dict, **kwargs)
        else:
            drawing = self.assemble_svg(bezier_points, positions, global_stroke_width, w=w, padded_lseg=padded_lseg, num_strokes_to_paint=num_strokes_to_paint,visual_attribute_dict=visual_attribute_dict)
        return drawing
    
    def _tokens_to_image_tensor(self, 
                                tokens:Tensor, 
                                post_process:bool = True, 
                                num_strokes_to_paint: int = 0,
                                render_w:int = 480) -> Tensor:
        tokens = tokens.clone().detach()
        bezier_points, positions, visual_attribute_dict = self.decode(tokens, ignore_special_tokens=True, return_visual_attribute_dict=True)
        padded_lseg = self.lseg + 2
        stroke_width = self.lseg / 3.0 * 0.4
        original_center_positions = positions / self.full_image_res * 72
        if post_process:
            return_tensor = get_fixed_svg_render(bezier_points, original_center_positions, "min_dist_clip", stroke_width, padded_lseg, render_w, 4.5, 
                                                 num_strokes_to_paint=num_strokes_to_paint, max_position_value=72,
                                                 visual_attribute_dict=visual_attribute_dict)
        else:
            drawing = self.assemble_svg(bezier_points, positions, stroke_width, w=render_w, padded_lseg=padded_lseg, num_strokes_to_paint=num_strokes_to_paint,visual_attribute_dict=visual_attribute_dict)
            return_tensor = drawing_to_tensor(drawing)
        return return_tensor
    
class RasterVQTokenizer(nn.Module):
    """
    Tokenizer for the VSQ. It tokenizes the patches of the raster image and their center positions + some special tokens + text conditioning.

    Args:
        - vq_model (Vector_VQVAE): VQVAE model to use for patch tokenization
        - full_image_res (int): Full resolution of the rasterized SVGs
        - tokens_per_patch (int): Number of tokens per patch
        - text_encoder_str (str): huggingface string of the BERT text encoder to use, default: bert-base-uncased
        - device (str, optional): Device to use. Defaults to "cpu".
        - use_text_encoder_only (bool, optional): Whether to use the text encoder only. Defaults to False. Used to bnenefit from special token mapping and text tokenization without the need for a VQVAE model.
    """

    def __init__(self, 
                 vq_model: VSQ, 
                 patch_size: int,
                 num_tiles_per_row:int, 
                 tokens_per_patch:int,
                 do_tokenize_positions: bool = True, 
                 text_encoder_str: str = "bert-base-uncased", 
                 device = "cpu",
                 use_text_encoder_only: bool = False,
                 codebook_size:int = None,
                 **kwargs) -> None:

        super(RasterVQTokenizer, self).__init__()
        self.text_encoder_str = text_encoder_str
        self.patch_size = patch_size
        self.tokens_per_patch = tokens_per_patch
        self.num_tiles_per_row = num_tiles_per_row
        self.max_num_pos_tokens = self.num_tiles_per_row ** 2
        self.device = device
        self.do_tokenize_positions = do_tokenize_positions
        self.use_text_encoder_only = use_text_encoder_only
        self.full_image_res = patch_size * num_tiles_per_row
        if self.use_text_encoder_only:
            self.vq_model = None
            self.codebook_size = codebook_size
        else:
            self.vq_model = vq_model.to(device)
            self.codebook_size = self.vq_model.codebook_size
        
        self.text_tokenizer: PreTrainedTokenizerBase = BertTokenizer.from_pretrained(self.text_encoder_str)
        assert self.text_tokenizer.vocab_size < 65535, "VQTokenizer only supports 16-bit np.ushort encoded tokens, but the text tokenizer exceeds that."

        # CLS and SEP are handled by the text embedding model
        self.special_token_mapping = {
            "<SOS>": 0,  # start of sequence
            "<BOS>": 1,  # beginning of SVG, separates text tokens from SVG
            "<EOS>": 2,  # end of sequence
            "<PAD>": 3,  # padding
        }

        self.all_possible_positions = self._calculate_patch_centers(self.patch_size, self.num_tiles_per_row)

        self.start_of_patch_token_idx = len(self.special_token_mapping)
        self.start_of_pos_token_idx = self.start_of_patch_token_idx + self.codebook_size
        self.num_tokens = self.start_of_pos_token_idx + self.max_num_pos_tokens
        
    def _is_position(self, token: int) -> bool:
        return token >= self.start_of_pos_token_idx and token <= self.num_tokens

    def _is_patch(self, token: int) -> bool:
        return token >= self.start_of_patch_token_idx and token < self.start_of_pos_token_idx
    
    def _get_patch_idx_range(self) -> Tuple[int, int]:
        return self.start_of_patch_token_idx, self.start_of_pos_token_idx
    
    def _get_pos_idx_range(self) -> Tuple[int, int]:
        return self.start_of_pos_token_idx, self.num_tokens
    
    def tokenize_patches(self, patches: Tensor) -> Tensor:
        """
        Tokenizes the patches of the rasterized SVGs.

        Args:
            patches (Tensor): Tensor of shape (num_patches, channels, patch_res, patch_res)

        Returns:
            Tensor: Tensor of shape (num_patches, self.tokens_per_patch)
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Tokenizing patches is not supported when using the text encoder only.")
        with torch.no_grad():
            _, indices = self.vq_model.encode(patches, quantize=True)
        indices = indices.flatten().cpu()
        return indices + self.start_of_patch_token_idx
    
    def _calculate_patch_centers(self, patch_size, num_tiles_per_row):
        """
        Calculate the center positions of patches in an image.

        indexing is "ij", so the first dimension is the y-axis and the second dimension is the x-axis.
        """

        indices = torch.arange(0, num_tiles_per_row)

        # Calculate the center positions
        centers = (indices * patch_size + patch_size // 2).float()

        # Create a grid of center positions
        grid_x, grid_y = torch.meshgrid(centers, centers, indexing="xy")
        center_positions = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

        return center_positions

    def tokenize_positions(self, positions: Tensor) -> Tensor:
        """
        Tokenizes the positions of the patches of the raster image.

        Args:
            positions (Tensor): Tensor of shape (num_pos, 2)

        Returns:
            Tensor: Tensor of shape (num_pos, 1)
        """

        if self.use_text_encoder_only:
            raise NotImplementedError("Tokenizing positions is not supported when using the text encoder only.")
        idxs = []
        for pos in positions:
            try:
                idx = torch.where((self.all_possible_positions == pos.float()).all(dim=1))[0].item()
                idxs.append(idx)
            except:
                raise ValueError(f"Position {pos} not found in possible positions: ", self.all_possible_positions)
        position_tokens = torch.stack(idxs).int()
        return position_tokens + self.start_of_pos_token_idx
    
    def tokenize_text(self, text: str) -> Tensor:
        """
        Tokenizes the conditional text.

        Args:
            text (str): Text to tokenize

        Returns:
            Tensor: Tensor of shape (num_tokens) without any padding but with special tokens [CLS] and [SEP]
        """
        tokens = torch.tensor(self.text_tokenizer.encode(text, add_special_tokens=True), device = self.device)
        return tokens.cpu()

    def forward(self):
        pass
    
    def tokenize(self, patches: Tensor, text:str, return_np_uint16:bool = False, positions: Tensor = None) -> Union[Tensor, Tensor] | Union[np.ndarray, np.ndarray]:
        """
        Tokenizes the patches and positions of the rasterized SVGs. Padding is done in the dataloader dynamically to avoid requiring a fixed context length during pre-tokenization.

        Args:
            - patches (Tensor): Tensor of shape (num_patches, channels, patch_res, patch_res)
            - text (str): conditional text
            - return_np_uint16 (bool, optional): Whether to return the tokens as np.uint16. Defaults to False.
            - positions (Tensor): Tensor of shape (num_pos, 2), pass None for automatic calculation

        Returns:
            - start_token: [<SOS>], either Tensor or np.ndarray
            - text_tokens: [<CLS>, ...text..., <SEP>], no padding, CLS and SEP come from text tokenizer, either Tensor or np.ndarray
            - vq_tokens: [<BOS>, patch_tokens, pos_token, patch_tokens, pos_token, ...], no padding, either Tensor or np.ndarray
            - end_token: [<EOS>], either Tensor or np.ndarray
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Tokenizing patches/positions is not supported when using the text encoder only.")
        patch_tokens = self.tokenize_patches(patches).cpu()
        if positions is not None:
            pos_tokens = self.tokenize_positions(positions)
        else:
            pos_tokens = torch.tensor([]).int()
        text_tokens = self.tokenize_text(text)
        if self.tokens_per_patch == 1:
            vq_tokens = torch.stack([patch_tokens, pos_tokens], dim=1).reshape(-1).int() if positions is not None else patch_tokens
        else:
            raise NotImplementedError("Merging not implemented for tokens_per_patch > 1")
        
        # NOTE: this is now done manually in the tokenization script as <SOS> needs to be put before the text tokens but I want to keep text and SVG tokens separate
        start_token = (self.special_token_mapping["<SOS>"]) * torch.ones(1).int()
        end_token = (self.special_token_mapping["<EOS>"]) * torch.ones(1).int()
        bos_token = (self.special_token_mapping["<BOS>"]) * torch.ones(1).int()

        vq_tokens = torch.cat([bos_token, vq_tokens], dim=0)

        # final_tokens = torch.cat([start_token, vq_tokens, end_token], dim=0)

        if return_np_uint16:
            vq_tokens = vq_tokens.numpy().astype(np.ushort)
            text_tokens = text_tokens.numpy().astype(np.ushort)
            start_token = start_token.numpy().astype(np.ushort)
            end_token = end_token.numpy().astype(np.ushort)
        
        return start_token, text_tokens, vq_tokens, end_token
    
    def decode_patches(self, tokens: Tensor, raster:bool = False) -> Tensor:
        """
        Decodes the patches from the tokens into bezier points.

        Args:
            tokens (Tensor): Tensor of shape (num_patches, self.tokens_per_patch)
            raster (bool, optional): Whether to return the rasterized patches. Defaults to False.

        Returns:
        if raster:
            Tensor: Tensor of shape (num_patches, channels, patch_res, patch_res)
        else:
            Tensor: Tensor of shape (num_patches, num_points, 2)
            dict : visual attribute dict
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Decoding patches is not supported when using the text encoder only.")
        with torch.no_grad():
            out, _ = self.vq_model.decode_from_indices(tokens - self.start_of_patch_token_idx)
        if raster:
            return out[0]
        else:
            return out[2], out[3]  # bezier points and visual attribute dict
    
    def decode_positions(self, tokens: Tensor) -> Tensor:
        """
        Decodes the positions from the tokens.

        Args:
            tokens (Tensor): Tensor of shape (num_pos, 1)

        Returns:
            Tensor: Tensor of shape (num_pos, 2)
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Decoding positions is not supported when using the text encoder only.")
        tokens = tokens - self.start_of_pos_token_idx
        positions = self.all_possible_positions[tokens]
        if positions.dim() > 2:
            positions = positions.squeeze(1)
        return positions
    
    def decode_text(self, tokens: Tensor) -> str:
        """
        Decodes the text from the tokens.

        Args:
            tokens (Tensor): Tensor of shape (num_tokens)

        Returns:
            str: Decoded text
        """
        text = self.text_tokenizer.decode(tokens, skip_special_tokens=True)
        return text
    
    def decode(self, tokens: Tensor, ignore_special_tokens: bool = False, only_patch_tokens:bool=False):
        """
        Decodes the patches and positions from the tokens.

        Args:
            tokens (Tensor): Tensor of shape (num_tokens)
            ignore_special_tokens (bool, optional): Whether to ignore the required special tokens like BOS and EOS. Defaults to False.
            only_patch_tokens (bool, optional): Whether the sequence contains only patch tokens. Defaults to False.

        Returns:
            Tuple[Tensor, Tensor]: Tuple of tensors of shape (num_patches, channels, patch_res, patch_res) and (num_pos, 2)
        """
        if self.use_text_encoder_only:
            raise NotImplementedError("Decoding patches/positions is not supported when using the text encoder only.")
        # remove all occurence of <PAD> token
        tokens = tokens[tokens != self.special_token_mapping["<PAD>"]]

        assert tokens.ndim == 1, f"Tokens should be 1D, got shape {tokens.shape}"
        assert tokens.size(0) > 3, f"Tokens should have at least 4 elements, got {tokens.size(0)}"
        if not ignore_special_tokens:
            assert tokens[0] == self.special_token_mapping["<BOS>"], f"First token should be <BOS>, got {tokens[0]}"
            assert tokens[-1] == self.special_token_mapping["<EOS>"] or tokens[-1] == self.special_token_mapping["<PAD>"], f"Last token should be <EOS> or <PAD>, got {tokens[-1]}"
        if tokens[-1] == self.special_token_mapping["<EOS>"]:
            tokens = tokens[:-1]
        if tokens[0] == self.special_token_mapping["<BOS>"]:
            tokens = tokens[1:]
        if self._is_patch(tokens[-1]) and not only_patch_tokens:
            # print("[INFO] Last token is a patch token, removing it.")
            tokens = tokens[:-1]
        if self.tokens_per_patch == 1 and not only_patch_tokens:
            assert tokens.size(0) % 2 == 0, f"Number of tokens should be even, got {tokens.size(0)}"
        if only_patch_tokens:
            patch_tokens = tokens
            positions = self.all_possible_positions
        else:
            patch_tokens = tokens[::2]
            pos_tokens = tokens[1::2]
            positions = self.decode_positions(pos_tokens)
        bezier_points, visual_attribute_dict = self.decode_patches(patch_tokens)
        return bezier_points, visual_attribute_dict, positions
    

    def assemble_svg(self, 
                     bezier_points: Tensor, 
                     visual_attribute_dict:dict,
                     center_positions: Tensor = None,
                     w=480) -> Drawing:
        # assert len(bezier_points) == len(self.all_possible_positions), f"Number of bezier points ({len(bezier_points)}) does not match number of possible positions of patches ({len(self.all_possible_positions)})."
        points_diff_to_center = bezier_points - 0.5
        scaled_points_diff_to_center = points_diff_to_center * (self.full_image_res / self.num_tiles_per_row)
        if center_positions is not None:
            global_positions = scaled_points_diff_to_center + center_positions[:,None,:].repeat(1, scaled_points_diff_to_center.size(1), 1)
        else:
            global_positions = scaled_points_diff_to_center + self.all_possible_positions[:len(bezier_points),None,:].repeat(1, scaled_points_diff_to_center.size(1), 1)

        global_positions = global_positions / self.full_image_res
        reconstructed_drawing = shapes_to_drawing(global_positions, stroke_width=None, w=w, mode="circles", visual_attribute_dict=visual_attribute_dict)
        return reconstructed_drawing
    
    def _tokens_to_image_tensor(self, 
                                tokens:Tensor,  
                                only_patch_tokens:bool=False) -> Tensor:
        
        bezier_points, visual_attribute_dict, positions = self.decode(tokens, ignore_special_tokens=True, only_patch_tokens=only_patch_tokens)
        drawing = self.assemble_svg(bezier_points, visual_attribute_dict, positions, w=480)
        return_tensor = drawing_to_tensor(drawing)
        return return_tensor