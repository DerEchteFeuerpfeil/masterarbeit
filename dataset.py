import os
import torch
from torch import Tensor
from typing import Any, List, Optional, Sequence, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from PIL import Image
import glob
import pandas as pd
import numpy as np
import string
from utils import drawing_to_tensor,svg2paths2, disvg, raster, get_single_paths, get_similar_length_paths, check_for_continouity, get_rasterized_segments, all_paths_to_max_diff, Path, svg_string_to_tensor, rgb_tensor_to_svg_color
from tokenizer import VQTokenizer
import copy
import random
import math

class VQDataset(Dataset):
    """
    main input here is the csv_path to a split.csv, which can be created using datasets/make_final_stage2_csv.py
    new addition: the csv needs a column called "index_in_numpy_array", which points each sample to an index in the vq_token numpy array

    - fraction_of_class_only_inputs: float (default 0.2), fraction of samples where the text input is only the "class" entry of the dataframe
    - fraction_of_blank_inputs: float (default 0.1), fraction of samples where the text input is empty
    - use_given_text_tokens_only: bool (default False), if True, the text input will always be the already tokenized file
    - shuffle_vq_order: bool (default False), if True, `<SOS>, <CLS>, t_1, ..., t_n, <SEP>, <BOS>, v_1, p_1, v_2, p_2, ... v_m, p_m, <EOS>` will become `<SOS>, <CLS>, t_1, ..., t_n, <SEP>, <BOS>, v_i, p_i, v_i+1, p_i+1, ..., v_m, p_m, v_1, p_1, ..., v_i-1, p_i-1, <EOS>` for random index i
    its not really "shuffling", but more cutting the sequence into two parts and switching their order
    """
    def __init__(self,
                 csv_path:str,
                 vq_token_npy_path:str,
                 tokenizer: VQTokenizer,
                 context_length: int,
                 dataset:str,
                 max_text_token_length: int = 50,
                 min_context_length: int = 10,
                fraction_of_class_only_inputs:float= 0.0,
                fraction_of_blank_inputs:float= 0.1,
                fraction_of_iconshop_chatgpt_inputs:float= 0.3,
                fraction_of_full_description_inputs:float=0.6,
                 shuffle_vq_order:bool=True,
                 use_pre_computed_text_tokens_only: bool=False,
                 train:bool = True,
                 subset:str=None,
                 return_ids:bool=False,
                 **kwargs):
        super(VQDataset, self)#.__init__()

        print(f"[INFO] These keywords were provided in VQDataset but are not used: {kwargs.keys()}")

        self.split = pd.read_csv(csv_path)
        self.train = train
        self.subset = subset
        self.return_ids = return_ids

        self.context_length = context_length
        self.min_context_length = min_context_length
        self.max_text_token_length = max_text_token_length

        sum_of_fractions = fraction_of_class_only_inputs + fraction_of_blank_inputs + fraction_of_iconshop_chatgpt_inputs + fraction_of_full_description_inputs
        if subset is not None:
            assert subset in self.split["class"].unique(), f"Subset {subset} not found in the dataset."
            self.split = self.split[self.split["class"] == subset].reset_index(drop=True)
        assert dataset in ["figr8", "fonts"], f"Dataset must be either 'figr8' or 'fonts', got {dataset}."
        assert sum_of_fractions == 1., f"All fractions must be equal to 1, got {sum_of_fractions}."

        if (not train or train is None) and fraction_of_full_description_inputs < 1.0:
            raise ValueError(f"Youre trying to validate/test with only {fraction_of_full_description_inputs} description inputs, must be 1.0")

        self.fraction_of_class_only_inputs = fraction_of_class_only_inputs
        self.fraction_of_blank_inputs = fraction_of_blank_inputs
        self.fraction_of_iconshop_chatgpt_inputs = fraction_of_iconshop_chatgpt_inputs
        self.fraction_of_full_description_inputs = fraction_of_full_description_inputs
        self.dataset = dataset

        if dataset == "figr8":
            assert "class" in self.split.columns if self.fraction_of_class_only_inputs > 0 else True, "Column 'class' is required for figr8 dataset."
            assert "iconshop_description" in self.split.columns if self.fraction_of_iconshop_chatgpt_inputs > 0 else True, "Column 'iconshop_description' is required for figr8 dataset."
            assert "description" in self.split.columns if self.fraction_of_full_description_inputs > 0 else True, "Column 'description' is required for figr8 dataset."

        self.use_pre_computed_text_tokens_only = use_pre_computed_text_tokens_only
        self.shuffle_vq_order = shuffle_vq_order

        self.tokenizer = tokenizer
        self.tokenizer.use_text_encoder_only = True

        self.bert_cls_token = self.tokenizer.text_tokenizer.get_vocab().get("[CLS]")
        self.bert_sep_token = self.tokenizer.text_tokenizer.get_vocab().get("[SEP]")
        self.bert_pad_token = self.tokenizer.text_tokenizer.get_vocab().get("[PAD]")
        self.sos_token = self.tokenizer.special_token_mapping.get("<SOS>")
        self.bos_token = self.tokenizer.special_token_mapping.get("<BOS>")
        self.eos_token = self.tokenizer.special_token_mapping.get("<EOS>")
        self.pad_token = self.tokenizer.special_token_mapping.get("<PAD>")

        # load pre-computed vq tokens
        self.vq_token_npy_path = vq_token_npy_path
        numpy_array = np.load(vq_token_npy_path)
        self.vq_numpy_array = np.split(numpy_array, np.where(numpy_array == self.bos_token)[0])[1:]

        if not len(self.vq_numpy_array) == len(self.split) and self.subset is None:
            print(f"[WARNING] Number of samples in the numpy array and the csv file do not match. Numpy array has {len(self.vq_numpy_array)} samples, csv has {len(self.split)} samples.")
        
        if train is None:
            self.split = self.split[self.split["split"] == "test"].reset_index(drop=True)
            # self.split = self.split.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            if train:
                self.split = self.split[self.split["split"] == "train"].reset_index(drop=True)
            else:
                self.split = self.split[self.split["split"] == "val"].reset_index(drop=True)

        self.split["index_in_numpy_array"] = self.split["index_in_numpy_array"].astype(int)
        samples_before_filtering = len(self.split)

        self.split = self.split[self.split["text_token_length"] <= self.max_text_token_length]

        # TODO add font blacklisting here
        self.split = self.split[self.split["vq_token_length"] + self.max_text_token_length + 2 <= self.context_length]
        self.split = self.split[self.split["vq_token_length"] >= self.min_context_length]

        samples_after_filtering = len(self.split)
        if samples_before_filtering > 0:
            print(f"[INFO] Filtered {samples_before_filtering - samples_after_filtering} samples because they were too long or too short. That is {np.round((samples_before_filtering - samples_after_filtering) / samples_before_filtering * 100, decimals=2)}% of the dataset.")
        else:
            print(f"[WARNING] No samples found for {'train' if train else 'test'} split.")

    def _get_padded_text_tokens(self, text_tokens: np.ndarray):
        padded_text = np.append(text_tokens, np.zeros(self.max_text_token_length - len(text_tokens), dtype=np.ushort) + self.bert_pad_token)
        return padded_text
    
    def _get_padded_vq_tokens(self, vq_tokens: np.ndarray):
        if vq_tokens[0] != self.bos_token:
            vq_tokens = np.concatenate([np.array([self.bos_token]), vq_tokens])
        vq_with_eos = np.append(vq_tokens, np.zeros(1, dtype=np.ushort) + self.eos_token)
        final_padded_vq = np.append(vq_with_eos, np.zeros(self.context_length - self.max_text_token_length - len(vq_with_eos) - 1, dtype=np.ushort) + self.pad_token)  # -1 because SOS token is prefixed to the sequence later
        return final_padded_vq

        # assert len(self.text_tokens) == len(self.vq_tokens), "Text and VQ tokens should have the same shape."
        # assert self.text_tokens[0,0] == bert_cls_token, "First token in text tokens should be the BERT CLS token."
        # assert self.vq_tokens[0,0] == bos_token, "First token in VQ tokens should be the BOS token."
        # assert self.text_attention_masks[0,0] == 1, "First token in text attention masks should be 1."

    def __len__(self):
        return len(self.split)
    
    def _get_tokenized_text(self, row):
        if self.dataset == "fonts":
            text_to_tokenize = np.random.choice([row["class"], 
                                                 row["description"], 
                                                 ""],
                                             p=[self.fraction_of_class_only_inputs, 
                                                self.fraction_of_full_description_inputs, 
                                                self.fraction_of_blank_inputs])
            if text_to_tokenize in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and len(text_to_tokenize) == 1:
                text_to_tokenize = f"capital {text_to_tokenize}"
            text_tokens = self.tokenizer.tokenize_text(text_to_tokenize)
            return text_tokens
        elif self.dataset == "figr8":
            text_to_tokenize = np.random.choice([row.get("class"), 
                                                 row.get("iconshop_description"), 
                                                 row.get("description"),
                                                 ""],
                                                p=[self.fraction_of_class_only_inputs, 
                                                   self.fraction_of_iconshop_chatgpt_inputs, 
                                                   self.fraction_of_full_description_inputs,
                                                   self.fraction_of_blank_inputs])
            if text_to_tokenize is None:
                text_to_tokenize = "None"
            text_tokens = self.tokenizer.tokenize_text(text_to_tokenize)
            return text_tokens

    def __getitem__(self, idx:int):
        """
        IMPORTANT
        text tokens have their special tokens and padding already included.
        vq tokens have their special tokens (BOS and EOS) and padding already included.
        only SOS needs to be prefixed after the data is loaded.
        """
        row = self.split.iloc[idx]

        if self.use_pre_computed_text_tokens_only:
            text_tokens = np.load(row["text_token_path"])
        else:
            text_tokens = self._get_tokenized_text(row)
        
        # it could be the case that the calculation of text token lengths was only done for one kind of description which could cause an error here if the other is longer
        if len(text_tokens) > self.max_text_token_length:
            last_token = text_tokens[-1:]
            text_tokens = text_tokens[:self.max_text_token_length-1]
            text_tokens = torch.concat([text_tokens, last_token])
        text_tokens = self._get_padded_text_tokens(text_tokens)

        # vq_tokens = np.load(row["vq_token_path"])
        vq_tokens = self.vq_numpy_array[row["index_in_numpy_array"]]
        if len(vq_tokens) != row["vq_token_length"]:
            print(f"[ERROR] Length of vq tokens in numpy array and csv file do not match. Numpy array has {len(vq_tokens)} tokens, csv has {row['vq_token_length']})")

        if self.shuffle_vq_order:
            try:
                i = np.random.randint(5, len(vq_tokens) - 5)
            except:
                # if something goes wrong, just take the middle of the min-sequence
                i = self.min_context_length//2
            if self.tokenizer._is_position(vq_tokens[i]):
                i -= self.tokenizer.tokens_per_patch
            vq_tokens = np.concatenate([np.array([self.bos_token]),vq_tokens[i:], vq_tokens[1:i]])
        vq_tokens = self._get_padded_vq_tokens(vq_tokens)
        text_attention_mask = (text_tokens != self.bert_pad_token).astype(np.int64)

        text_tokens = torch.from_numpy(text_tokens.astype(np.int32)).long()
        vq_tokens = torch.from_numpy(vq_tokens.astype(np.int32)).long()
        vq_targets = torch.roll(vq_tokens, -1)
        vq_targets[-1] = self.pad_token
        attention_mask = torch.from_numpy(text_attention_mask.astype(np.int32)).long()
        if self.return_ids:
            return text_tokens, attention_mask, vq_tokens, vq_targets, torch.ones(1).to(text_tokens.device)*self.pad_token, row["id"]
        return text_tokens, attention_mask, vq_tokens, vq_targets, torch.ones(1).to(text_tokens.device)*self.pad_token

class VQDataModule(LightningDataModule):

    def __init__(
        self,
        csv_path: str,
        dataset:str,
        vq_token_npy_path: str,
        tokenizer: VQTokenizer,
        context_length: int,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size:int = 32,
        num_workers: int = 0,
        min_context_length: int = 10,
        fraction_of_class_only_inputs: float = 0.0,
        fraction_of_blank_inputs: float = 0.1,
        fraction_of_iconshop_chatgpt_inputs: float = 0.3,
        fraction_of_full_description_inputs: float = 0.6,
        shuffle_vq_order:bool=False,
        use_pre_computed_text_tokens_only: bool=False,
        subset:str=None,
        **kwargs,
    ):
        super().__init__()

        self.csv_path = csv_path
        self.dataset = dataset
        self.vq_token_npy_path= vq_token_npy_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.min_context_length = min_context_length
        self.fraction_of_class_only_inputs = fraction_of_class_only_inputs
        self.fraction_of_blank_inputs = fraction_of_blank_inputs
        self.fraction_of_iconshop_chatgpt_inputs = fraction_of_iconshop_chatgpt_inputs
        self.fraction_of_full_description_inputs = fraction_of_full_description_inputs
        self.shuffle_vq_order = shuffle_vq_order
        self.use_pre_computed_text_tokens_only = use_pre_computed_text_tokens_only
        self.subset = subset


    def setup(self, stage: Optional[str] = None, return_ids:Optional[bool]=False) -> None:
        if stage not in ["train", "test", "val"]:
            stage = None

        if stage is None or stage == "train":
            self.train_dataset = VQDataset(
                self.csv_path,
                self.vq_token_npy_path,
                tokenizer=self.tokenizer,
                context_length=self.context_length,
                dataset=self.dataset,
                train=True,
                min_context_length=self.min_context_length,
                fraction_of_class_only_inputs = self.fraction_of_class_only_inputs,
                fraction_of_blank_inputs = self.fraction_of_blank_inputs,
                fraction_of_iconshop_chatgpt_inputs=self.fraction_of_iconshop_chatgpt_inputs,
                fraction_of_full_description_inputs=self.fraction_of_full_description_inputs,
                shuffle_vq_order = self.shuffle_vq_order,
                use_pre_computed_text_tokens_only = self.use_pre_computed_text_tokens_only,
                subset=self.subset,
                return_ids=return_ids,
            )

        if stage is None or stage == "val":
            self.val_dataset = VQDataset(
                self.csv_path,
                self.vq_token_npy_path,
                tokenizer=self.tokenizer,
                context_length=self.context_length,
                dataset=self.dataset,
                train=False,
                min_context_length=self.min_context_length,
                fraction_of_class_only_inputs = 0.0,
                fraction_of_blank_inputs = 0.0,
                fraction_of_iconshop_chatgpt_inputs=0.0,
                fraction_of_full_description_inputs=1.0,
                shuffle_vq_order = self.shuffle_vq_order,
                use_pre_computed_text_tokens_only = self.use_pre_computed_text_tokens_only,
                subset=self.subset,
                return_ids=return_ids,
            )
        if stage is None or stage == "test":
            self.test_dataset = VQDataset(
                self.csv_path,
                self.vq_token_npy_path,
                tokenizer=self.tokenizer,
                context_length=self.context_length,
                dataset=self.dataset,
                train=None,
                min_context_length=self.min_context_length,
                fraction_of_class_only_inputs = 0.0,
                fraction_of_blank_inputs = 0.0,
                fraction_of_iconshop_chatgpt_inputs=0.0,
                fraction_of_full_description_inputs=1.0,
                shuffle_vq_order = self.shuffle_vq_order,
                use_pre_computed_text_tokens_only = self.use_pre_computed_text_tokens_only,
                subset=self.subset,
                return_ids=return_ids,
            )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )
    
class VSQDataset(Dataset):
    """
    Glyphazzn dataset that requires already normalized SVGs. Yields patches, positions, and labels. The label is the index of string.printable -> label = string.printable.index(label)

    Requires the following structure:
    top_level_dir (simplified svgs)
    |________________________________________
    |                   |                   |
    train               test                split.csv (with columns: file_path, class, split, description)
    |                   |
    0-9, a-z, A-Z       0-9, a-z, A-Z
    |                   |
    *.svg               *.svg

    Args:
        - top_level_dirs: paths to the top level directory of all simplified SVGs
        - channels: number of channels for the rasterized images
        - width: width/height of the rasterized images
        - train: whether to use the train or test split
        - subset:
        - individual_min_length: minimum length of a path segment to qualify for being a single shape layer
        - individual_max_length: maximum length of a path segment, everything longer than this will be cropped into multiple segments
        - stroke_width: stroke width for rasterization
        - max_shapes_per_svg: maximum number of shape layers per svg file, can be tuned for VRAM savings
    """

    def __init__(self,
                 csv_path: List[str],
                 channels: int,
                 width: int,
                 train: bool = True,
                 individual_min_length: float = 1.,
                 individual_max_length: float = 10.,
                 stroke_width: float = 0.3,
                 max_shapes_per_svg: int = 64,
                 use_single_paths:bool = False,
                 return_index = False,
                 return_filename = False,
                 subset_class:str=None,
                 use_random_stroke_widths:bool = False,
                 color_mode:str = None,
                 return_visual_attributes:bool = False,
                 **kwargs):
        super(VSQDataset, self)
        print(f"[INFO] These keywords were provided in VSQDataset but are not used: {kwargs.keys()}")
        assert not (return_filename and return_index), "can only additionally return filename or index, not both"
        self.csv_path = csv_path
        self.individual_min_length = individual_min_length
        self.individual_max_length = individual_max_length
        self.stroke_width = stroke_width
        self.max_shapes_per_svg = max_shapes_per_svg
        self.channels = channels
        self.width = width
        self.train = train
        self.use_single_paths = use_single_paths
        self.return_index = return_index
        self.return_filename = return_filename
        self.subset_class = subset_class
        self.use_random_stroke_widths = use_random_stroke_widths
        self.color_mode = color_mode
        self.return_visual_attributes = return_visual_attributes

        assert color_mode in ["palette", None, "random", "None"], "color mode must be either 'palette' or None, fully random is not yet implemented"

        self.stroke_width_range = (0.1, 0.5)
        self.color_palette = [
            [0.1, 0.2, 0.3],  # Dark Blue
            [0.2, 0.6, 0.2],  # Forest Green
            [0.8, 0.2, 0.2],  # Red
            [0.9, 0.9, 0.1],  # Yellow
            [0.3, 0.3, 0.8],  # Royal Blue
            [0.5, 0.0, 0.5],  # Purple
            [1.0, 0.5, 0.0],  # Orange
            [0.0, 0.5, 0.5],  # Teal
            [0.0, 0.0, 0.0]   # Black
        ]

        self.df = pd.read_csv(csv_path)
        self.class2id = {id_name: class_name for class_name, id_name in enumerate(self.df["class"].unique())}
        if train is None:
            self.df = self.df[self.df["split"] == "test"].reset_index(drop=True)
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            if train:
                self.df = self.df[self.df["split"] == "train"].reset_index(drop=True)
            else:
                self.df = self.df[self.df["split"] == "val"].reset_index(drop=True)

        if subset_class is not None and subset_class in self.df["class"].unique():
            self.df = self.df[self.df["class"] == subset_class].reset_index(drop=True)

    def crop_path_into_segments(self, path:Path, length:float = 5.):
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

    def get_similar_length_paths(self, single_paths, max_length: float = 5., filter_min_length:bool = False):
        """
        splits all the paths into similar length segments if they're too long
        """
        similar_length_paths = []
        if filter_min_length:
            prev_len = len(single_paths)
            single_paths = [x for x in single_paths if x.length() >= self.individual_min_length]
            after_len = len(single_paths)
            if after_len >= 0.8 * len(prev_len):
                print("More than 80% of paths were removed because they were too short. This is likely an error.")
        for path in single_paths:
            if path.length() < self.individual_min_length:
                similar_length_paths.append(path)
                continue
            try:
                segments = self.crop_path_into_segments(path, length=max_length)
                similar_length_paths.extend(segments)
            except AssertionError:
                print("Error while cropping path into segments, skipping...")
                continue
        return similar_length_paths
    
    def get_similar_length_paths_from_index(self, index, max_length: float = 5.):
        svg_path = self.df.iloc[index].simplified_svg_file_path
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        single_paths = get_single_paths(paths)
        sim_length_paths = self.get_similar_length_paths(single_paths, max_length=max_length)
        return sim_length_paths
    
    def _get_local_stroke_width(self):
        if self.use_random_stroke_widths:
            return random.uniform(*self.stroke_width_range)
        else:
            return self.stroke_width

    def __getitem__(self, index) -> tuple:
        svg_path = self.df.iloc[index]["simplified_svg_file_path"]
        label = self.df.iloc[index]["class"]
        label = self.class2id[label]
        if "description" in self.df.columns:
            description = self.df.iloc[index]["description"]#
        elif "desc" in self.df.columns:
            description = self.df.iloc[index]["desc"]
        else:
            description = "class: "+str(self.df.iloc[index]["class"])
         
        try:
            paths, attributes, svg_attributes = svg2paths2(svg_path)
        except Exception as e:
            print(f"[ERROR] Could not load {svg_path}. Exception: {e}")
            return torch.ones(2,3,128,128), torch.ones(2).int(), torch.ones(2,2), "EMPTY"
        if self.use_single_paths:
            single_paths = get_single_paths(paths)
            single_paths = self.get_similar_length_paths(single_paths, self.individual_max_length, filter_min_length=False)
        else:
            single_paths = self.get_similar_length_paths(paths, self.individual_max_length)
        
        assert check_for_continouity(single_paths), "paths are not continous"
        # select a random slice of the paths of length max_shapes_per_svg
        single_paths = [path for path in single_paths if path.length() > 0.]
        if len(single_paths) > self.max_shapes_per_svg:
            indices = random.sample(range(len(single_paths)), self.max_shapes_per_svg)
            single_paths = [single_paths[i] for i in indices]
        render_stroke_width = self._get_local_stroke_width()
        if self.color_mode == "palette":
            random_colors = [torch.tensor(random.choice(self.color_palette), dtype=torch.float32) for i in range(len(single_paths))]
            random_rgb_colors = torch.stack(random_colors)
            random_colors = [rgb_tensor_to_svg_color(color) for color in random_colors]
        elif self.color_mode == "random":
            random_colors = [torch.rand(3, dtype=torch.float32) for i in range(len(single_paths))]
            random_rgb_colors = torch.stack(random_colors)
            random_colors = [rgb_tensor_to_svg_color(color) for color in random_colors]
        else:
            random_rgb_colors = None
            random_colors = None
        rasterized_segments, centers = get_rasterized_segments(single_paths, 
                                                               render_stroke_width, 
                                                               self.individual_max_length, 
                                                               svg_attributes, 
                                                               centered=True, 
                                                               height=self.width, 
                                                               width=self.width,
                                                               colors = random_colors,
                                                               fill=False)
        viewbox_dims = [float(x) for x in svg_attributes["viewBox"].split(" ")]
        assert viewbox_dims[0] == 0 and viewbox_dims[1] == 0, f"viewbox has to be 0 0 at the start, got {viewbox_dims[0]} {viewbox_dims[1]}"
        assert viewbox_dims[2] == viewbox_dims[3], f"viewbox has to be same dimensions at the end, got {viewbox_dims[2]} {viewbox_dims[3]}"
        imgs = torch.stack(rasterized_segments)  # (n_shapes, channels, width, width)
        centers = torch.tensor(centers)  # (n_shapes, 2)
        centers = centers / viewbox_dims[2] # scale them to be in [0, 1] range
        labels = torch.ones(imgs.size(0)) * label
        if self.return_index:
            return imgs, labels.int(), centers, description, index
        elif self.return_filename:
            return imgs, labels.int(), centers, description, svg_path
        elif self.return_visual_attributes:
            return imgs, labels.int(), centers, description, {"colors": random_rgb_colors, "stroke_width": render_stroke_width}
        else:
            return imgs, labels.int(), centers, description
        
    def _get_split_path_count(self, index) -> int:
        svg_path = self.df.iloc[index]["simplified_svg_file_path"]

        try:
            paths, attributes, svg_attributes = svg2paths2(svg_path)
        except Exception as e:
            print(f"[ERROR] Could not load {svg_path}. Exception: {e}")
            return -1
        
        single_paths = self.get_similar_length_paths(paths, self.individual_max_length)
        single_paths = [path for path in single_paths if path.length() > 0.]

        return len(single_paths)
            
    
    def _get_full_item(self, index:int) -> List[Tensor]:
        """
        This function is intended to be used by the tokenization process.
        """
        svg_path = self.df.iloc[index]["simplified_svg_file_path"]
        label = self.df.iloc[index]["class"]
        label = self.class2id[label]
        description = self.df.iloc[index]["description"]

        paths, attributes, svg_attributes = svg2paths2(svg_path)
        if self.use_single_paths:
            single_paths = get_single_paths(paths, self.individual_max_length)
        else:
            single_paths = self.get_similar_length_paths(paths, self.individual_max_length)
        assert check_for_continouity(single_paths), "paths are not continous"
        single_paths = [path for path in single_paths if path.length() > 0.]
        render_stroke_width = self._get_local_stroke_width()
        rasterized_segments, centers = get_rasterized_segments(single_paths, render_stroke_width, self.individual_max_length, svg_attributes, centered=True, height=self.width, width=self.width)
        viewbox_dims = [float(x) for x in svg_attributes["viewBox"].split(" ")]
        assert viewbox_dims[0] == 0 and viewbox_dims[1] == 0, f"viewbox has to be 0 0 at the start, got {viewbox_dims[0]} {viewbox_dims[1]}"
        assert viewbox_dims[2] == viewbox_dims[3], f"viewbox has to be same dimensions at the end, got {viewbox_dims[2]} {viewbox_dims[3]}"
        imgs = torch.stack(rasterized_segments)  # (n_shapes, channels, width, width)
        centers = torch.tensor(centers)  # (n_shapes, 2)
        centers = centers / viewbox_dims[2] # scale them to be in [0, 1] range
        imgs = torch.stack(rasterized_segments)  # (n_shapes, channels, width, width)
        if not isinstance(centers, Tensor):
            centers = torch.tensor(centers)  # (n_shapes, 2)
        labels = torch.ones(imgs.size(0)) * label
        return imgs, labels.int(), centers, description
    
    def _get_full_svg_drawing(self, index, width:int = 720, as_tensor:bool = False, override_global_stroke_width:float=None):
        svg_path = self.df.iloc[index].simplified_svg_file_path
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        if self.use_single_paths:
            single_paths = get_single_paths(paths)
        else:
            single_paths = self.get_similar_length_paths(paths, self.individual_max_length)
        # single_paths = get_single_paths(paths)
        single_paths = [path for path in single_paths if path.length() > 0.]
        if override_global_stroke_width is None:
            render_stroke_width = self._get_local_stroke_width()
        else:
            render_stroke_width = override_global_stroke_width
        # TODO add the color here
        drawing = disvg(single_paths, paths2Drawing=True, stroke_widths=[render_stroke_width]*len(single_paths), viewbox = svg_attributes["viewBox"],dimensions=(width, width))
        if as_tensor:
            return svg_string_to_tensor(drawing.tostring())
        else:
            return drawing

    def __len__(self):
        return len(self.df)

class VSQDatamodule(LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        train_batch_size: int,
        val_batch_size: int,
        channels: int,
        width: int,
        test_batch_size:int = 32,
        individual_max_length: float = 10.,
        max_shapes_per_svg:int=64,
        num_workers: int = 0,
        stroke_width: float = 0.3,
        subset:str = None,
        use_single_paths:bool = False,
        return_index = False,
        return_filename: bool = False,
        color_mode:str = None,
        use_random_stroke_widths:bool = False,
        **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.csv_path = csv_path
        self.channels = channels
        self.width = width
        self.num_workers = num_workers
        self.stroke_width = stroke_width
        self.individual_max_length = individual_max_length
        self.subset = subset
        self.max_shapes_per_svg = max_shapes_per_svg
        self.use_single_paths = use_single_paths
        self.return_index = return_index
        self.return_filename = return_filename
        self.color_mode = color_mode
        self.use_random_stroke_widths = use_random_stroke_widths

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in ["train", "test", "val"]:
            stage = None

        if stage is None or stage == "train":
            self.train_dataset = VSQDataset(
                self.csv_path,
                self.channels,
                self.width,
                train=True,
                individual_max_length=self.individual_max_length,
                stroke_width=self.stroke_width,
                max_shapes_per_svg=self.max_shapes_per_svg,
                use_single_paths=self.use_single_paths,
                return_index = self.return_index,
                return_filename = self.return_filename,
                subset_class=self.subset,
                use_random_stroke_widths=self.use_random_stroke_widths,
                color_mode=self.color_mode
            )

        if stage is None or stage == "val":
            self.val_dataset = VSQDataset(
                self.csv_path,
                self.channels,
                self.width,
                train=False,
                individual_max_length=self.individual_max_length,
                stroke_width=self.stroke_width,
                max_shapes_per_svg=self.max_shapes_per_svg,
                use_single_paths=self.use_single_paths,
                return_index = self.return_index,
                return_filename = self.return_filename,
                subset_class=self.subset,
                use_random_stroke_widths=self.use_random_stroke_widths,
                color_mode=self.color_mode
            )

        if stage is None or stage == "test":
            self.test_dataset = VSQDataset(
                self.csv_path,
                self.channels,
                self.width,
                train=None,
                individual_max_length=self.individual_max_length,
                stroke_width=self.stroke_width,
                max_shapes_per_svg=self.max_shapes_per_svg,
                use_single_paths=self.use_single_paths,
                return_index = self.return_index,
                return_filename = self.return_filename,
                subset_class=self.subset,
                use_random_stroke_widths=self.use_random_stroke_widths,
                color_mode=self.color_mode
            )

    #       ===============================================================

    def collate_fn(self, batch):
        if self.return_index:
            imgs, labels, centers, descriptions, idxs = zip(*batch)
        elif self.return_filename:
            imgs, labels, centers, descriptions, filenames = zip(*batch)
        else:
            imgs, labels, centers, descriptions = zip(*batch)
        imgs = torch.concat(imgs)
        labels = torch.concat(labels)
        centers = torch.concat(centers)
        if self.return_index:
            return imgs, labels, centers, descriptions, idxs
        elif self.return_filename:
            return imgs, labels, centers, descriptions, filenames
        else:
            return imgs, labels, centers, descriptions

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

class CenterShapeLayersFromSVGDataset(Dataset):
    """
    This dataset takes SVG files and preprocesses them into rasterized centered shape layers.

    Args:
        - csv_path: path to csv file with the following columns:
            - file_path: full path to svg file
            - class: class label
            - split: "train" or "test"
        - channels: number of channels for the rasterized images
        - width: width/height of the rasterized images
        - train: whether to use the train or test split
        - individual_min_length: minimum length of a path segment to qualify for being a single shape layer
        - individual_max_length: maximum length of a path segment, everything longer than this will be cropped into multiple segments
        - stroke_width: stroke width for rasterization
        - max_shapes_per_svg: maximum number of shape layers per svg file, can be tuned for VRAM savings
    """

    def __init__(self, 
                 csv_path: str,
                 channels: int,
                 width: int,
                 train: bool = True,
                 individual_min_length: float = 1.,
                 individual_max_length: float = 10.,
                 stroke_width: float = 0.3,
                 max_shapes_per_svg: int = 64,
                 **kwargs):
        super(CenterShapeLayersFromSVGDataset, self)
        self.csv_path = csv_path
        self.individual_min_length = individual_min_length
        self.individual_max_length = individual_max_length
        self.stroke_width = stroke_width
        self.max_shapes_per_svg = max_shapes_per_svg
        self.channels = channels
        self.width = width
        self.train = train
        self.split = pd.read_csv(self.csv_path)
        if train is not None:
            self.split = self.split[self.split["split"] == ("train" if self.train else "test")]
        else:
            print("[WARNING] Using the whole dataset! Train was None.")

    def crop_path_into_segments(self, path:Path, length:float = 5.):
        """
        a single input path is cropped into segments of approx length `length`. I say "approx" because we divide the path into same length segments, which will not be exactly `length` long.
        """
        segments = []
        num_iters = math.ceil(path.length() / length)
        for i in range(num_iters):
            cropped_segment = path.cropped(i/num_iters, (i+1)/num_iters)
            segments.append(cropped_segment)

        return segments

    def get_similar_length_paths(self, single_paths, max_length: float = 5.):
        similar_length_paths = []
        for path in single_paths:
            if path.length() < self.individual_min_length:
                continue
            try:
                segments = self.crop_path_into_segments(path, length=max_length)
                similar_length_paths.extend(segments)
            except AssertionError:
                print("Error while cropping path into segments, skipping...")
                continue
        return similar_length_paths
    
    def get_similar_length_paths_from_index(self, index, max_length: float = 5.):
        svg_path = self.split.iloc[index]["file_path"]
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        single_paths = get_single_paths(paths)
        sim_length_paths = self.get_similar_length_paths(single_paths, max_length=max_length)
        return sim_length_paths

    def __getitem__(self, index) -> tuple:
        svg_path = self.split.iloc[index]["file_path"]
        label = self.split.iloc[index]["class"]
        label = string.printable.index(label)
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        single_paths = get_single_paths(paths)
        # queue = copy.deepcopy(single_paths)
        sim_length_paths = self.get_similar_length_paths(single_paths, self.individual_max_length)
        assert check_for_continouity(sim_length_paths), "paths are not continous"
        # select a random slice of the paths of length max_shapes_per_svg
        if len(sim_length_paths) > self.max_shapes_per_svg:
            start_idx = random.randint(0, len(sim_length_paths) - self.max_shapes_per_svg)
            sim_length_paths = sim_length_paths[start_idx:start_idx+self.max_shapes_per_svg]
        rasterized_segments, centers = get_rasterized_segments(sim_length_paths, self.stroke_width, self.individual_max_length, svg_attributes, centered=True, height=self.width, width=self.width)
        imgs = torch.stack(rasterized_segments)  # (n_shapes, channels, width, width)
        centers = torch.tensor(centers)  # (n_shapes, 2)
        labels = torch.ones(imgs.size(0)) * label
        return imgs, labels, centers
    
    def _get_full_svg_drawing(self, index, width:int = 720):
        svg_path = self.split.iloc[index]["file_path"]
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        single_paths = get_single_paths(paths)
        return disvg(single_paths, paths2Drawing=True, stroke_widths=[self.stroke_width]*len(single_paths), viewbox = svg_attributes["viewBox"],dimensions=(width, width))

    def __len__(self):
        return len(self.split)

class CenterShapeLayersFromSVGDataModule(LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        train_batch_size: int,
        val_batch_size: int,
        channels: int,
        width: int,
        individual_max_length: float = 10.,
        num_workers: int = 0,
        stroke_width: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.csv_path = csv_path
        self.channels = channels
        self.width = width
        self.num_workers = num_workers
        self.stroke_width = stroke_width
        self.individual_max_length = individual_max_length

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = CenterShapeLayersFromSVGDataset(
            self.csv_path,
            self.channels,
            self.width,
            train=True,
            individual_max_length=self.individual_max_length,
            stroke_width=self.stroke_width
        )

        self.val_dataset = CenterShapeLayersFromSVGDataset(
            self.csv_path,
            self.channels,
            self.width,
            train=False,
            individual_max_length=self.individual_max_length,
            stroke_width=self.stroke_width
        )

    #       ===============================================================

    def collate_fn(self, batch):
        imgs, labels, centers = zip(*batch)
        imgs = torch.concat(imgs)
        labels = torch.concat(labels)
        centers = torch.concat(centers)
        return imgs, labels, centers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=16,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

class NewCausalSVGDataset(Dataset):
    """
    New FIGR8 dataset from a root directory for causal svg generation.
    Has:
        - centered images for prediction and visual MSE loss
        - absolute images for merged input
        - absolute and relative start/end points
        - stop signal vector


     ClassName
     |
     |
     |---I{svg_id}_{num_segments}_Segments_images_absolute.npy 
     |---I{svg_id}_{num_segments}_Segments_images_centered.npy 
     |---I{svg_id}_{num_segments}_Segments_positions.npy  
     |---split.csv
    """

    def __init__(self, root_path: str, context_length: int, channels: int, width: int, subset: List[str], **kwargs):
        super(NewCausalSVGDataset, self)
        self.context_length = context_length
        self.channels = channels
        self.width = width
        self.root_path = root_path
        self.subset = subset
        self.split = pd.read_csv(os.path.join(self.root_path, "split.csv"))
        before = len(self.split)
        self.split = self.split[self.split["segments"] < context_length]
        self.split = self.split[self.split["segments"] > 15]
        after = len(self.split)
        print(f"Removed {np.round((before - after) / before * 100, decimals=2)}% of samples because they have more segments than the context length.")
        if self.subset and len(self.subset) > 0:
            self.split = self.split[self.split['class'].isin(self.subset)]
        self.split = self.split[self.split["split"] == ("train" if kwargs["train"] else "test")]

    def __getitem__(self, index) -> tuple:
        centered_filename = self.split.iloc[index]["raster_filename_centered"]
        absolute_filename = self.split.iloc[index]["raster_filename_absolute"]
        positions_filename = self.split.iloc[index]["position_filename"]
        curr_class = self.split.iloc[index]["class"]

        centered_images = torch.from_numpy(np.load(os.path.join(self.root_path, curr_class, centered_filename)))
        absolute_images = torch.from_numpy(np.load(os.path.join(self.root_path, curr_class, absolute_filename)))
        positions = torch.from_numpy(np.load(os.path.join(self.root_path, curr_class, positions_filename))).to(torch.float32)
        positions = torch.flatten(positions[:, [2,3], :], start_dim=-2) # (t, 4, 2) -> (t, 4)

        num_timesteps = centered_images.shape[0]
        if centered_images.max() > 1:
            centered_images = centered_images / 255  # shift to [0-1] (imgs stored as uint8)
        if absolute_images.max() > 1:
            absolute_images = absolute_images / 255  # shift to [0-1] (imgs stored as uint8)

        # padding images with fewer features than CL
        pad_len = self.context_length - num_timesteps
        assert pad_len > 0, "context length must be greater than number of features of the dataset, did you set the context length correctly in the config?"
        pad = torch.ones(pad_len, *centered_images.shape[1:])  # 1 -> white -> no features
        
        centered_shape_layers = torch.concat((centered_images, pad), dim=0)  # Ground truth
        absolute_shape_layers = torch.concat((absolute_images, pad), dim=0)  # Ground truth
        
        position_pad = torch.ones(pad_len, *positions.shape[1:])
        positions = torch.concat((positions, position_pad), dim=0)

        # adding channel (Dataset is gray-scale, if you use 3 channels those are replicated)
        centered_shape_layers = centered_shape_layers[:, None].repeat((1, self.channels, 1, 1))
        absolute_shape_layers = absolute_shape_layers[:, None].repeat((1, self.channels, 1, 1))

        # merges absolute layers for full-image input
        merged_layers = absolute_shape_layers[0].unsqueeze(dim=0)
        for t in range(1 + 1, absolute_shape_layers.shape[0] + 1):
            merged_layers_t = torch.min(absolute_shape_layers[:t], dim = 0).values
            merged_layers = torch.cat((merged_layers, merged_layers_t.unsqueeze(dim=0)), dim=0)

        # input is all shifted one place to the right and starts with white canvas
        input_absolute_shape_layers = torch.concat((torch.ones(1, self.channels, *absolute_shape_layers.shape[-2:]), absolute_shape_layers[:-1]))
        input_centered_shape_layers = torch.concat((torch.ones(1, self.channels, *centered_shape_layers.shape[-2:]), centered_shape_layers[:-1]))
        input_merged_images = torch.concat((torch.ones(1, self.channels, *merged_layers.shape[-2:]), merged_layers[:-1]))
        
        # input is all shifted one place to the right and starts with pos (0, 0, 0, 0)
        input_positions = torch.concat((torch.zeros((1, 4)), positions[:-1]))

        # only take the positions of the first point as gt supervision signal
        gt_positions = positions[:,:2]

        # creating stop ground truth with 0: no stop, 1: stop, -1: padding
        stop_pad_len = pad_len - 1  # stop signals require one less padding than images
        stop_signals = torch.zeros(self.context_length)
        stop_signals[num_timesteps] = 1.
        if stop_pad_len >= 1:
            stop_signals[-stop_pad_len:] = -1.
        caption = f"black and white icon of a {self.split.iloc[index]['class']}"
        return input_absolute_shape_layers, input_centered_shape_layers, input_merged_images, stop_signals, caption, centered_shape_layers, input_positions, gt_positions

    def __len__(self):
        return len(self.split)

class NewCausalSVGDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int,
        val_batch_size: int,
        context_length: int,
        channels: int,
        width: int,
        num_workers: int = 0,
        subset: List = None,
        **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.root_path = data_path
        self.context_length = context_length
        self.channels = channels
        self.width = width
        self.num_workers = num_workers
        self.subset = subset
        if subset:
            print(f"Using subset of original dataset: {self.subset}")
        else:
            print("Executing on the whole dataset!")

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = NewCausalSVGDataset(
            self.root_path,
            self.context_length,
            self.channels,
            self.width,
            subset=self.subset,
            train=True,
        )

        self.val_dataset = NewCausalSVGDataset(
            self.root_path,
            self.context_length,
            self.channels,
            self.width,
            subset=self.subset,
            train=False,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=64,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class GenericRasterizedSVGDataset(Dataset):
    def __init__(self,
                 csv_path:str,
                 train:bool,
                 img_size:int = 128,
                 channels:int = 3,
                 fill:bool = True,
                 global_stroke_width:float = 0.7,
                 global_stroke_width_fraction:float = None,
                 subset:str=None,
                 path_column:str=None,
                 **kwargs) -> None:
        """
        requires the csv to have these columns: "class", "split", "simplified_svg_file_path"
        """
        super(GenericRasterizedSVGDataset).__init__()

        self.csv_path = csv_path
        self.train = train
        self.img_size = img_size
        self.channels = channels
        self.fill = fill
        self.stroke_width = global_stroke_width
        self.global_stroke_width_fraction = global_stroke_width_fraction
        if "subset" in kwargs:
            self.subset = kwargs["subset"]
        else:
            self.subset = subset

        self.df = pd.read_csv(self.csv_path)
        if path_column is not None and path_column in self.df.columns:
            self.svg_column = path_column
        elif "simplified_svg_file_path" in self.df.columns:
            self.svg_column = "simplified_svg_file_path"
        elif "filename" in self.df.columns:
            self.svg_column = "filename"
        else:
            raise ValueError("Could not find column with svg file paths. Please make sure that the column is named 'simplified_svg_file_path' or 'svg_file_path'.")
        self.class2idx = {class_name: idx for idx, class_name in enumerate(self.df["class"].unique())}
        self.idx2class = {idx: class_name for class_name, idx in self.class2idx.items()}

        if self.train is None:
            self.df = self.df[self.df["split"] == "test"].reset_index(drop=True)
        else:
            if self.train:
                self.df = self.df[self.df["split"] == "train"].reset_index(drop=True)
            else:
                self.df = self.df[self.df["split"] == "val"].reset_index(drop=True)

        if self.subset is not None:
            print(f"Using subset {self.subset}")
            if isinstance(self.subset, list):
                self.df = self.df[self.df["class"].isin(self.subset)].reset_index(drop=True)
            else:
                self.df = self.df[self.df["class"] == self.subset].reset_index(drop=True)

    def _rasterize_svg(self, svg_path, img_size, fill):
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        viewbox = svg_attributes["viewBox"]
        viewbox_sizes = [float(x) for x in viewbox.split(" ")]
        for i in range(len(attributes)):
            if "fill" in attributes[i]:
                attributes[i]["fill"] = "black" if fill else "none"
            if self.global_stroke_width_fraction is not None:
                attributes[i]["stroke-width"] = f"{self.global_stroke_width_fraction*viewbox_sizes[-1]}"
            else:
                attributes[i]["stroke-width"] = f"{self.stroke_width}"
        rasterized = disvg(paths, viewbox = svg_attributes["viewBox"], dimensions = (img_size, img_size), attributes = attributes, paths2Drawing=True)
        return drawing_to_tensor(rasterized)
    
    def __getitem__(self, index) -> Tensor:
        img = self._rasterize_svg(self.df.iloc[index][self.svg_column], self.img_size, self.fill)
        label = self.class2idx[self.df.iloc[index]["class"]]
        return img, label

    def __len__(self) -> int:
        return len(self.df)

class GenericRasterizedSVGDataModule(LightningDataModule):
    def __init__(self,
                 csv_path:str,
                 fill:bool,
                 img_size:int = 128,
                 channels:int = 3,
                 train_batch_size:int = 16,
                 val_batch_size: int = 16,
                 num_workers:int = 4,
                 **kwargs
                 ) -> None:
        super().__init__()

        self.csv_path = csv_path
        self.img_size = img_size
        self.channels = channels
        self.fill = fill
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = GenericRasterizedSVGDataset(
            self.csv_path,
            train=True,
            img_size=self.img_size,
            fill=self.fill,
            channels=self.channels,
            **self.kwargs
        )

        self.val_dataset = GenericRasterizedSVGDataset(
            self.csv_path,
            train=False,
            img_size=self.img_size,
            fill=self.fill,
            channels=self.channels,
            **self.kwargs
        )

        self.test_dataset = GenericRasterizedSVGDataset(
            self.csv_path,
            train=None,
            img_size=self.img_size,
            fill=self.fill,
            channels=self.channels,
            **self.kwargs
        )
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=16,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )



class GenericRasterDataset(Dataset):
    def __init__(self,
                 csv_path:str,
                 train:bool,
                 img_size:int = 128,
                 channels:int=3,
                 invert:bool = True,
                 subset:str|List=None,
                 **kwargs) -> None:
        super(GenericRasterDataset).__init__()

        self.csv_path = csv_path
        self.train = train
        self.invert = invert
        self.img_size = img_size
        self.channels = channels
        self.subset = subset

        self.df = pd.read_csv(self.csv_path)
        assert "split" in self.df.columns, "split column not found in csv file"
        assert "class" in self.df.columns, "class column not found in csv file"
        assert "full_path" in self.df.columns or "png_path" in self.df.columns, "full_path column and png_path column not found in csv file"

        if "full_path" in self.df.columns:
            self.path_col = "full_path"
        else:
            self.path_col = "png_path"

        # mapping of legacy splitting
        if "held_back" in self.df.split.unique():
            self.df.loc[self.df["split"] == "test", "split"] = "val"
            self.df.loc[self.df["split"] == "held_back", "split"] = "test"
        self.class2idx = {class_name: idx for idx, class_name in enumerate(self.df["class"].unique())}
        if train is None:
            split = "test"
        else:
            split = "train" if train else "val"
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.subset is not None:
            print(f"Using subset {self.subset}")
            if isinstance(self.subset, list):
                self.df = self.df[self.df["class"].isin(self.subset)].reset_index(drop=True)
            else:
                self.df = self.df[self.df["class"] == self.subset].reset_index(drop=True)
        self.split = split
        self.regular_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        self.invert_transforms = transforms.RandomInvert(p=1.0)


    def __getitem__(self, index) -> Tensor:
        img = Image.open(self.df.iloc[index][self.path_col])
        label = self.class2idx[self.df.iloc[index]["class"]]
        if self.invert:
            img = self.regular_transforms(img)
            # if img.mean() > 0.7: # 1.0 is white
            img = self.invert_transforms(img)
        else:
            img = self.regular_transforms(img)

        if self.channels != img.shape[0]:
            img = img.repeat(self.channels, 1, 1)
        return img, label
    
    def __len__(self) -> int:
        return len(self.df)

class GenericRasterDatamodule(LightningDataModule):
    def __init__(self,
                 csv_path:str,
                 img_size:int = 128,
                 channels:int = 3,
                 train_batch_size:int = 32,
                val_batch_size:int = 32,
                num_workers:int = 4,
                 invert:bool = True,
                 subset:str|List=None,
                 **kwargs) -> None:
        
        super().__init__()

        self.csv_path = csv_path
        self.img_size = img_size
        self.invert = invert
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.channels = channels
        self.subset = subset

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = GenericRasterDataset(
            self.csv_path,
            train=True,
            img_size=self.img_size,
            invert=self.invert,
            channels=self.channels,
            subset=self.subset
        )

        self.val_dataset = GenericRasterDataset(
            self.csv_path,
            train=False,
            img_size=self.img_size,
            invert=self.invert,
            channels=self.channels,
            subset=self.subset
        )
        self.test_dataset = GenericRasterDataset(
            self.csv_path,
            train=None,
            img_size=self.img_size,
            invert=self.invert,
            channels=self.channels,
            subset=self.subset
        )
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=16,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

class MNIST(Dataset):
    """
    MNIST dataset from a root directory:

    mnist
    |
    |--------------|
    training    testing
    |              |
    0-9           0-9
    """

    def __init__(self, root, train=True, transform=None):
        super(MNIST, self)
        self.root = root
        self.train = train
        self.transform = transform

        self.image_folder = os.path.join(root, "training" if train else "testing")

        self.image_paths = []
        self.labels = []

        for label in range(10):
            label_folder = os.path.join(self.image_folder, str(label))
            image_files = os.listdir(label_folder)
            for image_file in image_files:
                image_path = os.path.join(label_folder, image_file)
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        return image, label, "", ""  # has to fit to the experiment class, which is why I filled it with empty stuff

    def __len__(self):
        return len(self.image_paths)

import torch.nn.functional as F

class TiledMNIST(Dataset):
    """
    MNIST dataset from a root directory where each image is split into tiles:

    mnist
    |
    |--------------|
    training    testing
    |              |
    0-9           0-9
    """

    def __init__(self, 
                 root, 
                 train=True,
                 patch_size:int=128, 
                 transform=None,
                 num_tiles_per_row:int = 5,
                 random_colors:bool=False,
                 use_palette:bool=True,
                 total_padding:int=20,
                 return_filename:bool=False,):
        super(TiledMNIST, self)
        self.root = root
        self.train = train
        self.transform = transform
        self.patch_size = patch_size
        self.num_tiles_per_row = num_tiles_per_row
        self.random_colors = random_colors
        self.use_palette = use_palette
        self.total_padding = total_padding
        self.return_filename = return_filename

        self.image_folder = os.path.join(root, "training" if train else "testing")

        self.image_paths = []
        self.labels = []

        for label in range(10):
            label_folder = os.path.join(self.image_folder, str(label))
            image_files = os.listdir(label_folder)
            for image_file in image_files:
                image_path = os.path.join(label_folder, image_file)
                self.image_paths.append(image_path)
                self.labels.append(label)

    def _recolor_stroke(self, tensor_image, use_palette=True):
        """
        Adjust the color of the stroke in an MNIST digit image represented as a tensor.
        The function converts black strokes to a random RGB color in the range [0, 1].
        
        Args:
        tensor_image (torch.Tensor): A 3D tensor representing the image (C, H, W) in RGB format.
        
        Returns:
        torch.Tensor: The modified image tensor with recolored strokes.
        """

        if tensor_image.device != 'cpu':
            tensor_image = tensor_image.cpu()
        if tensor_image.dim() != 3 or tensor_image.shape[0] != 3:
            raise ValueError("The input tensor must have shape (3, H, W)")
        
        color_palette = [
            [0.1, 0.2, 0.3],  # Dark Blue
            [0.2, 0.6, 0.2],  # Forest Green
            [0.8, 0.2, 0.2],  # Red
            [0.9, 0.9, 0.1],  # Yellow
            [0.3, 0.3, 0.8],  # Royal Blue
            [0.5, 0.0, 0.5],  # Purple
            [1.0, 0.5, 0.0],  # Orange
            [0.0, 0.5, 0.5],  # Teal
            [0.0, 0.0, 0.0]   # Black
        ]

        if use_palette:
            random_color = torch.tensor(random.choice(color_palette), dtype=torch.float32)
        else:
            random_color = torch.rand(3, dtype=torch.float32)
        
        black_pixels = (tensor_image < 0.9).all(0)  # Allow a small threshold for robustness
        
        # Recolor the black pixels
        for i in range(3):
            tensor_image[i][black_pixels] = random_color[i]
        
        return tensor_image

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        filename = self.image_paths[index]

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        if self.random_colors:
            image = self._recolor_stroke(image, use_palette=self.use_palette)

        patches = []

        single_side_padding = self.total_padding // 2
        for i in range(0, image.shape[1], self.patch_size - single_side_padding * 2):
            for j in range(0, image.shape[2], self.patch_size - single_side_padding * 2):
                patch = image[:, i : i + self.patch_size - self.total_padding, j : j + self.patch_size - self.total_padding]
                patch = F.pad(patch, (single_side_padding, single_side_padding, single_side_padding, single_side_padding), value=1.)
                patches.append(patch)       
                
        patches = torch.stack(patches)

        if self.return_filename:  # returning nothing is fine as tokenizer calculates positions
            return patches, [label]*len(patches), "", str(label), filename  # has to fit to the experiment class, which is why I filled it with empty stuff
        else:
            return patches, [label]*len(patches), "", str(label)  # has to fit to the experiment class, which is why I filled it with empty stuff
        # return patches, label, "", ""  # has to fit to the experiment class, which is why I filled it with empty stuff

    def __len__(self):
        return len(self.image_paths)


class MNISTpp(Dataset):
    """
    MNISTpp dataset from a root directory. There are no labels available.

    mnistpp
    |
    |--------------|
    training    testing
    """

    def __init__(self, root, train=True, transform=None):
        super(MNISTpp, self)
        self.root = root
        self.train = train
        self.transform = transform

        self.image_folder = os.path.join(root, "training" if train else "testing")

        self.image_paths = []

        image_files = os.listdir(self.image_folder)
        for image_file in image_files:
            image_path = os.path.join(self.image_folder, image_file)
            self.image_paths.append(image_path)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        label = 0

        return image, label

    def __len__(self):
        return len(self.image_paths)


class Emoji(Dataset):
    """
    Emoji dataset from a root directory. There are no labels available.

    emoji
    |
    |--------------|
    training    testing
    """

    def __init__(self, root, train=True, transform=None):
        super(Emoji, self)
        self.root = root
        self.train = train
        self.transform = transform

        self.image_folder = root

        self.image_paths = glob.glob(self.image_folder+"/*.png")
        train_end_idx = int(len(self.image_paths) * 0.75)
        if(self.train):
            self.image_paths = sorted(self.image_paths)[:train_end_idx]
        else:
            self.image_paths = sorted(self.image_paths)[train_end_idx:]

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        return image, 0

    def __len__(self):
        return len(self.image_paths)


class NounProject(Dataset):
    """
    The Noun Project dataset from a root directory. Class labels are directories. No train/test split in the folder structure.

    nounproject
     |
     |--------------|--------------|--------------|
     airplane    basketball       ...           zebra
     |              |              |              |
    *.png         *.png          *.png          *.png
    """

    def __init__(self, root, train=True, transform=None):
        super(NounProject, self)
        self.root = root
        self.train = train
        self.transform = transform

        self.image_folder = root
        self.threshold = 128

        self.image_paths = []
        self.labels = []
        self._int_to_label = {}

        for i, label in enumerate(os.listdir(self.image_folder)):
            # important for CLIP sim later to have the real string label
            self._int_to_label[i] = label
            
            image_paths = sorted(
                glob.glob(os.path.join(self.image_folder, label) + "/*.png")
            )
            train_split_idx = int(len(image_paths) * 0.75)
            if train:
                split_image_paths = image_paths[:train_split_idx]
            else:
                split_image_paths = image_paths[train_split_idx:]
            for image_path in split_image_paths:
                self.image_paths.append(image_path)
                self.labels.append(i)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path)
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255,255,255))
            bg.paste(image, mask=image.split()[3])
            image = bg
        if self.transform is not None:
            image = self.transform(image)
            # binarized_image = image > self.threshold
            # binarized_image = binarized_image.float()

        return image, label

    def __len__(self):
        return len(self.image_paths)


class Figr8CausalSVGDataset(Dataset):
    """
    FIGR8 dataset from a root directory for causal svg generation

     Length
     |
     |---------------------------------------------|
     F{folder_id}_I{svg_id}_P{features}.npy    split.csv
    """

    def __init__(self, root_path: str, context_length: int, channels: int, width: int, subset: List, **kwargs):
        super(Figr8CausalSVGDataset, self)
        self.context_length = context_length
        self.channels = channels
        self.width = width
        self.root_path = root_path
        self.subset = subset
        self.split = pd.read_csv(os.path.join(self.root_path, "split.csv"))
        if self.subset and len(self.subset) > 0:
            self.split = self.split[self.split['class'].isin(self.subset)]
        self.split = self.split[self.split["split"] == ("train" if kwargs["train"] else "test")]

    def __getitem__(self, index) -> tuple:
        filename = self.split.iloc[index]["filename"]
        images = np.load(os.path.join(self.root_path, filename))
        images = torch.from_numpy(images)
        num_features = images.shape[0]
        if images.max() > 1:
            images = images / 255  # shift to [0-1] (imgs stored as uint8)

        # padding images with fewer features than CL
        pad_len = self.context_length - num_features
        assert pad_len > 0, "context length must be greater than number of features of the dataset, did you set the context length correctly in the config?"
        pad = torch.ones(pad_len, *images.shape[1:])  # 1 -> white -> no features
        shape_layers = torch.concat((images, pad), dim=0)  # Ground truth

        # adding channel (Dataset is gray-scale, if you use 3 channels those are replicated)
        shape_layers = shape_layers[:, None].repeat((1, self.channels, 1, 1))

        # merges layers for full-image supervision
        merged_layers = shape_layers[0].unsqueeze(dim=0)
        for t in range(1 + 1, shape_layers.shape[0] + 1):
            merged_layers_t = torch.min(shape_layers[:t], dim = 0).values
            merged_layers = torch.cat((merged_layers, merged_layers_t.unsqueeze(dim=0)), dim=0)

        # input is all shifted one place to the right and starts with white canvas
        images = torch.concat((torch.ones(1, self.channels, *images.shape[1:]), shape_layers[:-1]))
        merged_images = torch.concat((torch.ones(1, self.channels, *images.shape[-2:]), merged_layers[:-1]))

        # creating stop ground truth with 0: no stop, 1: stop, -1: padding
        stop_pad_len = pad_len - 1  # stop signals require one less padding than images
        stop_signals = torch.zeros(self.context_length)
        stop_signals[num_features] = 1.
        if stop_pad_len >= 1:
            stop_signals[-stop_pad_len:] = -1.
        caption = f"An image of {self.split.iloc[index]['class']}"
        return images, shape_layers, stop_signals, caption, merged_layers, merged_images

    def __len__(self):
        return len(self.split)


class DummyCausalSVGDataset(Dataset):
    """
    returns black dummy images as the full composite images that act as input 
    returns individual shape renderings for loss calculation. full image at index T requires the network to predict shape rendering at index T.
    returns stop signal vector
    """
    def __init__(self, context_length: int, channels: int, width: int, **kwargs):
        super(DummyCausalSVGDataset, self)
        self.context_length = context_length
        self.channels = channels
        self.width = width

    def __getitem__(self, index) -> Tensor:
        image = torch.ones((self.channels, self.width, self.width))
        stop_idx = torch.randint(1, (self.context_length - 1), (1,))

        stop_signals = torch.zeros(self.context_length)
        stop_signals[stop_idx:] = 1.

        images = torch.stack([image]*self.context_length, 0)
        shape_layers = images

        return images, shape_layers, stop_signals
    
    def __len__(self):
        return 500


class MNISTforCSVG(Dataset):
    """
    MNIST dataset from a root directory for causal svg generation.

    mnist
    |
    |--------------|
    training    testing
    |              |
    0-9           0-9
    """

    def __init__(self, root, context_length: int = 2, train=True, transform=None):
        super(MNISTforCSVG, self)
        assert context_length > 1, "context length must be greater than 1"
        self.root = root
        self.train = train
        self.transform = transform
        self.context_length = context_length

        self.image_folder = os.path.join(root, "training" if train else "testing")

        self.image_paths = []
        self.labels = []

        for label in range(10):
            label_folder = os.path.join(self.image_folder, str(label))
            image_files = os.listdir(label_folder)
            for image_file in image_files:
                image_path = os.path.join(label_folder, image_file)
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        
        width = image.size(1)
        white_image = torch.ones((3, width, width))
        input_images = torch.stack([white_image] + [image] + [white_image]*(max(self.context_length-2, 0)), dim=0)
        gt_shape_layers = torch.stack([image] + [white_image]*(self.context_length-1), dim=0)

        # 0 is not stop, 1 is stop, -1 is padding, makes masking easier
        stop_signals = torch.cat([torch.Tensor([0.]), torch.Tensor([1.]), torch.Tensor([-1.]*(max(self.context_length-2, 0)))], dim=0)

        return input_images, gt_shape_layers, stop_signals#, label

    def __len__(self):
        return len(self.image_paths)


class MNISTDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: int = 128,
        num_tiles_per_row:int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        random_colors:bool=False,
        use_palette:bool=True,
        padding_frac:float = 0.1,
        return_filename = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_tiles_per_row = num_tiles_per_row
        self.random_colors = random_colors
        self.use_palette = use_palette
        self.padding_frac = padding_frac
        self.total_padding = (int(self.patch_size * self.padding_frac) // 2) * 2
        self.return_filename = return_filename

    def collate_fn(self, batch):
        if self.return_filename:
            patches, labels, _, descriptions, filenames = zip(*batch)  # (tiles, channels, height, width)
        else:
            patches, labels, _, descriptions = zip(*batch)  # (tiles, channels, height, width)
        patches = torch.stack(patches, dim=0)  # (bs, tiles, channels, height, width)
        # use einops to merge bs and tiles dimensions
        if patches.dim() == 5:
            patches = patches.view(-1, *patches.shape[2:])
        return patches, labels, "", descriptions, filenames

    def setup(self, stage: Optional[str] = None) -> None:
        # =========================  MNIST Dataset  =========================
        new_dimension = (self.patch_size - self.total_padding) * self.num_tiles_per_row

        train_transforms = transforms.Compose(
            [
                transforms.Resize(new_dimension, antialias=True),
                transforms.RandomInvert(1.0),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(new_dimension, antialias=True),
                transforms.RandomInvert(1.0),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = TiledMNIST(
            self.data_dir,
            train=True,
            transform=train_transforms,
            num_tiles_per_row=self.num_tiles_per_row,
            random_colors=self.random_colors,
            patch_size = self.patch_size,
            total_padding = self.total_padding,
            use_palette=self.use_palette,
            return_filename = self.return_filename

        )

        self.val_dataset = TiledMNIST(
            self.data_dir,
            train=False,
            transform=val_transforms,
            num_tiles_per_row=self.num_tiles_per_row,
            random_colors=self.random_colors,
            patch_size = self.patch_size,
            total_padding = self.total_padding,
            use_palette=self.use_palette,
            return_filename = self.return_filename
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )


class MNISTppDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        # =========================  MNIST Dataset  =========================

        train_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.RandomInvert(1.0),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.RandomInvert(1.0),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = MNISTpp(
            self.data_dir,
            train=True,
            transform=train_transforms,
        )

        self.val_dataset = MNISTpp(
            self.data_dir,
            train=False,
            transform=val_transforms,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


class EmojiDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = Emoji(
            self.data_dir,
            train=True,
            transform=train_transforms,
        )

        self.val_dataset = Emoji(
            self.data_dir,
            train=False,
            transform=val_transforms,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


class CausalSVGDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int,
        val_batch_size: int,
        context_length: int,
        channels: int,
        width: int,
        num_workers: int = 0,
        subset: List = None,
        **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.root_path = data_path
        self.context_length = context_length
        self.channels = channels
        self.width = width
        self.num_workers = num_workers
        self.subset = subset
        if subset:
            print(f"Using subset of original dataset: {self.subset}")
        else:
            print("Executing on the whole dataset!")

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = Figr8CausalSVGDataset(
            self.root_path,
            self.context_length,
            self.channels,
            self.width,
            subset=self.subset,
            train=True,
        )

        self.val_dataset = Figr8CausalSVGDataset(
            self.root_path,
            self.context_length,
            self.channels,
            self.width,
            subset=self.subset,
            train=False,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=100,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )


class NounProjectDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        # TODO think about scaling [0.8, 1.2] and translating [-2.5, 2.5] from DeepSVG
        train_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(5.0, fill=256),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = NounProject(
            self.data_dir,
            train=True,
            transform=train_transforms,
        )

        self.val_dataset = NounProject(
            self.data_dir,
            train=False,
            transform=val_transforms,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


class DummyCausalSVGDataModule(LightningDataModule):
    def __init__(
        self,
        context_length: int, 
        channels: int, 
        width: int
    ):
        super().__init__()

        self.context_length = context_length
        self.channels = channels
        self.width = width

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = DummyCausalSVGDataset(
            self.context_length,
            self.channels,
            self.width,
            train=True,
        )

        self.val_dataset = DummyCausalSVGDataset(
            self.context_length,
            self.channels,
            self.width,
            train=False,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=8,
            num_workers=1,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=8,
            num_workers=1,
            shuffle=True,
            pin_memory=False,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=8,
            num_workers=1,
            shuffle=True,
            pin_memory=False,
        )


class MNISTDatasetCSVG(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        context_length: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.context_length = context_length

    def setup(self, stage: Optional[str] = None) -> None:
        # =========================  MNIST Dataset  =========================

        train_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.RandomInvert(1.0),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(self.patch_size, antialias=True),
                transforms.RandomInvert(1.0),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = MNISTforCSVG(
            self.data_dir,
            train=True,
            transform=train_transforms,
            context_length=self.context_length
        )

        self.val_dataset = MNISTforCSVG(
            self.data_dir,
            train=False,
            transform=val_transforms,
            context_length=self.context_length
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )