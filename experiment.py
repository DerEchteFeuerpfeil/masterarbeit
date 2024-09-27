import gc
import os
import random
from typing import List, Tuple, Union
import torch
from torch import Tensor
from torch import optim
import wandb
from models import VectorVAEnLayers, VSQ, VQ_SVG_Stage2
import pytorch_lightning as pl
from utils import log_images, log_all_images, get_side_by_side_reconstruction, add_points_to_image, get_merged_image_for_logging, interpolate_rows, width_pred_to_local_stroke_width
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from torchmetrics.functional.multimodal import clip_score
from tokenizer import VQTokenizer
# import torch_optimizer as optim_
from dataset import VSQDatamodule, MNISTDataset
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

class SVG_VQVAE_Stage2_Experiment(pl.LightningModule):
    def __init__(self,
                 model: VQ_SVG_Stage2,
                 tokenizer: VQTokenizer,
                 num_batches_train: int,
                 num_batches_val: int,
                 total_training_steps:int,
                 lr: float = 0.0003,
                 weight_decay: float = 0.0,
                 scheduler_gamma: float = 0.99,
                 train_log_interval: float = 0.05,
                 val_log_interval: float = 0.1,
                 metric_log_interval: float = 0.1,
                 manual_seed: int = 42,
                 wandb: bool = False,
                 post_process: bool = True,
                 warmup_steps:int = 0,
                 **kwargs) -> None:
        super(SVG_VQVAE_Stage2_Experiment, self).__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        assert train_log_interval < 1 and train_log_interval >= 0, f"train log interval should be a fraction of the total number of batches in [0, 1), got {train_log_interval}"
        assert metric_log_interval < 1 and metric_log_interval >= 0, f"metric log interval should be a fraction of the total number of batches in [0, 1), got {metric_log_interval}"
        self.train_log_interval = max(1, int(train_log_interval * num_batches_train))
        self.val_log_interval = max(1, int(val_log_interval * num_batches_val))
        self.train_metric_log_interval = max(1, int(metric_log_interval * num_batches_train))
        self.val_metric_log_interval = max(1, int(metric_log_interval * num_batches_val))
        self.manual_seed = manual_seed
        self.curr_device = None
        self.wandb = wandb
        self.post_process = post_process
        self.total_training_steps = total_training_steps
        self.warmup_steps = warmup_steps

    def forward(self, text_tokens: Tensor, text_attention_mask: Tensor, vq_tokens: Tensor, logging=False,
                **kwargs) -> list:
        out, logging_dict = self.model.forward(text_tokens, text_attention_mask, vq_tokens, logging=logging, **kwargs)
        return out, logging_dict

    def _generate_rasterized_sample(self, text_tokens: Tensor, text_attention_mask: Tensor, vq_tokens: Tensor,
                                    temperature:float = 0.0, sampling_method: str = None, sampling_kwargs:dict = {},
                                    post_process: bool = True, draw_context_red: bool = True) -> Tensor:
        """
        Args:
            - text_tokens (Tensor): (1, t)
            - text_attention_mask (Tensor): (1, t)
            - vq_tokens (Tensor): (1, input_context_len_you_want)
        """
        num_input_context_tokens = vq_tokens.shape[-1] // 2
        with torch.no_grad():
            generation, reason = self.model.generate(text_tokens, text_attention_mask, vq_tokens,
                                                     temperature=temperature, sampling_method=sampling_method,sampling_kwargs=sampling_kwargs)
            if generation.ndim > 1:
                generation = generation[0]
        if draw_context_red:
            return self.tokenizer._tokens_to_image_tensor(generation, post_process=post_process,
                                                          num_strokes_to_paint=num_input_context_tokens)
        else:
            return self.tokenizer._tokens_to_image_tensor(generation, post_process=post_process)

    def _get_clip_score_for_batch(self, text_tokens: Tensor, text_attention_mask: Tensor, vq_tokens: Tensor,
                                  post_process: bool = True, temperatures:List=None) -> Tuple[Tensor, List, List]:
        """
        gets clip scores for 0-context generations of a batch of text tokens
        """
        with torch.no_grad():
            bs = text_tokens.shape[0]
            texts = [self.tokenizer.decode_text(text_tokens[i]) for i in range(bs)]
            # filter out empty texts
            relevant_idxs = [i for i in range(bs) if len(texts[i]) > 0]
            generations = [
                self._generate_rasterized_sample(
                    text_tokens[i:i + 1, :],
                    text_attention_mask[i:i + 1, :],
                    vq_tokens[i:i + 1, :1],
                    post_process=post_process,
                    temperature=temperatures[i] if temperatures is not None else 0.0
                ).to(self.curr_device) for i in relevant_idxs
            ]
            texts = [texts[i] for i in relevant_idxs]
            metric = clip_score(generations, texts, "openai/clip-vit-base-patch16")
        return metric, generations, texts

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        text_tokens, text_attention_mask, vq_tokens, vq_targets, pad_token = batch
        self.curr_device = text_tokens.device

        if batch_idx % self.train_log_interval == 0 and self.wandb:
            out, logging_dict = self.forward(text_tokens, text_attention_mask, vq_tokens, logging=True)
            text_condition = self.tokenizer.decode_text(text_tokens[0])
            rasterized_gt = self.tokenizer._tokens_to_image_tensor(vq_targets[:1], post_process=self.post_process)

            # every third batch use temp = 0
            if batch_idx % (self.train_log_interval * 3) == 0:
                temperature = 0.0
            else:
                temperature = random.uniform(0.2, 1.5)

            context_0_generation = self._generate_rasterized_sample(text_tokens[:1, :], text_attention_mask[:1, :],
                                                                    vq_tokens[:1, :1], post_process=self.post_process,
                                                                    temperature=temperature)
            context_5_generation = self._generate_rasterized_sample(text_tokens[:1, :], text_attention_mask[:1, :],
                                                                    vq_tokens[:1, :6], post_process=self.post_process,
                                                                    temperature=temperature)
            context_10_generation = self._generate_rasterized_sample(text_tokens[:1, :], text_attention_mask[:1, :],
                                                                     vq_tokens[:1, :11], post_process=self.post_process,
                                                                     temperature=temperature)
            images = [rasterized_gt, context_0_generation, context_5_generation, context_10_generation]
            self.trainer.logger.log_image(
                key="train/rasterized_samples",
                caption=[f"GT: {text_condition}, temp: {round(temperature, ndigits=2)}, VQ context is marked red."] * len(images),
                images=images,
            )
        else:
            out, logging_dict = self.forward(text_tokens, text_attention_mask, vq_tokens, logging=False)

        if batch_idx % self.train_metric_log_interval == 0 and self.wandb:
            with torch.no_grad():
                num_samples = 8
                temperatures = [random.uniform(0.0, 1.5) for _ in range(num_samples)]
                clip_score_metric, generations, texts = self._get_clip_score_for_batch(
                    text_tokens[:num_samples],
                    text_attention_mask[:num_samples],
                    vq_tokens[:num_samples],
                    post_process=self.post_process,
                    temperatures=temperatures
                )
                self.log("train/clip_score", clip_score_metric, rank_zero_only=True, logger=True, on_step=True)
                self.trainer.logger.log_image(
                    key="train/generated_samples",
                    caption=[text+f", temp: {round(temperatures[i], ndigits=2)}" for i, text in enumerate(texts)],
                    images=generations,
                )

        pred_logits = out  # (b, vq_token_len)
        pred_logits = pred_logits.reshape(-1, pred_logits.shape[-1])

        targets = vq_targets.view(-1)

        # mask out pad token for loss calculation
        mask = targets != pad_token[0]
        pred_logits = pred_logits[mask]
        targets = targets[mask]

        # This is logging a table of tokens to the wandb dashboard
        if batch_idx % self.train_log_interval == 0 and self.wandb:
            target_unique_values, target_counts = torch.unique(targets.detach().cpu(), return_counts=True)
            pred_unique_values, pred_counts = torch.unique(pred_logits.detach().cpu().argmax(dim=1), return_counts=True)
            df = pd.DataFrame(zip(target_unique_values.tolist(), target_counts.tolist()),
                              columns=["token_idx", "target_count"])
            df_pred = pd.DataFrame(zip(pred_unique_values.tolist(), pred_counts.tolist()),
                                   columns=["token_idx", "pred_count"])
            df = pd.merge(df, df_pred, on='token_idx', how='outer').fillna(0)
            df["target_count"] = df["target_count"].astype(int)
            df["pred_count"] = df["pred_count"].astype(int)
            sorted_df = df.sort_values(by='target_count', ascending=False).reset_index(drop=True)
            self.trainer.logger.log_table("train/target_pred_token_counts", dataframe=sorted_df)

        loss_dict = self.model.loss_function(
            targets=targets,
            pred_logits=pred_logits,
        )

        self.log_dict(loss_dict, logger=True, rank_zero_only=True)
        self.log("train_loss", loss_dict["loss"].detach().item(), rank_zero_only=True)
        return loss_dict["loss"]

    def on_train_epoch_end(self):
        # gc.collect()
        # torch.cuda.empty_cache()
        return {}

    def validation_step(self, batch, batch_idx, optimizer_idx=0):

        text_tokens, text_attention_mask, vq_tokens, vq_targets, pad_token = batch
        self.curr_device = text_tokens.device

        with torch.no_grad():
            if batch_idx % self.val_log_interval == 0 and self.wandb:
                out, logging_dict = self.forward(text_tokens, text_attention_mask, vq_tokens, logging=True)
                text_condition = self.tokenizer.decode_text(text_tokens[0])

                # every third batch use temp = 0
                if batch_idx % (self.val_log_interval * 3) == 0:
                    temperature = 0.0
                else:
                    temperature = random.uniform(0.05, 0.7)

                rasterized_gt = self.tokenizer._tokens_to_image_tensor(vq_targets[:1], post_process=self.post_process)
                context_0_generation = self._generate_rasterized_sample(text_tokens[:1, :], text_attention_mask[:1, :],
                                                                        vq_tokens[:1, :1],
                                                                        post_process=self.post_process,
                                                                        temperature=temperature)
                context_5_generation = self._generate_rasterized_sample(text_tokens[:1, :], text_attention_mask[:1, :],
                                                                        vq_tokens[:1, :6],
                                                                        post_process=self.post_process,
                                                                        temperature=temperature)
                context_10_generation = self._generate_rasterized_sample(text_tokens[:1, :], text_attention_mask[:1, :],
                                                                         vq_tokens[:1, :11],
                                                                         post_process=self.post_process,
                                                                         temperature=temperature)
                
                images = [rasterized_gt, context_0_generation, context_5_generation, context_10_generation]
                self.trainer.logger.log_image(
                    key="val/rasterized_samples",
                    caption=[f"GT: {text_condition}, temp: {round(temperature, ndigits=2)}, VQ context is marked red."] * len(images),
                    images=images,
                )
            else:
                out, logging_dict = self.forward(text_tokens, text_attention_mask, vq_tokens, logging=False)

            if batch_idx % self.val_metric_log_interval == 0 and self.wandb:
                num_samples = 8
                temperatures = [random.uniform(0.0, 0.7) for _ in range(num_samples)]
                clip_score_metric, generations, texts = self._get_clip_score_for_batch(
                    text_tokens[:num_samples],
                    text_attention_mask[:num_samples],
                    vq_tokens[:num_samples],
                    post_process=self.post_process,
                    temperatures=temperatures
                )
                self.log("val/clip_score", clip_score_metric, rank_zero_only=True, logger=True)
                self.trainer.logger.log_image(
                    key="val/generated_samples",
                    caption=[text+f", temp: {round(temperatures[i], ndigits=2)}" for i, text in enumerate(texts)],
                    images=generations,
                )

        pred_logits = out  # (b, vq_token_len)
        pred_logits = pred_logits.reshape(-1, pred_logits.shape[-1])

        targets = vq_targets.view(-1)
        # mask out pad token for loss calculation
        mask = targets != pad_token[0]
        pred_logits = pred_logits[mask]
        targets = targets[mask]

        # disabled for now, doesnt serve purpose
        if False and batch_idx % self.val_log_interval == 0 and self.wandb:
            target_unique_values, target_counts = torch.unique(targets.detach().cpu(), return_counts=True)
            pred_unique_values, pred_counts = torch.unique(pred_logits.detach().cpu().argmax(dim=1), return_counts=True)
            df = pd.DataFrame(zip(target_unique_values.tolist(), target_counts.tolist()),
                              columns=["token_idx", "target_count"])
            df_pred = pd.DataFrame(zip(pred_unique_values.tolist(), pred_counts.tolist()),
                                   columns=["token_idx", "pred_count"])
            df = pd.merge(df, df_pred, on='token_idx', how='outer').fillna(0)
            df["target_count"] = df["target_count"].astype(int)
            sorted_df = df.sort_values(by='target_count', ascending=False).reset_index(drop=True)
            self.trainer.logger.log_table("val/target_pred_token_counts", dataframe=sorted_df)

        loss_dict = self.model.loss_function(
            targets=targets,
            pred_logits=pred_logits,
        )

        self.log("val_loss", loss_dict["loss"], sync_dist=True)
        return loss_dict["loss"]

    def on_validation_end(self) -> None:
        # if self.wandb:
        #     self.sample_images()
        # gc.collect()
        # torch.cuda.empty_cache()
        return {}

    def configure_optimizers(self):

        optims = []
        scheds = []

        param_group_1 = {'params': self.model.parameters(), 'lr': self.lr}
        param_groups = [param_group_1]

        if self.weight_decay is not None:
            optimizer = optim.AdamW(
                param_groups,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            # learning rates should be explicitly specified in the param_groups
            optimizer = optim.Adam(param_groups)
        optims.append(optimizer)

        try:
            warmup_steps = self.warmup_steps or 0
            if self.scheduler_gamma is not None:
                scheduler = get_cosine_schedule_with_warmup(optims[0],
                                                            num_warmup_steps=warmup_steps,
                                                            num_training_steps=self.total_training_steps,
                                                            last_epoch=self.current_epoch)
                scheds.append(scheduler)
                return optims, scheds
            else:
                scheduler = get_linear_schedule_with_warmup(optims[0],
                                                            warmup_steps,
                                                            self.total_training_steps,
                                                            last_epoch=self.current_epoch)
                scheds.append(scheduler)
                return optims, scheds
        except:
            pass
        return optims


class VectorVQVAE_Experiment_Stage1(pl.LightningModule):
    """
    Vector quantized pre-training of an autoencoder for SVG primitives.
    
    Input/Output are shape layers and positions.
    """

    def __init__(self,
                 model: VSQ,
                 lr: float = 0.0003,
                 schedule_pyramid_method: str = None,
                 weight_decay: float = 0.0,
                 scheduler_gamma: float = 0.99,
                 train_log_interval: float = 0.05,
                 val_log_interval:float = 0.1,
                 manual_seed: int = 42,
                 min_lr: float = 1.e-6,
                 step_lr_epoch_step_size: int = 30,
                 scheduler_type: str = "cosine",
                 wandb: bool = True,
                 datamodule = None,
                 max_epochs:int=300,
                 **kwargs) -> None:
        super(VectorVQVAE_Experiment_Stage1, self).__init__()

        # self.automatic_optimization = False


        assert train_log_interval < 1 and train_log_interval >= 0, f"train log interval should be a fraction of the total number of batches in [0, 1), got {train_log_interval}"
        # assert metric_log_interval < 1 and metric_log_interval >= 0, f"metric log interval should be a fraction of the total number of batches in [0, 1), got {metric_log_interval}"
        # self.train_log_interval = max(1, int(train_log_interval * num_batches_train))
        self.num_batches_train = len(datamodule.train_dataloader())
        self.num_batches_val = len(datamodule.val_dataloader())

        self.model = model
        self.lr = lr
        self.total_steps = max_epochs * self.num_batches_train
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.train_log_interval =  max(1, int(train_log_interval * self.num_batches_train))
        self.val_log_interval = max(1, int(val_log_interval * self.num_batches_val))
        self.manual_seed = manual_seed
        self.curr_device = None
        self.wandb = wandb
        self.datamodule = datamodule
        self.scheduler_type = scheduler_type
        self.step_size = step_lr_epoch_step_size
        self.schedule_pyramid_method = schedule_pyramid_method
        
        self.start_weights = torch.tensor([1/2, 1/2, 1/2, 1/4, 1/4, 1/8, 1/8])
        self.end_weights = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        interpolation_epochs = min(10, max_epochs)

        if self.schedule_pyramid_method is not None and self.schedule_pyramid_method.lower() == "linear":
            self.pyramid_weight_schedule = interpolate_rows(self.start_weights, self.end_weights, interpolation_epochs, method="linear")
        elif self.schedule_pyramid_method is not None and self.schedule_pyramid_method.lower() == "exponential":
            self.pyramid_weight_schedule = interpolate_rows(self.start_weights, self.end_weights, interpolation_epochs, method="exponential")
        else:
            self.pyramid_weight_schedule = self.start_weights.unsqueeze(0)

    def forward(self, input_images: Tensor, logging=False,**kwargs) -> list:
        out, logging_dict = self.model.forward(input_images, logging=logging, **kwargs)
        return out, logging_dict
    
    def training_step(self, batch, batch_idx):
        all_center_shapes, labels, centers, descriptions = batch
        self.curr_device = all_center_shapes.device
        bs = all_center_shapes.shape[0]
        channels = all_center_shapes.shape[1]
        if batch_idx % self.train_log_interval == 0 and self.wandb:
            out, logging_dict = self.forward(all_center_shapes, logging=True)
        else:
            out, logging_dict = self.forward(all_center_shapes, logging=False)  # out is [reconstructions, input, all_points, vq_loss]
        reconstructions=out[0]
        inputs = all_center_shapes
        all_points = out[2]
        vq_loss=out[3]
        # all_points.retain_grad()

        if self.schedule_pyramid_method is not None:
            if self.current_epoch < len(self.pyramid_weight_schedule):
                pyramid_weights = self.pyramid_weight_schedule[self.current_epoch]
            else:
                pyramid_weights = self.pyramid_weight_schedule[-1]
        else:
            pyramid_weights = self.start_weights

        loss_dict = self.model.loss_function(
            reconstructions=reconstructions[:,:channels,:,:],
            gt_images=inputs,
            vq_loss=vq_loss,
            points=all_points,
            pyramid_weights=pyramid_weights
        )
    
        # always log the first batch and variable amount of timesteps up to 10
        if batch_idx % self.train_log_interval == 0 and self.wandb:
            with torch.no_grad():
                logging_dict = {f"train/{key}": value for key, value in logging_dict.items()}
                wandb.log(logging_dict)
                random_idx = random.randint(0, len(self.datamodule.train_dataset) - 1)
                if isinstance(self.datamodule, VSQDatamodule):
                    dataset_name = "glyphazzn"
                elif isinstance(self.datamodule, MNISTDataset):
                    dataset_name = "mnist"
                side_by_side_recons, drawing = get_side_by_side_reconstruction(self.model, 
                                                                               self.datamodule.train_dataset, 
                                                                               idx = random_idx, 
                                                                               device = self.curr_device, 
                                                                               dataset_name=dataset_name,
                                                                               return_drawing=True)
                wandb.log({"train/side_by_side_recons":wandb.Image(side_by_side_recons, caption="side by side reconstructions of training sample")})
                
                svg_out_dir = os.path.join(self.logger.save_dir, "svgs")
                if not os.path.exists(svg_out_dir):
                    os.makedirs(svg_out_dir, exist_ok=True)
                drawing.saveas(os.path.join(svg_out_dir, f"epoch={self.current_epoch}batchidx={batch_idx}_reconsdrawing.svg"))
                
                if reconstructions.shape[0] > 25:
                    log_amount = 25
                else:
                    log_amount = reconstructions.shape[0]

                if isinstance(self.datamodule, VSQDatamodule):
                    log_reconstructions = add_points_to_image(all_points, reconstructions[:,:3,:,:], image_scale=reconstructions.shape[-1], with_gradient=True, only_start_end=True)
                elif isinstance(self.datamodule, MNISTDataset):
                    log_reconstructions = reconstructions[:,:3,:,:]

                # Log input against prediction
                log_images(
                    log_reconstructions[:log_amount],
                    inputs[:log_amount],
                    log_key="train/reconstruction",
                    captions="input (left) vs. reconstruction (right)"
                )

        self.log_dict(loss_dict, logger=True, prog_bar=True)

        # opt = self.optimizers()
        # opt.zero_grad()
        # loss = loss_dict["loss"]
        # all_points.retain_grad()
        # self.manual_backward(loss)
        # #print(all_points.grad)
        # opt.step()


        return loss_dict["loss"]


    def validation_step(self, batch, batch_idx):
        # print("before batch")
        with torch.no_grad():
            all_center_shapes, label, centers, descriptions = batch
            # print("got batch")
            self.curr_device = all_center_shapes.device
            bs = all_center_shapes.shape[0]
            channels = all_center_shapes.shape[1]

            out, logging_dict = self.forward(all_center_shapes)
            # print("after fwd")
            reconstructions=out[0]
            inputs = all_center_shapes
            all_points = out[2]
            vq_loss=out[3]
            assert vq_loss.dim() <= 1, f"vq_loss should be a 1D tensor, but got {vq_loss.dim()}"

            # for validation we only track MSE
            pyramid_weights = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # print("before loss")
            loss_dict = self.model.loss_function(
                reconstructions=reconstructions[:,:channels,:,:],
                gt_images=inputs,
                vq_loss=vq_loss,
                points=all_points,
                pyramid_weights=pyramid_weights
            )
            # print("after loss")
            # log_reconstructions = add_points_to_image(all_points, reconstructions[:,:3,:,:], image_scale=reconstructions.shape[-1])
            if batch_idx % self.val_log_interval == 0 and self.wandb:
                # print("we loggin")
                logging_dict = {f"val/{key}": value for key, value in logging_dict.items()}
                wandb.log(logging_dict)
                random_idx = random.randint(0, len(self.datamodule.val_dataset) - 1)
                if isinstance(self.datamodule, VSQDatamodule):
                    dataset_name = "glyphazzn"
                elif isinstance(self.datamodule, MNISTDataset):
                    dataset_name = "mnist"
                # print("before sbs")
                side_by_side_recons = get_side_by_side_reconstruction(self.model, self.datamodule.val_dataset, idx = random_idx, device = self.curr_device, dataset_name=dataset_name)
                # print("after sbs")
                wandb.log({"val/side_by_side_recons":wandb.Image(side_by_side_recons, caption="side by side reconstructions of validation sample")})
                if reconstructions.shape[0] > 25:
                    log_amount = 25
                else:
                    log_amount = reconstructions.shape[0]

                # print("before points")
                if isinstance(self.datamodule, VSQDatamodule):
                    log_reconstructions = add_points_to_image(all_points[:log_amount], reconstructions[:log_amount,:3,:,:], image_scale=reconstructions.shape[-1], with_gradient=True, only_start_end=True)
                elif isinstance(self.datamodule, MNISTDataset):
                    log_reconstructions = reconstructions[:log_amount,:3,:,:]
                # print("after points")

                # Log input against prediction
                log_images(
                    log_reconstructions[:log_amount],
                    inputs[:log_amount],
                    log_key="val/reconstruction",
                    captions="input (left) vs. reconstruction (right)"
                )
        mapped_loss_dict = {f"val_{key}": val for key, val in loss_dict.items()}
        self.log_dict(mapped_loss_dict, logger=True, prog_bar=True)
        # print("we done")
        return loss_dict["loss"]

    
    def configure_optimizers(self):

        optims = []
        scheds = []

        param_group_1 = {'params': self.model.parameters(), 'lr': self.lr}
        param_groups = [param_group_1]

        if self.weight_decay is not None:
            optimizer = optim.AdamW(
                param_groups,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            # learning rates should be explicitly specified in the param_groups
            optimizer = optim.Adam(param_groups)
        optims.append(optimizer)

        if self.scheduler_type == "cosine":
            scheds.append(CosineAnnealingLR(optimizer, T_max=self.total_steps, eta_min=self.min_lr))
            return optims, scheds
        elif self.scheduler_type == "step":
            scheds.append(StepLR(optimizer, step_size=self.step_size, gamma=self.scheduler_gamma))
            return optims, scheds
        elif self.scheduler_type == "exponential":
            try:
                if self.scheduler_gamma is not None:
                    scheduler = optim.lr_scheduler.ExponentialLR(optims[0], gamma = self.scheduler_gamma)
                    scheds.append(scheduler)
                    return optims, scheds
            except:
                return optims
        elif self.scheduler_type == "none":
            return optims
        else:
            raise Exception(f"Unknown scheduler for this training: {self.scheduler_type}")

    # def on_train_batch_end(self, output, batch, batch_index):
    #     # Perform evaluation after every eval_steps steps
    #     if batch_index % self.eval_steps == 0:
    #         self.trainer.fit_loop.epoch_loop.val_loop.run()



class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.beta_scale = 2.0 # introduced with Im2Vec
        self.lr = params["LR"]

        if "offset_LR" in params and self.model.decoder.offset_mode == "learnable":
            self.offset_LR = params["offset_LR"]
        else:
            self.offset_LR = None

        if "offset_warmup" in params and self.model.decoder.offset_mode in ["learnable", "optimizable"]:
            self.offset_warmup = params["offset_warmup"]
        else:
            self.offset_warmup = False
        
        if "log_fid" in self.params.keys():
            self.log_fid = True if self.params["log_fid"] else False
            if self.log_fid:
                self.fid = FrechetInceptionDistance(feature=768, reset_real_features=False, normalize=True)
        else:
            self.log_fid = False
            print("Not logging FID score. To enable, add 'log_fid: True' to the 'exp_params' of the config .yaml file")

        if "clip_sim_model" in self.params.keys() and "clip_prompt_suffix" in self.params.keys():
                self.log_clip_sim = True if self.params["clip_sim_model"] in ['openai/clip-vit-base-patch16', 'openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14-336', 'openai/clip-vit-large-patch14'] else False
                if self.log_clip_sim:
                    self.clip_prompt_suffix = self.params["clip_prompt_suffix"]
                    self.clip_sim = CLIPScore(model_name_or_path=self.params["clip_sim_model"])
                else:
                    self.log_clip_sim = False
                    print(f"""Not logging CLIP similarity score with: {self.params["clip_sim_model"]}. To enable, add one of ['openai/clip-vit-base-patch16', 'openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14-336', 'openai/clip-vit-large-patch14']
                           to the 'clip_sim_model' parameter in 'exp_params' of the config .yaml file""")

        else:
            self.log_clip_sim = False
            print("Not logging CLIP similarity score. To enable, add 'clip_sim_model' to the 'exp_params' of the config .yaml file with one of ['openai/clip-vit-base-patch16', 'openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14-336', 'openai/clip-vit-large-patch14'] and add 'clip_prompt_suffix' also.")

        
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)
    
    def on_train_epoch_start(self):
        if self.offset_warmup:
            for name, param in self.model.named_parameters():
                if "decoder.offset" in name or "encoder." in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        # regularely log training reconstructions
        if batch_idx % self.params["train_log_interval"] == 0:
            results = self.forward(real_img, labels = labels, log_path_length=True)
            if results[0].shape[0] > 10:
                log_amount = 10
            else:
                log_amount = results[0].shape[0]
            log_images(results[0][:log_amount], real_img[:log_amount], log_key="training")
            train_loss = self.model.loss_function(*results,
                                            M_N = self.params['kld_weight'],
                                            optimizer_idx=optimizer_idx,
                                            batch_idx = batch_idx,
                                            log_loss_images = False)#was true
            with torch.no_grad():
                self.sample_images()
        else:
            results = self.forward(real_img, labels = labels)
            train_loss = self.model.loss_function(*results,
                                                M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                                optimizer_idx=optimizer_idx,
                                                batch_idx = batch_idx)

        self.log_dict(train_loss, sync_dist=True, prog_bar=True)

        # Custom processing for Im2Vec
        if(isinstance(self.model, VectorVAEnLayers)):
            path = random.randint(7, 25)
            if self.params['resample_circle_segments']:
                self.model.redo_features(path)

        return train_loss['loss']
    
    def on_train_epoch_end(self):
        if self.offset_warmup:
            for name, param in self.model.named_parameters():
                param.requires_grad = True
        if isinstance(self.model, VectorVAEnLayers):
            if self.current_epoch % 25 ==0:
                new_beta = self.model.beta * self.beta_scale
                self.model.beta = min(new_beta, 4)
        gc.collect()
        torch.cuda.empty_cache()
        return {}

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        print("Entering validation step.")
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val for key, val in val_loss.items()}, sync_dist=True, prog_bar=True)

        return val_loss["loss"]

        
    def on_validation_end(self) -> None:
        print("Entering on_validation_end.")
        with torch.no_grad():
            self.sample_images()

        gc.collect()
        torch.cuda.empty_cache()
        
    def sample_images(self):
        print("Sampling images for wandb.")
        # Get sample reconstruction image            
        if(self.log_clip_sim or self.log_fid):
            num_of_samples = 100
        else:
            num_of_samples = 10
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input[:num_of_samples].to(self.curr_device)
        test_label = test_label[:num_of_samples].to(self.curr_device)

        with torch.no_grad():
            # test_input, test_label = batch
            recons = self.model.generate(test_input, labels = test_label, verbose=True)
        
        # make sure there are no small negative numbers for rendering
        dummy = torch.nn.ReLU()
        recons = dummy(recons)
        
        log_images(recons[:5], test_input[:5], log_key="val_recons")

        # if(self.logger.save_dir is not None):
        #     vutils.save_image(recons.data[:10],
        #                     os.path.join(self.logger.save_dir , 
        #                                 "Reconstructions", 
        #                                 f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                     normalize=True,
        #                     nrow=5)

        try:
            print("sampling from model.")
            samples = self.model.multishape_sample(num_of_samples, return_points=False, device=test_input.device)
            if not isinstance(samples, Tensor):
                samples = torch.stack(samples, dim=0)
            if samples.ndim >4:
                samples = samples[:,0,:,:,:]
            samples = dummy(samples[:,:3,:,:]).cpu() # drop the alpha channel for metric calculation
            log_images(samples[:5], samples[5:10], log_key="samples")
            # if(self.logger.save_dir is not None):
            #     vutils.save_image(samples.cpu().data,
            #                     os.path.join(self.logger.save_dir , 
            #                                 "Samples",      
            #                                 f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
            #                     normalize=True,
            #                     nrow=5)
            # Log FID score
            if self.log_fid:
                self.fid.update(test_input, real = True)

                # log reconstruction fid
                self.fid.update(recons, real = False)
                fid_recon_score = self.fid.compute()

                # log sample fid
                self.fid.update(samples, real = False)
                fid_sample_score = self.fid.compute()

                self.fid.reset()

                self.logger.log_metrics({"val_recons_FID" : fid_recon_score, "val_sample_FID" : fid_sample_score})#, sync_dist=True ,prog_bar=True)

            if self.log_clip_sim:
                
                _label_translate_dict = self.trainer.datamodule.val_dataset._int_to_label
                # was used to test VRAM usage
                # _max_clip_calculations = 100
                
                self.clip_sim.update(recons, [_label_translate_dict[label]+self.clip_prompt_suffix for label in test_label.cpu().numpy()])
                clip_sim_recon_score = self.clip_sim.compute()
                self.clip_sim.reset()

                self.clip_sim.update(samples, [_label_translate_dict[label]+self.clip_prompt_suffix for label in test_label.cpu().numpy()[:num_of_samples]])
                clip_sim_sample_score = self.clip_sim.compute()
                self.clip_sim.reset()

                self.logger.log_metrics({"val_recons_CLIP_sim" : clip_sim_recon_score, "val_sample_CLIP_sim" : clip_sim_sample_score})#, sync_dist=True ,prog_bar=True)
                
        except Exception as e:
            print(f"[ERROR] at sampling")
            pass

    def _configure_optimizers(self):

        optims = []
        scheds = []
        # if self.model.only_auxillary_training:
        #     print('Learning Rate changed for auxillary training')
        #     self.params['LR'] = 0.00001
        param_group_1 = {'params': self.model.parameters(), 'lr': self.lr}
        param_groups = [param_group_1]
        if(self.params["weight_decay"] is not None):
            optimizer = optim.AdamW(param_groups,
                                lr=self.lr,
                                weight_decay=self.params['weight_decay'])
        else:
            optimizer = optim.Adam(param_groups,
                                lr=self.lr)
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        # try:
        #     if self.params['LR_2'] is not None:
        #         optimizer2 = optim_.AdamP(getattr(self.model,self.params['submodel']).parameters(),
        #                                 lr=self.params['LR_2'])
        #         optims.append(optimizer2)
        # except:
        #     pass

        # scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
        #                                              gamma = self.params['scheduler_gamma'], last_epoch=450)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optims[0], 'min', verbose=True, factor=self.params['scheduler_gamma'], min_lr=0.0001, patience=int(self.model.memory_leak_epochs/7))
        scheduler = optim.lr_scheduler.CyclicLR(optims[0], self.params['LR']*0.1, self.params['LR'], mode='exp_range',
                                                     gamma = self.params['scheduler_gamma'],cycle_momentum=False)
        # scheduler = optim.lr_scheduler.OneCycleLR(optims[0], max_lr=self.params['LR'], steps_per_epoch=130, epochs=2000)
        # scheduler = GradualWarmupScheduler(optims[0], multiplier=1, total_epoch=20,
        #                                           after_scheduler=scheduler)

        scheds.append({
         'scheduler': scheduler,
         'monitor': 'val_loss', # Default: val_loss
         'interval': 'epoch',
         'frequency': 1,
        },)

        # Check if another scheduler is required for the second optimizer
        try:
            if self.params['scheduler_gamma_2'] is not None:
                scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                              gamma = self.params['scheduler_gamma_2'])
                scheds.append(scheduler2)
        except:
            pass
        print('USING WARMUP SCHEDULER')
        return optims, scheds

    def configure_optimizers(self):

        optims = []
        scheds = []

        param_groups = []

        if(self.offset_LR is not None):
            offset_params = []
            other_params = []
            # Separate parameters for the 'offset' layer and other model parameters
            for name, param in self.model.named_parameters():
                if(name == "decoder.offset.bias" or name == "decoder.offset.weight"):
                    offset_params.append(param)
                else:
                    other_params.append(param)

            # Set different learning rates for different parameter groups
            param_group_1 = {'params': other_params, 'lr': self.lr}
            param_group_2 = {'params': offset_params, 'lr': self.offset_LR}

            param_groups = [param_group_1, param_group_2]
        else:
            param_group_1 = {'params': self.model.parameters(), 'lr': self.lr}
            param_groups = [param_group_1]


        if(self.params["weight_decay"] is not None):
            optimizer = optim.AdamW(param_groups,
                                lr=self.lr,
                                weight_decay=self.params['weight_decay'])
        else:
            optimizer = optim.Adam(param_groups,
                                lr=self.lr)
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            pass
        return optims
