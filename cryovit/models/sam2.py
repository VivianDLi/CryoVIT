# """SAMv2 model for 2D/3D tomogram segmentation, using the existing library to support training and fine-tuning."""
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union
import logging
import wandb

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import numpy as np
from sam2.modeling.sam2_base import SAM2Base

from cryovit.models.base_model import BaseModel
from cryovit.models.sam2_blocks import PromptPredictor, LoRAMaskDecoderFactory
from cryovit.config import BaseModel as BaseModelConfig
from cryovit.types import BatchedTomogramData

sam2_model = ("facebook/sam2.1-hiera-tiny", {"config": "sam2.1_hiera_t.yaml", "weights": "sam2.1_hiera_tiny.pt"}) # the tiny variant of SAMv2.1
medical_sam2_model = ("wanglab/MedSAM2", {"config": "sam2.1_hiera_t.yaml", "weights": "MedSAM2_latest.pt"}) # fine-tuned on medical data SAMv2

MAX_SAM_DEPTH = 255 # Temporary maximum depth (number of slices) for SAMv2 (due to implementation error in CUDA - https://github.com/pytorch/pytorch/issues/142228)
# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0

class SAM2(BaseModel):
    """Lightning wrapper over the SAM2 model."""
    
    def __init__(self, sam_model: "SAM2Train", custom_kwargs, **kwargs) -> None:
        """Initializes the SAM2 model with specific convolutional and synthesis blocks."""
        super(SAM2, self).__init__(**kwargs)
        self.model = sam_model(**custom_kwargs)
        self.prompt_predictor = PromptPredictor()
        
        self.freeze_parameters()
        self.log_masks = True # whether to log predicted masks during training for debugging
        
    def freeze_parameters(self):
        """Freezes all model parameters except for the prompt predictor and mask decoder."""
        for p in self.model.image_encoder.parameters():
            p.requires_grad = False
        for p in self.model.sam_prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.model.memory_encoder.parameters():
            p.requires_grad = False
        for p in self.model.memory_attention.parameters():
            p.requires_grad = False

    def configure_optimizers(self) -> Optimizer:
        """Configures the optimizer with the initialization parameters."""
        prompt_params = {
            "params": self.prompt_predictor.parameters(),
            "lr": 3e-5,
        }
        decoder_params = {
            "params": self.model.parameters(),
        }
        return torch.optim.AdamW(
            [
                prompt_params,
                decoder_params
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _masked_predict(
        self, batch: BatchedTomogramData
    ) -> Dict[str, Tensor]:
        """Override trainer _masked_predict to handle masking for the prompt predictor."""
        out = self(batch)
        y_true = batch.labels # (B, D, H, W)
        y_pred_full, mask_pred_full = out["preds"], out["prompts"]

        mask = (y_true > -1.0).detach()

        y_pred = torch.masked_select(y_pred_full, mask).view(-1, 1)
        y_true = torch.masked_select(y_true, mask).view(-1, 1)
        mask_pred = torch.masked_select(mask_pred_full, mask).view(-1, 1)

        return {
            "preds": y_pred,
            "masks": mask_pred,
            "labels": y_true,
            "preds_full": y_pred_full,
            "masks_full": mask_pred_full
        }

    def _do_step(self, batch: BatchedTomogramData, batch_idx: int, prefix: Literal["train", "val", "test"]) -> float:
        """Override trainer do_step to handle losses for the prompt predictor."""
        out_dict = self._masked_predict(batch)
        
        y_pred, y_true = out_dict["preds"], out_dict["labels"]

        losses = self.compute_losses(y_pred, y_true)
        mask_loss = self.compute_losses(out_dict["masks"], y_true)["dice_loss"]
        losses["mask_loss"] = mask_loss
        losses["total"] = losses["total"] + mask_loss

        for _, m_fn in self.metric_fns[prefix.upper()].items():
            m_fn(y_pred, y_true)

        self.log_stats(losses, prefix, batch.num_tomos)
        if self.training and self.log_masks:
            # debug logging predicted masks
            raw_image = batch.tomo_batch[0, batch.num_slices // 2].detach().cpu()[[0]]
            pred_image = out_dict["preds_full"][0, batch.num_slices // 2].detach().cpu().unsqueeze(0)
            mask_image = out_dict["masks_full"][0, batch.num_slices // 2].detach().cpu().unsqueeze(0)
            prompt_image = (mask_image > 0.9).float()
            combined_image = torch.cat([raw_image, pred_image, mask_image, prompt_image], dim=-1) # concat along width
            wandb.log({"raw-pred-mask-prompt": [wandb.Image(combined_image, caption="Raw | Pred | Mask | Prompt")]}, commit=False)
        
        return losses["total"]

    def forward(self, data: BatchedTomogramData) -> Dict[str, Tensor]:
        C, H, W = data.tomo_batch.shape[-3:] # [H, W]
        # Expand channels for expected RGB input
        if C == 1:
            data.tomo_batch = data.tomo_batch.expand(-1, -1, 3, -1, -1)
            C = 3
        # Truncate if too many slices
        do_truncate = data.num_slices > MAX_SAM_DEPTH
        do_resize = (H != self.model.image_size or W != self.model.image_size)
        if do_truncate:
            logging.warning(f"Truncating input tomogram from {data.num_slices} to {MAX_SAM_DEPTH} slices for SAM2 model.")
            data.tomo_batch = data.tomo_batch[:, :MAX_SAM_DEPTH]
            data.tomo_sizes = torch.clamp(data.tomo_sizes, max=MAX_SAM_DEPTH)
            data.min_slices = min(data.min_slices, MAX_SAM_DEPTH)
        if do_resize:
            # Resize the input tomogram batch to the target size
            data.tomo_batch = F.interpolate(data.tomo_batch, size=(C, self.model.image_size, self.model.image_size), mode="trilinear", align_corners=False)

        flat_tensor = data.flat_tomo_batch # [BxDxCxHxW] -> [[BxD]xCxHxW]
        backbone_out = self.model.image_encoder(flat_tensor)
        flat_box_prompts, flat_mask_prompts = self.prompt_predictor(backbone_out["backbone_fpn"][0], num_batches=data.num_tomos) # flat tensor form
        binary_flat_mask_prompts = (flat_mask_prompts > 0.9).bool() # binarize the mask prompts for SAM2 input conservatively
        preds = self.model(data, flat_box_prompts, binary_flat_mask_prompts) # forward pass through SAM2
        masks = flat_mask_prompts.view(data.num_tomos, data.num_slices, *flat_mask_prompts.shape[-2:]) # reshape to [B, D, H, W]
        
        out = {
            "preds": preds,
            "prompts": masks
        }
        
        # Pad outputs if truncated
        if do_truncate:
            pad_size = (0, 0, 0, 0, 0, data.num_slices - MAX_SAM_DEPTH)
            for k in out:
                out[k] = F.pad(out[k], pad_size, mode="constant", value=0)
        if do_resize:
            # Upsample the output to the original size
            for k in out:
                out[k] = F.interpolate(out[k], size=(H, W), mode="bilinear", align_corners=False)
            # Resize the input tomogram batch to the original size
            data.tomo_batch = F.interpolate(data.tomo_batch, size=(C, H, W), mode="trilinear", align_corners=False)
        return out
        
    def load_sam_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = False, assign: bool = True) -> Tuple:
        """Override load_state_dict to handle loading of SAM2 weights."""
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def compile(self) -> None:
        """Compiles the model image encoder for training."""
        self.model.image_encoder.forward = torch.compile(self.model.image_encoder.forward)
        self.model.memory_encoder.forward = torch.compile(self.model.memory_encoder.forward)
        self.model.memory_attention.forward = torch.compile(self.model.memory_attention.forward)

class SAM2Train(SAM2Base):
    """SAMv2 model implementation."""

    def __init__(self, image_encoder: nn.Module, memory_attention: nn.Module, memory_encoder: nn.Module, num_init_cond_slices: Tuple[int, int] = (1, 1), rand_init_cond_slices: Tuple[bool, bool] = (True, False), **kwargs) -> None:
        """Initializes the CryoVIT model with specific convolutional and synthesis blocks."""
        super(SAM2Train, self).__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        self.num_init_cond_slices = num_init_cond_slices
        self.rand_init_cond_slices = rand_init_cond_slices
                
    def _apply_lora_to_mask_decoder(self):
        """Delay applying LoRA to the mask decoder until after loading weights."""
        decoder_factory = LoRAMaskDecoderFactory(lora_r=128, lora_alpha=128) # Using alpha=r
        self.sam_mask_decoder = decoder_factory.apply(self.sam_mask_decoder)

    def forward(self, data: BatchedTomogramData, box_prompts, mask_prompts) -> Tensor:
        """Forward pass for the SAMv2 model."""
        backbone_out = self.forward_image(data.flat_tomo_batch)
        mid_slice_idx = data.num_slices // 2
        backbone_out = self.prepare_prompt_inputs(backbone_out, box_prompts, mask_prompts, data, start_slice_idx=mid_slice_idx)
        out = self.forward_tracking(backbone_out, data)
        out = torch.sigmoid(out)
        return out

    def prepare_prompt_inputs(self, backbone_out: Dict[str, Any], box_prompts: Dict[str, Any], mask_prompts: Dict[str, Any], data: BatchedTomogramData, start_slice_idx: int = 0) -> Dict[str, Any]:
        """Prepare predicted masks."""
        backbone_out["num_slices"] = data.num_slices
        # Setup prompt parameters
        if self.training:
            num_init_cond_slices = self.num_init_cond_slices[0]
            rand_init_cond_slices = self.rand_init_cond_slices[0]
        else:
            num_init_cond_slices = self.num_init_cond_slices[1]
            rand_init_cond_slices = self.rand_init_cond_slices[1]
        assert num_init_cond_slices >= 1, "Number of initial conditioning slices must be at least 1."
        if rand_init_cond_slices and num_init_cond_slices > 1:
            # Randomly select number of initial conditioning slices
            num_init_cond_slices = np.random.randint(1, num_init_cond_slices + 1)
        
        # Select initial conditioning slices
        if num_init_cond_slices == 1:
            init_cond_slices = [start_slice_idx]
        else:
            init_cond_slices = [start_slice_idx] + np.random.choice(
                a=range(start_slice_idx + 1, data.min_slices),
                size=num_init_cond_slices - 1,
                replace=False
            ).tolist()
        backbone_out["init_cond_slices"] = init_cond_slices
        backbone_out["slices_not_in_init_cond"] = [n for n in range(data.num_slices) if n not in init_cond_slices]

        # Prepare mask and box inputs for slices
        backbone_out["box_inputs_per_slice"] = {}
        backbone_out["mask_inputs_per_slice"] = {}
        for n in init_cond_slices:
            idxs = data.index_to_flat_batch(n)
            backbone_out["box_inputs_per_slice"][n] = box_prompts[idxs] * self.sam_prompt_encoder.input_image_size[0]
            backbone_out["mask_inputs_per_slice"][n] = mask_prompts[idxs]

        return backbone_out

    def forward_tracking(self, backbone_out: Dict[str, Any], data: BatchedTomogramData, return_dict: bool = False) -> Union[Dict[str, Any], Tensor]:
        """Forward video tracking on each slice."""
        # Prepare backbone features
        # backbone_out is [[BxD]xCxHxW]
        # vision_feats and vision_pos_embeds are [(HW), (BD), C]
        _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)
        
        # Start loop over slices
        num_slices = backbone_out["num_slices"]
        init_cond_slices = backbone_out["init_cond_slices"]
        # First process initial conditioning slices, then condition on them for memory
        processing_order = init_cond_slices + backbone_out["slices_not_in_init_cond"]
        # Use "frame" instead of "slice" to match with SAM2 implementation
        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        for slice_id in processing_order:
            flat_idxs = data.index_to_flat_batch(slice_id)
            # Get image features for the current slice
            current_vision_feats = [x[:, flat_idxs] for x in vision_feats]
            current_vision_pos_embeds = [x[:, flat_idxs] for x in vision_pos_embeds]

            current_out = self.track_step(
                slice_id=slice_id,
                is_init_cond_slice=slice_id in init_cond_slices,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                box_inputs=backbone_out["box_inputs_per_slice"].get(slice_id, None),
                mask_inputs=backbone_out["mask_inputs_per_slice"].get(slice_id, None),
                output_dict=output_dict,
                num_slices=num_slices,
            )

            add_output_as_cond_slice = slice_id in init_cond_slices
            if add_output_as_cond_slice:
                output_dict["cond_frame_outputs"][slice_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][slice_id] = current_out
        if return_dict:
            return output_dict

        # turn 'output_dict' into a batched tensor for loss function (expects [B, D, H, W] output)
        all_slice_outputs = {}
        all_slice_outputs.update(output_dict["cond_frame_outputs"])
        all_slice_outputs.update(output_dict["non_cond_frame_outputs"])
        pred_output = []
        for slice_id, output_dict in all_slice_outputs.items():
            # Upsample to original size (from low-res masks)
            preds = F.interpolate(output_dict["pred_masks"], scale_factor=4, mode="bilinear", align_corners=False)
            pred_output.append(preds)
        total_output = torch.cat(pred_output, dim=1)

        return total_output

    def track_step(self, slice_id: int, is_init_cond_slice: bool, current_vision_feats: Tensor, current_vision_pos_embeds: Tensor, feat_sizes: Tensor, box_inputs: Optional[Tensor], mask_inputs: Optional[Tensor], output_dict: Dict[str, Any], num_slices: int) -> None:
        # Run the tracking step for the current slice
        current_out, sam_outputs, _, _ = self._track_step(slice_id, is_init_cond_slice, current_vision_feats, current_vision_pos_embeds, feat_sizes, box_inputs, mask_inputs, output_dict, num_slices, False, None)

        # Only save essential outputs to reduce memory usage
        (
            low_res_multimasks,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs
        # Combine multimask outputs into one mask by taking the max
        if low_res_multimasks is not None:
            low_res_masks = torch.max(low_res_multimasks, dim=1, keepdim=True).values
        current_out["pred_masks"] = low_res_masks
        current_out["obj_ptr"] = obj_ptr
        
        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future slices)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            None,
            True, # run_mem_encoder
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out
    
    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        box_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        current_out = {"point_inputs": None, "box_inputs": box_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert box_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, None)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                box_inputs=box_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    def _forward_sam_heads(
        self,
        backbone_features,
        box_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """Forward SAM prompt encoders and mask heads."""
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts by padding with an empty point with label -1
        sam_point_coords = torch.zeros(B, 1, 2, device=device)
        sam_point_labels = -1 * torch.ones(B, 1, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=box_inputs,
            masks=sam_mask_prompt,
        )
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

#### Model Creation and Loading ####

def create_sam_model_from_weights(cfg: BaseModelConfig, sam_dir: Path) -> SAM2:
    configs = _download_model_weights(sam_dir)
    assert cfg.name in configs, f"Model {cfg.name} was not found in available SAMv2 models. Available models are {configs.keys()}."

    file_paths = configs[cfg.name]
    
    # Merge configs together
    model_cfg_path = file_paths["config"]
    model_cfg = OmegaConf.load(model_cfg_path)["model"]
    model_cfg._target_ = "cryovit.models.sam2.SAM2Train" # Use cryovit SAM2 as target
    model_cfg.image_size = 256 # Set image size to 512 (crop size for training)
    model_cfg.use_mask_input_as_output_without_sam = False # use sam memory and mask decoder
    model_cfg._partial_ = True
    
    model = instantiate(cfg, sam_model=model_cfg, custom_kwargs=cfg.custom_kwargs)
    sd = torch.load(file_paths["weights"], map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_sam_state_dict(sd)
    if missing_keys:
        logging.error(missing_keys)
        raise RuntimeError()
    if unexpected_keys:
        logging.error(unexpected_keys)
        raise RuntimeError()
    model.model._apply_lora_to_mask_decoder() # Apply LoRA after loading weights
    model.configure_optimizers() # Configure optimizers after setting requires_grad

    return model

def _download_model_weights(sam_dir: Path) -> Dict[str, Dict[str, Path]]:
    from huggingface_hub import snapshot_download
    
    # Download base SAMv2 model
    sam2_repo, sam2_config = sam2_model
    if not ((sam_dir / sam2_config["weights"]).exists() and (sam_dir / sam2_config["config"]).exists()): 
        snapshot_download(repo_id = sam2_repo, repo_type="model", local_dir=sam_dir)
    sam2_config = {k: sam_dir / v for k, v in sam2_config.items()}
    
    # Download Medical-SAMv2
    medsam_repo, medsam_config = medical_sam2_model
    if not ((sam_dir / medsam_config["weights"]).exists() and (sam_dir / medsam_config["config"]).exists()):
        snapshot_download(repo_id = medsam_repo, repo_type="model", local_dir=sam_dir)
    medsam_config = {k: sam_dir / v for k, v in medsam_config.items()}
    
    return {
        "SAM2": sam2_config,
        "MedSAM": medsam_config
    }
    
def _mask_to_box(masks: Tensor) -> Tensor:
    """Converts a binary mask to bounding box coordinates."""
    device = masks.device
    B, _, H, W = masks.shape
    xs = torch.arange(W, device=device, dtype=torch.int32)
    ys = torch.arange(H, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing='xy')
    grid_xs = grid_xs.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
    grid_ys = grid_ys.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, W).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, H).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack([min_xs, min_ys, max_xs, max_ys], dim=-1)
    return bbox_coords
    
def _sample_box_from_mask(masks: Tensor, noise: float = 0.1, noise_bound: int = 20, top_left_label: int = 2, bottom_right_label: int = 3) -> Tuple[Tensor, Tensor]:
    """Samples a bounding box from a mask."""
    device = masks.device
    B, _, H, W = masks.shape
    box_coords = _mask_to_box(masks)
    box_labels = torch.tensor([top_left_label, bottom_right_label], device=device, dtype=torch.int).repeat(B)

    if noise > 0.0:
        if not isinstance(noise_bound, Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand((B, 1, 4), device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)
    
        box_coords = box_coords + box_noise
        img_bounds = torch.tensor([W, H, W, H], device=device) - 1
        box_coords = torch.clamp(box_coords, min=torch.zeros_like(img_bounds), max=img_bounds)
        
    box_coords = box_coords.reshape(-1, 2, 2) # (B, 2, 2)
    box_labels = box_labels.reshape(-1, 2) # (B, 2)
    return box_coords, box_labels