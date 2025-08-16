# """SAMv2 model for 2D/3D tomogram segmentation, using the existing library to support training and fine-tuning."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import logging

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import select_closest_cond_frames, get_1d_sine_pe

from cryovit.models.base_model import BaseModel
from cryovit.models.prompt_predictor import PromptPredictor
from cryovit.config import BaseModel as BaseModelConfig
from cryovit.types import BatchedTomogramData

sam2_model = ("facebook/sam2.1-hiera-large", {"config": "sam2.1_hiera_l.yaml", "weights": "sam2.1_hiera_large.pt"}) # the large variant of SAMv2.1
sam2_small_model = ("facebook/sam2.1-hiera-tiny") # for creating model for medical sam2
medical_sam2_model = ("wanglab/MedSAM2", {"config": "sam2.1_hiera_t.yaml", "weights": "MedSAM2_latest.pt"}) # fine-tuned on medical data SAMv2

class SAM2(BaseModel):
    """Lightning wrapper over the SAM2 model."""
    
    def __init__(self, sam_model: "SAM2Train", **kwargs) -> None:
        """Initializes the SAM2 model with specific convolutional and synthesis blocks."""
        super(SAM2, self).__init__(**kwargs)
        self.model = sam_model
        
    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = False, assign: bool = True) -> Tuple:
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, data: BatchedTomogramData) -> Tensor:
        # Expand channels for expected RGB input
        if data.tomo_batch.size(2) == 1:
            data.tomo_batch = data.tomo_batch.expand(-1, -1, 3, -1, -1)
        return self.model(data)

class SAM2Train(SAM2Base):
    """SAMv2 model implementation."""

    def __init__(self, image_encoder: nn.Module, memory_attention: nn.Module, memory_encoder: nn.Module, prob_use_pt_input: Tuple[float, float] = (0.5, 0), num_init_cond_slices: Tuple[int, int] = (1, 1), rand_init_cond_slices: Tuple[bool, bool] = (True, False), num_learnable_prompt_tokens: int = 10, freeze_image_encoder: bool = True, freeze_prompt_encoder: bool = True, **kwargs) -> None:
        """Initializes the CryoVIT model with specific convolutional and synthesis blocks."""
        super(SAM2Train, self).__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        self.prob_use_pt_input = prob_use_pt_input
        self.num_init_cond_slices = num_init_cond_slices
        self.rand_init_cond_slices = rand_init_cond_slices

        self.prompt_predictor = PromptPredictor(self.sam_prompt_encoder)
        self.learnable_prompts = torch.nn.Parameter(torch.randn(1, num_learnable_prompt_tokens, 256) / np.sqrt(256))

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
                
        if freeze_prompt_encoder:
            for p in self.sam_prompt_encoder.parameters():
                p.requires_grad = False

    def forward(self, data: BatchedTomogramData) -> Tensor:
        """Forward pass for the SAMv2 model."""
        flat_tensor = data.batch_tensor_to_flat_tensor(data.tomo_batch) # [BxDxCxHxW] -> [[BxD]xCxHxW]

        # precompute image features on all slices before tracking for mask generation
        backbone_out = self.forward_image(flat_tensor)
        backbone_out = self.prepare_prompt_inputs(backbone_out, data)
        out = self.forward_tracking(backbone_out, data)

        return torch.sigmoid(out)

    def prepare_prompt_inputs(self, backbone_out: Dict[str, Any], data: BatchedTomogramData, start_slice_idx: int = 0) -> Dict[str, Any]:
        """Prepare a grid-point mask for automatic prompting or an input mask from labeled data."""
        backbone_out["num_slices"] = data.max_slices
        
        # Setup prompt parameters
        if self.training:
            prob_use_pt_input = self.prob_use_pt_input[0]
            num_init_cond_slices = self.num_init_cond_slices[0]
            rand_init_cond_slices = self.rand_init_cond_slices[0]
        else:
            prob_use_pt_input = self.prob_use_pt_input[1]
            num_init_cond_slices = self.num_init_cond_slices[1]
            rand_init_cond_slices = self.rand_init_cond_slices[1]
        assert num_init_cond_slices >= 1, "Number of initial conditioning slices must be at least 1."
        if rand_init_cond_slices and num_init_cond_slices > 1:
            # Randomly select number of initial conditioning slices
            num_init_cond_slices = np.random.randint(1, num_init_cond_slices + 1)
        use_pt_input = np.random.rand() < prob_use_pt_input
        backbone_out["use_pt_input"] = use_pt_input
        
        # Select initial conditioning slices
        num_init_cond_slices = 10
        if num_init_cond_slices == 1:
            init_cond_slices = [start_slice_idx]
        else:
            init_cond_slices = [start_slice_idx] + np.random.choice(
                a=range(start_slice_idx + 1, data.min_slices),
                size=num_init_cond_slices - 1,
                replace=False
            ).tolist()
        backbone_out["init_cond_slices"] = init_cond_slices
        backbone_out["slices_not_in_init_cond"] = [n for n in range(start_slice_idx, backbone_out["num_slices"]) if n not in init_cond_slices]
        
        # Prepare mask and point inputs for each initial conditioning slice
        backbone_out["mask_inputs_per_slice"] = {}
        backbone_out["point_inputs_per_slice"] = {}
        B = data.total_slices
        prompts = self.learnable_prompts.repeat(B, 1, 1)  # [B, num_learnable_prompts, embed_dim]
        flat_box_prompts, flat_mask_prompts = self.prompt_predictor(backbone_out["backbone_fpn"], prompts) # flat tensor form
        for n in init_cond_slices:
            idxs = data.index_to_flat_batch(n)
            if not use_pt_input:
                backbone_out["mask_inputs_per_slice"][n] = flat_mask_prompts[idxs]
            else:
                backbone_out["point_inputs_per_slice"][n] = flat_box_prompts[idxs]
            
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
            "cond_frame_preds": {},
            "non_cond_frame_outputs": {},
            "non_cond_frame_preds": {},
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
                point_inputs=backbone_out["point_inputs_per_slice"].get(slice_id, None),
                mask_inputs=backbone_out["mask_inputs_per_slice"].get(slice_id, None),
                output_dict=output_dict,
                num_slices=num_slices,
            )
            
            # Get predictions from the current output
            preds = current_out["pred_masks_high_res"]

            add_output_as_cond_slice = slice_id in init_cond_slices
            if add_output_as_cond_slice:
                output_dict["cond_frame_outputs"][slice_id] = current_out
                output_dict["cond_frame_preds"][slice_id] = preds
            else:
                output_dict["non_cond_frame_outputs"][slice_id] = current_out
                output_dict["non_cond_frame_preds"][slice_id] = preds

        if return_dict:
            return output_dict

        # turn 'output_dict' into a batched tensor for loss function (expects [B, D, H, W] output)
        all_slice_outputs = {}
        all_slice_outputs.update(output_dict["cond_frame_preds"])
        all_slice_outputs.update(output_dict["non_cond_frame_preds"])
        B, D, _, H, W = data.tomo_batch.shape
        total_output = torch.zeros((B, D, H, W), dtype=torch.float32)
        for slice_id, preds in all_slice_outputs.items():
            flat_idxs = data.index_to_flat_batch(slice_id)
            total_output[flat_idxs, slice_id, ...] = preds

        return total_output

    def track_step(self, slice_id: int, is_init_cond_slice: bool, current_vision_feats: Tensor, current_vision_pos_embeds: Tensor, feat_sizes: Tensor, point_inputs: Optional[Tensor], mask_inputs: Optional[Tensor], output_dict: Dict[str, Any], num_slices: int) -> None:
        # Run the tracking step for the current slice
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(slice_id, is_init_cond_slice, current_vision_feats, current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs, output_dict, num_slices, False, None)

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        
        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future slices)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            True, # run_mem_encoder
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory. Overriding SAM2 implementation to remove memory pinning."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

#### Model Creation and Loading ####

def create_sam_model_from_weights(cfg: BaseModelConfig, sam_dir: Path) -> SAM2:
    configs = _download_model_weights(sam_dir)
    assert cfg.name in configs, f"Model {cfg.name} was not found in available SAMv2 models. Available models are {configs.keys()}."

    file_paths = configs[cfg.name]
    
    # Merge configs together
    model_cfg_path = file_paths["config"]
    model_cfg = OmegaConf.load(model_cfg_path)["model"]
    model_cfg._target_ = "cryovit.models.sam2.SAM2Train" # Use cryovit SAM2 as target
    model_cfg.image_size = 512 # Set image size to 512 (crop size for training)
    config = OmegaConf.merge(model_cfg, cfg.custom_kwargs)
    
    model = instantiate(cfg, sam_model=config)
    sd = torch.load(file_paths["weights"], map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(sd)
    valid_missing_keys = [k for k in missing_keys if "prompt_predictor" not in k and k != "learnable_prompts"] # ignore added parameters
    if valid_missing_keys:
        logging.error(valid_missing_keys)
        raise RuntimeError()
    if unexpected_keys:
        logging.error(unexpected_keys)
        raise RuntimeError()

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
        snapshot_download(repo_id = sam2_small_model, repo_type="model", local_dir = sam_dir)
        snapshot_download(repo_id = medsam_repo, repo_type="model", local_dir=sam_dir)
    medsam_config = {k: sam_dir / v for k, v in medsam_config.items()}
    
    return {
        "SAM2": sam2_config,
        "MedSAM": medsam_config
    }