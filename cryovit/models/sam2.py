# """SAMv2 model for 2D/3D tomogram segmentation, using the existing library to support training and fine-tuning."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torch import Tensor
import numpy as np
from sam2.modeling.sam2_base import SAM2Base

from cryovit.models.base_model import BaseModel
from cryovit.types import BatchedTomogramData, SAMPromptMethod

sam2_model = ("facebook/sam2.1-hiera-large", {"config": "sam2.1-hiera-l.yaml", "weights": "sam2.1_hiera_large.pt"}) # the large variant of SAMv2.1
sam2_small_model = ("facebook/sam2-hiera-small") # for creating model for medical sam2
medical_sam2_model = ("jiayuanz3/MedSAM2_pretrain", {"config": "sam2-hiera-s.yaml", "weights": "MedSAM2_pretrain.pth"}) # fine-tuned on medical data SAMv2

class SAM2(BaseModel, SAM2Base):
    """SAMv2 model implementation."""

    def __init__(self, prob_use_pt_input: Tuple[float, float] = (0.5, 0), prob_use_box_input: Tuple[float, float] = (0.5, 0), prob_sample_from_gt: float = 0.1, num_slices_to_correct: Tuple[int, int] = (2, 1), rand_slices_to_correct: Tuple[bool, bool] = (True, False), num_init_cond_slices: Tuple[int, int] = (1, 1), rand_init_cond_slices: Tuple[bool, bool] = (True, False), add_corrected_slices_as_cond: bool = True, num_correction_pt_per_slice: int = 7, grid_points_per_side: Optional[int] = None, prompt_method: SAMPromptMethod = SAMPromptMethod.NONE, freeze_image_encoder: bool = False, **kwargs) -> None:
        """Initializes the CryoVIT model with specific convolutional and synthesis blocks."""
        super(SAM2, self).__init__(**kwargs)
        self.prob_use_pt_input = prob_use_pt_input
        self.prob_use_box_input = prob_use_box_input
        self.prob_sample_from_gt = prob_sample_from_gt
        self.num_slices_to_correct_for = num_slices_to_correct
        self.rand_slices_to_correct_for = rand_slices_to_correct
        self.num_init_cond_slices = num_init_cond_slices
        self.rand_init_cond_slices = rand_init_cond_slices
        self.add_corrected_slices_as_cond = add_corrected_slices_as_cond
        self.num_correction_pt_per_slice = num_correction_pt_per_slice
        self.grid_points_per_side = grid_points_per_side
        self.prompt_method = prompt_method

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def _prepare_backbone_features_per_slice(self, flat_batch: torch.FloatTensor, slice_idxs: torch.LongTensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Prepares backbone features for a specific slice."""
        if slice_idxs.numel() > 1:
            unique_slice_ids, inv_ids = torch.unique(slice_idxs, return_inverse=True)
        else:
            unique_slice_ids, inv_ids = slice_idxs, None
            
        # Compute image features for unique slices
        tomo_slices = flat_batch[unique_slice_ids]
        backbone_out = self.forward_image(tomo_slices)
        _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)
        
        # Inverse map unique slice ids to final features
        if inv_ids is not None:
            tomo_slices = tomo_slices[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]
        
        return tomo_slices, vision_feats, vision_pos_embeds, feat_sizes

    def forward(self, data: BatchedTomogramData) -> Tensor:
        """Forward pass for the SAMv2 model."""
        if self.prompt_method == SAMPromptMethod.POINT_MASK:
            # if using point/mask input, only use labeled slices as input
            batch_idxs, slice_idxs = data.labeled_slices
            data = data.subset(batch_idxs, slice_idxs)
            flat_tensor = data.batch_tensor_to_flat_tensor(data.labeled_tomos_as_batch_tensor)
        else:
            flat_tensor = data.batch_tensor_to_flat_tensor(data.tomo_batch) # [BxDxCxHxW] -> [[BxD]xCxHxW]

        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all slices before tracking
            backbone_out = self.forward_image(flat_tensor)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, data)
        out = self.forward_tracking(backbone_out, data)

        return torch.sigmoid(out)

    def prepare_prompt_inputs(self, backbone_out: Dict[str, Any], data: BatchedTomogramData, start_slice_idx: int = 0) -> Dict[str, Any]:
        """Prepare a grid-point mask for automatic prompting or an input mask from labeled data."""
        # Load the ground truth masks on all labeled slices
        # masks are then of shape [B, 1, H, W]
        slice_idxs = [data.index_to_slice_batch(idx)[0] for idx in range(data.max_slices)]
        gt_masks_per_slice = {
            slice_id: data.labels[slice_idx].unsqueeze(1) for slice_id, slice_idx in enumerate(slice_idxs)
        }
        backbone_out["gt_masks_per_slice"] = gt_masks_per_slice
        backbone_out["num_slices"] = data.max_slices
        
        # Setup prompt parameters
        if self.training:
            prob_use_pt_input = self.prob_use_pt_input[0]
            prob_use_box_input = self.prob_use_box_input[0]
            num_slices_to_correct = self.num_slices_to_correct[0]
            rand_slices_to_correct = self.rand_slices_to_correct[0]
            num_init_cond_slices = self.num_init_cond_slices[0]
            rand_init_cond_slices = self.rand_init_cond_slices[0]
        else:
            prob_use_pt_input = self.prob_use_pt_input[1]
            prob_use_box_input = self.prob_use_box_input[1]
            num_slices_to_correct = self.num_slices_to_correct[1]
            rand_slices_to_correct = self.rand_slices_to_correct[1]
            num_init_cond_slices = self.num_init_cond_slices[1]
            rand_init_cond_slices = self.rand_init_cond_slices[1]
        assert num_init_cond_slices >= 1, "Number of initial conditioning slices must be at least 1."
        if rand_init_cond_slices and num_init_cond_slices > 1:
            # Randomly select number of initial conditioning slices
            num_init_cond_slices = np.random.randint(1, num_init_cond_slices + 1)
        use_pt_input = np.random.rand() < prob_use_pt_input
        if self.prompt_method == SAMPromptMethod.POINT_MASK and rand_slices_to_correct and num_slices_to_correct > num_init_cond_slices:
            # Randomly select number of slices to correct (excluding initial conditioning slices)
            num_slices_to_correct = np.random.randint(num_init_cond_slices, num_slices_to_correct + 1)
        backbone_out["use_pt_input"] = use_pt_input
        
        # Select initial conditioning slices
        if num_init_cond_slices == 1:
            init_cond_slices = [start_slice_idx]
        else:
            init_cond_slices = [start_slice_idx] + np.random.choice(
                range(start_slice_idx + 1, data.min_slices),
                num_init_cond_slices - 1,
                replace=False
            ).tolist()
        backbone_out["init_cond_slices"] = init_cond_slices
        backbone_out["slices_not_in_init_cond"] = [n for n in range(start_slice_idx, backbone_out["num_slices"]) if n not in init_cond_slices]
        
        # Prepare mask and point inputs for each initial conditioning slice
        backbone_out["mask_inputs_per_slice"] = {}
        backbone_out["point_inputs_per_slice"] = {}     
        valid_idxs = data.index_to_slice_batch(n)[0]
        match self.prompt_method:
            case SAMPromptMethod.NONE:
                # Use an empty mask input for unprompted segmentation
                backbone_out["mask_inputs_per_slice"] = {
                    n: torch.zeros_like(data.labels[valid_idxs, n, ...].shape, dtype=torch.float32) for n in init_cond_slices
                }
            case SAMPromptMethod.POINT_MASK:
                for n in init_cond_slices:
                    if not use_pt_input:
                        backbone_out["mask_inputs_per_slice"][n] = gt_masks_per_slice[n]
                    else:
                        use_box_input = np.random.rand() < prob_use_box_input
                        if use_box_input:
                            points, labels = _generate_box_mask(gt_masks_per_slice[n])
                        else:
                            points, labels = _generate_point_mask(gt_masks_per_slice[n])
                        point_inputs = {"point_coords": points, "point_labels": labels}
                        backbone_out["point_inputs_per_slice"][n] = point_inputs
            case SAMPromptMethod.GRID:
                if self.training:
                    raise ValueError("Grid prompting is not supported during training.")
                if self.grid_points_per_side is None:
                    raise ValueError("Grid prompting requires points_per_side to be set.")
                # Create a grid-point mask for automatic prompting
                backbone_out["point_inputs_per_slice"] = {
                    n: _generate_grid_point_mask(
                        self.grid_points_per_side,
                        data.labels[valid_idxs, n, ...].shape
                    ) for n in init_cond_slices
                }
        
        # Sample slices to add correction clicks
        if not use_pt_input:
            correction_slices = []
        elif num_slices_to_correct == num_init_cond_slices:
            correction_slices = init_cond_slices
        else:
            assert num_slices_to_correct > num_init_cond_slices, "Number of slices to correct must be greater than number of initial conditioning slices."
            correction_slices = init_cond_slices + np.random.choice(
                backbone_out["slices_not_in_init_cond"],
                num_slices_to_correct - num_init_cond_slices,
                replace=False
            ).tolist()
        backbone_out["slices_to_add_correction_pt"] = correction_slices
            
        return backbone_out

    def forward_tracking(self, backbone_out: Dict[str, Any], data: BatchedTomogramData, return_dict: bool = False) -> Union[Dict[str, Any], Tensor]:
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare backbone features
            # backbone_out is [[BxD]xCxHxW]
            # vision_feats and vision_pos_embeds are [(HW), (BD), C]
            _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)
        
        # Start loop over slices
        num_slices = backbone_out["num_slices"]
        init_cond_slices = backbone_out["init_cond_slices"]
        slices_to_add_correction_pt = backbone_out["slices_to_add_correction_pt"]
        # First process initial conditioning slices, then condition on them for memory
        processing_order = init_cond_slices + backbone_out["slices_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {}
        }
        for slice_id in processing_order:
            flat_idxs = data.index_to_flat_batch(slice_id)
            # Get image features for the current frame
            if img_feats_already_computed:
                current_vision_feats = [x[:, flat_idxs] for x in vision_feats]
                current_vision_pos_embeds = [x[:, flat_idxs] for x in vision_pos_embeds]
            else:
                _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self._prepare_backbone_features_per_frame(data.batch_tensor_to_flat_tensor(data.tomo_batch), flat_idxs)

            current_out = self.track_step(
                slice_id=slice_id,
                is_init_cond_frame=slice_id in init_cond_slices,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_slice"].get(slice_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(slice_id, None),
                gt_masks=backbone_out["masks_per_frame"].get(slice_id, None),
                slices_to_add_correction_pt=slices_to_add_correction_pt,
                output_dict=output_dict,
                num_slices=num_slices,
            )
            
            # Get predictions from the current output
            preds = current_out["pred_masks"]
            
            add_output_as_cond_frame = slice_id in init_cond_slices
            if add_output_as_cond_frame:
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

    def track_step(self, slice_id: int, is_init_cond_frame: bool, current_vision_feats: Tensor, current_vision_pos_embeds: Tensor, feat_sizes: Tensor, point_inputs: Optional[Tensor], mask_inputs: Optional[Tensor], gt_masks: Optional[Tensor], slices_to_add_correction_pt: Optional[List[int]], output_dict: Dict[str, Any], num_slices: int) -> None:
        if slices_to_add_correction_pt is None:
            slices_to_add_correction_pt = []
        # Run the tracking step for the current slice
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(slice_id, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs, output_dict, num_slices, False, None)

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
        
        for slice_idx in slices_to_add_correction_pt:
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame,
                point_inputs,
                gt_masks,
                high_res_features,
                pix_feat,
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                object_score_logits,
                current_out,
            )
            _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = final_sam_outputs
            
        # Use final prediction (after correction steps)
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
    
    def _iter_correct_pt_sampling(self, is_init_cond_frame: bool, point_inputs: Optional[Tensor], gt_masks: Optional[Tensor], high_res_features: Tensor, pix_feat: Tensor, low_res_multimasks: Tensor, high_res_multimasks: Tensor, ious: Tensor, low_res_masks: Tensor, high_res_masks: Tensor, object_score_logits: Tensor, current_out: Dict[str, Any]) -> Tuple[Optional[Tensor], Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Iteratively samples correction points and updates the model's predictions."""
        assert gt_masks is not None, "Ground truth masks must be provided for correction point sampling."
        all_pred_masks = [low_res_masks]
        all_pred_high_res_masks = [high_res_masks]
        all_pred_multimasks = [low_res_multimasks]
        all_pred_high_res_multimasks = [high_res_multimasks]
        all_pred_ious = [ious]
        all_point_inputs = [point_inputs]
        all_object_score_logits = [object_score_logits]
        
        for _ in range(self.num_correction_pt_per_frame):
            # sample a new point from the error between prediction and ground-truth
            # (with a small probability, directly sample from GT masks instead of errors)
            if self.training and self.prob_sample_from_gt > 0:
                sample_from_gt = (
                    self.rng.random() < self.prob_sample_from_gt
                )
            else:
                sample_from_gt = False
            # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
            pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
            new_points, new_labels = _generate_point_mask(
                gt_masks=gt_masks,
                pred_masks=pred_for_new_pt
            )
            # add new points
            if point_inputs is None:
                point_inputs = {"point_coords": new_points, "point_labels": new_labels}
            else:
                point_inputs = {"point_coords": torch.cat([point_inputs["point_coords"], new_points], dim=1), "point_labels": torch.cat([point_inputs["point_labels"], new_labels], dim=1)}
            # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
            # For tracking, this means that when the user adds a correction click, we also feed
            # the tracking output mask logits along with the click as input to the SAM decoder.
            mask_inputs = low_res_masks
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            if self.use_act_ckpt_iterative_pt_sampling and not multimask_output:
                sam_outputs = torch.utils.checkpoint.checkpoint(
                    self._forward_sam_heads,
                    backbone_features=pix_feat,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    use_reentrant=False,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                )
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                _,
                object_score_logits,
            ) = sam_outputs
            all_pred_masks.append(low_res_masks)
            all_pred_high_res_masks.append(high_res_masks)
            all_pred_multimasks.append(low_res_multimasks)
            all_pred_high_res_multimasks.append(high_res_multimasks)
            all_pred_ious.append(ious)
            all_point_inputs.append(point_inputs)
            all_object_score_logits.append(object_score_logits)

        # Concatenate the masks along channel (to compute losses on all of them,
        # using `MultiStepIteractiveMasks`)
        current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
        current_out["multistep_pred_masks_high_res"] = torch.cat(
            all_pred_high_res_masks, dim=1
        )
        current_out["multistep_pred_multimasks"] = all_pred_multimasks
        current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
        current_out["multistep_pred_ious"] = all_pred_ious
        current_out["multistep_point_inputs"] = all_point_inputs
        current_out["multistep_object_score_logits"] = all_object_score_logits

        return point_inputs, sam_outputs

#### Model Creation and Loading ####

def create_sam_model_from_weights(cfg: BaseModel, sam_dir: Path) -> SAM2:
    configs = _download_model_weights(sam_dir)
    assert cfg.name in configs, f"Model {cfg.name} was not found in available SAMv2 models. Available models are {configs.keys()}."

    file_paths = configs[cfg.name]
    
    # Merge configs together
    model_cfg_path = file_paths["config"]
    model_cfg = OmegaConf.load(model_cfg_path)["model"]
    del model_cfg._target_ # Use cryovit SAM2 as target
    extra_cfg = cfg.copy()
    del extra_cfg.name
    config = OmegaConf.merge(model_cfg, extra_cfg)
    
    model = instantiate(config, _recursive_=True)
    sd = torch.load(file_paths["weights"], map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(sd)
    if missing_keys:
        logging.error(missing_keys)
        raise RuntimeError()
    if unexpected_keys:
        logging.error(unexpected_keys)
        raise RuntimeError()

    return model

def _download_model_weights(sam_dir: Path) -> Dict[str, Dict[str, Path]]:
    from huggingface_hub import snapshot_download
    
    # Download base SAMv2 model
    sam2_repo, sam2_config = sam2_model
    snapshot_download(repo_id = sam2_repo, repo_type="model", local_dir=sam_dir)
    sam2_config = {k: sam_dir / v for k, v in sam2_config.items()}
    
    # Download Medical-SAMv2
    medsam_repo, medsam_config = medical_sam2_model
    snapshot_download(repo_id = sam2_small_model, repo_type="model", local_dir = sam_dir)
    snapshot_download(repo_id = medsam_repo, repo_type="model", local_dir=sam_dir)
    medsam_config = {k: sam_dir / v for k, v in medsam_config.items}
    
    return {
        "sam2": sam2_config,
        "medsam": medsam_config
    }

#### Mask Generation ####

def _generate_grid_point_mask(points_per_side: int, img_size: Tuple[int, int, int]) -> Tensor:
    """Generates a grid-point mask for automatic prompting."""
    offset = 1 / (2 * points_per_side)
    grid_side = torch.linspace(offset, 1 - offset, steps=points_per_side)
    grid_x, grid_y = torch.meshgrid(grid_side, grid_side)
    float_points = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1).view(-1, 2) # [N, 2]
    points_scale = torch.Tensor(img_size)[-2:].view(1, 2) # [1, 2]
    return float_points * points_scale

def _generate_point_mask(gt_masks: torch.BoolTensor, pred_masks: Optional[torch.BoolTensor] = None, padding: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a point mask by sampling from error/gt masks. Uses the RITM sampling method with more details at https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_utils.py#L252.
    
    Args:
        gt_masks (torch.Tensor): Ground truth masks of shape [B, 1, H, W].
        pred_masks (torch.Tensor): Predicted masks of shape [B, 1, H, W].
        padding (bool): Whether to pad the masks to ensure points are within the image bounds.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the coordinates of the points ([B, N, 2] tensor) and their corresponding labels ([B, N] tensor).
    """
    import cv2
    
    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and pred_masks.dtype == torch.bool, "Masks must be boolean tensors."
    assert gt_masks.size(1) == 1 and pred_masks.shape == gt_masks.shape, "Masks must have shape [B, 1, H, W]."
    
    B, _, H, W = gt_masks.shape
    device = gt_masks.device
    
    # Calculate false positive and false negative regions for correcting points
    fp_masks = ~gt_masks & pred_masks
    fn_masks = gt_masks & ~pred_masks
    
    points = torch.zeros(B, 1, 2, device=device, dtype=torch.float)
    labels = torch.ones(B, 1, device=device, dtype=torch.int32)
    for b in range(B):
        fp_mask = fp_masks[b, 0] # [H, W]
        fn_mask = fn_masks[b, 0] # [H, W]
        if padding:
            # Pad the masks to ensure points are within the image bounds
            fp_mask = torch.nn.functional.pad(fp_mask, (1, 1, 1, 1), mode="constant", value=False)
            fn_mask = torch.nn.functional.pad(fn_mask, (1, 1, 1, 1), mode="constant", value=False)
            
        # Compute distance of points in FN/FP region to the boundary
        fp_mask_dt = cv2.distanceTransform(fp_mask.cpu().numpy().astype(np.uint8), cv2.DIST_L2, 0)
        fn_mask_dt = cv2.distanceTransform(fn_mask.cpu().numpy().astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            # Remove padding from the distance transform
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
        
        # Take the point with the largest boundary distance
        flat_fp_mask_dt = torch.from_numpy(fp_mask_dt, device=device).flatten()
        flat_fn_mask_dt = torch.from_numpy(fn_mask_dt, device=device).flatten()
        fp_argmax = torch.argmax(flat_fp_mask_dt)
        fn_argmax = torch.argmax(flat_fn_mask_dt)
        is_positive = flat_fn_mask_dt[fn_argmax] > flat_fp_mask_dt[fp_argmax] # furthest point is in FN region?
        pt_idx = fn_argmax if is_positive else fp_argmax
        points[b, 0, 0] = pt_idx % W # x coordinate
        points[b, 0, 1] = pt_idx // W # y coordinate
        labels[b, 0] = int(is_positive)
        
    

def _generate_box_mask(gt_masks: torch.BoolTensor, noise: float = 0.1, noise_bound: int = 20, top_left_label: int = 2, bottom_right_label: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a point mask by sampling from the noised version of a given 'bbox'.
    
    Args:
        gt_masks (torch.Tensor): Ground truth masks of shape [B, 1, H, W].
        noise (float): Noise level to add to the bounding box coordinates as a fraction of the box width and height.
        noise_bound (int): maximum amount of noise (in pure pixels) to add to the bounding box coordinates.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the coordinates of the points ([B, N, 2] tensor) and their corresponding labels ([B, N] tensor).
    """
    device = gt_masks.device
    
    # Get bounding boxes for ground truth masks
    B, _, H, W = gt_masks.shape
    xs = torch.arange(W, device=device, dtype=torch.int32)
    ys = torch.arange(H, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs.view(1, 1, H, W).expand(B, -1, -1, -1)  # [B, 1, H, W]
    grid_ys = grid_ys.view(1, 1, H, W).expand(B, -1, -1, -1)  # [B, 1, H, W]
    min_xs = torch.min(torch.where(gt_masks, grid_xs, torch.tensor(W, device=device)).flatten(-2), dim=-1).values
    min_ys = torch.min(torch.where(gt_masks, grid_ys, torch.tensor(H, device=device)).flatten(-2), dim=-1).values
    max_xs = torch.max(torch.where(gt_masks, grid_xs, torch.tensor(-1, device=device)).flatten(-2), dim=-1).values
    max_ys = torch.max(torch.where(gt_masks, grid_ys, torch.tensor(-1, device=device)).flatten(-2), dim=-1).values
    box_coords = torch.stack([min_xs, min_ys, max_xs, max_ys], dim=-1)  # [B, 1, 4]
    
    # Get labels
    box_labels = torch.tensor([top_left_label, bottom_right_label], device=device, dtype=torch.int32).repeat(B) # [B, 2]
    
    # Add noise to bounding box coordinates
    if noise > 0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device, dtype=torch.int32)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        noise_x = torch.min(bbox_w * noise, noise_bound)
        noise_y = torch.min(bbox_h * noise, noise_bound)
        box_noise = torch.rand_like(box_coords) * 2 - 1 # [-1, 1]
        box_noise = box_noise * torch.stack([noise_x, noise_y, noise_x, noise_y], dim=-1)

        box_coords = box_coords + box_noise
        box_coords.clamp_(min=torch.zeros(4, device=device), max=torch.tensor([W, H, W, H], device=device) - 1)
    
    box_coords = box_coords.reshape(-1, 2, 2) # [B, 1, 4] -> [B, 2, 2]
    box_labels = box_labels.reshape(-1, 2)  # [B, 2]
    return box_coords.to(dtype=torch.float), box_labels