# """SAMv2 model for 2D/3D tomogram segmentation, using the existing library to support training and fine-tuning."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
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

# ####### Training specific params #######
# # box/point input and corrections
# prob_to_use_pt_input_for_train: 0.5
# prob_to_use_pt_input_for_eval: 0.0
# prob_to_use_box_input_for_train: 0.5  # 0.5*0.5 = 0.25 prob to use box instead of points
# prob_to_use_box_input_for_eval: 0.0
# prob_to_sample_from_gt_for_train: 0.1  # with a small prob, sampling correction points from GT mask instead of prediction errors
# num_frames_to_correct_for_train: 2  # iteratively sample on random 1~2 frames (always include the first frame)
# num_frames_to_correct_for_eval: 1  # only iteratively sample on first frame
# rand_frames_to_correct_for_train: True  # random #init-cond-frame ~ 2
# add_all_frames_to_correct_as_cond: True  # when a frame receives a correction click, it becomes a conditioning frame (even if it's not initially a conditioning frame)
# # maximum 2 initial conditioning frames
# num_init_cond_frames_for_train: 2
# rand_init_cond_frames_for_train: True  # random 1~2
# num_correction_pt_per_frame: 7
# use_act_ckpt_iterative_pt_sampling: false

class SAM2(BaseModel, SAM2Base):
    """SAMv2 model implementation."""

    def __init__(self, prob_use_pt_input: Tuple[float, float] = (0.5, 0), prob_use_box_input: Tuple[float, float] = (0.5, 0), num_slices_to_correct: Tuple[int, int] = (2, 1), rand_slices_to_correct: Tuple[bool, bool] = (True, False), num_init_cond_slices: Tuple[int, int] = (1, 1), rand_init_cond_slices: Tuple[bool, bool] = (True, False), add_corrected_slices_as_cond: bool = True, num_corrections_pt_per_slice: int = 7, grid_points_per_side: Optional[int] = None, prompt_method: SAMPromptMethod = SAMPromptMethod.NONE, freeze_image_encoder: bool = False, **kwargs) -> None:
        """Initializes the CryoVIT model with specific convolutional and synthesis blocks."""
        super(SAM2, self).__init__(**kwargs)
        self.prob_use_pt_input = prob_use_pt_input
        self.prob_use_box_input = prob_use_box_input
        self.num_slices_to_correct_for = num_slices_to_correct
        self.rand_slices_to_correct_for = rand_slices_to_correct
        self.num_init_cond_slices = num_init_cond_slices
        self.rand_init_cond_slices = rand_init_cond_slices
        self.add_corrected_slices_as_cond = add_corrected_slices_as_cond
        self.num_corrections_pt_per_slice = num_corrections_pt_per_slice
        self.grid_points_per_side = grid_points_per_side
        self.prompt_method = prompt_method

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def forward(self, data: BatchedTomogramData) -> Tensor:
        """Forward pass for the SAMv2 model."""
        flat_tensor = data.batch_tensor_to_flat_tensor(data.tomo_batch) # [BxDxCxHxW] -> [[BxD]xCxHxW]
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all slices before tracking
            # Note: data["input"] is B, D, H, W, may need to reformat (and check label shape)
            
            backbone_out = self.forward_image(flat_tensor)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, data)
        previous_stages_out = self.forward_tracking(backbone_out, data)

        return previous_stages_out

    def prepare_prompt_inputs(self, backbone_out: Dict[str, Any], data: BatchedTomogramData, start_slice_idx: int = 0) -> Dict[str, Any]:
        """Prepare a grid-point mask for automatic prompting or an input mask from labeled data."""
        # Load the ground truth masks on all labeled slices
        # masks are then of shape [B, 1, H, W]
        slice_idxs = [data.index_to_slice_batch(idx) for idx in range(data.max_slices)]
        gt_masks_per_slice = {
            slice_id: data.labels[slice_idx].unsqueeze(1) for slice_id, slice_idx in enumerate(slice_idxs) if data.labels[slice_idx].max() > 0 # i.e., mask has labels
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
        if self.prompt_method == SAMPromptMethod.POINT and rand_slices_to_correct and num_slices_to_correct > num_init_cond_slices:
            # Randomly select number of slices to correct (excluding initial conditioning slices)
            num_slices_to_correct = np.random.randint(num_init_cond_slices, num_slices_to_correct + 1)
        backbone_out["use_pt_input"] = use_pt_input
        
        # Select initial conditioning slices
        labeled_slices = list(gt_masks_per_slice.keys())
        # If using point input, select initial conditioning slices from labeled slices
        start_idx = start_slice_idx if not use_pt_input else labeled_slices[0]
        idx_choices = range(start_slice_idx + 1, data.min_slices)
        if num_init_cond_slices == 1 or len(labeled_slices) == 1:
            init_cond_slices = [start_idx]
        else:
            init_cond_slices = [start_slice_idx] + np.random.choice(
                range(start_slice_idx + 1, data.min_slices),
                num_init_cond_slices - 1,
                replace=False
            ).tolist()
        backbone_out["init_cond_slices"] = init_cond_slices
        backbone_out["slices_not_in_init_cond"] = [n for n in range(start_slice_idx, backbone_out["num_slices"]) if n not in init_cond_slices]
        
        # Prepare either empty mask input for unprompted segmentation or grid-point mask input for automatic prompting
        for n in init_cond_slices:
            valid_idxs = data.index_to_slice_batch(n)[0]
            match self.prompt_method:
                case SAMPromptMethod.NONE:
                    # Use an empty mask input for unprompted segmentation
                    backbone_out["mask_inputs_per_slice"] = {
                        n: torch.zeros_like(data.labels[valid_idxs, n, ...].shape, dtype=torch.float32)
                    }
                case SAMPromptMethod.POINT:
                    pass
                case SAMPromptMethod.AUTO:
                    if self.grid_points_per_side is None:
                        raise ValueError("Automatic prompting requires points_per_side to be set.")
                    # Create a grid-point mask for automatic prompting
                    backbone_out["mask_inputs_per_slice"] = {
                        n: self._generate_grid_point_mask(
                            self.grid_points_per_side,
                            data.labels[valid_idxs, n, ...].shape
                        )
                    }
        
        # Sample the initial conditioning frame (a random frame with a mask)
        backbone_out["init_cond_slices"] = [np.random.choice(list(gt_masks_per_slice.keys()))]
        backbone_out["slices_not_in_init_cond"] = [t for t in range(backbone_out["num_slices"]) if t not in backbone_out["init_cond_slices"]]
        # Prepare masks for initial conditioning frame
        backbone_out["mask_inputs_per_frame"] = {}
        for t in backbone_out["init_cond_slices"]:
            backbone_out["mask_inputs_per_frame"] = gt_masks_per_slice[t]
            
        return backbone_out

    def forward_tracking(self, backbone_out: Dict[str, Any], data: BatchedTomogramData) -> Dict[str, Any]:
        """Forward video tracking on each frame."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare backbone features
            _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)
        
        # Start loop over slices
        num_slices = backbone_out["num_slices"]
        init_cond_slices = backbone_out["init_cond_slices"]
        # First process initial conditioning slices, then condition on them for memory
        processing_order = init_cond_slices + backbone_out["slices_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {}
        }
        for frame_id in processing_order:
            # Get image features for the current frame
            if img_feats_already_computed:
                current_vision_feats = vision_feats[frame_id]
                current_vision_pos_embeds = vision_pos_embeds[frame_id]
            else:
                _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self._prepare_backbone_features_per_frame(data["input"], frame_id)
                
            current_out = self.track_step(
                frame_id=frame_id,
                is_init_cond_frame=frame_id in init_cond_slices,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=None,
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(frame_id, None),
                gt_masks=backbone_out["masks_per_frame"].get(frame_id, None),
                output_dict=output_dict,
                num_slices=num_slices,
            )
            
            add_output_as_cond_frame = frame_id in init_cond_slices
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][frame_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][frame_id] = current_out
                
        return output_dict
    
    def track_step(self, frame_id: int, is_init_cond_frame: bool, current_vision_feats: Tensor, current_vision_pos_embeds: Tensor, feat_sizes: Tensor, point_inputs: Optional[Tensor], mask_inputs: Optional[Tensor], gt_masks: Optional[Tensor], output_dict: Dict[str, Any], num_slices: int) -> None:
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(frame_id, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, None, mask_inputs, output_dict, num_slices, False, None)
        
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
    
    def _generate_grid_point_mask(self, points_per_side: int, img_size: Tuple[int, int, int]) -> Tensor:
        """Generates a grid-point mask for automatic prompting."""
        offset = 1 / (2 * points_per_side)
        grid_side = torch.linspace(offset, 1 - offset, steps=points_per_side)
        grid_x, grid_y = torch.meshgrid(grid_side, grid_side)
        points = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1).view(-1, 2) # []
        
# select n_random init_cond_frames to initialize SAMv2 memory -> define a processing order (cond_frames first, then the rest of the video frames)
# iterate through processing order
    # for each image, retrieve features if computed, otherwise compute features with image_encoder
    # track step -> add masks back to memory
    
# select random conditioning frames from unlabelled frames
# try using 0-tensors for prompts?, or, from https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/automatic_mask_generator.py#L36, generating a grid of point prompts
    # use this for inference
    # for training, use the provided labels
# also need a method to select the mask that most closely matches a target -> for loss calculation and evaluation

# set the label for the frame used for initial conditioning to -1 (i.e., ignore when calculating loss and metrics)


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