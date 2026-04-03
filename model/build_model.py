import os
import torch
import torch.nn as nn
import pkg_resources
from iopath.common.file_io import g_pathmgr
from typing import Optional, List
from sam3.model.maskformer_segmentation import UniversalSegmentationHead
from sam3.model.model_misc import (
    DotProductScoring,
    MLP,
)
from sam3.model.sam3_image import Sam3Image
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT
from sam3.model.sam3_video_base import Sam3VideoBase
from sam3.model.sam3_tracker_base import Sam3TrackerBase
from sam3.train.matcher import BinaryHungarianMatcherV2

from sam3.model_builder import (
        _create_vit_neck,
        _create_position_encoding,
        _create_sam3_transformer,
        _create_segmentation_head,
        _create_geometry_encoder,
        download_ckpt_from_hf,
        _create_tracker_maskmem_backbone,
        _create_tracker_transformer,
    )


def _create_text_encoder(bpe_path: str, adapt_dim: float = 1, adaptformer_stages: Optional[List[int]] = None,) -> VETextEncoder:
    """Create SAM3 text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
        adapt_dim=adapt_dim,
        adaptformer_stages=adaptformer_stages,
    )


def _create_vit_backbone(compile_mode=None, adapt_dim: float = 1, adaptformer_stages: Optional[List[int]] = None,):
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
        adapt_dim=adapt_dim,
        adaptformer_stages=adaptformer_stages
    )

def _create_vision_backbone(
    compile_mode=None, enable_inst_interactivity=True,
    adapt_dim: float = 1,
    adaptformer_stages: Optional[List[int]] = None,
) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone(compile_mode=compile_mode, adapt_dim=adapt_dim, adaptformer_stages=adaptformer_stages)
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    # Visual neck
    return vit_neck

def build_train_tracker(
        with_backbone: bool = False,
        checkpoint_path: Optional[str] = None,
        apply_temporal_disambiguation: bool = False
) -> Sam3TrackerBase:

    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()

    backbone = None
    if with_backbone:
        vision_backbone = _create_vision_backbone()
        backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)

    model = Sam3TrackerBase(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        # SAM parameters
        multimask_output_in_sam=True,
        # Evaluation
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        # Multimask
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        # Additional settings
        # Mask overlap
        non_overlap_masks_for_mem_enc=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        # SAM decoder settings
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        use_memory_selection=apply_temporal_disambiguation,
    )

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    return model


def build_sam3_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF=False,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    # adaptformer
    vision_adaptformer_stages: Optional[List[int]] = None,
    vision_adapt_dim: float = 1,
    text_adaptformer_stages: Optional[List[int]] = None,
    text_adapt_dim: float = 1,
) -> Sam3VideoBase:
    """
    Build SAM3 Few shot model.

    Args:
        checkpoint_path: Optional path to checkpoint file
        bpe_path: Path to the BPE tokenizer file

    Returns:
        Sam3VideoInferenceWithInstanceInteractivity: The instantiated dense tracking model
    """
    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )

    # Build Tracker module
    tracker = build_train_tracker(apply_temporal_disambiguation=apply_temporal_disambiguation)

    # Build Detector components
    visual_neck = _create_vision_backbone(adapt_dim=vision_adapt_dim, adaptformer_stages=vision_adaptformer_stages)
    text_encoder = _create_text_encoder(bpe_path, adapt_dim=text_adapt_dim, adaptformer_stages=text_adaptformer_stages)

    from model.vision_text_encoder import VisionTextEncoder
    backbone = VisionTextEncoder(scalp=1, visual=visual_neck, text=text_encoder)

    # backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer(has_presence_token=has_presence_token)
    segmentation_head: UniversalSegmentationHead = _create_segmentation_head()
    input_geometry_encoder = _create_geometry_encoder()

    # Create main dot product scoring
    main_dot_prod_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    main_dot_prod_scoring = DotProductScoring(
        d_model=256, d_proj=256, prompt_mlp=main_dot_prod_mlp
    )

    # Create matcher for training
    matcher = BinaryHungarianMatcherV2(
        focal=True,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        alpha=0.25,
        gamma=2,
        stable=False,
    )

    # Build Detector module
    detector = Sam3Image(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=has_presence_token,
        matcher=matcher,
    )

    # Build the main SAM3 video model
    if apply_temporal_disambiguation:
        model = Sam3VideoBase(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=15,
            hotstart_unmatch_thresh=8,
            hotstart_dup_thresh=8,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=16,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
        )
    else:
        # a version without any heuristics for ablation studies
        model = Sam3VideoBase(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=0,
            hotstart_unmatch_thresh=0,
            hotstart_dup_thresh=0,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=0,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
        )

    # Load checkpoint if provided
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    if checkpoint_path is not None:
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        missing_keys, unexpected_keys = model.load_state_dict(
            ckpt, strict=strict_state_dict_loading
        )
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    model.to(device=device)
    return model


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"checkpoint not found: {checkpoint_path}")
        return model, None

    print(f"load checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # add sam3 prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[f"sam3.{k}"] = v
    state_dict = new_state_dict

    incompatible_keys = model.load_state_dict(state_dict, strict=False)

    if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys:
        print(f"Successfully loaded {checkpoint_path}")
    else:
        print(f"Loaded {checkpoint_path} with {len(incompatible_keys.missing_keys)} missing "
              f"and {len(incompatible_keys.unexpected_keys)} unexpected keys.")

    return model, incompatible_keys

from model.model import SAM3VLFS

def build_sam3_vlfs_model(device: str = 'cuda', channel_factor: float = 0.25, checkpoint_path: str = './weights/sam3.pt') -> SAM3VLFS:

    text_adaptformer_stages = list(range(0, 24, 2))
    vision_adaptformer_stages = list(range(0, 32, 2))

    sam3_model = build_sam3_model(
        device=device,
        load_from_HF=False,
        apply_temporal_disambiguation=False,
        vision_adaptformer_stages=vision_adaptformer_stages,
        vision_adapt_dim=channel_factor,
        text_adaptformer_stages=text_adaptformer_stages,
        text_adapt_dim=channel_factor,
    )
    
    model = SAM3VLFS(sam3_model)
    
    # load checkpoint
    model, incompatible_keys = load_checkpoint(model, checkpoint_path)
    
    # freeze sam3
    if incompatible_keys and incompatible_keys.missing_keys:
        missing_keys_set = set(incompatible_keys.missing_keys)
        for name, p in model.named_parameters():
            p.requires_grad = (name in missing_keys_set)
    else:
        print("freeze everything")
        for name, p in model.named_parameters():
            p.requires_grad = False

    trainable_keywords = ["tracker.mask_downsample", "tracker.transformer", "tracker.maskmem_backbone"]

    for name, p in model.named_parameters():
        if any(key in name for key in trainable_keywords):
            p.requires_grad = True

    return model
