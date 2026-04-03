import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Any, Tuple

from sam3.model.sam3_video_base import Sam3VideoBase
from sam3.model.data_misc import FindStage


class SAM3VLFS(nn.Module):
    def __init__(
        self, 
        sam3_model: Sam3VideoBase, 
        resolution: int = 1008,
    ):
        super().__init__()
        self.sam3 = sam3_model
        self.resolution = resolution
        self.image_size = self.sam3.tracker.image_size
    
    @property
    def device(self):
        return next(self.sam3.parameters()).device
    
    def forward(
        self, 
        samples: torch.Tensor, 
        prompt_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        SAM3-VLFS forward
        Args:
            samples: [B, T, C, H, W]
            prompt_dict:
                'shots': int, support set
                'text': str, class description
        Returns:
            Dict containing 'pred_masks': [B*T, H, W]
        """
        B, T, C, H, W = samples.shape
        n_shots = prompt_dict.get('shots', 1)
        
        all_pred_masks = []
        all_tracker_masks = []
        all_det_masks = []
        
        for b in range(B):
            # get text prompt
            if b in prompt_dict and isinstance(prompt_dict[b], dict) and 'text' in prompt_dict[b]:
                text_prompt = prompt_dict[b]['text']
            else:
                text_prompt = prompt_dict.get('text', "object")
            
            batch_samples = samples[b]  # [T, C, H, W]
            
            # backbone feature
            backbone_output, text_outputs = self.sam3.detector.backbone.forward(
                samples=batch_samples,
                captions=[text_prompt],
            )
            
            # tracker feature
            backbone_output = self._build_tracker_features(backbone_output)
            
            inference_state = self._init_inference_state_with_features(
                images=batch_samples,
                text_prompt=text_prompt,
                text_outputs=text_outputs,
                backbone_output=backbone_output,
            )

            for t in range(T):
                is_support = t < n_shots
                if is_support:
                    pred_mask = self._process_support_frame(
                        inference_state=inference_state,
                        frame_idx=t,
                        batch_idx=b,
                        prompt_dict=prompt_dict,
                        num_frames=T,
                    )
                    tracker_mask = torch.zeros((1, 1, H, W), device=samples.device, dtype=samples.dtype)
                    det_mask = torch.zeros((1, 1, H, W), device=samples.device, dtype=samples.dtype)
                else:
                    pred_mask, tracker_mask, det_mask = self._process_query_frame(
                        inference_state=inference_state,
                        frame_idx=t,
                        num_frames=T,
                    )

                final_mask = F.interpolate(
                    pred_mask,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                all_pred_masks.append(final_mask)
                all_tracker_masks.append(tracker_mask)
                all_det_masks.append(det_mask)

        outputs = {
            "pred_masks": torch.cat(all_pred_masks, dim=0).squeeze(1),
            "tracker_masks": torch.cat(all_tracker_masks, dim=0).squeeze(1),
            "det_masks": torch.cat(all_det_masks, dim=0).squeeze(1),
        }

        return outputs

    def _build_tracker_features(self, backbone_output: Dict[str, Any]) -> Dict[str, Any]:
        """Create SAM3 tracker feature from backbone"""
        sam3_features = backbone_output["backbone_fpn"]

        sam_mask_decoder = self.sam3.tracker.sam_mask_decoder
        tracker_fpn = [
            sam_mask_decoder.conv_s0(sam3_features[0]),
            sam_mask_decoder.conv_s1(sam3_features[1]),
            sam3_features[2],
        ]
        
        backbone_output["tracker_fpn"] = tracker_fpn
        return backbone_output
    
    def _init_inference_state_with_features(
        self,
        images: Tensor,
        text_prompt: str,
        text_outputs: Dict[str, Any],
        backbone_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Init inference state"""
        T = images.shape[0]
        H, W = images.shape[2:]
        
        feature_cache = {}
        for t in range(T):
            frame_backbone_fpn = [
                fpn[t:t+1] for fpn in backbone_output["backbone_fpn"]
            ]
            frame_tracker_fpn = [
                fpn[t:t+1] for fpn in backbone_output["tracker_fpn"]
            ]
            
            vision_pos_enc = backbone_output["vision_pos_enc"]
            if isinstance(vision_pos_enc, list):
                frame_vision_pos_enc = [pos[t:t+1] for pos in vision_pos_enc]
            else:
                frame_vision_pos_enc = vision_pos_enc[t:t+1]
            
            tracker_backbone_out = {
                "backbone_fpn": frame_tracker_fpn,
                "vision_pos_enc": frame_vision_pos_enc,
            }
            
            frame_backbone_out = {
                "backbone_fpn": frame_backbone_fpn,
                "vision_pos_enc": frame_vision_pos_enc,
                "vision_features": frame_backbone_fpn[-1],
            }
            
            feature_cache[t] = (
                images[t:t+1],
                {
                    "backbone_out": frame_backbone_out,
                    "tracker_backbone_out": tracker_backbone_out,
                }
            )
        
        return {
            "images": images,
            "num_frames": T,
            "image_size": self.image_size,
            "orig_height": H,
            "orig_width": W,
            "text_prompt": text_prompt,
            "text_outputs": text_outputs,
            "output_dict": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            "feature_cache": feature_cache,
        }

    def _get_tracker_features(
        self,
        inference_state: Dict,
        frame_idx: int
    ) -> Tuple[List[Tensor], List[Tensor], List[Tuple[int, int]]]:
        """Get tracker features"""
        assert frame_idx in inference_state["feature_cache"], \
            f"Frame {frame_idx} features not cached."

        _, backbone_cache = inference_state["feature_cache"][frame_idx]
        tracker_backbone_out = backbone_cache["tracker_backbone_out"]

        backbone_fpn = tracker_backbone_out["backbone_fpn"]
        vision_pos_enc = tracker_backbone_out["vision_pos_enc"]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in backbone_fpn]

        vision_feats = [
            x.flatten(2).permute(2, 0, 1) for x in backbone_fpn
        ]

        if isinstance(vision_pos_enc, list):
            vision_pos_embeds = [
                pos.flatten(2).permute(2, 0, 1) for pos in vision_pos_enc
            ]
        else:
            vision_pos_embeds = [
                vision_pos_enc.flatten(2).permute(2, 0, 1)
            ]

        return vision_feats, vision_pos_embeds, feat_sizes

    def _process_support_frame(
        self,
        inference_state: Dict,
        frame_idx: int,
        batch_idx: int,
        prompt_dict: Dict,
        num_frames: int,
    ) -> Tensor:
        """Support img process: encoder for memory"""
        images = inference_state["images"]
        output_dict = inference_state["output_dict"]

        img = images[frame_idx:frame_idx+1]

        gt_mask = None
        if batch_idx in prompt_dict and frame_idx in prompt_dict[batch_idx]:
            gt_mask = prompt_dict[batch_idx][frame_idx].get('prompt')

        if gt_mask is None:
            gt_mask = torch.zeros(1, 1, img.shape[-2], img.shape[-1], device=self.device)

        gt_mask = gt_mask.to(self.device).float()
        if gt_mask.dim() == 2:
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
        elif gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)

        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self._get_tracker_features(inference_state, frame_idx)
        )

        mask_for_tracker = F.interpolate(
            gt_mask,
            size=(self.sam3.tracker.image_size, self.sam3.tracker.image_size),
            mode="bilinear",
            align_corners=False,
        )

        current_out = self.sam3.tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=img,
            point_inputs=None,
            mask_inputs=mask_for_tracker,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=False,
            run_mem_encoder=True,
        )

        output_dict["cond_frame_outputs"][frame_idx] = current_out

        return current_out['pred_masks']

    def _process_query_frame(
        self,
        inference_state: Dict,
        frame_idx: int,
        num_frames: int,
    ) -> Tuple[Tensor, ...]:
        """Query inference: Tracker(vision branch) + Detector(text Branch)"""
        images = inference_state["images"]
        output_dict = inference_state["output_dict"]
        text_outputs = inference_state["text_outputs"]
        feature_cache = inference_state["feature_cache"]

        img = images[frame_idx:frame_idx+1]
        _, backbone_cache = feature_cache[frame_idx]
        backbone_out = backbone_cache["backbone_out"]

        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self._get_tracker_features(inference_state, frame_idx)
        )

        # Visual Branch
        current_out = self.sam3.tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=img,
            point_inputs=None,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=False,
            run_mem_encoder=True,
        )

        output_dict["non_cond_frame_outputs"][frame_idx] = current_out

        tracker_mask = current_out.get("pred_masks", None)

        tracker_mask = F.interpolate(
            tracker_mask,
            size=(self.sam3.tracker.image_size, self.sam3.tracker.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Text Branch
        find_input = FindStage(
            img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        geometric_prompt = self.sam3.detector._get_dummy_prompt()
        full_backbone_out = {**backbone_out, **text_outputs}

        det_out = self.sam3.detector.forward_grounding(
            backbone_out=full_backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )

        pred_logits = det_out["pred_logits"]
        pred_masks = det_out["pred_masks"]

        pred_probs = pred_logits.sigmoid()
        if "presence_logit_dec" in det_out:
            presence_probs = det_out["presence_logit_dec"].sigmoid()
            pred_probs = pred_probs * presence_probs

        score_threshold = 0.6
        pred_probs_squeezed = pred_probs.squeeze(-1).squeeze(0)
        high_conf_mask = pred_probs_squeezed > score_threshold

        if high_conf_mask.any():
            high_conf_indices = high_conf_mask.nonzero(as_tuple=True)[0]
            det_masks = pred_masks[0, high_conf_indices]
            det_mask = det_masks.max(dim=0, keepdim=True)[0].unsqueeze(0)
        else:
            best_idx = pred_probs_squeezed.argmax()
            det_mask = pred_masks[0:1, best_idx:best_idx+1]

        det_mask = F.interpolate(
            det_mask,
            size=(self.sam3.tracker.image_size, self.sam3.tracker.image_size),
            mode="bilinear",
            align_corners=False,
        )

        if tracker_mask.dim() == 3:
            tracker_mask = tracker_mask.unsqueeze(0)

        pred_mask = torch.max(tracker_mask, det_mask)

        return pred_mask, tracker_mask, det_mask
