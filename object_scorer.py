import os
import cv2
import sys
import torch
import itertools
from PIL import Image
from pathlib import Path
import numpy as np


# Append paths to sys.path for importing modules from other directories
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Import from GroundingDINO
from masks_finder import MasksFinder
from clip_retriever import ClipRetriever
from letsdoit.utils.misc import SpatialPrimitive
from letsdoit.utils.object_instance import ObjectInstance
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Import from segment_anything
from segment_anything import (
    sam_hq_model_registry,
    SamPredictor
)

class ObjectScorer:
    def __init__(self):
        # Define paths
        self.image_path = "/teamspace/studios/this_studio/Grounded-Segment-Anything/42445132_5423.937.png"
        self.model_config_path = "/teamspace/studios/this_studio/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.grounded_checkpoint = "/teamspace/studios/this_studio/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "outputs" 
        sam_version = "vit_h"
        sam_hq_checkpoint = "/teamspace/studios/this_studio/Grounded-Segment-Anything/sam_hq_vit_h.pth"

        self.box_threshold = 0.3
        self.text_threshold = 0.25

        os.makedirs(self.output_dir, exist_ok=True)
        self.model = self._load_model()
        self.predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(self.device))

    def get_object_masks(self, image_path, objects):
        self._load_image(image_path)
        size = self.image_pil.size
        H, W = size[1], size[0]

        boxes_filt, pred_phrases, confidence = self._get_grounding_output(self.model, self.image, objects, self.box_threshold, self.text_threshold, device=self.device)
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, self.image_cv2.shape[:2]).to(self.device)

        masks, _, _ = self.predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )

        return pred_phrases, confidence, masks, boxes_filt

    def _load_image(self, image_path):
        # Load PIL image
        self.image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.image, _ = transform(self.image_pil, None)

        # Load CV2 image
        image_cv2 = cv2.imread(image_path)
        self.image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.image_cv2)        

    def _load_model(self):
        args = SLConfig.fromfile(self.model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.grounded_checkpoint, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    def _get_grounding_output(self, model, image, caption, box_threshold, text_threshold, device="cpu"):
        caption = caption.lower().strip() + "." if not caption.endswith(".") else caption
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        confidence = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase)
            confidence.append(logit.max().item())
        return boxes_filt, pred_phrases, confidence

def evaluate_instruction(instruction: dict, mf: MasksFinder, retriever: ClipRetriever, data_dir: Path):
    for idx, primitive in enumerate(instruction['spatial_primitives']):
        spatial_prim = get_primitive(primitive['primitive'])
        scores = get_primitive_score(mf, retriever, primitive['target_object'], primitive['reference_object'], spatial_prim, data_dir)
        #TODO: handle scores across all spatial primitives for a description, pick the target object instance corresponding to the highest overall score
    return scores

def get_primitive_score(mf: MasksFinder, retriever: ClipRetriever, target_obj: str, ref_obj: str, primitive: SpatialPrimitive, data_dir: Path):
    # Get the objects of interest that you need to look for from the query

    look_list = [target_obj, ref_obj]
    retriever.generate_image_features(data_dir / 'color_rotated') # TODO: move this outside of the function afterwards
    
    # Get the best image matches for the target and reference objects 
    best_imgs = retriever.retrieve_best_images_for_objects(look_list)

    objects_masks = mf.get_objects_masks_from_imgs(best_imgs, look_list)
    pairs_scores = []

    for mask in objects_masks:
        if mask is None:
            print(f"Skipping image {img_path.name} because it does not contain any mask info")
            continue
        # Find all instances of target and reference objects
        #TODO: account for the case where you might have a mask with both ref and target in it
        target_idxs = [i for i, obj in enumerate(mask['pred_phrases']) if target_obj in obj]
        ref_idxs = [i for i, obj in enumerate(mask['pred_phrases']) if ref_obj in obj]
        img_path = Path(mask['image_path'])

        #TODO: how to handle situation where some or none of the objects are visible in a single image? For now - just considering images where both visible
        if len(target_idxs) == 0 or len(ref_idxs) == 0:
            print(f"Skipping image {img_path.name} because it does not contain either target and reference objects")
            continue

        # Create all possible pairs of target-reference objects
        pairs = list(itertools.product(target_idxs, ref_idxs))

        # Evaluate each pair
        for ti, ri in pairs:
            target = ObjectInstance(bbox=mask['boxes_filt'][ti].numpy(), 
                                    mask=mask['masks'][ti].cpu().squeeze().numpy(), 
                                    label=mask['pred_phrases'][ti], 
                                    confidence=mask['confidence'][ti],
                                    img_path=mask['image_path'],
                                    )
            reference = ObjectInstance(bbox=mask['boxes_filt'][ri].numpy(), 
                                    mask=mask['masks'][ri].cpu().squeeze().numpy(), 
                                    label=mask['pred_phrases'][ri], 
                                    confidence=mask['confidence'][ri],
                                    img_path=mask['image_path'],
                                    )
            
            score = get_spatial_primitive_score(primitive, target, reference) #TODO: extend with other spatial primitives, link to queries
            pairs_scores.append({
                'target': target_obj,
                'reference': ref_obj,
                'primitive': primitive,
                'score': score
            })

    return pairs_scores

def get_spatial_primitive_score(primitive: SpatialPrimitive, target: ObjectInstance, reference: ObjectInstance):
    res = None
    match primitive:
        case SpatialPrimitive.ABOVE:
            pass
        case SpatialPrimitive.BELOW:
            pass
        case SpatialPrimitive.FRONT:
            pass
        case SpatialPrimitive.BEHIND:
            pass
        case SpatialPrimitive.RIGHT:
            pass
        case SpatialPrimitive.LEFT:
            pass
        case SpatialPrimitive.CONTAINS:
            pass
        case SpatialPrimitive.NEXT_TO:
            res = evaluate_next_to(target, reference)
        case SpatialPrimitive.BETWEEN:
            pass

    return res

def evaluate_next_to(target: ObjectInstance, reference: ObjectInstance):
    # Calculate the 3D center coordinates for both target and reference objects
    target_center_3d = target.get_mask_center_3d()
    reference_center_3d = reference.get_mask_center_3d()

    # Calculate the Euclidean distance between the 3D centers of the target and reference objects
    distance = np.linalg.norm(np.array(target_center_3d) - np.array(reference_center_3d))

    # Final score = - (object confidence x distance)
    score = - target.confidence * reference.confidence * distance

    # Return the negative calculated distance as the score for 'next to' spatial relation (higher score -> better match)
    return -score
