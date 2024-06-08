import os
import sys
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# Append paths to sys.path for importing modules from other directories
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Import from GroundingDINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from pipeline.object_instance import ObjectInstance

# Import from segment_anything
from segment_anything import (
    sam_hq_model_registry,
    SamPredictor
)

class MasksFinder:
    def __init__(self, box_threshold=0.3, text_threshold=0.25):
        # Define paths
        self.model_config_path = "/teamspace/studios/this_studio/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.grounded_checkpoint = "/teamspace/studios/this_studio/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "outputs" 
        sam_version = "vit_h"
        sam_hq_checkpoint = "/teamspace/studios/this_studio/Grounded-Segment-Anything/sam_hq_vit_h.pth"

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        os.makedirs(self.output_dir, exist_ok=True)
        self.model = self._load_model()
        self.mask_predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(self.device))


    def get_masks_from_imgs(self, images: List[np.ndarray], objects: str) -> List[dict]:
        """
        Given a list of images extract masks around objects that fit the 'objects' prompt.

        Parameters
        ----------
        image_paths : List[str]
            list of paths to images
        objects : str
            Single string that represent a list of objects separated by a '.'.
            E.g. 'bed. drawer. radiator.'

        Returns
        -------
        List[ObjectInstance]
        For each image returns an ObjectInstance object.
        """
        if isinstance(objects, list):
            objects = self._objects_list2str(objects)

        masks = []
        bboxes = []
        confidences = []
        image_ids = []
        labels = []
        for image_idx, image in enumerate(tqdm(images, desc='Extracting the masks')):
            masks_, bboxes_, confidences_ = self.get_masks_from_img(image, objects)
            if len(masks_) == 0:
                continue

            masks += masks_
            bboxes += bboxes_
            confidences += confidences_
            image_ids += [image_idx] * len(masks_)
            labels += [objects] * len(masks_)

        return image_ids, masks, bboxes, confidences, labels


    def get_masks_from_img(self, image: np.ndarray, objects: str) -> List | None:
        """Same as for self.get_objects_masks_from_imgs but for only one image"""

        if isinstance(objects, list):
            objects = self._objects_list2str(objects)

        image_tensor, image_pil, image_cv2 = self._convert_image(image)
        size = image_pil.size
        H, W = size[1], size[0]

        boxes_filt, pred_phrases, confidence = self._get_grounding_output(self.model, image_tensor, objects, self.box_threshold, 
                                                                          self.text_threshold, device=self.device)
        if boxes_filt.shape[0 ]== 0:
            return [], [], []
        
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
            boxes_filt[i] = constrain_box(boxes_filt[i], (W, H))

        masks = self._predict_masks(image_cv2, boxes_filt)
        masks = [mask.cpu().squeeze().numpy() for mask in masks]
        bboxes = [bbox.cpu().numpy() for bbox in boxes_filt]
        
        return masks, bboxes, confidence

    @staticmethod
    def _objects_list2str(objects):
        objects_str = '. '.join(objects)
        return objects_str

        
    def _predict_masks(self, image_cv2, boxes_filt):
        self.mask_predictor.set_image(image_cv2)
        transformed_boxes = self.mask_predictor.transform.apply_boxes_torch(boxes_filt, 
                                                                            image_cv2.shape[:2]).to(self.device)

        masks, _, _ = self.mask_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )

        return masks

    
    @staticmethod
    def _convert_image(image: np.ndarray):

        image_cv2 = image  # following the naming of the original function
        image_pil = Image.fromarray(image).convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)

        return image_tensor, image_pil, image_cv2


    def _load_model(self):
        args = SLConfig.fromfile(self.model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.grounded_checkpoint, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _ = model.eval()
        return model


    def _get_grounding_output(self, model, image, caption, box_threshold, text_threshold, device=torch.device("cpu")):
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


    def _find_geometric_center(self, mask):
        """
        Calculates the geometric center (centroid) of a binary mask.

        Parameters:
        - mask (numpy.ndarray): A binary image where the object's mask is represented as 1s.

        Returns:
        - (int, int): The (x, y) coordinates of the centroid.
        """


    @staticmethod
    def _show_mask(mask, ax, random_color=False):
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    @staticmethod
    def _show_box(box, ax, label):
        x0, y0, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        ax.text(x0, y0, label)


    def show_prediction(self, objects_masks: dict):
        # take as input objects_masks, as defined in the self.get_objects_masks_from_img

        image_path = objects_masks['image_path']
        _, _, image_cv2 = self._load_image(image_path)

        masks = objects_masks['masks']
        boxes_filt = objects_masks['boxes_filt']
        pred_phrases = objects_masks['pred_phrases']

        plt.figure(figsize=(10, 10))
        plt.imshow(image_cv2)
        for mask in masks:
            self._show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            self._show_box(box.numpy(), plt.gca(), label)

        plt.axis('off')


def unrotate_masks(masks: List[np.ndarray], orientations: List[int]) -> List[np.ndarray]:
    masks_unrotated = []
    for mask, orientation in zip(masks, orientations):
        match orientation:
            case 0: # original is upright
                k = 0
                axes = (0, 1)
            case 1: # original is left -> CCW 90deg to unrotate
                k = 1
                axes = (0, 1)
            case 2: # original is upside-down -> 180deg rotation
                k = 2
                axes = (0, 1)
            case 3: # original is right -> CW 90deg to unrotate
                k = 1
                axes = (1, 0)
        mask_unrot = np.rot90(mask, k=k, axes=axes)
        masks_unrotated.append(mask_unrot)
    return masks_unrotated

def unrotate_bboxes(bboxes: List[np.ndarray], img_dims: List[Tuple[float]], orientations: List[int]):
    bboxes_unrotated = []
    for bbox, img_shape, orientation in zip(bboxes, img_dims, orientations):
        u1, v1, u2, v2 = bbox
        height, width = img_shape
        match orientation:
            case 0: # original is upright
                new_u1, new_v1 = u1, v1
                new_u2, new_v2 = u2, v2
            case 1: # original is left -> CCW 90deg to unrotate
                new_u1, new_v1 = v1, width - u1
                new_u2, new_v2 = v2, width - u2
            case 2: # original is upside-down -> 180deg rotation
                new_u1, new_v1 = width - u1, height - v1
                new_u2, new_v2 = width - u2, height - v2
            case 3: # original is right -> CW 90deg to unrotate
                new_u1, new_v1 = height - v1, u1
                new_u2, new_v2 = height - v2, u2
        bboxes_unrotated.append(np.array([new_u1, new_v1, new_u2, new_v2]))
    return bboxes_unrotated
    
def constrain_box(box, image_dims):
    # Safeguard for bbox running out of bounds for some reason
    w, h = image_dims
    tmp = torch.vstack([torch.zeros(box.shape), box])
    box = tmp.max(axis=0).values
    tmp = torch.vstack([torch.tensor([w, h, w, h]), box])
    box = tmp.min(axis=0).values
    return box
