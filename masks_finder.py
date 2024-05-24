import os
import sys
from typing import List

import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))

# Append paths to sys.path for importing modules from other directories
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Import from GroundingDINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from utils.object_instance import ObjectInstance

# Import from segment_anything
from segment_anything import (
    sam_hq_model_registry,
    SamPredictor
)

class MasksFinder:
    def __init__(self):
        # Define paths
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
        self.mask_predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(self.device))


    def get_objects_masks_from_imgs(self, image_paths: List[str], objects: str) -> List[dict]:
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

        all_instances = []
        for image_path in tqdm(image_paths):
            object_instances = self.get_objects_masks_from_img(image_path, objects)
            if object_instances is None:
                continue
            all_instances = all_instances + object_instances # Do + instead of append to keep all ObjectInstance objects within the same list for a single object type
        return all_instances


    def get_objects_masks_from_img(self, image_path: str, objects: str) -> List | None:
        """Same as for self.get_objects_masks_from_imgs but for only one image"""

        if isinstance(objects, list):
            objects = self._objects_list2str(objects)

        image, image_pil, image_cv2 = self._load_image(image_path)
        size = image_pil.size
        H, W = size[1], size[0]

        boxes_filt, pred_phrases, confidence = self._get_grounding_output(self.model, image, objects, self.box_threshold, 
                                                                          self.text_threshold, device=self.device)
        if boxes_filt.shape[0 ]== 0:
            return None
        
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        masks = self._predict_masks(image_cv2, boxes_filt)

        objects_instances_list = []
        for i, label in enumerate(pred_phrases):
            objects_instances_list.append(ObjectInstance(bbox=boxes_filt[i].numpy(),
                                                         mask=masks[i].cpu().squeeze().numpy(),
                                                         label=label,
                                                         confidence=confidence[i],
                                                         img_path=image_path
                                                         ))

        return objects_instances_list

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
    def _load_image(image_path):
        # Load PIL image
        image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)

        # Load CV2 image
        image_cv2 = cv2.imread(image_path)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        return image, image_pil, image_cv2


    def _load_model(self):
        args = SLConfig.fromfile(self.model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.grounded_checkpoint, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
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
