
import os
from typing import List

import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from torch.nn import CosineSimilarity
from tqdm import tqdm



class ClipRetriever:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_features = None

    def generate_image_features(self, images: List[np.ndarray]):
        """
        Generate clip features for images under the path_images_dir
        """

        inputs = self.processor(images=images, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)

        self.image_features = image_features


    def retrieve_best_images_for_object(self, object: str, topk=10) -> List[Image.Image]:
        """
        Take in input an object (e.g. 'big red lamp', 'bed', etc.)
        Return the list of most relevant images for that object.
        """
        assert_err = "you first need to generate the image features using the self.generate_image_features function"
        assert self.image_features is not None, assert_err

        inputs = self.processor(text=[object], return_tensors="pt")

        text_features = self.model.get_text_features(**inputs)

        cos = CosineSimilarity()
        scores = cos(text_features, self.image_features)
        best_indices = torch.sort(scores, descending=True).indices
    
        return best_indices[:topk]

    
    def retrieve_best_images_for_objects(self, objects: List[str], topk=10, return_paths=True) -> List[Image.Image]:
        """
        Take in input a list of objects (every prompt is a different object e.g. ['drawer', 'big lamp'])
        Return the list of most relevant images that contain that combination of objects.
        """
        object_prompt = ' '.join(objects)

        # TODO: Here maximize the distance between the images in the feature space to improve the diversity

        return self.retrieve_best_images_for_object(object_prompt, topk=topk, return_paths=return_paths)

