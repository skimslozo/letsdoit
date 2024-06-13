import os
from typing import List, Union

import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from torch.nn import CosineSimilarity
from tqdm import tqdm


class ClipRetriever:
    def __init__(self, 
                 topk=10, 
                 max_similarity_thresh=0.97,
                 image_features: Union[torch.Tensor, str]=None):
                 
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.topk = topk
        self.max_similarity_thresh = max_similarity_thresh
        self.image_features = None

    def generate_image_features(self, images: List[np.ndarray]):
        """
        Generate clip features for images under the path_images_dir
        """

        inputs = self.processor(images=images, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)

        self.image_features = image_features
            

    def update_image_features(self, image_features: Union[torch.Tensor, str]):
        # check the image_features type
        err_msg = "image_features needs to be a torch.Tensor or the path to a stored torch.Tensor!"
        assert isinstance(image_features, torch.Tensor) or isinstance(image_features, str), err_msg

        if isinstance(image_features, str):
            # load the tensor from path
            assert os.path.isfile(image_features), "image_features must be the path to an existing file!"
            self.image_features = torch.load(image_features)
        else:
            # assign the tensor directly
            self.image_features = image_features


    def retrieve_best_images_for_object(self, object: str) -> List[Image.Image]:
        """
        Take in input an object (e.g. 'big red lamp', 'bed', etc.)
        Return the list of most relevant images for that object.
        Output only indices of images that have at most a cosine similarity of max_similarity_tresh. Use 1 to 
        Use max_similarity_thresh to make sure that the output images are different enough from each others. Use 1 to 
        undo this feature.
        """
        assert_err = "you first need to generate the image features using the self.generate_image_features function"
        assert self.image_features is not None, assert_err
        assert 0 < self.max_similarity_thresh <= 1, 'max_similarity_thresh needs to be > 0 and <= 1'

        inputs = self.processor(text=[object], return_tensors="pt")

        text_features = self.model.get_text_features(**inputs)

        cos = CosineSimilarity()
        scores = cos(text_features, self.image_features)
        best_indices_all = torch.sort(scores, descending=True).indices.cpu().numpy()

        best_indices_selected = []
        n_selected = 0

        if self.max_similarity_thresh >= 1:
            best_indices_selected = best_indices_all[:self.topk]

        else:
            # output topk indices of images that have a similarity score of at most max_similarity_thresh with each others
            for idx1 in best_indices_all:
                if len(best_indices_selected) == self.topk:
                    break
                max_similarity = 0
                if len(best_indices_selected) > 0:
                    similarities = cos(self.image_features[[idx1]], self.image_features[best_indices_selected])
                    max_similarity = similarities.max()
                        
                if max_similarity < self.max_similarity_thresh:
                    best_indices_selected.append(idx1)
                
    
        return best_indices_selected

    
    def retrieve_best_images_for_objects(self, objects: List[str]) -> List[Image.Image]:
        """
        Take in input a list of objects (every prompt is a different object e.g. ['drawer', 'big lamp'])
        Return the list of most relevant images that contain that combination of objects.
        """
        object_prompt = ' '.join(objects)
        return self.retrieve_best_images_for_object(object_prompt)

    def __del__(self):
        del self.model 
        del self.processor
        del self.max_similarity_thresh
        del self.image_features
        del self
