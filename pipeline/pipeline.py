from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import List
import warnings

import numpy as np
import open3d as o3d
import scipy
from torch.nn import CosineSimilarity
from transformers import CLIPProcessor, CLIPModel

from pipeline.masks_matcher import MasksMatcher
from pipeline.clip_retriever import ClipRetriever
from pipeline.masks_finder import MasksFinder, unrotate_masks, unrotate_bboxes
from pipeline.masks_merger import MasksMerger
from pipeline.object_instance import ObjectInstance, initialize_object_instances, plot_instances_3d, generate_masks_features, filter_instances
from pipeline.object_3d import Object3D, plot_objects_3d, denoise_objects_3d, filter_objects_3d
from dataloader.dataloader import DataLoader
from utils.misc import select_ids, number2str

from scoring.primitive import SpatialPrimitive, SpatialPrimitivePair, get_primitive
from scoring.spatial_graph import GraphNode, retrieve_best_action_object


class Pipeline:

    def __init__(self,
                 path_dataset,
                 path_instructions,
                 data_asset_type='wide',
                 data_split='train',
                 loader_sample_freq=1,
                 retriever_topk=10,
                 retriever_max_similarity_thresh=0.97,
                 finder_box_threshold=0.3,
                 finder_text_threshold=0.25,
                 merger_dist_thresh=0.1, 
                 merger_geo_similarity_thresh=0.7, 
                 merger_feat_similarity_thresh=0.7,
                 merger_n_points=1000,
                 path_submission_folder=None):

        self.path_dataset = path_dataset

        # avoid warnings in the initialization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.loader = DataLoader(self.path_dataset, split=data_split)
            print('Initializing ClipRetriver...')
            self.retriever = ClipRetriever(topk=retriever_topk, 
                                           max_similarity_thresh=retriever_max_similarity_thresh)
            print('Initializing the MasksFinder...')
            self.masks_finder = MasksFinder(box_threshold=finder_box_threshold, 
                                            text_threshold=finder_text_threshold)
            print('Initializing MasksMerger...')
            self.masks_merger = MasksMerger(dist_thresh=merger_dist_thresh,
                                            geo_similarity_thresh=merger_geo_similarity_thresh,
                                            feat_similarity_thresh=merger_feat_similarity_thresh,
                                            n_points=merger_n_points)
            print('Initializing ClipModel...')
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            print('Initializing ClipProcessor...')
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print('Done!')

        self.data_asset_type = data_asset_type
        self.path_instructions = path_instructions
        self.instruction_list = self._load_instruction_list()
        self.loader_sample_freq = loader_sample_freq
        self.path_submission_folder = path_submission_folder


    def run(self):
        action_objects = []
        for instruction_block in self.instruction_list:
            # TODO: save the masks as file somewhere
            action_objects.append(self._run_instruction_block(instruction_block))

        return action_objects


    def _run_instruction_block(self, instruction_block: dict):
        visit_id = str(instruction_block['visit_id'])

        self.pcd = self.loader.parser.get_highres_reconstruction(visit_id, self.loader.get_video_ids(visit_id)[0])

        # Get both original and upright-rotated images and depths as outputs
        dict_data = self._load_data(visit_id)
        self.retriever.generate_image_features(dict_data['images_rotated'])

        object_labels = instruction_block['all_objects']
        instructions = instruction_block['instructions']

        objects = []
        for object_label in object_labels:
            print(f"Extracting objects for '{object_label}'")
            objects.extend(self._get_objects_3d(object_label, dict_data))
        
        action_objects = []
        for instruction in instructions:
            action_object = retrieve_best_action_object(instruction, objects)

            if self.path_submission_folder is not None:
                self._save_instruction_result(visit_id, instruction, [action_object])
            
            action_objects.append(action_object)

        return action_objects

    
    def _save_instruction_result(self, visit_id: str, instruction: dict, action_objects: List[Object3D]):
        """
        Save the masks of the action_objects according to the opensun3d_track2 submission format
        """
        instruction_id = instruction["desc_id"]
        file_name = visit_id + '_'+ instruction_id

        action_masks = [action_object.pcd_mask.astype(np.uint8) for action_object in action_objects]
        confidences = [1 for _ in action_objects]  # TODO: decide how to assign the mask confidence

        dir_predicted_masks = 'predicted_masks'  # following submission convention
        dir_predicted_masks_abs = os.path.join(self.path_submission_folder, dir_predicted_masks)

        # create folders if they are not there yet
        if not os.path.isdir(self.path_submission_folder):
            os.mkdir(self.path_submission_folder)
        if not os.path.isdir(dir_predicted_masks_abs):
            os.mkdir(dir_predicted_masks_abs)

        path_predicted_masks = []

        # first create file with list of all the masks
        with open(os.path.join(self.path_submission_folder, file_name + '.txt'), "w") as f:

            for i, confidence in enumerate(confidences):
                path_predicted_mask = os.path.join(dir_predicted_masks, f"{file_name}_{number2str(i)}.txt")
                f.write(f"{path_predicted_mask} {confidence}\n")
                path_predicted_masks.append(path_predicted_mask)

        # save the masks as txt files
        for mask, path_predicted_mask in zip(action_masks, path_predicted_masks):
            with open(os.path.join(self.path_submission_folder, path_predicted_mask), 'w') as f:
                for val in mask:
                    f.write(f"{str(val)}\n")
    

    def _load_data(self, visit_id: str) -> dict:

        images, images_rotated, image_paths, intrinsics, poses, orientations = self.loader.get_images(visit_id, 
                                                                                                      asset_type=self.data_asset_type, 
                                                                                                      sample_freq=self.loader_sample_freq)
        depths, depths_rotated, depth_paths, _, _, _ = self.loader.get_depths(visit_id, 
                                                                              asset_type=self.data_asset_type, 
                                                                              sample_freq=self.loader_sample_freq)
                                                                    
        dict_data = {'images': images,
                     'images_rotated': images_rotated,
                     'image_paths': image_paths,
                     'intrinsics': intrinsics,
                     'poses': poses,
                     'orientations': orientations,
                     'depths': depths,
                     'depths_rotated': depths_rotated,
                     'depth_paths': depth_paths}
        
        return dict_data


    def _load_instruction_list(self) -> List[dict]:
        # load instructions from json file
        with open(self.path_instructions, 'r') as file:
            instruction_list = json.load(file)
        return instruction_list


    def _get_objects_3d(self, object_label: str, dict_data: dict) -> List[Object3D]:
        object_instances = self._get_object_instances(object_label, dict_data)
        objects_3d = self.masks_merger(object_instances, self.pcd)
        denoise_objects_3d(objects_3d)
        return objects_3d

    # For an object, get a list of corresponding ObjectInstances
    def _get_object_instances(self, object_label: str, dict_data: dict) -> List[ObjectInstance]:
        best_indices = self.retriever.retrieve_best_images_for_object(object_label)
        best_images = select_ids(dict_data['images'], best_indices)
        best_images_rotated = select_ids(dict_data['images_rotated'], best_indices)
        best_image_paths = select_ids(dict_data['image_paths'], best_indices)
        best_intrinsics = select_ids(dict_data['intrinsics'], best_indices)
        best_poses = select_ids(dict_data['poses'], best_indices)
        best_orientations = select_ids(dict_data['orientations'], best_indices)
        best_depths = select_ids(dict_data['depths'], best_indices)
        best_depth_paths = select_ids(dict_data['depth_paths'], best_indices)

        # Masks we get here as outputs are for the upright-rotated images
        image_ids, masks, bboxes, confidences, labels = self.masks_finder.get_masks_from_imgs(best_images_rotated, object_label)

        # Rotate masks and bboxes back to the rotation of the original image
        mask_image_sizes = [best_images_rotated[idx].shape[:-1] for idx in image_ids]
        mask_image_orientations = select_ids(best_orientations, image_ids)
        masks_unrotated = unrotate_masks(masks=masks, orientations=dict_data['orientations'])
        bboxes_unrotated = unrotate_bboxes(bboxes=bboxes, img_dims=mask_image_sizes, orientations=mask_image_orientations)
        image_features = generate_masks_features(self.clip_processor, self.clip_model, select_ids(best_images_rotated, image_ids), bboxes, masks)

        dict_object_instances = {'images': select_ids(best_images, image_ids),
                                 'image_names': [Path(img_path).name.replace('.png', '') for img_path in select_ids(best_image_paths, image_ids)],
                                 'depths': select_ids(best_depths, image_ids),
                                 'bboxes': bboxes_unrotated,
                                 'masks': masks_unrotated,
                                 'labels': labels,
                                 'confidences': confidences,
                                 'intrinsics': select_ids(best_intrinsics, image_ids),
                                 'extrinsics': select_ids(best_poses, image_ids),
                                 'orientations': mask_image_orientations,
                                 'image_features': image_features,
        }

        object_instances = initialize_object_instances(**dict_object_instances)

        return object_instances
            

