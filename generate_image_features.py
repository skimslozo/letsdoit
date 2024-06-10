import yaml
import torch
from tqdm import tqdm

from dataloader.dataloader import DataLoader
from pipeline.clip_retriever import ClipRetriever
from dataloader.data_parser import rotate_pose


path_config = '/teamspace/studios/this_studio/letsdoit/config/config.yml'
batch_size = 5


def generate_image_features():
    # generate image features for all the dataset and save them

    with open(path_config, 'r') as file:
        cfg = yaml.safe_load(file)

    loader = DataLoader(cfg['path_dataset'], cfg['data_split'])
    retriever = ClipRetriever()

    visit_ids = loader.visit_ids
    for visit_id in visit_ids:
        print(f'Generating image features for visit_id: {visit_id}')
        image_paths, intrinsics, poses, orientations = loader.get_images(visit_id,
                                                                         asset_type=cfg['data_asset_type'], 
                                                                         sample_freq=cfg['loader_sample_freq'])

        path_image_features = loader.get_image_features_path(visit_id)
        image_features = None

        pbar = tqdm(total=len(image_paths))

        for i1, i2 in zip(range(0, len(image_paths), batch_size), range(batch_size, len(image_paths)+batch_size, batch_size)):
            batch_image_paths = image_paths[i1:i2]
            batch_orientations = orientations[i1:i2]


            images_rotated = [rotate_pose(loader.parser.read_rgb_frame(img_path), orientation) 
                              for img_path, orientation in 
                              zip(batch_image_paths, batch_orientations)]
        
        
            retriever.generate_image_features(images_rotated)
            del(images_rotated)
            
            if image_features is None:
                image_features = retriever.image_features
            else:
                image_features = torch.cat((image_features, retriever.image_features), 0)
            pbar.update(batch_size)

        pbar.close()
        torch.save(image_features, path_image_features)




if __name__ == '__main__':
    generate_image_features()