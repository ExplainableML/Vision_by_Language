import os

import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal

import PIL
import PIL.Image
import torch
from torch.utils.data import Dataset


#####################################
### CIRR/CIRCO/FASHIONIQ-relevant Dataloaders.

class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield :a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions'] when
             split in ['train', 'val']
            - ['reference_image', 'reference_name', 'relative_captions'] when split == test
    """

    def __init__(self, dataset_path: Union[Path, str], split: Literal['train', 'val', 'test'], dress_types: List[str],
                 mode: Literal['relative', 'classic'], preprocess: callable, no_duplicates: Optional[bool] = False,blip_transform: callable = None):
        """
        :param dataset_path: path to the FashionIQ dataset
        :param split: dataset split, should be in ['train, 'val', 'test']
        :param dress_types: list of fashionIQ categories, each category should be in ['dress', 'shirt', 'toptee']
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
            - In 'relative' mode the dataset yield dict with keys:
                - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions']
                 when split in ['train', 'val']
                - ['reference_image', 'reference_name', 'relative_captions'] when split == test
        :param preprocess: function which preprocesses the image
        :param no_duplicates: if True, the dataset will not yield duplicate images in relative mode, does not affect classic mode
        """
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.mode = mode
        self.dress_types = dress_types
        self.split = split
        self.no_duplicates = no_duplicates
        self.blip_transform = blip_transform

        # Validate the inputs
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(dataset_path / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # Remove duplicats from
        if self.no_duplicates:
            seen = set()
            new_triplets = []
            for triplet in self.triplets:
                if triplet['candidate'] not in seen:
                    seen.add(triplet['candidate'])
                    new_triplets.append(triplet)
            self.triplets = new_triplets

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(dataset_path / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index) -> dict:
        try:
            if self.mode == 'relative':
                relative_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']

                if self.split in ['train', 'val']:
                    reference_image_path = self.dataset_path / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.triplets[index]['target']
                    target_image_path = self.dataset_path / 'images' / f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    blip_ref_img = self.blip_transform(PIL.Image.open(reference_image_path).convert('RGB'))
                    blip_target_img = self.blip_transform(PIL.Image.open(target_image_path).convert('RGB'))

                    return {
                        'reference_image': reference_image,
                        'blip_ref_img': blip_ref_img,
                        'blip_target_img': blip_target_img,
                        'reference_name': reference_name,
                        'target_image': target_image,
                        'target_name': target_name,
                        'relative_captions': relative_captions
                    }

                elif self.split == 'test':
                    reference_image_path = self.dataset_path / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    blip_ref_img = self.blip_transform(PIL.Image.open(reference_image_path).convert('RGB'))

                    return {
                        'reference_image': reference_image,
                        'blip_ref_img': blip_ref_img,
                        'reference_name': reference_name,
                        'relative_captions': relative_captions
                    }

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = self.dataset_path / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                blip_img = self.blip_transform(PIL.Image.open(image_path).convert('RGB'))
                return {
                    'image': image,
                    'blip_img': blip_img,
                    'image_name': image_name
                }

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRRDataset(Dataset):
    """
   CIRR dataset class for PyTorch dataloader.
   The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_caption', 'group_members']
             when split in ['train', 'val']
            - ['reference_image', 'reference_name' 'relative_caption', 'group_members', 'pair_id'] when split == test
    """

    def __init__(self, dataset_path: Union[Path, str], split: Literal['train', 'val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable, no_duplicates: Optional[bool] = False,blip_transform: callable = None):
        """
        :param dataset_path: path to the CIRR dataset
        :param split: dataset split, should be in ['train', 'val', 'test']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
                - In 'relative' mode the dataset yield dict with keys:
                    - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_caption',
                    'group_members'] when split in ['train', 'val']
                    - ['reference_image', 'reference_name' 'relative_caption', 'group_members', 'pair_id'] when split == test
        :param preprocess: function which preprocesses the image
        :param no_duplicates: if True, the dataset will not yield duplicate images in relative mode, does not affect classic mode
        """
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.no_duplicates = no_duplicates
        self.blip_transform = blip_transform

        if split == "test":
            split = "test1"
            self.split = "test1"

        # Validate inputs
        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(dataset_path / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # Remove duplicates from triplets
        if self.no_duplicates:
            seen = set()
            new_triplets = []
            for triplet in self.triplets:
                if triplet['reference'] not in seen:
                    seen.add(triplet['reference'])
                    new_triplets.append(triplet)
            self.triplets = new_triplets

        # get a mapping from image name to relative path
        with open(dataset_path / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index) -> dict:
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                relative_caption = self.triplets[index]['caption']

                if self.split in ['train', 'val']:
                    reference_image_path = self.dataset_path / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = self.dataset_path / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    blip_ref_img = self.blip_transform(PIL.Image.open(reference_image_path).convert('RGB'))
                    return {
                        'reference_image': reference_image,
                        'blip_ref_img': blip_ref_img,
                        'reference_name': reference_name,
                        'target_image': target_image,
                        'target_name': target_hard_name,
                        'relative_caption': relative_caption,
                        'group_members': group_members
                    }

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    reference_image_path = self.dataset_path / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    blip_ref_img = self.blip_transform(PIL.Image.open(reference_image_path).convert('RGB'))
                    return {
                        'reference_image': reference_image,
                        'blip_ref_img': blip_ref_img,
                        'reference_name': reference_name,
                        'relative_caption': relative_caption,
                        'group_members': group_members,
                        'pair_id': pair_id
                    }

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = self.dataset_path / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)

                return {
                    'image': image,
                    'image_name': image_name
                }

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRCODataset(Dataset):
    """
    CIRCO dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions', 'shared_concept',
             'gt_img_ids', 'query_id'] when split == 'val'
            - ['reference_image', 'reference_name', 'relative_captions', 'shared_concept', 'query_id'] when split == test
    """

    def __init__(self, dataset_path: Union[str, Path], split: Literal['val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable, blip_transform: callable = None):
        """
        Args:
            dataset_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        dataset_path = Path(dataset_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = dataset_path
        self.blip_transform = blip_transform

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val', 'test_gt']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(dataset_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [dataset_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(dataset_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id]
            if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(PIL.Image.open(reference_img_path))
            blip_ref_img = self.blip_transform(PIL.Image.open(reference_img_path).convert('RGB'))

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(PIL.Image.open(target_img_path))

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_image': reference_img,
                    'reference_name': reference_img_id,
                    'blip_ref_img': blip_ref_img,
                    'target_image': target_img,
                    'target_name': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,

                }

            elif self.split == 'test':
                return {
                    'reference_image': reference_img,
                    'reference_name': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                    'blip_ref_img': blip_ref_img,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = self.preprocess(PIL.Image.open(img_path))
            return {
                'image': img,
                'image_name': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


#####################################
### GENECIS-relevant Dataloaders.

class COCODataset(Dataset):

    def __init__(self, transform=None, blip_transform = None,root_dir='/shared-local/skarthik63/coco/val2017/') -> None:
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.blip_transform = blip_transform
    
    def load_blip_sample(self, sample):
    
        val_img_id = sample['val_image_id']
        fpath = os.path.join(self.root_dir, f'{val_img_id:012d}.jpg')
        orig_img = PIL.Image.open(fpath)
        img = self.blip_transform(orig_img)
        return img

    def load_sample(self, sample):
        val_img_id = sample['val_image_id']
        fpath = os.path.join(self.root_dir, f'{val_img_id:012d}.jpg')
        orig_img = PIL.Image.open(fpath)
        
        if self.transform is not None:
            img = self.transform(orig_img)

        return img


class COCOValSubset(COCODataset):

    def __init__(self, val_split_path, tokenizer=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(val_split_path) as f:
            val_samples = json.load(f)

        self.val_samples = val_samples
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        
        """
        Follow same return signature as CIRRSubset
        """

        sample = self.val_samples[index]
        orig_reference = sample['reference']

        target = sample['target']
        gallery = sample['gallery']
        caption = sample['condition']
        reference = self.load_sample(orig_reference)
        target = self.load_sample(target)
        blip_ref_img = self.load_blip_sample(orig_reference)
        gallery = [self.load_sample(i) for i in gallery]

        if self.transform is not None:
            gallery = torch.stack(gallery)
            gallery_and_target = torch.cat([target.unsqueeze(0), gallery])
        else:
            gallery_and_target = [target] + gallery

        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        # By construction, target_rank = 0
        return reference, caption, blip_ref_img, gallery_and_target,0  

    def __len__(self):
        return len(self.val_samples)

DILATION = 0.7
PAD_CROP = True

def expand2square(pil_img, background_color=(0, 0, 0)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = PIL.Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = PIL.Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class VAWDataset(Dataset):

    def __init__(self, transform=None, image_dir='/shared-local/skarthik63/Visual_Genome/VG_All/',blip_transform=None) -> None:
        super().__init__()

        self.image_dir = image_dir
        self.transform = transform
        self.blip_transform = blip_transform
        self.dilate = DILATION
        self.pad_crop = PAD_CROP

    def load_cropped_image(self, img):

        image_id = img['image_id']
        bbox = img['instance_bbox']
        
        # Get image
        path = os.path.join(self.image_dir, f'{image_id}.jpg')
        im = PIL.Image.open(path).convert('RGB')
        im_width, im_height = im.size

        width = bbox[2]     # Width of bounding box
        height = bbox[3]    # Height of bounding box


        if self.dilate:
            orig_left, orig_top = bbox[0], bbox[1]
            left, top = max(0, orig_left - self.dilate * width), max(0, orig_top - self.dilate * height)
            right, bottom = min(im_width, left + (1 + self.dilate) * width), min(im_height, top + (1 + self.dilate) * height)
        else:
            left, top = bbox[0], bbox[1]
            right, bottom = bbox[0] + width, bbox[1] + height

        im = im.crop((left, top, right, bottom))
        
        if self.pad_crop:
            if im.mode == 'L':
                bg_color = (0,)
            else:
                bg_color = (0, 0, 0)
            im = expand2square(im, bg_color)

        return im 
    def load_sample(self, sample):
        im = self.load_cropped_image(sample)
        im = self.transform(im)
        return im

    def load_blip_sample(self, sample):
        im = self.load_cropped_image(sample)
        im = self.blip_transform(im)
        return im


class VAWValSubset(VAWDataset):

    def __init__(self, val_split_path, tokenizer=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(val_split_path) as f:
            val_samples = json.load(f)

        self.val_samples = val_samples
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        
        """
        Follow same return signature as CIRRSubset
            (Except for returning reference object at the end)
        """

        sample = self.val_samples[index]
        orig_reference = sample['reference']

        target = sample['target']
        gallery = sample['gallery']
        caption = sample['condition']
        reference = self.load_sample(orig_reference)
        target = self.load_sample(target)
        blip_ref_img = self.load_blip_sample(orig_reference)
        gallery = [self.load_sample(i) for i in gallery]

        if self.transform is not None:
            gallery = torch.stack(gallery)
            gallery_and_target = torch.cat([target.unsqueeze(0), gallery])
        else:
            gallery_and_target = [target] + gallery

        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        # By construction, target_rank = 0
        return reference, caption, blip_ref_img,gallery_and_target, 0

    def __len__(self):
        return len(self.val_samples)