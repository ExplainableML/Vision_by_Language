import json
import os
from typing import List, Dict, Union

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import tqdm


@torch.no_grad()
def fiq(
    device: torch.device,
    predicted_features: torch.Tensor,
    target_names: List,
    index_features: torch.Tensor,
    index_names: List,
    split: str='val',
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the Fashion-IQ validation set fiven the dataset, pseudo tokens and reference names.
    Computes Recall@10 and Recall@50.
    """
    # Move the features to the device
    index_features = torch.nn.functional.normalize(index_features).to(device)
    predicted_features = torch.nn.functional.normalize(predicted_features).to(device)

    # Compute the distances
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Check if the target names are in the top 10 and top 50
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    output_metrics = {
        'Recall@1': (torch.sum(labels[:, :1]) / len(labels)).item() * 100,
        'Recall@5': (torch.sum(labels[:, :5]) / len(labels)).item() * 100,
        'Recall@10': (torch.sum(labels[:, :10]) / len(labels)).item() * 100,
        'Recall@50': (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    }
    return output_metrics
    

@torch.no_grad()
def cirr(
    device: torch.device, 
    predicted_features: torch.Tensor, 
    reference_names: List, 
    targets: Union[np.ndarray,List], 
    target_names: List, 
    index_features: torch.Tensor, 
    index_names: List, 
    query_ids: Union[np.ndarray,List],
    preload_dict: Dict[str, Union[str, None]],
    split: str='val',    
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRR validation set given the dataset, pseudo tokens and the reference names.
    Computes Recall@1, 5, 10 and 50. If given a test set, will generate submittable file.
    """   
    # Put on device.
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    if distances.ndim == 3:
        # If there are multiple features per instance, we average.
        distances = distances.mean(dim=1)
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    resize = len(sorted_index_names) if split == 'test' else len(target_names)
    reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(resize, -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)
    
    # Compute the subset predictions and ground-truth labels
    targets = np.array(targets)
    group_mask = (sorted_index_names[..., None] == targets[:, None, :]).sum(-1).astype(bool)

    if split == 'test':
        sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)
        pairid_to_retrieved_images, pairid_to_group_retrieved_images = {}, {}
        for pair_id, prediction in zip(query_ids, sorted_index_names):
            pairid_to_retrieved_images[str(int(pair_id))] = prediction[:50].tolist()
        for pair_id, prediction in zip(query_ids, sorted_group_names):
            pairid_to_group_retrieved_images[str(int(pair_id))] = prediction[:3].tolist()            

        submission = {'version': 'rc2', 'metric': 'recall'}
        group_submission = {'version': 'rc2', 'metric': 'recall_subset'}

        submission.update(pairid_to_retrieved_images)
        group_submission.update(pairid_to_group_retrieved_images)

        submissions_folder_path = os.path.join(os.getcwd(), 'data', 'test_submissions', 'cirr')
        os.makedirs(submissions_folder_path, exist_ok=True)

        with open(os.path.join(submissions_folder_path, preload_dict['test']), 'w') as file:
            json.dump(submission, file, sort_keys=True)
        with open(os.path.join(submissions_folder_path, f"subset_{preload_dict['test']}"), 'w') as file:
            json.dump(group_submission, file, sort_keys=True)                        
        return None
            
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))    
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    output_metrics = {f'recall@{key}': (torch.sum(labels[:, :key]) / len(labels)).item() * 100 for key in [1, 5, 10, 50]}
    output_metrics.update({f'group_recall@{key}': (torch.sum(group_labels[:, :key]) / len(group_labels)).item() * 100 for key in [1, 2, 3]})

    return output_metrics


@torch.no_grad()
def circo(
    device: torch.device, 
    predicted_features: torch.Tensor, 
    targets: Union[np.ndarray,List], 
    target_names: List, 
    index_features: torch.Tensor, 
    index_names: List,
    query_ids: Union[np.ndarray,List],
    preload_dict: Dict[str, Union[str, None]],
    split: str='val',
    **kwargs
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRCO validation set given the pseudo tokens and the reference names.
    Computes mAP@5, 10, 25 and 50. If test-split, generates submittable file.
    """
    # Load the model
    # Put on device.
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)
    
    ### Compute Test Submission in case of test split.
    if split == 'test':
        print('Generating test submission file!')
        similarity = predicted_features @ index_features.T
        if similarity.ndim == 3:
            # If there are multiple features per instance, we average.
            similarity = similarity.mean(dim=1)                    
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        # Return prediction dict to submit.
        queryid_to_retrieved_images = {
            query_id: query_sorted_names[:50].tolist() for (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)            
        }
        
        submissions_folder_path = os.path.join(os.getcwd(), 'data', 'test_submissions', 'circo')
        os.makedirs(submissions_folder_path, exist_ok=True)
        with open(os.path.join(submissions_folder_path, preload_dict['test']), 'w') as file:
            json.dump(queryid_to_retrieved_images, file, sort_keys=True)        
        return None
    
    ### Directly compute metrics when using validation split.
    retrievals = [5, 10, 25, 50]
    recalls = {key: [] for key in retrievals}
    maps = {key: [] for key in retrievals}
        
    for predicted_feature, target_name, sub_targets in tqdm.tqdm(zip(predicted_features, target_names, targets), total=len(predicted_features), desc='Computing Metric.'):
        sub_targets = np.array(sub_targets)[np.array(sub_targets) != '']  # remove trailing empty strings added for collate_fn
        similarity = predicted_feature @ index_features.T
        if similarity.ndim == 2:
            # If there are multiple features per instance, we average.
            similarity = similarity.mean(dim=0)
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        map_labels = torch.tensor(np.isin(sorted_index_names, sub_targets), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

        for key in retrievals:
            maps[key].append(float(torch.sum(precisions[:key]) / min(len(sub_targets), key)))

        assert target_name == sub_targets[0], f"Target name not in GTs {target_name} {sub_targets}"
        single_gt_labels = torch.tensor(sorted_index_names == target_name)
        
        for key in retrievals:
            recalls[key].append(float(torch.sum(single_gt_labels[:key])))

    output_metrics = {f'mAP@{key}': np.mean(item) * 100 for key, item in maps.items()}
    output_metrics.update({f'recall@{key}': np.mean(item) * 100 for key, item in recalls.items()})
    return output_metrics


@torch.no_grad()
def genecis(
    device: torch.device, 
    predicted_features: torch.Tensor, 
    index_features: torch.Tensor, 
    index_ranks: List,
    topk: List[int] = [1, 2, 3],
    **kwargs    
) -> Dict[str, float]:
    
    predicted_features = torch.nn.functional.normalize(predicted_features.float(), dim=-1).to(device)
    index_features = torch.nn.functional.normalize(index_features.float(), dim=-1).to(device)
    
    # Compute similarity
    if predicted_features.ndim == 3:
        similarities = predicted_features.bmm(index_features.permute(0,2,1)).mean(dim=1)
    else:
        similarities = (predicted_features[:, None, :] * index_features).sum(dim=-1)

    # # Sort the similarities in ascending order (closest example is the predicted sample)
    _, sort_idxs = similarities.sort(dim=-1, descending=True)                   # B x N

    # Compute recall at K
    if isinstance(index_ranks, list):
        index_ranks = torch.stack(index_ranks)
    index_ranks = index_ranks.to(device)
    
    output_metrics = {f'R@{k}': get_recall(sort_idxs[:, :k], index_ranks) * 100 for k in topk}

    return output_metrics


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
def get_recall(indices, targets): #recall --> wether next item in session is within top K recommended items or not
    """
    Code adapted from: https://github.com/hungthanhpham94/GRU4REC-pytorch/blob/master/lib/metric.py
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B) or (BxN): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """

    if len(targets.size()) == 1:
        # One hot label branch
        targets = targets.view(-1, 1).expand_as(indices)
        hits = (targets == indices).nonzero()
        if len(hits) == 0: return 0
        n_hits = (targets == indices).nonzero()[:, :-1].size(0)
        recall = float(n_hits) / targets.size(0)
        return recall
    else:        
        # Multi hot label branch
        recall = []
        for preds, gt in zip(indices, targets):            
            max_val = torch.max(torch.cat([preds, gt])).int().item()
            preds_binary = torch.zeros((max_val + 1,), device=preds.device, dtype=torch.float32).scatter_(0, preds, 1)
            gt_binary = torch.zeros((max_val + 1,), device=gt.device, dtype=torch.float32).scatter_(0, gt.long(), 1)
            success = (preds_binary * gt_binary).sum() > 0
            recall.append(int(success))        
        return torch.Tensor(recall).float().mean()
    



















# ########## TO UPDATE, CURRENTLY BROKEN ############
# @torch.no_grad()
# def fiq_generate_val_predictions(clip_model: clip.model.CLIP, blip_model:callable,query_dataset: torch.utils.data.Dataset, dress_type: str, preload_dict: Dict[str, Union[str,None]]) -> Tuple[torch.Tensor, List[str]]:
#     """
#     Generates features predictions for the validation set of Fashion IQ.
#     """    

#     # Create data loader
#     relative_val_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_size=32, num_workers=10,
#                                      pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)
    
#     predicted_features_list = []
#     target_names_list = []
    
#     # Compute features
#     for batch in tqdm.tqdm(relative_val_loader):
#         reference_names = batch['reference_name']
#         target_names = batch['target_name']
#         relative_captions = batch['relative_captions']

#         flattened_captions = np.array(relative_captions).T.flatten().tolist()
#         input_captions = [
#             f"{flattened_captions[i].strip('.?, ')} and {flattened_captions[i + 1].strip('.?, ')}" for
#             i in range(0, len(flattened_captions), 2)]
#         orig_input_captions = input_captions
#         input_captions = [f"a photo of $ that {in_cap}" for in_cap in input_captions]
        

#         blip_image = batch['blip_ref_img'].to(device)
#         captions = []
#         for i in range(blip_image.size(0)):
#             img = blip_image[i].unsqueeze(0)
#             caption = blip_model.generate({'image': img, "prompt": prompts.fiq_blip2_prompt(dress_type)})
#             captions.append(caption[0])

#         modified_captions = []
#         for i in range(len(captions)):
#             instruction = orig_input_captions[i]
#             img_caption = captions[i]
#             final_prompt = prompts.fiq_fashion_prompt(dress_type) + '\n' + "Image Content: " + img_caption
#             final_prompt = final_prompt + '\n' + 'Instruction: '+ instruction
#             resp = openai_api.openai_completion(final_prompt)

#             resp = resp.split('\n')
#             print(resp)
#             ## extract edited description
#             description = ""
#             for line in resp:
                    
#                 if line.startswith('Edited Description:'):
#                     description = line.split(':')[1].strip()
#                     modified_captions.append(description)
#                     break
#             if description == "":
#                 modified_captions.append(orig_input_captions[i]) 
            
#         predicted_features = torch.nn.functional.normalize(
#             clip_model.encode_text(clip.tokenize(modified_captions,truncate=True).to(device)))
#         #predicted_features = torch.nn.functional.normalize((torch.nn.functional.normalize(text_features) + torch.nn.functional.normalize(text_features_reversed)) / 2)
#         # predicted_features = torch.nn.functional.normalize((text_features + text_features_reversed) / 2)

#         predicted_features_list.append(predicted_features)
#         target_names_list.extend(target_names)

#     predicted_features = torch.vstack(predicted_features_list)
#     return predicted_features, target_names_list

# @torch.no_grad()
# def fiq_compute_val_metrics(query_dataset: torch.utils.data.Dataset, clip_model: clip.model.CLIP, blip_model:callable, index_features: torch.Tensor,
#                             index_names: List[str], dress_type:str, preload_dict: Dict[str, Union[str,None]]) \
#         -> Dict[str, float]:
#     """
#     Compute the retrieval metrics on the FashionIQ validation set given the dataset, pseudo tokens and the reference names
#     """

#     # Generate the predicted features
#     predicted_features, target_names = fiq_generate_val_predictions(
#         clip_model, blip_model,query_dataset, dress_type, preload_dict=preload_dict)

#     # Move the features to the device
#     index_features = index_features.to(device)
#     predicted_features = predicted_features.to(device)

#     # Normalize the features
#     index_features = torch.nn.functional.normalize(index_features.float())

#     # Compute the distances
#     distances = 1 - predicted_features @ index_features.T
#     sorted_indices = torch.argsort(distances, dim=-1).cpu()
#     sorted_index_names = np.array(index_names)[sorted_indices]

#     # Check if the target names are in the top 10 and top 50
#     labels = torch.tensor(
#         sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
#     assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

#     # Compute the metrics
#     recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
#     recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
#     recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
#     recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100


#     return {'fiq_recall_at1': recall_at1,
#             'fiq_recall_at5': recall_at5,
#             'fiq_recall_at10': recall_at10,
#             'fiq_recall_at50': recall_at50}

# @torch.no_grad()
# def fiq(args: argparse.Namespace, dataset_path: str, dress_type: str, clip_model_name: str, blip_model:callable,
#                        preprocess: callable,blip_transform:callable, preload_dict: Dict[str, Union[str,None]]) -> Dict[str, float]:
#     """
#     Compute the retrieval metrics on the FashionIQ validation set given the pseudo tokens and the reference names
#     """
#     # Load the model
#     clip_model, _ = clip.load(clip_model_name, device=utils.device, jit=False)
#     clip_model = clip_model.float().eval().requires_grad_(False)

#     # Extract the index features

#     index_features, index_names = utils.extract_image_features(args, classic_val_dataset, clip_model, preload=preload_dict['img_features'])
#     #index_features, index_names = extract_image_captions(classic_val_dataset, blip_model)

#     # Define the relative dataset
#     query_dataset = datasets.FashionIQDataset(dataset_path, 'val', [dress_type], 'relative', preprocess,blip_transform=blip_transform)

#     return fiq_compute_val_metrics(query_dataset, clip_model, blip_model,index_features, index_names,
#                                    dress_type,preload_dict=preload_dict)
# ################################################3


