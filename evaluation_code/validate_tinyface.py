### Source: https://github.com/mk-minchul/AdaFace/tree/master

import sys, os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse

sys.path.append(os.path.join(os.getcwd()))

from . import data_utils, tinyface_helper

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm


def fuse_features_with_norm(stacked_embeddings, stacked_norms, fusion_method='norm_weighted_avg'):

    assert stacked_embeddings.ndim == 3 # (n_features_to_fuse, batch_size, channel)
    if stacked_norms is not None:
        assert stacked_norms.ndim == 3 # (n_features_to_fuse, batch_size, 1)
    else:
        assert fusion_method not in ['norm_weighted_avg', 'pre_norm_vector_add']

    if fusion_method == 'norm_weighted_avg':
        weights = stacked_norms / stacked_norms.sum(dim=0, keepdim=True)
        fused = (stacked_embeddings * weights).sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'pre_norm_vector_add':
        pre_norm_embeddings = stacked_embeddings * stacked_norms
        fused = pre_norm_embeddings.sum(dim=0)
        fused, fused_norm = l2_norm(fused, axis=1)
    elif fusion_method == 'average':
        fused = stacked_embeddings.sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'concat':
        fused = torch.cat([stacked_embeddings[0], stacked_embeddings[1]], dim=-1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'faceness_score':
        raise ValueError('not implemented yet. please refer to https://github.com/deepinsight/insightface/blob/5d3be6da49275602101ad122601b761e36a66a01/recognition/_evaluation_/ijb/ijb_11.py#L296')
        # note that they do not use normalization afterward.
    else:
        raise ValueError('not a correct fusion method', fusion_method)

    return fused, fused_norm


def infer(rank, model, dataloader, use_flip_test, fusion_method):
    model.eval()
    features = []
    norms = []
    with torch.no_grad():
        for images, idx in tqdm(dataloader):
            feature = model(images.to(rank))
            norm = torch.norm(feature, 2, 1, True)
            feature = torch.div(feature, norm)

            if use_flip_test:
                fliped_images = torch.flip(images, dims=[3])
                
                flipped_feature = model(fliped_images.to(rank))
                flipped_norm = torch.norm(flipped_feature, 2, 1, True)
                flipped_feature = torch.div(flipped_feature, flipped_norm)

                stacked_embeddings = torch.stack([feature, flipped_feature], dim=0)
                stacked_norms = torch.stack([norm, flipped_norm], dim=0)

                fused_feature, fused_norm = fuse_features_with_norm(stacked_embeddings, stacked_norms, fusion_method=fusion_method)
                features.append(fused_feature.cpu().numpy())
                norms.append(fused_norm.cpu().numpy())
            else:
                features.append(feature.cpu().numpy())
                norms.append(norm.cpu().numpy())

    features = np.concatenate(features, axis=0)
    norms = np.concatenate(norms, axis=0)
    return features, norms


def tinyface_eval(rank, model, **kwargs):
    data_root = kwargs["eval_path"]
    fusion_method = kwargs["fusion_method"]
    training_desc = kwargs["eval_desc"]
    use_flip_test =  kwargs["use_flip_test"]
    batch_size_eval = kwargs["batch_size_eval"]
    output_path = kwargs["output"]
    result_file = "result_flipto" + str(use_flip_test) + "_" + fusion_method + "_" + training_desc + ".csv"
    os.makedirs(output_path, exist_ok=True)
    tinyface_test = tinyface_helper.TinyFaceTest(tinyface_root=data_root, alignment_dir_name='tinyface_aligned')
    img_paths = tinyface_test.image_paths
    dataloader = data_utils.prepare_dataloader(img_paths,  batch_size_eval, num_workers=0)

    features, norms = infer(rank, model, dataloader, use_flip_test=use_flip_test, fusion_method=fusion_method)
    results = tinyface_test.test_identification(features, ranks=[1,5,20])
    print(results)
    pd.DataFrame({'rank':[1,5,20], 'values':results}).to_csv(os.path.join(output_path, result_file))
