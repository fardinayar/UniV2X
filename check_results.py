#!/usr/bin/env python3

import pickle
import torch

# Load the results
with open('output/results_to_submit.pkl', 'rb') as f:
    results = pickle.load(f)

print('Type of results:', type(results))

if isinstance(results, dict):
    print('Results is a dictionary with keys:', list(results.keys()))
    if 'results' in results:
        actual_results = results['results']
        print('Number of samples in results:', len(actual_results))
        if len(actual_results) > 0:
            sample = actual_results[0]
        else:
            print('No samples found in results')
            exit()
    else:
        print('Keys in results dict:', list(results.keys()))
        # Take the first value if it's a dict
        sample = list(results.values())[0]
        if isinstance(sample, list) and len(sample) > 0:
            sample = sample[0]
elif isinstance(results, list):
    print('Number of samples:', len(results))
    sample = results[0]
else:
    print('Unexpected results type')
    exit()

print('\nKeys in first sample:')
for key in sample.keys():
    value = sample[key]
    if isinstance(value, torch.Tensor):
        print(f'  {key}: {type(value).__name__} shape={value.shape}')
    elif hasattr(value, '__len__') and not isinstance(value, str):
        print(f'  {key}: {type(value).__name__} len={len(value)}')
    else:
        print(f'  {key}: {type(value).__name__} = {value}')

# Check pts_bbox content if it exists
if 'pts_bbox' in sample:
    print('\nKeys in pts_bbox:')
    pts_bbox = sample['pts_bbox']
    for key in pts_bbox.keys():
        value = pts_bbox[key]
        if isinstance(value, torch.Tensor):
            print(f'  {key}: {type(value).__name__} shape={value.shape}')
        elif hasattr(value, '__len__') and not isinstance(value, str):
            print(f'  {key}: {type(value).__name__} len={len(value)}')
        else:
            print(f'  {key}: {type(value).__name__}')

# Check if expected keys for submission are present
expected_keys = ['token', 'boxes_3d', 'scores_3d', 'labels_3d', 'track_scores', 'track_ids', 
                 'boxes_3d_det', 'scores_3d_det', 'labels_3d_det']

print('\nChecking for expected submission keys:')
for key in expected_keys:
    if key in sample:
        print(f'  ✓ {key}: FOUND')
    elif key in sample.get('pts_bbox', {}):
        print(f'  ✓ {key}: FOUND in pts_bbox')
    else:
        print(f'  ✗ {key}: MISSING')
        
# Check if we need to extract data from pts_bbox
if 'pts_bbox' in sample:
    pts_bbox = sample['pts_bbox']
    print('\nAvailable in pts_bbox that might be useful:')
    useful_keys = ['boxes_3d', 'scores_3d', 'labels_3d', 'track_scores', 'track_ids']
    for key in pts_bbox.keys():
        if any(useful_key in key for useful_key in useful_keys):
            print(f'  {key}: {type(pts_bbox[key])}') 