#!/usr/bin/env python3

import mmcv
import numpy as np
import os
import json
import pickle
from tqdm import tqdm
from pyquaternion import Quaternion

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

def create_test_pkl_wo_label(data_root, save_path):
    """Create test pkl file without ground truth labels for wo_label inference"""
    
    # Load cooperative data info
    coop_data_info_path = os.path.join(data_root, 'cooperative/data_info.json')
    coop_data_infos = load_json(coop_data_info_path)
    
    # Load vehicle and infrastructure data info
    veh_data_info_path = os.path.join(data_root, 'vehicle-side/data_info.json')
    veh_data_infos = load_json(veh_data_info_path)
    
    inf_data_info_path = os.path.join(data_root, 'infrastructure-side/data_info.json')
    inf_data_infos = load_json(inf_data_info_path)
    
    # Create mappings
    veh_sample_mappings = {}
    inf_sample_mappings = {}
    
    for veh_info in veh_data_infos:
        veh_sample_mappings[veh_info['frame_id']] = veh_info
    
    for inf_info in inf_data_infos:
        inf_sample_mappings[inf_info['frame_id']] = inf_info
    
    test_infos = []
    command_data = {}
    
    print(f"Processing {len(coop_data_infos)} cooperative samples...")
    
    for idx, coop_data_info in enumerate(tqdm(coop_data_infos)):
        veh_frame_id = coop_data_info['vehicle_frame']
        inf_frame_id = coop_data_info['infrastructure_frame']
        
        # Get vehicle and infrastructure sample info
        veh_sample_info = veh_sample_mappings[veh_frame_id]
        inf_sample_info = inf_sample_mappings[inf_frame_id]
        
        # Create basic info structure
        info = {
            'token': veh_frame_id,
            'frame_idx': idx,
            'scene_token': veh_sample_info['sequence_id'],
            'location': veh_sample_info['intersection_loc'],
            'timestamp': float(veh_sample_info['pointcloud_timestamp']),
            'prev': '',  # Will be filled later if needed
            'next': '',  # Will be filled later if needed
            'other_agent_info_dict': {}
        }
        
        # Vehicle sensor info
        info['lidar_path'] = os.path.join('vehicle-side', veh_sample_info['pointcloud_path'].replace('pcd','bin'))
        info['lidar2ego_rotation'] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
        info['lidar2ego_translation'] = [0.0, 0.0, 0.0]
        info['ego2global_rotation'] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
        info['ego2global_translation'] = [0.0, 0.0, 0.0]
        
        # Camera info for vehicle
        info['cams'] = {}
        camera_type = 'VEHICLE_CAM_FRONT'
        info['cams'][camera_type] = {}
        info['cams'][camera_type]['data_path'] = os.path.join('vehicle-side', veh_sample_info['image_path'])
        info['cams'][camera_type]['cam_intrinsic'] = np.eye(3)  # Default identity matrix
        info['cams'][camera_type]['lidar2cam_rotation'] = np.eye(3)
        info['cams'][camera_type]['lidar2cam_translation'] = np.zeros(3)
        info['cams'][camera_type]['sensor2lidar_rotation'] = np.eye(3)
        info['cams'][camera_type]['sensor2lidar_translation'] = np.zeros(3)
        info['cams'][camera_type]['sensor2ego_rotation'] = np.eye(3)
        info['cams'][camera_type]['sensor2ego_translation'] = np.zeros(3)
        
        # Infrastructure agent info
        other_agent_info = {
            'token': inf_frame_id,
            'frame_idx': idx,
            'scene_token': veh_sample_info['sequence_id'],
            'location': veh_sample_info['intersection_loc'],
            'timestamp': float(inf_sample_info['pointcloud_timestamp']),
            'prev': '',
            'next': '',
            'system_error_offset': coop_data_info.get('system_error_offset', {'delta_x': 0, 'delta_y': 0})
        }
        
        # Infrastructure sensor info
        other_agent_info['lidar_path'] = os.path.join('infrastructure-side', inf_sample_info['pointcloud_path'].replace('pcd','bin'))
        other_agent_info['lidar2ego_rotation'] = [1.0, 0.0, 0.0, 0.0]
        other_agent_info['lidar2ego_translation'] = [0.0, 0.0, 0.0]
        other_agent_info['ego2global_rotation'] = [1.0, 0.0, 0.0, 0.0]
        other_agent_info['ego2global_translation'] = [0.0, 0.0, 0.0]
        other_agent_info['VehLidar2InfLidar_rotation'] = np.eye(3)
        other_agent_info['VehLidar2InfLidar_translation'] = np.zeros(3)
        
        # Infrastructure camera info
        other_agent_info['cams'] = {}
        camera_type = 'INF_CAM_FRONT'
        other_agent_info['cams'][camera_type] = {}
        other_agent_info['cams'][camera_type]['data_path'] = os.path.join('infrastructure-side', inf_sample_info['image_path'])
        other_agent_info['cams'][camera_type]['cam_intrinsic'] = np.eye(3)
        other_agent_info['cams'][camera_type]['lidar2cam_rotation'] = np.eye(3)
        other_agent_info['cams'][camera_type]['lidar2cam_translation'] = np.zeros(3)
        other_agent_info['cams'][camera_type]['sensor2lidar_rotation'] = np.eye(3)
        other_agent_info['cams'][camera_type]['sensor2lidar_translation'] = np.zeros(3)
        other_agent_info['cams'][camera_type]['sensor2ego_rotation'] = np.eye(3)
        other_agent_info['cams'][camera_type]['sensor2ego_translation'] = np.zeros(3)
        
        # Add sweeps and can_bus (empty for wo_label)
        info['sweeps'] = {}
        info['can_bus'] = np.zeros(18)
        other_agent_info['sweeps'] = {}
        other_agent_info['can_bus'] = np.zeros(18)
        
        # Empty annotations for wo_label
        info['gt_boxes'] = np.empty((0, 7))
        info['gt_names'] = np.array([])
        info['gt_ins_tokens'] = np.array([])
        info['gt_inds'] = np.array([])
        info['anno_tokens'] = np.array([])
        info['valid_flag'] = np.array([])
        info['num_lidar_pts'] = np.array([])
        info['timestamps'] = np.array([])
        info['visibility_tokens'] = np.array([])
        info['gt_velocity'] = np.empty((0, 2))
        info['prev_anno_tokens'] = np.array([])
        info['next_anno_tokens'] = np.array([])
        
        info['other_agent_info_dict']['model_other_agent_inf'] = other_agent_info
        test_infos.append(info)
        
        # Create a dummy command (straight driving)
        command_data[veh_frame_id] = np.array([1.0, 0.0, 0.0])  # [straight, left, right]
    
    # Save the test pickle file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    metadata = dict(version='v1.0-trainval')
    data = dict(infos=test_infos, metadata=metadata)
    
    print(f"Saving {len(test_infos)} test samples to {save_path}")
    mmcv.dump(data, save_path)
    
    # Save command file
    command_path = save_path.replace('.pkl', '_command.pkl')
    print(f"Saving command data to {command_path}")
    mmcv.dump(command_data, command_path)
    
    return save_path, command_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    args = parser.parse_args()
    
    create_test_pkl_wo_label(args.data_root, args.save_path) 