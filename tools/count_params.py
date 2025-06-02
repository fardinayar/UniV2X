from __future__ import division
import sys
import argparse
import torch
import mmcv
import os
import warnings
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from os import path as osp
from collections import defaultdict, OrderedDict

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger
import sys
sys.path.append('/media/jvn-server/185A27335A270CD6/fardin/UniV2X')
warnings.filterwarnings("ignore")

from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent


def build_hierarchy_tree(param_details):
    """
    Build a hierarchical tree structure from flat module names
    
    Args:
        param_details: Dict with module names as keys and param info as values
    
    Returns:
        dict: Hierarchical tree structure
    """
    tree = defaultdict(lambda: {'children': defaultdict(dict), 'params': {'total': 0, 'trainable': 0, 'frozen': 0}})
    
    for module_name, params in param_details.items():
        parts = module_name.split('.')
        current = tree
        
        # Navigate/create the tree structure
        for i, part in enumerate(parts):
            if part not in current:
                current[part] = {'children': defaultdict(dict), 'params': {'total': 0, 'trainable': 0, 'frozen': 0}}
            
            # Add parameters to this level and all parent levels
            current[part]['params']['total'] += params['total']
            current[part]['params']['trainable'] += params['trainable'] 
            current[part]['params']['frozen'] += params['frozen']
            
            if i == len(parts) - 1:
                # This is a leaf node (actual parameter-containing module)
                current[part]['is_leaf'] = True
                current[part]['direct_params'] = params
            else:
                current[part]['is_leaf'] = False
                current = current[part]['children']
    
    return dict(tree)


def print_hierarchy(tree, name="Model", indent=0, logger=None, show_only_leaves=False):
    """
    Print the hierarchical parameter structure
    
    Args:
        tree: Hierarchical tree structure
        name: Name of current level
        indent: Current indentation level
        logger: Logger for output
        show_only_leaves: If True, only show leaf nodes with direct parameters
    """
    indent_str = "  " * indent
    arrow = "└─ " if indent > 0 else ""
    
    if indent == 0:
        if logger is None:
            print(f"\n{'='*80}")
            print(f"Hierarchical Parameter Count for {name}")
            print(f"{'='*80}")
        else:
            logger.info(f"\n{'='*80}")
            logger.info(f"Hierarchical Parameter Count for {name}")
            logger.info(f"{'='*80}")
    
    # Sort items by name for consistent output
    sorted_items = sorted(tree.items())
    
    for module_name, module_info in sorted_items:
        params = module_info['params']
        is_leaf = module_info.get('is_leaf', False)
        has_children = len(module_info['children']) > 0
        
        # Determine if we should show this node
        should_show = True
        if show_only_leaves and not is_leaf and has_children:
            should_show = False
        
        if should_show and params['total'] > 0:
            if logger is None:
                print(f"{indent_str}{arrow}{module_name:30s} | Total: {params['total']:>10,} | Trainable: {params['trainable']:>10,} | Frozen: {params['frozen']:>10,}")
            else:
                logger.info(f"{indent_str}{arrow}{module_name:30s} | Total: {params['total']:>10,} | Trainable: {params['trainable']:>10,} | Frozen: {params['frozen']:>10,}")
        
        # Recursively print children
        if has_children:
            print_hierarchy(module_info['children'], module_name, indent + 1, logger, show_only_leaves)


def count_parameters(model, name="Model", recursive=True, logger=None):
    """
    Count parameters in a model recursively with hierarchical grouping
    
    Args:
        model: PyTorch model to count parameters for
        name: Name of the model for logging
        recursive: If True, count parameters for each submodule
        logger: Logger for output
    
    Returns:
        dict: Dictionary with parameter counts
    """
    if logger is None:
        print(f"\n{'='*50}")
        print(f"Parameter count for {name}")
        print(f"{'='*50}")
    else:
        logger.info(f"\n{'='*50}")
        logger.info(f"Parameter count for {name}")
        logger.info(f"{'='*50}")
    
    total_params = 0
    trainable_params = 0
    param_details = {}
    
    if recursive:
        # Count parameters for each named module
        for module_name, module in model.named_modules():
            if module_name == "":  # Skip the root module to avoid double counting
                continue
                
            module_params = sum(p.numel() for p in module.parameters(recurse=False))
            module_trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
            
            if module_params > 0:  # Only show modules that have parameters
                param_details[module_name] = {
                    'total': module_params,
                    'trainable': module_trainable,
                    'frozen': module_params - module_trainable
                }
        
        # Build and display hierarchical structure
        if param_details:
            hierarchy_tree = build_hierarchy_tree(param_details)
            
            # Show full hierarchy
            print_hierarchy(hierarchy_tree, name, logger=logger, show_only_leaves=False)
            
            # Show leaf nodes only (modules with direct parameters)
            if logger is None:
                print(f"\n{'-'*80}")
                print(f"LEAF MODULES ONLY (modules with direct parameters)")
                print(f"{'-'*80}")
            else:
                logger.info(f"\n{'-'*80}")
                logger.info(f"LEAF MODULES ONLY (modules with direct parameters)")
                logger.info(f"{'-'*80}")
            
            for module_name, params in sorted(param_details.items()):
                if logger is None:
                    print(f"{module_name:50s} | Total: {params['total']:>10,} | Trainable: {params['trainable']:>10,} | Frozen: {params['frozen']:>10,}")
                else:
                    logger.info(f"{module_name:50s} | Total: {params['total']:>10,} | Trainable: {params['trainable']:>10,} | Frozen: {params['frozen']:>10,}")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    if logger is None:
        print(f"\n{'='*80}")
        print(f"{'TOTAL PARAMETERS':50s} | Total: {total_params:>10,} | Trainable: {trainable_params:>10,} | Frozen: {frozen_params:>10,}")
        print(f"{'='*80}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        if total_params > 0:
            print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    else:
        logger.info(f"\n{'='*80}")
        logger.info(f"{'TOTAL PARAMETERS':50s} | Total: {total_params:>10,} | Trainable: {trainable_params:>10,} | Frozen: {frozen_params:>10,}")
        logger.info(f"{'='*80}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {frozen_params:,}")
        if total_params > 0:
            logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'details': param_details if recursive else {},
        'hierarchy': build_hierarchy_tree(param_details) if recursive and param_details else {}
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Count parameters in a model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--work-dir', help='the dir to save logs', default='./param_counts')
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='whether to count parameters recursively for each module')
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='disable recursive parameter counting')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    
    if args.no_recursive:
        args.recursive = False
    
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plugin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./param_counts',
                                osp.splitext(osp.basename(args.config))[0])

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # init the logger
    import time
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'param_count_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO', name='param_counter')

    logger.info(f'Config file: {args.config}')
    logger.info(f'Recursive counting: {args.recursive}')

    # build other agent models
    other_agent_names = []
    for key in cfg.keys():
        if 'model_other_agent' in key:
            other_agent_names.append(key)

    model_other_agents = {}
    other_agent_param_counts = {}
    
    for other_agent_name in other_agent_names:
        logger.info(f'Building {other_agent_name}...')
        model_other_agent = build_model(
            cfg.get(other_agent_name),
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        model_other_agent.init_weights()
        
        # Load pretrained weights if specified
        other_agent_model_pretrained_from = cfg.get(other_agent_name).get('load_from', None)
        if other_agent_model_pretrained_from:
            logger.info(f'Loading pretrained weights for {other_agent_name} from {other_agent_model_pretrained_from}')
            checkpoint = load_checkpoint(model_other_agent, other_agent_model_pretrained_from, 
                                       map_location='cpu', revise_keys=[(r'^model_ego_agent\.', '')])

        model_other_agents[other_agent_name] = model_other_agent
        
        # Count parameters for this other agent
        other_agent_param_counts[other_agent_name] = count_parameters(
            model_other_agent, f"{other_agent_name}", args.recursive, logger)

    # build ego_agent model
    logger.info('Building model_ego_agent...')
    model_ego_agent = build_model(
        cfg.model_ego_agent,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model_ego_agent.init_weights()
    
    # Load pretrained weights if specified
    ego_agent_model_pretrained_from = cfg.model_ego_agent.get('load_from', None)
    if ego_agent_model_pretrained_from:
        logger.info(f'Loading pretrained weights for ego_agent from {ego_agent_model_pretrained_from}')
        checkpoint = load_checkpoint(model_ego_agent, ego_agent_model_pretrained_from, 
                                   map_location='cpu', revise_keys=[(r'^model_ego_agent\.', '')])

    # Count parameters for ego agent
    ego_agent_param_counts = count_parameters(model_ego_agent, "Ego Agent Model", args.recursive, logger)

    # build multi_agent model
    logger.info('Building MultiAgent model...')
    model_multi_agents = MultiAgent(model_ego_agent, model_other_agents)

    # Count parameters for multi-agent model
    multi_agent_param_counts = count_parameters(model_multi_agents, "Multi-Agent Model", args.recursive, logger)

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("PARAMETER COUNT SUMMARY")
    logger.info(f"{'='*80}")
    
    logger.info(f"Ego Agent Model:")
    logger.info(f"  Total: {ego_agent_param_counts['total']:,}")
    logger.info(f"  Trainable: {ego_agent_param_counts['trainable']:,}")
    logger.info(f"  Frozen: {ego_agent_param_counts['frozen']:,}")
    
    total_other_params = 0
    total_other_trainable = 0
    for other_agent_name, counts in other_agent_param_counts.items():
        logger.info(f"{other_agent_name}:")
        logger.info(f"  Total: {counts['total']:,}")
        logger.info(f"  Trainable: {counts['trainable']:,}")
        logger.info(f"  Frozen: {counts['frozen']:,}")
        total_other_params += counts['total']
        total_other_trainable += counts['trainable']
    
    logger.info(f"All Other Agents Combined:")
    logger.info(f"  Total: {total_other_params:,}")
    logger.info(f"  Trainable: {total_other_trainable:,}")
    logger.info(f"  Frozen: {total_other_params - total_other_trainable:,}")
    
    logger.info(f"Multi-Agent Model (Total):")
    logger.info(f"  Total: {multi_agent_param_counts['total']:,}")
    logger.info(f"  Trainable: {multi_agent_param_counts['trainable']:,}")
    logger.info(f"  Frozen: {multi_agent_param_counts['frozen']:,}")
    
    logger.info(f"{'='*80}")

    # Save summary to a separate file
    summary_file = osp.join(cfg.work_dir, f'param_summary_{timestamp}.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Parameter Count Summary for {args.config}\n")
        f.write(f"Generated at: {timestamp}\n\n")
        f.write(f"Ego Agent Model: {ego_agent_param_counts['total']:,} total, {ego_agent_param_counts['trainable']:,} trainable\n")
        f.write(f"Other Agents Combined: {total_other_params:,} total, {total_other_trainable:,} trainable\n")
        f.write(f"Multi-Agent Model: {multi_agent_param_counts['total']:,} total, {multi_agent_param_counts['trainable']:,} trainable\n")

    logger.info(f"Parameter count summary saved to: {summary_file}")
    print(f"Parameter counting completed. Check logs at: {log_file}")


if __name__ == '__main__':
    main() 