#!/usr/bin/env python3

import os
import argparse

import sys
sys.path.append('../')

from core.solver import Solver
# from data_loader import get_loader
from data_loader.OpportunityDataset import get_loader
from torch.backends import cudnn
from config import OpportunityOpt
from torch.backends import cuda


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    # cudnn.benchmark = True
    # cudnn.deterministic = True # try to remove the cuda error CUDNN_STATUS_INTERNAL_ERROR
    # cudnn.allow_tf32 = True
    # cuda.matmul.allow_tf32 = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    opportunity_loader = None

    # FIXME
    if config.dataset in ['Opportunity']:
        print("selected attributes (positions): ", config.selected_attrs, ", users: ", OpportunityOpt['users'], ", classes: ", OpportunityOpt['classes'], "\n")

        # opportunity_loader = get_loader(sensor_data_file_path=config.opportunity_sensor_dir, users=OpportunityOpt['users'],
        #                                 positions=OpportunityOpt['positions'], selected_attrs_activities=OpportunityOpt['classes'], batch_size=config.batch_size,
        #                                 dataset='Opportunity', mode=config.mode,
        #                                 num_workers=config.num_workers)


        # all users, all activities, but positions are different (config.selected_attrs)
        opportunity_loader = get_loader(sensor_data_file_path=config.opportunity_sensor_dir, users=OpportunityOpt['users'],
                                        positions=config.selected_attrs, activities=OpportunityOpt['classes'], batch_size=config.batch_size,
                                        dataset='Opportunity', mode=config.mode,
                                        num_workers=config.num_workers)
    
    solver = Solver(opportunity_loader, config)

    if config.mode == 'train':
        print("Train")
        solver.train()
    elif config.mode == 'test':
        print("test")
        # solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--style_dim', type=int, default=4, help='dimension of domain labels (1st dataset)') # CelebA dataset (why five? isn't it 40?) => (black hair, blond hair, brown hair, male , young)?
    parser.add_argument('--channel_dim', type=int, default=6)
    parser.add_argument('--window_size', type=int, default=60, help='image resolution') # 2d size?
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    # parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--dataset', type=str, default='Opportunity', choices=['Opportunity', 'Pamap2', 'Both'])
    # parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    # parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters', type=int, default=20, help='number of total iterations for training D')
    # parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--num_iters_decay', type=int, default=10, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    # parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
    #                     default=['stand', 'walk', 'sit', 'lie'])
                        # default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the Opportunity dataset',
                        default=['RUA', 'LLA', 'L_Shoe', 'Back']) # set it as four different positions out of 7

    # Test configuration.
    # parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    parser.add_argument('--test_iters', type=int, default=20, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    # FIXME: Add directories for human activity recognition datasets
    parser.add_argument('--opportunity_sensor_dir', type=str, default='/mnt/sting/adiorz/mobile_sensing/datasets/opportunity_std_scaled_all.csv')
    # parser.add_argument('--attr_path', type=str, default='data/list_attr_opportunity.txt')
    parser.add_argument('--attr_path', type=str, default='data/attribute_list_opportunity.txt')

    parser.add_argument('--log_dir', type=str, default='/mnt/sting/adiorz/mobile_sensing/logs/transferability/opp_logs')
    parser.add_argument('--model_save_dir', type=str, default='opportunity/models')
    parser.add_argument('--sample_dir', type=str, default='opportunity/samples')
    parser.add_argument('--result_dir', type=str, default='opportunity/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)