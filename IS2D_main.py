#-*- coding:utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse
import warnings
warnings.filterwarnings('ignore')

from utils.get_functions import get_save_path
from utils.save_functions import save_metrics
from IS2D_Experiment.biomedical_2dimage_segmentation_experiment import BMISegmentationExperiment

def IS2D_main(args) :
    print("Hello! We start experiment for 2D Image Segmentation!")

    try:
        args.train_dataset_dir = os.path.join(args.data_path, args.train_data_type)
        args.test_dataset_dir = os.path.join(args.data_path, args.test_data_type)
    except TypeError:
        print("join() argument must be str, bytes, or os.PathLike object, not 'NoneType'")
        print("Please explicitely write the dataset type")
        sys.exit()

    if args.train_data_type in ['PolypSegData', 'DSB2018', 'ISIC2018', 'COVID19', 'BUSI']:
        args.num_channels = 3
        args.image_size = 352
        args.num_classes = 1
        args.metric_list = ['DSC', 'IoU', 'WeightedF-Measure', 'S-Measure', 'E-Measure', 'MAE']
    else:
        print("Wrong Train dataset...")
        sys.exit()

    experiment = BMISegmentationExperiment(args)
    test_results = experiment.inference()
    model_dirs = get_save_path(args)

    print("Save MADGNet Test Results...")
    save_metrics(args, test_results, model_dirs, args.final_epoch)

if __name__=='__main__' :
    parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself!')
    parser.add_argument('--data_path', type=str, default='dataset/BioMedicalDataset')
    parser.add_argument('--train_data_type', type=str, required=False, choices=['PolypSegData', 'DSB2018', 'ISIC2018', 'COVID19', 'BUSI'])
    parser.add_argument('--test_data_type', type=str, required=False, choices=['CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB',
                                                                                            'DSB2018', 'MonuSeg2018', 'ISIC2018', 'PH2', 'COVID19', 'COVID19_2', 'BUSI', 'STU'])

    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--save_path', type=str, default='model_weights')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--seed_fix', default=False, action='store_true')
    parser.add_argument('--final_epoch', type=int, default=200)
    parser.add_argument('--step', type=int, default=10)

    # MFMSNet parameter
    parser.add_argument('--cnn_backbone', type=str, required=False, default='resnest50')
    parser.add_argument('--scale_branches', type=int, required=False, default=3)
    parser.add_argument('--frequency_branches', type=int, required=False, default=16)
    parser.add_argument('--frequency_selection', type=str, required=False, default='top')
    parser.add_argument('--block_repetition', type=int, required=False, default=1)
    parser.add_argument('--min_channel', type=int, required=False, default=32)
    parser.add_argument('--min_resolution', type=int, required=False, default=8)

    args = parser.parse_args()

    # Skin Cancer Segmentation Generalizability Test
    if args.train_data_type == 'ISIC2018':
        for test_data_type in ['ISIC2018', 'PH2']:
            args.test_data_type = test_data_type
            IS2D_main(args)

    # COVID19 lesion Segmentation Generalizability Test
    if args.train_data_type == 'COVID19':
        for test_data_type in ['COVID19', 'COVID19_2']:
            args.test_data_type = test_data_type
            IS2D_main(args)

    # Polyp Segmentation Generalizability Test
    if args.train_data_type == 'PolypSegData':
        for test_data_type in ['CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            args.test_data_type = test_data_type
            IS2D_main(args)

    # Ultrasound Tumor Segmentation Generalizability Test
    if args.train_data_type == 'BUSI':
        for test_data_type in ['BUSI', 'STU']:
            args.test_data_type = test_data_type
            IS2D_main(args)

    # Cell Segmentation Generalizability Test
    if args.train_data_type == 'DSB2018':
        for test_data_type in ['DSB2018', 'MonuSeg2018']:
            args.test_data_type = test_data_type
            IS2D_main(args)