"""
@author: ShaktiWadekar
"""

import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='process file paths')
    
    # collect the paths first
    parser.add_argument('--dataset_csv_path', default='driving_log.csv',
                        type=str,
                        help='csv file contains path to all images, \
                            steering and throttle data for training and \
                            testing. Default: driving_log.csv in current \
                            directory')
    
    parser.add_argument('--model_save_path', default='trained_models', type=str,
                        help='path to save model. Default: current directory')
    parser.add_argument('--model_load_path', default='', type=str,
                        help='path to load model. Default: current directory')
    
    # input image arguments
    parser.add_argument('--IMAGE_HEIGHT', default=80, type=int,
                        help='input image resized to this height. Default 80')
    parser.add_argument('--IMAGE_WIDTH', default=160, type=int,
                        help='input image resized to this width. Default 160')
    
    # training arguments
    parser.add_argument('--batch_size', default=128, type=int,\
                         help='')
    parser.add_argument('--train_epochs', default=500, type=int,\
                         help='')
    parser.add_argument('--optimizer', default='adam',\
                         help='')
    parser.add_argument('--save_rate', default=50, type=int,\
                         help='save model every save_rate epoch. Default: every 200th epoch')
        
    # Used BOTH in throttle training and drive
    parser.add_argument('--steering_model_epoch', default=400, type=int, \
                         help='Used for loading specific steering model \
                             during throttle training and drive')
    parser.add_argument('--throttle_model_epoch', default=400, type=int, \
                         help='Used for loading specific throttle model \
                             during only drive')
    
    # testing arguments
    parser.add_argument('--test_epochs', default=1, type=int,\
                         help='')
    parser.add_argument('--test_rate', default=10, type=int,\
                         help='test model every test_rate epoch. Default: every 10th epoch')
    
    
    # important
    parser.add_argument('--cuda', default='True', type=bool,\
                         help='Important *******Should be True when training on GPU. \
                             Should be False when driving on track *********')
                            
    # general
    parser.add_argument('--log_dir', default='', type=str,\
                        help='')
    parser.add_argument('--log_file', default='log.json', type=str,\
                        help='')
    parser.add_argument('--plot_file', default='plot.png', type=str,\
                        help='')
    parser.add_argument('--auto_plot', default='True', type=bool,\
                        help='')
    parser.add_argument('--clean_start', default='False', type=bool,\
                        help='')
        
    # drive
    parser.add_argument('--fullpath_DriveNet', default='model_DriveNet.pth', type=str,\
                        help='DriveNet file path directory+file. \
                            Default current: Currentdirectory+model_DriveNet.pth')
    parser.add_argument('--look_for_lower', default=False, type=bool,\
                        help='Set it True if you want to automatically check and load \
                            lower steering_epoch throttle model')
    
    
    args = parser.parse_args()
    return args