import os

import torch

def get_deivce() :
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("You are using \"{}\" device.".format(device))

    return device

def get_save_path(args):
    save_model_path = '{}'.format(args.train_data_type)


    model_dirs = os.path.join(args.save_path, save_model_path)
    if not os.path.exists(os.path.join(model_dirs, 'model_weights')): os.makedirs(os.path.join(model_dirs, 'model_weights'))
    if not os.path.exists(os.path.join(model_dirs, 'test_reports')): os.makedirs(os.path.join(model_dirs, 'test_reports'))

    return model_dirs