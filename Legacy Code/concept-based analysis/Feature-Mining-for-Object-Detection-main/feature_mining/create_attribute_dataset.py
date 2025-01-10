import argparse
import sys
import yaml
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, '/root/siqi/Feature-Mining-for-Object-Detection')

import torch

from feature_mining.hook import *
from feature_mining.utils import *
from utils.torch_utils import *
from utils.datasets import *
from models.models import *
from models import models

from nuimages import NuImages


def prepare_hook(model:Darknet, target:int, target_type) -> ForwardHook:
    target_module = model.module_list[target]
    for module in target_module.modules():
        # if the layer is tha activation layer, add a hook
        if isinstance(module, target_type):
            fh = ForwardHook(module)
            fh.hook()
            break
    return fh

def save_dataset(data: torch.Tensor, label: torch.Tensor, filename:str):
    # TODO: Add labels
    # Results of the model
    torch.save({
            'data' : data,
            'label' : label
        }, f'{filename}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=28, help='The layer for creating dataset')
    parser.add_argument('--device', default='cpu', help='Cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img-size', type=int, default=1600, help='Inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=12, help='Total batch size for all GPUs')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4-tiny-25.cfg', help='model.yaml path')
    parser.add_argument('--yaml', type=str, default='data/nuimages.yaml', help='data.yaml path')
    parser.add_argument('--data-type', type=str, default='val', help='train or val')
    parser.add_argument('--nuimage-data', type=str, default='/root/data/nuimages/images/', help='nuimages path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='Weights path')
    parser.add_argument('--filename', type=str, default='/root/data/activation_outputs/val_data_28.pt', help='Wights path')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    opt = parser.parse_args()

    # Seclect device
    device = select_device(opt.device, batch_size=opt.batch_size)
    print(torch.cuda.device_count())
    # if opt.device == 'cpu':
    #     device = 'cpu'
    # else:
    #     device = 'cuda:' + opt.device

    ###############  Load NuImages  ###############
    dataroot = opt.nuimage_data
    data_type = opt.data_type
    nuim = NuImages(dataroot=dataroot, version='v1.0-'+data_type, verbose=True, lazy=False)

    attribute_dict = {}
    for attri in nuim.attribute:
        attribute_dict[attri['token']] = attri['name']

    # Attributes for samples in training set
    data_attributes = {}
    for ob in nuim.object_ann:
        sample_token = ob['sample_data_token']

        # Get object's file name
        path = dataroot + nuim.get('sample_data', sample_token)['filename']
        if path not in data_attributes and ob['attribute_tokens']:
            data_attributes[path] = set()
        
        # Get object's attribute
        for attribute_token in ob['attribute_tokens']:
            data_attributes[path].add(attribute_dict[attribute_token])


    ###############  Load model  ###############
    # print(device)
    # Create model
    model = Darknet(opt.cfg).to(device)
    # Load checkpoint
    ckpt = torch.load(opt.weights, map_location=device)
    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)

    # Evaluation mode
    model.eval()

    # Prepare the hook
    target_type = (torch.nn.LeakyReLU, )
    # target_type = (torch.nn.Conv2d, )
    hook = prepare_hook(model, opt.layer, target_type)

    ###############  Load dataset  ###############
    imgsz = check_img_size(opt.img_size, s=64)
    batch_size = opt.batch_size
    stride = 64
    pad = 0.5


    with open(opt.yaml) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    # print(data)
    nc = int(data['nc'])
    if data_type == 'train':
        dataset_path = data['train']
    elif data_type == 'val':
        dataset_path = data['val']

    dataloader, _ = create_dataloader(dataset_path, imgsz, batch_size, stride, rect=True, pad=pad)  # dataloader

    ###############  Get activation values  ###############
    activation_outputs = None
    attributes = []
    num = 0;
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        # print(paths)
        img = img.to(device, non_blocking=True).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # Feed the images to the model
        with torch.no_grad():
            model(img)  # forward

        # Get activation values
        output = hook.module_output.cpu()

        # Concat activation outputs
        if activation_outputs is not None:
            activation_outputs = torch.cat((activation_outputs, output), 0)
        else:
            activation_outputs = output.clone()
        
        # Add attributes as labels
        for path in paths:
            if path in data_attributes:
                # If the image has 
                attributes.append(data_attributes[path])
            else:
                attributes.append({})
        
        num += batch_size
        if num > 2000:
            break
            

    ###############  Create dataset  ###############

    save_dataset(activation_outputs, attributes, opt.filename)