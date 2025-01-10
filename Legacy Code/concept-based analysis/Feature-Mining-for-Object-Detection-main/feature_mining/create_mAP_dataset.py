import argparse
import sys
import yaml
from tqdm import tqdm
sys.path.insert(0, '/root/siqi/Feature-Mining-for-Object-Detection')

import torch

from feature_mining.hook import *
from feature_mining.utils import *
from utils.torch_utils import *
from utils.datasets import *
from models.models import *
from models import models


def prepare_hook(model:Darknet, target:int) -> ForwardHook:
    target_module = model.module_list[target]
    for module in target_module.modules():
        # if the layer is tha activation layer, add a hook
        if isinstance(module, (torch.nn.LeakyReLU)):
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
    parser.add_argument('--layer', type=int, default=35, help='The layer for creating dataset')
    parser.add_argument('--device', default='cpu', help='Cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img-size', type=int, default=1600, help='Inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=1, help='Total batch size for all GPUs')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4-tiny-25.cfg', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/nuimages_sub.yaml', help='data.yaml path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='Weights path')
    parser.add_argument('--filename', type=str, default='activation_outputs/output.pt', help='Wights path')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    opt = parser.parse_args()

    ###############  Load model  ###############
    device = select_device(opt.device, batch_size=opt.batch_size)
    # device = 'cuda:1'
    print(device)
    # Create model
    model = Darknet(opt.cfg).to(device)
    # Load checkpoint
    ckpt = torch.load(opt.weights, map_location=device)
    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)

    # Evaluation mode
    model.eval()

    # Prepare the hook
    hook = prepare_hook(model, opt.layer)

    ###############  Load dataset  ###############
    imgsz = check_img_size(opt.img_size, s=64)
    batch_size = opt.batch_size
    # source = opt.data
    stride = 64
    pad = 0.5


    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    print(data)
    nc = int(data['nc'])
    train_path = data['train']
    # val_path = data['val']
    test_path = data['val']

    # dataset = LoadImages(source, img_size=imgsz, auto_size=64)
    testloader, testset = create_dataloader(test_path, imgsz, batch_size, stride, rect=True, pad=pad)  # testloader


    ###############  Get activation values  ###############
    activation_outputs = None
    mAPs = []
    i = 0
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(testloader)):
        # img = torch.from_numpy(img).to(device).float()
        img = img.to(device, non_blocking=True).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)

        # Feed the images to the model
        with torch.no_grad():
            pred = model(img)  # forward
            output = non_max_suppression(pred[0], conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)
            assert len(output)==1, "batch size should be 1"
            img = torch.squeeze(img)
            output = output[0]
            labels = targets[targets[:, 0] == 0, 1:].to(device, non_blocking=True)
            mAP = mAP_per_image(img, output, labels, nc, device)
            # print(mAP)

        # Get activation values
        output = hook.module_output.cpu()

        # Concat activation outputs
        if activation_outputs is not None:
            activation_outputs = torch.vstack((activation_outputs, output))
        else:
            activation_outputs = output.clone()
        i += 1
        if i >= 100:
            break
        
        # Add mAP as labels
        mAPs.append(mAP)
            
    ###############  Create dataset  ###############

    save_dataset(activation_outputs, mAPs, opt.filename)

    
