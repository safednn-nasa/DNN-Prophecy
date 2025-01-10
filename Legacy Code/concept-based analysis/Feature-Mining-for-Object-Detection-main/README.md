# Feature-Mining-for-Object-Detection

The code is adopted from [WongKinYiu's](https://github.com/WongKinYiu/PyTorch_YOLOv4) github repository (commit [101](https://github.com/WongKinYiu/PyTorch_YOLOv4/commit/eb5f1663ed0743660b8aa749a43f35f505baa325)). 

[Overall Plan](https://docs.google.com/document/d/1739mbYTs2FOIsYrOcZhWkwYcO2mbURmI-YExTN1rmeQ/edit?usp=sharing) and [Feature Visualization](https://docs.google.com/document/d/1-HHrRup6-xev_rSDCNA18WLtSQLlxuwqBmX7Ox48iuM/edit?usp=sharing
)  
For references on training yoloV4 model on new datasets, we referred to [1](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects) and [2](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV3_V4_tiny_training.ipynb)

## Setup
```
git clone https://github.com/Jaskiee/Feature-Mining-for-Object-Detection.git
sudo docker run --gpus all -it --rm -p 8081:8888 --ipc=host -v `pwd`/Feature-Mining-for-Object-Detection:/feature-mining -v `pwd`/data/:/root/data nvcr.io/nvidia/pytorch:22.07-py3
```

Insider docker containter,
```
pip install grad-cam
```

## Usage
Command used to train the model: 
```shell
python3 train.py --device 0,1 --batch-size 64 --img 1600 1600 --data nuimages.yaml --cfg 'cfg/yolov4-tiny-25.cfg' --name 'yolov4-tiny-new' --weights 'init.pt'
```
Command used to test the model: 
```shell
python3 test.py --device 0 --batch-size 64 --img 1600 --data nuimages.yaml --cfg 'cfg/yolov4-tiny-25.cfg' --name 'yolov4-tiny-new' --weights 'best.pt'
```
