# ECMNet:Lightweight Semantic Segmentation with Efficient CNN-Mamba Network
This repository is an official PyTorch implementation of our paper "ECMNet:Lightweight Semantic Segmentation with Efficient CNN-Mamba Network".



## Train

```
# cityscapes
python train.py --dataset cityscapes --train_type train --max_epochs 1000 --lr 4.5e-2 --batch_size 5

# camvid
python train.py --dataset camvid --train_type train --max_epochs 1000 --lr 1e-3 --batch_size 8
```



## Test

```
# cityscapes
python test.py --dataset cityscapes --checkpoint ${CHECKPOINT_FILE}

# camvid
python test.py --dataset camvid --checkpoint ${CHECKPOINT_FILE}
```

## Predict
only for cityscapes dataset
```
python predict.py --dataset cityscapes 
```

## Results

- Please refer to our article for more details.

| Methods |  Dataset   | Input Size | mIoU(%) |
| :-----: | :--------: | :--------: | :-----: |
| ECMNet  | Cityscapes |  1024x1024  |  70.6   |
| ECMNet  |   CamVid   |  360x480   |  73.6   |




