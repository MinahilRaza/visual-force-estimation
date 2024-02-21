# Force Estimation using Convolutional Neural Networks

## Training

The following command runs the training:

```bash
python3 train.py --batch_size 8 --lr 0.00001 --num_epochs 5 --train_runs 1 2 3 4
```

## Image Preprocessing

The following command crops and resizes the images to 256 x 256:
```
python3 resize_crop_images.py -d <DIR>
```