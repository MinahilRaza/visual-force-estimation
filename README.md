# Force Estimation using Convolutional Neural Networks

## Training

The following command runs the training:

```bash
python3 train.py --batch_size 8 --lr 0.00001 --num_epochs 5 --train_runs 1 2 3 4
```

## Evaluation

The following command evaluates a model, where the weights are stored in a `.pth` file at `<PATH_TO_WEIGHTS>`:

```bash
python3 evaluate.py --weights <PATH_TO_WEIGHTS> --run <RUN_NR> --pdf
```
The flag `--pdf` is used to create plots in pdf format, otherwise they are created in png format.

## Image Preprocessing

The following command crops and resizes the images to 256 x 256:
```
python3 resize_crop_images.py -d <DIR>
```