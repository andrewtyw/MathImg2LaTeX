# MathImg2LaTeX
Convert pictures of mathematical formulas to LaTeX expressions
## Method
### Model architecture
The Encoder is ResNet, and the Decoder is Transformer Layers.

### Training process
First, use MLM(Mask Language Model) to pretrain Model. Then fine-tune Model. Google used MLM when pre-training Bert and got a certain improvement, so here also considers using it.

***
## How to run?
### 1, Get training data
There are 10k training samples in  `/resources/data/` (far from enough to train a good model). Run the following to unzip.
```
sh get_data.sh
```
It is more recommended to use personalized training data. The requirements of training data: 
- Put training images here: `/resources/data/imgs` , and put training labels here: `/resources/data/labels` (the same as data samples)
- the file name of the image and the label must be the same. e.g. `/resources/data/imgs/1_0.png` corresponds to `/resources/data/labels/1_0.txt`

### 2, Data pre-process
```
python prepare_data.py --train_val_test_ratio 7:2:1
```
### 3, MLM pretrain
```
python  train.py --MLM_pretrain_mode True --lr 1e-3 --d_model 256 --dim_feedforward 256 --batch_size 64 --epoches 30
```
### 3, fine-tune
```
python  train.py --from_MLM True --lr 3e-4 --d_model 256 --dim_feedforward 256 --batch_size 64 --epoches 30 
```
***
## Evaluation
Put your images you need to predict here: `/resources/evaluate_imgs`
```
python  evaluate.py --img_name img1.png
```
f ( f ( \sin \alpha ) ) = 2

## Result
### Evaluation Metrics
- Exact Match: **Proportion** that the prediction is exactly the same with ground truth.
- Edit distance: Given the prediction and ground truth, the minimum number of operations required to transform prediction into ground truth.


## Acknowledgement
- Some of the code is adopted from the project https://github.com/kingyiusuen/image-to-latex.
