## Introduction
This is [Negative-Aware Attention Framework for Image-Text Matching](https://www.researchgate.net/publication/360642414_Negative-Aware_Attention_Framework_for_Image-Text_Matching), source code of NAAF. The paper is accepted by CVPR2022 and its Chinese blog can be found [here](https://www.cnblogs.com/lemonzhang/p/16456403.html). It is built on top of the [SCAN](https://github.com/kuanghuei/SCAN) in PyTorch. 

![image](https://github.com/CrossmodalGroup/NAAF/blob/main/Framework%20Overview.jpg)
## Requirements and Installation
We recommended the following dependencies.

* Python 3.6
* [PyTorch](http://pytorch.org/) 1.8.0
* [NumPy](http://www.numpy.org/) (>1.19.5)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
* The specific required environment can be found [here](https://drive.google.com/file/d/1jLhd1GU6W3YrKeADM5g4qQxJoYt1lXx5/view?usp=sharing)

## Pretrained model
If you don't want to train from scratch, you can download the pretrained NAAF model from [here](https://drive.google.com/file/d/1e3I5Uk2UGHPql4KLIrQW5L7ek3ih34rh/view?usp=sharing)(for Flickr30K model) and [here](https://drive.google.com/file/d/1NpZZYXmmejgd_nam79IdETSYIuRjo-7p/view?usp=sharing)(for Flickr30K model without using GloVe). The performance of this pretrained single model is as follows, in which some Recall@1 values are even better than results produced by our paper:
```bash
rsum: 507.9
Average i2t Recall: 91.3
Image to text: 80.6 95.4 98.0 1.0 2.0
Average t2i Recall: 78.0
Text to image: 60.0 83.9 89.9 1.0 7.4
```
## Performance
We provide our NAAF model performance (single or ensemble) under different text backbones, where readers can choose the appropriate performance for a fair comparison:

![image](https://github.com/CrossmodalGroup/NAAF/blob/main/Performance-Bi-GRU.png)
![image](https://github.com/CrossmodalGroup/NAAF/blob/main/Performance-GloVe.png)

## Download data
Download the dataset files. We use the image feature created by SCAN, downloaded [here](https://github.com/kuanghuei/SCAN). The vocabulary required by GloVe has been placed in the 'vocab' folder of the project (for Flickr30K and MSCOCO).

## Training

```bash
python train.py --data_path "$DATA_PATH" --data_name f30k_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/log --logg_path runs/runX/logs --model_name "$MODEL_PATH" 
```

Arguments used to train Flickr30K models and MSCOCO models are similar with those of SCAN:

For Flickr30K:

| Method      | Arguments |
| :---------: | :-------: |
|  NAAF   | `--lambda_softmax=20 --num_epoches=20 --lr_update=10 --learning_rate=.0005 --embed_size=1024 --batch_size=128 `|

For MSCOCO:

| Method      | Arguments |
| :---------: | :-------: |
|  NAAF   | `--lambda_softmax=20 --num_epoches=20 --lr_update=10 --learning_rate=.0005 --embed_size=1024 --batch_size=256 `|

## Evaluation

Test on Flickr30K
```bash
python test.py
```

To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`--data_name coco_precomp`.

```bash
python testall.py
```

To ensemble model, specify the model_path in test_stack.py, and run
```bash
python test_stack.py
```

## Reference

If you found this code useful, please cite the following paper:
```
@inproceedings{zhang2022negative,
  title={Negative-Aware Attention Framework for Image-Text Matching},
  author={Zhang, Kun and Mao, Zhendong and Wang, Quan and Zhang, Yongdong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15661--15670},
  year={2022}
}
```

