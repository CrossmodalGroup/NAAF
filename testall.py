# -------------------------------------------------------------------------------------
# Negative-Aware Attention Framework for Image-Text Matching  implementation based on SCAN
# https:.
# "Negative-Aware Attention Framework for Image-Text Matching"
# Kun Zhang, Zhendong Mao, Quan Wang, Yongdong Zhang
#
# Writen by Kun Zhang, 2022
# -------------------------------------------------------------------------------------

from vocab import Vocabulary
import evaluation
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# RUN_PATH = "/mnt/data10t/bakuphome20210617/zhangkun/neg_2021_10_7/runs/runX/checkpoint/checkpoint_16_523.2.pth.tar"

# RUN_PATH = "/mnt/data10t/bakuphome20210617/zhangkun/neg_2021_10_7/runs/runX/checkpoint/checkpoint_16_523.2.pth.tar"

RUN_PATH = "/mnt/data10t/bakuphome20210617/zhangkun/neg_2021_10_22/runs/runX/checkpoint/model_best.pth.tar"



DATA_PATH = "/mnt/data10t/bakuphome20210617/zhangkun/data/"
# evaluation.evalrank(RUN_PATH, data_path=DATA_PATH, split="dev",fold5=False)
evaluation.evalrank(RUN_PATH, data_path=DATA_PATH, split="testall",fold5=False)
