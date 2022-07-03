# -------------------------------------------------------------------------------------
# Negative-Aware Attention Framework for Image-Text Matching  implementation based on SCAN
# https://github.com/CrossmodalGroup/NAAF
# "Negative-Aware Attention Framework for Image-Text Matching"
# Kun Zhang, Zhendong Mao, Quan Wang, Yongdong Zhang
#
# Writen by Kun Zhang, 2022
# -------------------------------------------------------------------------------------

from vocab import Vocabulary
import evaluation
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RUN_PATH = "/home/zhangkun/NAAF/runs/runX/checkpoint/checkpoint_16.pth_80.6_60.0_507.9.tar"
DATA_PATH = "/home/zhangkun/data/"

evaluation.evalrank(RUN_PATH, data_path=DATA_PATH, split="test")
