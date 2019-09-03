# coding=utf-8
import os

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std

def mkdir( rank, experiment_name):
    """传进来一个实验名， 生成 ./logs ./result ./model 生成对应文件夹"""
    folder = os.path.exists("logs/"+experiment_name )

    if not folder:
        os.makedirs("logs/" + experiment_name)
    if not os.path.exists("logs/"+experiment_name + "/DDPG_" + str(rank)):
        os.makedirs("logs/" + experiment_name + "/DDPG_" + str(rank))

    folder = os.path.exists("result/" + experiment_name)
    if not folder:
        os.makedirs("result/" + experiment_name)

    folder = os.path.exists("model/" + experiment_name)
    if not folder:
        os.makedirs("model/" + experiment_name)
