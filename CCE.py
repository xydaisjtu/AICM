# encoding:utf-8
import sys
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from utils import *
import pprint
import math
import matplotlib.pyplot as plt

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('CACM')
    parser.add_argument('--seq_path', nargs='+',
                        help='The path the click sequence file')
    parser.add_argument('--plot_label', nargs='+',
                        help='The label of the plot')
    parser.add_argument('--file_name', type=str, default='CCE.png',
                        help='The label of the plot')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    for path in args.seq_path:
        assert os.path.isfile(path), 'file: {} does not exist!'.format(path)
    assert len(args.seq_path) == len(args.plot_label)
    file_num = len(args.seq_path)
    L1s = [[0.0 for i in range(10)] for _ in range(file_num)]
    for idx, file_path in enumerate(args.seq_path):
        file = open(file_path, 'r')
        lines = file.readlines()
        cnt = 0
        for line in lines:
            items = line.strip().split('\t')
            logits = eval(items[0])
            pred_clicks = eval(items[1])
            true_clicks = eval(items[2])
            for i in range(10):
                if true_clicks[i]:
                    L1s[idx][i] += -math.log(logits[i])
                else:
                    L1s[idx][i] += -math.log(1 - logits[i])
            cnt += 1
        for i in range(10):
            L1s[idx][i] = -math.log(L1s[idx][i] / cnt)
        L1 = sum(L1s[idx]) / 10
        print('{}, {}'.format(L1s[idx], L1))
    
    plt.figure()
    colors = ['red', 'blue', 'orange', 'green']
    for idx, plot_label in enumerate(args.plot_label):
        plt.plot(range(1, 11), L1s[idx], marker='o', color=colors[idx], label=plot_label)
    plt.legend()
    plt.xlabel('Rank')
    plt.ylabel('Avg cross entropy')
    plt.savefig('{}.png'.format(args.file_name))
