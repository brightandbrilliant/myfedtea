import os
import pickle
from typing import List
import numpy as np


def read_anchors(datapath: str):
    with open(datapath, 'r') as f:
        content = f.read()
        anchor_list = eval(content)

    assert type(anchor_list) == list
    # print(anchor_list)
    return anchor_list


def parse_anchors(anchors: List, point: int):
    for i in range(0, len(anchors)):
        anchors[i][1] = anchors[i][1] - point
    return anchors


if __name__ == "__main__":
    datapath = "../dataset/dblp/anchors.txt"
    point = 9086
    anchor_list = read_anchors(datapath)
    anchor_list = parse_anchors(anchor_list, point)
    print(anchor_list)
