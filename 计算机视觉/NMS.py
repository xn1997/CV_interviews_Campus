from typing import List


class Box:
    x: int
    y: int
    w: int
    h: int
    score: float

    def __init__(self, x, y, w, h, score):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score


def iou(box1: Box, box2: Box):
    """
    计算两个box的IOU
    :param box1:
    :param box2:
    :return:
    """
    xmin = max(box1.x, box2.x)
    ymin = max(box1.y, box2.y)
    xmax = min(box1.x + box1.w, box2.x + box2.w)
    ymax = min(box1.y + box1.h, box2.y + box2.h)
    inter_area = max(0, xmax - xmin) * max(0, ymax - ymin)
    area1 = box1.w * box1.h
    area2 = box2.w * box2.h
    iou = inter_area / (area1 + area2 - inter_area)
    return iou


def nms(boxes: List[Box], threshold: float):
    ret = []  # 返回值
    # 1. 按照置信度排序
    boxes = sorted(boxes, key=lambda x: x.score, reverse=True)  # 降序排序
    index = [_ for _ in range(0, len(boxes))]  # 记录所有待NMS的box索引，迭代维护这个数组
    while len(index) > 0:
        # 2. 选出score最高的box
        box_cur = boxes[index[0]]
        ret.append(box_cur)
        # 3. 计算其与剩余box的IOU，过滤非极大值
        for i in range(len(index) - 1, 0, -1):  # 倒序实现对index的删除操作(一定是用index的索引删除才行)
            if iou(box_cur, boxes[index[i]]) > threshold:
                del index[i]
        del index[0]
        # 此时index只保留了剩余待nms的box
    return ret


import numpy as np
from copy import deepcopy
if __name__ == '__main__':
    boxes = np.array([[100, 100, 210, 210, 0.72],
                  [250, 250, 420, 420, 0.8],
                  [220, 220, 320, 330, 0.92],
                  [100, 100, 210, 210, 0.72],
                  [230, 240, 325, 330, 0.81],
                  [220, 230, 315, 340, 0.9]])
    tmp = []
    for box in boxes:
        a = Box(*box)
        tmp.append(deepcopy(a))
    boxes = tmp
    import matplotlib.pyplot as plt


    def plot_bbox(dets: List[Box], c='k'):
        x1 = [_.x for _ in dets]
        y1 = [_.y for _ in dets]
        x2 = [_.w for _ in dets]
        y2 = [_.h for _ in dets]

        plt.plot([x1, x2], [y1, y1], c)
        plt.plot([x1, x1], [y1, y2], c)
        plt.plot([x1, x2], [y2, y2], c)
        plt.plot([x2, x2], [y1, y2], c)
        plt.title(" nms")


    plt.figure(1)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    plt.sca(ax1)
    plot_bbox(boxes, 'k')  # before nms

    keep = nms(boxes, threshold=0.7)
    plt.sca(ax2)
    plot_bbox(keep, 'r')  # after
    plt.show()
