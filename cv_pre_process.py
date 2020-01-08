import os
import cv2
import numpy as np
import random
import pickle
import sys

def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=70):
    """
    Call in a loop to create a terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = "█" * filledLength + ' ' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def cv_pre_process(pictures_path, shuffle=False):
    pictures = os.listdir(pictures_path)
    if shuffle:
        random.shuffle(pictures)
    for i in range(len(pictures)):
        fp = pictures_path + "/" + pictures[i]
        img = cv2.imread(fp)
        img = cv2.resize(src=img, dsize=(224, 224))
        img = np.array(img >= 200, dtype=np.uint8) * 255
        img = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY)
        img = cv2.medianBlur(src=img, ksize=13)
        img = cv2.Laplacian(src=img, ddepth=cv2.CV_8U, ksize=5)
        yield img


def load_data(path_1_in_func, path_2_in_func, filename):
    n1 = len(os.listdir(path_1_in_func))
    n2 = len(os.listdir(path_2_in_func))
    assert n1 == n2, '相似图片组含图数量不等.'
    group_1 = cv_pre_process(path_1_in_func)
    group_2 = cv_pre_process(path_2_in_func)
    group_1_copy = cv_pre_process(path_1_in_func)
    group_2_shuffle = cv_pre_process(path_2_in_func, shuffle=True)

    paired = []
    for img_from_1, img_from_2, j in zip(group_1, group_2, range(1, n1 + 1)):
        paired.append([img_from_1, img_from_2])
        printProgress(j, n1, prefix="create paired progress:")
    paired_labels = [1] * n1
    unpaired = []
    for img_from_1_copy, img_from_2_shuffle, j in zip(group_1_copy, group_2_shuffle, range(1, n1 + 1)):
        unpaired.append([img_from_1_copy, img_from_2_shuffle])
        printProgress(j, n1, prefix="create unpaired progress:")
    unpaired_labels = [0] * n1

    x = paired + unpaired
    y = paired_labels + unpaired_labels
    idx = list(range(2*n1))
    random.shuffle(idx)
    saver = {
        'X': np.array([x[i] for i in idx]),
        'Y': np.array([y[i] for i in idx]),
        'n': n1,
    }
    with open(filename, 'wb') as fp:
        pickle.dump(saver, fp)


if __name__ == '__main__':
    load_data('./demo_1/group_1', './demo_1/group_2', './demo_1_224.pkl')
