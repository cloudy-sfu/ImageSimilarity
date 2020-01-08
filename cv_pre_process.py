import os
import cv2
import numpy as np
import random
import pickle
import sys


def print_process(iteration, total, prefix='', suffix='', decimals=1, bar_length=70):
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
    format_str = "{0:." + str(decimals) + "f}"
    percent = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "█" * filled_length + ' ' * (bar_length - filled_length)
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
        img = _process_core(fp)
        yield img


def _process_core(fp):
    img = cv2.imread(fp)
    try:
        img = cv2.resize(src=img, dsize=(100, 100))
    except AssertionError:
        img = np.zeros((100, 100))
        return img
    img = np.array(img >= 200, dtype=np.uint8) * 255
    img = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY)
    img = cv2.medianBlur(src=img, ksize=13)
    img = cv2.Laplacian(src=img, ddepth=cv2.CV_8U, ksize=5)
    return img


def load_data(path_1_in_func, path_2_in_func, filename):
    fp1 = os.listdir(path_1_in_func)
    if 'desktop.ini' in fp1:
        fp1.remove('desktop.ini')
    n1 = len(fp1)
    fp2 = os.listdir(path_2_in_func)
    if 'desktop.ini' in fp2:
        fp2.remove('desktop.ini')
    n2 = len(fp2)
    assert n1 == n2, '相似图片组含图数量不等.'
    group_1 = cv_pre_process(path_1_in_func)
    group_2 = cv_pre_process(path_2_in_func)
    group_1_copy = cv_pre_process(path_1_in_func)
    group_2_shuffle = cv_pre_process(path_2_in_func, shuffle=True)

    paired = []
    for img_from_1, img_from_2, j in zip(group_1, group_2, range(1, n1 + 1)):
        paired.append([img_from_1, img_from_2])
        print_process(j, n1, prefix="create paired progress:")
    paired_labels = [1] * n1
    unpaired = []
    for img_from_1_copy, img_from_2_shuffle, j in zip(group_1_copy, group_2_shuffle, range(1, n1 + 1)):
        unpaired.append([img_from_1_copy, img_from_2_shuffle])
        print_process(j, n1, prefix="create unpaired progress:")
    unpaired_labels = [0] * n1

    x = paired + unpaired
    y = paired_labels + unpaired_labels
    idx = list(range(2 * n1))
    random.shuffle(idx)
    saver = {
        'X': np.array([x[i] for i in idx]),
        'Y': np.array([y[i] for i in idx]),
        'n': n1,
    }
    with open(filename, 'wb') as fp:
        pickle.dump(saver, fp)


def load_data_batch_test(path_1_in_func, path_2_in_func, filename):
    fp1 = os.listdir(path_1_in_func)
    if 'desktop.ini' in fp1:
        fp1.remove('desktop.ini')
    n1 = len(fp1)
    fp2 = os.listdir(path_2_in_func)
    if 'desktop.ini' in fp2:
        fp2.remove('desktop.ini')
    n2 = len(fp2)
    assert n1 == n2, '相似图片组含图数量不等.'
    group_1 = cv_pre_process(path_1_in_func)
    group_2 = cv_pre_process(path_2_in_func)

    paired = []
    for img_from_1, img_from_2, j in zip(group_1, group_2, range(1, n1 + 1)):
        paired.append([img_from_1, img_from_2])
        yield 100 * j / n1
    x = paired
    saver = {
        'X': np.array([x[i] for i in range(n1)]),
        'n': n1,
    }
    with open(filename, 'wb') as fp:
        pickle.dump(saver, fp)


def load_data_single_test(img_1, img_2):
    img_1 = _process_core(img_1)
    img_2 = _process_core(img_2)
    paired = np.array([[img_1, img_2]])
    return paired


def load_data_valid(path_1_in_func, path_2_in_func, filename):
    fp1 = os.listdir(path_1_in_func)
    if 'desktop.ini' in fp1:
        fp1.remove('desktop.ini')
    n1 = len(fp1)
    fp2 = os.listdir(path_2_in_func)
    if 'desktop.ini' in fp2:
        fp2.remove('desktop.ini')
    n2 = len(fp2)
    assert n1 == n2, '相似图片组含图数量不等.'
    group_1 = cv_pre_process(path_1_in_func)
    group_2 = cv_pre_process(path_2_in_func)
    group_1_copy = cv_pre_process(path_1_in_func)
    group_2_shuffle = cv_pre_process(path_2_in_func, shuffle=True)

    paired = []
    for img_from_1, img_from_2, j in zip(group_1, group_2, range(1, n1 + 1)):
        paired.append([img_from_1, img_from_2])
        yield 50 * j / n1
    paired_labels = [1] * n1
    unpaired = []
    for img_from_1_copy, img_from_2_shuffle, j in zip(group_1_copy, group_2_shuffle, range(1, n1 + 1)):
        unpaired.append([img_from_1_copy, img_from_2_shuffle])
        yield 50 * (1 + j / n1)
    unpaired_labels = [0] * n1

    x = paired + unpaired
    y = paired_labels + unpaired_labels
    idx = list(range(2 * n1))
    random.shuffle(idx)
    saver = {
        'X': np.array([x[i] for i in idx]),
        'Y': np.array([y[i] for i in idx]),
        'n': n1,
    }
    with open(filename, 'wb') as fp:
        pickle.dump(saver, fp)
