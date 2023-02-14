# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''
#to run: python test_LPRNet.py --show true




from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os

Acc_fig=0

Labelzcopy=[]
Targets=[]

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--test_img_dirs', default="./data/GeneratedImages", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=1, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')

    args = parser.parse_args()

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        labels=labels[:7]
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def test(net):
    lab=[]
    args = get_parser()

    # lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    # device = torch.device("cpu")
    # lprnet.to(device)
    # print("Successful to build network!")

    # # load pretrained model
    # if args.pretrained_model:
    #     lprnet.load_state_dict(torch.load(args.pretrained_model,map_location=torch.device('cpu')))
    #     print("load pretrained model successful!")
    # else:
    #     print("[Error] Can't found pretrained mode, please check!")
    #     return False

    test_img_dirs = "./data/GeneratedImages"
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    #print(len(test_dataset))
    try:
        lab=Greedy_Decode_Eval(net, test_dataset, args)
    finally:
        cv2.destroyAllWindows()
        return lab
    

def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()

    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)

        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])
        imgs = images.numpy().copy()

        images = Variable(images)

        # forward

        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        preb_labels123 = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            top1 = list()
            top2 = list()
            top3 = list()
            for j in range(preb.shape[1]):
                copy = preb[:, j]
                #print(copy[np.argmin(copy, axis=0)])
                ind = np.argmax(copy, axis=0)
                preb_label.append(ind)
                top1.append([ind, copy[ind]])
                if ind != 67:
                    copy[ind] = -1000
                ind = np.argmax(copy, axis=0)
                top2.append([ind, copy[ind]])
                if ind != 67:
                    copy[ind] = -1000
                ind = np.argmax(copy, axis=0)
                top3.append([ind, copy[ind]])
            no_repeat_blank_label = list()
            post_process = list()
            nrbl1 = list()
            nrbl2 = list()
            nrbl3 = list()

            pre_c = preb_label[0]
            pre_c1 = top1[0]
            pre_c2 = top2[0]
            pre_c3 = top3[0]

            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            if pre_c1[0] != len(CHARS) - 1:
                nrbl1.append(pre_c1)
            if pre_c2[0] != len(CHARS) - 1:
                nrbl2.append(pre_c2)
            if pre_c3[0] != len(CHARS) - 1:
                nrbl3.append(pre_c3)

            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            for i, c in enumerate(top1): # dropout repeate label and blank label
                if (pre_c1[0] == c[0]) or (c[0] == len(CHARS) - 1):
                    if c[0] == len(CHARS) - 1:
                        pre_c1 = c
                    continue
                nrbl1.append(c)
                post_process.append(i)
                pre_c1 = c
            nrbl2 = np.array(top2)[post_process].tolist()
            nrbl3 = np.array(top3)[post_process].tolist()
            for i, x in enumerate(nrbl2):
                nrbl2[i][0] = int(x[0])
            for i, x in enumerate(nrbl3):
                nrbl3[i][0] = int(x[0])
            


            preb_labels.append(no_repeat_blank_label)
            preb_labels123.append([nrbl1, nrbl2, nrbl3])

            Labelzcopy=preb_labels123.copy()
            Targets=targets.copy()
        for i, label in enumerate(preb_labels123):
            # show image and its predict label
            if args.show:
                show123(label[0], label[1], label[2], targets[i])

        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1
    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    #print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    #print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))
    return Labelzcopy

def show(img, label, target):
    img = np.transpose(img, (1, 2, 0))
    img *= 128.
    img += 127.5
    img = img.astype(np.uint8)

    lb = ""
    for i in label:
        lb += CHARS[i]
    tg = ""
    for j in target.tolist():
        tg += CHARS[int(j)]

    flag = "F"
    if lb == tg:
        flag = "T"
    # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
    img = cv2ImgAddText(img, lb, (0, 0))
    cv2.imshow("test", img)
    print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)
    cv2.waitKey()
    cv2.destroyAllWindows()

def show123(top1, top2, top3, target):
    lb1 = ""
    conf1 = list()
    for i in top1:
        lb1 += CHARS[i[0]]
        conf1.append(int(i[1]))
    lb2 = ""
    conf2 = list()
    for i in top2:
        lb2 += CHARS[i[0]]
        conf2.append(int(i[1]))
    lb3 = ""
    conf3 = list()
    for i in top3:
        lb3 += CHARS[i[0]]
        conf3.append(int(i[1]))
    tg = ""
    for j in target.tolist():
        tg += CHARS[int(j)]
    print(f"target: {tg}\ntop1  : {lb1}   {conf1}\ntop2  : {lb2}   {conf2}\ntop3  : {lb3}   {conf3}\n")
    return conf1, conf2, conf3




def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    test()
