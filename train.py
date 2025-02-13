from PIL import Image
import glob
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms

def train():
    # prepare data
    name_to_index = class_name_to_class_index("/root/fastestimator_data/tiny_imagenet/tiny-imagenet-200/train")
    train_data = tinydataset(folderPath="/root/fastestimator_data/tiny_imagenet/tiny-imagenet-200/train", name_to_index=name_to_index)
    val_data = tinydataset(folderPath="/root/fastestimator_data/tiny_imagenet/tiny-imagenet-200/val", name_to_index=name_to_index)
    train_loader = DataLoader(train_data, batch_size= 128, num_workers=16, shuffle= True)
    val_loader = DataLoader(val_data, batch_size= 128, num_workers=16, shuffle= True)

    # create network, optimizer, loss
    model = ResNet9(input_size= (3,64,64), classes = 200)
    num_epochs = 20
    opt = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    step = 0 
    best_acc = float('-inf') # set the initial acc as infinite small
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights_dir = "/home/joyce/dl_lab/1_basic_workflow/save_dir"

    model.to(device)
    for epoch in range(num_epochs):
        print("epoch: {}".format(epoch))
        print("start training......")
        model.train()
        for x_train, y_train in train_loader:
            opt.zero_grad()
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            logits = model(x_train)
            loss = loss_fn(logits, y_train)
            loss.backward()
            opt.step()
            step = step + 1
            if step % 100 == 1:
                print("epoch :{}, step:{}, loss:{}".format(epoch, step, loss))
        print("start validation......")
        model.eval()
        total_correct_guess = 0
        total_sample = 0
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            logits = model(x_val) # 128(batch size) * 200(class_num)
            correct_guess = torch.argmax(logits,dim = 1) #return the index of each sample, get 128*1
            num_correct_guess = torch.sum(correct_guess == y_val) # sum all 1 and 0, when equal it is 1.
            total_correct_guess = total_correct_guess + num_correct_guess.to("cpu")##????? to cpu
            num_samples = x_val.shape[0]
            total_sample = total_sample + num_samples
        acc = total_correct_guess / total_sample
        if acc > best_acc:
            best_acc = acc
            modelPath = os.path.join(save_weights_dir, "weights.th")
            print("saving wights to :{}".format(modelPath))
            torch.save(model.state_dict(), modelPath)
        print("epoch:{}, best_accuracy:{}".format(epoch, best_acc))



if __name__ == "__main__":
    train()
  