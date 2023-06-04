from flask import Flask, render_template, request, redirect,  flash, abort, url_for, session
from underwater import app
from fileinput import filename

from models import CC_Module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import time
from options import opt
import math
import shutil
from tqdm import tqdm



@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template("project.html")
@app.route('/beautify', methods = ['POST'])  
def beautify():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f.filename)
        CHECKPOINTS_DIR = opt.checkpoints_dir
        # INP_DIR = opt.testing_dir_inp
        # CLEAN_DIR = opt.testing_dir_gt

        # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'    

        device = 'cpu'    

        ch = 3

        network = CC_Module()
        checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR,"netG_295.pt"),map_location=torch.device('cpu'))
        print(checkpoint['epoch'])
        print(checkpoint['mse_loss'])
        print(checkpoint['vgg_loss'])
        print(checkpoint['total_loss'])
        network.load_state_dict(checkpoint['model_state_dict'])
        network.eval()
        network.to(device)

        result_dir = './underwater/static/enhanced_images/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)


        # total_files = os.listdir(INP_DIR)
        st = time.time()
        # with tqdm(total=len(total_files)) as t:

        #     for m in total_files:
        print(f.filename)
        img=cv2.imread(f.filename)
        img = img[:, :, ::-1]   
        img = np.float32(img) / 255.0
        h,w,c=img.shape
        train_x = np.zeros((1, ch, h, w)).astype(np.float32)
        train_x[0,0,:,:] = img[:,:,0]
        train_x[0,1,:,:] = img[:,:,1]
        train_x[0,2,:,:] = img[:,:,2]
        dataset_torchx = torch.from_numpy(train_x)
        dataset_torchx=dataset_torchx.to(device)
        output=network(dataset_torchx)
        output = (output.clamp_(0.0, 1.0)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        output = output[:, :, ::-1]
        cv2.imwrite(os.path.join(result_dir + str("output.png")), output)

                # t.set_postfix_str("name: {} | old [hw]: {}/{} | new [hw]: {}/{}".format(str(m), h,w, output.shape[0], output.shape[1]))
                # t.update(1)
                
        end = time.time()
        print('Total time taken in secs : '+str(end-st))
        return render_template("project.html", data='output.png',epoch = checkpoint['epoch'],  mse_loss = checkpoint['mse_loss'],vgg_loss = checkpoint['vgg_loss'],total_loss = checkpoint['total_loss'],)