import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torchvision.utils import save_image
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import LivDet2015
import csv
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
################


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.model_net = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.BCELoss()
        self.objective = config.objective
        if self.objective == 'classification':
            print("Using cross entropy loss ")
            self.criterion = torch.nn.CrossEntropyLoss()
        self.augmentation_prob = config.augmentation_prob
        

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        
        # Training settings
        self.num_epochs = config.num_epochs
        
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        #self.val_step = config.val_step
        self.val_step = 1000000

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type =='LivDet2015':
            self.model_net = LivDet2015(in_ch=self.img_ch, num_classes=2)
     
            
            
        self.optimizer = optim.SGD(list(self.model_net.parameters()),
                                      lr=0.000001, momentum=0.9)
        self.model_net.to(self.device)

        #self.print_network(self.model_net, self.model_type)
    
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        for name_train, param in model.named_parameters():
            if param.requires_grad:
                print (name_train)
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.model_net.zero_grad()
        self.optimizer.zero_grad()

    def compute_accuracy(self,SR,GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

    def tensor2img(self,x):
        img = (x[:,0,:,:]>x[:,1,:,:]).float()
        img = img*255
        return img


    def train(self):
        """Train encoder, generator and discriminator."""
        model_net_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
        print("-------> started Training <------")
        if os.path.isfile(model_net_path):
            # Load the pretrained Encoder
            self.model_net.load_state_dict(torch.load(model_net_path))
            print('%s is Successfully Loaded from %s'%(self.model_type,model_net_path))
        else:
            # Train for Encoder
            lr = self.lr
            best_model_net_score = 0.
            
        for epoch in range(self.num_epochs):

            self.model_net.train(True)
            epoch_loss = 0
            acc_train = 0
            
            
            if self.objective =='classification':
                for i, (images, labels) in enumerate(self.train_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    pred_output = self.model_net(images)
                    loss = self.criterion(pred_output,labels)
                    
                    epoch_loss += loss.item()
                    
                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()
                    pred_probs = F.softmax(pred_output,dim=1)
                    _, preds = torch.max(pred_probs.data, 1)
                    print(preds,labels)
                    acc_train += torch.sum(preds == labels.data)
                    

            # Print the log info
            print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f' % (
                  epoch+1, self.num_epochs, \
                  epoch_loss,\
                  acc_train.item()/len(self.train_loader.dataset)))

            if (epoch+1)%10==0:
                print("saving model")
                model_net_path_epoch = os.path.join(self.model_path, 'epoch-%d-%s-%d-%.4f-%d-%.4f.pkl' %(epoch,self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
                saved_epoch = self.model_net.state_dict()
                torch.save(saved_epoch,model_net_path_epoch)
                print("saving at---> ", model_net_path_epoch)

            # Decay learning rate
            if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print ('Decay learning rate to lr: {}.'.format(lr))
            
    def test(self):
        print("YAAAAAy Testing")     
        #del self.model_net
        #del best_model_net
        self.build_model()
        model_net_test_load_model='epoch-1-ABU_Net-250-0.0002-162-0.5965.pkl'
        model_net_path=os.path.join(self.model_path,model_net_test_load_model)
        print("@@@@@@@@@@@",model_net_path)
        self.model_net.load_state_dict(torch.load(model_net_path))

        self.model_net.train(False)
        self.model_net.eval()

        test_img_dir=self.result_path+'/'+str(self.num_epochs)+"/result_images/"
        if not os.path.exists(test_img_dir):
            os.makedirs(test_img_dir)
        acc_test = 0
        if self.objective =='classification':
            for i, (images, label) in enumerate(self.test_loader):
                images = images.to(self.device)
                label = label.to(self.device)
                pred_output = self.model_net(images)
                pred_probs = F.sigmoid(pred_output)
                loss = self.criterion(pred_probs,label)
                _, preds = torch.max(pred_probs.data, 1)
                acc_test += torch.sum(preds == labels.data) 
        print('Loss: %.4f, \n[Training] Acc: %.4f' % (
              loss, \
              acc_test/len(self.train_loader.dataset)))
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  