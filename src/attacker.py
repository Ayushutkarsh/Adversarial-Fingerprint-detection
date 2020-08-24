import os
import sys
import numpy as np

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from evaluation import *
from network import LivDet2015
from roi import Roi
import torchvision
import torchvision.transforms as transforms
#from models import *
from utils import progress_bar
from torch.autograd import Variable
from torchvision import datasets, transforms
from deepfool import deepfool
from torchvision.utils import save_image

from differential_evolution import differential_evolution

class Attacker(object):
    def __init__(self, config, attack_loader):

        # Data loader
        self.attack_loader = attack_loader

        # Models
        self.model_net = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.objective = config.objective
        self.criterion = torch.nn.CrossEntropyLoss()
        self.augmentation_prob = config.augmentation_prob
        self.config =config
        self.roi=Roi()
        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        print("@@@@@@@@@@@@@@@@@@@@@@@ LR B1 & B2 for Adam ------> ",self.lr,self.beta1,self.beta2)

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
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.objective = config.objective
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type =='LivDet2015':
            self.model_net = LivDet2015(in_ch=self.img_ch, num_classes=2)
            print('Building LivDet2015 model again for attacking')
     
            

        self.optimizer = optim.SGD(list(self.model_net.parameters()),
                                      lr=0.000001, momentum=0.9)
        self.model_net.to(self.device)

        # self.print_network(self.model_net, self.model_type)
    
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data
    
    def reset_grad(self):
        """Zero the gradient buffers."""
        self.model_net.zero_grad()
        self.optimizer.zero_grad()
    
    
    def perturb_image(self,xs, img):
        if xs.ndim < 2:
                xs = np.array([xs])
        batch = len(xs)
        imgs = img.repeat(batch, 1, 1, 1)
        xs = xs.astype(int)
        count = 0
        for x in xs:
            if self.img_ch==3:
                pixels = np.split(x, len(x)/5)
            elif self.img_ch==1:
                pixels = np.split(x, len(x)/3)
                
                for pixel in pixels:
                    if self.img_ch==3:
                        x_pos, y_pos, r, g, b = pixel
                        imgs[count, 0, x_pos, y_pos] = (r/255.0-0.4914)/0.2023
                        imgs[count, 1, x_pos, y_pos] = (g/255.0-0.4822)/0.1994
                        imgs[count, 2, x_pos, y_pos] = (b/255.0-0.4465)/0.2010
                       
                    elif self.img_ch==1:
                        x_pos, y_pos, grey = pixel
                        imgs[count, 0, x_pos, y_pos] = (grey-0.1307)/0.3081
                        
                count += 1

        return imgs
    def predict_classes(self,xs, img, target_calss, net, minimize=True):
        imgs_perturbed = self.perturb_image(xs, img.clone())
        #input_image = Variable(imgs_perturbed, volatile=True).cuda()
        input_image = Variable(imgs_perturbed, volatile=True).to(self.device)
        predictions = F.softmax(net(input_image)).data.cpu().numpy()[:, target_calss]

        return predictions if minimize else 1 - predictions
    
    def attack_success(self,x, img, target_calss, net, targeted_attack=False, verbose=False):

        attack_image = self.perturb_image(x, img.clone())
        #input_image = Variable(attack_image, volatile=True).cuda()
        input_image = Variable(attack_image, volatile=True).to(self.device)
        confidence = F.softmax(net(input_image)).data.cpu().numpy()[0]
        predicted_class = np.argmax(confidence)

        if (verbose):
                print ("Confidence: %.4f"%confidence[target_calss])
        if (targeted_attack and predicted_class == target_calss) or (not targeted_attack and predicted_class != target_calss):
                return True
            
    
    def attack(self,img, label, net, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):
        # img: 1*3*W*H tensor
        # label: a number

        targeted_attack = target is not None
        target_calss = target if targeted_attack else label
        image_size = 227
        img_numpy = img.numpy()[0,:,:,:]
        img_numpy = np.reshape(img_numpy, (image_size, image_size, 3))
        print(type(img_numpy),"<-------------- type ",img_numpy.shape)
        objs = self.roi.get_roi(img_numpy,w=8, threshold=.5)
        
        print(objs[0][1].start, objs[0][0].start, objs[0][1].stop, objs[0][0].stop)
        
        if self.img_ch==3:
            #bounds = [(0,image_size), (0,image_size), (0,255), (0,255), (0,255)] * pixels
            bounds = [(objs[0][1].start,objs[0][1].stop), (objs[0][0].start,objs[0][0].stop), (0,255), (0,255), (0,255)] * pixels
        elif self.img_ch==1:
            #bounds = [(0,image_size), (0,image_size), (0,1)] * pixels
            bounds = [(objs[0][1].start,objs[0][1].stop), (objs[0][0].start,objs[0][0].stop), (0,1)] * pixels

        popmul = max(1, popsize/len(bounds))

        predict_fn = lambda xs: self.predict_classes(
                xs, img, target_calss, net, target is None)
        callback_fn = lambda x, convergence: self.attack_success(
                x, img, target_calss, net, targeted_attack, verbose)

        # print("type.popmul", type(popmul))
        inits = np.zeros([int(popmul*len(bounds)), len(bounds)])
        for init in inits:
                for i in range(pixels):
                    if self.img_ch == 3:
                        init[i*5+0] = np.random.random()*image_size
                        init[i*5+1] = np.random.random()*image_size
                        init[i*5+2] = np.random.normal(128,127)
                        init[i*5+3] = np.random.normal(128,127)
                        init[i*5+4] = np.random.normal(128,127)
                        
                    elif self.img_ch==1:
                        init[i*3+0] = np.random.random()*image_size
                        init[i*3+1] = np.random.random()*image_size
                        init[i*3+2] = np.random.normal(-1,1)
                        

        attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
                recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)

        attack_image = self.perturb_image(attack_result.x, img)
        #attack_var = Variable(attack_image, volatile=True).cuda()
        attack_var = Variable(attack_image, volatile=True).to(self.device)
        predicted_probs = F.softmax(net(attack_var)).data.cpu().numpy()[0]

        predicted_class = np.argmax(predicted_probs)

        if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_calss):
            return 1, attack_result.x.astype(int),attack_var
        return 0, [None], attack_var
    
    def attack_all(self,net, loader, pixels=1, targeted=False, maxiter=75, popsize=400, verbose=False):

        total_fake_fp = 0.0
        success = 0
        success_rate = 0

        for batch_idx, (image_input, target,imagename) in enumerate(loader):
                if type(imagename) is tuple:
                    imagename = imagename[0]
                #print("image_input.shape", image_input.shape,target)
                #img_var = Variable(image_input, volatile=True).cuda()
                img_var = Variable(image_input, volatile=True).to(self.device)
                #prior_probs = F.softmax(net(img_var))
                prior_probs = F.softmax(F.sigmoid(net(img_var)))
                #print(prior_probs)
                _, indices = torch.max(prior_probs, 1)
                
                if target[0] ==1:
                    #If the image is live, we dont need to perform attack on it
                    continue
                if target[0] == 0 and target[0] != indices.data.cpu()[0]:
                    #Actual label is fake but prediction already live, attack not possible 
                    continue

                total_fake_fp += 1
                target = target.numpy()

                targets = [None] if not targeted else range(10)
                print("targeted mode", targeted)

                for target_calss in targets:
                        if (targeted):
                                if (target_calss == target[0]):
                                        continue
                        print("Running attack for target",target[0],"and pred",indices.data.cpu()[0])
                        flag, x, attack_var = self.attack(image_input, target[0], net, target_calss, pixels=pixels, maxiter=maxiter, popsize=popsize, verbose=verbose)
                        print("flag==>", flag)

                        success += flag
                        if flag == 1:
                            print("1 positive attack recorded")
                            save_image(img_var,'./dataset/adversarial_data/FGSM/' + imagename+'_purturbed.png')
                                
                
        if (targeted):
            success_rate = float(success)/(9*total_fake_fp)
        else:
            success_rate = float(success)/total_fake_fp
        return success_rate
    
    def FGSM_attack_all(self,net, loader, maxiter=400):

        total_fake_fp = 0.0
        success = 0.0
        success_rate=0.0
        learning_rate = 1e-3
        epsilon = 0.01
        

        for batch_idx, (image_input, target, imagename) in enumerate(loader):
                image_input = image_input.to(self.device)
                target = target.to(self.device)
                if type(imagename) is tuple:
                    imagename = imagename[0]
                
                net.eval() # To switch off Dropouts and batchnorm
                #img_var = Variable(image_input,requires_grad=True).cuda()
                img_var = Variable(image_input,requires_grad=True).to(self.device)
                
                #prior_probs = F.softmax(net(img_var))
                prior_probs = F.softmax(F.sigmoid(net(img_var)))
                _, indices = torch.max(prior_probs, 1)
                
                if target[0] ==1:
                    #If the image is live, we dont need to perform attack on it
                    
                    continue
                if target[0] == 0 and target[0] != indices.data.cpu()[0]:
                    #Actual label is fake but prediction already live, attack not possible 
                    continue

                total_fake_fp += 1.0
                
                for i in range(maxiter):
                    
                    pred_output = net(img_var)
                    prior_probs = F.softmax(F.sigmoid(pred_output))
                    _,indices = torch.max(prior_probs, 1)
                    if target[0] != indices.data.cpu()[0]:
                        #If after perturbations, misclassification occurs, its a +ve attack
                        print("1 positive attack recorded", indices.data.cpu()[0] )
                        success +=1.0
                        save_image(img_var,'./dataset/adversarial_data/FGSM/'+imagename+'_purturbed.png')
                        break
                    
                    loss = self.criterion(pred_output,target)
                    loss.backward()
                    img_var_grad = torch.sign(img_var.grad.data)
                    img_var = img_var.data + epsilon * img_var_grad
                    img_var.requires_grad = True
        success_rate = success/total_fake_fp
        print(" Total correctly recognized fake images are ---> ",total_fake_fp)
        print(" Successful attacks rate ----->", success_rate)

        return success_rate
    
    def DeepFool_attack_all(self,net, loader, maxiter=400):

        total_fake_fp = 0.0
        success = 0.0
        success_rate=0.0
        learning_rate = 1e-3
        epsilon = 0.01
        

        for batch_idx, (image_input, target,imagename) in enumerate(loader):
                if type(imagename) is tuple:
                    imagename = imagename[0]
                image_input = image_input.to(self.device)
                target = target.to(self.device)
                
                net.eval() # To switch off Dropouts and batchnorm
                #prior_probs = F.softmax(net(img_var))
                prior_probs = F.softmax(F.sigmoid(net(image_input)))
                _, indices = torch.max(prior_probs, 1)
                
                if target[0] ==1:
                    #If the image is live, we dont need to perform attack on it
                    
                    continue
                if target[0] == 0 and target[0] != indices.data.cpu()[0]:
                    #Actual label is fake but prediction already live, attack not possible 
                    continue

                total_fake_fp += 1.0
                r, loop_i, label_orig, label_pert, pert_image = deepfool(image_input[0], net,max_iter=maxiter)
                print("Original label = ", np.int(label_orig))
                print("Perturbed label = ", np.int(label_pert))
                if np.int(label_orig) == 0 and np.int(label_pert)== 1:
                    print("1 positive attack recorded")
                    save_image(pert_image,'./dataset/adversarial_data/DeepFool/'+imagename+'_purturbed.png')
                    success+=1.0
                
        success_rate = success/total_fake_fp
        print(" Total correctly recognized fake images are ---> ",total_fake_fp)
        print(" Successful attacks rate ----->", success_rate)

        return success_rate
    
    
    def train(self):
        model_net_path = './models/epoch-9-LivDet2015-200-0.0010-70-0.0000.pkl'
        
        if os.path.isfile(model_net_path):
            # Load the pretrained Encoder
            self.model_net.load_state_dict(torch.load(model_net_path))
            print('%s is Successfully Loaded from %s'%(self.model_type,model_net_path))
            cudnn.benchmark = True
            print("-------> starting Attack <------")
            if self.config.attack_type == 'DE':
                results = self.attack_all(self.model_net, self.attack_loader, pixels=self.config.pixels, targeted=self.config.targeted, maxiter=self.config.maxiter, popsize=self.config.popsize, verbose=False)
                
            elif self.config.attack_type =='FGSM':
                results =self.FGSM_attack_all(self.model_net,self.attack_loader,maxiter=self.config.maxiter)
            
            elif self.config.attack_type == 'DeepFool':
                results = self.DeepFool_attack_all(self.model_net, self.attack_loader, maxiter = self.config.maxiter)
            print(results)
            print ("Final success rate: ",results)
        else:
            print('Cannot find trained model, Cannot attack this network before training')
        
        
