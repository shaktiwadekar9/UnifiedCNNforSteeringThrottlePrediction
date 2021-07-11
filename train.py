#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ShaktiWadekar
"""

import arguments
import utils
from dataloader import SimulationDataset
from model import Net

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import os
import numpy as np
import random



# For reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True) # present in versions from 1.8



class TrainModel():
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.input_shape = (self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH)
        self.steering_or_throttle = None
        self.net = Net()
        
        if self.cfg.cuda == True:
            self.device = torch.device("cuda")
            self.num_workers = 8  # can increase to 16,32 and ...
        else:
            self.device = torch.device("cpu")
            self.num_workers = 0
            
        if (self.cfg.cuda):
            self.net.to(self.device)
            
        
    def load_data(self):       
        
        # load trainset
        trainset = SimulationDataset("train", csv_path=self.cfg.dataset_csv_path,
                steering_or_throttle=self.steering_or_throttle,
                transforms=transforms.Compose([                 
                    utils.RandomChoose(['center'],
                                       steering_or_throttle = self.steering_or_throttle),          
                    utils.Preprocess(self.input_shape),
                    utils.RandomTranslate(10, 10,
                                          steering_or_throttle = self.steering_or_throttle),
                    utils.RandomHorizontalFlip(steering_or_throttle = self.steering_or_throttle),
                    utils.ToTensor(),
                    utils.Normalize([0.1, 0.4, 0.4], [0.9, 0.6, 0.5])
            ]))
        
        self.trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, 
                                                       batch_size=self.cfg.batch_size, 
                                                       num_workers=self.num_workers)
                                                       #pin_memory=True)

        # load testset
        testset = SimulationDataset("test", csv_path=self.cfg.dataset_csv_path,
                steering_or_throttle=self.steering_or_throttle,
                transforms=transforms.Compose([
                    utils.RandomChoose(['center'],
                                       steering_or_throttle = self.steering_or_throttle),
                    utils.Preprocess(self.input_shape),
                    utils.ToTensor(),
                    utils.Normalize([0.1, 0.4, 0.4], [0.9, 0.6, 0.5])
            ]))
        
        self.testloader = torch.utils.data.DataLoader(testset, shuffle=False,
                                                      batch_size=self.cfg.batch_size,
                                                      num_workers=self.num_workers)
                                                      #pin_memory=True)
        
    
    def load_steering_model(self, file_epoch):
        filepath = os.path.join(self.cfg.model_save_path, 'model_steering_E{}.pth'.format(file_epoch))
        print(filepath)
        
        # if file not found, decrement epoch, until we find file
        while os.path.exists(filepath) == False: # exists() returns True or False
            file_epoch -= 1 #decrement epoch and check
            if file_epoch < 0:
                raise ValueError("Epoch given for steering model does NOT exist")
            
        self.net.load_state_dict(torch.load(filepath, map_location=self.device))
        print("Throttle model loaded with model_steering_E{}.pth".format(file_epoch))
        
        #return file_epoch
        
    
    def save_model(self, epoch):
        if self.steering_or_throttle == "steering":
            full_path = os.path.join(self.cfg.model_save_path, \
                                'model_steering_E{}.pth'.format(epoch))               
        elif self.steering_or_throttle == "throttle": 
            full_path = os.path.join(self.cfg.model_save_path, \
                                'model_throttle_SME{}_E{}.pth'\
                                    .format(self.cfg.steering_model_epoch,\
                                            epoch))       
        else:
            raise ValueError('steering_or_throttle variable must be either \
                             "steering" or "throttle"' )       
        torch.save(self.net.state_dict(), full_path)
        
    
    def train(self):
        
        writer = SummaryWriter()  ## Tensorboard: step1

        # set train mode
        self.net.train()

        # loss definition
        if (self.cfg.cuda):
            criterion = nn.MSELoss().cuda()
        else:
            criterion = nn.MSELoss()
            
        # optimizer
        if self.steering_or_throttle == "steering":
            if self.cfg.optimizer == 'adam':
                optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
            elif self.cfg.optimizer == 'adadelta':
                optimizer = optim.Adadelta(self.net.parameters(), lr=1.0, rho=0.9, 
                                           eps=1e-06, weight_decay=0)
            else:
                optimizer = optim.SGD(self.net.parameters(), lr=0.0001, momentum=0.9)        
                
        elif self.steering_or_throttle == "throttle":
            
            # load all the weights from steering model and next freeze the covn layers
            self.load_steering_model(self.cfg.steering_model_epoch) #loads 400th epoch model by default
            
            ################ Freeze layers code [two steps + verify] ######################
            # step 1: Freeze the conv layers until we see first fully connected layer
            for name, param in self.net.named_parameters():                
                if name == "fc1.weight": # In 'name.weight', 'name' must match with name provided in model.py
                    break
                else:
                    param.requires_grad = False
            
            # step 2 : only fully connected layer weights sent to optimizer
            if self.cfg.optimizer == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                       lr=0.0001)
            elif self.cfg.optimizer == 'adadelta':
                optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.net.parameters()),
                                           lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
            else:
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()),
                                      lr=0.0001, momentum=0.9)
            
            # step 3  [verify that conv layers are frozen]
            false_list = []
            true_list = []            
            for name, param in self.net.named_parameters():
                if param.requires_grad == False:
                    false_list.append(name) 
                elif param.requires_grad == True: # written for readability/clarity
                    true_list.append(name)                         
            print("list of layers with requires_grad = False :",false_list)
            print("list of layers with requires_grad = True :",true_list)            
            ######################################################################            
        
        else:
            raise ValueError('steering_or_throttle variable must be either \
                             "steering" or "throttle"')
            
        
        for epoch in range(self.cfg.train_epochs):  # loop over the dataset multiple times

            train_loss, running_loss = 0, 0
            
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                # To prefetch the data, non_blocking=True here and 
                # pin_memory=True in dataloader can be added.
                inputs, labels = data
                if (self.cfg.cuda):
                    inputs = inputs.cuda() # inputs.cuda(non_blocking=True)
                    labels = labels.cuda() # labels.cuda(non_blocking=True)
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if (self.cfg.cuda):
                    outputs = self.net(inputs).cuda() # .cuda(non_blocking=True)
                else:
                    outputs = self.net(inputs)

                # Remove one dimension
                outputs = outputs.squeeze()
                
                # loss
                loss = criterion(outputs, labels)
                
                # get the gradients
                loss.backward()
                
                # update parameters
                optimizer.step()

                # store loss for visulaizaiton/tracking
                running_loss += loss.item()
                del loss

                # print statistics
                if i % 100 == 0: # print every 100 batches
                    print(f'{self.steering_or_throttle}: '+'Running Loss: [%d, %5d] loss: %.6f' % 
                          (epoch + 1, i + 1, running_loss / (i+1)))
            
            train_loss = running_loss / len(self.trainloader) 
            print(f'{self.steering_or_throttle}: '+ \
                  'Epoch Loss: Epoch {%d}: MSE on the traintset: %.6f' % \
                      (epoch+1, train_loss))
            
            ## Sending training loss of each epoch to tensorboard [## Tensorboard: step2]
            writer.add_scalar(f"{self.steering_or_throttle}_Training_Loss",train_loss,epoch+1)  ##Tensorboard
            
            # saving models at save_rate
            if ((epoch + 1) % self.cfg.save_rate == 0):
                self.save_model(epoch+1)
                
            if ((epoch + 1) % self.cfg.test_rate == 0):
                    tmp_res = self.test()
                    
                    ## Sending testing loss of each epoch to tensorboard [## Tensorboard: step2]
                    ## Tensorboard  ## new map created when new variable used
                    writer.add_scalar(f"{self.steering_or_throttle}_Testing_Loss",tmp_res,epoch+1)
                    
        print('Finished Training')
        writer.close()  ## Tensorboard: step3
    
    
    def test(self):
        # set test mode
        self.net.eval()

        if (self.cfg.cuda):
            criterion = nn.MSELoss().cuda()
        else:
            criterion = nn.MSELoss()

        test_loss, running_loss = 0, 0

        for epoch in range(self.cfg.test_epochs):  # loop over the dataset multiple times
            for data in self.testloader:
                inputs, labels = data
                if (self.cfg.cuda):
                    inputs = inputs.cuda() # inputs.cuda(non_blocking=True)
                    labels = labels.cuda() # labels.cuda(non_blocking=True))

                if (self.cfg.cuda):
                    outputs = self.net(inputs).cuda() # .cuda(non_blocking=True)
                else:
                    outputs = self.net(inputs)

                # Remove one dimension
                outputs = outputs.squeeze()
                
                # Compute mean squared error
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                del loss

        if (self.cfg.test_epochs > 0):
            test_loss = running_loss / (len(self.testloader) * self.cfg.test_epochs) 

        print(f'{self.steering_or_throttle}: '+ 'MSE of the network on the testset: %.6f' % (test_loss))
        # set train mode
        self.net.train()

        return test_loss



if  __name__ =='__main__':
    
    cfg = arguments.parse_args()
    if torch.cuda.is_available() == True:
        cfg.cuda = True
        #selecting GPU
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"   ## making specific gpu/gpus available to code
    else:
        cfg.cuda = False
    
    
    
    # Steering Training
    model = TrainModel(cfg=cfg)
    model.steering_or_throttle = "steering"
    model.load_data()
    model.train()
    print("Steering training completed")
    
    # Throttle Training
    # Will load weights from the steering model and train model for throttle by
    # freezing the conv layers
    model = TrainModel(cfg=cfg)
    model.steering_or_throttle = "throttle"
    model.load_data()
    model.train()
    print(f"Throttle training completed with\
          model_steering_E{cfg.steering_model_epoch}")