#parsing command line arguments
import argparse
#decoding camera images
import base64
#for frametimestamp saving
from datetime import datetime
#reading and writing files
import os
#high level file operations
import shutil
#matrix math
import numpy as np
#real-time server
import socketio
#concurrent networking 
import eventlet
#web server gateway interface
import eventlet.wsgi
#image manipulation
from PIL import Image
#web framework
from flask import Flask
#input output
from io import BytesIO
# Load model
from model import DriveNet

import torch
from torchvision import transforms
import arguments
import utils

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)


class DriveModel():
    
    def __init__(self, cfg):
        self.net = DriveNet()
        self.cfg = cfg
        self.input_shape = (self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.cfg.cuda = True
        else:
            self.device = torch.device("cpu")
            self.cfg.cuda = False
    
    def save_model(self):
        torch.save(self.net.state_dict(), self.cfg.fullpath_DriveNet)  # default: saves in current directory
        
    def load_drive_model(self):
        if self.cfg.cuda == False:
            self.net.load_state_dict(torch.load(self.cfg.fullpath_DriveNet, map_location="cpu"))
        else:
            self.net.load_state_dict(torch.load(self.cfg.fullpath_DriveNet))
    
    ### loading weights with verification
    def load_weights_dict(self):
        
        ## loading weights using dictionary
        steering_dict = torch.load(
                            os.path.join(self.cfg.model_save_path,
                            'model_steering_E{}.pth'.format(self.cfg.steering_model_epoch)),
                            map_location=self.device)
                            
        #_________________________________________________________________________________________
        # if file not found, decrement steering epoch, until we steering model file is found
        throttlefilepath = os.path.join(os.path.join(self.cfg.model_save_path,
                            'model_throttle_SME{}_E{}.pth'.format(self.cfg.steering_model_epoch\
                                                                  ,self.cfg.throttle_model_epoch)))
        if os.path.exists(throttlefilepath) == False:
            if self.cfg.look_for_lower == True:
                while os.path.exists(throttlefilepath) == False: # exists() returns True or False
                    print("WARNING: Required throttle model not available, trying lower steering_epoch trained file")
                    self.cfg.steering_model_epoch -= 1 #decrement epoch and check
                    if self.cfg.steering_model_epoch < 0:
                        raise ValueError("Epoch given for steering model and lower does NOT exist")
                        
                    throttlefilepath = os.path.join(self.cfg.model_save_path,
                                    'model_throttle_SME{}_E{}.pth'.format(self.cfg.steering_model_epoch\
                                                                          ,self.cfg.throttle_model_epoch))
                throttle_dict = torch.load(
                                    throttlefilepath,
                                    map_location=self.device)
            else:
                raise ValueError('Throttle file does NOT exist' )
        else:
            throttle_dict = torch.load(
                                    throttlefilepath,
                                    map_location=self.device)
        print('model_throttle_SME{}_E{}.pth'\
                        .format(self.cfg.steering_model_epoch\
                                ,self.cfg.throttle_model_epoch) + 'throttle model loaded')
        #_________________________________________________________________________________________
        
        hybrid_dict = torch.load(self.cfg.fullpath_DriveNet,
                                 map_location=self.device) # default: finds in current directory
        # loading Conv layers
        for (hy_name,ph), (st_name,ps), (th_name, pt) in zip(hybrid_dict.items(),
                                                steering_dict.items(),
                                                throttle_dict.items()):
            
            if st_name[0:2] == "fc":
                break
            else:
                #check if throttle and steering conv layers are same
                if ps.data.ne(pt.data).sum() > 0:
                    # if NOT same, give warining and copy conv weights from steering/throttle model
                    hybrid_dict[hy_name].data.copy_(ps.data)
                    print("Steering layer {} NOT same as Throttle layer {}".format(st_name,th_name))
                    print("Steering layer {} copied to hybrid layer {}".format(st_name,hy_name))
                else:
                    # if same, copy conv weights from steering/throttle model
                    hybrid_dict[hy_name].data.copy_(ps.data)
                    print("Steering layer {} same as Throttle layer {}".format(st_name,th_name))
                    print("Steering layer {} copied to hybrid layer {}".format(st_name,hy_name))
                
        # loading fully connected layers
        for (st_name,p1), (th_name, p2) in zip(steering_dict.items(),
                                                throttle_dict.items()):            
            if st_name[0:2] == "fc":
                hybrid_dict[f"{st_name[0:3]}_steering{st_name[3:]}"].data.copy_(p1.data)
                print(f"Steering layer {st_name} copied to hybrid layer {st_name[0:3]}_steering{st_name[3:]}")
                
            if th_name[0:2] == "fc":
                hybrid_dict[f"{th_name[0:3]}_throttle{th_name[3:]}"].data.copy_(p2.data)
                print(f"Throttle layer {th_name} copied to hybrid layer {th_name[0:3]}_throttle{th_name[3:]}")
                
        ##***** important.... saving dict in pth format*******##
        torch.save(hybrid_dict, self.cfg.fullpath_DriveNet)
        
    
    def predict(self, image, preloaded=False):
        # set test mode
        self.net.eval()

        if (not preloaded):
            self.load_drive_model()
            print('Loaded Model')

        composed=transforms.Compose([
            utils.Preprocess(self.input_shape),
            utils.ToTensor(),
            utils.Normalize([0.1, 0.4, 0.4], [0.9, 0.6, 0.5])
        ])
        # Target gets discareded
        sample = {'image': image, 'target': 0}
        sample  = composed(sample)
        inputs = sample['image']
        
        # Add single batch diemension
        inputs = inputs.unsqueeze(0)

        if (self.cfg.cuda):
            inputs = inputs.cuda() # .cuda(non_blocking=True)

        if (self.cfg.cuda):
            outputs = self.net(inputs).cuda() # .cuda(non_blocking=True)
        else:
            outputs = self.net(inputs)

        # set train mode
        self.net.train()

        return outputs

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:        
        # Send input image and get the steering and throttle predicitons
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        output = model.predict(image, preloaded=True)
        steering_pred = float(output[0])
        throttle_pred = float(output[1])
        
        print("network prediction -> (steering angle: {:.3f}, \
              throttle: {:.3f})".format(steering_pred, throttle_pred))

        send_control(steering_pred, throttle_pred)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    
    cfg = arguments.parse_args()
    
    # Create the DriveNet model architecture
    model = DriveModel(cfg=cfg)
    
    # Loading of weights in the model is done using weights dictionaryies
    ## step1: Save DriveNet weights to dictionary
    model.save_model()
    ## step2: loading weights in DriveNet dictionary from steering and throttle model dictionaries
    model.load_weights_dict()
    ## step3: Load the DriveNet model with updated DriveNet dictionary
    model.load_drive_model()
    
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
