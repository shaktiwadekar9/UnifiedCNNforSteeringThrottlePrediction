This repo holds source code for [ICRA 2021 Workshop paper](https://linklab-uva.github.io/icra-autonomous-racing/contributed_papers/paper10.pdf)

<div align="center">

# UnifiedCNN_steering_throttle
[![arXiv](https://img.shields.io/badge/arxiv.org-2105.01799-b31b1b.svg)](https://arxiv.org/abs/2105.01799)
[![ICRA](https://img.shields.io/badge/ICRA-2021-blue)](https://linklab-uva.github.io/icra-autonomous-racing/contributed_papers/paper10.pdf)

</div>


### Table of contents

1. [Installation](#1-Installation) 
2. [Training](#2-Training)
3. [Testing](#3-Testing)
4. [Acknowledgement](#4-Acknowledgement)


https://www.youtube.com/watch?v=On0RhWkMLW4

https://www.youtube.com/watch?v=ChaoakkGMgs

### 1. Installation: Create environment for training and testing

**STEPS for installing packages required for running this code**

>For users not using conda:

>Need to install packages provided in general_environment.yml file and install pytorch using command given on official website (official website command includes torchvision and cudatookit)

**STEP 1** (conda users): Use .yml to create new conda environment

_command_:

```bash
conda env create -f general_environment.yml
```

yml file: general_environment.yml


>If drive.py is giving socket error, then remove current version and install specific version of packages mentioned below.
>python-socketio=4.5.1,
>python-engineio=3.11.2,
>flask-socketio=4.3.1


<!--
##### Creating my EXACT env 
(WARNING: check the cuda and cudatoolkit version suitable for your gpu):
yaml file:UnifiedSteeringThrottle.yml
Cuda: 10.1
cudatoolkit:cudatoolkit=10.1.243=h6bb024c_0
pytorch version:pytorch=1.5.1=py3.6_cuda10.1.243_cudnn7.6.3_0 
-->


**STEP 2** (conda users): Activate the newly created environment

_command_:

```bash
conda activate new_environment_name
```

**STEP 3**: Install pytorch from the official website using conda in this new environment

example: conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

This will install following packages:
pytorch,
torchvision,
cudatoolkit


### 2. Training

**STEP 1**: Get the training data from udacity simulator.

>How to generate the training data using udacity simulator?:

**Installing udacity simulator**: 

```bash
Use Term 1 simulator: Version 1  Windows64. 
```

**Term 1 Version 1 version is stable. Version 2 simulator files tend to oftern crash at startup.**

[This code is verified with udacity simulator on windows]

**Collect training data**: Go in training mode, then click the record button to choose which folder you want to save your training data in. Two things will be saved, One is a Folder with name 'IMG' and second is a 'driving_log.csv' file. driving_log.csv file has the paths to training images and for each image it has corresponding steering angle, throttle value and speed associated with it. For more info on simulator see [Udacity simulator](https://github.com/udacity/self-driving-car-sim)


**STEP 2**: Run the training

*Before running this command create 'trained_models' folder  in the current directory,
if the directory is not already present.*

_command_:

```bash
python train.py --dataset_csv_path driving_log.csv
```

OR

```bash
python train.py --dataset_csv_path driving_log.csv --train_epochs 1000 --steering_model_epoch 800
```

**How is training code running?**
>Model needs images and steering / throttle values to train, which are obtained from the .csv file. 

>First, steering training occurs. Net() is loaded from model.py and trained for steering prediction. The model files are saved by name model_steering_E{epoch_number}. Model is saved every 'save_rate' epochs. Default value of save_rate is 50. So model is saved every 50 epochs. You can change it using --save_rate argument. Trained files are saved in 'trained_models' folder. Saved Steering models will look like this for example model_steering_E50, model_steering_E200 and etc.

>Second, throttle training occurs. Before training of throttle starts, following things happen:
>1. Net() is loaded from model.py
>2. All the weights of throttle model are **initialized** with trained steering model. The trained steering model is choosen based on the --steering_model_epoch argument. Default value is 400. So by default the code will look for 'model_steering_E400' in 'trained_models'. You can change this value using --steering_model_epoch argument.
>3. **Convolutional layer weights are frozen** using requrie_grad=False. Therefore, Convolutional layers weights are NOT trained during throttle training, they are the same weights as the trained steering model convolutional layer weights.
>4. Now during throttle training, **only fully connected weights are trained**.

>The trained throttle models are saved by name model_throttle_SME{steering_model_epoch}_E{epoch_number}. SME{steering_model_epoch}, indicates that convolutional layer weights of model_steering_E{steering_model_epoch} was used for throttle training. So with above command which has steeirng_model_epoch as 1000, the saved trained files will look like this model_throttle_SME1000_E50, model_throttle_SME1000_E600 and etc.

**Mandatory arguments to be given**
--dataset_csv_path:

This is the path to csv file. Udacity simulator generates driving_log.csv file,
along with the IMG folder which has the training images.


**Arguments for flexibility**

--train_epochs : Default 500

--save_rate : Default 50. Model is saved every 50 epochs

--steering_model_epoch: Depends on save_rate. **Important argument for throttle training**

>**Steering Model saved on this {steering_model_epoch} epoch will be used to train the throttle model.**

>**Throttle training uses trained convolutional layers from steering model.**

>**These convolutional layers are frozen and not trained during throttle training.**

>**Only the fully connected layers in the throttle model are trained during throttle training.**


**Visualize training on tensorboard**

'runs' folder will be created in the current directory from previous step.

_command_:

```bash
tensorboard --logdir=runs
```

This command will print the link to be used in browser for live plot visualization.
Example link: http://localhost:6006/

### 3. Testing
_command_:

```bash
python drive.py --steering_model_epoch 800 --throttle_model_epoch 700
```


drive.py uses DriveNet() from model.py for steering and throttle prediction.

**There are three parts to DriveNet model,**
1. Conv layers
2. Fully connected layers for steering
3. Fully connected layers for throttle

>Conv layers are loaded from the steering model.

>Fully connected layers for steering are loaded from steering model.

>Fully connected layers for throttle are loaded from throttle model.

**Steering model: model_steering_E{steering_model_epoch}**

**Throttle model: model_throttle_SME{steering_model_epoch}_E{throttle_model_epoch}**


--steering_model_epoch: Depends on save_rate

--throttle_model_epoch: Depends on save_rate

**Note:**

>Trained steering models which are closer in terms of epochs to the trained steering model which was used for throttle training, also works during testing/driving on track on the simulator. That is the reason for introducing '--look_for_lower' argument. 

>'look_for_lower' can be set to 'True' or 'False'. Default value is 'False'.

>If set to 'True'. The code will automatically look for trained steering models which are trained and saved at lower epochs.

```bash
python drive.py --steering_model_epoch 800 --throttle_model_epoch 700 --look_for_lower True
```

So if model_steering_E{steering_model_epoch} is not available and 'look_for_lower' is true, the code will look for trained steering models by decrementing steering_model_epoch number and look for models like  model_steering_E799, model_steering_E798 and so on until it finds the file or reaches 0.



### 4. Acknowledgement

Parts of the code are from [Reference1](https://github.com/shaktiwadekar9/Udacity-Self-driving-car-simulator-pytorch-code) and [Reference2](https://github.com/pgebert/autonomous_car_simulation).
