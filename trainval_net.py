# --------------------------------------------------------
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import datetime





'''
function to take in and set the variables sent through the command line
'''
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network') # The parser is just an argparse object which is from a built in python class
  parser.add_argument('--dataset', dest='dataset', # defining the dataset
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net', # neural network type
                    help='vgg16, res101',
                    default='res101', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=15, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval', # this is the just visual stuff here
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval', # this is the just visual stuff here
                      help='number of iterations to display',
                      default=10000, type=int)
  
  parser.add_argument('--depth', dest='depth',
                      help='whether use depth or not',
                      default=False, type=bool) # Custom, whether to use depth or not
  parser.add_argument('--save_dir', dest='save_dir',  # this is where the output is saved
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--pretrained', dest='if_pretrained',				# Custom whether to use pretrained caffe weights or not
                      help='Whether to use pretrained weights or not',
                      default=True, type=bool)
  parser.add_argument('--track_dir', dest='track_dir',				# Custom Track Directory
                      help='Path to use for keeping track directory',
                      default="track", type=str)
  
# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',        ######
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',	# choose whether to resume training a net or start fresh
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and display
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args







class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
  	 
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data










if __name__ == '__main__':

  args = parse_args() # function call to get the command line variables. Note that the variable are returned incapsulated in the 'args' local variable which acts as a goto reference for all the command line variables


   
  """
  Track stuff here:
  """
  track_dir = args.track_dir
  if not os.path.exists(track_dir):
    os.makedirs(track_dir)
  track_file = open("{}/{}.txt".format(track_dir,args.session),'a+',0)
  #track_file.truncate()
   

  
  
  
  
  


  print('Called with args:')
  print(args)
  
  
  args.depth = False # Custom 
  args.if_pretrained = False # Custom
  
    
    
  
  '''
  Setting important variables in the 'args' local variable. Note that here we are storing further information in the 'args' variable.
  This is DB related info
  '''
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  # New addition to work with the KITTI dataset in the form of KITTIVOC. We will copy this from the first if statement as we use the pascal_voc data architecture
  # Note that the same name of the dataset is used whether we use rgb or rgbd for kitti data set
  elif args.dataset == "kitti_voc": 																								
      args.imdb_name = "kitti_voc_trainval"
      args.imdbval_name = "kitti_voc_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']      
  elif args.dataset == "pascal_voc_0712":																							# This is also used for pascal dataset if we want to use the 2007 and the 2012 dataset together
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)		# Loading the correct config file for the neural network







  """
	This code below checks whether config file is present or not.
	if it is present, then config variable is modified in config.py file using the cfg_file
	else, the config variable is modified in config.py file using the set_cfgs variable defined earlier
	Note that the config variable in config.py is __C and after the following code is executed, a lot of details regarding training net are stored in __C
	The __C variable is available to us in this script as the cfg variable imported from config.py
	So now we have to sources of information, the cfg and the args variable
  """
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)


  print('Using config:')
  pprint.pprint(cfg)
  
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")



  """
	till now we have only defined important configuration variables which are used in processed later
  """











  """
	This is where we carry out the database related tasks
	we load the database into defined variables
	most DB related configuration takes place here
  """
  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = False
  cfg.USE_GPU_NMS = args.cuda
  cfg.TRAIN.USE_DEPTH = args.depth  
  cfg.TRAIN.IF_PRETRAINED = args.if_pretrained
  
  print(args.depth, args.if_pretrained)    

  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name) # where the magic happens. Once we have the correct imdb, roidb, ratio list and ratio index, we don't care what happens to the data at a later stage.
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))
  track_file.write('{} roidb entries\n'.format(len(roidb)))
  
  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset # creating the directory used to store the net we will eventually create after training. This piece of code is a little out place
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)		# the main variable used to store the dataset in batch groups and to cycle through the batches
                            
  
                            
                          
       







       
       
       
       
  """
	Some CUDA stuff here
	This initializes 4 important variables which are used as storage units for local data as the net is cycling through the batch data
  """
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)
  
  
  

      	

  if args.cuda:
    cfg.CUDA = True
  
  
  
  
 






  track_file.write("#"*100)
  track_file.write("\n{}, {}, {}\n".format(args.net,args.imdb_name,args.session))
  track_file.write("#"*100)  
  track_file.write("\n") 















  """
  initilize the network here.
  !!!!!!! 	Note that this is where the PRETRAINED models/weights are loaded iff pretrained=True. 
  """
    
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=args.if_pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=args.if_pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=args.if_pretrained, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=args.if_pretrained, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum














  """
	Neural network configuration happens here
  """
  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.cuda:
    fasterRCNN.cuda()

  if args.resume:		# !!!!!!!! If we want to resume training of some model, then this piece of code loads this model and then overwrites some variables, model weights, and optimizer
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model']) # Note that this was already initialized in resnet101.py or vgg16.py. The model is being overwritten here
    optimizer.load_state_dict(checkpoint['optimizer']) # optimizer is being overwritten too
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)























  """
  This is where the training loop takes place
  """
  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")


  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)
    
    for step in range(iters_per_epoch):
      data = next(data_iter)	# Loading the next batch of data to be used in training
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])
      

      	
      fasterRCNN.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes) # function call to train on the loaded data in the current batch

      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
      loss_temp += loss.item()

      # backward
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        track_file.write("[session {}][epoch {}][iter {}/{}] loss: {}, lr: {}, time: {}\n".format(args.session, epoch, step, iters_per_epoch, loss_temp, lr,datetime.datetime.now().time()))
        track_file.write("\t\t\tfg/bg=({}/{}), time cost: {}\n".format(fg_cnt, bg_cnt, end-start))
        track_file.write("\t\t\trpn_cls: {}, rpn_box: {}, rcnn_cls: {}, rcnn_box {}\n\n".format(loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()

    
    save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name)) # this is executed after every epoch. The model is stored after every epoch
    track_file.write('save model: {}\n\n'.format(save_name))
  if args.use_tfboard:		 # Some exception occurs here
    logger.close()
  track_file.close()
