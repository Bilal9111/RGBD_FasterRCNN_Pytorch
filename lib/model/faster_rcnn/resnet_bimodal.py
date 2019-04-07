from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn_bimodal import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb














"""
########################################################################################################
This is the secondary resnet file responsible for creating the RESNET architecture.
It supports the RGBD (bimodal) training and testing as well
########################################################################################################
"""




















__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}







########################################################################################################
#
# The two classes in this section are completely useless to us as they are only used in resnet18 and resnet34
#
# This class is completely useless to us 
def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

# This class is completely useless to us
class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out
########################################################################################################








# This is basically just a neural network module used as a lower level block in our neural network design
class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    #print(inplanes, planes)
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

# This is basically just a neural network module used as a lower level block in our neural network design
class Reverse_Bottleneck2(nn.Module):
  expansion = 2

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Reverse_Bottleneck2, self).__init__()
    #print(inplanes, planes)
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, int(planes /2), kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(int(planes /2))
    self.relu = nn.ReLU(inplace=True)
    self.stride = stride
    if downsample == True:
     downsample = nn.Sequential(
         nn.Conv2d(inplanes, int(planes / 2),
               kernel_size=1, stride=stride, bias=False),
         nn.BatchNorm2d(int(planes / 2)))
    self.downsample = downsample
    

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)
      
    out += residual
    out = self.relu(out)

    return out



# This is basically just a neural network module used as a lower level block in our neural network design
class Reverse_Bottleneck(nn.Module):
  expansion = 2

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Reverse_Bottleneck, self).__init__()
    #print(inplanes, planes)
    self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,padding=1, bias=False) # change
    self.bn1 = nn.BatchNorm2d(inplanes)
    
    self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    
    self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,padding=1,bias=False)
    self.bn3 = nn.BatchNorm2d(planes)
    
    self.relu = nn.ReLU(inplace=True)
    self.stride = stride

  def forward(self, x):
  	 residual = x
  	 out = self.conv1(x)
  	 out = self.bn1(out)
  	 out = self.relu(out)
  	 
  	 out = self.conv2(out)  	 
  	 out = self.bn2(out)
  	 out = self.relu(out)
  	 
  	 residual = self.conv3(residual)
  	 residual = self.bn3(residual)
  	 
  	 out += residual
  	 out = self.relu(out)
  	 return out



# This is the higher level implementation of the our resnet101 architecture. Over here we use the Bottleneck class, above, to define the middle layers in the neural network
class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
     
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    #self.avgpool = nn.AvgPool2d(7)
    #self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x


















#######################################################################################################
#
# These are the set of functions which are used by the main resnet class below to initialise the proper architecture
#
# This function is useless to us
def resnet18(pretrained=False): 
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model

# This function is useless to us
def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3]) # Here we initialise the ResNet class with the Bottleneck structure as the lower level block. The list used as a parameter contains the number of layers which need to be constructed in the ResNet architecture.
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model
  


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model



#######################################################################################################




















# This is the main class which makes everything happen
class resnet(_fasterRCNN): # The _fasterRCNN is basically means that resnet class is a derived class which uses _fasterRCNN as a base class
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic) # Initialising the base class here

  def _init_modules(self):
    resnet = resnet50() # Initialise the proper ResNet architecture
    resnet_d_p = resnet50()
    
    
    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      state_dict_rgbd = state_dict
      resnet.load_state_dict({k:v for k,v in state_dict_rgbd.items() if k in resnet.state_dict()},strict=False)
      resnet_d_p.load_state_dict({k:v for k,v in state_dict_rgbd.items() if k in resnet_d_p.state_dict()},strict=False)
      


    # Build resnet.
    """
    NOTE: The resnet101() function is used here which uses the Bottleneck and the ResNet class. The resnet101() returns a neural network but the network is never used directly. 
    It is used to create the RCNN_base and RCNN_top which are then used for rest of the computations.
    """
    self.RCNN_base_a = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3) # Defining the RCNN_base which is later used by the functions in the base class _fasterRCNN
        
    self.RCNN_base_b = nn.Sequential(resnet_d_p.conv1, resnet_d_p.bn1,resnet_d_p.relu,
      resnet_d_p.maxpool,resnet_d_p.layer1,resnet_d_p.layer2,resnet_d_p.layer3) # Defining the RCNN_base which is later used by the functions in the base class _fasterRCNN
    
    
    
    
    self.RCNN_top_a = nn.Sequential(resnet.layer4)  # Defining the RCNN_top which is later used by the functions in the base class _fasterRCNN
    self.RCNN_top_b = nn.Sequential(resnet_d_p.layer4)  # Defining the RCNN_top which is later used by the functions in the base class _fasterRCNN
    
    
    

    
    
    """ Upsampling goes here: """
    #self.conv_upsample = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
    self.rev_bottle1 = Reverse_Bottleneck2(2048,1024,downsample=True)    
    self.rev_bottle4 = Reverse_Bottleneck2(512,1024)
    self.rev_bottle5 = Reverse_Bottleneck2(512,1024)
    
    self.upsample1 = nn.Sequential(self.rev_bottle1,self.rev_bottle4,self.rev_bottle5)#self.rev_bottle6)#,self.rev_bottle7)
    #self.rev_bottle8 = Reverse_Bottleneck2(512,1024)
    self.rev_bottle9 = Reverse_Bottleneck2(512,512,downsample=True)
    self.rev_bottle10 = Reverse_Bottleneck2(256,512)
    self.upsample2 = nn.Sequential(self.rev_bottle9,self.rev_bottle10)
    
    
    
    
    
    
    for m in self.upsample1.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        
    for m in self.upsample2.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    
    
    
    
    
    
    
    
    
    
    """ Fully connected layers go here: """
    self.relu_fc = nn.ReLU(inplace=True)
    self.fc2 = nn.Linear(256, 128)    
    self.fc3 = nn.Linear(128, 128)        
    self.fc_layer = nn.Sequential(self.fc2,self.relu_fc,self.fc3,self.relu_fc)
    
    
    
    
    
    
    print("The number of classes in n_classes is: %d" %(self.n_classes))
    
    self.RCNN_cls_score = nn.Linear(128, self.n_classes) # not to be changed
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(128, 4) # not to be changed
    else:
      self.RCNN_bbox_pred = nn.Linear(128, 4 * self.n_classes) # not to be changed
    
    
    
    
    
    
    
    """ Fix blocks """
    # RGB path
    for p in self.RCNN_base_a[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base_a[1].parameters(): p.requires_grad=False
    # Depth path    
    for p in self.RCNN_base_b[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base_b[1].parameters(): p.requires_grad=False



    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base_a[6].parameters(): p.requires_grad=False
      for p in self.RCNN_base_b[6].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base_a[5].parameters(): p.requires_grad=False
      for p in self.RCNN_base_b[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base_a[4].parameters(): p.requires_grad=False
      for p in self.RCNN_base_b[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base_a.apply(set_bn_fix)
    self.RCNN_top_a.apply(set_bn_fix)
    self.RCNN_base_b.apply(set_bn_fix)
    self.RCNN_top_b.apply(set_bn_fix)










  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base_a.eval()
      self.RCNN_base_a[5].train()
      self.RCNN_base_a[6].train()
      self.RCNN_base_b.eval()
      self.RCNN_base_b[5].train()
      self.RCNN_base_b[6].train()
      
      self.upsample1.train()
      self.upsample2.train()
      self.fc_layer.train()
      
      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base_a.apply(set_bn_eval)
      self.RCNN_top_a.apply(set_bn_eval)
      self.RCNN_base_b.apply(set_bn_eval)
      self.RCNN_top_b.apply(set_bn_eval)
     
      
      
  def _head_to_tail(self, pool5,pool6):
    fc7 = self.RCNN_top_a(pool5)
    fc8 = self.RCNN_top_b(pool6)
    fc = fc7+fc8
    return fc 