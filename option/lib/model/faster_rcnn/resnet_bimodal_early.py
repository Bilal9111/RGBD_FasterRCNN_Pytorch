from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn_bimodal_early import _fasterRCNN

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




# This is the higher level implementation of the our resnet101 architecture. Over here we use the Bottleneck class, above, to define the middle layers in the neural network
class ResNet_Merge(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
  	

    super(ResNet_Merge, self).__init__()


    # RGB pre-processing here:
    self.inplanes = 64
    self.a_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
    self.a_bn1 = nn.BatchNorm2d(64)
    self.a_relu = nn.ReLU(inplace=True)
    self.a_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
    self.a_layer1 = self._make_layer(block, 64, layers[0]) 
    
    # Depth pre-processing here:
    self.inplanes = 64
    self.b_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
    self.b_bn1 = nn.BatchNorm2d(64)
    self.b_relu = nn.ReLU(inplace=True)
    self.b_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
    self.b_layer1 = self._make_layer(block, 64, layers[0]) 
    
    # Post-processing here: 
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
  model = ResNet_Merge(Bottleneck, [3, 4, 6, 3])
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
class resnet(_fasterRCNN): 
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    _fasterRCNN.__init__(self, classes, class_agnostic) # Initialising the base class here
    self.weight_string = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var','layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.conv3.weight', 'layer1.0.bn3.weight', 'layer1.0.bn3.bias', 'layer1.0.bn3.running_mean', 'layer1.0.bn3.running_var', 'layer1.0.downsample.0.weight', 'layer1.0.downsample.1.weight', 'layer1.0.downsample.1.bias', 'layer1.0.downsample.1.running_mean', 'layer1.0.downsample.1.running_var', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.conv3.weight', 'layer1.1.bn3.weight', 'layer1.1.bn3.bias', 'layer1.1.bn3.running_mean', 'layer1.1.bn3.running_var', 'layer1.2.conv1.weight', 'layer1.2.bn1.weight', 'layer1.2.bn1.bias', 'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.conv2.weight', 'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var', 'layer1.2.conv3.weight', 'layer1.2.bn3.weight', 'layer1.2.bn3.bias', 'layer1.2.bn3.running_mean', 'layer1.2.bn3.running_var']


  def _init_modules(self):
    resnet = resnet50()
    
    
    if self.pretrained == True:
      
      # Loading weights      
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      
      # Preprocessing weights:
      for weight_st in self.weight_string:
      	state_dict['a_'+weight_st] = state_dict[weight_st]
      	state_dict['b_'+weight_st] = state_dict[weight_st]
  
      # Initialising weights
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()},strict=False)
      
      	
      


    # Build resnet.
    """
    NOTE: The resnet101() function is used here which uses the Bottleneck and the ResNet class. The resnet101() returns a neural network but the network is never used directly. 
    It is used to create the RCNN_base and RCNN_top which are then used for rest of the computations.
    """
    # Preprocess 
    self.RCNN_preprocess_a = nn.Sequential(resnet.a_conv1, resnet.a_bn1,resnet.a_relu,
      resnet.a_maxpool,resnet.a_layer1)
      
    self.RCNN_preprocess_b = nn.Sequential(resnet.b_conv1, resnet.b_bn1,resnet.b_relu,
      resnet.b_maxpool,resnet.b_layer1)
    
    # Postprocess
    self.RCNN_postprocess = nn.Sequential(resnet.layer2,resnet.layer3)
     
    # Top layer
    self.RCNN_top = nn.Sequential(resnet.layer4)  # Defining the RCNN_top which is later used by the functions in the base class _fasterRCNN
    
    
    
    
    
    
    print("The number of classes in n_classes is: %d" %(self.n_classes))
    
    self.RCNN_cls_score = nn.Linear(2048, self.n_classes) # not to be changed
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4) # not to be changed
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes) # not to be changed
    
    
    
    
    
    """

    # Fix blocks
    for p in self.RCNN_preprocess_a[0].parameters(): p.requires_grad=False
    for p in self.RCNN_preprocess_a[1].parameters(): p.requires_grad=False
    for p in self.RCNN_preprocess_b[0].parameters(): p.requires_grad=False
    for p in self.RCNN_preprocess_b[1].parameters(): p.requires_grad=False


    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_postprocess[1].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_postprocess[0].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_preprocess_a[4].parameters(): p.requires_grad=False
      for p in self.RCNN_preprocess_b[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False
    self.RCNN_preprocess_a.apply(set_bn_fix)
    self.RCNN_preprocess_b.apply(set_bn_fix)    
    self.RCNN_postprocess.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)
    

   """






  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_preprocess_a.train()
      self.RCNN_preprocess_b.train()
      self.RCNN_postprocess.train()
      
      """
      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_preprocess_a.apply(set_bn_eval)
      self.RCNN_preprocess_b.apply(set_bn_eval)    
      self.RCNN_postprocess.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)
      """
      
      
  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5)
    return fc7.mean(3).mean(2) 