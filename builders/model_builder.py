from model.SQNet import SQNet
from model.LinkNet import LinkNet
from model.SegNet import SegNet
from model.UNet import UNet
from model.ENet import ENet
from model.ERFNet import ERFNet
from model.CGNet import CGNet
from model.EDANet import EDANet
from model.ESNet import ESNet
from model.ESPNet import ESPNet
from model.LEDNet import LEDNet
#from model.ESPNet_v2.SegmentationModel import EESPNet_Seg
from model.ContextNet import ContextNet
from model.FastSCNN import FastSCNN
from model.DABNet import DABNet
from model.FSSNet import FSSNet
from model.FPENet import FPENet
from model.LETNet import LETNet
#from model.SCTNet import SCTNet
from model.CMTFNet import CMTFNet
from model.LightMUNet import LightMUNet
#from model.UMambaBot import UMambaBot
from model.RS3Mamba import RS3Mamba
from model.Samba import Samba
from model.ECMNet import ECMNet

def build_model(model_name, num_classes):
    if model_name == 'SQNet':
        return SQNet(classes=num_classes)
    elif model_name == 'LinkNet':
        return LinkNet(classes=num_classes)
    elif model_name == 'SegNet':
        return SegNet(classes=num_classes)
    elif model_name == 'UNet':
        return UNet(classes=num_classes)
    elif model_name == 'ENet':
        return ENet(classes=num_classes)
    elif model_name == 'ERFNet':
        return ERFNet(classes=num_classes)
    elif model_name == 'CGNet':
        return CGNet(classes=num_classes)
    elif model_name == 'EDANet':
        return EDANet(classes=num_classes)
    elif model_name == 'ESNet':
        return ESNet(classes=num_classes)
    elif model_name == 'ESPNet':
        return ESPNet(classes=num_classes)
    elif model_name == 'LEDNet':
        return LEDNet(classes=num_classes)
    elif model_name == 'CMTFNet':
        return CMTFNet(classes=num_classes)
    # elif model_name == 'SCTNet':
    #     return SCTNet(classes=num_classes)
    # elif model_name == 'ESPNet_v2':
    #     return EESPNet_Seg(classes=num_classes)
    elif model_name == 'LightMUNet':
        return LightMUNet(out_channels=num_classes)
    # elif model_name == 'UMambaBot':
    #     return UMambaBot(num_classes=num_classes)
    elif model_name == 'ECMNet':
        return ECMNet(classes=num_classes)
    elif model_name == 'RS3Mamba':
        return RS3Mamba(classes=num_classes)
    elif model_name == 'Samba':
        return Samba(num_classes=num_classes)
    elif model_name == 'ContextNet':
        return ContextNet(classes=num_classes)
    elif model_name == 'FastSCNN':
        return FastSCNN(classes=num_classes)
    elif model_name == 'DABNet':
        return DABNet(classes=num_classes)
    elif model_name == 'FSSNet':
        return FSSNet(classes=num_classes)
    elif model_name == 'FPENet':
        return FPENet(classes=num_classes)
    elif model_name == 'LETNet':
        return LETNet(classes=num_classes)
