from .resnet18 import ResNet18EDLSoftplus
from .vgg16_bn import VGG16BNEDL
from .wrn28_10 import WideResNet28x10EDL

MODELS = {'resnet18': ResNet18EDLSoftplus, 'vgg16_bn': VGG16BNEDL,
          'wrn28_10': WideResNet28x10EDL}

def create_model(backbone):
    return MODELS[backbone]()
