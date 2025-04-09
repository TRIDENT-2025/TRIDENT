from .unimodel_seq import Visual_ResNet, Visual_MobileNet
from .unimodel_audio import Audio_Net, GP_VGG

def get_model(name, n_classes, m=None):
    if name.startswith('resnet'):
        if m is None:
            raise ValueError("Parameter 'm' is required for ResNet models.")
        return Visual_ResNet(name=name, n_classes=n_classes, m=m)
    elif name.startswith('mobilenet'):
        return Visual_MobileNet(n_classes=n_classes)
    elif name.startswith('lenet'):
        return Audio_Net(n_classes=n_classes)
    elif name.startswith('vgg'):
        return GP_VGG(n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
