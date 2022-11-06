import pytorch_influence_functions as ptif
import torchvision
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.transforms.functional as transF

# Return a square image
class SquarePad:
    image_padding = 'constant'

    def __init__(self, image_padding):
        self.image_padding = image_padding

    def __call__(self, image):
        # Reference: https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/10
        if self.image_padding == 'none':
            return image
        elif self.image_padding == 'short_side':
            # Calculate the size of paddings
            max_size = max(image.size)
            pad_left, pad_top = [(max_size - size) // 2 for size in image.size]
            pad_right, pad_bottom = [max_size - (size + pad) for size, pad in zip(image.size, [pad_left, pad_top])]
            padding = (pad_left, pad_top, pad_right, pad_bottom)
            return transF.pad(image, padding, 0, 'constant')

        # TODO: Support more padding modes. E.g. pad both sides to given image size 
        else:
            raise NotImplementedError

# Supplied by the user:

model = torchvision.models.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2],
                                num_classes=9)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
model = model.to("cuda")


means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
transforms = transforms.Compose([
    SquarePad("none"),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(means, stds)
])

trainset = ImageFolder("/Robustar2/dataset/train", transform=transforms)
testset = ImageFolder("/Robustar2/dataset/test", transform=transforms)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=16, shuffle=False, num_workers=8)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=16, shuffle=True, num_workers=8)



ptif.init_logging()

def get_default_config():
    """Returns a default config file"""
    config = {
        'outdir': 'outdir',
        'seed': 42,
        'gpu': 0,
        'dataset': 'CIFAR10',
        'num_classes': 9,
        'test_sample_start_per_class': False,
        'test_sample_num': 1,
        'test_start_index': 0,
        'recursion_depth': 200,
        'r_averaging': 1,
        'scale': None,
        'damp': None,
        'calc_method': 'img_wise',
        'log_filename': None,
    }
    return config

config = get_default_config()

influences, harmful, helpful = ptif.calc_img_wise(config, model, trainloader, testloader)
print(influences, harmful, helpful)


