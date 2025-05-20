import torch
import torchvision.models as models
import timm
from torch import nn
from torchsummary import summary
import torchinfo
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

class MusicRecNet(nn.Module):
    def __init__(self, 
                 n_mels, 
                 n_frames, 
                 filters: list,
                 add_dropout: bool = False):
        super(MusicRecNet, self).__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.filters = filters
        self.add_dropout = add_dropout

        modules = nn.Sequential()

        in_channels = 1

        for i, out_channels in enumerate(filters):
            modules.add_module(f'conv{i}', nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)))
            modules.add_module(f'relu{i}', nn.ReLU())
            modules.add_module(f'pool{i}', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            if add_dropout:
                modules.add_module(f'dropout{i}', nn.Dropout(0.25))
            modules.add_module(f'batchnorm{i}', nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        self.cnn = modules
        self.flatten = nn.Flatten()
        dense_out = 16
        self.dense = nn.Linear(in_channels * (n_mels // 2 ** len(filters)) * (n_frames // 2 ** len(filters)), dense_out)
        self.relu = nn.ReLU()
        if add_dropout:
            self.dropout = nn.Dropout(0.3)
        self.batchnorm = nn.BatchNorm1d(dense_out)
        self.dense_2 = nn.Linear(dense_out, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.relu(x)
        if self.add_dropout:
            x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.dense_2(x)
        x = self.softmax(x)
        return x
    
class AlexNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, fine_tune=True):
        super(AlexNet, self).__init__()

        self.alexnet = models.alexnet(weights="DEFAULT" if pretrained else None)
        if not (fine_tune and pretrained):
            for param in self.alexnet.parameters():
                param.requires_grad = False
        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)  # Change the last layer to output 7 classes

    def forward(self, x):
        x = self.alexnet(x)
        return x

class Wav2Vec2(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, fine_tune=True):
        super(Wav2Vec2, self).__init__()
        self.wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=num_classes)
        if not (fine_tune and pretrained):
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.wav2vec2(x).logits
        return x

    

    
def get_model(model_name, num_classes, pretrained, fine_tune):

    if model_name == 'alexnet':
        model = AlexNet(num_classes=num_classes, pretrained=pretrained, fine_tune=fine_tune)

    elif model_name in timm.list_models():
        
        if model_name.startswith('resnet') or model_name.startswith('resnext'):
            model_name = model_name + '.tv_in1k'
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        if not fine_tune:
            # freeze all layers except the last one
            for index, (name, param) in enumerate(model.named_parameters()):
                if index < len(list(model.named_parameters())) - 2:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    elif model_name == 'wav2vec2':
        model = Wav2Vec2(num_classes=num_classes, pretrained=pretrained, fine_tune=fine_tune)

    else:
        raise ValueError(f"Model {model_name} is not supported.")

    return model

if __name__ == "__main__":

    # model_name = 'vgg13_bn'
    # print(f'----- {model_name} model -----')
    # model = get_model(model_name, num_classes=8, pretrained=True, fine_tune=True).to(device='cuda')


    # summary(model, (3, 227, 227))
    # x = torch.randn(32, 3, 227, 227).to(device='cuda')
    # y = model(x)
    # print(f'Input : {x.size()}')
    # print(f'Output : {y.size()}')

    # model_name = 'resnet50'
    # print(f'----- {model_name} model -----')
    # model = timm.create_model(model_name, pretrained=True)
    # model = model.to(device='cuda')
    # summary(model, (3, 227, 227))
    # x = torch.randn(32, 3, 227, 227).to(device='cuda')
    # y = model(x)
    # y_forward = model.forward_features(x)
    # print(f'Input : {x.size()}')
    # print(f'Output : {y.size()}')
    # print(f'Output forward_features : {y_forward.size()}')

    model_name = 'wav2vec2'
    print(f'----- {model_name} model -----')
    model = get_model(model_name, num_classes=7, pretrained=True, fine_tune=True).to(device='cuda')
    torchinfo.summary(model, (1, 48000))
    x = torch.randn(32, 48000).to(device='cuda')
    y = model(x)

    print(f'Input : {x.size()}')
    print(f'Output : {y.size()}')
