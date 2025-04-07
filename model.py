import torch
from torch import nn
from torchsummary import summary
  
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

if __name__ == "__main__":
    
    n_mels = 16
    n_samples = 3 * 22050
    n_frames = 1 + (n_samples - 2048) // 512
    print(f'n_frames : {n_frames}, n_samples : {n_samples}, n_mels : {n_mels}')
    model_RecNet = MusicRecNet(n_mels=n_mels, n_frames=n_frames, filters=[32, 64, 128])
    summary(model_RecNet, (1, n_mels, n_frames))

    print(f'----- MusicRecNet model -----')
    x = torch.randn(32, 1, n_mels, n_frames)
    y = model_RecNet(x)
    print(f'Input : {x.size()}')
    print(f'Output : {y.size()}')     