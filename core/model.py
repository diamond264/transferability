import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


# class DownBlock(nn.Module):
#     def __init__(self, c_dim, dim_in, extraction_rate):
#         super(DownBlock, self).__init__()
#         self.c_dim = c_dim
#         dim_out = dim_in*c_dim//extraction_rate
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, kernel_size=(3, c_dim), stride=(1, 1), padding=(1, 0), bias=False),
#             nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
#             # FIXME: Not sure whether LeakyRelu has to be here
#             nn.LeakyReLU(inplace=True))
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
#             nn.BatchNorm2d(dim_in, affine=True, track_running_stats=True),
#             # FIXME: Not sure whether LeakyRelu has to be here
#             nn.LeakyReLU(inplace=True))
#
#         dim_out = dim_out/c_dim
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(dim_out+dim_in, 2*dim_in, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
#             nn.BatchNorm2d(2*dim_in, affine=True, track_running_stats=True),
#             # FIXME: Not sure whether LeakyRelu has to be here
#             nn.LeakyReLU(inplace=True))
#
#         self.down = nn.AvgPool2d(kernel_size=(2, 1))
#
#     def forward(self, x):
#         conv1_res = self.conv1(x)
#         conv2_res = self.conv2(x)
#         batch_size, f_dim, s_dim, conv_dim = conv2_res.shape
#         conv2_res = conv2_res.view([batch_size, f_dim, self.c_dim, -1])
#         x = torch.cat([conv1_res, conv2_res], dim=1)
#         x = self.conv3(x)
#         x = self.down(x)
#         return x

class DownBlock(nn.Module):
    def __init__(self, c_dim, dim_in, extraction_rate):
        super(DownBlock, self).__init__()
        self.c_dim = c_dim
        dim_out = dim_in
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            # FIXME: Not sure whether LeakyRelu has to be here
            nn.LeakyReLU(inplace=True))
        
        dim_out = dim_in * 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            # FIXME: Not sure whether LeakyRelu has to be here
            nn.LeakyReLU(inplace=True))

        self.down = nn.MaxPool2d(kernel_size=(2, 1), return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, indices = self.down(x)
        return x, indices


class UpBlock(nn.Module):
    def __init__(self, dim_in, extraction_rate):
        super(UpBlock, self).__init__()
        
        self.up = nn.MaxUnpool2d(kernel_size=(2, 1), padding=(0, 0))

        dim_out = dim_in
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            # FIXME: Not sure whether LeakyRelu has to be here
            nn.LeakyReLU(inplace=True))
        
        dim_in = dim_out
        dim_out = dim_in//2
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            # FIXME: Not sure whether LeakyRelu has to be here
            nn.LeakyReLU(inplace=True))

    def forward(self, x, indices):
        x = self.up(x, indices)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Generator(nn.Module):
    """Generator network."""
    # FIXME: Default parameters should be fixed
    # def __init__(self, conv_dim=64, c_dim=5, repeat_num=3, extraction_rate=3):
    def __init__(self, conv_dim=32, c_dim=4, repeat_num=3, extraction_rate=2):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1+2*c_dim, conv_dim, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), bias=False))
        layers.append(nn.BatchNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(inplace=True))
        self.downconv = nn.Sequential(*layers)

        # Down-sampling layers.
        curr_dim = conv_dim
        self.down1 = DownBlock(c_dim=c_dim, dim_in=curr_dim, extraction_rate=extraction_rate)
        curr_dim = curr_dim * 2
        self.down2 = DownBlock(c_dim=c_dim, dim_in=curr_dim, extraction_rate=extraction_rate)
        curr_dim = curr_dim * 2

        # for i in range(2):
        #     layers.append(DownBlock(c_dim=c_dim, dim_in=curr_dim, extraction_rate=extraction_rate))
        #     curr_dim = curr_dim * 2

        # Bottleneck layers.
        bottleneck_layers = []
        for i in range(repeat_num):
            bottleneck_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.residualconv = nn.Sequential(*bottleneck_layers)

        # Up-sampling layers
        self.up1 = UpBlock(dim_in=curr_dim, extraction_rate=extraction_rate)
        curr_dim = curr_dim // 2
        self.up2 = UpBlock(dim_in=curr_dim, extraction_rate=extraction_rate)
        curr_dim = curr_dim // 2

        # for i in range(2):
        #     layers.append(UpBlock(dim_in=curr_dim, extraction_rate=extraction_rate))
        #     curr_dim = curr_dim // 2

        last_layers = []
        last_layers.append(nn.Conv2d(curr_dim, 1, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), bias=False))
        last_layers.append(nn.Tanh())
        self.lastconv = nn.Sequential(*last_layers)

    # src_c: source domain (one-hot vector)
    # tgt_c: target domain (one-hot vector)
    def forward(self, x, src_c, tgt_c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        src_c = src_c.view(src_c.size(0), src_c.size(1), 1, 1)
        src_c = src_c.repeat(1, 1, x.size(2), x.size(3))
        tgt_c = tgt_c.view(tgt_c.size(0), tgt_c.size(1), 1, 1)
        tgt_c = tgt_c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, src_c, tgt_c], dim=1)

        x = self.downconv(x)
        x, indices1 = self.down1(x)
        x, indices2 = self.down2(x)
        
        x = self.residualconv(x)
        
        x = self.up1(x, indices2)
        x = self.up2(x, indices1)
        x = self.lastconv(x)
        return x


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    # def __init__(self, win_len=120, channel_dim=6, conv_dim=64, style_dim=5, context_dim=20):
    def __init__(self, win_len=60, channel_dim=6, conv_dim=32, style_dim=4, context_dim=4):
        super(Discriminator, self).__init__()
        self.win_len = win_len
        self.channel_dim = channel_dim
        self.conv_dim = conv_dim

        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=(5, 3), stride=(1, 3), padding=(2, 0)))
        layers.append(nn.BatchNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(0.01))

        layers.append(nn.AvgPool2d(kernel_size=(3, 1)))

        layers.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0)))
        layers.append(nn.BatchNorm2d(conv_dim*2, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(0.01))

        layers.append(nn.AvgPool2d(kernel_size=(2, 1)))
        self.main_cv = nn.Sequential(*layers)

        layers_fc0 = []
        layers_fc0.append(nn.Linear(win_len//6 * channel_dim//3 * conv_dim*2, 1024))
        layers_fc0.append(nn.ReLU(True))
        # layers_fc0.append(nn.Linear(2048, 1024))
        # layers_fc0.append(nn.ReLU(True))
        layers_fc0.append(nn.Linear(1024, 500))
        layers_fc0.append(nn.ReLU(True))

        self.main_fc = nn.Sequential(*layers_fc0)

        layers_fc1 = []
        layers_fc1.append(nn.Linear(500, 256))
        # layers_fc1.append(F.relu())
        layers.append(nn.ReLU(True))
        layers_fc1.append(nn.Linear(256, 128))
        # layers_fc1.append(F.relu())
        layers.append(nn.ReLU(True))
        layers_fc1.append(nn.Linear(128, 64))
        # layers_fc1.append(F.relu())
        layers.append(nn.ReLU(True))
        layers_fc1.append(nn.Linear(64, 1))
        self.fc1 = nn.Sequential(*layers_fc1)

        layers_fc2 = []
        layers_fc2.append(nn.Linear(500, 256))
        # layers_fc2.append(F.relu())
        layers.append(nn.ReLU(True))
        layers_fc2.append(nn.Linear(256, 128))
        # layers_fc2.append(F.relu())
        layers.append(nn.ReLU(True))
        layers_fc2.append(nn.Linear(128, 64))
        # layers_fc2.append(F.relu())
        layers.append(nn.ReLU(True))
        layers_fc2.append(nn.Linear(64, style_dim))
        self.fc2 = nn.Sequential(*layers_fc2)

        layers_fc3 = []
        layers_fc3.append(nn.Linear(500, 256))
        # layers_fc3.append(F.relu())
        layers.append(nn.ReLU(True))
        layers_fc3.append(nn.Linear(256, 128))
        # layers_fc3.append(F.relu())
        layers.append(nn.ReLU(True))
        layers_fc3.append(nn.Linear(128, 64))
        # layers_fc3.append(F.relu())
        layers.append(nn.ReLU(True))
        layers_fc3.append(nn.Linear(64, context_dim))
        self.fc3 = nn.Sequential(*layers_fc3)
        
    def forward(self, x):
        print(x)
        h = self.main_cv(x)
        h = h.view(-1, self.win_len//6 * self.channel_dim//3 * self.conv_dim*2)
        h = self.main_fc(h)
        # label for real/fake
        out_fake = self.fc1(h)
        # label for style
        out_style = self.fc2(h)
        # label for context
        out_context = self.fc3(h)
        return out_fake, out_style, out_context
