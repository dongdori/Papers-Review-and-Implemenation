import torch
import torch.nn as nn

# Generator
class Upblock(nn.Module):
    def __init__(self, factor):
        super(Upblock, self).__init__()
        self.convtranspose = nn.ConvTranspose2d(in_channels = n_gf * (2**factor),
                                                out_channels = n_gf*(2**(factor-1)),
                                                kernel_size = (4,4),
                                                stride = 2,
                                                padding = 1,
                                                bias = False)
        self.bn = nn.BatchNorm2d(n_gf * 2**(factor-1))
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
    def forward(self, input):
        x = self.convtranspose(input)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # upsampling
            nn.ConvTranspose2d(n_z, n_gf*8, (4,4), 1, 0, bias = False),
            nn.BatchNorm2d(n_gf*8),
            nn.ReLU(True),
            Upblock(factor = 3),
            Upblock(factor = 2),
            Upblock(factor = 1),
            nn.ConvTranspose2d(n_gf, 3, 4, 2, 1),
            # final output
            nn.Tanh()
            )
    def forward(self, input):
        return self.main(input)
      
# Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, n_df, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(n_df, n_df * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_df * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(n_df * 2, n_df * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_df * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(n_df * 4, n_df * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_df * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(n_df * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
