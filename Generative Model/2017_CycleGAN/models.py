################# Generator ########################
class Generator(nn.Module):
    def __init__(self, n_residual_blocks = 6):
        super(Generator, self).__init__()
        # downsampling
        layers = [nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace = True),
                 DownBlock(64, 128),
                 DownBlock(128, 256)]
        # residual layers
        for i in range(n_residual_blocks):
            layers += [ResidualBlock(256)]
        # upsampling
        upblocks_layer = [UpBlock(256, 128),
                    UpBlock(128, 64),
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(64, 3, 7),
                    nn.Tanh()]
        layers += upblocks_layer
        self.model = nn.Sequential(*layers)
    def forward(self, input):
        return self.model(input) 
    
        
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride = 2, padding = 1),
            nn.InstanceNorm2d(out_channels), 
            nn.ReLU(inplace = True)
            )
    def forward(self, input):
        return self.block(input)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace = True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, input):
        return input + self.block(input)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 2, padding = 1, output_padding = 1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )
    def forward(self, input):
        return self.block(input)


############## Discriminator ################
class Discriminator(nn.Module):
    def __init__(self, channels = [3, 64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.channels = channels
        # set layers
        layers = []
        for i in range(len(channels)-1):
            layers += self.downsample(channels[i], channels[i+1])
        # fully convolutional classfication layer
        layers += [nn.Conv2d(self.channels[-1], 1, 4, padding = 1)]
        # set model
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        x = self.model(input)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

    def downsample(self, input_channel, output_channel):
        if input_channel == 3:
            layers = [nn.Conv2d(input_channel, output_channel, kernel_size = 4, stride = 2, padding = 1),
                      nn.LeakyReLU(0.2, inplace = True)]
        elif output_channel == self.channels[-1]:
            layers = [nn.Conv2d(input_channel, output_channel, kernel_size = 4, padding = 1),
                        nn.InstanceNorm2d(output_channel),
                        nn.LeakyReLU(0.2, inplace = True)]
        else:
            layers = [nn.Conv2d(input_channel, output_channel, kernel_size = 4, stride = 2, padding = 1),
                        nn.InstanceNorm2d(output_channel),
                        nn.LeakyReLU(0.2, inplace = True)]
        return layers
