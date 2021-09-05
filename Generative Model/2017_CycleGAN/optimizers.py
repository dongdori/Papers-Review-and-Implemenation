# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG.parameters(), netF.parameters()), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_X = torch.optim.Adam(netD_X.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_Y = torch.optim.Adam(netD_Y.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Learning rate scheduler with Linear Decay
class LinearDecay():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LinearDecay(n_epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X,
                                                     lr_lambda=LinearDecay(n_epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y,
                                                     lr_lambda=LinearDecay(n_epochs, start_epoch, decay_epoch).step)
