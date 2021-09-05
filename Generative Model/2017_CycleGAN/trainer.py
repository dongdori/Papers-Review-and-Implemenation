# Trainer
class Trainer:
    def __init__(self, n_epochs = 200, cur_epoch = 0, lr_schedulers = None, optimizers = None):
        self.n_epochs = n_epochs
        self.cur_epoch = cur_epoch
        self.lr_schedule_G, self.lr_schedule_D_X, self.lr_schedule_D_Y = lr_schedulers
        self.optimizer_G, self.optimizer_D_X, self.optimizer_D_Y = optimizers
        self.cycle_criterion = nn.L1Loss()
        self.adv_criterion = nn.MSELoss()
    
    def train(self, dataloader):
        for epoch in range(self.cur_epoch, self.n_epochs):
            print('{}th EPOCH \n'.format(epoch))
            for i, batch in enumerate(dataloader):
                img_s1, img_s2 = batch['s1'].to(device), batch['s2'].to(device)
                # 1. update Generators (G and F)
                optimizer_G.zero_grad()
                ## compute gan loss
                fake_s2 = netG(img_s1)
                pred_dy= netD_Y(fake_s2)
                fake_s1 = netF(img_s2)
                pred_dx = netD_X(fake_s1)
                label_G = torch.Tensor(dataloader.batch_size).fill_(1.0).to(device)
                label_F = torch.Tensor(dataloader.batch_size).fill_(1.0).to(device)
                gan_loss = nn.MSELoss()(pred_dy, label_G) + nn.MSELoss()(pred_dx, label_F)
                ## compute cyclic loss
                rec_s1 = netF(netG(img_s1))
                rec_s2 = netG(netF(img_s2))
                cycle_loss = nn.L1Loss()(rec_s1, img_s1) + nn.L1Loss()(rec_s2, img_s2)
                # optimizer step for Generators
                G_loss = gan_loss + 10*cycle_loss
                G_loss.backward(retain_graph = True)
                self.optimizer_G.step()
                
                # 2. update Discriminator D_X
                optimizer_D_X.zero_grad()
                label_D_X_real = torch.Tensor(dataloader.batch_size).fill_(1.0).to(device)
                label_D_X_fake = torch.Tensor(dataloader.batch_size).fill_(0.0).to(device)
                D_X_real = nn.MSELoss()(netD_X(img_s1), label_D_X_real)
                D_X_fake = nn.MSELoss()(netD_X(fake_s1.detach()), label_D_X_fake)
                adv_loss_D_X = D_X_real + D_X_fake
                adv_loss_D_X.backward(retain_graph = True)
                self.optimizer_D_X.step()
                
                # 3. update Discriminator D_Y 
                optimizer_D_Y.zero_grad()
                label_D_Y_real = torch.Tensor(dataloader.batch_size).fill_(1.0).to(device)
                label_D_Y_fake = torch.Tensor(dataloader.batch_size).fill_(0.0).to(device)
                D_Y_real = nn.MSELoss()(netD_Y(img_s2), label_D_Y_real)
                D_Y_fake = nn.MSELoss()(netD_Y(fake_s2.detach()), label_D_Y_fake)
                adv_loss_D_Y = D_Y_real + D_Y_fake
                adv_loss_D_Y.backward(retain_graph = True)
                self.optimizer_D_Y.step()
                if i % 50 == 0:
                    print('[{} / {}] : D_Y(fake) : {} | D_Y(real) : {} | cyc_loss : {} \n'.format(
                        i, len(dataloader), D_Y_fake.item(), D_Y_real.item(), cycle_loss))
            # learning scheduler step progress
            self.lr_schedule_G.step()
            self.lr_schedule_D_X.step()
            self.lr_schedule_D_Y.step()
            # increase cur_epoch instance by 1
            self.cur_epoch += 1
            # save updated parameters of Generators and Discriminators
            torch.save(netG.state_dict(), './output/netG.pth')
            torch.save(netF.state_dict(), './output/netF.pth')
            torch.save(netD_X.state_dict(), './output/netD_X.pth')
            torch.save(netD_Y.state_dict(), './output/netD_Y.pth')
