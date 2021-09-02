import torch
import torch.nn as nn

class Trainer:
    def __init__(self, num_epochs, dataloader):
        self.num_epochs = num_epochs
        self.dataloader = dataloader
    def train(self):
        for epoch in range(self.num_epochs):
            print('{}th Epoch'.format(epoch + 1))
            # for each batch in dataset
            for i, data in enumerate(self.dataloader):
                # 1. Update Discriminator
                D.zero_grad()
                ## train with real images
                data = data.to(device)
                label = torch.full((batch_size,), 1.0, dtype = torch.float, device = device)
                output = D(data).view(-1)
                errD_real = loss_fn(output, label)
                errD_real.backward()
                D_x = output.mean().item() # average value of output of D given real images

                ## train with fake images
                noise = torch.randn(batch_size, n_z, 1, 1, device = device)
                fake = G(noise)
                label.fill_(0.0)
                output = D(fake.detach()).view(-1)
                errD_fake = loss_fn(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item() # average value of output of D given fake images

                optim_D.step()

                # 2. Update Generator
                G.zero_grad()
                label.fill_(1.0) # in perspective of generator,
                                # goal(label) is to make D to return 1.0, given fake images
                output = D(fake).view(-1)
                errG = loss_fn(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()

                optim_G.step()

                if i % 100 == 0: 
                    print('{}th epoch {}% complete | D(x) : {}, D(G(z)) : {}'.format(
                        epoch+1, round(i/len(dataloader)*100,1), round(D_x, 2), round(D_G_z2, 2)))
