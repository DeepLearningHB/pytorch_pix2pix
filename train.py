import torch
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 3"
from dataset import *
from optimizer import *
from model import *
from torch.autograd import Variable
def main():
    total_epoch = 300
    learning_rate = 1e-4
    weight_decay = 1e-5
    adam_betas = (0.5, 0.999)
    learning_step = 15
    early_stopping = 5
    l1_lambda = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_saved_path = './weights'
    os.makedirs(weight_saved_path, exist_ok=True)

    train_data = Pix2PixDataset(True)
    valid_data = Pix2PixDataset(False)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)

    G = Generator().to(device)
    D = Discriminator().to(device)

    G = nn.DataParallel(G, output_device=0)
    D = nn.DataParallel(D, output_device=0)

    loss_bce = nn.BCELoss()
    l1_loss = nn.L1Loss()

    optim_G = AdamW_GCC2(G.parameters(), lr=learning_rate, betas=adam_betas, weight_decay=weight_decay)
    optim_D = AdamW_GCC2(D.parameters(), lr=learning_rate, betas=adam_betas, weight_decay=weight_decay)



    for ep in range(total_epoch):
        D_losses = []
        G_losses = []

        avg_G_loss = 0
        avg_D_loss = 0
        count = 0
        for idx, batch in train_loader:
            batch_x = batch['data_x'].to(device)
            batch_y = batch['data_y'].to(device)

            # training Discriminator
            D.zero_grad()
            out_D = D(batch_x, batch_y).squeeze()
            D_real_loss = loss_bce(out_D, torch.ones(out_D.size()).to(device))

            out_G = G(batch_x)
            out_D = D(batch_x, out_G).squeeze()
            D_fake_loss = loss_bce(out_D, torch.zeros(out_D.size()).to(device))

            D_total_loss = 0.5 * (D_real_loss + D_fake_loss)
            D_losses.append(D_total_loss.item())
            avg_D_loss += D_total_loss.item()
            D_total_loss.backward()
            optim_D.step()


            # training generator
            G.zero_grad()

            out_G = G(batch_x)
            out_D = D(batch_x, out_G).squeeze()

            G_loss = loss_bce(out_D, torch.ones(out_D) + l1_lambda * l1_loss(out_G, batch_y))
            G_losses.append(G_loss.item())
            avg_G_loss += G_loss.item()
            G_loss.backward()
            optim_G.step()
            count += 1


        avg_D_loss /= count
        avg_G_loss /= count

        print("[Training] Generator Loss: %.5f Discriminator Loss: %.5f" % (avg_G_loss, avg_D_loss))









if __name__ == "__main__":
    main()