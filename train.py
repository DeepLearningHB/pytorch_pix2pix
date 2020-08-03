import torch
import torch.nn as nn
from dataset import *
from optimizer import *
from model import *
def main():
    total_epoch = 300
    learning_rate = 1e-4
    weight_decay = 1e-5
    adam_betas = (0.5, 0.999)
    learning_step = 15
    early_stopping = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_saved_path = './weights'
    os.makedirs(weight_saved_path, exist_ok=True)

    train_data = Pix2PixDataset(True)
    valid_data = Pix2PixDataset(False)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)

    G = Generator().to(device)
    D = Discriminator().to(device)

    loss_bce = nn.BCELoss()
    l1_loss = nn.L1Loss()

    optim_G = AdamW_GCC2(G.parameters(), lr=learning_rate, betas=adam_betas, weight_decay=weight_decay)
    optim_D = AdamW_GCC2(D.parameters(), lr=learning_rate, betas=adam_betas, weight_decay=weight_decay)



    for ep in range(total_epoch):
        D_losses = []
        G_losses = []

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







if __name__ == "__main__":
    main()