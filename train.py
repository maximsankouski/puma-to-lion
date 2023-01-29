import os
import config
from dataloader import get_datasets
from discriminator import Discriminator
from generator import Generator
from utils import denorm
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image


def train(disc_p, disc_l, gen_p, gen_l, dl, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, sched_disc, sched_gen, eps):
    sample_dir_3 = 'fake_pumas'
    os.makedirs(sample_dir_3, exist_ok=True)
    sample_dir_3 = 'fake_lions'
    os.makedirs(sample_dir_3, exist_ok=True)

    losses_d = []
    losses_g_p = []
    losses_g_l = []
    losses_full = []

    for epoch in range(eps):
        real_lions = 0
        fake_lions = 0
        loss_d_per_epoch = []
        loss_g_p_per_epoch = []
        loss_g_l_per_epoch = []
        full_loss_per_epoch = []

        loop = tqdm(dl)

        for idx, (puma, lion) in enumerate(loop):
            lion = lion.to(config.DEVICE)
            puma = puma.to(config.DEVICE)

            # Train discriminators
            with torch.cuda.amp.autocast():
                # Gen fake puma
                fake_puma = gen_p(lion)
                d_p_real = disc_p(puma)
                d_p_fake = disc_p(fake_puma.detach())
                d_p_real_loss = mse(d_p_real, torch.ones_like(d_p_real))
                d_p_fake_loss = mse(d_p_fake, torch.zeros_like(d_p_fake))
                d_p_loss = d_p_real_loss + d_p_fake_loss

                # Gen fake lion
                fake_lion = gen_l(puma)
                d_l_real = disc_l(lion)
                d_l_fake = disc_l(fake_lion.detach())
                real_lions += d_l_real.mean().item()
                fake_lions += d_l_fake.mean().item()
                d_l_real_loss = mse(d_l_real, torch.ones_like(d_l_real))
                d_l_fake_loss = mse(d_l_fake, torch.zeros_like(d_l_fake))
                d_l_loss = d_l_real_loss + d_l_fake_loss

                d_loss = (d_p_loss + d_l_loss) / 2

            opt_disc.zero_grad()
            d_scaler.scale(d_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
            sched_disc.step()

            loss_d_per_epoch.append(d_loss.item())

            # Train generators
            with torch.cuda.amp.autocast():
                # Adversarial losses
                d_p_fake = disc_p(fake_puma)
                d_l_fake = disc_l(fake_lion)
                loss_g_p = mse(d_p_fake, torch.ones_like(d_p_fake))
                loss_g_l = mse(d_l_fake, torch.ones_like(d_l_fake))

                loss_g_p_per_epoch.append(loss_g_p.item())
                loss_g_l_per_epoch.append(loss_g_l.item())

                # Cycle losses
                cycle_lion = gen_l(fake_puma)
                cycle_puma = gen_p(fake_lion)
                cycle_lion_loss = l1(lion, cycle_lion)
                cycle_puma_loss = l1(puma, cycle_puma)

                # Full loss
                full_loss = (
                    loss_g_l
                    + loss_g_p
                    + cycle_lion_loss * config.CYCLE_LOSS_LAMBDA
                    + cycle_puma_loss * config.CYCLE_LOSS_LAMBDA
                )

                full_loss_per_epoch.append(full_loss.item())

            opt_gen.zero_grad()
            g_scaler.scale(full_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
            sched_gen.step()

            if idx == 950:
                save_image(denorm(fake_puma), f"fake_pumas/puma{epoch}.jpg")
                save_image(denorm(fake_lion), f"fake_lions/lion{epoch}.jpg")

            loop.set_postfix(lion_real=real_lions / (idx + 1), lion_fake=fake_lions / (idx + 1))

        # Record losses
        losses_d.append(np.mean(loss_d_per_epoch))
        losses_g_p.append(np.mean(loss_g_p_per_epoch))
        losses_g_l.append(np.mean(loss_g_l_per_epoch))
        losses_full.append(np.mean(full_loss_per_epoch))

        # Log losses
        print("Epoch [{}/{}], loss_d: {:.4f}, loss_g_p: {:.4f}, loss_g_l: {:.4f}, losses_full: {:.4f}".format(
            epoch + 1,
            eps,
            losses_d[-1],
            losses_g_p[-1],
            losses_g_l[-1],
            losses_full[-1])
        )

    return losses_d, losses_g_p, losses_g_l, losses_full


def main():
    pumalion_train, pumalion_test = get_datasets(config.PUMA_ROOT, config.LION_ROOT)
    dataloader = DataLoader(pumalion_train, batch_size=config.BATCH_SIZE, shuffle=True)
    # dataloader_test = DataLoader(pumalion_test, batch_size=config.BATCH_SIZE, shuffle=False)

    disc_p = Discriminator().to(config.DEVICE)
    disc_l = Discriminator().to(config.DEVICE)
    gen_p = Generator().to(config.DEVICE)
    gen_l = Generator().to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_p.parameters()) + list(disc_l.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_l.parameters()) + list(gen_p.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    sched_disc = optim.lr_scheduler.MultiStepLR(
        opt_disc,
        milestones=[x for x in range(int(config.EPOCHS/2), config.EPOCHS)],
        gamma=0.98
    )
    sched_gen = optim.lr_scheduler.MultiStepLR(
        opt_gen,
        milestones=[x for x in range(int(config.EPOCHS/2), config.EPOCHS)],
        gamma=0.98
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    history = train(
        disc_p,
        disc_l,
        gen_p,
        gen_l,
        dataloader,
        opt_disc,
        opt_gen,
        l1,
        mse,
        d_scaler,
        g_scaler,
        sched_disc,
        sched_gen,
        config.EPOCHS
    )

    losses_d, losses_g_p, losses_g_l, losses_full = history
    return losses_d, losses_g_p, losses_g_l, losses_full


def get_generated_images_test(dl, gen_p, gen_l):
    images_fake_puma = []
    images_fake_lion = []
    for puma, lion in dl:
        images_fake_lion.append((gen_l(puma)).cpu().detach())
        images_fake_puma.append((gen_p(lion)).cpu().detach())
    images_fake_puma = torch.cat(images_fake_puma)
    images_fake_lion = torch.cat(images_fake_lion)
    return images_fake_puma, images_fake_lion


if __name__ == "__main__":
    main()
