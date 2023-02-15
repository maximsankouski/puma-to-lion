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


def train(disc_p, disc_l, gen_p, gen_l, dl, opt_disc_p, opt_disc_l, opt_gen, l1, mse, d_scaler, g_scaler, sched_disc_p, sched_disc_l, sched_gen, eps):
    sample_dir_3 = 'fake_pumas'
    os.makedirs(sample_dir_3, exist_ok=True)
    sample_dir_3 = 'fake_lions'
    os.makedirs(sample_dir_3, exist_ok=True)

    losses_d_p = []
    losses_d_l = []
    losses_g_p = []
    losses_g_l = []
    cycle_p_losses = []
    cycle_l_losses = []
    identity_p_losses = []
    identity_l_losses = []
    losses_full = []

    lion_real_scores = []
    lion_fake_scores = []

    for epoch in range(eps):
        real_lions = 0
        fake_lions = 0
        loss_d_p_per_epoch = []
        loss_d_l_per_epoch = []
        loss_g_p_per_epoch = []
        loss_g_l_per_epoch = []
        cycle_p_loss_per_epoch = []
        cycle_l_loss_per_epoch = []
        identity_p_loss_per_epoch = []
        identity_l_loss_per_epoch = []
        full_loss_per_epoch = []

        lion_real_per_epoch = []
        lion_fake_per_epoch = []

        loop = tqdm(dl)

        for idx, (puma, lion) in enumerate(loop):
            lion = lion.to(config.DEVICE)
            puma = puma.to(config.DEVICE)

            # Train puma discriminator
            opt_disc_p.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                # Gen fake puma
                fake_puma = gen_p(lion)
                d_p_real = disc_p(puma)
                d_p_fake = disc_p(fake_puma.detach())
                d_p_real_loss = mse(d_p_real, torch.ones_like(d_p_real))
                d_p_fake_loss = mse(d_p_fake, torch.zeros_like(d_p_fake))
                d_p_loss = d_p_real_loss + d_p_fake_loss

            d_scaler.scale(d_p_loss).backward()
            d_scaler.step(opt_disc_p)
            d_scaler.update()

            loss_d_p_per_epoch.append(d_p_loss.item())

            # Train lion discriminator
            opt_disc_l.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                # Gen fake lion
                fake_lion = gen_l(puma)
                d_l_real = disc_l(lion)
                d_l_fake = disc_l(fake_lion.detach())
                real_lions += d_l_real.mean().item()
                fake_lions += d_l_fake.mean().item()
                d_l_real_loss = mse(d_l_real, torch.ones_like(d_l_real))
                d_l_fake_loss = mse(d_l_fake, torch.zeros_like(d_l_fake))
                d_l_loss = d_l_real_loss + d_l_fake_loss

            d_scaler.scale(d_l_loss).backward()
            d_scaler.step(opt_disc_l)
            d_scaler.update()

            loss_d_l_per_epoch.append(d_l_loss.item())

            # Train generators
            opt_gen.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                # Adversarial losses
                d_p_fake_output = disc_p(fake_puma)
                d_l_fake_output = disc_l(fake_lion)
                loss_g_p = mse(d_p_fake_output, torch.ones_like(d_p_fake_output))
                loss_g_l = mse(d_l_fake_output, torch.ones_like(d_l_fake_output))

                loss_g_p_per_epoch.append(loss_g_p.item())
                loss_g_l_per_epoch.append(loss_g_l.item())

                # Cycle losses
                cycle_lion = gen_l(fake_puma)
                cycle_puma = gen_p(fake_lion)
                cycle_lion_loss = l1(lion, cycle_lion)
                cycle_puma_loss = l1(puma, cycle_puma)

                cycle_p_loss_per_epoch.append(cycle_puma_loss.item())
                cycle_l_loss_per_epoch.append(cycle_lion_loss.item())

                # Identity losses
                identity_puma = gen_p(puma)
                identity_lion = gen_l(lion)
                identity_puma_loss = l1(puma, identity_puma)
                identity_lion_loss = l1(lion, identity_lion)

                identity_p_loss_per_epoch.append(identity_puma_loss.item())
                identity_l_loss_per_epoch.append(identity_lion_loss.item())

                # Full loss
                full_loss = (
                        loss_g_l
                        + loss_g_p
                        + cycle_lion_loss * config.CYCLE_LOSS_LAMBDA
                        + cycle_puma_loss * config.CYCLE_LOSS_LAMBDA
                        + identity_puma_loss * config.IDENTITY_LAMBDA
                        + identity_lion_loss * config.IDENTITY_LAMBDA
                )

                full_loss_per_epoch.append(full_loss.item())

            g_scaler.scale(full_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            # Save images to check the progress in learning
            if idx == 700:
                save_image(denorm(lion), f"fake_pumas/{epoch + 1}lion.jpg")
                save_image(denorm(fake_puma), f"fake_pumas/{epoch + 1}fake_puma.jpg")
                save_image(denorm(puma), f"fake_lions/{epoch + 1}puma.jpg")
                save_image(denorm(fake_lion), f"fake_lions/{epoch + 1}fake_lion.jpg")

            # Show and record possibility to recognize real and fake lions
            lion_real = real_lions / (idx + 1)
            lion_fake = fake_lions / (idx + 1)
            loop.set_postfix(lion_real, lion_fake)

            lion_real_per_epoch.append(lion_real)
            lion_fake_per_epoch.append(lion_fake)

            # Scheduler steps.
            if epoch > (config.EPOCHS - config.SCHEDULER_STEPS):
                sched_disc_p.step()
                sched_disc_l.step()
                sched_gen.step()

            losses_d_p.append(np.mean(loss_d_p_per_epoch))
            losses_d_l.append(np.mean(loss_d_l_per_epoch))
            losses_g_p.append(np.mean(loss_g_p_per_epoch))
            losses_g_l.append(np.mean(loss_g_l_per_epoch))
            cycle_p_losses.append(np.mean(cycle_p_loss_per_epoch))
            cycle_l_losses.append(np.mean(cycle_l_loss_per_epoch))
            identity_p_losses.append(np.mean(identity_p_loss_per_epoch))
            identity_l_losses.append(np.mean(identity_l_loss_per_epoch))
            losses_full.append(np.mean(full_loss_per_epoch))

            lion_real_scores.append(np.mean(lion_real_per_epoch))
            lion_fake_scores.append(np.mean(lion_fake_per_epoch))

            # Log losses
            print(
                "Epoch [{}/{}],"
                "loss_d_p: {:.4f},"
                "loss_d_l: {:.4f},"
                "loss_g_p: {:.4f},"
                "loss_g_l: {:.4f},"
                "cycle_p_losses: {:.4f},"
                "cycle_l_losses: {:.4f},"
                "identity_p_losses: {:.4f},"
                "identity_l_losses: {:.4f},"
                "losses_full: {:.4f}".format(
                    epoch + 1,
                    eps,
                    losses_d_p[-1],
                    losses_d_l[-1],
                    losses_g_p[-1],
                    losses_g_l[-1],
                    cycle_p_losses[-1],
                    cycle_l_losses[-1],
                    identity_p_losses[-1],
                    identity_l_losses[-1],
                    losses_full[-1])
            )

        return (
            losses_d_p,
            losses_d_l,
            losses_g_p,
            losses_g_l,
            cycle_p_losses,
            cycle_l_losses,
            identity_p_losses,
            identity_l_losses,
            losses_full,
            lion_real_scores,
            lion_fake_scores
        )


def main():
    pumalion_train, pumalion_test = get_datasets(config.PUMA_ROOT, config.LION_ROOT)
    dataloader = DataLoader(pumalion_train, batch_size=config.BATCH_SIZE, shuffle=True)
    # dataloader_test = DataLoader(pumalion_test, batch_size=config.BATCH_SIZE, shuffle=False)

    disc_p = Discriminator().to(config.DEVICE)
    disc_l = Discriminator().to(config.DEVICE)
    gen_p = Generator().to(config.DEVICE)
    gen_l = Generator().to(config.DEVICE)

    opt_disc_p = optim.Adam(
        list(disc_p.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_disc_l = optim.Adam(
        list(disc_l.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_l.parameters()) + list(gen_p.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    sched_disc_p = optim.lr_scheduler.ExponentialLR(opt_disc_p, gamma=0.95)
    sched_disc_l = optim.lr_scheduler.ExponentialLR(opt_disc_l, gamma=0.95)
    sched_gen = optim.lr_scheduler.ExponentialLR(opt_gen, gamma=0.95)

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    (
        losses_d_p,
        losses_d_l,
        losses_g_p,
        losses_g_l,
        cycle_p_losses,
        cycle_l_losses,
        identity_p_losses,
        identity_l_losses,
        losses_full,
        lion_real_scores,
        lion_fake_scores
    ) = train(
        disc_p,
        disc_l,
        gen_p,
        gen_l,
        dataloader,
        opt_disc_p,
        opt_disc_l,
        opt_gen,
        l1,
        mse,
        d_scaler,
        g_scaler,
        sched_disc_p,
        sched_disc_l,
        sched_gen,
        config.EPOCHS
    )

    return (
        losses_d_p,
        losses_d_l,
        losses_g_p,
        losses_g_l,
        cycle_p_losses,
        cycle_l_losses,
        identity_p_losses,
        identity_l_losses,
        losses_full,
        lion_real_scores,
        lion_fake_scores
    )


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
