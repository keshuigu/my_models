import argparse
import inspect
import torch
import os
from accelerate import Accelerator
from loguru import logger as log
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.config import train_config
from src.utils import init_dirs, init_logs, init_seeds, log_config, save_labeled_PIL_imgs, save_labeled_imgs
from src.data import get_dataset, get_transforms
from src.models import get_vqgan_model, get_discriminator

# export PYTHONPATH=/mnt/share/workspace/research/my_models
def get_config_class(config_name):
    members = inspect.getmembers(train_config, inspect.isclass)
    for name, member in members:
        if name == config_name:
            return member
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='conf')
    args = parser.parse_args()
    # 创建配置对象
    config_name = args.config
    config_class = get_config_class(config_name)
    if config_class is None:
        log.error("Config class not found. Please check the config name.")
    config = config_class()

    # 不要颠倒顺序
    sample_dir, logs_dir, tensorboard_dir = init_dirs(config)
    init_logs(config)
    init_seeds(config)
    log.info("Config: \n{}", log_config(config_class))
    dataset_train, dataset_test = get_dataset(config, transform=get_transforms(config))

    train_dataloader = DataLoader(dataset_train, batch_size=config.train_loader_batchSize, shuffle=True,
                                  num_workers=config.train_loader_numWorkers)

    if dataset_test is not None:
        test_dataloader = DataLoader(dataset_test, batch_size=config.test_loader_batchSize, shuffle=True,
                                     num_workers=config.test_loader_numWorkers)

    model = get_vqgan_model(config)
    discriminator = get_discriminator(config)

    if config.model_vae_pretrained is not None:
        model = model.from_pretrained(config.model_vae_pretrained)

    if config.model_dis_pretrained is not None:
        discriminator = discriminator.from_pretrained(config.model_dis_pretrained)

    optimizer1 = torch.optim.AdamW(model.parameters(), lr=config.train_optim_lr)
    optimizer2 = torch.optim.Adam(discriminator.model_vqLPIPSWithDiscriminator.discriminator.parameters(),
                                  lr=config.train_optim_lr, betas=(0.5, 0.9))

    lr_scheduler1 = get_cosine_schedule_with_warmup(
        optimizer=optimizer1,
        num_warmup_steps=config.train_optim_warmupSteps,
        num_training_steps=(len(train_dataloader) * config.train_epochsNum),
    )

    lr_scheduler2 = get_cosine_schedule_with_warmup(
        optimizer=optimizer2,
        num_warmup_steps=config.train_optim_warmupSteps,
        num_training_steps=(len(train_dataloader) * config.train_epochsNum),
    )

    accelerator = Accelerator(
        mixed_precision=config.train_accelerate_mixedPrecision,
        gradient_accumulation_steps=config.train_accelerate_gradientAccumulationSteps,
        log_with="tensorboard",
        project_dir=tensorboard_dir,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("train_example")

    model, discriminator, optimizer1, optimizer2, train_dataloader, test_dataloader, lr_scheduler1 = accelerator.prepare(
        model, discriminator, optimizer1, optimizer2, train_dataloader, train_dataloader, lr_scheduler1
    )

    global_step = 0

    for epoch in range(config.train_epochsNum):
        if config.model_disc_start < global_step:
            log.info("Begin Disc Train on Epoch {}", epoch)

        process_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        process_bar.set_description(f"Epoch {epoch}")

        model.train()
        if config.debug:
            if epoch > config.debug_time:
                break

        for step, batch in enumerate(train_dataloader):
            if config.debug:
                if step > config.debug_time:
                    break
            images, labels = batch
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            h = model.encode(images).latents
            (quant, emb_loss, (_, _, ind)) = model.VQ(h, False)
            dec = model.decode(quant).sample
            global_step += 1
            if step % 2 == 0:
                model.train()  # 设置为训练模式
                discriminator.eval()
                loss, log_dict_ae = discriminator(emb_loss,
                                                  images,
                                                  dec,
                                                  optimizer_idx=0,
                                                  global_step=global_step,
                                                  last_layer=model.decoder.conv_out.weight, split="train", )
                accelerator.backward(loss)
                optimizer1.step()
                lr_scheduler1.step()
                process_bar.update(1)
            else:
                model.eval()
                discriminator.train()
                disc_loss, log_dict_disc = discriminator(emb_loss,
                                                         images,
                                                         dec,
                                                         optimizer_idx=1,
                                                         global_step=global_step,
                                                         last_layer=model.decoder.conv_out.weight, split="train")
                accelerator.backward(disc_loss)
                optimizer2.step()
                lr_scheduler2.step()
                process_bar.update(1)
        log.info("End of train Epoch: {} Global Step:{}", epoch, global_step)
        if accelerator.is_main_process:
            if epoch % config.train_savePerEpochs == 0:
                model.save_pretrained(os.path.join(config.init_outputDir, "vae-%d" % epoch))
                discriminator.save_pretrained(os.path.join(config.init_outputDir, "dis"))
            if epoch % config.train_evalPerEpoch == 0:
                model.eval()
                discriminator.eval()
                for step, batch in enumerate(test_dataloader):
                    if config.debug:
                        if step > config.debug_time:
                            break
                    images, labels = batch
                    B, C, H, W = images.shape
                    h = model.encode(images).latents
                    (quant, emb_loss, (_, _, ind)) = model.VQ(h, False)
                    dec = model.decode(quant).sample
                    aeloss, log_dict_ae = discriminator(emb_loss,
                                                        images,
                                                        dec,
                                                        optimizer_idx=0,
                                                        global_step=global_step,
                                                        last_layer=model.decoder.conv_out.weight, split="val")
                    discloss, log_dict_disc = discriminator(emb_loss,
                                                            images,
                                                            dec,
                                                            optimizer_idx=1,
                                                            global_step=global_step,
                                                            last_layer=model.decoder.conv_out.weight,
                                                            split="val")
                    if step == 0:
                        if config.model_inChannels == 3:
                            postProcess = transforms.Compose([
                                transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # 反归一化
                                transforms.Lambda(lambda x: x.clamp(0, 255)),
                                transforms.ToPILImage(),
                            ])
                        elif config.model_inChannels == 1:
                            postProcess = transforms.Compose([
                                transforms.Normalize(mean=[-1], std=[2]),
                                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                transforms.Lambda(lambda x: x.clamp(0, 255)),
                                transforms.ToPILImage(),
                            ])
                        else:
                            raise ValueError("model_inChannels should be 1 or 3")

                        img_raws = [postProcess(i) for i in images]
                        save_labeled_PIL_imgs(img_raws, torch.randn([B]),
                                              os.path.join(sample_dir, f"test-{epoch}-raw.png"), 4, 4)
                        img_aes = [postProcess(i) for i in dec]
                        save_labeled_PIL_imgs(img_aes, torch.randn([B]),
                                              os.path.join(sample_dir, f"test-{epoch}-aed.png"), 4, 4)

                    logs = {"loss-test-gen": aeloss, "loss-test-disc": discloss}
                    accelerator.log(logs, step=epoch)
        process_bar.close()
        log.info("End of test Epoch: {} Global Step:{}", epoch, global_step)
