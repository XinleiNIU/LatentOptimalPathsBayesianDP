import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from audio import Audio
from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample,synth_toy, get_mask_from_lengths
from dataset import Dataset

from evaluate import evaluate

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
# from model.BayesianPDA import BayesianPDA
from model.BayesianDTW import BayesianDTW

from matplotlib import pyplot as plt
import pdb

from torch import autograd

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cleanup():
    dist.destroy_process_group()

def moving_average(loss,n=5):

    # Build a tensor list
    loss_list = []

    if len(loss_list) >= n:
        # drop the first loss
        loss_list =  loss_list[1:,:]
  
    loss_list.append(loss.data)

    baseline = loss-sum(loss_list)/len(loss_list)
    
    if baseline.all() == 0:
        return loss.detach()
    else:
        return baseline


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)



def prepare(train_config, preprocess_config, rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    """ Distribute the dataloader """

    if train_config["dataset"] == 'ryanspeech':
        print('Trainning the model on RyanSpeech dataset ...')          
        # Get dataset
        dataset = Dataset(
            "train.txt", preprocess_config, train_config, sort=True, drop_last=True
        )
        batch_size = train_config["optimizer"]["batch_size"]
        group_size = 1  # Set this larger than 1 to enable sorting in Dataset
        assert batch_size * group_size < len(dataset)
        audio_processor = Audio(preprocess_config)

    else:
        raise("wrong dataset name!")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size*group_size, 
                        pin_memory=pin_memory, 
                        num_workers=num_workers, 
                        drop_last=False, 
                        shuffle=True, 
                        sampler=sampler,
                        collate_fn=dataset.collate_fn,)
    
    return dataloader, audio_processor


def main(rank, args, configs, world_size):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    setup(rank,world_size)
    device = rank

    # Get data loader 
    dataloader, audio_processor = prepare(train_config,preprocess_config,rank,world_size)
    # Get model and optimizer
    model, optimizer = get_model(args, configs, device, train=True)
    model = model.to(rank)
    # wrap the model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    num_param = get_param_num(model)
    print("Number of BDPVAE Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    debug_step = train_config["step"]["debug_step"]
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    length_weight = train_config["length"]["length_weight"]
    kl_weight_init = train_config["kl"]["kl_weight_init"]
    kl_weight_end = train_config["kl"]["kl_weight_end"]
    kl_weight_inc_epochs = train_config["kl"]["kl_weight_increase_epoch"]
    kl_weight_step = (kl_weight_end - kl_weight_init) / kl_weight_inc_epochs
    mel_weight = train_config["length"]["mel_weight"]

    alpha = model_config["common"]["alpha"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    # reduction factor computation



    while True:
        
        kl_weight = kl_weight_init + kl_weight_step * epoch if epoch <= kl_weight_inc_epochs else kl_weight_end

        inner_bar = tqdm(total=len(dataloader), desc="Epoch {}".format(epoch), position=1)
        for batchs in dataloader:
            dataloader.sampler.set_epoch(epoch)
            for batch in batchs:
                # batch: ids, raw_texts,speakers,texts,text_lens,max(text_lens),mels,mel_lens,max(mel_lens)
                batch = to_device(batch, rank)
                print(step)
                if step == 1:
                    with torch.no_grad():
                        model.module.init(speakers = batch[2:][0], text_inputs=batch[2:][1], mel_lengths=batch[2:][5], text_lengths=batch[2:][2])#,f0 = batch[2:][7])
    
            
                (predictions, mel_l2, kl_divergence, length_l2,logprob, latent_samples,mask,text_embd,W)= model(
                    speakers = batch[2:][0], inputs = batch[2:][1], text_lengths = batch[2:][2], max_src_len = batch[2:][3], # f0 = batch[2:][7],
                    mel_targets = batch[2:][4], mel_lengths = batch[2:][5], max_mel_len = batch[2:][6],
                    reduce_loss=False)

                if train_config["optimizer"]["baseline"]:
                    # Reinforce moving average baseline
                    baseline_loss = moving_average(mel_l2.detach(), n = 10)
                    reinforce_loss = torch.mean(baseline_loss*logprob)
                else:
                    reinforce_loss = torch.mean(mel_l2.detach()*logprob)
                    # reinforce_loss = torch.mean(logprob)


                # Take average between batch
                mel_l2 = torch.mean(mel_l2)
                kl_divergence = torch.mean(kl_divergence)
                length_l2 = torch.mean(length_l2)

                # Total loss
                # total_loss =  mel_weight * mel_l2 + length_weight * length_l2 \
                #      + kl_weight * torch.max(kl_divergence, torch.tensor(0., device=device)) - reinforce_loss + reinforce_loss.detach()

                total_loss =  mel_weight * mel_l2 +  length_weight * length_l2 \
                     + kl_weight * torch.max(kl_divergence, torch.tensor(0., device=device)) - reinforce_loss + reinforce_loss.detach()      
                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()     
                # total_loss.backward() #retain_graph=True)
                # optimizer.step_and_update_lr()
                # optimizer.zero_grad()

                if train_config["dataset"] in ['timit_pho', 'popcs2'] and step % log_step == 0:
                    # Calculate MAD 
                    mel_len = batch[2:][5]
                    # len_mask = get_mask_from_lengths(mel_len)
                    pred_mel2ph = (torch.argmax(latent_samples,2)+1)
                    MAD = torch.mean(torch.sum(pred_mel2ph == batch[2:][-1],-1)/mel_len)
                    # MAD = torch.mean(torch.abs(latent_samples.sum(1) - batch[2:][-1]))
                    train_logger.add_scalar("Accuracy/mad", MAD, step)
                    # pdb.set_trace()
                    # # Calculate MAD 
                    # MAD = torch.mean(torch.abs(latent_samples.sum(1) - batch[2:][-1]))
                    # train_logger.add_scalar("Accuracy/mad", MAD, step)

                if step % debug_step == 0:
                    spect = spec_embs.cpu().detach().numpy()
                    fig_sample, axes = plt.subplots(1, 1, squeeze=False)
                    axes[0][0].imshow(spect[0].T,origin="lower")
                    plt.savefig("./spect_feature.png")
                    for name, param in policy_model.named_parameters():
                        print(name, param.grad)
                        pdb.set_trace()
                    pdb.set_trace()

                if step % log_step == 0:
                    losses = [l.item() for l in list([total_loss, mel_l2, kl_divergence, length_l2, reinforce_loss])]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, KLD Loss: {:.4f}, Duration Loss: {:.4f}, Reinforce Loss:{:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses, kl_weight=kl_weight)


                

                if step % synth_step == 0 and rank == 0:


                    if train_config["dataset"] == 'toy':
                        fig,latent_figs = synth_toy(mel_input, mel_lengths,predictions,batchs['duration'],latent_samples,model_config,preprocess_config)
                        log(
                            train_logger,
                            fig=fig,
                            tag="Training/step_{}_{}".format(step, 'toy data'),
                        )
                        log(
                            train_logger,
                            fig=latent_figs,
                            tag="Training_Duration_Matrix/step_{}_{}".format(step, 'toy data'),
                        )   
                    else:

                        fig, wav_reconstruction, wav_prediction, tag, latent_figs, text_fig= synth_one_sample(
                            batch,
                            predictions,
                            latent_samples,
                            text_embd,
                            vocoder,
                            audio_processor,
                            model_config,
                            preprocess_config,
                        )
                        log(
                            train_logger,
                            fig=fig,
                            tag="Training/step_{}_{}".format(step,tag),
                            step = step
                        )
                        log(
                            train_logger,
                            fig=latent_figs,
                            tag="Training_latent_alignment/step_{}_{}".format(step,tag),
                            step = step
                        )
                        sampling_rate = preprocess_config["preprocessing"]["audio"][
                            "sampling_rate"
                        ]
                        log(
                            train_logger,
                            audio=wav_reconstruction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_reconstructed".format(step, tag),
                            step = step
                        )
                        log(
                            train_logger,
                            audio=wav_prediction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_synthesized".format(step, tag),
                            step = step
                        )

                if step % save_step == 0 and rank == 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)
            inner_bar.update(1)
            # cleanup()
        epoch += 1


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--pretrain", type=str, default=False)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    world_size = torch.cuda.device_count()
    # main(args, configs)
    mp.spawn(main,args=(args, configs, world_size), nprocs = world_size)
