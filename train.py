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
from utils.tools import to_device, log, synth_one_sample,synth_toy,get_mask_from_lengths
from dataset import Dataset

# from model.BayesianPDA import BayesianPDA
from model.BayesianDTW import BayesianDTW

from matplotlib import pyplot as plt
import pdb

from torch import autograd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.manual_seed(0)
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True 

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



def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs


    # Get dataset
    if train_config["dataset"] == 'ryanspeech':
        print('Trainning the model on RyanSpeech dataset ...')          
        # Get dataset
        dataset = Dataset(
            "train.txt", preprocess_config, train_config, sort=True, drop_last=True
        )
        batch_size = train_config["optimizer"]["batch_size"]
        group_size = 1  # Set this larger than 1 to enable sorting in Dataset
        assert batch_size * group_size < len(dataset)
        loader = DataLoader(
            dataset,
            batch_size=batch_size * group_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )
        audio_processor = Audio(preprocess_config)
    else:
        raise("wrong dataset name!")

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = model.to(device)
    num_param = get_param_num(model)
    print("Number of BDPVAE-TTS Parameters:", num_param)

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
    def _get_reduction_factor(ep):
        i = 0
        while i < len(intervals) and intervals[i] <= ep:
            i += 1
        i = i - 1 if i > 0 else 0
        return rfs[i]
    # scaler =  torch.cuda.amp.GradScaler()

    while True:
        
        kl_weight = kl_weight_init + kl_weight_step * epoch if epoch <= kl_weight_inc_epochs else kl_weight_end

        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            
            for batch in batchs:
                # batch: ids, raw_texts,speakers,texts,text_lens,max(text_lens),mels,mel_lens,max(mel_lens),f0s,durations
                batch = to_device(batch, device)
            
                if train_config["dataset"] == 'toy':
                    # pdb.set_trace()
                    text_inputs = batchs['text'].to(device)
                    text_lengths = batchs['text_length'].to(device)
                    mel_lengths = batchs['mel_length'].to(device)
                    mel_input = batchs['mel'].to(device)
                    # Cut data
                    mel_max = mel_lengths.max()
                    text_max = text_lengths.max()
                    mel_input = mel_input[:,:,:mel_max].permute(0,2,1)
                    text_inputs = text_inputs[:,:text_max]
                    duration_inputs = batchs['duration'].to(device)
                    # pdb.set_trace()
                if step == 1:
                    with torch.no_grad():
                        model.init(speakers = batch[2:][0], text_inputs=batch[2:][1], mel_lengths=batch[2:][5], text_lengths=batch[2:][2],f0 = batch[2:][7],)
                        # model.init(text_inputs=text_inputs, mel_lengths=mel_lengths, text_lengths=text_lengths)
                                    # speakers,
            
                (predictions, mel_l2, kl_divergence, length_l2,logprob, latent_samples,mask,text_embd,W)= model(
                    speakers = batch[2:][0], inputs = batch[2:][1], text_lengths = batch[2:][2], max_src_len = batch[2:][3], f0 = batch[2:][7],
                    mel_targets = batch[2:][4], mel_lengths = batch[2:][5], 
                    reduce_loss=False)

                if train_config["optimizer"]["baseline"]:
                    # Reinforce moving average baseline
                    baseline_loss = moving_average(mel_l2.detach())
                    reinforce_loss = torch.mean(baseline_loss*logprob)
                else:
                    reinforce_loss = torch.mean(mel_l2.detach()*logprob)
                    # reinforce_loss = torch.mean(logprob)


                # Take average between batch
                mel_l2 = torch.mean(mel_l2)
                kl_divergence = torch.mean(kl_divergence)
                length_l2 = torch.mean(length_l2)

                # Total loss  #+ length_weight * length_l2 \
                total_loss =  mel_weight * mel_l2 \
                     + kl_weight * torch.max(kl_divergence, torch.tensor(0., device=device)) - reinforce_loss + reinforce_loss.detach()
      
                # Backward
                total_loss = total_loss / grad_acc_step
                
                total_loss.backward() #retain_graph=True)
                optimizer.step_and_update_lr()
                optimizer.zero_grad()
                # scaler.scale(total_loss).backward()
                if train_config["dataset"] in ['timit_pho','popcs'] and step % log_step == 0:
                    # Calculate MAD 
                    mel_len = batch[2:][5]
                    len_mask = get_mask_from_lengths(mel_len)
                    pred_mel2ph = (torch.argmax(latent_samples,2)+1)
                    MAD = torch.mean(torch.sum(pred_mel2ph == batch[2:][-1],-1)/mel_len)
                    pdb.set_trace()
                    # MAD = torch.mean(torch.abs(latent_samples.sum(1) - batch[2:][-1]))
                    train_logger.add_scalar("Accuracy/mad", MAD, step)
                
                # if step % grad_acc_step == 0:
                #     # Clipping gradients to avoid gradient explosion
                #     nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                #     # Update weights
                #     optimizer1.step_and_update_lr()
                #     optimizer1.zero_grad()
                #     # scaler.step(optimizer)
                #     # scaler.update()
                #     # optimizer.zero_grad()

                # (predictions, mel_l2, kl_divergence, length_l2, reinforce_loss,latent_samples, W, mask, logprobs, dec_alignments, decoded_initial,text_embd,det,std_norm) = model(
                #     batch[2:][0], batch[2:][1],batch[2:][2], batch[2:][3], batch[2:][4], batch[2:][5],
                #     reduce_loss=False)
                
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

                

                if step % synth_step == 0:


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

                        fig, wav_reconstruction, wav_prediction, tag, latent_figs, attn_figs, dec_init_fig, text_fig= synth_one_sample(
                            batch,
                            predictions,
                            latent_samples,
                            dec_alignments,
                            decoded_initial,
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


                if step % save_step == 0:
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
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default=False)
    parser.add_argument("--restore_step", type=int, default=0)
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

    main(args, configs)
