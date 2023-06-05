import re
import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pdb
import librosa 
import math

matplotlib.use("Agg")



def to_device(data, device):
    if len(data) == 9:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)

    if len(data) == 7:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len, durations) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        durations = torch.from_numpy(durations).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len,durations)

    if len(data) == 10:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        durations = torch.from_numpy(durations).to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            durations,
        )

    if len(data) == 11:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            f0,
            durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        f0 = torch.from_numpy(f0).to(device)
        durations = torch.from_numpy(durations).to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            f0,
            durations,
        )

def log(
    logger, step=None, losses=None, kl_weight=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/kl_loss", losses[2], step)
        logger.add_scalar("Loss/duration_loss", losses[3], step)
        logger.add_scalar("Loss/reinforce_loss", losses[4], step)
        # logger.add_scalar("Loss/det_term", losses[5], step)
        # logger.add_scalar("Loss/logprob_N(0,1)", losses[6], step)


    if kl_weight is not None:
        logger.add_scalar("Training/kl_weight", kl_weight, step)

    if fig is not None:
        logger.add_figure(tag, fig, step)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len,device = lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = ids < lengths.unsqueeze(1).expand(-1, max_len)

    return mask

def get_inverse_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len,lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

def get_2d_mask_from_lengths(x_len,y_len):
    '''
        Return float type 2d mask given two 1d masks
    '''
    x_mask = get_mask_from_lengths(x_len)
    y_mask = get_mask_from_lengths(y_len)
    return torch.bmm(x_mask.unsqueeze(2).float(),y_mask.unsqueeze(1).float())


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def synth_toy(
        mel_target,
        mel_len,
        predictions,
        text_dur,
        latent_samples,
        model_config,
        preprocess_config):


    mel_len = mel_len[0]
    
    # reduced_mel_len = reduced_mel_lens[0].item()
    mel_target = mel_target[0, :, :mel_len].cpu().detach().transpose(0,1)
    mel_prediction = predictions[0, :mel_len,:].cpu().detach().transpose(0, 1)
    latent = latent_samples[0].cpu().detach().numpy().transpose(1, 0)

    # sample_fig = plot_samples([latent,],["Alignment samples"])

    gt_dur = plot_dur2align(text_dur,latent,True)

    fig_dur = concate_fig(gt_dur,latent)
    fig = plot_mel(
        [
            mel_prediction,
            mel_target,
        ],
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram", "Decoder Alignment"],
    )

    return fig,fig_dur
    
def concate_fig(gt,pred, save_dir=None):
    fig, axes = plt.subplots(2, 1, squeeze=False)
 
    axes[0][0].imshow(gt, origin="lower")
    axes[0][0].set_aspect(2.5, adjustable="box")
    axes[0][0].set_ylim(0, gt.shape[0])
    axes[0][0].set_title('GT', fontsize="medium")

    axes[1][0].imshow(pred, origin="lower")
    axes[1][0].set_aspect(2.5, adjustable="box")
    axes[1][0].set_ylim(0, pred.shape[0])
    axes[1][0].set_title('Pred', fontsize="medium")
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close()

    return fig

def synth_one_sample(
        targets,
        predictions,
        latent_samples,
        text_embd,
        vocoder,
        audio_processor,
        model_config,
        preprocess_config,
        # random_samples
        ):
    clip_norm = preprocess_config["preprocessing"]["mel"]["normalize"]
    max_abs_value = preprocess_config["preprocessing"]["mel"]["max_abs_value"]
    min_level_db = preprocess_config["preprocessing"]["mel"]["min_level_db"]
    ref_level_db = preprocess_config["preprocessing"]["audio"]["ref_level_db"]

    basename = targets[1][0]
    src_len = targets[4][0].item()
    mel_len = targets[7][0].item()
    text_feature = text_embd[0].cpu().detach().numpy().transpose(1,0)

    mel_target = targets[6][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[0, :mel_len].detach().transpose(0, 1)

    latent = latent_samples[0,:mel_len,:src_len].cpu().detach().numpy().transpose(1, 0)
    # random = random_samples[0,:mel_len,:src_len].cpu().detach().numpy().transpose(1, 0)
    
    # If has duration GT
    # if len(targets) == 10:
    #     text_dur = targets[9][0].cpu().detach().numpy()
    #     text_dur_mat = duration2mat(text_dur,src_len,mel_len,overlap=True)
    #     # pdb.set_trace()
    #     # sample_fig = plot_hline([latent,text_dur_mat,random],['blue','green','orange'],["Sampled latent", "Ground-Truth","Random W"])
    #     sample_fig = plot_step([latent,text_dur_mat],['blue','green','orange'],["Sampled latent", "Ground-Truth","Random W"])
    #     # sample_fig = plot_samples([latent],["Alignment samples"])
    #     # GT_fig = plot_samples([text_dur_mat],["Alignment GT"])

    # else:
    sample_fig = plot_samples([latent,],["Alignment samples"])
    text_fig = plot_samples([text_feature],["Embed text feature"])

    if clip_norm:
        mel_target = audio_processor._denormalize(mel_target.cpu().numpy())
        mel_prediction = audio_processor._denormalize(mel_prediction.cpu().numpy())
    mel_target = mel_target + ref_level_db
    mel_prediction = mel_prediction + ref_level_db

    fig = plot_mel(
        [
            mel_prediction,
            mel_target,
        ],
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram", "Decoder Alignment"],
    )
    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = audio_processor.inv_preemphasize(
            vocoder_infer(
                torch.from_numpy(mel_target).to(device).unsqueeze(0),
                vocoder,
                model_config,
                preprocess_config,
            )[0]
        )
        wav_prediction = audio_processor.inv_preemphasize(
            vocoder_infer(
                torch.from_numpy(mel_prediction).to(device).unsqueeze(0),
                vocoder,
                model_config,
                preprocess_config,
            )[0]
        )
    else:
        wav_reconstruction = audio_processor.inv_mel_spectrogram(mel_target)
        wav_reconstruction = audio_processor.inv_preemphasize(wav_reconstruction)
        wav_prediction = audio_processor.inv_mel_spectrogram(mel_prediction)
        wav_prediction = audio_processor.inv_preemphasize(wav_prediction)

    return fig, wav_reconstruction, wav_prediction, basename,sample_fig,text_fig


def synth_samples(
        targets,
        predictions,
        pred_lens,
        text_lens,
        latent_samples,
        vocoder,
        audio_processor,
        model_config,
        preprocess_config,
        path):
    clip_norm = preprocess_config["preprocessing"]["mel"]["normalize"]
    max_abs_value = preprocess_config["preprocessing"]["mel"]["max_abs_value"]
    min_level_db = preprocess_config["preprocessing"]["mel"]["min_level_db"]
    ref_level_db = preprocess_config["preprocessing"]["audio"]["ref_level_db"]
    if preprocess_config["dataset"] == "timit_pho":
        MAD = torch.mean(torch.abs(latent_samples.sum(1) - targets[-1]))

    predictions = predictions.detach().cpu().numpy()
    if clip_norm:
        predictions = audio_processor._denormalize(predictions)
    predictions = predictions + ref_level_db

    basenames = targets[0]
    texts = targets[1]

    for i in range(len(targets[0])):
        basename = basenames[i]
        text = texts[i]
        src_len = text_lens[i].item()
        mel_len = pred_lens[i].item()
        latent = latent_samples[i].cpu().detach().numpy().transpose(1, 0)

        mel_prediction = np.transpose(predictions[i, :mel_len], [1, 0])

        sample_fig = plot_samples([latent,],["Alignment samples"],save_dir=os.path.join(path, "{}_latent.png".format(basename)))

        fig = plot_mel(
            [
                mel_prediction,
            ],
            ["Synthetized Spectrogram"],
            save_dir=os.path.join(path, "{}.png".format(basename)),
        )



    from .model import vocoder_infer

    mel_predictions = np.transpose(predictions, [0, 2, 1])
    lengths = pred_lens * preprocess_config["preprocessing"]["audio"]["frame_shift_sample"]

    if vocoder is not None:
        wav_predictions = list()
        for wav_prediction in vocoder_infer(
                torch.from_numpy(mel_predictions).cuda(),
                vocoder, model_config, preprocess_config, lengths=lengths
            ):
            wav_predictions.append(audio_processor.inv_preemphasize(wav_prediction))
    else:
        wav_predictions = list()
        for sample in range(len(mel_predictions)):
            wav_prediction = audio_processor.inv_mel_spectrogram(mel_predictions[sample])
            wav_prediction = audio_processor.inv_preemphasize(wav_prediction)
            wav_predictions.append(wav_prediction)


    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        audio_processor.save_wav(os.path.join(path, "{}.wav".format(basename)), wav)
    # return mcd


def cumulative_sum(l):
    new_list=[]
    j=0
    for i in range(0,len(l)):
        j+=l[i]
        new_list.append(j)
    return new_list

def plot_mel(data, titles, save_dir=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    for i in range(len(data)):
        mel = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close()

    return fig

def plot_dur2align(data,latent,return_matrix=True,save_dir=None,plot_all=False):
    text_len = latent.shape[0]
    zeta = data[0,:text_len]
    matrix = np.zeros_like(latent)
    for i,value in enumerate(zeta):
        value = int(value)
        if i == 0 :
            pre = 0
        matrix[i,pre:pre+value] = 1
        pre = pre+value
    if return_matrix:
        return matrix
    else:
        fig, axes = plt.subplots(1, 1, squeeze=False)
        axes[0][0].imshow(matrix, origin="lower")
        axes[0][0].set_aspect(2.5, adjustable="box")
        axes[0][0].set_ylim(0, matrix.shape[0])
        axes[0][0].set_title('Duration to Alignment', fontsize="small")
        axes[0][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[0][0].set_anchor("W")
        if save_dir is not None:
            plt.savefig(save_dir)
        plt.close()

        return fig


def duration2mat(data,text_len,spec_len,overlap):
    matrix = np.zeros((text_len,spec_len))
    if overlap == True:
        for i,value in enumerate(data):
                value = int(value)
                if i == 0 :
                    pre = 0
                matrix[i,pre:pre+value] = 1
                pre = pre+value-1
        return matrix
    else:
        for i,value in enumerate(data):
            value = int(value)
            if i == 0 :
                pre = 0
            matrix[i,pre:pre+value] = 1
            pre = pre+value
        return matrix

def plot_samples(data,titles,save_dir=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    for i in range(len(data)):
        latent = data[i]
        axes[i][0].imshow(latent, origin="lower")
        axes[i][0].set_aspect(2, adjustable="box")
        axes[i][0].set_ylim(0, latent.shape[0])
        # axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")
        axes[i][0].set_xlabel("Spectrogram")
        axes[i][0].set_ylabel("Phoneme sequence")
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close()

    return fig      

def plot_multi_attn(attn_keys, attn_values, save_dir=None):
    figs = list()
    for i, attn in enumerate(attn_values):
        fig = plt.figure()
        num_head = attn.shape[0]
        for j, head_ali in enumerate(attn):
            ax = fig.add_subplot(2, num_head // 2, j + 1)
            ax.set_xlabel('Audio timestep (reduced)') if j >= num_head-2 else None
            ax.set_ylabel('Text timestep') if j % 2 == 0 else None
            # pdb.set_trace()
            im = ax.imshow(head_ali, aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax)
        # plt.tight_layout()
        fig.suptitle(attn_keys[i], fontsize=10)
        figs.append(fig)
        if save_dir is not None:
            plt.savefig(save_dir[i])
        plt.close()

    return figs


def plot_attn(attn_keys,attn_values,save_dir = None):

    figs = list()
    for i, attn in enumerate(attn_values):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Audio timestep') 
        ax.set_ylabel('Hidden') 
        im = ax.imshow(attn, origin='lower')
        # fig.colorbar(im, ax=ax)
        # plt.tight_layout()
        fig.suptitle(attn_keys[i], fontsize=10)
        figs.append(fig)
        if save_dir is not None:
            plt.savefig(save_dir[i])
        plt.close()
    return figs

def plot_hline(inputs,color,label,save_dir=None):
    fig,ax = plt.subplots(1,1)
    for idx,data in enumerate(inputs):
        y = range(data.shape[0])
        xmax = cumulative_sum(data.sum(1))
        xmin = [0]+xmax[:-1]
        ax.hlines(y,xmin,xmax,color = color[idx],label= label[idx])

    ax.set_xlabel("Spectrogram frame")
    ax.set_ylabel("Phoneme sequence")
    plt.legend(loc=2)
    return fig

def plot_step(inputs,color,label,save_dir=None):
    fig,ax = plt.subplots(1,1)
    for idx,data in enumerate(inputs):

        coords = np.argwhere(data==1)
        xmin = np.zeros(data.shape[0])
        xmax = np.zeros(data.shape[0])
        y = range(data.shape[0])
        # plot hlines
        for row in range(data.shape[0]):
            cor = np.where(coords[:,0] == row)
        
            xmin[row] = coords[cor[0][0]][1]
            xmax[row] = coords[cor[0][-1]][1]

        ax.hlines(y,xmin,xmax,color = color[idx],label= label[idx])
        # plot vlines
        ymin = np.zeros(data.shape[1])
        ymax = np.zeros(data.shape[1])
        x = range(data.shape[1])
        for col in range(data.shape[1]):
            cor = np.where(coords[:,1] == col)

            ymin[col] = coords[cor[0][0]][0]
            ymax[col] = coords[cor[0][-1]][0]

        ax.vlines(x,ymin,ymax,color=color[idx])

        # plot diag
        for row in range(data.shape[0]-1):
            cor  = np.where(coords[:,0] == row)
            corn = np.where(coords[:,0] == row+1)
            # pdb.set_trace()
            if coords[cor[0][-1]][1] == coords[corn[0][0]][1] - 1:

                ident2 = np.array([coords[cor[0][-1]][-1],coords[corn[0][0]][-1]])
                ident1 = np.array([coords[cor[0][-1]][0],coords[corn[0][0]][0]])
                ax.plot(ident2,ident1,color = color[idx])


    ax.set_xlabel("Spectrogram frame",fontsize=12)
    ax.set_ylabel("Phoneme sequence", fontsize=12)
    plt.legend(loc=2)
    return fig

def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
