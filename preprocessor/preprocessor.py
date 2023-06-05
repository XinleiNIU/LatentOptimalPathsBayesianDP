import os
import random
import json

import tgt
import librosa
import numpy as np
from tqdm import tqdm

from audio import Audio
from text import grapheme_to_phoneme, text_to_sequence
from utils.tools import read_lexicon
from g2p_en import G2p
import pdb 
import math
from sklearn.preprocessing import StandardScaler
from pypinyin import pinyin, Style

random.seed(1234)


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.frame_shift_sample = config["preprocessing"]["audio"]["frame_shift_sample"]
        self.frame_length_sample = config["preprocessing"]["audio"]["frame_length_sample"]
        self.clip_norm = config["preprocessing"]["mel"]["normalize"]
        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.g2p = G2p()
        self.audio_processor = Audio(config)
        self.lexicon = read_lexicon(config["path"]["lexicon_path"])
        self.cleaner = config["preprocessing"]["text"]["text_cleaners"]

    def build_alignment(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        idx = 0
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
                if ".wav" not in wav_name:
                    continue
                speaker_id = wav_name.split("-")[1]
                basename = wav_name.split(".")[0]
                if speaker_id not in speakers:
                    speakers[speaker_id] = idx 
                    idx +=1
                ret = self.process_utterance_pho(speaker, speaker_id, basename, self.clip_norm)
      
                if ret is None:
                    continue
                else:
                    info, n = ret
                out.append(info)

                n_frames += n
             
        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        print(
            "Total time: {} hours".format(
                n_frames * self.frame_shift_sample / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        print("Processing Data ...")
        out = list()
        n_frames = 0

        speakers = {}
        idx = 0
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            # speakers[speaker] = i

            for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
                speaker_id = wav_name.split("-")[1]
                basename = wav_name.split(".")[0]
                if speaker_id not in speakers:
                    speakers[speaker_id] = idx 
                    idx +=1

                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]

                ret = self.process_utterance(speaker, speaker_id, basename, self.clip_norm)
                if ret is None:
                    continue
                else:
                    info, n = ret
                out.append(info)

                n_frames += n

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        print(
            "Total time: {} hours".format(
                n_frames * self.frame_shift_sample / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

 
    def process_utterance(self, speaker,speaker_id, basename, clip_norm=False):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))

        # Compute mel-scale spectrogram from raw audio
        mel_spectrogram = self.audio_processor.get_mel_from_wav(wav_path, clip_norm=clip_norm)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Get phoneme
        phone = grapheme_to_phoneme(raw_text, self.g2p, self.lexicon)
        text = "{" + " ".join(phone) + "}"

        # Save files
        mel_filename = "{}-mel-{}.npy".format(speaker_id, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker_id, text, raw_text]),
            mel_spectrogram.shape[1],
        )

    
    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)
        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))
        return min_value, max_value

    def cumulative_sum(self,l):
        new_list=[]
        j=0
        for i in range(0,len(l)):
            j+=l[i]
            new_list.append(j)
        return new_list


    def preprocess_mandarin(self, text):

        phones = []
        pinyins = [
            p[0]
            for p in pinyin(
                text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
            )
        ]
        for p in pinyins:
            if p in self.lexicon:
                phones += self.lexicon[p]
            else:
                phones.append("sp")

        phones = "{" + " ".join(phones) + "}"
        return phones, pinyins

