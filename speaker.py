import argparse
from resemblyzer import VoiceEncoder, preprocess_wav
import pyreaper
import torch
import torchaudio
import random
from os.path import join, exists, basename
from os import makedirs, walk
from tqdm import tqdm
from glob import glob
#import json
import pickle
import parselmouth

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = VoiceEncoder()


def get_f0_with_parselmouth(filepath):
    snd = parselmouth.Sound(filepath)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    return pitch_values.mean(), pitch_values.max(), pitch_values.min()


def get_f0_with_pyreaper(filepath):

    x, fs = torchaudio.load(filepath)
    x = x * 32768.0
    x = x.to(torch.int16)
    pm_times, pm, f0_times, f0, corr = pyreaper.reaper(x.squeeze().numpy(), fs)

    return f0.mean(), f0.max(), f0.min()


def extract_embedding(filepath):
    '''
    Embeddings from: Generalized End-To-End Loss for Speaker Verification
    '''
    wav = preprocess_wav(filepath)
    file_embedding = encoder.embed_utterance(wav)
    embedding = torch.tensor(file_embedding)
    return embedding


def calculate_average(speaker_dict):
    if 'train' not in speaker_dict:
        return {}

    speaker_avg_dict = {'train': {}}

    for speaker_id in speaker_dict['train']:
        samples = speaker_dict['train'][speaker_id]

        emb_avg = torch.zeros(256)
        f0_mean_avg = 0
        f0_max_avg = 0
        f0_min_avg = 0

        total_samples = len(samples)

        for embedding, (f0_mean, f0_max, f0_min) in samples:
            emb_avg += embedding
            f0_mean_avg += f0_mean
            f0_max_avg += f0_max
            f0_min_avg += f0_min

        emb_avg /= total_samples
        f0_mean_avg /= total_samples
        f0_max_avg /= total_samples
        f0_min_avg /= total_samples

        speaker_avg_dict['train'][speaker_id] = \
            emb_avg.tolist(), (f0_mean_avg, f0_max_avg, f0_min_avg)

    return speaker_avg_dict


def generate_list_dict_from_dict(filelist_dict, output_filepath):
    '''
    Embeddings from: Generalized End-To-End Loss for Speaker Verification
    '''
    speaker_dict= {'train': {}}

    for speaker_id in tqdm(filelist_dict):

        for filepath in filelist_dict[speaker_id]:
            #print("Processing file: {}".format(filepath))
            # Load audio file
            if not exists(filepath):
                print("file {} doesnt exist!".format(filepath))
                continue

            speaker_id = str(basename(filepath).split('-')[0])

            embedding = extract_embedding(filepath)
            #f0_mean, f0_max, f0_min = get_f0(filepath)
            f0_mean, f0_max, f0_min = get_f0_with_parselmouth(filepath)

            if speaker_id not in speaker_dict['train']:
                speaker_dict['train'][speaker_id] = [
                    (
                        embedding.tolist(),
                        (f0_mean, f0_max, f0_min)
                    )
                ]
            else:
                speaker_dict['train'][speaker_id].append(
                    (
                        embedding.tolist(),
                        (f0_mean, f0_max, f0_min)
                    )
                )
    return speaker_dict



def generate_list_dict_from_list(filelist_train, filelist_val, root_folder, output_filepath):
    '''
    Embeddings from: Generalized End-To-End Loss for Speaker Verification
    '''
    speaker_dict= {'train': {}}

    for filepath in tqdm(filelist_train):
        speaker_id = str(filepath)
        filepath = join(root_folder, filepath)
        #print("Processing file: {}".format(filepath))
        # Load audio file
        if not exists(filepath):
            print("file {} doesnt exist!".format(filepath))
            continue

        #speaker_id = str(basename(filepath).split('-')[0])
        embedding = extract_embedding(filepath)
        #f0_mean, f0_max, f0_min = get_f0(filepath)
        f0_mean, f0_max, f0_min = get_f0_with_parselmouth(filepath)
        f0_mean, f0_max, f0_avg = int(f0_mean), int(f0_max), ((f0_mean + f0_max) / 2)
        speaker_dict['train'][speaker_id] = embedding.numpy(), (f0_mean, f0_max, f0_avg)


    speaker_dict['valid'] = {}

    for filepath in tqdm(filelist_val):
        speaker_id = str(filepath)
        filepath = join(root_folder, filepath)
        #print("Processing file: {}".format(filepath))
        # Load audio file
        if not exists(filepath):
            print("file {} doesnt exist!".format(filepath))
            continue

        #speaker_id = str(basename(filepath).split('-')[0])
        embedding = extract_embedding(filepath)
        #f0_mean, f0_max, f0_min = get_f0(filepath)
        f0_mean, f0_max, f0_min = get_f0_with_parselmouth(filepath)
        f0_mean, f0_max, f0_avg = int(f0_mean), int(f0_max), ((f0_mean + f0_max) / 2)
        speaker_dict['valid'][speaker_id] = embedding.numpy(), (f0_mean, f0_max, f0_avg)

    return speaker_dict

def select_files(filelist, total_files_per_speaker=5):

    random.shuffle(filelist)
    filelist_dict = {}
    for filepath in filelist:
        # Load audio file
        if not exists(filepath):
            continue
        #speaker_id = str(basename(filepath).split('-')[0])
        speaker_id = filepath
        if speaker_id not in filelist_dict:
            filelist_dict[speaker_id] = []

        if len(filelist_dict[speaker_id]) < total_files_per_speaker:
            filelist_dict[speaker_id].append(filepath)

    return filelist_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--input_train', default="feature/lab/train.tsv")
    parser.add_argument('-v', '--input_val', default="feature/lab/valid.tsv")
    parser.add_argument('-d', '--dataset_dir', default="dataset", help='Dataset root folder')
    parser.add_argument('-o', '--output', default='feature/lab/spk2info.dict', help='Output folder')
    args = parser.parse_args()

    '''
    filelist = []
    for root, dirs, files in walk(args.input):
        for file in files:
            if file.endswith(".wav"):
                #print(join(root, file))
                filelist.append(join(root, file))

    filelist.sort()
    #makedirs(args.output, exist_ok=True)

    filelist_dict = select_files(filelist)
    '''

    with open(args.input_train, "r") as file:
        data = file.readlines()[1:]
    filelist_train = [line.split("\t")[0] for line in data]

    with open(args.input_val, "r") as file:
        data = file.readlines()[1:]
    filelist_val = [line.split("\t")[0] for line in data]

    speaker_list_dict = generate_list_dict_from_list(filelist_train, filelist_val, args.dataset_dir, args.output)

    del filelist_train
    del filelist_val

    with open(args.output, 'wb') as file:
        pickle.dump(speaker_list_dict, file)


if __name__ == "__main__":
    main()
