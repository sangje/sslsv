import os
from operator import itemgetter

import torch
import numpy as np
import soundfile as sf

from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve

from sslsv.data.utils import load_audio
from scipy import spatial


def extract_embeddings_from_batch(curr_batch_data, model):
    batch = np.array(curr_batch_data)
    B, N, T = batch.shape
    #print(B, N, T)
    batch = batch.reshape((B * N, T))
    #batch = batch.reshape((B ,N , T))
    #print(batch.shape)

    with torch.no_grad():
        feats = model(torch.FloatTensor(batch).cuda()).detach().cpu().numpy()
    feats = normalize(feats, axis=1)
    feats = feats.reshape((B, N, -1))
    feats = feats.mean(axis=1)
    return feats


def extract_embeddings_origi(
    model,
    dataset_config,
    batch_size=64,
    num_frames=6
):
    # Get a list of unique utterances
    utterances = set()
    for line in open(dataset_config.trials):
        target, a, b = line.rstrip().split(' ')
        utterances.add(a)
        utterances.add(b)

    # Determine embeddings for each unique utterance
    embeddings = {}
    curr_batch_ids = []
    curr_batch_data = []
    for utterance in utterances:
        if len(curr_batch_ids) == batch_size:
            feats = extract_embeddings_from_batch(curr_batch_data, model)
            for i in range(len(curr_batch_ids)):
                uttid, data = curr_batch_ids[i], feats[i]
                embeddings[uttid] = data
            curr_batch_ids, curr_batch_data = [], []

        # Store current utterance id and data
        audio_path = os.path.join(dataset_config.base_path, 'voxceleb1', utterance)
        
        data = load_audio(audio_path, dataset_config.frame_length, num_frames=num_frames)
        curr_batch_ids.append(utterance)
        curr_batch_data.append(data)

    # Register remaining samples (if nb samples % 64 != 0)
    if len(curr_batch_ids) != 0:
        feats = extract_embeddings_from_batch(curr_batch_data, model)
        for i in range(len(curr_batch_ids)):
            uttid, data = curr_batch_ids[i], feats[i]
            embeddings[uttid] = data

    return embeddings

def extract_embeddings(
    model,
    dataset_config,
    batch_size=32,
    num_frames=6
):
    # Get a list of unique utterances
    utterances = set()
    for line in open(dataset_config.trials):
        label, a, b = line.rstrip().split(' ')
        
        utterances.add(a)
        utterances.add(b)

    # Determine embeddings for each unique utterance
    embeddings = {}
    curr_batch_ids = []
    curr_batch_data = []
    for utterance in utterances:
        if len(curr_batch_ids) == batch_size:
            feats = extract_embeddings_from_batch(curr_batch_data, model)
            for i in range(len(curr_batch_ids)):
                uttid, data = curr_batch_ids[i], feats[i]
                embeddings[uttid] = data
            curr_batch_ids, curr_batch_data = [], []

        # Store current utterance id and data
        audio_path = os.path.join(dataset_config.base_path, 'voxceleb1', utterance)
        #data = np.load(audio_path, allow_pickle=True)
        data = load_audio(audio_path, dataset_config.frame_length, num_frames=num_frames)
        curr_batch_ids.append(utterance)
        curr_batch_data.append(data)

    # Register remaining samples (if nb samples % 64 != 0)
    if len(curr_batch_ids) != 0:
        feats = extract_embeddings_from_batch(curr_batch_data, model)
        for i in range(len(curr_batch_ids)):
            uttid, data = curr_batch_ids[i], feats[i]
            embeddings[uttid] = data
            

    return embeddings

def score_trials(trials_path, embeddings):
    scores, labels = [], []
    for line in open(trials_path):
        target, a, b = line.rstrip().split(' ')
        score = cosine_similarity(embeddings[a].reshape(1,-1), embeddings[b].reshape(1,-1)).item(0)
        score = (score + 1) / 2
        
        label = int(target)

        #score = 1 - cosine(embeddings[a], embeddings[b])
        #score = abs(score)
        #total = str(score)[:8] +" " +  a + " " + b
        scores.append(score)
        labels.append(label)

    return scores, labels

def score_trials22(trials_path, embeddings):
    scores, labels = [], []
    for line in open(trials_path):
        target, a, b = line.rstrip().split(' ')

        score = 1 - cosine(embeddings[a], embeddings[b])
        label = int(target)

        scores.append(score)
        labels.append(label)

    return scores, labels

def save_score(scores):
         with open('/media/user/Samsung_T5/sslsv/result/scores.txt', 'w', encoding='UTF-8') as f:
                  for i in range(len(scores)):
                           line = scores[i] +  '\n'
                           f.write(line)

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    #print(fpr, tpr, thresholds)
    fnr = 1 - tpr    
    idxE = np.nanargmin(np.abs(fnr - fpr))
    #idxE = np.abs(fnr - fpr).values
    #idxE = idxE.idxmin()
    eer  = max(fpr[idxE], fnr[idxE]) * 100
    return eer


def compute_error_rates(scores, labels):
      # Sort scores from smallest to largest.
      # Scores are the thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      labels = [labels[i] for i in sorted_indexes]
      
      # Determine false negative rates and false positive rates for each threshold.
      fnrs = []
      fprs = []
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i -1] + labels[i])
              fprs.append(fprs[i -1] + 1 - labels[i])

      fnrs_norm = sum(labels)
      #print("fn    : ", fnrs_norm)
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      fprs_norm = len(labels) - fnrs_norm
      #print("fp   : ",fprs_norm)
      fprs = [1 - x / float(fprs_norm) for x in fprs]

      return fnrs, fprs


def compute_min_dcf(fnrs, fprs, p_target=0.01, c_miss=1, c_fa=1):
    # Equations are from Section 3 of
    # NIST 2016 Speaker Recognition Evaluation Plan

    # Equation (2)
    min_c_det = float('inf')
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
    
    # Equations (3) and (4)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    return min_dcf


def evaluate(embeddings, trials):

    #print("111    ", embeddings)
    #print("222    ", trials)
    scores, labels = score_trials(trials, embeddings)
    #print(scores)
    #print(labels)

    eer = compute_eer(scores, labels)
    #print(eer)

    fnrs, fprs = compute_error_rates(scores, labels)
    min_dcf_001 = compute_min_dcf(fnrs, fprs, p_target=0.01)

    return { 'test_eer': eer, 'test_min_dcf_001': min_dcf_001 }
