"""
Bert-VITS TTS module for SillyTavern Extras

Reference:
    - https://github.com/fishaudio/Bert-VITS2
"""
import glob
import json
import os

import torch
from io import BytesIO
from flask import request, Response
from scipy.io import wavfile

from . import utils, commons
from .models import SynthesizerTrn
from .text import cleaned_text_to_sequence, get_bert
from .text.cleaner import clean_text
from .text.symbols import symbols

DEBUG_PREFIX = "<Bert-VITS2 module>"
hps = None
net_g = None
device = 'cpu'


def initialize(model_dir, gpu_mode=False):
    global hps, net_g, device

    if net_g is not None:
        print(DEBUG_PREFIX, "Already initialized")

    device = 'cuda' if gpu_mode else 'cpu'
    print(DEBUG_PREFIX, f"Initializing Bert-VITS2 module on {device}")
    hps = utils.get_hparams_from_file(os.path.join(model_dir, "config.json"))
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    ).to(device)

    net_g.eval()

    print(DEBUG_PREFIX, f"Loading checkpoint from {model_dir}...")
    checkpoints = sorted(glob.glob(os.path.join(model_dir, "G_*.pth")))
    if checkpoints:
        utils.load_checkpoint(checkpoints[-1], net_g, None, skip_optimizer=True)
    else:
        raise FileNotFoundError("No checkpoints `G_*.pth` found in model directory")
    print(DEBUG_PREFIX, "Initialization complete")


def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str, device)
    del word2ph
    assert bert.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JA":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language


def infer(text, speaker, language, sdp_ratio=0.2, noise_scale=0.5, noise_scale_w=0.9, length_scale=1.0):
    global hps, net_g

    if net_g is None:
        raise RuntimeError("Module not initialized")

    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps)
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        speakers = torch.LongTensor([hps.data.spk2id[speaker]]).to(device)
        audio = net_g.infer(
            x_tst,
            x_tst_lengths,
            speakers,
            tones,
            lang_ids,
            bert,
            ja_bert,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )[0][0, 0].data.cpu().float().numpy()

    return audio


def generate_tts():
    print(DEBUG_PREFIX, "Received TTS request for ", request.args)

    speaker = request.args.get("speaker")
    text = request.args.get("text").replace("\n", "")
    sdp_ratio = float(request.args.get("sdp_ratio", 0.2))
    noise = float(request.args.get("noise", 0.5))
    noisew = float(request.args.get("noisew", 0.9))
    length = float(request.args.get("length", 1.0))
    language = request.args.get("language", "ZH")
    if length >= 2:
        raise RuntimeError("Too big length")
    if len(text) >= 250:
        raise RuntimeError("Too long text")
    if None in (speaker, text):
        raise RuntimeError("Missing speaker or text")
    if language not in ("JA", "ZH"):
        raise RuntimeError("Invalid language")

    audio = infer(text, speaker, language, sdp_ratio, noise, noisew, length)

    with BytesIO() as wav:
        wavfile.write(wav, hps.data.sampling_rate, audio)
        torch.cuda.empty_cache()
        return Response(wav.getvalue(), mimetype='audio/wav')


def get_speakers():
    speakers = list(hps.data.spk2id.keys())
    return Response(json.dumps(speakers), mimetype='application/json')
