#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waveform-Mixer TTS Trainer
==========================

This single-file program does the following end-to-end:
1) Reads a CSV that maps reference audio filename, validation audio filename, and target text.
2) Infers speaker attributes (gender + simple pitch stats) from the reference audio using inaSpeechSegmenter + librosa.
3) Generates an initial synthetic audio for the target text, conditioned on detected gender (best-effort via Edge-TTS or Coqui TTS).
4) Slices both the reference audio and the initial synthetic audio into 0.1s chunks using two modes:
   - Mode 1: start at 0.0s, stride 0.1s
   - Mode 2: start at 0.05s, stride 0.1s (discard the first 0.05s)
   All slices are saved AS RAW WAVEFORM ARRAYS (.npy) — NO embeddings are saved.
5) Builds a waveform-mixing autoregressive model with 6 stacked “decoder-like” blocks,
   where each block = attention (over raw slices via cross-correlation) + linear layer.
   The model directly outputs raw waveform segments of 0.1s each. No fixed vocabulary is used.
6) Trains by looping over slice folders; for each folder, the model autoregressively generates an audio of the
   same duration as the validation audio (0.1s per step) and computes time-domain MSE loss against the validation audio.

Notes:
- This is research/prototype code meant to reflect the requested design precisely. It is intentionally explicit
  and avoids compressing slice waveforms into learned embeddings. Attention uses raw waveform cross-correlation.
- For TTS, the code tries Edge-TTS (internet) first; if unavailable, it falls back to Coqui TTS or finally a dumb tone
  so preprocessing can still proceed. Replace with your preferred offline TTS if needed.
- Dependencies (examples):
    pip install torch torchaudio librosa soundfile pandas numpy tqdm inaSpeechSegmenter edge-tts TTS
  Also ensure ffmpeg is installed for inaSpeechSegmenter.

Run examples:
  # Preprocess: generate initial synth audios and slices
  python waveform_mixer_tts_training.py preprocess \
    --csv data/manifest.csv \
    --ref_dir data/ref_wavs \
    --val_dir data/val_wavs \
    --synth_dir data/initial_synth \
    --slices_root data/slices \
    --sr 16000

  # Train for 5 epochs, saving checkpoints
  python waveform_mixer_tts_training.py train \
    --csv data/manifest.csv \
    --ref_dir data/ref_wavs \
    --val_dir data/val_wavs \
    --synth_dir data/initial_synth \
    --slices_root data/slices \
    --checkpoints runs/wmmixer \
    --epochs 5 --lr 1e-4 --batch_accum 1 --sr 16000

"""
from __future__ import annotations
import os
import sys
import io
import json
import math
import time
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# Audio I/O
try:
    import soundfile as sf
except Exception as e:
    sf = None

try:
    import librosa
except Exception as e:
    librosa = None

# Optional gender detection
try:
    from inaSpeechSegmenter import Segmenter
    from inaSpeechSegmenter.export_funcs import seg2csv
except Exception:
    Segmenter = None

# Optional TTS engines
EDGE_TTS_AVAILABLE = False
COQUI_TTS_AVAILABLE = False
try:
    import edge_tts  # async
    EDGE_TTS_AVAILABLE = True
except Exception:
    EDGE_TTS_AVAILABLE = False

try:
    from TTS.api import TTS as COQUI_TTS
    COQUI_TTS_AVAILABLE = True
except Exception:
    COQUI_TTS_AVAILABLE = False

# ---------- AISHELL-3 CSV builder ----------
# ---------- AISHELL-3 CSV builder (full-connect) ----------
import re

_CJK = re.compile(r'[\u4e00-\u9fff]')
_PUNC = set('，。？！、；：（）《》“”‘’—…,.!?;:()[]<>\"\'- ')


def _contains_han(s: str) -> bool:
    return any(_CJK.match(ch) for ch in s)

def _han_only_from_tokens(tokens):
    kept = []
    for tok in tokens:
        # 保留：含汉字的 token（通常是单字）或常见中英文标点
        if _contains_han(tok) or tok in _PUNC:
            kept.append(tok)
    return ''.join(kept)

def build_csv_for_aishell3(data_root: str, out_csv: str):
    train_wav_root = os.path.join(data_root, 'train', 'wav')
    test_wav_root  = os.path.join(data_root, 'test',  'wav')
    content_path   = os.path.join(data_root, 'test', 'content.txt')

    # 1) 收集 train/test 的 wav（相对路径），按 speaker 聚类
    def list_wavs_by_spk(root):
        spk2wavs = {}
        for spk in sorted(os.listdir(root)):
            spk_dir = os.path.join(root, spk)
            if not os.path.isdir(spk_dir):
                continue
            wavs = sorted([f for f in os.listdir(spk_dir) if f.lower().endswith('.wav')])
            if wavs:
                spk2wavs[spk] = [f"{spk}/{w}" for w in wavs]
        return spk2wavs

    train_by_spk = list_wavs_by_spk(train_wav_root)
    test_by_spk  = list_wavs_by_spk(test_wav_root)

    # 2) 读取 content.txt，建立 test 相对路径 -> 去拼音文本 的映射
    val_text = {}
    with open(content_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            first = parts[0]
            tokens = parts[1:]

            # 标准化成 test 相对路径（SSBxxxx/xxx.wav）
            if '/' in first:
                spk = first.split('/')[0]
                base = os.path.basename(first)
                val_rel = f"{spk}/{base}"
            else:
                base = first  # e.g., SSB00050353.wav
                spk  = os.path.splitext(base)[0][:7]  # 'SSB0005'
                val_rel = f"{spk}/{base}"

            # 去拼音：保留汉字与常见标点
            text = _han_only_from_tokens(tokens)
            val_text[val_rel] = text

    # 3) 对每个同时存在于 train 和 test 的 speaker 做全连接 (train×test)
    rows = []
    common_spk = sorted(set(train_by_spk.keys()) & set(test_by_spk.keys()))
    for spk in common_spk:
        train_wavs = train_by_spk[spk]
        test_wavs  = [v for v in test_by_spk[spk] if v in val_text]  # 只用 content.txt 里有标注的
        for ref_rel in train_wavs:
            for val_rel in test_wavs:
                rows.append([ref_rel, val_rel, val_text[val_rel]])

    pd.DataFrame(rows, columns=['ref_name', 'val_name', 'text']).to_csv(out_csv, index=False)

# ---------------------------
# Utility: audio load/save
# ---------------------------
def load_audio(path: str, sr: int) -> np.ndarray:
    """Load mono audio at given sample rate using soundfile+librosa fallback."""
    if sf is None:
        raise RuntimeError("soundfile is required: pip install soundfile")
    wav, file_sr = sf.read(path, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if librosa is not None and file_sr != sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    elif file_sr != sr:
        # naive resample if librosa missing
        ratio = sr / float(file_sr)
        idx = (np.arange(int(len(wav)*ratio)) / ratio).astype(np.float32)
        wav = np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)
        file_sr = sr
    wav = wav.astype(np.float32)
    # normalize to [-1, 1]
    mx = np.max(np.abs(wav)) + 1e-9
    wav = (wav / mx).astype(np.float32)
    return wav


def save_wav(path: str, wav: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if sf is None:
        raise RuntimeError("soundfile is required: pip install soundfile")
    sf.write(path, wav.astype(np.float32), sr)


# ---------------------------
# Gender + pitch estimation
# ---------------------------
def detect_gender_and_pitch(audio_path: str, sr: int) -> Tuple[str, Dict[str, float]]:
    """Return (gender, extras). Gender in {'M','F','U'}; extras may include pitch stats.
    Uses inaSpeechSegmenter if available; pitch via librosa.pyin if available.
    """
    gender = 'U'
    extras: Dict[str, float] = {}
    try:
        wav = load_audio(audio_path, sr)
        if Segmenter is not None:
            seg = Segmenter(vad_engine='smn', detect_gender=True)
            ann = seg(wav, sample_rate=sr)
            # ann is a list of tuples like ('female', start, end), ('male', start, end), ...
            dur_m = sum(e - s for (lab, s, e) in ann if lab == 'male')
            dur_f = sum(e - s for (lab, s, e) in ann if lab == 'female')
            if dur_m > dur_f and dur_m > 0:
                gender = 'M'
            elif dur_f > 0:
                gender = 'F'
        # pitch (rough)
        if librosa is not None:
            fmin, fmax = 50, 600
            try:
                f0, vflag, _ = librosa.pyin(wav, fmin=fmin, fmax=fmax, sr=sr)
                f0 = f0[~np.isnan(f0)]
                if f0.size > 0:
                    extras['f0_median'] = float(np.median(f0))
                    extras['f0_mean'] = float(np.mean(f0))
            except Exception:
                pass
    except Exception as e:
        print(f"[warn] gender/pitch detection failed on {audio_path}: {e}")
    return gender, extras


# ---------------------------
# Speaker info (gender) loader
# ---------------------------
def load_spk_info_gender(spk_info_path: Optional[str]) -> Dict[str, str]:
    """
    Parse spk-info.txt with format:
    speakerID  ageGroup  gender  accent
    - whitespace separated
    - lines starting with '#' are comments
    Returns dict {speakerID: 'M'|'F'|'U'}
    """
    mapping: Dict[str, str] = {}
    if not spk_info_path:
        return mapping
    if not os.path.exists(spk_info_path):
        print(f"[warn] spk-info not found: {spk_info_path}")
        return mapping

    with open(spk_info_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 3:
                continue
            spk, age_grp, gender = parts[0], parts[1], parts[2]
            g = gender.strip().lower()
            if g in ("female", "f", "woman", "girl"):
                mapping[spk] = "F"
            elif g in ("male", "m", "man", "boy"):
                mapping[spk] = "M"
            else:
                mapping[spk] = "U"
    print(f"[info] Loaded {len(mapping)} speaker genders from spk-info.txt")
    return mapping


# ---------------------------
# Initial TTS synthesis (best-effort)
# ---------------------------
import asyncio
import edge_tts

async def _synthesize_edge_tts_async(text: str, voice: str, sr: int, out_path: str):
    # 让服务端直接返回 16 kHz 16bit mono PCM（WAV RIFF）
    fmt = "riff-16khz-16bit-mono-pcm" if sr == 16000 else "riff-24khz-16bit-mono-pcm"
    comm = edge_tts.Communicate(text, voice=voice, output_format=fmt)
    with open(out_path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])

def synthesize_initial_tts(text: str, gender: str, sr: int, out_path: str, backend: Optional[str] = None) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if backend == "edge":
        if EDGE_TTS_AVAILABLE:
            try:
                voice = "zh-CN-YunxiNeural" if gender == "M" else "zh-CN-XiaoxiaoNeural"

                async def _edge_save_mp3(text, voice, mp3_path):
                    comm = edge_tts.Communicate(text, voice=voice)
                    await comm.save(mp3_path)

                tmp_mp3 = out_path + ".tmp.mp3"
                asyncio.run(_edge_save_mp3(text, voice, tmp_mp3))

                import librosa
                wav, _ = librosa.load(tmp_mp3, sr=sr, mono=True)
                wav = wav.astype(np.float32)
                sf.write(out_path, wav, sr)
                try: os.remove(tmp_mp3)
                except Exception: pass
                return "edge"
            except Exception as e:
                print(f"[warn] Edge-TTS failed: {e}")
        print("[warn] Edge-TTS unavailable, falling back to dummy")
        backend = "dummy"

    if backend == 'coqui':
        if COQUI_TTS_AVAILABLE:
            try:
                if gender == 'M':
                    model_name = "tts_models/en/vctk/vits"; speaker = "p243"
                else:
                    model_name = "tts_models/zh-CN/baker/tacotron2-DDC-GST"; speaker = None
                tts = COQUI_TTS(model_name)
                wav = np.asarray(tts.tts(text=text, speaker=speaker, speed=1.0), dtype=np.float32)
                if librosa is not None and tts.synthesizer.output_sample_rate != sr:
                    wav = librosa.resample(wav, tts.synthesizer.output_sample_rate, sr)
                save_wav(out_path, wav, sr)
                return "coqui"
            except Exception as e:
                print(f"[warn] Coqui TTS failed: {e}")
        print("[warn] Coqui TTS unavailable, falling back to dummy")
        backend = 'dummy'

    if backend == 'none':
        save_wav(out_path, np.zeros(int(sr*0.1), dtype=np.float32), sr)
        return "none"

    if backend == 'dummy':
        print("[warn] Falling back to dummy tone for initial synth")
        dur = max(1.0, min(10.0, len(text) / 6.0))
        t = np.linspace(0, dur, int(sr*dur), endpoint=False)
        freq = 140.0 if gender == 'M' else 220.0
        wav = 0.15 * np.sin(2*np.pi*freq*t).astype(np.float32)
        save_wav(out_path, wav, sr)
        return "dummy"


# ---------------------------
# Slicing
# ---------------------------
def slice_into_modes(wav: np.ndarray, sr: int, slice_sec: float = 0.1) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Return (mode1_slices, mode2_slices) where each slice has length slice_len samples.
    Mode1: start at 0s, step slice_sec. Mode2: start at 0.05s, step slice_sec (drop first 0.05s).
    The last partial chunk is dropped to keep all slices same length.
    """
    L = int(round(slice_sec * sr))
    hop = L
    # Mode 1
    n1 = len(wav) // L
    m1 = [wav[i*L:(i+1)*L].copy() for i in range(n1)]
    # Mode 2 (phase-shifted)
    offset = int(round(0.05 * sr))
    start = offset
    n2 = (len(wav) - start) // L
    m2 = [wav[start + i*L : start + (i+1)*L].copy() for i in range(n2)]
    return m1, m2

def save_ref_slices(folder: str, ref_m1: List[np.ndarray], ref_m2: List[np.ndarray]):
    os.makedirs(folder, exist_ok=True)
    for i, a in enumerate(ref_m1):
        np.save(os.path.join(folder, f"ref_mode1_{i:05d}.npy"), a.astype(np.float32))
    for i, a in enumerate(ref_m2):
        np.save(os.path.join(folder, f"ref_mode2_{i:05d}.npy"), a.astype(np.float32))

def save_synth_slices(folder: str, syn_m1: List[np.ndarray], syn_m2: List[np.ndarray]):
    os.makedirs(folder, exist_ok=True)
    for i, a in enumerate(syn_m1):
        np.save(os.path.join(folder, f"synth_mode1_{i:05d}.npy"), a.astype(np.float32))
    for i, a in enumerate(syn_m2):
        np.save(os.path.join(folder, f"synth_mode2_{i:05d}.npy"), a.astype(np.float32))

def save_slices_folder(folder: str,
                       ref_m1: List[np.ndarray], ref_m2: List[np.ndarray],
                       syn_m1: List[np.ndarray], syn_m2: List[np.ndarray]):
    os.makedirs(folder, exist_ok=True)
    # Save as raw waveforms .npy (explicitly NOT embeddings)
    def _save_group(prefix: str, arrs: List[np.ndarray]):
        for i, a in enumerate(arrs):
            np.save(os.path.join(folder, f"{prefix}_{i:05d}.npy"), a.astype(np.float32))
    _save_group('ref_mode1', ref_m1)
    _save_group('ref_mode2', ref_m2)
    _save_group('synth_mode1', syn_m1)
    _save_group('synth_mode2', syn_m2)

# ---------- Completeness & health checks ----------
def _count_files(folder: str, prefix: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(1 for fn in os.listdir(folder) if fn.startswith(prefix) and fn.endswith('.npy'))

def _expected_counts_from_len(n_samples: int, sr: int, slice_sec: float = 0.1) -> Tuple[int, int]:
    L = int(round(slice_sec * sr))
    n1 = n_samples // L
    offset = int(round(0.05 * sr))
    n2 = max(0, (n_samples - offset) // L)
    return n1, n2

def _is_good_wav(path: str, sr: int) -> bool:
    try:
        wav = load_audio(path, sr)
        if wav is None or wav.size < int(sr * 0.1):
            return False
        if not np.isfinite(wav).all():
            return False
        # 允许静音片段，但整段几乎全零算坏
        if np.max(np.abs(wav)) < 1e-5:
            return False
        return True
    except Exception:
        return False

def ref_slices_complete(ref_path: str, ref_folder: str, sr: int) -> bool:
    """Check if ref slices are already complete for given ref audio."""
    if not os.path.isdir(ref_folder):
        return False
    try:
        wav = load_audio(ref_path, sr)
    except Exception:
        return False
    n1, n2 = _expected_counts_from_len(len(wav), sr)
    c1 = _count_files(ref_folder, 'ref_mode1')
    c2 = _count_files(ref_folder, 'ref_mode2')
    meta_ok = os.path.exists(os.path.join(ref_folder, 'meta.json'))
    return (n1 > 0) and (c1 == n1) and (c2 == n2) and meta_ok

def synth_complete(synth_wav_path: str, synth_folder: str, sr: int,
                   require_meta: bool = True) -> Tuple[bool, Optional[str]]:
    """Check if synth wav + slices are complete. Returns (is_complete, backend_if_known)."""
    backend = None
    if not _is_good_wav(synth_wav_path, sr):
        return False, backend
    try:
        wav = load_audio(synth_wav_path, sr)
    except Exception:
        return False, backend
    n1, n2 = _expected_counts_from_len(len(wav), sr)
    c1 = _count_files(synth_folder, 'synth_mode1')
    c2 = _count_files(synth_folder, 'synth_mode2')
    meta_path = os.path.join(synth_folder, 'meta.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            backend = meta.get('synth_backend')
        except Exception:
            backend = None
    meta_ok = (not require_meta) or os.path.exists(meta_path)
    ok = os.path.isdir(synth_folder) and (n1 > 0) and (c1 == n1) and (c2 == n2) and meta_ok
    return ok, backend

# ====================== MP globals for preprocess ======================
# ======= MP globals for preprocess (split ref & val) =======
G_REF_DIR = None
G_VAL_DIR = None
G_SYNTH_DIR = None
G_SLICES_ROOT = None
G_SR = None
G_TTS_BACKEND = None
G_DETECT_GENDER = None
G_SPK_GENDER_MAP = {}
G_REF2TEXT = {}   # 仅作 ref→text 的占位（不再用于合成）
G_VAL2TEXT = {}   # 新增：val→text（用于初始合成）

def _mp_init_ref(ref_dir, slices_root, sr, tts_backend, detect_gender, spk_gender_map):
    global G_REF_DIR, G_SLICES_ROOT, G_SR, G_TTS_BACKEND, G_DETECT_GENDER, G_SPK_GENDER_MAP
    G_REF_DIR = ref_dir
    G_SLICES_ROOT = slices_root
    G_SR = sr
    G_TTS_BACKEND = tts_backend
    G_DETECT_GENDER = detect_gender
    G_SPK_GENDER_MAP = spk_gender_map or {}

def _mp_init_val(val_dir, synth_dir, slices_root, sr, tts_backend, detect_gender, spk_gender_map, val2text):
    global G_VAL_DIR, G_SYNTH_DIR, G_SLICES_ROOT, G_SR, G_TTS_BACKEND, G_DETECT_GENDER, G_SPK_GENDER_MAP, G_VAL2TEXT
    G_VAL_DIR = val_dir
    G_SYNTH_DIR = synth_dir
    G_SLICES_ROOT = slices_root
    G_SR = sr
    G_TTS_BACKEND = tts_backend
    G_DETECT_GENDER = detect_gender
    G_SPK_GENDER_MAP = spk_gender_map or {}
    G_VAL2TEXT = val2text or {}

def _mp_init(ref_dir, synth_dir, slices_root, sr, tts_backend, detect_gender, spk_gender_map, ref2text):
    """每个子进程启动时调用，设置本进程的全局配置。"""
    global G_REF_DIR, G_SYNTH_DIR, G_SLICES_ROOT, G_SR, G_TTS_BACKEND, G_DETECT_GENDER, G_SPK_GENDER_MAP, G_REF2TEXT
    G_REF_DIR = ref_dir
    G_SYNTH_DIR = synth_dir
    G_SLICES_ROOT = slices_root
    G_SR = sr
    G_TTS_BACKEND = tts_backend
    G_DETECT_GENDER = detect_gender
    G_SPK_GENDER_MAP = spk_gender_map or {}
    G_REF2TEXT = ref2text or {}

def process_one_ref(ref_name: str):
    ref_path = os.path.join(G_REF_DIR, ref_name)
    ref_stem = os.path.splitext(os.path.basename(ref_name))[0]
    ref_folder = os.path.join(G_SLICES_ROOT, "ref", ref_stem)
    meta_path = os.path.join(ref_folder, "meta.json")

    # 已完整？→ 跳过
    try:
        if ref_slices_complete(ref_path, ref_folder, G_SR):
            return True
    except Exception:
        pass

    # 重新切片
    ref_wav = load_audio(ref_path, G_SR)
    ref_m1, ref_m2 = slice_into_modes(ref_wav, G_SR)
    save_ref_slices(ref_folder, ref_m1, ref_m2)

    spk = ref_name.split('/')[0].split('\\')[0]
    gender = G_SPK_GENDER_MAP.get(spk, "U") if not G_DETECT_GENDER else detect_gender_and_pitch(ref_path, G_SR)[0]
    meta = {
        "ref_audio": os.path.abspath(ref_path),
        "sample_rate": G_SR, "slice_len": int(round(0.1*G_SR)),
        "speaker_id": spk, "gender": gender, "kind": "ref",
        "status": "ok"
    }
    os.makedirs(ref_folder, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return True


def process_one_val(val_name: str):
    val_path = os.path.join(G_VAL_DIR, val_name)
    val_stem = os.path.splitext(os.path.basename(val_name))[0]
    synth_path = os.path.join(G_SYNTH_DIR, f"{val_stem}.wav")
    synth_folder = os.path.join(G_SLICES_ROOT, "synth", val_stem)
    meta_path = os.path.join(synth_folder, "meta.json")

    # 读文本与性别来源
    text = G_VAL2TEXT.get(val_name, "你好")
    spk = val_name.split('/')[0].split('\\')[0]

    # 已完整？→ 跳过（允许根据 meta 中 backend 判断是否需要重生成）
    complete, backend_old = synth_complete(synth_path, synth_folder, G_SR)
    if complete and backend_old != "dummy":
        return True

    # 需要重新合成或修复切片
    if G_DETECT_GENDER:
        gender, extras = detect_gender_and_pitch(val_path, G_SR)
        gender_src = "detector(val)"
    else:
        gender, extras = G_SPK_GENDER_MAP.get(spk, "U"), {}
        gender_src = "spk-info" if spk in G_SPK_GENDER_MAP else "unknown"

    # 若 wav 不存在或损坏，则重合成
    backend_used = backend_old if complete else None
    if not _is_good_wav(synth_path, G_SR):
        backend_used = synthesize_initial_tts(text if text else "你好", gender, G_SR, synth_path, backend=G_TTS_BACKEND)

    # 重新读取合成音频并切片（无论是刚合成还是只是缺切片）
    synth_wav = load_audio(synth_path, G_SR)
    syn_m1, syn_m2 = slice_into_modes(synth_wav, G_SR)
    save_synth_slices(synth_folder, syn_m1, syn_m2)

    meta = {
        "val_audio": os.path.abspath(val_path),
        "synth_audio": os.path.abspath(synth_path),
        "text": text, "speaker_id": spk,
        "gender": gender, "gender_source": gender_src,
        "sample_rate": G_SR, "slice_len": int(round(0.1*G_SR)),
        "synth_backend": backend_used or G_TTS_BACKEND,
        "kind": "synth",
        "status": "ok"
    }
    os.makedirs(synth_folder, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return True



# ---------------------------
# Preprocess pipeline
# ---------------------------
def preprocess(csv_path: str, ref_dir: str, val_dir: str,
               synth_dir: str, slices_root: str, sr: int,
               tts_backend: str = 'dummy',
               detect_gender: bool = False,
               num_workers: int = 0,
               spk_info: Optional[str] = None):

    df = pd.read_csv(csv_path)
    col_ref, col_val, col_text = df.columns[:3]
    print(f"[info] Using CSV columns: ref='{col_ref}', val='{col_val}', text='{col_text}'")

    os.makedirs(synth_dir, exist_ok=True)
    os.makedirs(slices_root, exist_ok=True)

    # speaker -> gender（当 detect_gender=False 时使用）
    spk_gender_map = load_spk_info_gender(spk_info) if not detect_gender else {}

    # 唯一集合
    unique_refs = sorted(set(df[col_ref].astype(str).tolist()))
    unique_vals = sorted(set(df[col_val].astype(str).tolist()))

    # 为每个 val 取“第一条文本”
    first_rows_val = df.groupby(col_val, as_index=False).first()
    val2text = dict(zip(first_rows_val[col_val].astype(str), first_rows_val[col_text].astype(str)))

    # === Phase 1: 只处理 ref → 切 ref_mode{1,2} 到 slices_root/ref/{ref_stem} ===
    if num_workers and num_workers > 0:
        from multiprocessing import Pool
        with Pool(
            processes=num_workers,
            initializer=_mp_init_ref,
            initargs=(ref_dir, slices_root, sr, tts_backend, detect_gender, spk_gender_map),
        ) as pool:
            for _ in tqdm(pool.imap_unordered(process_one_ref, unique_refs),
                          total=len(unique_refs), desc="Preprocess [REF slices]"):
                pass
    else:
        _mp_init_ref(ref_dir, slices_root, sr, tts_backend, detect_gender, spk_gender_map)
        for ref_name in tqdm(unique_refs, desc="Preprocess [REF slices]"):
            process_one_ref(ref_name)

    # === Phase 2: 只处理 val → 合成 & 切 synth_mode{1,2} 到 slices_root/synth/{val_stem} ===
    if num_workers and num_workers > 0:
        from multiprocessing import Pool
        with Pool(
            processes=num_workers,
            initializer=_mp_init_val,
            initargs=(val_dir, synth_dir, slices_root, sr, tts_backend, detect_gender, spk_gender_map, val2text),
        ) as pool:
            for _ in tqdm(pool.imap_unordered(process_one_val, unique_vals),
                          total=len(unique_vals), desc="Preprocess [SYNTH by VAL]"):
                pass
    else:
        _mp_init_val(val_dir, synth_dir, slices_root, sr, tts_backend, detect_gender, spk_gender_map, val2text)
        for val_name in tqdm(unique_vals, desc="Preprocess [SYNTH by VAL]"):
            process_one_val(val_name)





# ---------------------------
# Model components
# ---------------------------
class TextCondEncoder(nn.Module):
    """Map a Unicode string (sequence of codepoints) to a small continuous vector WITHOUT a fixed vocab.
    Implementation: codepoints normalized to [0,1] -> 1D conv -> global pooling.
    """
    def __init__(self, out_dim: int = 64, max_len: int = 256):
        super().__init__()
        self.max_len = max_len
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, out_dim, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, texts: List[str]) -> torch.Tensor:
        # batch of strings -> (B, 1, T)
        arrs = []
        for s in texts:
            cps = [ord(ch) for ch in s]
            cps = cps[:self.max_len]
            if len(cps) == 0:
                cps = [0]
            arr = np.array(cps, dtype=np.float32)
            if arr.size < self.max_len:
                arr = np.pad(arr, (0, self.max_len - arr.size), mode='constant')
            arr = arr / 65535.0
            arrs.append(arr[None, :])

        device = self.conv1.weight.device  # 关键：与卷积权重同设备
        dtype  = self.conv1.weight.dtype

        # 直接在目标设备上创建
        x = torch.as_tensor(np.stack(arrs, axis=0), dtype=dtype, device=device)  # (B, 1, T)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return x


class DecoderBlock(nn.Module):
    """One decoder-like block = attention over raw slices (via cross-correlation) + linear projection.
    - Query: previous generated 0.1s waveform segment (L)
    - Memory: all input slices (S, L) from ref & initial synth, both slicing modes
    - Text condition: vector used to modulate attention logits (FiLM-like)
    The block outputs a new segment (L), intended to be the deformed, mixed result of multiple slices.
    """
    def __init__(self, slice_len: int, text_dim: int = 64):
        super().__init__()
        self.slice_len = slice_len
        # Linear (time-wise) projection; keep it strictly linear as requested
        self.linear = nn.Linear(slice_len, slice_len, bias=True)
        # Small 1D conv for deformation of the mixed signal (still linear w.r.t. input)
        self.defconv = nn.Conv1d(1, 1, kernel_size=9, padding=4, bias=False)
        # Text gating to tweak attention logits
        self.text_to_logit_bias = nn.Linear(text_dim, 1)
        self.text_to_mix_gain = nn.Linear(text_dim, 1)
        self.norm = nn.LayerNorm(slice_len)

    def forward(self, prev_seg: torch.Tensor, slices: torch.Tensor,
                slices_mask: torch.Tensor, text_cond: torch.Tensor) -> torch.Tensor:
        """
        prev_seg:   (B, L)
        slices:     (B, S, L)
        slices_mask:(B, S)  True=有效, False=padding
        text_cond:  (B, C)
        """
        B, S, L = slices.shape
        dot = torch.einsum('bl,bsl->bs', prev_seg, slices) / math.sqrt(L)
        bias = self.text_to_logit_bias(text_cond)  # (B,1)
        logits = dot + bias

        if slices_mask is not None:
            logits = logits.masked_fill(~slices_mask, float('-inf'))  # 屏蔽 padding 切片

        attn = torch.softmax(logits, dim=-1)  # (B,S)
        mixed = torch.einsum('bs,bsl->bl', attn, slices)
        gain = torch.sigmoid(self.text_to_mix_gain(text_cond))
        mixed = mixed * gain

        x = self.defconv(mixed.unsqueeze(1)).squeeze(1)
        x = x + prev_seg
        x = self.linear(self.norm(x))
        return x


class WaveformMixerModel(nn.Module):
    def __init__(self, slice_len: int, n_layers: int = 6, text_dim: int = 64):
        super().__init__()
        self.slice_len = slice_len
        self.text_enc = TextCondEncoder(out_dim=text_dim)
        self.layers = nn.ModuleList([DecoderBlock(slice_len, text_dim) for _ in range(n_layers)])

    def forward_generate(self, slices: torch.Tensor, slices_mask: torch.Tensor,
                         texts: List[str], total_steps: int) -> torch.Tensor:
        device = slices.device
        B, S, L = slices.shape
        text_cond = self.text_enc(texts).to(device)
        seg = torch.zeros(B, L, device=device, dtype=slices.dtype)
        outs = []
        for _ in range(total_steps):
            x = seg
            for layer in self.layers:
                x = layer(x, slices, slices_mask, text_cond)
            seg = x
            outs.append(seg)
        return torch.cat(outs, dim=-1)  # (B, total_steps*L)


# ---------------------------
# Dataset loader for slice folders
# ---------------------------
class SliceFolderDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, ref_dir: str, val_dir: str, slices_root: str, sr: int):
        self.df = pd.read_csv(csv_path)
        self.col_ref, self.col_val, self.col_text = self.df.columns[:3]
        self.ref_dir = ref_dir
        self.val_dir = val_dir
        self.slices_root = slices_root
        self.sr = sr
        self.slice_len = int(round(0.1 * sr))

    def __len__(self):
        return len(self.df)

    def _load_slices_stack_pair(self, ref_folder: str, synth_folder: str) -> np.ndarray:
        def _collect(folder: str, prefixes: List[str]) -> List[np.ndarray]:
            arrs = []
            for prefix in prefixes:
                pat = [fn for fn in sorted(os.listdir(folder)) if fn.startswith(prefix) and fn.endswith('.npy')]
                for fn in pat:
                    a = np.load(os.path.join(folder, fn)).astype(np.float32)
                    a = a[:self.slice_len] if a.shape[0] >= self.slice_len else np.pad(a, (0, self.slice_len - a.shape[0]))
                    arrs.append(a)
            return arrs

        if not os.path.isdir(ref_folder):
            raise RuntimeError(f"Ref slices folder not found: {ref_folder}")
        if not os.path.isdir(synth_folder):
            raise RuntimeError(f"Synth slices folder not found: {synth_folder}")

        ref_arrs  = _collect(ref_folder,  ['ref_mode1', 'ref_mode2'])
        synth_arrs= _collect(synth_folder,['synth_mode1', 'synth_mode2'])
        if not ref_arrs or not synth_arrs:
            raise RuntimeError(f"Empty slices: ref={len(ref_arrs)} synth={len(synth_arrs)}")
        stack = np.stack(ref_arrs + synth_arrs, axis=0)  # (S, L)
        return stack

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        ref_name = str(row[self.col_ref])
        val_name = str(row[self.col_val])
        text = str(row[self.col_text])

        ref_stem = os.path.splitext(os.path.basename(ref_name))[0]
        val_stem = os.path.splitext(os.path.basename(val_name))[0]

        ref_folder = os.path.join(self.slices_root, "ref", ref_stem)
        synth_folder = os.path.join(self.slices_root, "synth", val_stem)

        slices = self._load_slices_stack_pair(ref_folder, synth_folder)  # (S,L)

        val_path = os.path.join(self.val_dir, val_name)
        val_wav = load_audio(val_path, self.sr)

        L = self.slice_len
        T = int(math.ceil(len(val_wav) / L))
        target = np.pad(val_wav, (0, T*L - len(val_wav)))

        meta = {
            'text': text,
            'T': T,
            'orig_len': len(val_wav),
            'ref_stem': ref_stem,
            'val_stem': val_stem,
        }
        return slices.astype(np.float32), target.astype(np.float32), meta


from typing import Sequence
import numpy as np
import torch

def make_collate_fn(slice_len: int, S_max: int):
    def collate_fn(batch: Sequence):
        # batch: list of (slices_np(S,L), target_np(T*L), meta_dict)
        B = len(batch)
        # 统计 T 与 S
        T_list = [int(b[2]['T']) for b in batch]
        T_max = max(T_list)
        S_list = [b[0].shape[0] for b in batch]
        S_cap_list = [min(s, S_max) for s in S_list]
        S_pad = max(S_cap_list)

        L = slice_len
        slices_tensor = torch.zeros(B, S_pad, L, dtype=torch.float32)
        slices_mask   = torch.zeros(B, S_pad, dtype=torch.bool)
        targets       = torch.zeros(B, T_max*L, dtype=torch.float32)
        time_mask     = torch.zeros(B, T_max, dtype=torch.bool)
        texts: List[str] = []

        for i, (s_np, tgt_np, meta) in enumerate(batch):
            s = s_np  # (S,L)
            S_i = s.shape[0]
            k = min(S_i, S_max)
            # 随机子采样（或全部使用）
            if S_i > k:
                idx = np.random.choice(S_i, size=k, replace=False)
                s = s[idx]
            else:
                k = S_i
            slices_tensor[i, :k] = torch.from_numpy(s[:k])
            slices_mask[i, :k] = True

            T_i = int(meta['T'])
            targets[i, :T_i*L] = torch.from_numpy(tgt_np[:T_i*L])
            time_mask[i, :T_i] = True

            texts.append(meta['text'])

        return {
            "slices": slices_tensor,      # (B,S_pad,L)
            "slices_mask": slices_mask,   # (B,S_pad)
            "targets": targets,           # (B,T_max*L)
            "time_mask": time_mask,       # (B,T_max)
            "texts": texts,               # list[str] 长度 B
            "T_max": T_max,
        }
    return collate_fn

# ---------------------------
# Training loop
# ---------------------------
@dataclass
class TrainConfig:
    csv: str
    ref_dir: str
    val_dir: str
    slices_root: str
    checkpoints: str
    epochs: int = 5
    lr: float = 1e-4
    batch_accum: int = 1
    sr: int = 16000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    grad_clip: float = 1.0
    train_split: float = 0.7   # 新增：训练/验证划分比例
    seed: int = 1337           # 新增：随机种子（用于可复现实验/划分）
    batch_size: int = 16         # 新增：真实并行的 batch
    max_slices: int = 512        # 新增：每个样本最多用多少个切片(S)；多的随机采样、少的零填充
    amp: bool = True             # 新增：是否用自动混合精度

def train(cfg: TrainConfig):
    # ====== 初始化 ======
    os.makedirs(cfg.checkpoints, exist_ok=True)

    # 全局随机种子
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    ds = SliceFolderDataset(cfg.csv, cfg.ref_dir, cfg.val_dir, cfg.slices_root, cfg.sr)
    slice_len = ds.slice_len
    device_is_cuda = str(cfg.device).startswith('cuda') and torch.cuda.is_available()

    model = WaveformMixerModel(slice_len=slice_len, n_layers=6, text_dim=64).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # ✅ 新 API（可消除你看到的 FutureWarning）
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and device_is_cuda)
        autocast_ctx = lambda: torch.amp.autocast('cuda', enabled=cfg.amp and device_is_cuda)
    except Exception:
        # 兼容旧版本 PyTorch
        scaler = torch.cuda.amp.GradScaler('cuda',enabled=cfg.amp and device_is_cuda)
        autocast_ctx = lambda: torch.cuda.amp.autocast('cuda',enabled=cfg.amp and device_is_cuda)

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    # ====== ✅ 划分 train/val 索引（之前缺失，导致 NameError） ======
    n_total = len(ds)
    if n_total < 2:
        raise RuntimeError(f"Dataset too small: {n_total} samples")

    idx_all = np.arange(n_total)
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(idx_all)
    n_train = max(1, int(round(n_total * float(cfg.train_split))))
    n_train = min(n_train, n_total - 1)  # 确保 val 至少 1 个
    train_idx = idx_all[:n_train].tolist()
    val_idx   = idx_all[n_train:].tolist()

    # ====== 其余准备 ======
    collate = make_collate_fn(slice_len, cfg.max_slices)

    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4,
        collate_fn=collate
    )
    val_loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size,
        sampler=torch.utils.data.SequentialSampler(val_idx),
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4,
        collate_fn=collate
    )

    best_val = None          # ✅ 新增：记录最好验证集损失
    global_step = 0          # ✅ 新增：记录优化步数
    grad_acc = max(1, cfg.batch_accum)

    # ====== 训练多个 epoch ======
    for epoch in range(1, cfg.epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        num_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]")
        for step, batch in enumerate(pbar, start=1):
            slices = batch["slices"].to(cfg.device, non_blocking=True)        # (B,S,L)
            slices_mask = batch["slices_mask"].to(cfg.device, non_blocking=True)
            targets = batch["targets"].to(cfg.device, non_blocking=True)      # (B,T*L)
            time_mask = batch["time_mask"].to(cfg.device, non_blocking=True)  # (B,T)
            texts = batch["texts"]
            T_max = int(batch["T_max"])
            L = slice_len

            with autocast_ctx():
                pred = model.forward_generate(slices, slices_mask, texts, total_steps=T_max)  # (B,T*L)
                time_mask_pts = time_mask.repeat_interleave(L, dim=1)                         # (B,T*L)
                mse = (pred - targets) ** 2
                loss = (mse * time_mask_pts).sum() / time_mask_pts.sum().clamp_min(1)

            loss_to_back = loss / grad_acc
            if scaler.is_enabled():
                scaler.scale(loss_to_back).backward()
            else:
                loss_to_back.backward()

            if step % grad_acc == 0:
                if cfg.grad_clip and cfg.grad_clip > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += float(loss.detach().cpu())
            num_train_batches += 1
            pbar.set_postfix({
                "train_loss": f"{running_loss / num_train_batches:.4f}",
                "B": slices.shape[0], "S": slices.shape[1], "T_max": T_max
            })

        avg_train = running_loss / max(1, num_train_batches)

        # ---- Val ----
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad(), autocast_ctx():
            for batch in val_loader:
                slices = batch["slices"].to(cfg.device, non_blocking=True)
                slices_mask = batch["slices_mask"].to(cfg.device, non_blocking=True)
                targets = batch["targets"].to(cfg.device, non_blocking=True)
                time_mask = batch["time_mask"].to(cfg.device, non_blocking=True)
                texts = batch["texts"]
                T_max = int(batch["T_max"])
                L = slice_len

                pred = model.forward_generate(slices, slices_mask, texts, total_steps=T_max)
                time_mask_pts = time_mask.repeat_interleave(L, dim=1)
                mse = (pred - targets) ** 2
                loss = (mse * time_mask_pts).sum() / time_mask_pts.sum().clamp_min(1)

                val_loss_sum += float(loss.detach().cpu())
                val_batches += 1

        avg_val = val_loss_sum / max(1, val_batches)
        print(f"[epoch {epoch}] train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        # ---- 保存 ckpt ----
        ckpt = {
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'slice_len': slice_len,
            'train_loss': avg_train,
            'val_loss': avg_val,
            'split': {'train': len(train_idx), 'val': len(val_idx), 'ratio': cfg.train_split},
            'seed': cfg.seed,
        }
        torch.save(ckpt, os.path.join(cfg.checkpoints, f"ckpt_epoch_{epoch:03d}.pt"))
        if best_val is None or avg_val < best_val:
            best_val = avg_val
            torch.save(ckpt, os.path.join(cfg.checkpoints, "ckpt_best.pt"))




# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waveform-Mixer TTS Training")
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_prep = sub.add_parser('preprocess', help='Generate initial synth and slice dataset')
    p_prep.add_argument('--csv', required=True, help='CSV path with columns: ref_name, val_name, text')
    p_prep.add_argument('--ref_dir', required=True)
    p_prep.add_argument('--val_dir', required=True)
    p_prep.add_argument('--synth_dir', required=True)
    p_prep.add_argument('--slices_root', required=True)
    p_prep.add_argument('--sr', type=int, default=16000)
    p_prep.add_argument('--tts_backend', choices=['edge', 'coqui', 'dummy', 'none'], default='dummy')
    p_prep.add_argument('--detect_gender', action='store_true')
    p_prep.add_argument('--num_workers', type=int, default=0)
    p_prep.add_argument('--spk_info', type=str, default=None,
                        help='Path to spk-info.txt (used when --detect_gender is OFF)')

    p_train = sub.add_parser('train', help='Train the waveform mixer model')
    p_train.add_argument('--csv', required=True)
    p_train.add_argument('--ref_dir', required=True)
    p_train.add_argument('--val_dir', required=True)
    p_train.add_argument('--slices_root', required=True)
    p_train.add_argument('--checkpoints', required=True)
    p_train.add_argument('--epochs', type=int, default=5)
    p_train.add_argument('--lr', type=float, default=1e-4)
    p_train.add_argument('--batch_accum', type=int, default=2)
    p_train.add_argument('--sr', type=int, default=16000)
    p_build = sub.add_parser('build_csv', help='Build (ref,val,text) CSV from AISHELL-3 layout')
    p_build.add_argument('--data_root', required=True)
    p_build.add_argument('--out_csv', required=True)
    p_train.add_argument('--train_split', type=float, default=0.7)
    p_train.add_argument('--seed', type=int, default=1337)
    p_train.add_argument('--batch_size', type=int, default=16)
    p_train.add_argument('--max_slices', type=int, default=512)
    p_train.add_argument('--no_amp', action='store_true')


    args = parser.parse_args()

    if args.cmd == 'preprocess':
        preprocess(csv_path=args.csv,
                   ref_dir=args.ref_dir,
                   val_dir=args.val_dir,
                   synth_dir=args.synth_dir,
                   slices_root=args.slices_root,
                   sr=args.sr,
                   tts_backend=args.tts_backend,
                   detect_gender=args.detect_gender,
                   num_workers=args.num_workers,
                   spk_info=args.spk_info)  # 新增

    elif args.cmd == 'train':
        cfg = TrainConfig(csv=args.csv,
                          ref_dir=args.ref_dir,
                          val_dir=args.val_dir,
                          slices_root=args.slices_root,
                          checkpoints=args.checkpoints,
                          epochs=args.epochs,
                          lr=args.lr,
                          batch_accum=args.batch_accum,
                          sr=args.sr,
                          train_split = args.train_split,  # 新增
                          seed = args.seed,
                          batch_size = args.batch_size,
                          max_slices = args.max_slices,
                          amp = (not args.no_amp)
                          )  # 新增
        train(cfg)
    elif args.cmd == 'build_csv':
        build_csv_for_aishell3(args.data_root, args.out_csv)
