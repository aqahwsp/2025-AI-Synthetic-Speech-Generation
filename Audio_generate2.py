#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 批处理脚本：
- 读取 ./Task/ 下的 csv（包含列：utt, reference_speech, text）
- 对每条数据进行：降噪/人声保留 -> 参考音频长度补偿(按题意规则) ->
  * 短样本(<=10s)：零样本生成
  * 长样本(>10s)：拼接至>60s后触发“小微调”（尽力调用 WebUI 同款训练入口），再合成
- 结果写入 ./result/synthesized_speech_{utt}.wav，并生成 ./result/result.csv

注意：
1) 本脚本优先调用 GPT-SoVITS 自带的降噪工具（tools/cmd-denoise.py 或 UVR5）进行“仅保留人声”。
   若环境缺失，则回退到简易频谱门限降噪（效果有限，只为兜底）。
2) “小微调”环节尝试复用 webui.py 里的 open1abc / open1Ba / open1Bb 流程（需要预训练权重、依赖等已就绪）。
   如果训练入口不可用，则自动回退到零样本推理，保证流程不中断。
3) 推理阶段调用 GPT_SoVITS.inference_webui.get_tts_wav（无参考文本 ref_free 模式）。
   该模式在 v1/v2/v2Pro 系列可用；若当前模型版本不支持 ref_free，会自动给参考文本占位。
"""
import sys
import os
BASE = os.path.dirname(os.path.abspath(__file__))  # .../Audiogenerate/
PRE  = os.path.join(BASE, "GPT_SoVITS_main", "GPT_SoVITS", "pretrained_models")

# BERT / HuBERT (必须是本地已下载好的目录)
os.environ["bert_path"] = os.path.join(PRE, "chinese-roberta-wwm-ext-large")
os.environ["cnhubert_base_path"] = os.path.join(PRE, "chinese-hubert-base")

# 指定 GPT / SoVITS 权重（本地文件）
os.environ["gpt_path"]    = os.path.join(PRE, "s1v3.ckpt")
os.environ["sovits_path"] = os.path.join(PRE, "v2Pro", "s2Gv2ProPlus.pth")
os.environ["sv_path"]     = os.path.join(PRE, "sv", "pretrained_eres2netv2w24s4ep4.ckpt")
# 可选：彻底离线（即使联网也不去 HuggingFace）
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRA_PATHS = [
    BASE_DIR,                                        # GPT_SoVITS_main/
    os.path.join(BASE_DIR, "GPT_SoVITS_main"),
    os.path.join(BASE_DIR, "GPT_SoVITS_main", "GPT_SoVITS"),
    os.path.join(BASE_DIR, "GPT_SoVITS_main", "GPT_SoVITS", "pretrained_models"),
    os.path.join(BASE_DIR, "GPT_SoVITS_main", "GPT_SoVITS", "eres2net"),
    os.path.join(BASE_DIR, "GPT_SoVITS_main", "GPT_SoVITS", "BigVGAN"),
    os.path.join(BASE_DIR, "GPT_SoVITS_main", "tools"),
    os.path.join(BASE_DIR, "GPT_SoVITS_main", "tools", "uvr5"),
]
for p in EXTRA_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)

# ===== 1) 放在文件顶部常量/工具函数区 =====

def ensure_nltk_data(nltk_dir: str):
    try:
        import nltk, zipfile, glob, shutil
    except Exception as e:
        print("[warn] nltk 未安装，跳过 NLTK 资源准备。"); return

    os.makedirs(nltk_dir, exist_ok=True)
    nltk.data.path.insert(0, nltk_dir)

    needed = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for probe, pkg in needed:
        try:
            nltk.data.find(probe)
        except Exception as e:
            # 可能是 BadZipFile / LookupError 等：先清理，再重下
            base = probe.split("/")[0]  # e.g. "tokenizers" 或 "taggers"
            try:
                for z in glob.glob(os.path.join(nltk_dir, base + "*.zip")):
                    os.remove(z)
                shutil.rmtree(os.path.join(nltk_dir, base), ignore_errors=True)
            except Exception:
                pass
            ok = False
            try:
                ok = nltk.download(pkg, download_dir=nltk_dir, quiet=True)
            except Exception as _:
                pass
            if not ok:
                print(f"[warn] NLTK 资源 {pkg} 下载失败，继续运行（可能不影响推理）。")



# ==== 自动定位仓库根目录 & 脚本文件 ====
def _find_repo_root():
    # 兼容两种常见布局：项目根即仓库根；或仓库放在 GPT_SoVITS_main/ 下
    candidates = [
        BASE,
        os.path.join(BASE, "GPT_SoVITS_main"),
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "GPT_SoVITS")) and os.path.isdir(os.path.join(cand, "tools")):
            return cand
    return BASE

REPO_ROOT = _find_repo_root()

def _find_file(name: str):
    """在 REPO_ROOT 下优先按常见位置找脚本文件，不行再递归搜索"""
    common = [
        os.path.join(BASE, name),
        os.path.join(REPO_ROOT, name),
        os.path.join(REPO_ROOT, "GPT_SoVITS", name),
        os.path.join(REPO_ROOT, "tools", name),  # 例如 tools/xxx.py
    ]
    for p in common:
        if os.path.isfile(p):
            return p
    try:
        matches = glob.glob(os.path.join(REPO_ROOT, "**", name), recursive=True)
        for m in matches:
            if os.path.isfile(m):
                return m
    except Exception:
        pass
    return None

os.environ["version"] = "v2ProPlus"  # inference_webui 会读取这个环境变量

from config import pretrained_sovits_name, pretrained_gpt_name, infer_device

import io
import re
import json
import math
import glob
import time
import shutil
import warnings
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import soundfile as sf

# 尽量延后导入重型依赖，避免在无环境时直接报错
try:
    import librosa
except Exception:
    librosa = None

# ---------- 基础路径 ----------
TASK_DIR = "./Task"
RESULT_DIR = "./result"
WORK_DIR = "./_work_gsv"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# ---------- 工具函数 ----------
def log(msg: str):
    print(f"[gsv-pipeline] {msg}", flush=True)

def sec_of(wav_path: str) -> float:
    try:
        info = sf.info(wav_path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        # 兜底用 librosa
        if librosa is not None:
            y, sr = librosa.load(wav_path, sr=None, mono=True)
            return len(y) / sr
        raise

def load_mono(wav_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """以 float32 单声道读取"""
    data, sr = sf.read(wav_path, always_2d=False)
    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = np.mean(data, axis=1)
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    if target_sr is not None and librosa is not None and sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return data, sr

def save_wav(path: str, data: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, data, sr)

def simple_noise_reduce(y: np.ndarray, sr: int, prop=0.1) -> np.ndarray:
    """非常简易的频谱门限降噪（兜底方案，仅当官方工具不可用时启用）。
       prop: 取前 prop 比例的音频估计噪声阈值"""
    if librosa is None:
        # 没有 librosa 就原样返回
        return y
    n = max(1, int(len(y) * prop))
    noise_sample = y[:n]
    # 估算噪声均值/方差
    noise_std = np.std(noise_sample) + 1e-6
    # 简单硬阈值
    return np.where(np.abs(y) < 2.5 * noise_std, 0.0, y)

def detect_language(text: str) -> str:
    """非常粗糙的语言猜测，用于 .list 生成：返回 'zh'|'ja'|'en'|..."""
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[\u3040-\u30ff]", text):  # 日文假名
        return "ja"
    # 其他一律当英文
    return "en"

def repeat_to_between(y: np.ndarray, sr: int, min_sec: float, max_sec: float) -> np.ndarray:
    """把音频 y 循环拼接至时长落在 [min_sec, max_sec]，尽量不超过 max_sec"""
    cur = len(y) / sr
    if cur >= min_sec and cur <= max_sec:
        return y
    # 循环拼接
    rep = 1
    while cur * rep < min_sec:
        rep += 1
    y_rep = np.tile(y, rep)
    # 若超出 max_sec，做裁剪
    max_len = int(max_sec * sr)
    if len(y_rep) > max_len:
        y_rep = y_rep[:max_len]
    return y_rep

def repeat_to_over(y: np.ndarray, sr: int, over_sec: float) -> np.ndarray:
    """把音频 y 循环拼接到 > over_sec"""
    rep = int(math.ceil((over_sec * sr) / max(1, len(y))))
    return np.tile(y, rep)

def is_valid_wav(path: str, min_frames: int = 1) -> bool:
    """快速检查 wav 是否“生成完成”：能读到合法头且帧数>0"""
    try:
        info = sf.info(path)
        return (info.frames or 0) >= min_frames and (info.samplerate or 0) > 0
    except Exception:
        return False


# ---------- 优先调用 GPT-SoVITS 内置降噪/人声保留 ----------
def have_gsv_denoise() -> bool:
    if os.path.exists("tools/cmd-denoise.py"):
        return True
    # UVR5 权重是否存在
    if os.path.isdir("tools/uvr5/uvr5_weights"):
        return True
    return False

def run_gsv_denoise(in_wav, out_wav,
                    mode=os.getenv("GSV_DENOISE","mild"),   # none/mild/strong
                    extract_vocals=os.getenv("GSV_EXTRACT_VOCAL","1")=="1",
                    uvr_model=os.getenv("UVR_MODEL","HP2")) -> bool:
    """
    1) 优先 tools/cmd-denoise.py（批处理接口，要求目录）
    2) 再尝试 UVR5（若仅有 UVR5 权重——需要你本地已下载模型权重）
    3) 返回 True 表示成功产出文件
    """
    # a) cmd-denoise.py
    tool_py = _find_file("cmd-denoise.py")
    if tool_py and os.path.basename(os.path.dirname(tool_py)) == "tools":
        repo_cwd = os.path.dirname(os.path.dirname(tool_py))  # tools 的上一级就是仓库根
        tmp_in = os.path.join(WORK_DIR, "_denoise_in")
        tmp_out = os.path.join(WORK_DIR, "_denoise_out")
        shutil.rmtree(tmp_in, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)
        os.makedirs(tmp_in, exist_ok=True)
        os.makedirs(tmp_out, exist_ok=True)
        base = os.path.basename(in_wav)
        shutil.copy2(in_wav, os.path.join(tmp_in, base))
        cmd = [sys.executable, "-s", tool_py, "-i", tmp_in, "-o", tmp_out, "-p", "float32"]
        log(f"denoise via cmd: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=repo_cwd)
            cand = os.path.join(tmp_out, base)
            if os.path.exists(cand):
                shutil.copy2(cand, out_wav)
                return True
            wavs = sorted(glob.glob(os.path.join(tmp_out, "*.wav")))
            if wavs:
                shutil.copy2(wavs[0], out_wav)
                return True
        except Exception as e:
            log(f"cmd-denoise failed: {e}")

    # b) UVR5 权重检测（目录在仓库根的 tools/uvr5/uvr5_weights）
    if os.path.isdir(os.path.join(REPO_ROOT, "tools", "uvr5", "uvr5_weights")):
        log("UVR5 weights detected, please install a CLI wrapper for batch separation if needed.")
        # 这里留空，避免阻塞
        pass

    return False


# ---------- GPT-SoVITS 推理（零样本/小微调后） ----------

# 替换你文件里的 synth_once_cli 为下面版本
import shlex

def synth_once_cli(ref_wav: str, text: str, out_wav: str,
                   ref_language_ui: str = "多语种混合", text_language_ui: Optional[str] = None) -> bool:
    """
    通过命令行完成一次推理：以【模块方式】运行 GPT_SoVITS.inference_cli
    cwd 统一为仓库根（REPO_ROOT），这样 tools/* 等包能被正确找到。
    """
    def _guess_lang_ui(t: str) -> str:
        code = detect_language(t)
        return {"zh": "中文", "ja": "日文", "en": "英文"}.get(code, "多语种混合")

    if text_language_ui is None:
        text_language_ui = _guess_lang_ui(text)

    gpt_model = os.environ.get("gpt_path") or pretrained_gpt_name["v2ProPlus"]
    s2_model  = os.environ.get("sovits_path") or pretrained_sovits_name["v2ProPlus"]

    tmp_dir = os.path.join(WORK_DIR, "_cli")
    os.makedirs(tmp_dir, exist_ok=True)
    ref_txt = os.path.join(tmp_dir, "ref.txt")
    tgt_txt = os.path.join(tmp_dir, "tgt.txt")
    with open(ref_txt, "w", encoding="utf-8") as f: f.write("")   # 空 -> ref-free
    safe = (text or "").strip()
    if safe and safe[0] not in "，。,.?!、：:;…—-":
        safe = ",，," + safe           # 句首“缓冲”
    if safe and safe[-1] not in "。.!?？！.":
        safe = safe + "。"
    with open(tgt_txt, "w", encoding="utf-8") as f:
        f.write(safe)

    out_dir = os.path.dirname(out_wav) or "."
    os.makedirs(out_dir, exist_ok=True)

    # === 新增：全部转成绝对路径，避免 cwd 变化带来的相对路径失效 ===
    ref_wav_abs = os.path.abspath(ref_wav)
    ref_txt_abs = os.path.abspath(ref_txt)
    tgt_txt_abs = os.path.abspath(tgt_txt)
    out_dir_abs = os.path.abspath(out_dir)

    repo_cwd = REPO_ROOT
    NLTK_DIR = os.path.join(REPO_ROOT, "_nltk_data")
    ensure_nltk_data(NLTK_DIR)
    env = os.environ.copy()
    env["NLTK_DATA"] = NLTK_DIR
    extra_paths = [
        repo_cwd,
        os.path.join(repo_cwd, "GPT_SoVITS"),
        os.path.join(repo_cwd, "tools"),
        os.path.join(repo_cwd, "tools", "uvr5"),
    ]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.pathsep.join(extra_paths), env.get("PYTHONPATH", "")]))
    cmd = [
        sys.executable, "-s", "-m", "GPT_SoVITS.inference_cli",
        "--gpt_model", gpt_model,
        "--sovits_model", s2_model,
        "--ref_audio", ref_wav_abs,
        "--ref_text", ref_txt_abs,            # 空文件 => ref_free
        "--ref_language", ref_language_ui,    # 中文/英文/日文
        "--target_text", tgt_txt_abs,         # 来自 ./Task CSV
        "--target_language", text_language_ui,
        "--output_path", out_dir_abs,
    ]
    log("infer via CLI: " + " ".join(shlex.quote(x) for x in cmd) + f"  (cwd={repo_cwd})")
    try:
        subprocess.run(cmd, check=True, cwd=repo_cwd, env=env)
        cli_out = os.path.join(out_dir_abs, "output.wav")  # 也改为绝对路径
        if os.path.exists(cli_out):
            if os.path.abspath(cli_out) != os.path.abspath(out_wav):
                shutil.move(cli_out, out_wav)
            return True
    except Exception as e:
        log(f"inference_cli failed: {e}")
    return False


# ---------- 小微调（尽量调用 WebUI 同款入口；失败则跳过） ----------
def try_finetune(exp_name: str, list_file: str, wav_dir: str, epochs: int = 30,
                 dpo_prefer: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """
    尝试运行 webui 里的 1abc(特征准备) + s2/s1 训练，成功则返回 (sovits_ckpt, gpt_ckpt)，否则 (None, None)
    - list_file: 形如 "wav|spk|lang|text" 的 .list（参见官方 README）
    """
    # 1) 准备 1abc 所需环境变量并按 webui 的脚本顺序运行
    # 提示：该流程依赖大量预训练模型与脚本，用户需按 README 配置好。
    try:
        # —— 准备必要路径
        exp_root = os.environ.get("exp_root", "logs")  # webui 默认
        sdir_rel = os.path.join(exp_root, exp_name)
        sdir = os.path.abspath(sdir_rel)
        os.makedirs(sdir, exist_ok=True)

        # 写入输入 list
        inp_text = os.path.join(sdir, "train.list")
        shutil.copy2(list_file, inp_text)

        # 1abc：等价于 webui.open1abc 所调用的 prepare_datasets 脚本
        # 这里直接复用那些脚本的 -s 入口（通过环境变量传参）
        env = os.environ.copy()

        # 在 try_finetune(...) 里，env = os.environ.copy() 之后补：
        repo_cwd = REPO_ROOT  # 已有
        extra_paths = [
            repo_cwd,
            os.path.join(repo_cwd, "GPT_SoVITS"),
            os.path.join(repo_cwd, "GPT_SoVITS","prepare_datasets"),
            os.path.join(repo_cwd, "tools"),
            os.path.join(repo_cwd, "tools", "uvr5"),
        ]
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [os.pathsep.join(extra_paths), env.get("PYTHONPATH", "")])
        )

        env.update({
            "inp_text": os.path.abspath(inp_text),
            "inp_wav_dir": os.path.abspath(wav_dir),
            "exp_name": exp_name,
            "opt_dir": sdir,
            "i_part": "0",
            "all_parts": "1",
        })
        # 1B/1C：按 WebUI 的分支逻辑执行 Pro 所需的 2-get-sv，并为 1C 传 s2G 和 s2 config
        version = os.environ.get("version", "v2")
        s2_cfg = (
            f"GPT_SoVITS/configs/s2{version}.json"
            if version in {"v2Pro", "v2ProPlus"}
            else "GPT_SoVITS/configs/s2.json"
        )
        # infer_device 来自 config.py；CPU 环境时会是 cpu 设备，没有 index 属性
        gpu_index = getattr(infer_device, "index", 0)
        env.update({
            "_CUDA_VISIBLE_DEVICES": str(gpu_index),
            "pretrained_s2G": pretrained_sovits_name.get(version, ""),
            "s2config_path": s2_cfg,
        })
        # 1B: hubert
        repo_cwd = REPO_ROOT  # 确保所有相对路径以仓库根为基准

        # —— 兼容性：显式设置 CN-HuBERT / BERT 路径到 env（两种命名都写上）
        cn_hubert_abs = os.environ.get("cnhubert_base_path") or os.path.join(REPO_ROOT, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base")
        bert_abs      = os.environ.get("bert_path")          or os.path.join(REPO_ROOT, "GPT_SoVITS", "pretrained_models", "chinese-roberta-wwm-ext-large")

        env["cnhubert_base_path"] = cn_hubert_abs  # 有的分支用这个
        env["cnhubert_path"]      = cn_hubert_abs  # 有的分支用这个（例如 config.cnhubert_path）
        env["cnhubert_base_dir"]  = cn_hubert_abs  # 2-get-hubert-wav32k.py 用这个
        env["bert_path"]          = bert_abs

        # 可选：把相对路径转绝对，避免 cwd 变化
        env["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
        env["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

        # 可选：检查并友好提示
        if not os.path.isdir(cn_hubert_abs):
            log(f"[fatal] CN-HuBERT 目录不存在：{cn_hubert_abs}，请把模型放到此处或改路径。")
            raise SystemExit(1)

        log("run 2-get-hubert-wav32k.py")
        subprocess.run([sys.executable, "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"],
                       check=True, env=env, cwd=repo_cwd)
        # 1B-Pro: 说话人向量
        if "Pro" in version:
            log("run 2-get-sv.py (Pro系列)")
            subprocess.run([sys.executable, "-s", "GPT_SoVITS/prepare_datasets/2-get-sv.py"],
                           check=True, env=env, cwd=repo_cwd)

        log("run 3-get-semantic.py")
        subprocess.run([sys.executable, "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"],
                       check=True, env=env, cwd=repo_cwd)

        # 2) SoVITS 训练（s2）
        # 根据 webui.open1Ba 写入 tmp_s2.json 再调用 s2_train.py
        # 直接使用默认的 s2 配置文件，让脚本在 exp_dir 下读取准备好的 2-name2text/6-name2semantic 等
        version = os.environ.get("version", "v2")  # 与当前环境一致即可
        s2_cfg = "GPT_SoVITS/configs/s2.json"
        if version in {"v2Pro", "v2ProPlus"}:
            s2_cfg = f"GPT_SoVITS/configs/s2{version}.json"
        if not os.path.exists(s2_cfg):
            log("warn: s2 config not found, skip s2 training")
        else:
            with open(s2_cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["data"]["exp_dir"] = sdir
            cfg["s2_ckpt_dir"] = sdir
            cfg["train"]["epochs"] = int(epochs)
            cfg["train"]["batch_size"] = int(20)  # 与题意无关，这里固定 20
            tmp_s2 = os.path.join(sdir, "tmp_s2.json")
            with open(tmp_s2, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False)
            cmd = [sys.executable, "-s", "GPT_SoVITS/s2_train.py", "--config", tmp_s2]
            log("train s2: " + " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=repo_cwd)

        # 3) GPT 训练（s1）
        # 参考 webui.open1Bb，修改 s1longer-v2.yaml 并传入 s1_train.py
        s1_yaml = "GPT_SoVITS/configs/s1longer-v2.yaml"
        if not os.path.exists(s1_yaml):
            log("warn: s1 yaml not found, skip s1 training")
        else:
            import yaml  # 训练阶段才导入
            with open(s1_yaml, "r", encoding="utf-8") as f:
                y = yaml.load(f, Loader=yaml.FullLoader)
            y["train"]["epochs"] = int(epochs)
            # 题意：能开 DPO 则 batch_size=8，否则 20。这里“尝试”为 True；失败由 s1_train 捕获。
            y["train"]["if_dpo"] = bool(dpo_prefer)
            y["train"]["batch_size"] = 8 if dpo_prefer else 20
            y["train"]["exp_name"] = exp_name
            y["train_semantic_path"] = os.path.join(sdir, "6-name2semantic.tsv")
            y["train_phoneme_path"] = os.path.join(sdir, "2-name2text.txt")
            y["output_dir"] = os.path.join(sdir, "logs_s1_v2")
            tmp_s1 = os.path.join(sdir, "tmp_s1.yaml")
            with open(tmp_s1, "w", encoding="utf-8") as f:
                yaml.safe_dump(y, f, allow_unicode=True)
            cmd = [sys.executable, "-s", "GPT_SoVITS/s1_train.py", "--config_file", tmp_s1]
            log("train s1: " + " ".join(cmd))
            try:
                subprocess.run(cmd, check=True, cwd=repo_cwd)
            except subprocess.CalledProcessError:
                # 如果 DPO 不可用，重试：关闭 DPO，batch_size=20
                log("retry s1 without DPO...")
                y["train"]["if_dpo"] = False
                y["train"]["batch_size"] = 20
                with open(tmp_s1, "w", encoding="utf-8") as f:
                    yaml.safe_dump(y, f, allow_unicode=True)
                subprocess.run(cmd, check=True, cwd=repo_cwd)

        # 尝试找最新权重
        def pick_last_weight(patterns: List[str]) -> Optional[str]:
            cand = []
            for p in patterns:
                cand += glob.glob(os.path.join(sdir, p))
            if not cand:
                return None
            cand.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return cand[0]

        sovits_ckpt = pick_last_weight(["logs_s2_*/G_*.pth", "logs_s2_*/G_*.ckpt", "logs_s2*/G_*.pth", "G_*.pth"])
        gpt_ckpt = pick_last_weight(["logs_s1*/epoch*.ckpt", "logs_s1*/last*.ckpt", "logs_s1*/*.ckpt", "*.ckpt"])
        return sovits_ckpt, gpt_ckpt

    except Exception as e:
        log(f"finetune failed: {e}")
        return None, None

# ---------- 主流程 ----------
def main():
    # 读 CSV
    csv_files = [f for f in os.listdir(TASK_DIR) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("Task 目录下未发现 csv 文件")
    csv_path = os.path.join(TASK_DIR, csv_files[0])
    df = pd.read_csv(csv_path)

    # 输出 CSV 初始化
    out_df = df.copy()
    out_df["synthesized_speech"] = ""

    for idx, row in df.iterrows():
        try:
            utt = int(row["utt"])
            ref_name = str(row["reference_speech"]).strip()
            target_text = str(row["text"]).strip()
        except Exception:
            log(f"非法行 idx={idx}，跳过")
            continue
        out_wav = os.path.join(RESULT_DIR, f"synthesized_speech_{utt}.wav")
        if os.path.exists(out_wav) and is_valid_wav(out_wav):
            out_df.loc[idx, "synthesized_speech"] = os.path.basename(out_wav)
            log(f"已存在，跳过 utt={utt} -> {out_wav}")
            continue

        ref_path = os.path.join(TASK_DIR, ref_name)
        if not os.path.exists(ref_path):
            log(f"参考音频不存在：{ref_path}，跳过")
            continue
        try:
            log(f"===== 处理 utt={utt} : {ref_name} =====")
            # 1) 降噪/人声保留
            clean_ref = os.path.join(WORK_DIR, f"utt{utt}_clean.wav")
            ok = False
            if have_gsv_denoise():
                ok = run_gsv_denoise(ref_path, clean_ref)
            if not ok:
                # 回退：简易降噪
                y, sr = load_mono(ref_path, target_sr=44100 if librosa is not None else None)
                y = simple_noise_reduce(y, sr, prop=0.1)
                save_wav(clean_ref, y, sr)

            # 2) 按题意规则处理长度 & 决定是否微调
            dur = sec_of(clean_ref)
            prompt_wav = os.path.join(WORK_DIR, f"utt{utt}_prompt.wav")
            need_finetune = False
            train_wav_dir = None  # 若微调，数据所在目录
            if dur < 3.0:
                y, sr = load_mono(clean_ref, target_sr=44100 if librosa is not None else None)
                y_new = repeat_to_between(y, sr, 3.0, 10.0)
                save_wav(prompt_wav, y_new, sr)
                need_finetune = False
            elif 3.0 <= dur <= 10.0:
                shutil.copy2(clean_ref, prompt_wav)
                need_finetune = False
            elif 10.0 < dur < 60.0:
                # 复制并合并到 >60s，用于小微调；同时生成一个 3~10s 的 prompt 给推理
                y, sr = load_mono(clean_ref, target_sr=44100 if librosa is not None else None)
                y_train = repeat_to_over(y, sr, 60.01)
                # prompt 取前 8s
                y_prompt = repeat_to_between(y, sr, 3.0, 10.0)
                save_wav(prompt_wav, y_prompt, sr)
                train_wav = os.path.join(WORK_DIR, f"utt{utt}_train.wav")
                save_wav(train_wav, y_train, sr)
                train_wav_dir = os.path.dirname(train_wav)
                need_finetune = True
            else:
                # dur >= 60s：直接微调；prompt 取前 8s
                y, sr = load_mono(clean_ref, target_sr=44100 if librosa is not None else None)
                y_prompt = repeat_to_between(y[: sr*10], sr, 3.0, 10.0)  # 限制到 <=10s
                save_wav(prompt_wav, y_prompt, sr)
                train_wav = os.path.join(WORK_DIR, f"utt{utt}_train.wav")
                save_wav(train_wav, y, sr)
                train_wav_dir = os.path.dirname(train_wav)
                need_finetune = True

            # 3) （可选）小微调：构造 .list 并调用 try_finetune
            sovits_ckpt = gpt_ckpt = None
            if need_finetune and train_wav_dir is not None:
                # 生成 list：wav|speaker|lang|text
                spk = f"spk{utt}"
                lang = detect_language(target_text)
                list_path = os.path.join(WORK_DIR, f"utt{utt}.list")
                with open(list_path, "w", encoding="utf-8") as f:
                    f.write(f"{os.path.join(train_wav_dir, os.path.basename(train_wav))}|{spk}|{lang}|{target_text}\n")

                exp_name = f"auto_utt_{utt}"
                sovits_ckpt, gpt_ckpt = try_finetune(exp_name, list_path, train_wav_dir, epochs=30, dpo_prefer=True)
                # 若拿到了新权重，切换权重
                if sovits_ckpt or gpt_ckpt:
                    try:
                        from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights
                        if sovits_ckpt:
                            change_sovits_weights(sovits_ckpt)
                        if gpt_ckpt:
                            change_gpt_weights(gpt_ckpt)
                        log(f"finetune weights ready: s2={sovits_ckpt}, s1={gpt_ckpt}")
                    except Exception as e:
                        log(f"warn: cannot switch weights automatically: {e}")

            # 4) 推理合成
            out_wav = os.path.join(RESULT_DIR, f"synthesized_speech_{utt}.wav")
            ok = synth_once_cli(prompt_wav, target_text, out_wav,
                                ref_language_ui="中文",  # 参考音频不确定时更安全
                                text_language_ui="中文")

            if not ok:
                log(f"推理失败，跳过 utt={utt}")
                continue

            out_df.loc[idx, "synthesized_speech"] = os.path.basename(out_wav)
            log(f"完成 utt={utt} -> {out_wav}")
        except Exception as e:
            # **修改点**：任意抛错 -> 不中断整个批处理，直接跳过本条
            log(f"处理 utt={utt} 异常，已跳过：{e}")
            continue
    # 保存 result.csv
    out_csv = os.path.join(RESULT_DIR, "result.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    log(f"结果已写入：{out_csv}")


if __name__ == "__main__":
    main()
