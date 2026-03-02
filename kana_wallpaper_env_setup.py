# -*- coding: utf-8 -*-
"""
Kana Wallpaper - ワンクリック環境セットアップ（venv + 依存 + モデル + 実行スクリプト生成）

目的：
  1) リポジトリ直下に venv を作成して依存関係をインストール
  2) _kana_state/models に必要なAIモデル（任意でダウンロード）を揃える
  3) 実行をワンタッチ化するための起動スクリプト（RUN_KANA_WALLPAPER.bat 等）を生成する
  4) 可能な範囲でGPU（CUDA/DirectML）環境を自動選択して入れる

使い方（基本）:
  python kana_wallpaper_env_setup.py

おすすめ（モデルも揃える）:
  python kana_wallpaper_env_setup.py --download-models

注意：
  * WindowsでNVIDIA GPUがある場合、PyTorchのCUDAビルドを入れます（自動判定）。
  * それ以外のGPU（AMD/Intel等）は、必要なら DirectML も選べます（Windowsのみ）。
  * ffmpeg/7z/rar 等の「外部コマンド」は pip では入らないため、存在チェックのみ行います。
  * コメントは日本語で統一。
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


# =========================
# 設定（後からここだけ編集）
# =========================

@dataclass
class SetupConfig:
    # venvの場所（リポジトリ直下推奨）
    VENV_DIRNAME: str = ".venv"

    # 出力・キャッシュ・モデル置き場
    STATE_DIR_BASENAME: str = "_kana_state"
    MODELS_SUBDIRNAME: str = "models"

    # 1クリック運用向け：最後にEnter待ちする（ダブルクリック起動時のウィンドウ即閉じ対策）
    PAUSE_AT_END: bool = True

    # GPUモード：auto / cuda / cpu / directml
    # auto：Windows + NVIDIAならcuda、それ以外はcpu（必要ならdirectmlを手動指定）
    GPU_MODE: str = "auto"

    # CUDA判定で選ぶPyTorchホイール（index-url）を固定したい時はここに書く
    # 例: "https://download.pytorch.org/whl/cu128"
    FORCE_TORCH_INDEX_URL: Optional[str] = None

    # Kana Wallpaperで使う主要パッケージ
    BASE_PIP_PACKAGES: Tuple[str, ...] = (
        "pip",  # まずpip自体の更新に使う
        "setuptools",
        "wheel",
        # ここから本体依存
        "pillow",
        "numpy",
        "opencv-python",
        "py7zr",
        "rarfile",
        "ultralytics",
    )

    # モデルファイル（_kana_state/models に置く）
    # ※YOLOは大きいので --download-yolo の時だけ落とす（初期はOFF）
    YUNET_URL: str = "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ANIMEFACE_CASCADE_URL: str = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
    YOLO_ANIMEFACE_URL: str = "https://huggingface.co/Fuyucchi/yolov8_animeface/resolve/main/yolov8x6_animeface.pt"

    YUNET_FILENAME: str = "face_detection_yunet_2023mar.onnx"
    ANIMEFACE_CASCADE_FILENAME: str = "lbpcascade_animeface.xml"
    YOLO_ANIMEFACE_FILENAME: str = "yolov8x6_animeface.pt"

    # ワンタッチ起動スクリプト名
    RUN_BAT_FILENAME: str = "RUN_KANA_WALLPAPER.bat"
    RUN_SH_FILENAME: str = "run_kana_wallpaper.sh"


CFG = SetupConfig()


# =========================
# ユーティリティ
# =========================

def print_hr(title: str = "") -> None:
    width = 72
    if title:
        line = f" {title} "
        pad = max(0, width - len(line))
        left = pad // 2
        right = pad - left
        print("=" * left + line + "=" * right)
    else:
        print("=" * width)


def run_cmd(cmd: list[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    # 外部コマンドを実行（標準出力はそのまま流す）
    print(f"\n[RUN] {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
    )


def which_any(names: Tuple[str, ...]) -> Optional[str]:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def repo_root_from_this_file() -> Path:
    # このsetupスクリプトのある場所をリポジトリのルートとして扱う
    return Path(__file__).resolve().parent


def state_dir(repo_root: Path) -> Path:
    return repo_root / CFG.STATE_DIR_BASENAME


def models_dir(repo_root: Path) -> Path:
    return state_dir(repo_root) / CFG.MODELS_SUBDIRNAME


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def venv_python_path(repo_root: Path) -> Path:
    venv_dir = repo_root / CFG.VENV_DIRNAME
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def safe_pause() -> None:
    if not CFG.PAUSE_AT_END:
        return
    try:
        input("\n完了しました。Enterキーで閉じます...")
    except Exception:
        pass


# =========================
# GPU / PyTorch セットアップ
# =========================

def detect_nvidia_cuda_version() -> Optional[Tuple[int, int]]:
    """
    nvidia-smi があれば CUDA Version を拾う（例: 'CUDA Version: 12.6'）
    """
    smi = which_any(("nvidia-smi",))
    if not smi:
        return None
    try:
        cp = subprocess.run([smi], capture_output=True, text=True, check=False)
        text = (cp.stdout or "") + "\n" + (cp.stderr or "")
        m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", text)
        if not m:
            return None
        return (int(m.group(1)), int(m.group(2)))
    except Exception:
        return None


def choose_torch_index_url(gpu_mode: str, cuda_ver: Optional[Tuple[int, int]]) -> Tuple[Optional[str], str]:
    """
    PyTorchのpipインストールに使う index-url を決める。
      - CUDA版: https://download.pytorch.org/whl/cu118 など
      - CPU版 : https://download.pytorch.org/whl/cpu
    """
    if CFG.FORCE_TORCH_INDEX_URL:
        return (CFG.FORCE_TORCH_INDEX_URL, f"forced ({CFG.FORCE_TORCH_INDEX_URL})")

    gpu_mode = (gpu_mode or "").lower().strip()

    if gpu_mode == "cpu":
        return ("https://download.pytorch.org/whl/cpu", "cpu")

    if gpu_mode == "cuda":
        # PyTorch公式の選択肢（cu118 / cu126 / cu128）を目安にする
        # CUDAバージョンが拾えない場合は互換性を優先してcu118へ
        if not cuda_ver:
            return ("https://download.pytorch.org/whl/cu118", "cuda (fallback cu118)")
        major, minor = cuda_ver
        # 12.8以上 → cu128、12.6以上 → cu126、それ未満 → cu118
        if (major, minor) >= (12, 8):
            return ("https://download.pytorch.org/whl/cu128", "cuda (cu128)")
        if (major, minor) >= (12, 6):
            return ("https://download.pytorch.org/whl/cu126", "cuda (cu126)")
        return ("https://download.pytorch.org/whl/cu118", "cuda (cu118)")

    # directml は別パッケージ（torch-directml）を入れるので index-url は不要
    if gpu_mode == "directml":
        return (None, "directml")

    # auto
    if cuda_ver:
        return choose_torch_index_url("cuda", cuda_ver)
    return choose_torch_index_url("cpu", None)


def infer_gpu_mode_auto() -> str:
    """
    autoのときの判断：
      - Windows + nvidia-smiあり → cuda
      - それ以外 → cpu（必要なら directml を明示指定）
    """
    cuda_ver = detect_nvidia_cuda_version()
    if os.name == "nt" and cuda_ver is not None:
        return "cuda"
    return "cpu"


# =========================
# venv / pip
# =========================

def ensure_venv(repo_root: Path) -> Path:
    venv_dir = repo_root / CFG.VENV_DIRNAME
    py = venv_python_path(repo_root)
    if py.exists():
        print(f"[OK] venv exists: {venv_dir}")
        return py

    print(f"[INFO] Creating venv: {venv_dir}")
    ensure_dir(venv_dir.parent)
    run_cmd([sys.executable, "-m", "venv", str(venv_dir)])
    if not py.exists():
        raise RuntimeError(f"venv python not found: {py}")
    return py


def pip_install(venv_py: Path, args: list[str]) -> None:
    run_cmd([str(venv_py), "-m", "pip"] + args)


def pip_install_packages(venv_py: Path, packages: Tuple[str, ...]) -> None:
    # pip自体の更新
    pip_install(venv_py, ["install", "-U", "pip", "setuptools", "wheel"])
    # 本体依存
    pkgs = [p for p in packages if p not in ("pip", "setuptools", "wheel")]
    if pkgs:
        pip_install(venv_py, ["install"] + list(pkgs))


# =========================
# モデルダウンロード
# =========================

def download_with_progress(url: str, dst: Path) -> None:
    """
    urllibでダウンロード（簡易プログレス表示）
    """
    ensure_dir(dst.parent)

    # 既にあればスキップ
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[SKIP] exists: {dst.name}")
        return

    print(f"[DL] {url}")
    print(f"     -> {dst}")

    req = urllib.request.Request(url, headers={"User-Agent": "KanaWallpaperSetup/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = resp.headers.get("Content-Length")
        total_size = int(total) if total and total.isdigit() else None

        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f:
            downloaded = 0
            t0 = time.time()
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = (downloaded / total_size) * 100.0
                    elapsed = max(0.001, time.time() - t0)
                    speed = downloaded / elapsed / (1024 * 1024)
                    print(f"\r     {pct:6.2f}%  {downloaded/1024/1024:8.1f}MB  {speed:5.1f}MB/s", end="")
                else:
                    print(f"\r     {downloaded/1024/1024:8.1f}MB", end="")
        print()

        tmp.replace(dst)
        print(f"[OK] downloaded: {dst.name}")


def ensure_model_files(repo_root: Path, download_small: bool, download_yolo: bool) -> None:
    mdir = models_dir(repo_root)
    ensure_dir(mdir)

    # README（必要ファイルが分かるように）
    readme = mdir / "README_models.txt"
    if not readme.exists():
        readme.write_text(
            "\n".join([
                "Kana Wallpaper models folder",
                "",
                "Place these files here (or let setup script download them):",
                f"- {CFG.YUNET_FILENAME}  (YuNet face detector ONNX)",
                f"- {CFG.ANIMEFACE_CASCADE_FILENAME}  (AnimeFace LBP cascade XML)",
                f"- {CFG.YOLO_ANIMEFACE_FILENAME}  (YOLOv8 anime face weights, large file)",
                "",
                "URLs:",
                f"- YuNet: {CFG.YUNET_URL}",
                f"- AnimeFace: {CFG.ANIMEFACE_CASCADE_URL}",
                f"- YOLO: {CFG.YOLO_ANIMEFACE_URL}",
                "",
                "Notes:",
                "- YOLO weights file is large; download it only if needed.",
            ]) + "\n",
            encoding="utf-8"
        )

    if download_small:
        download_with_progress(CFG.YUNET_URL, mdir / CFG.YUNET_FILENAME)
        download_with_progress(CFG.ANIMEFACE_CASCADE_URL, mdir / CFG.ANIMEFACE_CASCADE_FILENAME)

    if download_yolo:
        download_with_progress(CFG.YOLO_ANIMEFACE_URL, mdir / CFG.YOLO_ANIMEFACE_FILENAME)


# =========================
# 外部コマンドチェック
# =========================

def check_external_tools() -> None:
    print_hr("External tools check")

    tools = [
        ("ffmpeg", ("ffmpeg",)),
        ("ffprobe", ("ffprobe",)),
        ("7zip", ("7z", "7za")),
        ("rar", ("rar",)),
        ("unrar", ("unrar",)),
    ]
    for label, names in tools:
        p = which_any(names)
        if p:
            print(f"[OK] {label}: {p}")
        else:
            print(f"[WARN] {label}: not found in PATH")


# =========================
# 動作確認
# =========================

def verify_install(venv_py: Path) -> None:
    print_hr("Verify")
    # 主要モジュールの import と torch のGPU可用性チェック
    # NOTE: `python -c` は実行ログが長くなりがちなので、一時ファイルを生成して実行し、最後に削除します。
    code = r"""
import sys
print("python:", sys.version.replace("\n"," "))
try:
    import numpy as np
    import cv2
    from PIL import Image
    print("numpy:", np.__version__)
    print("opencv:", cv2.__version__)
    print("Pillow: OK")
except Exception as e:
    print("basic import failed:", e)

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
except Exception as e:
    print("torch check failed:", e)

try:
    import ultralytics
    print("ultralytics:", ultralytics.__version__)
except Exception as e:
    print("ultralytics check failed:", e)
"""
    tmp_dir = Path(tempfile.mkdtemp(prefix="kana_wallpaper_verify_"))
    tmp_py = tmp_dir / "verify_env.py"
    try:
        tmp_py.write_text(code, encoding="utf-8")
        run_cmd([str(venv_py), str(tmp_py)])
    finally:
        try:
            tmp_py.unlink()
        except Exception:
            pass
        try:
            tmp_dir.rmdir()
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# =========================
# ワンタッチ起動スクリプト生成
# =========================

def parse_version_from_name(name: str) -> Optional[int]:
    m = re.search(r"_v(\d+)", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def find_latest_launcher(repo_root: Path) -> Optional[Path]:
    # ざっくり：kana_wallpaper_launcher*.py を候補として、v番号最大を選ぶ
    cands = sorted(repo_root.glob("kana_wallpaper_launcher*.py"))
    if not cands:
        return None

    best = None
    best_v = -1
    for p in cands:
        v = parse_version_from_name(p.name)
        if v is None:
            # v番号が無いものは弱い候補として、更新日時で比較できるようにする
            # ただしv番号があるものが見つかったら基本そちらを優先
            continue
        if v > best_v:
            best_v = v
            best = p

    if best is not None:
        return best

    # v番号が無い候補しか無い場合は更新日時で一番新しいもの
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def write_run_bat(repo_root: Path, setup_script_name: str, launcher_name: str, auto_download_models: bool) -> None:
    """
    Windows向け：ダブルクリックで
      - venvが無ければsetup実行（小さいモデルDLも任意）
      - その後 launcher を venv python で起動
    """
    dl_flag = " --download-models" if auto_download_models else ""
    content = "\r\n".join([
        "@echo off",
        "setlocal",
        "cd /d \"%~dp0\"",
        "chcp 65001 >nul",
        "set PYTHONUTF8=1",
        "set PYTHONIOENCODING=utf-8",
        "",
        "if exist \".venv\\Scripts\\python.exe\" goto :RUN",
        "echo [INFO] venv not found. Running setup...",
        f"python \"{setup_script_name}\"{dl_flag} --no-pause",
        "",
        ":RUN",
        "if not exist \".venv\\Scripts\\python.exe\" (",
        "  echo [ERROR] venv python not found. Please run setup again.",
        "  pause",
        "  exit /b 1",
        ")",
        "",
        f"\".venv\\Scripts\\python.exe\" \"{launcher_name}\"",
        "pause",
        "endlocal",
        "",
    ])
    (repo_root / CFG.RUN_BAT_FILENAME).write_text(content, encoding="utf-8")


def write_run_sh(repo_root: Path, setup_script_name: str, launcher_name: str, auto_download_models: bool) -> None:
    """
    macOS/Linux向け：./run_kana_wallpaper.sh
    """
    dl_flag = " --download-models" if auto_download_models else ""
    content = "\n".join([
        "#!/usr/bin/env bash",
        "set -e",
        "cd \"$(dirname \"$0\")\"",
        "",
        "export PYTHONUTF8=1",
        "export PYTHONIOENCODING=utf-8",
        "",
        "if [ ! -x \".venv/bin/python\" ]; then",
        "  echo \"[INFO] venv not found. Running setup...\"",
        f"  python \"{setup_script_name}\"{dl_flag} --no-pause",
        "fi",
        "",
        "\".venv/bin/python\" \"" + launcher_name + "\"",
        "",
        "read -p \"Press Enter to close...\" _",
        "",
    ])
    p = repo_root / CFG.RUN_SH_FILENAME
    p.write_text(content, encoding="utf-8")
    # 実行権限を付与（可能な環境のみ）
    try:
        mode = p.stat().st_mode
        p.chmod(mode | 0o111)
    except Exception:
        pass


def make_run_scripts(repo_root: Path, setup_script_name: str, launcher_path: Path, auto_download_models: bool) -> None:
    print_hr("One-touch run scripts")
    print(f"[INFO] launcher: {launcher_path.name}")

    # Windows用
    write_run_bat(repo_root, setup_script_name, launcher_path.name, auto_download_models)

    # macOS/Linux用
    write_run_sh(repo_root, setup_script_name, launcher_path.name, auto_download_models)

    print(f"[OK] created: {CFG.RUN_BAT_FILENAME}")
    print(f"[OK] created: {CFG.RUN_SH_FILENAME}")


# =========================
# メイン
# =========================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Kana Wallpaper: venv + dependencies + models + one-touch run scripts"
    )
    ap.add_argument("--gpu", choices=["auto", "cuda", "cpu", "directml"], default=CFG.GPU_MODE,
                    help="GPU mode (default: auto)")
    ap.add_argument("--download-models", action="store_true",
                    help="Download small model files (YuNet + AnimeFace cascade)")
    ap.add_argument("--download-yolo", action="store_true",
                    help="Download YOLO animeface weights (large file)")
    ap.add_argument("--no-pause", action="store_true",
                    help="Do not wait for Enter at the end")
    ap.add_argument("--no-run-scripts", action="store_true",
                    help="Do not generate one-touch run scripts")
    ap.add_argument("--run-script-download-models", action="store_true",
                    help="In generated run script: auto-download small models when setup is needed")
    ap.add_argument("--launcher", default="",
                    help="Launcher filename to run (default: auto-detect latest kana_wallpaper_launcher*.py)")
    args = ap.parse_args()

    if args.no_pause:
        CFG.PAUSE_AT_END = False

    repo_root = repo_root_from_this_file()
    setup_script_name = Path(__file__).name

    print_hr("Kana Wallpaper Setup")
    print(f"Repo root : {repo_root}")
    print(f"State dir : {state_dir(repo_root)}")
    print(f"Models dir: {models_dir(repo_root)}")

    # Pythonバージョンの注意（PyTorchの最新安定は3.10以上が目安）
    pyver = sys.version_info
    if (pyver.major, pyver.minor) < (3, 10):
        print("[WARN] Python 3.10+ is recommended for recent PyTorch builds.")

    # GPUモード決定
    gpu_mode = args.gpu
    if gpu_mode == "auto":
        gpu_mode = infer_gpu_mode_auto()

    cuda_ver = detect_nvidia_cuda_version()
    if cuda_ver:
        print(f"[INFO] NVIDIA detected, CUDA Version: {cuda_ver[0]}.{cuda_ver[1]}")
    else:
        print("[INFO] NVIDIA CUDA not detected (nvidia-smi not found or no NVIDIA GPU).")

    print(f"[INFO] GPU mode: {gpu_mode}")

    # venv作成
    venv_py = ensure_venv(repo_root)

    print_hr("pip / packages")

    # torch
    if gpu_mode == "directml":
        if os.name != "nt":
            print("[WARN] directml is intended for Windows. Falling back to CPU torch.")
            gpu_mode = "cpu"

    index_url, label = choose_torch_index_url(gpu_mode, cuda_ver)
    if gpu_mode == "directml":
        # DirectMLは torch-directml を入れる
        pip_install(venv_py, ["install", "-U", "pip", "setuptools", "wheel"])
        pip_install(venv_py, ["install", "torch-directml"])
        print("[OK] torch-directml installed (directml).")
    else:
        # CPU / CUDA
        print(f"[INFO] Installing torch via index-url: {label}")
        pip_install(venv_py, ["install", "torch", "torchvision", "torchaudio", "--index-url", index_url])

    # Kana Wallpaper依存
    pip_install_packages(venv_py, CFG.BASE_PIP_PACKAGES)

    # モデルフォルダ作成（必要ならDL）
    print_hr("Models")
    ensure_model_files(repo_root, download_small=args.download_models, download_yolo=args.download_yolo)

    # 外部コマンドチェック
    check_external_tools()

    # 動作確認
    verify_install(venv_py)

    # ランチャー検出 & 実行スクリプト生成
    if not args.no_run_scripts:
        launcher_path: Optional[Path]
        if args.launcher.strip():
            launcher_path = (repo_root / args.launcher.strip()).resolve()
            if not launcher_path.exists():
                print(f"[WARN] launcher not found: {launcher_path.name}")
                launcher_path = None
        else:
            launcher_path = find_latest_launcher(repo_root)

        if launcher_path is None:
            print("[WARN] launcher script was not found. Run scripts will not be created.")
        else:
            make_run_scripts(
                repo_root=repo_root,
                setup_script_name=setup_script_name,
                launcher_path=launcher_path,
                auto_download_models=args.run_script_download_models,
            )

    print_hr("Done")
    print("Setup finished.")
    print(f"- venv : {repo_root / CFG.VENV_DIRNAME}")
    print(f"- models: {models_dir(repo_root)}")
    print(f"- run  : {CFG.RUN_BAT_FILENAME}  (Windows)")
    print(f"- run  : {CFG.RUN_SH_FILENAME}   (macOS/Linux)")

    print("\nTips:")
    print("  - If you want YOLO animeface, re-run setup with: --download-yolo")
    print("  - If ffmpeg/7z/rar are missing, install them and ensure they are in PATH.")
    print("  - For one-touch run on Windows, double-click RUN_KANA_WALLPAPER.bat")

    safe_pause()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
