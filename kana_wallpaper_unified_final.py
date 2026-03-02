# -*- coding: utf-8 -*-
"""
Kana Wallpaper - Unified FINAL（core）

複数の画像（フォルダ/アーカイブ/動画フレーム）から 1 枚の壁紙画像を生成します。
Windows では必要なら壁紙設定まで行います。

主な機能
- 入力: フォルダ（サブフォルダ）/ ZIP・7z・RAR / 動画フレーム抽出（任意）
- 動画抽出: random / uniform / scene / scene_best / best_bright / best_sharp / best_combo
  - grid では「タイムライン順（asc/desc）」で配置でき、動画の後半が欠けにくいよう全体から選別します。
- レイアウト: grid / hex / mosaic（uniform-height, uniform-width）/ quilt / stained-glass / random
- 抽出: SELECT_MODE（random/aesthetic/recent/oldest/name_asc/name_desc）＋ dHash 近似重複除去
- 並び/最適化: 色順（spectral/hilbert 等）＋ 近傍最適化（anneal/hill/checkerboard 等）
- 顔フォーカス: 低コスト検出（haar 等）/ AI（YOLO/YuNet/AnimeFace）＋ hex 用「顔欠け防止オートフィット」
- 仕上げ: halation / tonecurve / split tone / LUT(.cube) / NR / dehaze / shadow/highlight /
         clarity / unsharp / vibrance / grain / vignette / brightness(auto)

外部設定（おすすめ）
- ランチャーのエクスポート JSON（既定: kana_wallpaper_launcher_export.json）を読み込み、
  本体を編集せずに設定を上書きできます。
- 適用されるのは「本体の globals に存在するキーのみ」です（未知キーは無視されます）。
- JSON の場所は、EXTERNAL_LAUNCHER_CONFIG_PATH を編集して指定できます。

安全機能
- 入力画像が 0 枚のときは生成・壁紙設定を行わず中断します。
"""
from __future__ import annotations

HELP_TEXT = r"""
Kana Wallpaper - Unified FINAL（CLIヘルプ）
===========================================================

使い方
------
  py -3 kana_wallpaper_unified_final.py [TARGET ...] [OPTIONS]

TARGET（入力）
-------------
- フォルダ（サブフォルダ含む）
- アーカイブ（.zip / .7z / .rar）※設定で有効化している場合
- 動画（動画フレーム抽出を有効化している場合）

例：
  py -3 kana_wallpaper_unified_final.py .\images
  py -3 kana_wallpaper_unified_final.py "D:\Pictures" "E:\Photos"

OPTIONS（コマンドライン）
------------------------
ヘルプ
  -h / --help / -help
      このヘルプを表示して終了します。

サブフォルダ走査
  --recursive
      サブフォルダを含めて走査します（既定）。
  --top-only / --no-recursive
      トップ階層のみ走査します。

出力（保存先・ファイル名）
  --img-dir <dir>
      出力画像の保存先ディレクトリを指定します。
  --img-name <name>
      出力画像のベース名を指定します（拡張子は FORMAT に従います）。

ログ/記録（任意）
  --log-dir <dir>
      ログ類の保存先ディレクトリを指定します。
  --image / --no-image
      生成画像の保存を ON/OFF します。
  --logs / --no-logs
      使用リスト等の保存を ON/OFF します。
  --records / --no-records
      --logs / --no-logs の同義（互換用）です。

補足
----
- レイアウト、最適化、エフェクト、動画抽出、ZIP/7z/RAR対応などの多くは、
  コマンドラインではなく、スクリプト内の設定値／外部設定JSON／ランチャーで変更します。
- 外部設定JSON（ランチャーのエクスポート）を自動で読み込み、同名キーを上書きできます。
  - 既定: スクリプトと同じフォルダの kana_wallpaper_launcher_export.json
  - 置き場所は EXTERNAL_LAUNCHER_CONFIG_PATH を編集して指定できます。
  - 適用されるのは「本体の globals に存在するキーのみ」です（未知キーは無視されます）。
- 入力画像が 0 枚だった場合は、メッセージを表示して中断します（黒い壁紙を生成しません）。
- --logs/--records を有効にすると、画像パス（ZIP内パスを含む）が JSON/CSV/TXT に記録される場合があります。
  公開リポジトリへコミットしないよう、.gitignore の利用を推奨します。
"""

# =============================================================================
# 目次（core）
#   - 依存関係（インポート／任意依存）
#   - 外部設定（ランチャーエクスポートJSONの適用）
#   - 表示/UI（ANSI/Unicode・進捗・バナー）
#   - 入力スキャン（フォルダ／サブフォルダ／ZIP・7z・RAR／動画フレーム抽出）
#   - 動画抽出（random/uniform/scene/scene_best 等）とタイムライン並び
#   - 抽出（SELECT_MODE）と近似重複排除（dHash）
#   - レイアウト（grid／mosaic／hex／quilt／stained-glass／random）
#   - 並び（色順・前処理）/ 最適化（anneal/hill/checkerboard/spectral）
#   - レンダリング（貼り込み・マスク）
#   - 顔フォーカス（heuristic/AI/hex face-fit）
#   - エフェクト（光/色味/ディテール/明るさ/仕上げ）
#   - 保存・ログ（出力画像／キャッシュ／使用リスト／クラッシュログ／外部設定JSON）
#   - CLI（引数）/ エントリーポイント（main）
# =============================================================================
# 設定インデックス（ざっくり早見表）
#
# ランチャーがセットするキーと同名のものが多いです（直書き／環境変数／既定値）。
# 迷子になったら「この表 → 見出し → 変数名検索」が早いです。
#
# ■ 外部設定（ランチャーJSON）
#   - EXTERNAL_LAUNCHER_CONFIG_PATH : 外部設定JSONの場所（空ならcoreと同じフォルダ）
#   - EXTERNAL_LAUNCHER_CONFIG_FILE : 既定の外部設定JSON名（例: kana_wallpaper_launcher_export.json）
#
# ■ 入力・収集（スキャン）
#   - DEFAULT_TARGET_DIRS           : ダブルクリック時の既定探索先（複数）
#   - RECURSIVE                     : サブフォルダ走査（True/False）
#   - ZIP_SCAN_ENABLE               : ZIP内画像を候補に含める（on/off）
#   - SEVENZ_SCAN_ENABLE / RAR_SCAN_ENABLE : 7z/rar 内画像（任意）
#   - VIDEO_SCAN_ENABLE             : 動画フレーム抽出（任意）
#   - VIDEO_FRAME_FORMAT / VIDEO_FRAME_MAX_DIM : 抽出画像形式/最大辺（例: png, 1280）
#
# ■ 動画抽出（フレーム選別）
#   - VIDEO_FRAME_MODE              : "auto"（自動配分）/ "fixed"（固定）
#   - VIDEO_FRAME_PER_VIDEO_MAX     : 固定モード時の上限枚数
#   - VIDEO_FRAME_SELECT_MODE       : random / uniform / scene / scene_best / best_bright / best_sharp / best_combo
#   - VIDEO_SCENE_*                 : シーン切替検出・層化抽出（scene/scene_best 用）
#   - GRID_VIDEO_TIMELINE           : grid で抽出順（タイムスタンプ順）を保つ
#   - GRID_VIDEO_TIMELINE_ORDER     : asc / desc（時系列/逆順）
#
# ■ 抽出（SELECT_MODE）と近似重複排除
#   - SELECT_MODE                   : random / aesthetic / recent / oldest / name_asc / name_desc
#   - COUNT                         : 抽出枚数の目安（レイアウトにより使い方が異なる）
#   - SELECT_RANDOM_DEDUP / SELECT_DEDUP_ALWAYS : dHash近似重複排除
#   - DEDUPE_HAMMING                : 近似判定の許容距離（小さいほど厳しい）
#   - DHASH_CACHE_FILE              : 永続dHashキャッシュ（*.json）
#
# ■ レイアウト
#   - LAYOUT_STYLE                  : grid / hex / mosaic-uniform-height / mosaic-uniform-width / quilt / stained-glass / random
#   - ROWS / COLS                   : grid（行/列）
#   - HEX_TIGHT_ORIENT              : hex（row-shift/col-shift）
#   - QUILT_*                       : quilt（分割/最適化/顔フォーカス等）
#   - STAINED_GLASS_*               : stained-glass（ピース数/境界線/角数/角度/効果適用など）
#
# ■ 並び/最適化（例）
#   - MOSAIC_GLOBAL_ORDER           : spectral-hilbert 等（色順）
#   - MOSAIC_ENHANCE_PROFILE        : diagonal / hilbert / scatter
#   - GRID_OPTIMIZER / GRID_NEIGHBOR_OBJECTIVE : grid最適化（anneal 等）
#
# ■ 顔フォーカス（例）
#   - FACE_FOCUS_ENABLE / FACE_FOCUS_AI_ENABLE / FACE_FOCUS_AI_ALWAYS
#   - FACE_FOCUS_AI_BACKEND（yolov8_animeface / yunet / animeface）
#   - FACE_FOCUS_BIAS_Y             : 既定 0（全レイアウト共通の上下バイアス）
#   - HEX_FACE_SAFE_*               : hex の「顔欠け防止オートフィット」用（v123+）
#
# ■ エフェクト（まとめ）
#   - EFFECTS_ENABLE
#   - HALATION_*（ENABLE/INTENSITY/RADIUS/THRESHOLD/KNEE）
#   - TONECURVE_*（ENABLE/MODE/STRENGTH）
#   - SPLIT_TONE_*（ENABLE/SHADOW_* / HIGHLIGHT_* / BALANCE）
#   - LUT_*（ENABLE/FILE/STRENGTH）
#   - VIBRANCE_* / BW_EFFECT_ENABLE / SEPIA_*
#   - CLARITY_* / UNSHARP_* / DENOISE_* / DEHAZE_* / SHADOWHIGHLIGHT_*
#   - GRAIN_* / VIGNETTE_*
#   - BRIGHTNESS_MODE / AUTO_METHOD / AUTO_TARGET_MEAN / MANUAL_GAIN / MANUAL_GAMMA
#
# ■ 保存・ログ
#   - SAVE_IMAGE / IMAGE_SAVE_DIR / IMAGE_BASENAME
#   - SAVE_ARTIFACTS（使用リスト・キャッシュ等の保存）
#   - CRASH_LOG_FILE（例: kana_wallpaper_crash.log）
# =============================================================================
# =============================================================================
# セクション: 依存関係（import）
# =============================================================================

# =============================================================================
# 編集ガイド: レイアウト（モード）別に探しやすくする
#
# 迷子になったら：
#   1) このガイドで「モード名 → 主要キー」を見つける
#   2) 変数名で検索（Ctrl+F）
#   3) 該当の『セクション: レイアウト: ...』へジャンプ
#
# ■ grid
#   - GRID_* / GRID_OPTIMIZER_* / GRID_DIAG_* / GRID_RANDOM_* など
#
# ■ mosaic（uniform-height / uniform-width）
#   - MOSAIC_* / MOSAIC_DIAG_* / MOSAIC_OPTIMIZER_* など
#
# ■ hex
#   - HEX_* / HEX_DIAG_* / HEX_GLOBAL_* / HEX_FACE_FIT_* など
#
# ■ quilt
#   - QUILT_* / QUILT_WARP_* / QUILT_MERGE_* など
#
# ■ stained-glass（ステンドグラス）
#   - STAINED_GLASS_* （Voronoi生成 / 境界線 / 顔フォーカス / グローバル簡略化など）
#   - 重要：最大角数を厳密に下げすぎると隙間が出やすい（Voronoiの性質）
#
# ■ random（ランダム配置）
#   - RANDOM_* など
# =============================================================================
import sys, os, math, time, random, tempfile, csv, json, secrets, textwrap, threading, atexit
import subprocess, shutil
import hashlib
import logging
import re

# -----------------------------------------------------------------------------
# Ultralytics の冗長ログ抑制（配布向け）
# - ultralytics を import する前に設定しておくことで、環境行/summary の出力を抑えやすくします。
# - デバッグ時は FACE_FOCUS_DEBUG=True を使います（必要ならこのブロックをコメントアウト）。
# -----------------------------------------------------------------------------
try:
    os.environ.setdefault("YOLO_VERBOSE", "False")
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
except Exception as e:
    _kana_silent_exc('core:L188', e)
    pass
from logging.handlers import RotatingFileHandler
import io, zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Union, Any, Dict

# === KANA: Silent Exception Summary (to detect silent quality degradation) ===
# 目的: except Exception: pass / continue で握りつぶされる例外をカウントし、最後にまとめて表示する。
# - 落とさない方針は維持（例外は握りつぶす）しつつ、「何が起きてるか」を見える化する。
# 設定:
#   SILENT_EXC_SUMMARY_ENABLE (既定 True) : まとめ表示を有効化
#   SILENT_EXC_TOP_N          (既定 5)   : 上位N箇所を表示
#   SILENT_EXC_SAMPLE_MAX     (既定 5)   : サンプル（例外メッセージ）の最大件数
#   SILENT_EXC_VERBOSE        (既定 False): サンプル詳細も表示
try:
    from collections import Counter as _KANA_Counter
except Exception:
    _KANA_Counter = None  # type: ignore

_KANA_SILENT_EXC_TOTAL = 0
_KANA_SILENT_EXC_BY_WHERE = _KANA_Counter() if _KANA_Counter else None
_KANA_SILENT_EXC_SAMPLES = []
_KANA_SILENT_EXC_ATEXIT_REGISTERED = False

def _kana_silent_exc(where: str, e: Exception) -> None:
    """握りつぶす例外をカウントする（表示は atexit または明示呼び出しでまとめて行う）"""
    global _KANA_SILENT_EXC_TOTAL, _KANA_SILENT_EXC_BY_WHERE, _KANA_SILENT_EXC_SAMPLES
    _KANA_SILENT_EXC_TOTAL += 1
    try:
        if _KANA_SILENT_EXC_BY_WHERE is not None:
            _KANA_SILENT_EXC_BY_WHERE[where] += 1
    except Exception as e:
        _kana_silent_exc('core:L223', e)
        pass
    try:
        lim = int(globals().get("SILENT_EXC_SAMPLE_MAX", 5))
        if lim > 0 and len(_KANA_SILENT_EXC_SAMPLES) < lim:
            msg = str(e).replace("\n", " ").replace("\r", " ")
            if len(msg) > 200:
                msg = msg[:200] + "…"
            _KANA_SILENT_EXC_SAMPLES.append((where, e.__class__.__name__, msg))
    except Exception as e:
        _kana_silent_exc('core:L232', e)
        pass
def _kana_print_silent_exc_summary(_note_func) -> None:
    """まとめ表示（スパム防止のため1回だけ短く出す）"""
    try:
        if not bool(globals().get("SILENT_EXC_SUMMARY_ENABLE", True)):
            return
    except Exception:
        return
    global _KANA_SILENT_EXC_TOTAL, _KANA_SILENT_EXC_BY_WHERE, _KANA_SILENT_EXC_SAMPLES
    if not _KANA_SILENT_EXC_TOTAL:
        return
    try:
        top_n = int(globals().get("SILENT_EXC_TOP_N", 5))
    except Exception:
        top_n = 5
    parts = []
    try:
        if _KANA_SILENT_EXC_BY_WHERE is not None:
            for w, c in list(_KANA_SILENT_EXC_BY_WHERE.most_common(top_n)):
                parts.append(f"{w}×{c}")
    except Exception as e:
        _kana_silent_exc('core:L254', e)
        pass
    try:
        head = f"Silent exceptions: {_KANA_SILENT_EXC_TOTAL}"
        if parts:
            head += " | top: " + " / ".join(parts)
        _note_func(head)
        if bool(globals().get("SILENT_EXC_VERBOSE", False)):
            for w, cls, msg in _KANA_SILENT_EXC_SAMPLES:
                _note_func(f"  - {w}: {cls}: {msg}")
        else:
            # lead_overlay と SG_GLOBAL_SIMPLIFY は 1 件だけ短く出して原因特定を楽にする（スパムにならない）
            shown = set()
            for w, cls, msg in _KANA_SILENT_EXC_SAMPLES:
                ww = str(w)
                if ww in ("core:lead_overlay", "core:SG_GLOBAL_SIMPLIFY"):
                    if ww in shown:
                        continue
                    _note_func(f"  - {w}: {cls}: {msg}")
                    shown.add(ww)
                    if len(shown) >= 2:
                        break
    except Exception as e:
        _kana_silent_exc('core:L264', e)
        pass
def _kana_register_silent_exc_atexit(_note_func) -> None:
    """プロセス終了時に1回だけサマリを出す（ランチャー実行でも確実に表示される）"""
    global _KANA_SILENT_EXC_ATEXIT_REGISTERED
    if _KANA_SILENT_EXC_ATEXIT_REGISTERED:
        return
    try:
        import atexit as _atexit
        _atexit.register(lambda: _kana_print_silent_exc_summary(_note_func))
        _KANA_SILENT_EXC_ATEXIT_REGISTERED = True
    except Exception as e:
        _kana_silent_exc('core:L276', e)
        pass
# === /KANA: Silent Exception Summary ===

# -----------------------------------------------------------------------------
# 型エイリアス
# - ImageRef: Path または str（画像パス/アーカイブ内エントリ等の参照に使用）
# -----------------------------------------------------------------------------
ImageRef = Union[Path, str]

# =============================================================================
# セクション: 既定の入力フォルダ
# - ダブルクリック時に走査します（複数指定OK / サブフォルダ含む）
# - 相対パスは「このスクリプトの場所」基準です
# =============================================================================
DEFAULT_TARGET_DIRS = [
    (r".\\AI_images" if os.name == "nt" else r"./AI_images"),
]


# =============================================================================
# セクション: 出力/状態保存ベース（GitHub運用向け）
# - 生成画像・ログ・キャッシュ・ランチャーexport等の「副産物」をまとめて置く場所です。
# ✅ 基本：ここだけ触ればOK
#   - None: このスクリプトと同じ場所に "_kana_state" を作ってそこへ集約（おすすめ）
#   - 例: r"D:\kana_state"  または  r".\_out"（相対はこのスクリプト基準）
# =============================================================================
STATE_DIR: Optional[str] = None
STATE_DIR_BASENAME: str = "_kana_state"

def _core_state_dir() -> Path:
    """状態/副産物用の基準ディレクトリを返します（環境変数は使いません）。"""
    try:
        if STATE_DIR:
            p = Path(STATE_DIR).expanduser()
            if not p.is_absolute():
                p = (Path(__file__).resolve().parent / p).resolve()
        else:
            p = Path(__file__).resolve().parent / STATE_DIR_BASENAME
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        # 最後の砦：カレントへ（失敗しても致命にしない）
        try:
            p = Path.cwd() / STATE_DIR_BASENAME
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            return Path.cwd()


# =============================================================================
# セクション: 外部設定（ランチャーエクスポート）
# - 本体を書き換えず、JSONを読み込んで設定を上書きします
# =============================================================================
EXTERNAL_LAUNCHER_CONFIG_FILE = "kana_wallpaper_launcher_export.json"
EXTERNAL_LAUNCHER_CONFIG_PATH = str(_core_state_dir())  # 既定: STATE_DIR（=このスクリプトの隣/_kana_state）

def _coerce_like(old, v):
    """既存値 old の型に寄せて v を変換する（失敗したら v を返す）。"""
    try:
        if old is None:
            return v
        if isinstance(old, bool):
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.strip().lower() in ("1","true","yes","on","y")
            return bool(v)
        if isinstance(old, int) and not isinstance(old, bool):
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str):
                return int(float(v.strip()))
            return v
        if isinstance(old, float):
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                return float(v.strip())
            return v
        if isinstance(old, (list, tuple)):
            if isinstance(v, (list, tuple)):
                return list(v)
            if isinstance(v, str):
                return [v]
            return v
        if isinstance(old, str):
            return str(v)
    except Exception as e:
        _kana_silent_exc('core:L336', e)
        pass
    return v

def _load_external_launcher_config() -> Optional[Dict[str, Any]]:
    """ランチャーが書き出す外部設定JSONを読み込みます（環境変数は使わない）。"""
    try:
        raw = str(EXTERNAL_LAUNCHER_CONFIG_PATH or "").strip()
        if raw:
            p = Path(raw).expanduser()
            if p.is_dir():
                p = p / EXTERNAL_LAUNCHER_CONFIG_FILE
            cfg_path = p
        else:
            cfg_path = Path(__file__).resolve().parent / EXTERNAL_LAUNCHER_CONFIG_FILE

        if not cfg_path.exists():
            return None

        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None
def _apply_external_launcher_config() -> None:
    """外部設定JSONの値を globals() に適用する。"""
    try:
        cfg = _load_external_launcher_config()
        if not cfg:
            return
        g = globals()
        cfg = {k: v for (k, v) in cfg.items() if isinstance(k, str) and (not k.startswith("_"))}
        applied = 0
        applied_keys = []
        ignored_keys = []
        for k, v in cfg.items():
            if k not in g:
                ignored_keys.append(k)
                continue
            try:
                g[k] = _coerce_like(g.get(k, None), v)
                applied += 1
                applied_keys.append(k)
            except Exception as e:
                _kana_silent_exc('core:L393', e)
                pass
        # ログ（短め）
        if (applied > 0) or ignored_keys:
            src_p = str(globals().get("_EXTERNAL_LAUNCHER_CONFIG_LAST_PATH", ""))
            try:
                msg = f"外部設定を適用: {applied} keys"
                if ignored_keys:
                    msg += f" / ignored: {len(ignored_keys)}"
                if src_p:
                    msg += f" / src={src_p}"
                note(_lang(msg, msg))
            except Exception as e:
                _kana_silent_exc('core:L406', e)
                pass
    except Exception as e:
        _kana_silent_exc('core:L408', e)
        pass
# =============================================================================
# セクション: グローバル設定（キャンバス/基本パラメータ）
# =============================================================================
WIDTH, HEIGHT = 3840, 2160         # キャンバスサイズ（px）
MARGIN        = 0                  # 外周の余白（px）
GUTTER        = 1                  # 画像と画像の間隔（px）
FORMAT        = "png"              # "png" か "jpg"
BG_COLOR      = "#000000"          # 背景色（FIT時の余白やモザイクの隙間に見える色）

# レイアウトスタイルを指定します。
# - "grid"                  : 固定グリッド（ROWS×COLS）に均等配置
# - "hex"                   : 正六角・フラットトップ ハニカム充填
# - "mosaic-uniform-height" : 行の高さを一定にして横方向に詰めるモザイク
# - "mosaic-uniform-width"  : 列の幅を一定にして縦方向に詰めるモザイク
# - "quilt"                : Mondrian風（BSP分割）で大小ブロックを敷き詰め（隙間なし）
# - "random"                : 上記候補からランダムに選択
LAYOUT_STYLE = "grid"

# random の候補に含めるレイアウト（必要に応じて編集）
RANDOM_LAYOUT_CANDIDATES = [
    "grid",
    "hex",
    "mosaic-uniform-height",
    "mosaic-uniform-width",
    "quilt",
    "stained-glass",
]

# =============================================================================
# セクション: Grid（固定グリッド）
# =============================================================================

# Grid レイアウト用の行数と列数（目安）。ROWS×COLS 枚を上限として使用します。
ROWS, COLS = 5, 9
COUNT      = ROWS * COLS


try:
    import numpy as np  # オプション（無くてもOK）
except Exception:
    np = None

from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageStat, ImageEnhance, ImageDraw
# --- Archive scan summary (log formatting) ---
ARCHIVE_LOG_OPENED_THRESHOLD = 0  # kept for compatibility; opened is always shown

def _fmt_archives_summary(zip_entries, zip_opened, sevenz_entries, sevenz_opened, rar_entries, rar_opened):
    """Return compact archives summary string.
    Only show formats with entries>0. Show 'opened' only when it is large enough.
    Example: 'zip: entries=300, opened=2 | rar: entries=6088, opened=34'
    """
    parts = []
    def add(label, entries, opened):
        try:
            e = int(entries)
            o = int(opened)
        except Exception:
            return
        if e <= 0:
            return
        parts.append(f"{label}: entries={e}, opened={o}")

    add('zip', zip_entries, zip_opened)
    add('7z', sevenz_entries, sevenz_opened)
    add('rar', rar_entries, rar_opened)
    return " | ".join(parts) if parts else ""


# -----------------------------------------------------------------------------
# Pillow のリサンプリング互換性
# -----------------------------------------------------------------------------
# Pillow 10.0 以降では Image.LANCZOS などのトップレベルのリサンプリング定数が非推奨となり、
# Image.Resampling 列挙型へ移行しました。新旧両方のバージョンに対応するため、
# インポート時に Image.Resampling の有無をチェックし、存在する場合はそれを
# `Resampling` としてエイリアスします。存在しない場合は古い定数を同じ属性名で
# 持つダミークラスを用意します。本モジュール内のコードではトップレベルの定数ではなく、
# `Resampling.LANCZOS`、`Resampling.BILINEAR`、`Resampling.NEAREST` を参照するようにしてください。
try:
    # Pillow 9.1 以降は Image.Resampling が使えます
    Resampling = Image.Resampling  # type: ignore[attr-defined]
except AttributeError:
    class _Resampling:
        LANCZOS = Image.LANCZOS
        BILINEAR = Image.BILINEAR
        NEAREST = Image.NEAREST
    Resampling = _Resampling()

# 非推奨になった Image のリサンプリング定数を Resampling エイリアスにマッピングします。
# これらの属性を設定しておくことで、Image.LANCZOS などの参照が新しい Pillow でも
# 非推奨警告を出さずに解決されるようになります。
Image.LANCZOS = Resampling.LANCZOS
Image.BILINEAR = Resampling.BILINEAR
Image.NEAREST = Resampling.NEAREST


# -----------------------------------------------------------------------------
# 顔フォーカス用デバッグカウンタ（常に定義）
# -----------------------------------------------------------------------------
# どのレイアウト経路でも NameError を避けるため、モジュールロード時に初期化します。
# カウンタの意味（ログの読み方）
#   _FDBG（顔フォーカスの統計）
#     frontal/profile/upper/person : Haar/HOG 検出で候補になった回数（生ヒット数）
#     saliency/center       : サリエンシー/中心フォールバックを使った回数
#     reject_pos            : 位置条件で除外した回数（例：画面下すぎ等）
#     reject_ratio          : 縦横比など品質条件で除外した回数
#     errors                : OpenCV 処理で例外になった回数
#   _FDBG2（目検証・低品質扱いの統計）
#     eyes_ok/eyes_ng       : 目検証（strict_eyes）の成否回数
#     low_reject            : low-quality 扱いで除外した回数（allow_low 等）
_FDBG: Dict[str, Any] = {"cv2": None, "frontal":0, "profile":0, "anime":0, "ai":0, "upper":0, "person":0, "saliency":0, "center":0,
                         "reject_pos":0, "reject_ratio":0, "errors":0}
_FDBG2: Dict[str, Any] = {"eyes_ok":0, "eyes_ng":0, "low_reject":0,
                          "anime_face_ok":0, "anime_face_ng":0,
                          "anime_eyes_ok":0, "anime_eyes_ng":0, "ai_face_ok":0, "ai_face_ng":0}

# -----------------------------------------------------------------------------
# キャッシュ抑止（__pycache__を作らない）
# -----------------------------------------------------------------------------
sys.dont_write_bytecode = True

# -----------------------------------------------------------------------------
# 例外握りつぶし箇所のデバッグ（既定は無音）
# -----------------------------------------------------------------------------
# 既存コードには「except Exception: pass」が多数あります。
# ふだんは無音のまま動作を変えず、必要なときだけ原因を追えるようにします。
try:
    EXC_PASS_DEBUG
except NameError:
    EXC_PASS_DEBUG = False  # True のとき、握りつぶした例外を 1 回だけ警告表示

_EXC_PASS_WARNED = set()


# --- CPU描画の先読みヘルパ ---
def prefetch_ordered_safe(items, fn, ahead: int = 16, max_workers: int = 0):
    """thread pool で fn(item) を実行し、最大 `ahead` 件までタスクを同時実行して先読みします。

    元の順序のまま (item, result, exc) を yield します。
    exc が None でない場合は result は None になり、呼び出し側は同期パスにフォールバックできます。"""
    try:
        ahead = int(ahead)
    except Exception:
        ahead = 0
    if ahead <= 0:
        for item in items:
            try:
                yield item, fn(item), None
            except Exception as e:
                yield item, None, e
        return

    try:
        mw = int(max_workers)
    except Exception:
        mw = 0
    if mw <= 0:
        try:
            mw = max(1, int(os.cpu_count() or 4))
        except Exception:
            mw = 4

    from collections import deque
    it = iter(items)
    q = deque()
    with ThreadPoolExecutor(max_workers=mw) as ex:
        for _ in range(ahead):
            try:
                item = next(it)
            except StopIteration:
                break
            q.append((item, ex.submit(fn, item)))

        while q:
            item, fut = q.popleft()
            try:
                res = fut.result()
                yield item, res, None
            except Exception as e:
                yield item, None, e
            try:
                item2 = next(it)
            except StopIteration:
                item2 = None
            if item2 is not None:
                q.append((item2, ex.submit(fn, item2)))


def prefetch_ordered_mp_safe(items, fn, ahead: int = 16, max_workers: int = 0):
    """process pool で fn(item) を実行し、最大 `ahead` 件までタスクを同時実行して先読みします。

    元の順序のまま (item, result, exc) を yield します。
    exc が None でない場合は result は None になり、呼び出し側は同期パスにフォールバックできます。

    注意:
      - `fn` は pickle 可能（トップレベル関数）である必要があります。
      - Windows ではワーカー生成時に本モジュールが import されるため、import 時に重い処理をしないでください。"""
    try:
        ahead = int(ahead)
    except Exception:
        ahead = 0
    if ahead <= 0:
        for item in items:
            try:
                yield item, fn(item), None
            except Exception as e:
                yield item, None, e
        return

    try:
        mw = int(max_workers)
    except Exception:
        mw = 0
    if mw <= 0:
        try:
            mw = max(1, int(os.cpu_count() or 4))
        except Exception:
            mw = 4

    from collections import deque
    it = iter(items)
    q = deque()
    with ProcessPoolExecutor(max_workers=mw) as ex:
        for _ in range(ahead):
            try:
                item = next(it)
            except StopIteration:
                break
            q.append((item, ex.submit(fn, item)))

        while q:
            item, fut = q.popleft()
            try:
                res = fut.result()
                yield item, res, None
            except Exception as e:
                yield item, None, e
            try:
                item2 = next(it)
            except StopIteration:
                item2 = None
            if item2 is not None:
                q.append((item2, ex.submit(fn, item2)))


def _pf_worker_grid_render(arg):
    """grid タイル描画用のプロセス安全なワーカー。

    arg: (path, w, h, mode, grid_use_face_focus)
    returns: (kind, PIL.Image)"""
    p, w, h, mode, grid_use_ff = arg
    w = max(1, int(w))
    h = max(1, int(h))

    if bool(grid_use_ff):
        tile = _tile_render_cached(p, w, h, 'fill', use_face_focus=True)
        return 'fill_ff', tile

    if str(mode) == 'fit':
        tile = _tile_render_cached(p, w, h, 'fit', use_face_focus=False)
        return 'fit', tile

    tile = _tile_render_cached(p, w, h, 'fill', use_face_focus=False)
    return 'fill', tile


def _pf_worker_hex_render(item):
    """hex タイル描画用のプロセス安全なワーカー。

    item: (path, S)
    returns: (key, PIL.Image または None)"""
    p_t, S_t = item
    try:
        S_t = max(1, int(S_t))
        with open_image_safe(p_t, draft_to=(S_t, S_t), force_mode='RGB') as im_t:
            tile_t = _cover_square_face_focus(im_t, S_t, p_t)
        return str(p_t), tile_t
    except Exception:
        return str(p_t), None

def _pf_worker_mosaic_uh_render(item):
    """mosaic-uniform-height タイル描画用のプロセス安全なワーカー。

    item: (path, w, h)
    returns: PIL.Image"""
    p_t, wj_t, rhh = item
    wj_t = max(1, int(wj_t))
    rhh = max(1, int(rhh))
    with open_image_safe(p_t, draft_to=(wj_t, rhh), force_mode='RGB') as im_tt:
        return hq_resize(im_tt, (wj_t, rhh))


def _pf_worker_mosaic_uw_render(item):
    """mosaic-uniform-width タイル描画用のプロセス安全なワーカー。

    item: (path, w, h)
    returns: PIL.Image"""
    p_t, wj_t, h_t = item
    wj_t = max(1, int(wj_t))
    h_t = max(1, int(h_t))
    with open_image_safe(p_t, draft_to=(wj_t, h_t), force_mode='RGB') as im_tt:
        return hq_resize(im_tt, (wj_t, h_t))
# --- /CPU描画の先読みヘルパ ---

def _warn_exc_once(e: BaseException) -> None:
    """except Exception: pass された例外を、必要なときだけ 1 回だけ表示する。"""
    if not bool(globals().get("EXC_PASS_DEBUG", False)):
        return
    try:
        tb = e.__traceback__
        while tb and tb.tb_next:
            tb = tb.tb_next
        if tb:
            key = f"{tb.tb_frame.f_code.co_name}:{tb.tb_lineno}:{type(e).__name__}"
        else:
            key = f"unknown:{type(e).__name__}"
        if key in _EXC_PASS_WARNED:
            return
        _EXC_PASS_WARNED.add(key)
        msg = f"[WARN] swallowed exception at {key}: {e}"
        print(msg)
        _log_warn(msg)
    except Exception as _e:
        # ここで例外を出すと本末転倒なので、無音で戻る
        return


# -----------------------------------------------------------------------------
# 保存とログ（どこに何を残すか）
#   画像/リスト/統計などを Temp ではなく任意の場所に保存可能。
#   * 壁紙セットだけして画像を残さない、も選べます。
# -----------------------------------------------------------------------------
_P = Path  # 型注釈用の別名（互換のため残しています）

TEMP_DIR = _P(tempfile.gettempdir())

# 既定の入出力ベース（このスクリプトの隣/_kana_state に集約）
CORE_BASE_DIR: _P = _core_state_dir()

IMAGE_SAVE_DIR: _P = CORE_BASE_DIR      # 出力画像の保存先（変更可）
IMAGE_BASENAME: str = "kana_wallpaper_current"
LOG_SAVE_DIR:   _P = CORE_BASE_DIR      # 使用画像リスト等の保存先

SAVE_IMAGE:    bool = True         # 生成画像自体を保存するか
APPLY_WALLPAPER: bool = True      # 生成した画像を Windows 壁紙へ反映するか（連続出力用途などでOFFにできます）
SAVE_ARTIFACTS:bool = True         # 使用リスト・統計CSVなどの副産物を保存するか
DELETE_OLD_WHEN_DISABLED: bool = False  # 上2つを False にした時、古いファイルを消すか

# -----------------------------------------------------------------------------
# 表示（進捗・コンソールUI）とログ
#   ここは「よく触る」設定（言語/ASCII/Unicode/進捗）をまとめています。
# -----------------------------------------------------------------------------
VERBOSE        = True              # 処理の進捗や統計を表示

# 追加: 詳細デバッグログ（普段は OFF 推奨）
HEX_DEBUG_LOG  = False             # True で HexDBG の詳細を表示（VERBOSE 時のみ）
PROGRESS_EVERY = 1                 # 何枚ごとにバーを更新するか
PROGRESS_WIDTH = 40                # 進捗バーの横幅（文字数）
# 進捗バーの更新間隔。0.0 に設定すると時間ベースの制御を無効にし、
# PROGRESS_EVERY のステップ数のみで更新します。
# これにより処理完了後の一時停止感がなくなります。
PROGRESS_UPDATE_SECS = 0.016         # 進捗バーの更新間隔（秒）
PROGRESS_UPDATE_MODE = "secs"  # 進捗更新の方式: "every"=ステップ間隔（PROGRESS_EVERY）/ "secs"=時間間隔（PROGRESS_UPDATE_SECS）
UI_STYLE       = "ascii"           # "unicode" / "ascii"
UI_LANG = "ja"                     # "en" / "ja"
TREAT_AMBIGUOUS_WIDE = True        # East Asian Ambiguous を全角幅扱いにする（日本語向け）
FORCE_UTF8_CP  = False             # Windows コンソールを UTF-8 に切替（chcp 65001）
PROGRESS_BAR_STYLE = "segment"     # "segment"|"paint"

# -----------------------------------------------------------------------------
# ログ（任意）
#   既定では従来どおり print ベースで表示します。
#   LOG_ENABLE=True にすると、note()/banner()/警告などをファイルへ記録します。
#   ※ コンソール出力はそのまま（ログは基本ファイルのみ）
# -----------------------------------------------------------------------------
LOG_ENABLE = False
LOG_LEVEL  = "INFO"
LOG_FILE   = ""   # 空なら Temp に kana_wallpaper.log
CRASH_LOG_FILE = ""  # 空なら LOG_SAVE_DIR（あれば）→ Temp に kana_wallpaper_crash.log
LOG_MAX_BYTES    = 2_000_000
LOG_BACKUP_COUNT = 3

# 進捗バーとバナーに使うネオン調グラデ（RGB, 0-255）
#   - NEON_RANDOMIZE=True のときは、このリストから「連続色」を毎回ランダムに抜き出して使います。
#   - なので、このリストを“色相の順（虹の輪）”に並べて増やすほど、ランダムでも綺麗なグラデになります。
UNICODE_NEON_PALETTE = [
    (255,  64,  96),  # ネオン赤（ピンク寄り）
    (255,  96,  64),  # ネオン朱（コーラル）
    (255, 140,  64),  # ネオン橙
    (255, 200,  64),  # ネオン琥珀
    (255, 255,  80),  # ネオン黄
    (200, 255,  80),  # ネオン黄緑（チャート）
    (120, 255,  80),  # ネオンライム
    ( 80, 255, 120),  # ネオングリーン
    ( 80, 255, 200),  # ネオンミント
    ( 80, 255, 255),  # ネオンシアン
    ( 80, 200, 255),  # ネオン空色
    ( 80, 140, 255),  # ネオン蒼
    ( 80,  80, 255),  # ネオン青
    (140,  80, 255),  # ネオン藍
    (200,  80, 255),  # ネオン紫
    (255,  80, 255),  # ネオンマゼンタ
    (255,  80, 200),  # ネオン桃
    (255,  80, 140),  # ネオンホットピンク
    (255,  80,  96),  # ネオン赤（戻り）
]

# ---- セクション別の Unicode バナー用パレット -------------------
# セクションごとの配色。値は (R,G,B) の配列。自由にカスタム可。
# 注: NEON_RANDOMIZE=True のときは、基本的に UNICODE_NEON_PALETTE が優先されます。
#       ここは NEON_RANDOMIZE=False にしたときの“固定色”として効きます。
BANNER_PALETTES = {
    "scan":              [(120,255, 80),( 80,255,120),( 80,255,200),( 80,255,255),(255,255, 80)],  # スキャン系（爽やか）
    "render-grid":       [(255,200, 64),(255,140, 64),(255, 64, 96),(255, 80,200),(255,255, 80)],  # Grid 描画（熱め）
    "render-mosaic-h":   [( 80,255,255),( 80,255,200),(120,255, 80),(200,255, 80),(255,255, 80)],  # Mosaic 高さ均一（瑞々しい）
    "render-mosaic-w":   [(200, 80,255),(140, 80,255),( 80, 80,255),( 80,200,255),(255, 80,255)],  # Mosaic 幅均一（クール）
    "render-hex":        [(200, 80,255),( 80,255,255),(255, 80,255),(120,255, 80),( 80,140,255)],  # Hex（既定外キーだが保険）
    "opt-hill":          [(255,255, 80),(255,200, 64),(255,140, 64),(255, 64, 96),(255, 80,200)],  # 近傍最適化（hill）
    "opt-anneal":        [(255, 64, 96),(255,140, 64),(255,200, 64),(255,255, 80),(255, 80,140)],  # anneal
    "opt-checker":       [(255,255,255),( 80,255,255),(255, 80,255),(170,170,170),( 20, 20, 20)],  # チェッカー（白黒＋ネオン差し）
    "opt-spectral":      [( 80, 80,255),(140, 80,255),(200, 80,255),(255, 80,255),(255, 80,140)],  # スペクトル→Hilbert
    "preprocess":        [( 80,255,200),( 80,255,255),( 80,200,255),(200, 80,255),(120,255, 80)],  # 前処理（順序付け）
    "brightness":        [(255,255, 80),(255,200, 64),(255,140, 64),(255,255,255),(255, 64, 96)],  # 明るさ調整
    "done":              [( 80,255,120),( 80,255,255),(140, 80,255),(255, 80,255),(120,255, 80)],  # 完了
    "default":           [(255, 64, 96),(255,255, 80),( 80,255,120),( 80,255,255),( 80, 80,255),(200, 80,255),(255, 80,255),(255, 80,140)],  # 既定（虹）
}

# 進捗バーの文字（Unicode 推奨。ズレが出るなら "ascii" に）
BAR_FILL_CHAR  = "█"   # 全角幅2相当の環境でも幅計算を行うのでOK
BAR_EMPTY_CHAR = "░"

# ---- Unicode 装飾文字（おまけ表示） --------------------------------------------
# Unicodeのときだけ有効。ASCII時は自動で地味表示にフォールバック。
#   Quick Settings で UNICODE_BLING を設定済みの場合はそちらが優先されます。
UNICODE_BLING = True  # 再定義防止のため値を保持

# バナーや進捗のネオン配色：毎回ランダムにする
NEON_RANDOMIZE  = True

# 1回のパレットに使う色数（ランダム範囲）
NEON_COLORS_MIN_MAX = (3, 6)

# 現在のセクションとパレット（進捗バーでも共有）
CURRENT_SECTION  = "default"
CURRENT_PALETTE  = None

# レイアウト情報を重複表示しないためのガード
_PRINTED_LAYOUT_ONCE = False

# -----------------------------------------------------------------------------
# 乱数シード（再現性）
#   "random" で毎回変える／整数で固定再現可。
# -----------------------------------------------------------------------------
SHUFFLE_SEED = "random"            # 画像シャッフル用
OPT_SEED     = "same"              # 最適化用（"same"で SHUFFLE_SEED に追従）


def _seed_to_int(v):
    """seed の指定値を int に解決（失敗時は None）。 'random' は None。"""
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return int(v)
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        if s.lower() == "random":
            return None
        if s.isdigit():
            return int(s)
    except Exception as e:
        _kana_silent_exc('core:L878', e)
        pass
    return None


def _resolve_seed_aliases():
    """'same' を SHUFFLE_SEED に追従させる（本体を直接編集しやすくするため）。"""
    master = globals().get("SHUFFLE_SEED", "random")
    master_i = _seed_to_int(master)
    if master_i is None:
        # master が random の場合、alias はそのまま（random扱い）
        return

    opt = globals().get("OPT_SEED", "same")
    try:
        if isinstance(opt, str) and opt.strip().lower() in ("same", "follow", "shuffle"):
            globals()["OPT_SEED"] = int(master_i)
    except Exception as e:
        _kana_silent_exc('core:L895', e)
        pass
    hls = globals().get("HEX_LOCAL_OPT_SEED", None)
    try:
        if isinstance(hls, str) and hls.strip().lower() in ("same", "follow", "opt", "shuffle"):
            globals()["HEX_LOCAL_OPT_SEED"] = int(master_i)
        # None の場合は OPT_SEED を使う仕様なので何もしない
    except Exception as e:
        _kana_silent_exc('core:L903', e)
        pass
# -----------------------------------------------------------------------------
# 出力キャンバス（完成画像の土台）
#   壁紙として最終的に生成される “キャンバス” の大きさや見た目。
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# モザイクレイアウトで ROWS/COLS を固定行数・固定列数として利用するかどうか。
# MOSAIC_USE_ROWS_COLS を True にすると、mosaic‑uniform‑height では ROWS の値を縦方向の行数として厳守し、
# 横方向の枚数を自動的に調整します。mosaic‑uniform‑width では COLS の値を横方向の列数として厳守し、
# 縦方向の枚数を自動的に調整します。さらに、ROWS や COLS が正の値に設定されている場合は自動的に
# 固定行数／固定列数モードが有効になります。ROWS や COLS が 0 または負の場合は無視され、
# MOSAIC_USE_ROWS_COLS の設定で挙動が決まります。False の場合は従来通り、行数や列数は自動で決定されます。
try:
    MOSAIC_USE_ROWS_COLS
except NameError:
    MOSAIC_USE_ROWS_COLS = True

# -----------------------------------------------------------------------------
# アスペクト比を保ちながら矩形でトリミングする処理
# -----------------------------------------------------------------------------
# モザイク系レイアウトでは、固定サイズのタイルに合わせて画像をリサイズする必要があります。
# その際、元画像のアスペクト比を維持したままタイルの縦横サイズに合わせます。アスペクト比が異なる場合は、
# 一様なスケーリングで画像を拡大・縮小してターゲット矩形を「覆い」、
# そのうえで中央部分をトリミングします。以下のヘルパー関数はこの処理を行います。
# 顔を優先的に残す `_cover_rect_face_focus` のバリエーションを補完する役割もあります。
# 顔検出が無効な場合や失敗した場合に使用される汎用版です。

def _fit_rect_no_crop_no_upscale(im: Image.Image, cw: int, ch: int) -> Image.Image:
    """
    画像を (cw, ch) に「収める」ように貼り付ける（モザイク用）。

    ✅ クロップしない（cover しない）
    ✅ ズーム（拡大）しない（必要なら縮小のみ）
    ✅ 余白は背景色で埋める（元画像の平均色ベース）

    ※ モザイクは「そのまま貼る」方針のため、タイル内で画角を変えません。
    """
    # 入力が不正なら黒いタイル
    try:
        iw, ih = im.size
    except Exception:
        return Image.new("RGB", (max(1, cw), max(1, ch)), color=(0, 0, 0))
    cw = int(max(1, cw))
    ch = int(max(1, ch))
    if iw <= 0 or ih <= 0:
        return Image.new("RGB", (cw, ch), color=(0, 0, 0))

    # まず RGB へ（背景色計算と貼り付けを安定させる）
    try:
        src = im.convert("RGB")
    except Exception:
        src = im if getattr(im, "mode", None) == "RGB" else Image.new("RGB", (iw, ih), color=(0, 0, 0))

    # 背景色（平均色を 1x1 サンプルで高速に求める）
    try:
        bg = src.resize((1, 1), Resampling.BILINEAR).getpixel((0, 0))
        if not isinstance(bg, tuple):
            bg = (0, 0, 0)
        elif len(bg) >= 3:
            bg = (int(bg[0]), int(bg[1]), int(bg[2]))
        else:
            bg = (0, 0, 0)
    except Exception:
        bg = (0, 0, 0)

    # 縮小のみ（拡大しない）
    scale = min(cw / float(iw), ch / float(ih), 1.0)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))

    if new_w != iw or new_h != ih:
        try:
            src2 = src.resize((new_w, new_h), Resampling.LANCZOS)
        except Exception:
            src2 = src.resize((new_w, new_h), Resampling.BILINEAR)
    else:
        src2 = src

    out = Image.new("RGB", (cw, ch), color=bg)
    px = max(0, (cw - new_w) // 2)
    py = max(0, (ch - new_h) // 2)
    out.paste(src2, (px, py))
    return out


# === エフェクト ===
# エフェクト全体のON/OFF（ワンタッチ）。
# False にすると、ここで定義されたエフェクト群（Bloom/白黒/セピア/グレイン/ビブランス/ビネット/明るさ調整）を一括で無効化します。
try: EFFECTS_ENABLE
except NameError: EFFECTS_ENABLE = True

# ハレーション（Bloom）エフェクトを追加するかどうか。
# 元画像とガウシアンぼかし版を合成し、明るい部分をふんわり光らせる処理を行います。
# HALATION_INTENSITY と HALATION_RADIUS の値によって、強さと広がりを調整します。
try: HALATION_ENABLE
except NameError: HALATION_ENABLE = False
try: HALATION_INTENSITY
except NameError: HALATION_INTENSITY = 0.30
try: HALATION_RADIUS
except NameError: HALATION_RADIUS = 18

try: HALATION_THRESHOLD
except NameError: HALATION_THRESHOLD = 0.70  # 0.0〜1.0（明部抽出のしきい値。目安：0.60〜0.80）
try: HALATION_KNEE
except NameError: HALATION_KNEE = 0.08  # 0.0〜0.5（ソフトニー。しきい値付近の滑らかさ）
# 画像をグレースケール化するかどうか。
# True にすると、Bloom/Halation の後に白黒化を行います。False なら白黒化しません。
try: BW_EFFECT_ENABLE
except NameError: BW_EFFECT_ENABLE = False

# セピア調エフェクトを追加するかどうか。True なら暖色系のトーンを重ねます。
# 強度は SEPIA_INTENSITY で 0.0〜1.0 の範囲で調整可能です。
try: SEPIA_ENABLE
except NameError: SEPIA_ENABLE = False
try: SEPIA_INTENSITY
except NameError: SEPIA_INTENSITY = 0.03

# フィルムグレイン（粒子）エフェクトを追加するかどうか。True なら
# 輝度（明るさ）に“加算型”ノイズを加え、レトロな質感を与えます（色を濁しにくい方式）。
# 強度は GRAIN_AMOUNT で 0.0〜1.0 の範囲で調整します（値が大きいほどノイズが強くなります）。
try: GRAIN_ENABLE
except NameError: GRAIN_ENABLE = False
try: GRAIN_AMOUNT
except NameError: GRAIN_AMOUNT = 0.15  # 粒状ノイズの強さ（0.0〜1.0目安）

# 彩度ブースト（ビブランス）エフェクトを追加するかどうか。True にすると
# “彩度の低い色ほど”持ち上げ、元から鮮やかな色は上げすぎない（いわゆる vibrance）方式で
# 色の鮮やかさを調整します。VIBRANCE_FACTOR で調整可能です。
try:
    VIBRANCE_ENABLE
except NameError:
    VIBRANCE_ENABLE = False
try:
    VIBRANCE_FACTOR
except NameError:
    VIBRANCE_FACTOR = 1.0  # 1.0 が無調整。1.20〜1.40 が使いやすい目安（低彩度ほど強めに効く）

# スプリットトーン（影・ハイライトに色味を乗せる）エフェクト。
# 影（シャドウ）と明部（ハイライト）にそれぞれ指定した色相の“薄い色味”を乗せ、
# 写真を“作品っぽい”色に寄せます（セピアより現代的 / LUTより軽量）。
# - SPLIT_TONE_SHADOW_HUE       : 影に乗せる色相（0〜360）
# - SPLIT_TONE_SHADOW_STRENGTH  : 影の強さ（0.0〜1.0。目安 0.03〜0.10）
# - SPLIT_TONE_HIGHLIGHT_HUE    : 明部に乗せる色相（0〜360）
# - SPLIT_TONE_HIGHLIGHT_STRENGTH: 明部の強さ（0.0〜1.0。目安 0.03〜0.10）
# - SPLIT_TONE_BALANCE          : 影/明部のバランス（-1.0〜1.0。0.0 で中間）
try: SPLIT_TONE_ENABLE
except NameError: SPLIT_TONE_ENABLE = False
try: SPLIT_TONE_SHADOW_HUE
except NameError: SPLIT_TONE_SHADOW_HUE = 220.0
try: SPLIT_TONE_SHADOW_STRENGTH
except NameError: SPLIT_TONE_SHADOW_STRENGTH = 0.06
try: SPLIT_TONE_HIGHLIGHT_HUE
except NameError: SPLIT_TONE_HIGHLIGHT_HUE = 35.0
try: SPLIT_TONE_HIGHLIGHT_STRENGTH
except NameError: SPLIT_TONE_HIGHLIGHT_STRENGTH = 0.05
try: SPLIT_TONE_BALANCE
except NameError: SPLIT_TONE_BALANCE = 0.0

# トーンカーブ（階調）エフェクト。
# - "film"     : 影をほんのり持ち上げ、ハイライトを丸める“フィルムっぽい”階調
# - "liftgamma": 影〜中間をやや持ち上げる（ふんわり系）
# - "custom"   : 現状は film ベース（将来拡張用）
try: TONECURVE_ENABLE
except NameError: TONECURVE_ENABLE = False
try: TONECURVE_MODE
except NameError: TONECURVE_MODE = "film"   # "film" / "liftgamma" / "custom"
try: TONECURVE_STRENGTH
except NameError: TONECURVE_STRENGTH = 0.35  # 0.0〜1.0（目安：0.20〜0.45）


# クラリティ（局所コントラスト）エフェクトを追加するかどうか。
# True にすると、輝度（明るさ）成分の局所コントラストを持ち上げ、質感や立体感を出します。
# - CLARITY_AMOUNT : 強さ（0.0〜1.0 目安。0.08〜0.20 が使いやすい）
# - CLARITY_RADIUS : ぼかし半径（px。1.5〜3.0 が使いやすい）
try: CLARITY_ENABLE
except NameError: CLARITY_ENABLE = False
try: CLARITY_AMOUNT
except NameError: CLARITY_AMOUNT = 0.12
try: CLARITY_RADIUS
except NameError: CLARITY_RADIUS = 2.0


# アンシャープマスク（輪郭強調）エフェクトを追加するかどうか。
# True にすると、輝度（明るさ）成分の輪郭を少しだけ強調し、遠目の“キレ”を上げます。
# - UNSHARP_AMOUNT    : 強さ（0.0〜1.0 目安。0.25〜0.55 が使いやすい）
# - UNSHARP_RADIUS    : 半径（px。0.8〜1.8 が使いやすい）
# - UNSHARP_THRESHOLD : しきい値（0〜20。大きいほどノイズを拾いにくい / 目安 2〜6）
try: UNSHARP_ENABLE
except NameError: UNSHARP_ENABLE = False
try: UNSHARP_AMOUNT
except NameError: UNSHARP_AMOUNT = 0.35
try: UNSHARP_RADIUS
except NameError: UNSHARP_RADIUS = 1.2
try: UNSHARP_THRESHOLD
except NameError: UNSHARP_THRESHOLD = 3


# --- Shadow/Highlight（暗部救済・白飛び抑え） ---
# 暗部（シャドウ）を持ち上げ、明部（ハイライト）を抑えることで、
# 眠い/暗い/白飛び寄りの素材を“写真っぽく”整えます。
# 0.0〜1.0 で調整（目安：0.10〜0.35）。強すぎると不自然になりやすいです。
try: SHADOWHIGHLIGHT_ENABLE
except NameError: SHADOWHIGHLIGHT_ENABLE = False
try: SHADOW_AMOUNT
except NameError: SHADOW_AMOUNT = 0.22
try: HIGHLIGHT_AMOUNT
except NameError: HIGHLIGHT_AMOUNT = 0.18


# --- ノイズ除去（Noise Reduction） ---
# ノイズ除去を行うモードを選びます（壁紙用途の軽量版）。
#   - "off"   : 無効
#   - "light" : 輝度（明るさ）だけを薄くスムージング（軽い / 破綻しにくい）
#   - "median": 点ノイズ（塩胡椒）向け。細部が少し丸くなります
#   - "edge"  : エッジ保護（低解像度で軽量バイラテラル）※やや重い
#   - "heavy" : 強力（OpenCV があれば NLM / なければ強めバイラテラル）※重い
#
# DENOISE_STRENGTH は 0.0〜1.0（効き具合）。目安：0.15〜0.40
try: DENOISE_MODE
except NameError: DENOISE_MODE = "off"
try: DENOISE_STRENGTH
except NameError: DENOISE_STRENGTH = 0.25


# --- Dehaze（霞み抜き） ---
# 低コントラスト（霧/霞/白っぽい眠さ）を“抜く”エフェクトです。
# 目安：0.05〜0.20。強すぎると不自然になりやすいので、まずは弱め推奨です。
try: DEHAZE_ENABLE
except NameError: DEHAZE_ENABLE = False
try: DEHAZE_AMOUNT
except NameError: DEHAZE_AMOUNT = 0.10
try: DEHAZE_RADIUS
except NameError: DEHAZE_RADIUS = 24  # 解析半径（px）。大きいほど“大きな霞”に効きます（推奨 16〜40）。

# --- LUT（.cube）カラーグレーディング --------------------------------
# .cube 形式の 3D LUT を適用して、色の“世界観”を一発で切り替えます。
# LUT_FILE に .cube のパスを指定し、LUT_STRENGTH で効き具合（0.0〜1.0）を調整します。
# 目安：0.15〜0.40（薄めが上品）
try: LUT_ENABLE
except NameError: LUT_ENABLE = False
try: LUT_FILE
except NameError: LUT_FILE = ""  # .cube のファイルパス
try: LUT_STRENGTH
except NameError: LUT_STRENGTH = 0.30

try: VIGNETTE_ENABLE
except NameError: VIGNETTE_ENABLE = False
try: VIGNETTE_STRENGTH
except NameError: VIGNETTE_STRENGTH = 0.15  # ビネットの強さ（0.0〜1.0。目安：0.05〜0.30）
try: VIGNETTE_ROUND
except NameError: VIGNETTE_ROUND = 0.50  # 0.5 で左右上下が均等（おすすめ）

# -----------------------------------------------------------------------------
# 明るさ（背景を無視して自動調整）
#   “黒い余白やボーダー”の影響を受けないよう、貼り付け領域のマスクで統計。
#   - "auto"   : 目標平均輝度に合わせて gain/gamma を自動調整
#   - "manual" : MANUAL_* をそのまま適用
#   - "off"    : 無調整
# -----------------------------------------------------------------------------
BRIGHTNESS_MODE  = "off"          # "off" / "auto" / "manual"
AUTO_METHOD      = "hybrid"        # "gamma" / "gain" / "hybrid"
AUTO_TARGET_MEAN = 0.50            # 0.0〜1.0（目標の平均明るさ）
AUTO_GAIN_MIN,  AUTO_GAIN_MAX  = 0.75, 1.35
AUTO_GAMMA_MIN, AUTO_GAMMA_MAX = 0.3, 1.7
MANUAL_GAIN, MANUAL_GAMMA = 1.00, 1.00
# ガンマの極端さを和らげる係数（0=そのまま, 1=ガンマ無効化=常に1.0へ）
try: AUTO_GAMMA_SOFTEN
except NameError: AUTO_GAMMA_SOFTEN = 0.5

# --- Tempo（賑やか/静かの交互配置）ステージ設定 ---
try: ARRANGE_TEMPO_ENABLE
except NameError: ARRANGE_TEMPO_ENABLE = False # テンポON/OFF
try: ARRANGE_TEMPO_MODE
except NameError: ARRANGE_TEMPO_MODE = "alt" # "alt" | "2:1"

#   "pre"  : レンダラの最適化より前（入力順の事前整列）
#   "post" : レンダラの最適化の後（描画直前で強制整列）
#   "blend": 描画直前に小窓スワップで“軽く”交互化（配列全体は崩しすぎない）
try: ARRANGE_TEMPO_STAGE
except NameError: ARRANGE_TEMPO_STAGE = "blend"   # "pre" | "post" | "blend"

# blend 用の先読み窓幅（2〜4 推奨）
try: ARRANGE_TEMPO_WINDOW
except NameError: ARRANGE_TEMPO_WINDOW = 3

# True にすると、レイアウトの前に画像の順序を完全にランダム化します。
# この挙動は grid・hex・mosaic レイアウトに適用されます。
# 既定では元の並び順（色バランスの最適化なども含む）が維持されます。
# 毎回予測不能な並びにしたい場合は True に設定します。
# 再現性のあるシャッフルにしたい場合は OPT_SEED に整数値を指定してください。
# OPT_SEED が "random" 以外の場合、その値に基づいて同じ順序でシャッフルされます。
try:
    ARRANGE_FULL_SHUFFLE
except NameError:
    ARRANGE_FULL_SHUFFLE = False

# -----------------------------------------------------------------------------
# シャッフルの精密化（ハッシュシャッフル）
# -----------------------------------------------------------------------------
# 乱数器（random.Random）の内部状態に依存せず、seed と item から作ったハッシュ値で
# 安定ソートすることで “ハッシュシャッフル” を実現します。
#
# - seed が同じなら毎回同じ並び（再現性）
# - salt を変えると用途ごとに別の並び（抽出/フルシャッフル等）
# - 途中で乱数を使っても結果がぶれにくい（shuffle の結果が乱数消費に影響されにくい）

def _effective_seed(seed_value, bits: int = 128) -> int:
    """
    seed_value を int に正規化して返します。

        - None／"random"／変換不能: secrets.randbits(bits) で新しい seed を生成します
        - int に変換できる: その値を採用します
    """
    try:
        if seed_value is None:
            return secrets.randbits(bits)
        if isinstance(seed_value, str) and seed_value.lower() == "random":
            return secrets.randbits(bits)
        return int(seed_value)
    except Exception:
        return secrets.randbits(bits)


def _seed_to_16bytes(seed_int: int) -> bytes:
    """任意の int を 128bit（16バイト）に正規化する（符号は捨てて modulo）。"""
    try:
        v = int(seed_int)
    except Exception:
        v = 0
    v = v % (1 << 128)
    return v.to_bytes(16, "little", signed=False)


def _hash_shuffle_key(item, seed_bytes: bytes, salt_bytes: bytes) -> int:
    """item から安定ソート用のキー（整数）を作る。"""
    try:
        s = str(item)
    except Exception:
        s = repr(item)
    data = salt_bytes + b"\0" + s.encode("utf-8", "ignore")
    digest = hashlib.blake2b(data, key=seed_bytes, digest_size=16).digest()
    return int.from_bytes(digest, "little", signed=False)


def hash_shuffle(seq: Sequence, seed_value, salt: str = "") -> list:
    """ハッシュ方式でシャッフルした新しいリストを返す（元の seq は破壊しない）。"""
    seed_int = _effective_seed(seed_value)
    seed_bytes = _seed_to_16bytes(seed_int)
    salt_bytes = str(salt).encode("utf-8", "ignore")
    out = list(seq)
    # まれな衝突の順序ブレを避けるため、元の index もタイブレークに使う
    out = [(i, x) for i, x in enumerate(out)]
    out.sort(key=lambda t: (_hash_shuffle_key(t[1], seed_bytes, salt_bytes), t[0]))
    return [x for (_i, x) in out]


def hash_shuffle_inplace(lst: list, seed_value, salt: str = "") -> None:
    """ハッシュ方式で lst をその場でシャッフルする。"""
    try:
        lst[:] = hash_shuffle(lst, seed_value, salt)
    except Exception:
        # 最悪でも処理が止まらないように、例外時は通常 shuffle にフォールバック
        try:
            _sv = seed_value
            if _sv is None or (isinstance(_sv, str) and _sv.lower() == "random"):
                secrets.SystemRandom().shuffle(lst)
            else:
                random.Random(int(_sv)).shuffle(lst)
        except Exception as e:
            _warn_exc_once(e)
            pass
# -----------------------------------------------------------------------------
# Grid 近傍色差の最適化（隣接セルの色を似せる/離す）
#   目的(OBJECTIVE):
#     - "max": 色差を最大化 → バラけて“パズル感”
#     - "min": 色差を最小化 → グラデーション/連続感
#   アルゴリズム(GRID_OPTIMIZER):
#     - "hill"             : ヒルクライム（軽い・速い）
#     - "anneal"           : anneal（局所解脱出に強い／重い）
#     - "checkerboard"     : 明暗で2分割→市松→近傍貪欲
#     - "spectral-hilbert" : 色ベクトル→2D射影→Hilbert 曲線沿い
#     - "spectral-diagonal": 色ベクトルの射影を対角線方向に並べるスペクトル対角モード
# -----------------------------------------------------------------------------
GRID_NEIGHBOR_OBJECTIVE = "max"    # "max" or "min"
GRID_OPTIMIZER = "anneal"
GRID_DIAGONAL_DIRECTION = "random"   # "random","tlbr","trbl","bltr","brtl"
GRID_DIAGONAL_ZIGZAG   = True

# annealの調整（重いほど効く）
GRID_ANNEAL_STEPS   = 20000        # 総ステップ数
GRID_ANNEAL_T0      = 1.0          # 初期温度（大きいほど悪化移動も受理しやすい＝探索が広い）
GRID_ANNEAL_TEND    = 1e-3         # 終了温度（小さいほど収束が強い＝仕上げが締まる）
GRID_ANNEAL_REHEATS = 4            # 再加熱回数（0〜3程度）
GRID_ANNEAL_ENABLE = False       # True で“anneal(anneal)”を追加適用（Tune用）
# -----------------------------------------------------------------------------
# Mosaic の最適化（行/列の “詰まり具合” と “並びの色差”）
#   - バランス最適化: 行幅/列高の偏りを下げる（溢れ・バラつき抑制）
#   - 並び色差      : 各行/列の順番を最適化（swap／2opt）
#   - グローバル順序: 詰める前に全体の順序を整える（色の並びを先に決める）
# -----------------------------------------------------------------------------
# =============================================================================
# セクション: Mosaic（モザイク）
# =============================================================================
MOSAIC_BALANCE_ENABLE = True       # 行/列のバランス最適化を有効にする

# 行/列内の色差目的（"max"=バラけ、"min"=滑らか）
MOSAIC_NEIGHBOR_OBJECTIVE = "min"
MOSAIC_NEIGHBOR_ITERS_PER_LINE = 200

# 行/列の並び最適化アルゴリズム
#   "swap"／"2opt"／"swap+2opt"（粗→微の二段構え推奨）
MOSAIC_SEQ_ALGO = "swap+2opt"

# バランス最適化の挙動
BALANCE_EARLY_STOP       = True    # 改善停滞で早期終了
BALANCE_RESTART_ON_STALL = True    # 一度だけランダム再スタート

# 詰める前の “全体の並び” を決める（mosaic 系の前処理）
#   "none"／"spectral-hilbert"／"anneal"（フェイスフォーカス時の並び替えモード）
MOSAIC_GLOBAL_ORDER      = "spectral-hilbert"
MOSAIC_GLOBAL_OBJECTIVE  = "min"   # "max" or "min"
MOSAIC_GLOBAL_ITERS      = 40000   # anneal 時の反復量の目安

# モザイクの拡張割り当て（post-pack）
#  - 有効時は先にレイアウトの幾何を決め、その後タイルへ画像を割り当てる
#      タイル位置順（diagonal／Hilbert／checkerboard風）で割り当て、必要なら
#      ローカルk近傍アニールで微調整する。
MOSAIC_ENHANCE_ENABLE     = True   # True: post-pack割り当て（拡張）＋（任意）ローカル最適化を有効化
MOSAIC_ENHANCE_PROFILE    = "diagonal"  # "diagonal" / "hilbert" / "scatter"（旧: "checker"） ※ "off" は旧互換
MOSAIC_DIAGONAL_DIRECTION = "tl_br"  # 対角グラデの向き: "tl_br" / "br_tl" / "tr_bl" / "bl_tr"

MOSAIC_LOCAL_OPT_ENABLE   = True    # 初期配置後のローカル最適化（k近傍アニール）を回す
MOSAIC_LOCAL_OPT_STEPS    = 20000   # 総ステップ数（大きいほど強いが重い）
MOSAIC_LOCAL_OPT_REHEATS  = 4       # 再加熱回数（局所解脱出用。0〜3程度が目安）
MOSAIC_LOCAL_OPT_K        = 8       # k近傍グラフの k（>=3 推奨。大きいほど“広い近傍”を見て重い）
MOSAIC_POS_HILBERT_ORDER  = 10      # 位置ヒルベルトの次数（10 → 1024×1024 解像度相当）


# -----------------------------------------------------------------------------
# モザイクレイアウトのギャップレス拡張フラグ
# True にすると、uniform-height／uniform-width のモザイクで行や列を追加画像で拡張し、
# 右・下・左・上の各辺に余白が残らないようにします。
# 拡張後のモザイクは余剰部分を左右均等に切り取って中央に揃えます。
# これにより残った隙間が解消され、hex レイアウトと同様の挙動になります。
# 参考: ギャップレス拡張は「余白ゼロ」を優先するため、全体を少しオーバーフィルしてから中央で整列します。
#       post-pack（グラデ/散らし）と併用する場合でも、タイル内のアス比は維持されます。
# ギャップレス拡張を無効にするには、このフラグを False にするか、
# MOSAIC_GAPLESS_EXTEND="off" を環境変数またはコマンドラインで指定してください。
try:
    if isinstance(MOSAIC_GAPLESS_EXTEND, str):
        MOSAIC_GAPLESS_EXTEND = MOSAIC_GAPLESS_EXTEND.lower() != "off"
except NameError:
    MOSAIC_GAPLESS_EXTEND = True

# ギャップレス拡張で「不足分を同じ画像で埋める（繰り返し）」を許可するか。
# False（既定）なら、供給が尽きたらそこで拡張を止め、余白（背景）が残る場合があります。
try:
    if isinstance(MOSAIC_GAPLESS_ALLOW_REPEAT, str):
        MOSAIC_GAPLESS_ALLOW_REPEAT = MOSAIC_GAPLESS_ALLOW_REPEAT.lower() in ("1", "true", "yes", "on")
except NameError:
    MOSAIC_GAPLESS_ALLOW_REPEAT = False

# 供給画像（gapless拡張で追加される画像）でも近似重複を避けるか。
# True（既定）なら、すでに使った画像と dHash が近いものはスキップします。
try:
    if isinstance(MOSAIC_GAPLESS_SUPPLY_DEDUPE, str):
        MOSAIC_GAPLESS_SUPPLY_DEDUPE = MOSAIC_GAPLESS_SUPPLY_DEDUPE.lower() in ("1", "true", "yes", "on")
except NameError:
    MOSAIC_GAPLESS_SUPPLY_DEDUPE = True

# 供給画像の近似重複しきい値（Hamming距離）。None/未指定なら 0（完全一致のみ）を使います。
try:
    _t = globals().get("MOSAIC_GAPLESS_SUPPLY_DEDUPE_HAMMING", None)
    MOSAIC_GAPLESS_SUPPLY_DEDUPE_HAMMING = int(_t) if _t is not None else 0
except Exception:
    MOSAIC_GAPLESS_SUPPLY_DEDUPE_HAMMING = 0

# -----------------------------------------------------------------------------
# 自動補間／隙間検出フラグ
# True にすると、モザイク描画の最後に生成されたマスクを検査して、
# ギャップレス拡張後にも残る細い隙間（縦または横の筋）を検出・修正します。
# 検出は表示範囲内（margin〜margin+W または H）で完全に空白の行や列を探し、
# その隙間を取り除くようにクロップ位置を1ピクセル調整します。これにより
# 1ピクセル単位の丸め誤差による隙間を抑制しますが、モザイク全体が1ピクセル分
# ずれる可能性があります。この動作を無効にするには、このフラグを False にするか、
# MOSAIC_AUTO_INTERPOLATE="off" を環境変数やコマンドラインで指定してください。
try:
    if isinstance(MOSAIC_AUTO_INTERPOLATE, str):
        MOSAIC_AUTO_INTERPOLATE = MOSAIC_AUTO_INTERPOLATE.lower() != "off"
except NameError:
    MOSAIC_AUTO_INTERPOLATE = True


# -----------------------------------------------------------------------------
# リサンプル品質（縮小の画質）
#   "default": 直接 LANCZOS（十分高品質。軽め）
#   "hq"     : 段階的 BOX → 最終 LANCZOS（大幅縮小での滲み/モアレ低減）
# -----------------------------------------------------------------------------
RESAMPLE_MODE = "hq"

# -----------------------------------------------------------------------------
# 最適化ループのスケール（ヒルクライム/その他で参照）
#   処理時間と相談。値を上げるほど“粘る”。
# -----------------------------------------------------------------------------
OPT_ITERS = 1500

# Mosaic Uniform Width の割り当て方法:
#   "minheight" : もっとも低い列に順番に積み、高さバランスを優先します。
#   "roundrobin" : 列に順番に割り当て、グローバルな色順を崩しにくくします。
#   "snake"      : 左右に往復しながら割り当て、バランスと色順の両立を狙います。
MOSAIC_UW_ASSIGN = "roundrobin"

# Mosaic Uniform Width の列順の整列方法:
#   "first-rank" : 各列の最初の画像の全体順位に基づいてソートします。
#   "avg-rank"   : 各列内の画像順位の平均でソートします。
#   "avgLAB"     : 各列内画像の平均 LAB 値でソートし、色の流れを強調します。
#   "none"       : 列順を変更しません。
MOSAIC_UW_COL_ORDER = "avgLAB"

# Mosaic Uniform Height の割り当て（固定行数時）。Uniform Width と対称に使えるようにする。
# - "packed"     : 既存の詰め込み（左→右、収まらなければ次の行）
# - "minheight"  : （UHでは minwidth 相当）現在もっとも短い行へ割り当て（バランス優先）
# - "roundrobin" : 行へ順番に割り当て（色順を崩しにくい）
# - "snake"      : 往復しながら割り当て（バランスと色順の両立）
MOSAIC_UH_ASSIGN = MOSAIC_UW_ASSIGN

# Mosaic Uniform Height の行順。Uniform Width の MOSAIC_UW_COL_ORDER と対称。
# - 選択肢: "first-rank"／"avg-rank"／"avgLAB"／"none"
# None の場合は、旧設定 MOSAIC_UH_ORDER_ROWS に従って自動決定します。
MOSAIC_UH_ROW_ORDER = None


# Mosaic Uniform Height の行の並べ替えを色順にするかどうか。
# True にすると各行の平均 LAB 値に基づいて縦方向の色の流れを作ります。
MOSAIC_UH_ORDER_ROWS = True

# Mosaic (uniform-height) の行高の探索レンジ（ピクセル）。
# 画面高に対し「何行くらいで詰めるか」を決定します。
JUSTIFY_MIN_ROW_H = 30
JUSTIFY_MAX_ROW_H = 2160

# Mosaic (uniform-width) の列幅の探索レンジ（ピクセル）。
# 既定では行高と同じレンジを使用しますが、別途指定も可能です。
JUSTIFY_MIN_COL_W = JUSTIFY_MIN_ROW_H
JUSTIFY_MAX_COL_W = JUSTIFY_MAX_ROW_H

# 平均Labベクトルの簡易キャッシュ（プロセス内）
_AVG_LAB_CACHE = {}

# Grid セル内のリサイズ方法
#   - "fit"  : 画像のアスペクト比を維持。セルに収まるまで縮小（未充満の余白あり）
#   - "fill" : 画像の一部をトリミングしてセルを完全に満たす（余白なし）
MODE = "fill"

# =============================================================================
# セクション: Hex（ハニカム）
# =============================================================================
try: HEX_TIGHT_ENABLE
except NameError: HEX_TIGHT_ENABLE = True  # 六角形で隙間なく詰める機能を有効にするか。Trueで有効。
try: HEX_TIGHT_MAX_COLS
except NameError: HEX_TIGHT_MAX_COLS = 128  # 六角配置の最大列数。この数を超える場合は列数を制限します。
try: HEX_TIGHT_BG
except NameError: HEX_TIGHT_BG = None  # 六角配置で特別な背景色を指定します。None なら全体の BG_COLOR を使います。
try: HEX_TIGHT_GAP
except NameError: HEX_TIGHT_GAP = None   # ← None のときは GUTTER を採用。この値で六角タイル間の間隔を制御できます。
try: HEX_TIGHT_SEAM_EPS
except NameError: HEX_TIGHT_SEAM_EPS = 0  # 六角タイルの継ぎ目補正用ピクセル数。つなぎ目の隙間を埋めるための許容量。
try: HEX_TIGHT_DILATE
except NameError: HEX_TIGHT_DILATE = 1  # 六角マスクの膨張量。値を大きくすると角が丸くなりアンチエイリアスが強まります。
try: HEX_TIGHT_ORIENT
except NameError: HEX_TIGHT_ORIENT = "col-shift"   # "col-shift" / "row-shift" : ハニカムのシフト方向。列または行ごとにずらします。
try: HEX_TIGHT_EXTEND
except NameError: HEX_TIGHT_EXTEND = 2  # 画面外へ何層分タイルを拡張するか。端の空白を無くすための外周タイル数です。


# --- HEX Face-fit（六角で顔が欠けないようにクロップ補正） ---
# いずれも「タイルサイズ（正方窓 S）」に対する比率です（0.0〜1.0 の想定）。
# 例: HEX_FACE_SAFE_TOP=0.10 なら、顔BBoxの上に最低 10% の余白を確保する方向へ補正します。
try: HEX_FACE_SAFE_TOP
except NameError: HEX_FACE_SAFE_TOP = 0.10  # 上余白（額/頭の切れ防止）
try: HEX_FACE_SAFE_BOTTOM
except NameError: HEX_FACE_SAFE_BOTTOM = 0.06  # 下余白（顎の切れ防止）
try: HEX_FACE_SAFE_XCENTER_MIN
except NameError: HEX_FACE_SAFE_XCENTER_MIN = 0.33  # 顔中心Xの下限（左に寄りすぎ防止）
try: HEX_FACE_SAFE_XCENTER_MAX
except NameError: HEX_FACE_SAFE_XCENTER_MAX = 0.67  # 顔中心Xの上限（右に寄りすぎ防止）
try: HEX_FACE_SAFE_YCENTER_MIN
except NameError: HEX_FACE_SAFE_YCENTER_MIN = 0.40  # 顔中心Yの下限（上に寄りすぎ防止）
try: HEX_FACE_SAFE_YCENTER_MAX
except NameError: HEX_FACE_SAFE_YCENTER_MAX = 0.70  # 顔中心Yの上限（下に寄りすぎ防止）
try: HEX_FACE_SAFE_TOP_BAND
except NameError: HEX_FACE_SAFE_TOP_BAND = 0.40  # 上部帯（この範囲では斜めカットを強く意識）
try: HEX_FACE_SAFE_SIDE_TOP
except NameError: HEX_FACE_SAFE_SIDE_TOP = 0.22  # 上部帯の左右安全域（斜めカットの欠け防止）


# --- HEX 全体/局所最適化（色グラデ） ---
# hex でも grid/mosaic のように「色の並び最適化」を効かせるための設定。
#   HEX_GLOBAL_ORDER（全体の並び）:
#     - "inherit" : MOSAIC_GLOBAL_ORDER を流用
#     - 選択肢: "none"／"spectral-hilbert"／"anneal"
#   HEX_GLOBAL_OBJECTIVE（目的関数）:
#     - "min" : 近い色を近く（グラデ向き）
#     - "max" : 近い色を離す（散らし向き）
try: HEX_GLOBAL_ORDER
except NameError: HEX_GLOBAL_ORDER = "inherit"
try: HEX_GLOBAL_OBJECTIVE
except NameError: HEX_GLOBAL_OBJECTIVE = "max"
try: HEX_GLOBAL_ITERS
except NameError: HEX_GLOBAL_ITERS = 40000

try: HEX_DIAG_DIR
except NameError: HEX_DIAG_DIR = "tlbr"  # "tlbr"/"brtl"/"trbl"/"bltr" (diagonal direction for HEX spectral-diagonal)

# hex 専用: 6近傍（六方向）隣接コストでの局所最適化（anneal）
try: HEX_LOCAL_OPT_ENABLE
except NameError: HEX_LOCAL_OPT_ENABLE = True

# hex 最適化の簡易スイッチ（外部JSON/ランチャ用）
# - "inherit"(既定): HEX_LOCAL_OPT_ENABLE をそのまま使う（従来互換）
# - "off"/"none"/"false"/"0": 局所最適化を無効化（高速）
# - それ以外: 局所最適化を有効化（高品質）
try: HEX_OPTIMIZER
except NameError: HEX_OPTIMIZER = "inherit"

try: HEX_LOCAL_OPT_OBJECTIVE
except NameError: HEX_LOCAL_OPT_OBJECTIVE = "inherit"   # "inherit": HEX_GLOBAL_OBJECTIVE を継承 / "min" / "max"
try: HEX_LOCAL_OPT_STEPS
except NameError: HEX_LOCAL_OPT_STEPS = 20000         # 総ステップ数（大きいほど強いが重い）
try: HEX_LOCAL_OPT_REHEATS
except NameError: HEX_LOCAL_OPT_REHEATS = 4           # 再加熱回数（局所脱出用。0〜3程度が目安）
try: HEX_LOCAL_OPT_T0
except NameError: HEX_LOCAL_OPT_T0 = 1.0              # 初期温度（大きいほど悪化移動も受理しやすい）
try: HEX_LOCAL_OPT_TEND
except NameError: HEX_LOCAL_OPT_TEND = 1e-3           # 終了温度（小さいほど収束が強い／貪欲寄り）
try: HEX_LOCAL_OPT_MAX_DEG
except NameError: HEX_LOCAL_OPT_MAX_DEG = 6
try: HEX_LOCAL_OPT_SEED
except NameError: HEX_LOCAL_OPT_SEED = None  # None のときは OPT_SEED を使う

try: KANA_FORCE_HEX
except NameError:
    # （既定）hex を強制しません。"on" にすると LAYOUT_STYLE に関係なく hex レンダラを使います。
    # grid を優先したい場合や互換性重視なら "off" のままにしてください。
    KANA_FORCE_HEX = "off"
# -----------------------------------------------------------------------------
# Hex 補充シャッフル
# Hex レイアウトでは画面全体の可視セル数に合わせて候補画像を補充しますが、
# 既存の処理では補充画像がリストの末尾に追加されるため、描画順が偏り
# （左側と右側で見た目の均一性が損なわれる）ことがあります。
# このフラグを True にすると、補充後に画像リストをシャッフルして
# 補充した画像が均等に散らばるようにします。False にすると従来の挿入順のままです。
try: HEX_TOPUP_SHUFFLE
except NameError: HEX_TOPUP_SHUFFLE = True

# hex 補充の interleave 用フラグ（新規）。
#   True にすると六角レイアウトで可視タイル数が選択枚数を上回る場合、
#   追加分を元のリストに交互に挿入して全体の並びを均質化します。
#   False の場合は従来通り末尾に連結します。
try:
    HEX_TOPUP_INTERLEAVE
except NameError:
    # 六角レイアウトで補充した画像を元のリストに交互に挿入するかどうか。
    # True にすると、左側と右側で同じくらい均一に混ざります。
    HEX_TOPUP_INTERLEAVE = True


# -----------------------------------------------------------------------------
# Hex: タイル生成メモリキャッシュ（高速化／安定化）
#   Hex レイアウト描画では、同じ画像を何度も参照することがあります。
#   ProcessPool で「ユニーク画像のタイル生成」を先に作ってキャッシュしておくと draw が爆速になります。
#   ただしキャッシュ上限が小さいと、ユニーク数が上限を超えた瞬間にキャッシュが溢れて
#   “残りが1プロセス逐次生成”に落ちて急激に遅くなることがあります。
#
#   HEX_TILE_MEMCACHE_MAX : Hex 用タイルキャッシュの最大保持数（既定 512）
#     - 0 で無効（メモリ節約だが遅くなりやすい）
#     - 500枚以上など大量タイルで遅くなる場合は 1024〜4096 を目安に増やしてください。
# -----------------------------------------------------------------------------
try:
    HEX_TILE_MEMCACHE_MAX = int(HEX_TILE_MEMCACHE_MAX)
except NameError:
    HEX_TILE_MEMCACHE_MAX = 4096
#  0: 無制限（追い出しなし）／-1: 無効（キャッシュを使わない）
except Exception:
    HEX_TILE_MEMCACHE_MAX = 512

# -----------------------------------------------------------------------------
# 入力スキャン＆抽出（どの画像を何枚使うか）
#   サブフォルダ含めて集め、好みで“美選抜＋重複除去”をかけられます。
# -----------------------------------------------------------------------------
RECURSIVE   = True
IMAGE_EXTS  = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff",".jfif"}

# Zip 圧縮ファイル内の画像も候補に含める（.zip／.cbz）
#   - True にするとスキャン時に Zip を開いて中の画像を列挙します（候補数が増えます）。
#   - Zip が多い/巨大だとスキャンが遅くなるので、必要なときだけ ON 推奨です。
try:
    ZIP_SCAN_ENABLE
except NameError:
    ZIP_SCAN_ENABLE = True

try:
    ZIP_SCAN_EXTS
except NameError:
    ZIP_SCAN_EXTS = {".zip", ".cbz"}

# 7z / rar アーカイブ内画像も候補に含める（任意）
#   - py7zr / rarfile が無い場合はスキップします（警告は1回だけ）。
try:
    SEVENZ_SCAN_ENABLE
except NameError:
    SEVENZ_SCAN_ENABLE = True
try:
    RAR_SCAN_ENABLE
except NameError:
    RAR_SCAN_ENABLE = True
try:
    SEVENZ_SCAN_EXTS
except NameError:
    SEVENZ_SCAN_EXTS = {".7z", ".cb7"}
try:
    RAR_SCAN_EXTS
except NameError:
    RAR_SCAN_EXTS = {".rar", ".cbr"}

# 走査の暴走防止（Zip が大量にある環境向け）
try:
    ZIP_SCAN_MAX_ZIPS
except NameError:
    ZIP_SCAN_MAX_ZIPS = 200

try:
    ZIP_SCAN_MAX_MEMBERS_PER_ZIP
except NameError:
    ZIP_SCAN_MAX_MEMBERS_PER_ZIP = 50000

# Zip 内の隠し/メタフォルダをざっくり無視する
try:
    ZIP_SCAN_SKIP_HIDDEN
except NameError:
    ZIP_SCAN_SKIP_HIDDEN = True


# -----------------------------------------------------------------------------
# 動画からフレームを抽出して「候補画像」に混ぜる（任意）
#   - VIDEO_SCAN_ENABLE        : 動画フレーム抽出を有効化（on/off）
#   - VIDEO_SCAN_EXTS          : 対象拡張子（小文字）
#   - VIDEO_FRAMES_PER_VIDEO   : 動画1本あたりの抽出最大枚数（上限）
#   - VIDEO_FRAME_SELECT_MODE  : 抽出方式
#       "random"       : 無作為にフレームを選ぶ（軽い）
#       "uniform"      : 全体から等間隔に選ぶ（軽い）
#       "scene"        : シーン切替（大きな変化点）を優先して選ぶ（おすすめ）
#       "scene_best"   : scene の候補から「明るさ＋シャープさ」でベストを厳選（壁紙向き）
#       "best_bright"  : 候補を多めにサンプルし、明るいフレームを優先
#       "best_sharp"   : 候補を多めにサンプルし、シャープなフレームを優先
#       "best_combo"   : 明るさ＋シャープさの混合スコアで優先（バランス）
#   - VIDEO_FRAME_SCORE_CANDIDATES : best_* 用に評価する候補数（大きいほど厳選だが遅い）
#   - VIDEO_FRAME_MAX_DIM      : 保存するフレーム画像の最大辺（ディスク/メモリ節約）
#   - VIDEO_FRAME_CACHE_DIR    : 抽出フレームのキャッシュ先（未指定なら temp 配下）
#   - VIDEO_ASPECT_FROM_CONTAINER : コンテナのSAR/DAR（表示アス比）で補正する（ffprobe がある場合）
#
# 依存: opencv-python（cv2）が必要です。未導入の場合は自動的にスキップします。
# -----------------------------------------------------------------------------
try:
    VIDEO_SCAN_ENABLE
except NameError:
    VIDEO_SCAN_ENABLE = True

try:
    VIDEO_SCAN_EXTS
except NameError:
    VIDEO_SCAN_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".flv", ".wmv", ".mpg", ".mpeg"}

try:
    VIDEO_FRAMES_PER_VIDEO
except NameError:
    VIDEO_FRAMES_PER_VIDEO = 0


try:
    # 0 以下の場合は「自動配分」（必要枚数と動画本数から 1 本あたりの抽出枚数を決定）になります。
    # 明示的に無効にしたい場合は VIDEO_SCAN_ENABLE=False を使用してください。
    VIDEO_FRAMES_PER_VIDEO_CAP
except NameError:
    # 自動配分時の 1 本あたり上限（0 で上限なし）
    VIDEO_FRAMES_PER_VIDEO_CAP = 0

try:
    VIDEO_FRAMES_AUTO_MARGIN
except NameError:
    # 自動配分時の余裕係数（dHash 等で間引かれても不足しにくいように少し多めに抽出）
    VIDEO_FRAMES_AUTO_MARGIN = 1.5

try:
    VIDEO_FRAMES_AUTO_MIN_PER_VIDEO
except NameError:
    # 自動配分時の最小枚数（候補が十分ある場合でも、動画混ぜONなら最低この枚数は抽出）
    VIDEO_FRAMES_AUTO_MIN_PER_VIDEO = 3
try:
    VIDEO_FRAME_SELECT_MODE
except NameError:
    VIDEO_FRAME_SELECT_MODE = "random"

# scene/scene_best 用のパラメータ
try:
    VIDEO_SCENE_SCAN_FPS
except NameError:
    # シーン検出用の走査FPS（低いほど軽いが、細かい切替を見落としやすい）
    VIDEO_SCENE_SCAN_FPS = 3

try:
    VIDEO_SCENE_ANALYZE_MULT
except NameError:
    # scene/scene_best で「動画全体の均等サンプル」を何倍調べるか。大きいほど精度↑・時間↑
    VIDEO_SCENE_ANALYZE_MULT = 3.0

try:
    VIDEO_SCENE_THRESHOLD
except NameError:
    # シーン切替とみなす変化量しきい値（0〜1）。大きいほど厳しめ。
    VIDEO_SCENE_THRESHOLD = 0.35

try:
    VIDEO_SCENE_MIN_GAP_SEC
except NameError:
    # 近接した切替の連発を抑えるための最小間隔（秒）
    VIDEO_SCENE_MIN_GAP_SEC = 1.0

try:
    VIDEO_SCENE_OFFSET_SEC
except NameError:
    # シーン切替検出点から、実際に抜く位置を少し後ろへずらす（暗転直後回避）
    VIDEO_SCENE_OFFSET_SEC = 0.4

try:
    VIDEO_SCENE_BEST_WINDOW_SEC
except NameError:
    # scene_best で、オフセット後のどれくらいの範囲からベストを探すか（秒）
    VIDEO_SCENE_BEST_WINDOW_SEC = 0.8

try:
    VIDEO_FRAME_QUALITY_LUMA_MIN
except NameError:
    # 暗すぎるフレームを除外（0-255）
    VIDEO_FRAME_QUALITY_LUMA_MIN = 20

try:
    VIDEO_FRAME_QUALITY_LUMA_MAX
except NameError:
    # 白飛び気味のフレームを除外（0-255）
    VIDEO_FRAME_QUALITY_LUMA_MAX = 235

try:
    VIDEO_FRAME_QUALITY_SHARP_MIN
except NameError:
    # ブレ（低シャープ）を除外する最低値（ラプラシアン分散）。0 で無効。
    VIDEO_FRAME_QUALITY_SHARP_MIN = 0.0

try:
    GRID_VIDEO_TIMELINE
except NameError:
    # grid のみ：動画抽出フレームをタイムライン（時系列）順に固定します。
    #  "off" / "asc"（時系列） / "desc"（逆順）
    GRID_VIDEO_TIMELINE = "off"

try:
    GRID_VIDEO_TIMELINE_SPREAD
except NameError:
    # GRID_VIDEO_TIMELINE が asc/desc のとき、動画全体から満遍なくフレームを選びます。
    # （dHash 早期終了で後半が欠けるのを防ぐため）
    GRID_VIDEO_TIMELINE_SPREAD = True

try:
    GRID_VIDEO_TIMELINE_DEDUP_SCOPE
except NameError:
    # 'local'：序盤と似ているだけで後半が弾かれないよう、直近だけで比較します。
    # 'global'：通常のdedupe同様、全体比較（後半が欠けやすい場合あり）。
    GRID_VIDEO_TIMELINE_DEDUP_SCOPE = 'local'

try:
    GRID_VIDEO_TIMELINE_DEDUP_WINDOW
except NameError:
    # scope='local' のときだけ使用します。
    GRID_VIDEO_TIMELINE_DEDUP_WINDOW = 24

try:
    GRID_VIDEO_TIMELINE_FORCE_PICK
except NameError:
    # bin内で非重複が見つからない場合でも、そのbinから何か1枚は採用します。
    # （タイムライン全体のカバレッジ優先）
    GRID_VIDEO_TIMELINE_FORCE_PICK = True

try:
    VIDEO_FRAME_SCORE_CANDIDATES
except NameError:
    VIDEO_FRAME_SCORE_CANDIDATES = 16

try:
    VIDEO_FRAME_MAX_DIM
except NameError:
    VIDEO_FRAME_MAX_DIM = 1280


try:
    VIDEO_FRAME_CACHE_FORMAT
except NameError:
    # 動画フレームのキャッシュ保存形式（"png" / "jpg"）。デフォルトは png（推奨）。
    VIDEO_FRAME_CACHE_FORMAT = "png"

try:
    VIDEO_FRAME_JPEG_QUALITY
except NameError:
    # jpg を選んだ場合の画質（0-100）
    VIDEO_FRAME_JPEG_QUALITY = 92

try:
    VIDEO_FRAME_PNG_COMPRESSION
except NameError:
    # png を選んだ場合の圧縮レベル（0=無圧縮〜9=最大圧縮）。速度/容量バランスは 3〜6 目安。
    VIDEO_FRAME_PNG_COMPRESSION = 3

try:
    VIDEO_FRAME_CACHE_DIR
except NameError:
    VIDEO_FRAME_CACHE_DIR = str(CORE_BASE_DIR / "kana_wallpaper_video_frames_cache")

try:
    VIDEO_FRAME_CACHE_CLEAR_ON_START
except NameError:
    VIDEO_FRAME_CACHE_CLEAR_ON_START = False  # True で起動時に動画フレームキャッシュを毎回削除します（容量肥大化対策）

try:
    VIDEO_FRAME_CACHE_CLEAR_ON_END
except NameError:
    VIDEO_FRAME_CACHE_CLEAR_ON_END = True  # True で処理終了後に動画フレームキャッシュを削除します（容量肥大化対策）

try:
    VIDEO_DAR_LOG
except NameError:
    VIDEO_DAR_LOG = False  # True で動画の DAR/SAR 関連ログを表示します（VERBOSE 時のみ）

try:
    VIDEO_FRAME_CACHE_CLEAR_FORCE
except NameError:
    VIDEO_FRAME_CACHE_CLEAR_FORCE = False  # True で安全チェックを無視して削除します（非推奨）


try:
    SCAN_PRECOUNT_ENABLE
except NameError:
    # スキャン進捗の分母を安定させるため、事前にファイル数を数えます（大規模だと遅くなるため上限あり）
    SCAN_PRECOUNT_ENABLE = True

try:
    SCAN_PRECOUNT_MAX
except NameError:
    # 事前カウントがこの数を超えたら中止して「分母なし」表示へフォールバック
    SCAN_PRECOUNT_MAX = 50_000


try:
    VIDEO_ASPECT_FROM_CONTAINER
except NameError:
    VIDEO_ASPECT_FROM_CONTAINER = True


try:
    VIDEO_FFPROBE_PATH
except NameError:
    # ffprobe のパスを明示したい場合に指定（例: r"C:\\ffmpeg\\bin\\ffprobe.exe"）
    # 未指定なら PATH から自動検出します。
    VIDEO_FFPROBE_PATH = None

VIDEO_DIAG_SHOW_FFPROBE_LINE = False  # core側で ffprobe パス行を出す（既定False。ランチャー診断があるため）

# Zip 内エントリを表すキー（内部用）
ZIP_KEY_PREFIX = "zip://"
ZIP_KEY_SEP = "::"
SEVENZ_KEY_PREFIX = "7z://"
RAR_KEY_PREFIX = "rar://"


# 動画由来フレームを「表示用」に表すキー（内部用）
VIDEO_KEY_PREFIX = "video://"
VIDEO_KEY_SEP = "::"


# 抽出方法:
#   - "random"    : 無作為に選ぶ
#   - "aesthetic" : 明瞭度/コントラスト/エッジ等の簡易スコアで並べて上位を選ぶ
# -----------------------------------------------------------------------------
# 抽出モードの既定値
#   - "random"  : 画像リストをシャッフルして先頭から COUNT 枚を選びます。
#   - "aesthetic" : 明瞭度/コントラスト/エッジ等の簡易スコアで並べ替え、上位を選びます。
#   - "recent"    : 更新日時が新しい順に選びます（もっとも最近更新されたファイルが先頭）。
#   - "oldest"    : 更新日時が古い順に選びます。
#   - "name_asc"  : ファイル名の昇順（英数字順）に選びます。
#   - "name_desc" : ファイル名の降順に選びます。
SELECT_MODE = "random"

# -----------------------------------------------------------------------------
# 近似重複の除去設定
SHOW_RANDOM_DEDUP_PROGRESS = False  # 近似重複除去の走査バー表示（分母が大きく分かりづらいので既定OFF）

# いつでも近似重複除去（dHash）を有効にする（全抽出モード共通）
#   True : random / aesthetic / recent / oldest / name_asc / name_desc すべてで近似重複除去を試みます。
#   False: 各モードの個別スイッチ（SELECT_RANDOM_DEDUP / SELECT_RECENT_DEDUP / SELECT_SORT_DEDUP）に従います。
try: SELECT_DEDUP_ALWAYS
except NameError: SELECT_DEDUP_ALWAYS = True

#   pick_recent()／pick_sorted_generic() は、True の場合 dHash による近似重複除去を行います。
#   重複を厳しく除去すると、似た構図の新しい画像が多い場合に大幅に間引かれてしまい、
#   代わりに古い画像が大量に補充されることがあります。通常は False を推奨します。
#   必要に応じて値を変更してください。
SELECT_RECENT_DEDUP: bool = False   # recent 抽出時に dHash で近似重複を除去する（False 推奨）
SELECT_SORT_DEDUP:   bool = False   # oldest/name_asc/name_desc 抽出時に dHash で近似重複を除去する（False 推奨）

# ランダム抽出でも近似重複（dHash）を避けるか（既定は ON）
try: SELECT_RANDOM_DEDUP
except NameError: SELECT_RANDOM_DEDUP = True

# topped_up（補充）が発生したとき、似た画像が隣り合いにくいよう並び順を分散する
try: SPREAD_RANDOM_WHEN_TOPPED_UP
except NameError: SPREAD_RANDOM_WHEN_TOPPED_UP = True

# topped_up（補充）時に、可能な限り DEDUP_DHASH_THRESHOLD を満たす（= 近似扱いにならない）候補で埋める
#   - TOPUP_STRICT_DEDUP_ENABLE           : True なら『閾値を満たす候補』を優先して探す
#   - TOPUP_STRICT_DEDUP_TIMEOUT_SEC      : 探索の打ち切り時間（秒）
#   - TOPUP_STRICT_DEDUP_RELAX_IF_TIMEOUT : True なら時間内に埋まらなければ『遠い順』で妥協補充も許可（埋め切り優先）
try: TOPUP_STRICT_DEDUP_ENABLE
except NameError: TOPUP_STRICT_DEDUP_ENABLE = True
try: TOPUP_STRICT_DEDUP_TIMEOUT_SEC
except NameError: TOPUP_STRICT_DEDUP_TIMEOUT_SEC = 5.0
try: TOPUP_STRICT_DEDUP_RELAX_IF_TIMEOUT
except NameError: TOPUP_STRICT_DEDUP_RELAX_IF_TIMEOUT = True

# 重複近似排除（dHash の Hamming 距離しきい値）。値を大きくすると「近い画像」まで重複扱いになり、除去が強くなります。
DEDUPE_HAMMING = 8  # 近似重複排除：dHashの許容Hamming距離（大きいほど厳しい）

# しきい値の一貫性確保
try:
    DEDUP_DHASH_THRESHOLD  # type: ignore[name-defined]
except NameError:
    DEDUP_DHASH_THRESHOLD = DEDUPE_HAMMING

# -----------------------------------------------------------------------------
# dHash 計算の高速化（永続キャッシュ・先読み）
#   近似重複除去を有効にしたとき、同じ画像に対して dHash を何度も計算すると遅くなるため、
#   ここでは dHash の結果をファイルに保存して再利用できるようにします。
#
#   - 永続キャッシュ: 実行ディレクトリに .dhash_cache.json を自動生成（既定）
#   - 先読み（prefetch）: 近似重複除去の対象になりそうな画像の dHash を前もって計算
#
# 速度にしか影響しない設計にしてあるため、基本は ON 推奨です。
# -----------------------------------------------------------------------------
try: DHASH_CACHE_ENABLE
except NameError: DHASH_CACHE_ENABLE = True      # dHash 永続キャッシュを使う
try: LAB_CACHE_ENABLE
except NameError: LAB_CACHE_ENABLE = True       # 平均LabベクトルをdHashキャッシュに保存して再利用（checkerboard/spectral等の高速化）
try: FACE_CACHE_ENABLE
except NameError: FACE_CACHE_ENABLE = True  # 顔フォーカスのキャッシュで、AI（YOLO/YuNet/AnimeFaceなど）の結果だけは保存/再利用しない。
# - True（既定）: AIは毎回計算。CPU系（Haar/サリエンシー等）はキャッシュ可。
# - False: すべての顔候補（AI含む）をキャッシュして再利用（速度優先）。
try: FACE_CACHE_DISABLE_AI
except NameError: FACE_CACHE_DISABLE_AI = True
# 顔/上半身検出結果をdHashキャッシュに保存して再利用（face focus高速化）
try: DHASH_CACHE_FILE
except NameError: DHASH_CACHE_FILE = str(CORE_BASE_DIR / "kana_wallpaper.dhash_cache.json")  # dHash キャッシュ保存先（既定=C:\\kana_wallpaper）
try: DHASH_CACHE_MAX
except NameError: DHASH_CACHE_MAX = 200000      # キャッシュの上限（超えたら古いものから間引き）

try: DHASH_PREFETCH_ENABLE
except NameError: DHASH_PREFETCH_ENABLE = True   # dHash 先読みを使う
try: DHASH_PREFETCH_WORKERS
except NameError: DHASH_PREFETCH_WORKERS = max(1, min(8, (os.cpu_count() or 4)))  # スレッド数
try: DHASH_PREFETCH_AHEAD
except NameError: DHASH_PREFETCH_AHEAD = 0       # 先読み対象の枚数（0=自動。大きすぎると無駄が増えます）


# 描画プリフェッチ（レンダリング高速化／CPU）
#   大量タイルの「開く→リサイズ→（face-focus等）→貼る」のうち、
#   “開く/リサイズ”を先読みして、貼り付けループの待ち時間を減らします。
#
#   DRAW_PREFETCH_ENABLE  : True/False（全体ON/OFF）
#   DRAW_PREFETCH_AHEAD   : 先読み数。0 にするとプリフェッチ無効（既定 16）
#   DRAW_PREFETCH_WORKERS : 0 なら自動（CPUコア数）。固定したい場合は 4〜(コア数) を指定。
#     ※ 値を上げすぎるとディスクI/O・メモリ消費が増えます。
#        伸びが頭打ちになったら workers を下げるのが吉です。
# -----------------------------------------------------------------------------
try:
    if isinstance(DRAW_PREFETCH_ENABLE, str):
        DRAW_PREFETCH_ENABLE = DRAW_PREFETCH_ENABLE.strip().lower() not in ("0", "false", "off", "no")
except NameError:
    DRAW_PREFETCH_ENABLE = True

try:
    DRAW_PREFETCH_AHEAD = int(DRAW_PREFETCH_AHEAD)
except NameError:
    DRAW_PREFETCH_AHEAD = 16
except Exception:
    DRAW_PREFETCH_AHEAD = 16

try:
    DRAW_PREFETCH_WORKERS = int(DRAW_PREFETCH_WORKERS)
except NameError:
    DRAW_PREFETCH_WORKERS = 0
except Exception:
    DRAW_PREFETCH_WORKERS = 0


#   DRAW_PREFETCH_AUTO    : True/False（解像度が大きいときに先読み量を自動で控えめにします）
#   DRAW_PREFETCH_AHEAD_4K: 4K級(>=約12MP)の上限（既定 16）
#   DRAW_PREFETCH_AHEAD_8K: 8K級(>=約30MP)の上限（既定 8）
try:
    if isinstance(DRAW_PREFETCH_AUTO, str):
        DRAW_PREFETCH_AUTO = DRAW_PREFETCH_AUTO.strip().lower() not in ("0", "false", "off", "no")
except NameError:
    DRAW_PREFETCH_AUTO = True

try:
    DRAW_PREFETCH_AHEAD_4K = int(DRAW_PREFETCH_AHEAD_4K)
except NameError:
    DRAW_PREFETCH_AHEAD_4K = 16
except Exception:
    DRAW_PREFETCH_AHEAD_4K = 16

try:
    DRAW_PREFETCH_AHEAD_8K = int(DRAW_PREFETCH_AHEAD_8K)
except NameError:
    DRAW_PREFETCH_AHEAD_8K = 8
except Exception:
    DRAW_PREFETCH_AHEAD_8K = 8
#   DRAW_PREFETCH_BACKEND : 'thread' / 'process'（既定は環境依存だが、ここでは thread を既定にする）
try:
    DRAW_PREFETCH_BACKEND = str(DRAW_PREFETCH_BACKEND).strip().lower()
except NameError:
    DRAW_PREFETCH_BACKEND = "thread"
except Exception:
    DRAW_PREFETCH_BACKEND = "thread"


# -----------------------------------------------------------------------------
# 描画キャッシュ: タイルメモリキャッシュ（全レイアウト共通）
#   レンダリング時に同じ画像を繰り返し開く/変換するのを減らすための簡易キャッシュです。
#   既存コードは globals().get(..., default) で既定値を持っているため、
#   ここは“設定を見つけやすくする目的”で明示しています（挙動は同等）。
#
#   TILE_MEMCACHE_ENABLE    : True/False（既定 True）
#   TILE_MEMCACHE_MAX_ITEMS : 最大保持数（既定 512）
#   TILE_MEMCACHE_MAX_BYTES : 最大保持バイト数（既定 256MiB）
# -----------------------------------------------------------------------------
try: TILE_MEMCACHE_ENABLE
except NameError: TILE_MEMCACHE_ENABLE = True
try: TILE_MEMCACHE_MAX_ITEMS
except NameError: TILE_MEMCACHE_MAX_ITEMS = 512
try: TILE_MEMCACHE_MAX_BYTES
except NameError: TILE_MEMCACHE_MAX_BYTES = 256 * 1024 * 1024

try: FACE_FOCUS_ENABLE
except NameError: FACE_FOCUS_ENABLE = True         # 顔フォーカスを使うか: 顔検出によるクロップを有効にするかどうか。
try: FACE_FOCUS_RATIO
except NameError: FACE_FOCUS_RATIO = 0.42          # 顔の高さがタイルの何割になるよう縮尺調整。0.42なら約4割に顔をフィットさせます。

# 顔中心を上にずらす比率（負値で上寄せ）。-0.10 は上に 10% 移動。
try: FACE_FOCUS_BIAS_Y
except NameError:
    FACE_FOCUS_BIAS_Y = 0.0

# 水平方向の顔位置を調整するバイアス。
#   正の値で顔を右に、負の値で左にずらします。通常は 0.0（中央）です。
try:
    FACE_FOCUS_BIAS_X
except NameError:
    FACE_FOCUS_BIAS_X = 0.0

# 顔中心自動調整フラグ。
# True のとき、検出した顔の中心をタイル中央に合わせるために
# FACE_FOCUS_BIAS_X/Y の値を無視して自動計算します。
# False のときは従来のバイアス設定を使用します。
try:
    FACE_FOCUS_CENTER_FACE
except NameError:
    # 顔中心を自動的にタイル中央へ合わせるかどうか
    FACE_FOCUS_CENTER_FACE = False

# 検出した顔の幅がセル幅に対して占める割合を制限する係数。
#   1.0 なら顔の幅をセル幅いっぱいに拡大します。細いセルで全体が見えるよう
#   調整したい場合は 0.8 などに下げてください。
try:
    FACE_FOCUS_WIDTH_FRAC
except NameError:
    FACE_FOCUS_WIDTH_FRAC = 0.8

# Grid レイアウトの fill モードで顔フォーカスを有効にするかどうか。
# True にすると grid+fill 時に顔（目）がタイル内に来るようにクロップします。
try: GRID_FACE_FOCUS_ENABLE
except NameError: GRID_FACE_FOCUS_ENABLE = True

# Mosaic レイアウトでは face-focus を使いません（クロップ/ズームを避け、処理コストも削減）。


# すべてのレイアウトでフォーカス検出（Face→Person→Saliency）を適用するか。
# True のとき、grid/mosaic/quilt/hex/random など「fill クロップ」する箇所で毎回フォーカス推定を試みます。
# ※重い場合は False にして、各レイアウト個別の *_FACE_FOCUS_ENABLE で制御できます。
try:
    FACE_FOCUS_FORCE_ALL_MODES
except NameError:
    FACE_FOCUS_FORCE_ALL_MODES = True

# =============================================================================
# セクション: Quilt
# =============================================================================
# - QUILT_MAX_TILES   : 生成するタイル数の上限（0 なら入力枚数に合わせる）
# - QUILT_MIN_SHORT   : タイル短辺の最小値（px）。小さすぎて何の絵かわからない問題を防ぐ
# - QUILT_MAX_ASPECT  : タイル縦横比の最大値。max(w/h, h/w) がこれを超えるタイルは作らない
# - QUILT_SPLIT_RANGE : 分割比率（低,高）。極端な分割を避けて見栄えを安定させる
# - QUILT_STOP_PROB   : 途中で分割を止める確率（0で止めない）
# - QUILT_SPLIT_STYLE : 分割位置の選び方。
#     "classic" : 中央寄り（三角分布）
#     "mixed"   : 中央寄り＋たまに端寄り（切り刻み感UP）
#     "extreme" : 端寄りを多め（大胆な分割）
#     "uniform" : 一様（合法範囲で均等）
# - QUILT_MULTI_SPLIT_PROB : 分割直後にもう一度分割する確率（線を増やして複雑に）
# - QUILT_PICK_STYLE : 分割対象の選び方。
#     "topk"          : 既存（面積上位から選ぶ）
#     "area_weighted" : 面積で重み付けして選ぶ（偏りが減って複雑になりやすい）
# - QUILT_FACE_FOCUS_ENABLE : タイルごとに顔フォーカス（クロップ）を適用する（fill のときのみ）
QUILT_MAX_TILES   = 0
QUILT_MIN_SHORT   = 220
QUILT_MAX_ASPECT  = 2.5
QUILT_SPLIT_RANGE = (0.12, 0.88)
QUILT_STOP_PROB   = 0.0
QUILT_SPLIT_STYLE = "mixed"
QUILT_MULTI_SPLIT_PROB = 0.35
QUILT_PICK_STYLE = "area_weighted"
QUILT_FACE_FOCUS_ENABLE = True

# Quilt: タイル形状と画像の縦横比（アスペクト）をなるべく合わせる（任意）
#  - 縦に細長いタイルには縦長画像、横に細長いタイルには横長画像を優先して割り当てます。
#  - 画像が足りない場合は、残りから埋めます（"なるべく" の挙動）。
#  - Quilt の近傍anneal（最適化）を行う場合も、タイル種別（縦長/横長/中立）内で swap することで
#    この相性が崩れにくいようにします。
QUILT_ASPECT_MATCH_ENABLE = True
QUILT_ASPECT_MATCH_EPS = 0.12

# Quilt: ランダム感（不規則さ）を増やすための“重複抑制”
#  - 分割線の長さ（縦分割=高さ / 横分割=幅）や、タイル面積が
#    「同じくらいの値に偏らない」ように、分割候補を複数試して
#    “重複が少ない候補”を選びます。
#  - anneal（最適化）ではなく、分割“生成”の時点で効かせるため安定します。
QUILT_ANTIREPEAT_ENABLE = True
QUILT_ANTIREPEAT_TRIES = 24
QUILT_ANTIREPEAT_LEN_BIN = 8
QUILT_ANTIREPEAT_AREA_LOG_BIN = 0.18
QUILT_ANTIREPEAT_W_LEN = 1.0
QUILT_ANTIREPEAT_W_AREA = 0.6
QUILT_ANTIREPEAT_P_LEN = 1.2
QUILT_ANTIREPEAT_P_AREA = 1.1

# Quilt 分割方向の制御
#   - "auto"         : 既定（縦横の選択はサイズ/ランダムで決定）
#   - "alternate_vh" : 分割方向を 縦→横→縦→… と交互に試す（無理なら自動でフォールバック）
#   - "alternate_hv" : 分割方向を 横→縦→横→… と交互に試す（無理なら自動でフォールバック）
QUILT_SPLIT_ORIENT_MODE = "auto"

# Quilt 近傍最適化（任意）
#   - QUILT_ENHANCE_ENABLE      : quilt の“配置順（place）”と最適化を有効にする
#   - QUILT_ENHANCE_PROFILE     : "diagonal" / "hilbert" / "scatter" / "as_is"
#   - QUILT_DIAGONAL_DIRECTION  : 対角方向（"tl_br" など）
#   - QUILT_NEIGHBOR_OBJECTIVE  : "min"(滑らか) / "max"(ばらけ)
#   - QUILT_OPTIMIZER           : "none" / "anneal"
#   - QUILT_ANNEAL_ENABLE       : True のとき anneal を実行（Tune用）
#   - QUILT_ANNEAL_STEPS/T0/TEND/REHEATS : annealパラメータ
QUILT_ENHANCE_ENABLE     = False
QUILT_ENHANCE_PROFILE    = "hilbert"
QUILT_DIAGONAL_DIRECTION = "tl_br"
QUILT_NEIGHBOR_OBJECTIVE = "min"
QUILT_OPTIMIZER          = "none"
QUILT_ANNEAL_ENABLE      = False
QUILT_ANNEAL_STEPS       = 20000
QUILT_ANNEAL_T0          = 1.0
QUILT_ANNEAL_TEND        = 1e-3
QUILT_ANNEAL_REHEATS     = 4

# Face-focus が失敗したとき、人物（Person）を探すか。
# 既定では「上半身カスケード」を使います。
try:
    FACE_FOCUS_USE_PERSON
except NameError:
    FACE_FOCUS_USE_PERSON = True

# Person-focus の追加手段: HOG 人物検出（写真向け）。遅い場合は False にしてください。
try:
    PERSON_FOCUS_HOG_ENABLE
except NameError:
    PERSON_FOCUS_HOG_ENABLE = True

try:
    PERSON_FOCUS_HOG_MAX_DIM
except NameError:
    PERSON_FOCUS_HOG_MAX_DIM = 640


# =============================================================================
# セクション: Stained Glass（Voronoi / ステンドグラス）
# =============================================================================
# Voronoi 分割でできた多角形パネルに、画像を「fill（cover）→多角形マスク」で貼り込みます。
#
# - STAINED_GLASS_ORDER        : パネルの並び順（"hilbert" / "scan" / "diag" / "random" / "spiral"）
# - STAINED_GLASS_HILBERT_BITS : "hilbert" の量子化ビット（7=128x128）。大きいほど細かい順序づけ
# - STAINED_GLASS_POINT_JITTER : 種点のゆらぎ（0.0〜1.0）。大きいほど不規則で自然な割れ方
# - STAINED_GLASS_LEAD_WIDTH   : 境界（鉛線）の太さ（px）
# - STAINED_GLASS_LEAD_COLOR   : 境界（鉛線）の色（"#RRGGBB"）
# - STAINED_GLASS_LEAD_ALPHA   : 境界（鉛線）の不透明度（0.0〜1.0）
# - STAINED_GLASS_FACE_FOCUS_ENABLE : True のとき、パネル貼り込みに face-focus（AI/ヒューリスティック）を使う
# - STAINED_GLASS_MAX_CORNER_ANGLE_DEG : パネル内角の最大角度（度）。小さくすると“ほぼ一直線”の角を削って形をスッキリさせる（例: 170）
# - STAINED_GLASS_EFFECTS_APPLY_MODE : エフェクト適用範囲（"global" / "mask" / "mask_feather"）※既定 global
# - STAINED_GLASS_EFFECTS_INCLUDE_LEAD : True のとき境界線も含めてマスク系でエフェクト対象へ（global では無効）
#
try:
    STAINED_GLASS_ORDER
except NameError:
    STAINED_GLASS_ORDER = "hilbert"

try:
    STAINED_GLASS_HILBERT_BITS
except NameError:
    STAINED_GLASS_HILBERT_BITS = 7

try:
    STAINED_GLASS_POINT_JITTER
except NameError:
    STAINED_GLASS_POINT_JITTER = 0.55

try:
    STAINED_GLASS_GLOBAL_MESH_SIMPLIFY
except NameError:
    # ステンドグラス: メッシュ全体で角数制約を満たす（隙間を作りにくい実験的モード）
    STAINED_GLASS_GLOBAL_MESH_SIMPLIFY = True


try:
    STAINED_GLASS_GLOBAL_SIMPLIFY_MAX_COLLAPSES
except NameError:
    # グローバル簡略化: エッジコラプス上限（多いほど max_vertices に寄せやすいが遅くなる）
    # 既定は 800（従来のフォールバック値と同じ）
    STAINED_GLASS_GLOBAL_SIMPLIFY_MAX_COLLAPSES = 800


try:
    STAINED_GLASS_LEAD_WIDTH
except NameError:
    STAINED_GLASS_LEAD_WIDTH = 1

try:
    STAINED_GLASS_LEAD_COLOR
except NameError:
    STAINED_GLASS_LEAD_COLOR = "#101010"

try:
    STAINED_GLASS_LEAD_ALPHA
except NameError:
    STAINED_GLASS_LEAD_ALPHA = 0.2

try:
    STAINED_GLASS_LEAD_STYLE
except NameError:
    STAINED_GLASS_LEAD_STYLE = "outer"


# 見た目の境界を滑らかにする（マスク縁のアンチエイリアス）
# - 0.0: 無効（既定）
# - 0.6〜1.2: ほんのり滑らか（おすすめ帯）
try:
    STAINED_GLASS_MASK_FEATHER_PX
except NameError:
    STAINED_GLASS_MASK_FEATHER_PX = 0.0

# 多角形パネル境界のジャギー低減（パネルマスクを高解像度で描いて縮小）
# - 1: 無効（従来と同じ）
# - 2: 2倍で描いて縮小（おすすめ）
# - 3〜4: さらに滑らかだが重くなる
try:
    STAINED_GLASS_MASK_SUPERSAMPLE
except NameError:
    STAINED_GLASS_MASK_SUPERSAMPLE = 4

# lead（鉛線）を滑らかにする（αマスクの軽いぼかし）
# - 0.0: 無効（既定）
# - 0.6〜1.2: ほんのり滑らか（おすすめ帯）
try:
    STAINED_GLASS_LEAD_SMOOTH_PX
except NameError:
    STAINED_GLASS_LEAD_SMOOTH_PX = 0.0

# lead の生成方式:
# - "mask" : 従来方式（各パネルの縁マスクを合成）※境界が二重になり太く見えることがあります
# - "edges": 共有エッジを重複排除して描く（細い線が作りやすい / ジャギーが減る）
try:
    STAINED_GLASS_LEAD_METHOD
except NameError:
    STAINED_GLASS_LEAD_METHOD = "edges"

# edges 方式のときだけ使う：描画を高解像度で行ってから縮小して滑らかにする（1..2 推奨）
# - 1: 無効（そのまま描く）
# - 2: 2倍で描いて縮小（4Kでもメモリ安全な範囲）
try:
    STAINED_GLASS_LEAD_SUPERSAMPLE
except NameError:
    STAINED_GLASS_LEAD_SUPERSAMPLE = 2

# edges 方式の「同一エッジ判定」の量子化（大きいほど厳密 / 小さいほど同一視しやすい）
# - 4: 0.25px 単位（おすすめ）
try:
    STAINED_GLASS_LEAD_EDGE_QUANT
except NameError:
    STAINED_GLASS_LEAD_EDGE_QUANT = 4


# StainedGlass: 顔検出用の縮小最大辺（小さくすると速い／小さすぎると検出が落ちます）
try:
    STAINED_GLASS_FACE_DETECT_MAX_DIM
except NameError:
    STAINED_GLASS_FACE_DETECT_MAX_DIM = 512
try:
    STAINED_GLASS_FACE_FOCUS_ENABLE
except NameError:
    STAINED_GLASS_FACE_FOCUS_ENABLE = True

# StainedGlass: face-fit（多角形フレーム内に「目」が入るまで、顔画像を差し替える）
try:
    STAINED_GLASS_FACE_FIT_ENABLE
except NameError:
    # ※ STAINED_GLASS_FACE_FOCUS_ENABLE がONのときだけ有効
    STAINED_GLASS_FACE_FIT_ENABLE = True

# 顔画像を“太いパネル”に優先的に割り当てる（細いパネルでは顔を避ける）
try:
    STAINED_GLASS_FACE_PRIORITY_ENABLE
except NameError:
    STAINED_GLASS_FACE_PRIORITY_ENABLE = True

# face-fit の最大リトライ回数（画像差し替え）
try:
    STAINED_GLASS_FACE_FIT_MAX_TRIES
except NameError:
    STAINED_GLASS_FACE_FIT_MAX_TRIES = 6

# 「細いパネル判定」: bbox の短辺がこの値未満なら顔を避ける
try:
    STAINED_GLASS_FACE_FIT_MIN_SHORT_SIDE
except NameError:
    STAINED_GLASS_FACE_FIT_MIN_SHORT_SIDE = 96

# 「細いパネル判定」: 多角形の塗りつぶし面積 / bbox 面積 がこの値未満なら顔を避ける
try:
    STAINED_GLASS_FACE_FIT_MIN_FILL_RATIO
except NameError:
    STAINED_GLASS_FACE_FIT_MIN_FILL_RATIO = 0.45

# 「目がフレームに入っている」判定の安全マージン（px）
# None の場合は lead_w から自動推定（lead_w * 0.35 程度）
try:
    STAINED_GLASS_FACE_FIT_SAFE_MARGIN_PX
except NameError:
    STAINED_GLASS_FACE_FIT_SAFE_MARGIN_PX = None

# 目の高さ（顔bboxの上端からの割合）
try:
    STAINED_GLASS_FACE_FIT_EYE_Y_FRAC
except NameError:
    STAINED_GLASS_FACE_FIT_EYE_Y_FRAC = 0.30


# 目の高さを上下にずらした“代替チェック”の幅（顔bbox比）
# - 目の位置がズレやすい絵柄向け（例: 0.06〜0.10）
try:
    STAINED_GLASS_FACE_FIT_EYE_Y_ALT_DELTA
except NameError:
    STAINED_GLASS_FACE_FIT_EYE_Y_ALT_DELTA = 0.06

# 目チェックを厳格化（eye_y と eye_y±alt の両方が安全域に入ることを要求）
try:
    STAINED_GLASS_FACE_FIT_STRICT_EYE_BAND
except NameError:
    STAINED_GLASS_FACE_FIT_STRICT_EYE_BAND = False

# StainedGlass（顔寄せ）用の簡易パフォーマンス計測（必要なときだけ表示）
SG_PERF = {
    "yolo_sec": 0.0,
    "yolo_calls": 0,
}

# 細いパネルでの扱い
# - 'nofacefocus': thin では face_focus を切って貼る（目欠け事故を避ける）
# - 'reject': thin では顔画像を避ける（候補不足で遅くなる可能性あり）
try:
    STAINED_GLASS_FACE_FIT_THIN_MODE
except NameError:
    STAINED_GLASS_FACE_FIT_THIN_MODE = "nofacefocus"

# 目の“広がり”サンプル（顔bboxに対する割合）
# - 目が欠けやすい場合、X/Y を少し増やす（例: X=0.08, Y=0.05）と厳しくなります
try:
    STAINED_GLASS_FACE_FIT_EYE_SPREAD_X_FRAC
except NameError:
    STAINED_GLASS_FACE_FIT_EYE_SPREAD_X_FRAC = 0.06

try:
    STAINED_GLASS_FACE_FIT_EYE_SPREAD_Y_FRAC
except NameError:
    STAINED_GLASS_FACE_FIT_EYE_SPREAD_Y_FRAC = 0.04


# 目サンプルの安全係数（spread を一括で強めたいときに使う）
# - 1.0: そのまま / 1.3: 少し厳しめ / 1.6: かなり厳しめ
try:
    STAINED_GLASS_FACE_FIT_EYE_SPREAD_SCALE
except NameError:
    STAINED_GLASS_FACE_FIT_EYE_SPREAD_SCALE = 1.00

# 目のサンプル点がマスク内に入っている割合（1.0=全点必須 / 0.8=8割以上）
# 注: 旧名互換（探しやすさ用）
#  - STAINED_GLASS_FACE_FIT_OK_RATIO を編集しても効くようにする
try:
    STAINED_GLASS_FACE_FIT_OK_RATIO
except NameError:
    STAINED_GLASS_FACE_FIT_OK_RATIO = 1.0

try:
    STAINED_GLASS_FACE_FIT_POINT_OK_RATIO
except NameError:
    STAINED_GLASS_FACE_FIT_POINT_OK_RATIO = STAINED_GLASS_FACE_FIT_OK_RATIO


try:
    STAINED_GLASS_EFFECTS_APPLY_MODE
except NameError:
    # "global": 画像全体にエフェクト（境界で段差が出にくい）
    # "mask":   パネル領域のみ（境界が強い場合あり）
    # "mask_feather": マスクをぼかしてなじませる
    STAINED_GLASS_EFFECTS_APPLY_MODE = "global"

try:
    STAINED_GLASS_EFFECTS_INCLUDE_LEAD
except NameError:
    # apply_mode が mask 系のときだけ有効（global では無効）
    STAINED_GLASS_EFFECTS_INCLUDE_LEAD = True

try:
    STAINED_GLASS_MAX_CORNER_ANGLE_DEG
except NameError:
    STAINED_GLASS_MAX_CORNER_ANGLE_DEG = 179.5

# =============================================================================
# セクション: Face-focus（検出モデル / 厳密チェック / 実行時ゲート）
# =============================================================================
# --- デフォルト値と各種調整パラメータ ---
try: FACE_FOCUS_ZOOM_MIN
except NameError: FACE_FOCUS_ZOOM_MIN = 0.5  # 顔フォーカスズームの下限倍率（小さいほど引き気味になります）
try: FACE_FOCUS_ZOOM_MAX
except NameError: FACE_FOCUS_ZOOM_MAX = 1.0  # 顔フォーカスズームの上限倍率（大きいほど寄り気味になります）
try: FACE_FOCUS_MIN_EYE_DIST_FRAC
except NameError: FACE_FOCUS_MIN_EYE_DIST_FRAC = 0.18  # 目距離が小さいときのズーム過多を抑制（大きいほど抑えめ）
try: FACE_FOCUS_ALLOW_LOW
except NameError: FACE_FOCUS_ALLOW_LOW = False  # 顔検出精度が低くてもフォーカス処理を行うか
try: FACE_FOCUS_DEBUG
except NameError: FACE_FOCUS_DEBUG = True  # 顔検出・目検出時のデバッグ情報を表示するか
try: FACE_FOCUS_DEBUG_DETAIL
except NameError: FACE_FOCUS_DEBUG_DETAIL = False  # 候補(顔/アニメ顔)の詳細ログを出す（大量出力防止のため既定はFalse）
try: FACE_FOCUS_DEBUG_DETAIL_LIMIT
except NameError: FACE_FOCUS_DEBUG_DETAIL_LIMIT = 3  # 詳細ログを出す画像数の上限（0以下なら無制限）
try: FACE_FOCUS_DEBUG_DETAIL_MAX_CANDIDATES
except NameError: FACE_FOCUS_DEBUG_DETAIL_MAX_CANDIDATES = 8  # 1画像あたり表示する候補数の上限
try: FACE_FOCUS_ANIME_FACE_PREFER
except NameError: FACE_FOCUS_ANIME_FACE_PREFER = True  # Haarよりアニメ顔(eye-pair)推定を優先する
try: FACE_FOCUS_ANIME_FACE_ALWAYS
except NameError: FACE_FOCUS_ANIME_FACE_ALWAYS = False  # Trueなら毎回アニメ顔推定も走らせて比較する（重い場合はFalse推奨）
try: FACE_FOCUS_HAAR_SUSPICIOUS_Y
except NameError: FACE_FOCUS_HAAR_SUSPICIOUS_Y = 0.62
try: FACE_FOCUS_VALIDATOR_ENABLE
except NameError: FACE_FOCUS_VALIDATOR_ENABLE = True  # 候補に検証(validator)をかける
try: FACE_FOCUS_VALIDATOR_MAX_Y
except NameError: FACE_FOCUS_VALIDATOR_MAX_Y = 0.70  # center_y 正規化の上限（超えたら棄却）
try: FACE_FOCUS_VALIDATOR_EYE_MIN
except NameError: FACE_FOCUS_VALIDATOR_EYE_MIN = 1   # 目の最小数（未満は棄却）
  # Haar候補のcenter_yがこれより下なら「怪しい」とみなしてアニメ顔比較を行う
try: FACE_FOCUS_USE_PROFILE
except NameError: FACE_FOCUS_USE_PROFILE = True  # 横顔用の顔検出器を使用するか
try: FACE_FOCUS_USE_UPPER
except NameError: FACE_FOCUS_USE_UPPER = True  # 上半身検出器を使用するか
try: FACE_FOCUS_USE_SALIENCY
except NameError: FACE_FOCUS_USE_SALIENCY = True  # 顔/人物検出が失敗したときに視覚的顕著度（saliency）へフォールバックするか

# 顔検出の最小サイズ（短辺に対する割合）。AI/アニメ絵では顔が画面に対して小さめでも
# しっかり描かれていることが多いので、既定は 0.05（短辺の5%）にします。
try:
    FACE_FOCUS_MIN_FACE_FRAC
except NameError:
    FACE_FOCUS_MIN_FACE_FRAC = 0.05
try: FACE_FOCUS_STRICT_EYES
except NameError: FACE_FOCUS_STRICT_EYES = True  # 目検出が成功した場合のみ厳密にズームを適用するか
try: FACE_FOCUS_EYE_MIN
except NameError: FACE_FOCUS_EYE_MIN = 2  # 有効とみなす目検出の最小数

# アニメ/AI絵向け: 目（アイ）検出ヒューリスティック
# - Haar の eye カスケードが外しやすい「アニメ目」を補助的に拾います。
# - 目が検出できた場合は strict_eyes の判定にも利用します。
try: FACE_FOCUS_ANIME_EYES_ENABLE
except NameError: FACE_FOCUS_ANIME_EYES_ENABLE = True

# 目ペア推定の強化: Haar eye カスケードも使って「目らしい2点」を拾います。
try:
    FACE_FOCUS_ANIME_EYE_HAAR_ENABLE
except NameError:
    FACE_FOCUS_ANIME_EYE_HAAR_ENABLE = True
try:
    FACE_FOCUS_ANIME_EYE_HAAR_NEIGHBORS
except NameError:
    FACE_FOCUS_ANIME_EYE_HAAR_NEIGHBORS = 2

# アニメ/AI絵向け: 顔検出（eye-pair から擬似 face bbox を推定）
# - Haar の顔検出が外れた場合のみ試します（誤爆を避けるため）。
try: FACE_FOCUS_ANIME_FACE_ENABLE
except NameError: FACE_FOCUS_ANIME_FACE_ENABLE = True

# 目候補抽出のパラメータ（軽量なエッジベース）
FACE_FOCUS_ANIME_EYE_MIN_AREA_FRAC = 0.0015
try: FACE_FOCUS_ANIME_EYE_MIN_FRAC
except NameError: FACE_FOCUS_ANIME_EYE_MIN_FRAC = 0.06  # ROI短辺に対する最小サイズ（既定: 8%）
try: FACE_FOCUS_ANIME_EYE_MAX_W_FRAC
except NameError: FACE_FOCUS_ANIME_EYE_MAX_W_FRAC = 0.60  # ROI幅に対する最大幅（既定: 60%）
try: FACE_FOCUS_ANIME_EYE_MAX_H_FRAC
except NameError: FACE_FOCUS_ANIME_EYE_MAX_H_FRAC = 0.45  # ROI高さに対する最大高さ（既定: 45%）
try: FACE_FOCUS_ANIME_EYE_PAIR_MAX_DY_FRAC
except NameError: FACE_FOCUS_ANIME_EYE_PAIR_MAX_DY_FRAC = 0.35  # 2つの目の縦ズレ許容（既定: max(h)*0.35）

try: FACE_FOCUS_ANIME_FACE_MAX_DIM
except NameError: FACE_FOCUS_ANIME_FACE_MAX_DIM = 720  #

# =============================================================================
# セクション: AIモデル保存ベース（GitHub運用向け）
# - YOLO/YuNet/AnimeFace 等のモデル（.pt/.onnx/.xml）を置く場所です。
#
# ✅ 基本：ここだけ触ればOK
#   - None: STATE_DIR（=このスクリプトの隣/_kana_state）の中に "models" を作って参照します（おすすめ）
#   - 例: r"D:\kana_state\models"  または  r".\_kana_state\models"
#     ※相対はこのスクリプト基準
# =============================================================================
MODEL_DIR: Optional[str] = None
MODEL_SUBDIR_NAME: str = "models"

def _core_models_dir() -> Path:
    """AIモデル用の基準ディレクトリを返します（環境変数は使いません）。"""
    try:
        if MODEL_DIR:
            p = Path(MODEL_DIR).expanduser()
            if not p.is_absolute():
                p = (Path(__file__).resolve().parent / p).resolve()
        else:
            # 既定：STATE_DIR 配下（= _kana_state/models）
            p = _core_state_dir() / MODEL_SUBDIR_NAME
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        # 最後の砦：カレントへ（失敗しても致命にしない）
        try:
            p = Path.cwd() / MODEL_SUBDIR_NAME
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            return Path.cwd()

# -----------------------------------------------------------------------------

# AI顔検出（YuNet: OpenCV Zoo）
# -----------------------------------------------------------------------------
# アニメ絵で Haar/eye-pair が外れやすい場合の「最後の切り札」。
# 使うには onnx モデル（YuNet）を用意してください。
# 例: face_detection_yunet_2023mar.onnx（_kana_state/models に置けば既定で見つかります）
try: FACE_FOCUS_AI_ENABLE
except NameError: FACE_FOCUS_AI_ENABLE = False  # TrueでAI顔検出を有効化（既定OFF）  # デフォルトOFF（必要なら外部設定/ランチャーでON）
try: FACE_FOCUS_AI_ALWAYS
except NameError: FACE_FOCUS_AI_ALWAYS = False  # デフォルトOFF（常時AIは重いので明示的にON）
try: FACE_FOCUS_YUNET_MODEL
except NameError: FACE_FOCUS_YUNET_MODEL = str(_core_models_dir() / "face_detection_yunet_2023mar.onnx")
try: FACE_FOCUS_YUNET_SCORE
except NameError: FACE_FOCUS_YUNET_SCORE = 0.60  # 検出しきい値（高いほど厳しい）
try: FACE_FOCUS_YUNET_NMS
except NameError: FACE_FOCUS_YUNET_NMS = 0.30    # NMS しきい値（低いほど重複除去強め）
try: FACE_FOCUS_YUNET_TOPK
except NameError: FACE_FOCUS_YUNET_TOPK = 50

try: FACE_FOCUS_AI_BACKEND
except NameError: FACE_FOCUS_AI_BACKEND = "yolov8_animeface"  # "yunet"(既定) / "yolov8_animeface"

# YOLOv8（アニメ顔）: ultralytics を使う
# - 事前に: pip install ultralytics
# - 重み例: yolov8x6_animeface.pt（アニメ顔専用モデル）
try: FACE_FOCUS_YOLO_MODEL
except NameError: FACE_FOCUS_YOLO_MODEL = str(_core_models_dir() / "yolov8x6_animeface.pt")

# 代替YOLOモデル（任意）: YOLOv5アニメ顔モデルなどを試したいときに使う
# - 現在の実装では FACE_FOCUS_YOLO_MODEL を参照します。
#   手動で切り替える場合に備えて、パスだけ既定で用意します。
try: FACE_FOCUS_YOLO_MODEL_ALT
except NameError: FACE_FOCUS_YOLO_MODEL_ALT = str(_core_models_dir() / "yolov5x_anime.pt")

try: FACE_FOCUS_YOLO_CONF
except NameError: FACE_FOCUS_YOLO_CONF = 0.25  # しきい値（低いほど拾う）
try: FACE_FOCUS_YOLO_IOU
except NameError: FACE_FOCUS_YOLO_IOU = 0.50
try: FACE_FOCUS_YOLO_IMGSZ
except NameError: FACE_FOCUS_YOLO_IMGSZ = 1536
try: FACE_FOCUS_YOLO_DEVICE
except NameError: FACE_FOCUS_YOLO_DEVICE = ""  # ""=自動 / "0"=GPU0 / "cpu" 等
try: FACE_FOCUS_YOLO_MAXDET
except NameError: FACE_FOCUS_YOLO_MAXDET = 50     # 最大検出数

# AnimeFace（CPUカスケード）: nagadomi/lbpcascade_animeface のXMLを利用
# - 事前に lbpcascade_animeface.xml を用意して、_kana_state/models に置いてください。
# - YOLOを使わない（GPUが弱い/未搭載）環境向けの軽量フォールバックです。
try: FACE_FOCUS_ANIMEFACE_CASCADE
except NameError: FACE_FOCUS_ANIMEFACE_CASCADE = str(_core_models_dir() / "lbpcascade_animeface.xml")
try: FACE_FOCUS_ANIMEFACE_SCALE_FACTOR
except NameError: FACE_FOCUS_ANIMEFACE_SCALE_FACTOR = 1.10
try: FACE_FOCUS_ANIMEFACE_MIN_NEIGHBORS
except NameError: FACE_FOCUS_ANIMEFACE_MIN_NEIGHBORS = 3
try: FACE_FOCUS_ANIMEFACE_MIN_SIZE
except NameError: FACE_FOCUS_ANIMEFACE_MIN_SIZE = 24


try: FACE_FOCUS_TOP_FRAC
except NameError: FACE_FOCUS_TOP_FRAC = 0.70  # 顔や目検出領域の上部をどれだけ無視するか（割合）
# 顔の縦横比(幅/高さ)の許容レンジ
try: FACE_FOCUS_FACE_RATIO_MIN
except NameError: FACE_FOCUS_FACE_RATIO_MIN = 0.5
try: FACE_FOCUS_FACE_RATIO_MAX
except NameError: FACE_FOCUS_FACE_RATIO_MAX = 2.0


# =============================================================================
# セクション: コンソール表示・ログ出力ユーティリティ
# =============================================================================

# -----------------------------------------------------------------------------
# サブセクション: コンソールユーティリティ
# -----------------------------------------------------------------------------
ANSI_OK = False
UI = {"style":"unicode","emoji":False,"ansi":False}

def _enable_ansi():
    global ANSI_OK
    ANSI_OK = False
    if os.name != "nt":
        ANSI_OK = True; return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        kernel32.GetConsoleMode(h, ctypes.byref(mode))
        kernel32.SetConsoleMode(h, mode.value | 0x0004)
        ANSI_OK = True
    except Exception:
        ANSI_OK = False

def _maybe_force_utf8():
    if os.name=="nt" and FORCE_UTF8_CP:
        try: os.system("chcp 65001 >nul")
        except Exception: pass

def _wcwidth_char(ch: str) -> int:
    import unicodedata as ud
    # 結合文字（濁点など）は直前の文字に合成されるため表示幅 0
    try:
        if ud.combining(ch):
            return 0
    except Exception as e:
        _warn_exc_once(e)
        pass
    ew = ud.east_asian_width(ch)
    if ew in ("W", "F"):
        return 2
    # フラグ未定義でも True を既定に（日本語環境向け）
    if globals().get("TREAT_AMBIGUOUS_WIDE", True) and ew == "A":
        return 2
    return 1

def disp_width(s: str) -> int:
    w=0; i=0
    while i<len(s):
        if s[i]=="\x1b":
            j=i+1
            while j<len(s) and s[j]!="m": j+=1
            i=j+1; continue
        w+=_wcwidth_char(s[i]); i+=1
    return w

def pad_to_width(s: str, w: int, align: str = "left") -> str:
    """表示幅（全角=2など）を基準に、文字列をちょうど w 列に切り詰め／パディングします。

    - ANSI エスケープシーケンス（\x1b[...m）は幅 0 として扱います。
    - 結合文字（濁点など）は幅 0 として扱います。

    NOTE: `str.ljust()` は全角を 1 文字扱いするため、日本語を含むと枠線（|／│）がずれます。
          その対策としてこの関数を使います。
    """
    try:
        w = int(w)
    except Exception:
        w = 0
    if w <= 0:
        return ""

    out_parts = []
    vis = 0
    i = 0
    saw_ansi = False

    while i < len(s):
        # ANSI シーケンスは幅 0
        if s[i] == "\x1b":
            j = i + 1
            while j < len(s) and s[j] != "m":
                j += 1
            j = min(len(s), j + 1)
            out_parts.append(s[i:j])
            saw_ansi = True
            i = j
            continue

        ch = s[i]
        cw = _wcwidth_char(ch)
        if vis + cw > w:
            break
        out_parts.append(ch)
        vis += cw
        i += 1

    out = "".join(out_parts)

    # 色付き出力で途中で切った場合、リセットを付けておく（端末の色が漏れないように）
    if saw_ansi and "\x1b" in out and not out.endswith("\x1b[0m"):
        out += "\x1b[0m"

    pad = w - vis
    if pad <= 0:
        return out

    a = str(align).lower()
    if a in ("right", "r"):
        return " " * pad + out
    if a in ("center", "centre", "c"):
        left = pad // 2
        right = pad - left
        return " " * left + out + " " * right
    return out + " " * pad

BORDER={"h":"─","v":"│","tl":"┌","tr":"┐","bl":"└","br":"┘"}

def C(code: str, s: str):
    return f"\033[{code}m{s}\033[0m" if UI.get("ansi") else s


# -----------------------------------------------------------------------------
# サブセクション: 実行前チェック
# -----------------------------------------------------------------------------
def _logger_get():
    """モジュール用loggerを返します（常に安全）。"""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    lg = logging.getLogger('kana_wallpaper')
    lg.propagate = False
    # 既定は無音（print は従来どおり）
    if not lg.handlers:
        lg.addHandler(logging.NullHandler())
    _LOGGER = lg
    return lg

def _logger_setup_once():
    """LOG_ENABLE=True のときだけファイルログを設定します。"""
    global _LOGGER_READY
    if _LOGGER_READY:
        return
    _LOGGER_READY = True
    try:
        if not bool(globals().get('LOG_ENABLE', False)):
            return
    except Exception:
        return
    try:
        lg = _logger_get()
        # 既存ハンドラをクリア（重複防止）
        for h in list(lg.handlers):
            lg.removeHandler(h)
        level_name = str(globals().get('LOG_LEVEL', 'INFO') or 'INFO').upper()
        level = getattr(logging, level_name, logging.INFO)
        lg.setLevel(level)

        path = str(globals().get('LOG_FILE', '') or '').strip()
        if not path:
            base = str(globals().get('LOG_SAVE_DIR', '') or '').strip()
            if base:
                path = str(Path(base) / 'kana_wallpaper.log')
            else:
                path = os.path.join(tempfile.gettempdir(), 'kana_wallpaper.log')
        max_bytes = int(globals().get('LOG_MAX_BYTES', 2_000_000) or 2_000_000)
        backups = int(globals().get('LOG_BACKUP_COUNT', 3) or 3)
        try:
            fh = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding='utf-8')
        except Exception:
            fh = logging.FileHandler(path, encoding='utf-8')
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh.setFormatter(fmt)
        lg.addHandler(fh)
        globals()['_LOG_PATH_ACTIVE'] = path
        # ログ先だけ一度案内（LOG_ENABLE の時のみ）
        if bool(globals().get('VERBOSE', False)):
            try:
                note(f'log(file): {path}')
            except Exception as e:
                _kana_silent_exc('core:L2693', e)
                pass
    except Exception:
        # ログ設定失敗は本処理に影響させない
        return

def _log_info(msg: str):
    try:
        if not bool(globals().get('LOG_ENABLE', False)):
            return
        _logger_setup_once()
        _logger_get().info(str(msg))
    except Exception:
        return

def _log_warn(msg: str):
    try:
        if not bool(globals().get('LOG_ENABLE', False)):
            return
        _logger_setup_once()
        _logger_get().warning(str(msg))
    except Exception:
        return

def banner(title: str):
    if not VERBOSE:
        return
    _ensure_newline_if_bar_active()
    # i18n 済みタイトル
    try:
        t = _tr(title)
    except Exception:
        t = title

    _log_info(f'[BANNER] {t}')

    # セクション別パレットを決定し、グローバルに記録
    pal = _palette_for_title(t)
    globals()["CURRENT_PALETTE"] = pal

    # 枠・内側テキストをグラデ化（Unicode のときのみ）
    inner_raw = f" {t} "
    inner_w   = _disp_width_safe(inner_raw)

    if globals().get("UI_STYLE","ascii") == "unicode" and globals().get("UNICODE_BLING", False):
        # 上枠
        top_grad = _grad_list(inner_w, pal)
        top_line = "".join(_ansi_rgb(r,g,b,True)+"─" for (r,g,b) in top_grad) + "\x1b[0m"
        print(_ansi_rgb(*top_grad[0],True)+"┌" + top_line + _ansi_rgb(*top_grad[-1],True)+"┐" + "\x1b[0m")

        # 中身（左右の縦罫線は端の色、内側文字はグラデ）
        left_col  = _ansi_rgb(*top_grad[0], True)  + "│" + "\x1b[0m"
        right_col = _ansi_rgb(*top_grad[-1], True) + "│" + "\x1b[0m"
        inner_col = rainbow_text(inner_raw, bold=True, palette=pal)
        print(left_col + _pad_to_width_safe(inner_col, inner_w) + right_col)

        # 下枠（上と同じ配色で）
        bot_grad = top_grad
        bot_line = "".join(_ansi_rgb(r,g,b,True)+"─" for (r,g,b) in bot_grad) + "\x1b[0m"
        print(_ansi_rgb(*bot_grad[0],True)+"└" + bot_line + _ansi_rgb(*bot_grad[-1],True)+"┘" + "\x1b[0m")
    else:
        # ASCII フォールバック
        line = "-" * inner_w
        print("+" + line + "+")
        print("|" + _pad_to_width_safe(inner_raw, inner_w) + "|")
        print("+" + line + "+")


# --- KANA（face-cascade 自動ローダー） ---
def note(msg: str):
    _ensure_newline_if_bar_active()
    # 行ごとに翻訳して出力
    for ln in textwrap.dedent(str(msg)).splitlines():
        s = _tr(ln)
        # note() は先頭に箇条書き記号を付与する。呼び出し側が '•' を含めても二重にならないよう吸収する。
        try:
            ss = str(s).lstrip()
            if ss.startswith('• '):
                s = ss[2:].lstrip()
            elif ss.startswith('•'):
                s = ss[1:].lstrip()

            # 表記ゆれ吸収（UI統一）
            _t = str(s)
            _t = re.sub(r'\bquilt\b\s*[\(\（]\s*bsp\s*[\)\）]', 'quilt', _t, flags=re.I)
            _t = re.sub(r'\bQuilt\b', 'quilt', _t)
            s = _t
        except Exception as e:
            _kana_silent_exc('core:note', e)
            pass
        _log_info('• ' + str(s))
        print('  • ' + C('97', str(s)))


# 進捗バーは '\r' で同じ行を書き換えるため、最後に改行しないと次の枠表示が崩れます。
# 現在バー更新中かを記録し、枠/ログ出力前に改行して整列させます。
_BAR_ACTIVE = False

# AI 状態表示は、進捗バー更新中に print するとバーが「2行」になって見た目が崩れます。
# そのため、バー更新中は一旦保留し、バーの最終行（final=True）でまとめて表示します。
_AI_STATUS_PENDING = ""

def _flush_ai_status_pending():
    global _AI_STATUS_PENDING
    try:
        msg = str(_AI_STATUS_PENDING or "")
        if msg:
            _AI_STATUS_PENDING = ""
            # note() は自動で "  • " を付ける
            note(msg)
    except Exception:
        _AI_STATUS_PENDING = ""
        pass

def _ensure_newline_if_bar_active():
    global _BAR_ACTIVE
    if _BAR_ACTIVE:
        print()
        _BAR_ACTIVE = False


def _bar_prefix_norm(prefix: str) -> str:
    """進捗バーの prefix を固定幅の統一トークンへ正規化します（表示専用）。"""
    p = (prefix or "").strip().lower().replace("_", "-")
    # よくある別名を吸収し、表示トークンは6文字以内に丸めます（整列用）。
    mp = {
        "scan": "scan",
        "video": "video",
        "feat": "feat",
        "feature": "feat",
        "features": "feat",
        "aesthetic": "aesft",  # aesthetic feature compute
        "rank": "rank",
        "dedup": "dedup",
        "pca": "pca",
        "project": "proj",
        "proj": "proj",
        "prefetch": "prefet",
        "prefet": "prefet",
        "draw": "draw",
        "assign": "assign",
        "anneal": "anneal",
        "hill": "hill",
        "opt-col": "colopt",
        "opt-row": "rowopt",
        "colopt": "colopt",
        "rowopt": "rowopt",
        "select": "select",
    }
    tok = mp.get(p, p)
    tok = (tok or "")[:6]
    return tok

def bar(done: int, total: int, prefix: str="", final: bool=False):

    global _BAR_ACTIVE

    # --- prefix 正規化（表示専用） ---
    # prefixを固定幅で揃え、進捗バーが縦に揃って見やすくします。
    raw_p = (prefix or "").strip().lower()

    # 特例：過去の経路で 'select' が2段階で使われることがあります。
    # 互換維持のため、表示上は 'rank' → 'dedup' として区別します。
    try:
        if raw_p == "aesthetic":
            globals()["_BAR_SELECT_STAGE"] = 0
        if raw_p == "select":
            st = int(globals().get("_BAR_SELECT_STAGE", 0) or 0)
            prefix = "rank" if st <= 0 else "dedup"
            globals()["_BAR_SELECT_STAGE"] = st + 1
    except Exception as e:
        _kana_silent_exc('core:L2856', e)
        pass
    # 表示と間引き（throttling）キー用に正規化します。
    prefix_disp = _bar_prefix_norm(prefix)


    # 進捗更新方式の分岐（秒間隔 or ステップ間隔）
    mode = str(globals().get('PROGRESS_UPDATE_MODE', 'secs')).lower()
    if mode == 'every':
        ev = int(globals().get('PROGRESS_EVERY', 1) or 1)
        global _BAR_LAST_STEP
        try:
            _BAR_LAST_STEP
        except NameError:
            _BAR_LAST_STEP = {}
        key = prefix_disp or 'default'
        last = int(_BAR_LAST_STEP.get(key, -ev))
        if not final and (done < last + ev) and (done % ev != 0):
            return
        _BAR_LAST_STEP[key] = done
    else:
        global _BAR_LAST_TS
        try:
            _BAR_LAST_TS
        except NameError:
            _BAR_LAST_TS = {}
        import time as _t
        now = _t.monotonic()
        key = prefix_disp or 'default'
        last = float(_BAR_LAST_TS.get(key, 0.0))
        interval = float(globals().get('PROGRESS_UPDATE_SECS', 0.25))  # 進捗表示の更新間隔（秒）
        if not final and (now - last) < max(0.01, interval):
            return
        _BAR_LAST_TS[key] = now

    if not VERBOSE:
        return
    # total が未知（<=0）の場合は、カウンタ表示のみ（割合は出さない）
    try:
        _t = int(total)
    except Exception:
        _t = 0
    if _t <= 0:
        left = f"{prefix_disp}".ljust(7)
        # 行末のゴミが残らないように適度にパディング
        try:
            di = int(done)
        except Exception:
            di = 0
        s = (f"{left}{di}").ljust(100)
        end = "\n" if final else "\r"
        # 重複最終行の抑止（final が二重に呼ばれてもログが二重にならないようにする）
        try:
            if final and bool(globals().get("_BAR_LAST_FINAL", False)) and str(globals().get("_BAR_LAST_LINE", "")) == s:
                _BAR_ACTIVE = (not final)
                return
        except Exception as e:
            _kana_silent_exc('core:L2913', e)
            pass
        globals()["_BAR_LAST_LINE"] = s
        globals()["_BAR_LAST_FINAL"] = bool(final)
        # 重複行の抑止（同じ行を連続で出さない。unicode の体感速度改善にも効く）
        try:
            _hist = globals().setdefault("_BAR_PRINT_HIST", {})
            _hk = str(prefix_disp)
            _prev = _hist.get(_hk)
            if _prev and _prev.get("s") == s and bool(_prev.get("final")) == bool(final):
                _BAR_ACTIVE = (not final)
                return
            _hist[_hk] = {"s": s, "final": bool(final)}
        except Exception as e:
            _kana_silent_exc('core:L2926', e)
            pass
        print(s, end=end, flush=True)
        # 最終行が '' のまま残ると表示が崩れるため、状態を記録
        _BAR_ACTIVE = (not final)
        if final:
            _flush_ai_status_pending()
        return

    total = max(1, _t)
    try:
        done_i = int(done)
    except Exception:
        done_i = 0
    done = max(0, min(done_i, total))

    ratio = done / total
    width = max(10, int(PROGRESS_WIDTH))
    filled = int(width * ratio + 1e-9)
    empty  = width - filled
    pct    = f"{int(ratio*100):>3d}%"

    pal = globals().get("CURRENT_PALETTE") or BANNER_PALETTES["default"]
    bar_core = neon_bar(filled, empty, palette=pal)

    left  = f"{prefix_disp:<6} "
    vid_suffix = ""
    try:
        if str(prefix_disp) == "video":
            _vn = int(globals().get("_VIDEO_EXTRACT_VN", 0) or 0)
            _vi = int(globals().get("_VIDEO_EXTRACT_VIDX", 0) or 0)
            if _vn > 0 and _vi > 0:
                vid_suffix = f" v{_vi}/{_vn}"
    except Exception as e:
        _kana_silent_exc('core:L2959', e)
        pass
    right = f" {done}/{total} ({pct}){vid_suffix}"

    if globals().get("UI_STYLE","ascii") == "unicode" and globals().get("UNICODE_BLING", False):
        s = C("97", left) + bar_core + C("97", right)
    else:
        # bar core cache (reduces string allocations when progress updates frequently)
        try:
            _cache = globals().setdefault("_BAR_CORE_CACHE", {})
            _k = (int(PROGRESS_WIDTH), filled, BAR_FILL_CHAR, BAR_EMPTY_CHAR)
            _bar_core_cached = _cache.get(_k)
            if _bar_core_cached is None:
                _bar_core_cached = f"[{BAR_FILL_CHAR*filled}{BAR_EMPTY_CHAR*empty}]"
                _cache[_k] = _bar_core_cached
            _bar_core = _bar_core_cached
        except Exception:
            _bar_core = f"[{BAR_FILL_CHAR*filled}{BAR_EMPTY_CHAR*empty}]"
        s = f"{left}{_bar_core}{right}"
    end = "\n" if final else "\r"
    # 重複最終行の抑止（final が二重に呼ばれてもログが二重にならないようにする）
    try:
        if final and bool(globals().get("_BAR_LAST_FINAL", False)) and str(globals().get("_BAR_LAST_LINE", "")) == s:
            _BAR_ACTIVE = (not final)
            return
    except Exception as e:
        _kana_silent_exc('core:L2984', e)
        pass
    globals()["_BAR_LAST_LINE"] = s
    globals()["_BAR_LAST_FINAL"] = bool(final)
    # 重複行の抑止（同じ行を連続で出さない。unicode の体感速度改善にも効く）
    try:
        _hist = globals().setdefault("_BAR_PRINT_HIST", {})
        _hk = str(prefix_disp)
        _prev = _hist.get(_hk)
        if _prev and _prev.get("s") == s and bool(_prev.get("final")) == bool(final):
            _BAR_ACTIVE = (not final)
            return
        _hist[_hk] = {"s": s, "final": bool(final)}
    except Exception as e:
        _kana_silent_exc('core:L2997', e)
        pass
    print(s, end=end, flush=True)
    # 最終行が '' のまま残ると表示が崩れるため、状態を記録
    _BAR_ACTIVE = (not final)

# --- モードタグ用ヘルパー：一部設定が欠けていても落ちない安全設計 ---

# --- ログ補助（可読性向上；挙動は変えない） ---

def _note_config_summary(seed_used: int):
    """ログの可読性向上のため、設定サマリを“やや詳細”に出力します。

    - 1行目: 出力／レイアウト／抽出の要点（※fmt/bg/gutter/margin は表示しない）
    - 2行目: 最適化／スキャン／キャッシュ等
    - 3行目: エフェクト（ONの項目だけ）

    ※長くても省略せず、そのまま表示します（コンソール側で折り返されます）。
    """
    # --- 出力（キャンバス） ---
    try:
        out_wh = f"{int(globals().get('WIDTH', 0))}x{int(globals().get('HEIGHT', 0))}"
    except Exception:
        out_wh = "?x?"

    # --- レイアウト／抽出 ---
    try:
        layout = str(globals().get('LAYOUT_STYLE', 'grid') or 'grid').strip().lower()
    except Exception:
        layout = 'grid'
    try:
        sel = str(globals().get('SELECT_MODE', 'random') or 'random').strip().lower()
    except Exception:
        sel = 'random'
    try:
        full = bool(globals().get('ARRANGE_FULL_SHUFFLE', False))
    except Exception:
        full = False
    # dedup
    try:
        thr = int(globals().get('DEDUP_DHASH_THRESHOLD', globals().get('DEDUP_HAMMING', 0)) or 0)
    except Exception:
        thr = 0
    # grid timeline（grid のときだけ意味があります）
    try:
        gvt = str(globals().get('GRID_VIDEO_TIMELINE', 'off') or 'off').strip().lower()
    except Exception:
        gvt = 'off'

    # レイアウト詳細（表示用）
    layout_disp = layout
    if layout == 'grid':
        try:
            r = int(globals().get('ROWS', 0) or 0)
            c = int(globals().get('COLS', 0) or 0)
            if r > 0 and c > 0:
                layout_disp = f"grid({r}x{c})"
        except Exception as e:
            _kana_silent_exc('core:L3054', e)
            pass
    elif layout.startswith('mosaic'):
        layout_disp = layout
        # mosaic は COUNT を目安にすることが多いので、あれば一緒に出す
        try:
            cnt = int(globals().get('COUNT', 0) or 0)
            if cnt > 0:
                layout_disp = f"{layout}(count={cnt})"
        except Exception as e:
            _kana_silent_exc('core:L3063', e)
            pass
    elif layout == 'hex':
        try:
            cnt = int(globals().get('COUNT', 0) or 0)
            if cnt > 0:
                layout_disp = f"hex(count={cnt})"
        except Exception as e:
            _kana_silent_exc('core:L3070', e)
            pass
    elif layout == 'random':
        try:
            cnt = int(globals().get('COUNT', 0) or 0)
            if cnt > 0:
                layout_disp = f"random(count={cnt})"
        except Exception as e:
            _kana_silent_exc('core:L3077', e)
            pass
    # 目安枚数（grid は TARGET_COUNT、その他は COUNT を優先）
    count_disp = ""
    try:
        if layout == 'grid':
            t = int(globals().get('TARGET_COUNT', 0) or 0)
            if t > 0:
                count_disp = f"count={t}"
        else:
            c = int(globals().get('COUNT', 0) or 0)
            if c > 0:
                count_disp = f"count={c}"
    except Exception:
        count_disp = ""

    # layout_disp 側に count= を含めている場合は重複表示しない
    try:
        if isinstance(layout_disp, str) and ('count=' in layout_disp):
            count_disp = ""
    except Exception as e:
        _kana_silent_exc('core:L3098', e)
        pass
    # --- 最適化（表示用） ---
    opt_disp = ""
    try:
        if layout == 'grid':
            opt = str(globals().get('GRID_OPTIMIZER', 'off') or 'off').strip().lower()
            if opt and opt != 'off':
                extras = []
                try:
                    obj = str(globals().get('GRID_OBJECTIVE', '') or '').strip().lower()
                except Exception:
                    obj = ''
                if opt in ('anneal', 'simulated_anneal', 'simulated-anneal'):
                    try:
                        steps = int(globals().get('GRID_ANNEAL_STEPS', 0) or 0)
                    except Exception:
                        steps = 0
                    try:
                        reheats = int(globals().get('GRID_ANNEAL_REHEATS', 0) or 0)
                    except Exception:
                        reheats = 0
                    if steps > 0:
                        extras.append(f"{steps}x{reheats}" if reheats else f"{steps}")
                if obj in ('max', 'min'):
                    extras.append(f"obj={obj}")
                if opt in ('spectral-diagonal', 'spectral_diagonal', 'diagonal'):
                    try:
                        diag = str(globals().get('GRID_DIAGONAL_DIRECTION', '') or '').strip().lower()
                    except Exception:
                        diag = ''
                    if diag:
                        extras.append(f"diag={diag}")
                opt_disp = f"opt={opt}" + (f"({','.join(extras)})" if extras else "")
        elif layout.startswith('mosaic'):
            opt = str(globals().get('MOSAIC_OPTIMIZER', 'off') or 'off').strip().lower()
            if opt and opt != 'off':
                extras = []
                try:
                    obj = str(globals().get('MOSAIC_OBJECTIVE', '') or '').strip().lower()
                except Exception:
                    obj = ''
                if opt in ('anneal', 'simulated_anneal', 'simulated-anneal'):
                    try:
                        steps = int(globals().get('MOSAIC_ANNEAL_STEPS', 0) or 0)
                    except Exception:
                        steps = 0
                    try:
                        reheats = int(globals().get('MOSAIC_ANNEAL_REHEATS', 0) or 0)
                    except Exception:
                        reheats = 0
                    if steps > 0:
                        extras.append(f"{steps}x{reheats}" if reheats else f"{steps}")
                if obj in ('max', 'min'):
                    extras.append(f"obj={obj}")
                opt_disp = f"opt={opt}" + (f"({','.join(extras)})" if extras else "")
        elif layout == 'hex':
            opt = str(globals().get('HEX_OPTIMIZER', 'inherit') or 'inherit').strip().lower()
            if opt in ('none','false','0','disable','disabled','no'): opt = 'off'
            if opt == 'inherit': opt = ('off' if (not bool(globals().get('HEX_LOCAL_OPT_ENABLE', False))) else 'on')
            if opt and opt != 'off':
                extras = []
                try:
                    obj = str(globals().get('HEX_OBJECTIVE', '') or '').strip().lower()
                except Exception:
                    obj = ''
                if opt in ('anneal', 'simulated_anneal', 'simulated-anneal'):
                    try:
                        steps = int(globals().get('HEX_ANNEAL_STEPS', 0) or 0)
                    except Exception:
                        steps = 0
                    try:
                        reheats = int(globals().get('HEX_ANNEAL_REHEATS', 0) or 0)
                    except Exception:
                        reheats = 0
                    if steps > 0:
                        extras.append(f"{steps}x{reheats}" if reheats else f"{steps}")
                if obj in ('max', 'min'):
                    extras.append(f"obj={obj}")
                opt_disp = f"opt={opt}" + (f"({','.join(extras)})" if extras else "")
        elif layout == 'quilt':
            opt = str(globals().get('QUILT_OPTIMIZER', 'off') or 'off').strip().lower()
            if opt and opt != 'off':
                extras = []
                try:
                    obj = str(globals().get('QUILT_NEIGHBOR_OBJECTIVE', '') or '').strip().lower()
                except Exception:
                    obj = ''
                if opt in ('anneal', 'simulated_anneal', 'simulated-anneal', 'sa', 'simulated-annealing'):
                    try:
                        steps = int(globals().get('QUILT_ANNEAL_STEPS', 0) or 0)
                    except Exception:
                        steps = 0
                    try:
                        reheats = int(globals().get('QUILT_ANNEAL_REHEATS', 0) or 0)
                    except Exception:
                        reheats = 0
                    if steps > 0:
                        extras.append(f"{steps}x{reheats}" if reheats else f"{steps}")
                if obj in ('max', 'min'):
                    extras.append(f"obj={obj}")
                opt_disp = f"opt={opt}" + (f"({','.join(extras)})" if extras else "")
    except Exception:
        opt_disp = ""

    # --- スキャン ---
    try:
        rec = bool(globals().get('SCAN_RECURSIVE', True))
    except Exception:
        rec = True
    try:
        zip_en = bool(globals().get('ZIP_SCAN_ENABLE', False))
    except Exception:
        zip_en = False
    try:
        z7_en = bool(globals().get('SEVENZ_SCAN_ENABLE', False))
    except Exception:
        z7_en = False
    try:
        rar_en = bool(globals().get('RAR_SCAN_ENABLE', False))
    except Exception:
        rar_en = False
    try:
        vid_en = bool(globals().get('VIDEO_SCAN_ENABLE', False))
    except Exception:
        vid_en = False
    vid_note = ""
    if vid_en:
        try:
            vfmt = str(globals().get('VIDEO_FRAME_FORMAT', 'png') or 'png').strip().lower()
        except Exception:
            vfmt = 'png'
        try:
            vmax = int(globals().get('VIDEO_FRAME_MAX_DIM', 0) or 0)
        except Exception:
            vmax = 0
        if vmax > 0:
            vid_note = f"vid=on({vfmt},{vmax})"
        else:
            vid_note = f"vid=on({vfmt})"
    else:
        vid_note = "vid=off"
    scan_disp = f"scan: sub={'on' if rec else 'off'} zip={'on' if zip_en else 'off'} 7z={'on' if z7_en else 'off'} rar={'on' if rar_en else 'off'} {vid_note}"

    # --- キャッシュ（dHash） ---
    cache_disp = ''
    try:
        cf = str(globals().get('DHASH_CACHE_FILE', '') or '').strip()
    except Exception:
        cf = ''
    try:
        if cf:
            cache_disp = f"cache: dhash={Path(cf).name}"
    except Exception:
        if cf:
            cache_disp = "cache: dhash=(set)"

    # --- エフェクト（ONのみ） ---
    fx_line = ""
    try:
        fx_on = bool(globals().get('EFFECTS_ENABLE', False))
    except Exception:
        fx_on = False

    try:
        fx_parts = []
        fx_parts.append(f"effects={'on' if fx_on else 'off'}")
        if fx_on:
            # 光
            try:
                if bool(globals().get('HALATION_ENABLE', False)):
                    it = float(globals().get('HALATION_INTENSITY', 0.0) or 0.0)
                    ra = int(globals().get('HALATION_RADIUS', 0) or 0)
                    th = float(globals().get('HALATION_THRESHOLD', 0.0) or 0.0)
                    kn = float(globals().get('HALATION_KNEE', 0.0) or 0.0)
                    fx_parts.append(f"halation={it:.2f}@{ra}(thr={th:.2f},k={kn:.2f})")
            except Exception as e:
                _kana_silent_exc('core:L3275', e)
                pass
            # 色味／グレーディング
            try:
                if bool(globals().get('SPLIT_TONE_ENABLE', False)):
                    sh = float(globals().get('SPLIT_TONE_SHADOW_HUE', 0.0) or 0.0)
                    ss = float(globals().get('SPLIT_TONE_SHADOW_STRENGTH', 0.0) or 0.0)
                    hh = float(globals().get('SPLIT_TONE_HIGHLIGHT_HUE', 0.0) or 0.0)
                    hs = float(globals().get('SPLIT_TONE_HIGHLIGHT_STRENGTH', 0.0) or 0.0)
                    bal = float(globals().get('SPLIT_TONE_BALANCE', 0.0) or 0.0)
                    fx_parts.append(f"split=sh{sh:.0f}:{ss:.2f} hi{hh:.0f}:{hs:.2f} bal={bal:.2f}")
            except Exception as e:
                _kana_silent_exc('core:L3287', e)
                pass
            try:
                if bool(globals().get('TONECURVE_ENABLE', False)):
                    md = str(globals().get('TONECURVE_MODE', 'film') or 'film').strip()
                    st = float(globals().get('TONECURVE_STRENGTH', 0.0) or 0.0)
                    fx_parts.append(f"tonecurve={md}@{st:.2f}")
            except Exception as e:
                _kana_silent_exc('core:L3294', e)
                pass
            try:
                if bool(globals().get('LUT_ENABLE', False)):
                    lf = str(globals().get('LUT_FILE', '') or '').strip()
                    ls = float(globals().get('LUT_STRENGTH', 0.0) or 0.0)
                    if lf:
                        try:
                            nm = Path(lf).name
                        except Exception:
                            nm = lf
                        fx_parts.append(f"lut={nm}@{ls:.2f}")
            except Exception as e:
                _kana_silent_exc('core:L3306', e)
                pass
            try:
                if bool(globals().get('VIBRANCE_ENABLE', False)):
                    vb = float(globals().get('VIBRANCE_FACTOR', 1.0) or 1.0)
                    fx_parts.append(f"vibrance={vb:.2f}")
            except Exception as e:
                _kana_silent_exc('core:L3312', e)
                pass
            try:
                if bool(globals().get('BW_EFFECT_ENABLE', False)):
                    fx_parts.append("bw=on")
            except Exception as e:
                _kana_silent_exc('core:L3317', e)
                pass
            try:
                if bool(globals().get('SEPIA_ENABLE', False)):
                    sp = float(globals().get('SEPIA_INTENSITY', 0.0) or 0.0)
                    fx_parts.append(f"sepia={sp:.2f}")
            except Exception as e:
                _kana_silent_exc('core:L3323', e)
                pass
            # ディテール
            try:
                if bool(globals().get('CLARITY_ENABLE', False)):
                    ca = float(globals().get('CLARITY_AMOUNT', 0.0) or 0.0)
                    cr = float(globals().get('CLARITY_RADIUS', 0.0) or 0.0)
                    fx_parts.append(f"clarity={ca:.2f}@{cr:g}")
            except Exception as e:
                _kana_silent_exc('core:L3332', e)
                pass
            try:
                if bool(globals().get('UNSHARP_ENABLE', False)):
                    ua = float(globals().get('UNSHARP_AMOUNT', 0.0) or 0.0)
                    ur = float(globals().get('UNSHARP_RADIUS', 0.0) or 0.0)
                    ut = int(globals().get('UNSHARP_THRESHOLD', 0) or 0)
                    fx_parts.append(f"unsharp={ua:.2f}@{ur:g}t{ut}")
            except Exception as e:
                _kana_silent_exc('core:L3340', e)
                pass
            try:
                md = str(globals().get('DENOISE_MODE', 'off') or 'off').strip().lower()
                if md and md != 'off':
                    ds = float(globals().get('DENOISE_STRENGTH', 0.0) or 0.0)
                    fx_parts.append(f"nr={md}@{ds:.2f}")
            except Exception as e:
                _kana_silent_exc('core:L3347', e)
                pass
            try:
                if bool(globals().get('DEHAZE_ENABLE', False)):
                    da = float(globals().get('DEHAZE_AMOUNT', 0.0) or 0.0)
                    dr = int(globals().get('DEHAZE_RADIUS', 0) or 0)
                    fx_parts.append(f"dehaze={da:.2f}@{dr}")
            except Exception as e:
                _kana_silent_exc('core:L3354', e)
                pass
            # 明るさ
            try:
                if bool(globals().get('SHADOWHIGHLIGHT_ENABLE', False)):
                    sa = float(globals().get('SHADOW_AMOUNT', 0.0) or 0.0)
                    ha = float(globals().get('HIGHLIGHT_AMOUNT', 0.0) or 0.0)
                    fx_parts.append(f"shadow/high={sa:.2f}/{ha:.2f}")
            except Exception as e:
                _kana_silent_exc('core:L3363', e)
                pass
            try:
                bm = str(globals().get('BRIGHTNESS_MODE', 'off') or 'off').strip().lower()
                if bm == 'auto':
                    am = str(globals().get('AUTO_METHOD', 'hybrid') or 'hybrid').strip().lower()
                    tg = float(globals().get('AUTO_TARGET_MEAN', 0.0) or 0.0)
                    fx_parts.append(f"brightness=auto({am},tgt={tg:.2f})")
                elif bm == 'manual':
                    gn = float(globals().get('MANUAL_GAIN', 1.0) or 1.0)
                    gm = float(globals().get('MANUAL_GAMMA', 1.0) or 1.0)
                    fx_parts.append(f"brightness=manual(gain={gn:.3f},gamma={gm:.3f})")
            except Exception as e:
                _kana_silent_exc('core:L3375', e)
                pass
            # 仕上げ
            try:
                if bool(globals().get('GRAIN_ENABLE', False)):
                    ga = float(globals().get('GRAIN_AMOUNT', 0.0) or 0.0)
                    fx_parts.append(f"grain={ga:.2f}")
            except Exception as e:
                _kana_silent_exc('core:L3383', e)
                pass
            try:
                if bool(globals().get('VIGNETTE_ENABLE', False)):
                    vs = float(globals().get('VIGNETTE_STRENGTH', 0.0) or 0.0)
                    vr = float(globals().get('VIGNETTE_ROUND', 0.5) or 0.5)
                    fx_parts.append(f"vignette={vs:.2f}(r={vr:.2f})")
            except Exception as e:
                _kana_silent_exc('core:L3390', e)
                pass
        fx_line = "fx: " + " | ".join(fx_parts)
    except Exception:
        fx_line = ""

    # 1行目（Config）
    parts1 = [
        f"Config: out={out_wh}",
        f"layout={layout_disp}",
        f"select={sel}",
    ]
    if count_disp:
        parts1.append(count_disp)
    parts1.append(f"full_shuffle={'on' if full else 'off'}")
    if thr:
        parts1.append(f"dedup_thr={thr}")
    if layout == 'grid' and gvt in ('asc', 'desc'):
        parts1.append(f"grid_timeline={gvt}")
    parts1.append(f"seed={seed_used}")

    # 2行目（最適化／スキャン／キャッシュ）
    parts2 = []
    if opt_disp:
        parts2.append(opt_disp)
    parts2.append(scan_disp)
    if cache_disp:
        parts2.append(cache_disp)

    try:
        if parts2 and fx_line:
            note(', '.join(parts1) + "\n" + ' | '.join(parts2) + "\n" + fx_line)
        elif parts2:
            note(', '.join(parts1) + "\n" + ' | '.join(parts2))
        elif fx_line:
            note(', '.join(parts1) + "\n" + fx_line)
        else:
            note(', '.join(parts1))
    except Exception:
        # 失敗しても処理は続行（ログだけのベストエフォート）
        pass


def _note_face_focus_stats(d: dict, d2: dict):
    """
    Face-focus の統計を出力します（見やすさ優先）。

    表示方針:
    - picked: 実際に「顔フォーカス（クロップ中心決定）」に採用された検出ソースの内訳
    - eye-check: 目検証の結果（ok/ng/low）
    - heuristic: ルールベース（アニメ目検出など）の成功/失敗
    - model: 学習済みモデル（YOLO/YuNet等）の成功/失敗
    - rejected: 位置/比率などの理由で捨てた回数
    """
    try:
        # 実際に採用されたソース（= クロップ中心を決めたもの）
        # d は「採用された種類のカウント」を持っている前提
        picked_items = []
        total = 0

        # 表示名の統一（"ai" ではなく "model"）
        name_map = {
            "ai": "model",
            "frontal": "haar",
            "profile": "haar_profile",
            "anime": "heuristic",
            "upper": "upper",
            "person": "person",
            "saliency": "saliency",
            "center": "center",
        }

        for k in ("ai", "anime", "frontal", "profile", "upper", "person", "saliency", "center"):
            v = int(d.get(k, 0) or 0)
            if v:
                picked_items.append(f"{name_map.get(k, k)}={v}")
                total += v

        # 捨てた理由
        rej = []
        for k, lab in (("reject_pos", "pos"), ("reject_ratio", "ratio"), ("errors", "errors")):
            v = int(d.get(k, 0) or 0)
            if v:
                rej.append(f"{lab}={v}")

        # 目検証
        eyes = []
        for k, lab in (("eyes_ok", "ok"), ("eyes_ng", "ng"), ("low_reject", "low")):
            v = int(d2.get(k, 0) or 0)
            if v:
                eyes.append(f"{lab}={v}")

        # ルールベース（アニメ目検出など）
        heur = []
        for k, lab in (("anime_face_ok", "face_ok"), ("anime_face_ng", "face_ng"),
                       ("anime_eyes_ok", "eyes_ok"), ("anime_eyes_ng", "eyes_ng")):
            v = int(d2.get(k, 0) or 0)
            if v:
                heur.append(f"{lab}={v}")

        # 学習済みモデル（YOLO/YuNetなど）
        model = []
        for k, lab in (("ai_face_ok", "face_ok"), ("ai_face_ng", "face_ng")):
            v = int(d2.get(k, 0) or 0)
            if v:
                model.append(f"{lab}={v}")

        lines = []

        if picked_items:
            # total は「今回フォーカスが走った枚数」の近似（=採用カウントの総和）
            lines.append("picked: " + " ".join(picked_items) + f" (n={total})")

        if eyes:
            lines.append("eye-check: " + " ".join(eyes))

        if heur:
            lines.append("heuristic: " + " ".join(heur))

        # モデルが有効なら、0でも表示して「動いている/いない」を確認できるようにする
        if bool(globals().get("FACE_FOCUS_AI_ENABLE", False)):
            if model:
                lines.append("model: " + " ".join(model))
            else:
                lines.append("model: face_ok=0 face_ng=0")
        elif model:
            lines.append("model: " + " ".join(model))

        if rej:
            lines.append("rejected: " + " ".join(rej))

        if lines:
            note("Face-focus:")
            for ln in lines:
                note("  " + ln)
        else:
            note("Face-focus: (all zero)")
    except Exception as e:
        _kana_silent_exc('core:L3528', e)
        pass
def _mode_tag_for_console() -> str:
    """存在しない設定があっても落ちないように、表示用のレイアウト/モード名を返します。"""
    ls  = globals().get("LAYOUT_STYLE", "grid")
    md  = globals().get("MODE", "fill")
    mu  = globals().get("MOSAIC_UNIFORM", None)  # 古い版だと存在しないことがある

    # Grid（格子）
    if ls == "grid":
        return "Fill" if md == "fill" else "Fit"

    # Mosaic 系（表記ゆれ対応）
    if ls in ("mosaic", "mosaic-uniform-height", "mosaic-uniform-width"):
        if mu == "height": return "Uniform Height"
        if mu == "width":  return "Uniform Width"
        if "height" in ls: return "Uniform Height"
        if "width"  in ls: return "Uniform Width"
        return "Mosaic"

    # Quilt / Mondrian（大小ブロック）
    if ls in ("quilt", "quilt-bsp", "mondrian"):
        return "quilt"


    # その他（ランダム/カスタム文字列）
    if "mosaic" in ls:
        if "height" in ls: return "Uniform Height"
        if "width"  in ls: return "Uniform Width"
        return "Mosaic"

    # 最後の手段：生の文字列をそのまま返す
    return str(ls)

# --- 言語ヘルパー：UI_LANG に応じて JA / EN を返す ---
def _lang(msg_ja: str, msg_en: str) -> str:
    """UI_LANG が 'en' のとき英語、そうでなければ日本語を返す。"""
    return msg_en if globals().get("UI_LANG", "ja") == "en" else msg_ja


# --- 描画プリフェッチの先読み量を、解像度に応じて自動で抑制 ---
# 目的: 4K/8K など巨大出力時に、同時デコード数が増えすぎてメモリを食うのを防ぎます。
# 既定は互換維持のため「控えめにするだけ」（上限を設ける）で、ユーザーが手動値を小さくした場合は尊重します。

def _effective_draw_prefetch_ahead(out_w: int, out_h: int, base_ahead: int) -> int:
    try:
        base = int(base_ahead)
    except Exception:
        base = 0
    if base <= 0:
        return 0
    try:
        if not bool(globals().get('DRAW_PREFETCH_AUTO', True)):
            return base
        w = max(1, int(out_w))
        h = max(1, int(out_h))
        mp = (w * h) / 1_000_000.0
        # 4K=約8.3MP, 5K/6Kも考慮して 12MP を境界に。8K=約33MP。
        if mp >= 30.0:
            cap = int(globals().get('DRAW_PREFETCH_AHEAD_8K', 8))
        elif mp >= 12.0:
            cap = int(globals().get('DRAW_PREFETCH_AHEAD_4K', 16))
        else:
            cap = base
        # cap==0 を許容（自動でも無効化したい場合）
        if cap <= 0:
            return 0
        return int(min(base, cap))
    except Exception:
        return base


def _note_opt_improve_sumdelta(init_sum: float, best_sum: float, objective: str, accepted: int, steps: int, label: str = "ΣΔcolor") -> None:
    """統一表示: 近傍色差（ΣΔcolor）の改善量をまとめて表示する。

    Mosaic の表示形式に合わせて、
      init -> best (Δ=..., ...%) / accepted a/b
    を出します。

    - objective は "min"／"max"（表示の % は「改善量」を正にするため objective を考慮）
      - min（似せる）: best < init が改善 → Δ は負、% は正
      - max（散らす）: best > init が改善 → Δ は正、% は正

    - accepted/steps（受理率）の見方（目安）
      - 低すぎる（例: <0.1%）: 温度が低すぎる/探索が硬い → T0↑ や reheats↑ を検討
      - 高すぎる（例: >10%）: 温度が高すぎる/収束が甘い → T0↓ や Tend↓ を検討
      ※ 画像枚数や近傍 k によって適正は変わるため、主に“比較用の指標”として見る
    """
    try:
        obj = str(objective).lower().strip()
        ini = float(init_sum)
        fin = float(best_sum)
        delta = fin - ini  # 実値の変化（min改善ならマイナス）
        gain = delta if obj == "max" else (-delta)  # 改善量（改善ならプラス）
        pct = (gain / max(1e-9, abs(ini))) * 100.0
        note(f"{label}: init={ini:.3f} -> best={fin:.3f} (Δ={delta:+.3f}, {pct:+.4f}%) / accepted {int(accepted)}/{int(steps)}")
    except Exception:
        try:
            note(f"{label}: init={init_sum} -> best={best_sum} / accepted {accepted}/{steps}")
        except Exception as e:
            _warn_exc_once(e)
            pass
def init_console():
    _maybe_force_utf8(); _enable_ansi()
    UI["style"] = UI_STYLE if UI_STYLE in ("ascii","unicode") else "unicode"
    UI["ansi"]  = ANSI_OK and (UI["style"]=="unicode")

_TR_MAP = {
    # ===== バナー／フェーズ =====
    "処理中: Hex tiles (prefetch)": "Prefetch: Hex tiles",
    "事前生成: Hex tiles": "Prefetch: Hex tiles",
    "スキャン完了": "Scan complete",
    "処理中: Grid": "Rendering (Grid)",
    "描画中: Grid": "Rendering (Grid)",
    "処理中: Mosaic / Uniform Height": "Rendering (Mosaic - Uniform Height)",
    "描画中: Mosaic / Uniform Height": "Rendering (Mosaic - Uniform Height)",
    "処理中: Mosaic / Uniform Width":  "Rendering (Mosaic - Uniform Width)",
    "描画中: Mosaic / Uniform Width":  "Rendering (Mosaic - Uniform Width)",
    "最適化: Grid 近傍色差（hill）":        "Optimize: Grid neighbor color (hill)",
    "最適化: Grid anneal":      "Optimize: Grid anneal",
    "最適化: Grid anneal":              "Optimize: Grid anneal",
    "最適化: Grid 市松（checkerboard）":     "Optimize: Grid checkerboard",
    "最適化: Grid spectral→hilbert":   "Optimize: Grid spectral→hilbert",
    "前処理: Global order（spectral→hilbert）": "Preprocess: Global order (spectral→hilbert)",
    "前処理: Global order（anneal/hill）":            "Preprocess: Global order (anneal/hill)",

    "前処理: Hex 市松（checkerboard seed）": "Preprocess: Hex checkerboard seed",
    "前処理: Hex 市松(checkerboard seed)":  "Preprocess: Hex checkerboard seed",
    "前処理: Hex/Global（スペクトル→対角）": "Preprocess: Hex/Global (spectral→diagonal)",
    "前処理: Hex/Global(スペクトル→対角)":  "Preprocess: Hex/Global (spectral→diagonal)",
    "最適化: Grid スペクトル→対角スイープ": "Optimize: Grid spectral→diagonal sweep",
    "最適化: Mosaic バランス（行）":         "Optimize: Mosaic balance (rows)",
    "最適化: Mosaic バランス（列）":         "Optimize: Mosaic balance (columns)",
    "最適化: Mosaic 色差（行の並び）":       "Optimize: Mosaic color diff (row order)",
    "最適化: Mosaic 色差（列の並び）":       "Optimize: Mosaic color diff (column order)",
    "明るさ（自動・背景無視）":              "Brightness (auto, ignore background)",
    "明るさ（手動）":                        "Brightness (manual)",
    "完了":                                   "Done",

    # 追加分（ランダム抽出/明るさ/描画完了/壁紙更新）
    "ランダム抽出完了": "Random selection complete",
    "ランダム抽出Done": "Random selection done",
    "ランダム抽出":     "Random selection",
    "近似重複除去":   "near-duplicate filtering",
    # 抽出モードのラベル（更新順/古い順/名前順 など）
    "更新順抽出完了": "Recent selection complete",
    "更新順抽出Done": "Recent selection done",
    "更新順抽出":     "Recent selection",
    "並び替え抽出完了": "Sorted selection complete",
    "並び替え抽出Done": "Sorted selection done",
    "並び替え抽出":     "Sorted selection",
    "美選抜抽出完了":   "Aesthetic selection complete",
    "美選抜抽出Done":   "Aesthetic selection done",
    "美選抜抽出":       "Aesthetic selection",
    "明るさ 調整":      "Brightness adjustment",
    "明るさ調整":       "Brightness adjustment",
    "描画Done":         "Render done",
    "壁紙を更新しました": "Wallpaper updated",

    # ===== 補足／ラベル =====
    "候補": "Candidates",
    "保存": "Saved",
    "選抜": "Picked",
    "目的: 最大化(バラけ)": "Objective: maximize (diversify)",
    "目的: 最小化(似せる)": "Objective: minimize (similarize)",
    "行バランス": "Row balance",
    "列バランス": "Column balance",
    "ダブルクリック（既定の複数フォルダ）:": "Double-click (default folders):",

    # ===== 文中トークン =====
    "平均": "Mean",
    "目標": "target",
    "受理": "accepted",
    "採用": "accepted",
    "行あたり": "per row",
    "列あたり": "per column",
    "ゲイン": "gain",
    "ガンマ": "gamma",
    "モード/メソッド": "mode/method",
    "ΣΔ色(行)": "ΣΔcolor (row)",
    "ΣΔ色(列)": "ΣΔcolor (column)",
    "ΣΔ色":     "ΣΔcolor",
    "最適化: Grid スペクトル→対角スイープ":   "Optimize: Grid spectral→diagonal sweep",
    "対角スイープ": "diagonal sweep",

    # 追加トークン: 最適化の反復回数を英語UIに翻訳する
    # "反復" は optimization loops/iterations の意味
    "反復": "iterations",

    # === 追加翻訳（未翻訳だったバナー/ラベル） ===
    # 特徴量抽出フェーズ（score_and_pick で使用）
    "特徴量抽出": "Feature extraction",
    # 類似除去フェーズ（dHash）
    "類似除去 (dHash)": "Deduplication (dHash)",
    # 類似除去（汎用/フォールバック）
    "類似除去": "Deduplication",

}
# ===== 日本語UIでも英語のまま出したいログラベル（読みやすさ優先） =====
_TR_FORCE_EN = {
    "ダブルクリック（既定の複数フォルダ）:": "Double-click (default folders):",
    "ドラッグ＆ドロップ入力:": "D&D/CLI:",
    "サブフォルダを含めて走査します": "Scan includes subfolders",
    "候補:": "Candidates:",
    "抽出モード:": "Selection mode:",
    "近似重複除去:": "Near-duplicate filtering:",
    "選抜:": "Picked:",
    "行バランス:": "Row balance:",
    "列バランス:": "Column balance:",
    "目的:": "Objective:",
    "ΣΔ色(行):": "ΣΔColor(row):",
    "ΣΔ色(列):": "ΣΔColor(col):",
    "平均:": "Mean:",
    "モード/メソッド:": "Mode/Method:",
    "描画完了:": "Rendered:",
    "目標": "target",
    "行あたり": "per row",
    "列あたり": "per column",
}

def _tr(s: str) -> str:
    """UI_LANG == 'en' のとき、日本語メッセージを英語に置換して返します。
    部分一致置換のあと、記号・句読点などを英語 UI 向けに正規化します。
    """
    try:
        s = str(s)
    except Exception:
        return s
    lang = globals().get("UI_LANG", "ja")
    if lang != "en":
        # 日本語UIでも、数値ログなどは英語ラベルの方が読みやすいことがあるため、限定的に置換
        for jp, en in _TR_FORCE_EN.items():
            if jp in s:
                s = s.replace(jp, en)
        return s

    # まず語句を置換
    for jp, en in _TR_MAP.items():
        if jp in s:
            s = s.replace(jp, en)

    # 記号・句読点・全角括弧などを英語UI向けに正規化
    # 先頭の和文中点「・ 」は英語のダッシュに
    s = s.replace("・ ", "- ")
    # 全角括弧や記号を半角に
    s = (s.replace("（", "(").replace("）", ")")
           .replace("％", "%").replace("：", ":")
           .replace("！", "!").replace("。", ".")
           .replace("×", "x"))
    # 矢印（→/←）は幅ズレの原因になりやすいので ASCII に置換
    s = s.replace("→", "->")
    s = s.replace("←", "<-")
    return s

def fmt_num(x, digits=3, dash="—"):
    try:
        if x is None: return dash
        return f"{x:.{digits}f}"
    except Exception:
        return dash

# === モードラベルヘルパ ===
# 現在の SELECT_MODE に対する人間が読みやすいラベルを返します。
# UI_LANG が 'en' の場合は英語、そうでない場合は日本語で表示します。
def _ansi_rgb(r: int, g: int, b: int, bold: bool=False) -> str:
    return f"\x1b[{1 if bold else 0};38;2;{r};{g};{b}m"

def _grad_list(n: int, palette: list[tuple[int,int,int]]):
    """n個の色をパレット間を線形補間して生成"""
    if n <= 0: return []
    if len(palette) == 1: return [palette[0]] * n
    out=[]; segs=len(palette)-1
    for i in range(n):
        t = i/max(1,n-1)
        k = min(int(t*segs), segs-1)
        u = (t*segs) - k
        r = int(palette[k][0]*(1-u) + palette[k+1][0]*u)
        g = int(palette[k][1]*(1-u) + palette[k+1][1]*u)
        b = int(palette[k][2]*(1-u) + palette[k+1][2]*u)
        out.append((r,g,b))
    return out

def _palette_for_title(title: str) -> list[tuple[int,int,int]]:
    """バナーのタイトル（英/日どちらでも）からセクション色を決定。
    NEON_RANDOMIZE が有効な場合は、UNICODE_NEON_PALETTE からランダムな連続色を選んで
    使用します（色数は NEON_COLORS_MIN_MAX の範囲内）。
    """
    # キーワードは英語と日本語の両方をサポートし、セクションごとに既定のパレットを選ぶ
    s = (title or "").lower()
    if "scan complete" in s or "スキャン完了" in s:
        key = "scan"
    elif "rendering (grid)" in s or ((("描画中" in s) or ("処理中" in s)) and "grid" in s):
        key = "render-grid"
    elif "uniform height" in s or "mosaic / uniform height" in s or "高さ" in s:
        key = "render-mosaic-h"
    elif "uniform width" in s or "mosaic / uniform width" in s or "幅" in s:
        key = "render-mosaic-w"
    elif "anneal" in s:
        key = "opt-anneal"
    elif "checker" in s or "市松" in s:
        key = "opt-checker"
    elif "spectral" in s or "ヒルベルト" in s or "スペクトル" in s:
        key = "opt-spectral"
    elif "preprocess" in s or "前処理" in s:
        key = "preprocess"
    elif "brightness" in s or "明るさ" in s:
        key = "brightness"
    elif "done" in s or "完了" in s:
        key = "done"
    else:
        key = "default"
    base = BANNER_PALETTES.get(key, BANNER_PALETTES["default"])
    # NEON_RANDOMIZE が有効な場合は、色数を NEON_COLORS_MIN_MAX の範囲でランダムに選び、
    # UNICODE_NEON_PALETTE から連続色を選択する。そうでない場合は既定のパレットを返す。
    try:
        if globals().get("NEON_RANDOMIZE", False):
            palette_base = globals().get("UNICODE_NEON_PALETTE", base) or base
            lo, hi = globals().get("NEON_COLORS_MIN_MAX", (3, 6))
            n = max(1, min(len(palette_base), random.randint(int(lo), int(hi))))
            if len(palette_base) <= n:
                return palette_base[:]
            start = random.randint(0, len(palette_base)-1)
            out = []
            for i in range(n):
                out.append(palette_base[(start + i) % len(palette_base)])
            return out
    except Exception as e:
        _warn_exc_once(e)
        pass
    return base

def _grad_iter(n: int, palette: list[tuple[int,int,int]]):
    """n 個ぶん、パレットを滑らかに補間した RGB を逐次返します（ジェネレータ）。"""
    if n <= 0:
        return
    if len(palette) == 1:
        for _ in range(n):
            yield palette[0]
        return
    segs = len(palette) - 1
    for i in range(n):
        t = i / max(1, n-1)
        k = min(int(t * segs), segs-1)
        local = (t*segs) - k
        r = int(palette[k][0]*(1-local) + palette[k+1][0]*local)
        g = int(palette[k][1]*(1-local) + palette[k+1][1]*local)
        b = int(palette[k][2]*(1-local) + palette[k+1][2]*local)
        yield (r,g,b)

def rainbow_text(s: str, bold: bool=True, palette: list[tuple[int,int,int]]|None=None) -> str:
    if globals().get("UI_STYLE","ascii") != "unicode" or not globals().get("UNICODE_BLING", False):
        return s
    pal = palette or globals().get("CURRENT_PALETTE") or BANNER_PALETTES["default"]
    chars = list(s); out=[]
    for (r,g,b), ch in zip(_grad_list(len(chars), pal), chars):
        out.append(_ansi_rgb(r,g,b,bold) + ch)
    out.append("\x1b[0m")
    return "".join(out)

def neon_bar(fill_len: int, empty_len: int, palette: list[tuple[int,int,int]]|None=None) -> str:
    # ASCII／Bling 無効時は地味な進捗バー
    if globals().get("UI_STYLE","ascii") != "unicode" or not globals().get("UNICODE_BLING", False):
        return BAR_FILL_CHAR*fill_len + BAR_EMPTY_CHAR*empty_len
    # 使用するパレットを決定
    pal = palette or globals().get("CURRENT_PALETTE") or BANNER_PALETTES["default"]
    style = globals().get("PROGRESS_BAR_STYLE", "segment")
    total = max(1, int(fill_len + empty_len))  # バー全体の幅でグラデを敷く
    # paint スタイル: 背景色で塗りつぶす
    if str(style).lower().startswith("paint"):
        segs=[]
        # 全体グラデを生成し、塗られた部分だけを描く
        grad_full = list(_grad_iter(total, pal))
        for (r,g,b) in grad_full[:fill_len]:
            # 背景色を変えるだけのスペース
            segs.append(f"\x1b[1;48;2;{r};{g};{b}m ")
        # 未塗り部分は空白
        if empty_len>0:
            segs.append("\x1b[0m" + " "*empty_len)
        segs.append("\x1b[0m")
        return "".join(segs)
    # segment スタイル: 短冊表示。全体のグラデを先に作り、塗られた部分のみを使用
    segs=[]
    grad_full = _grad_iter(total, pal)
    for i,(r,g,b) in enumerate(grad_full):
        if i < fill_len:
            segs.append(_ansi_rgb(r,g,b,True) + BAR_FILL_CHAR)
        else:
            break
    if empty_len>0:
        segs.append("\x1b[90m" + BAR_EMPTY_CHAR*empty_len + "\x1b[0m")
    else:
        segs.append("\x1b[0m")
    return "".join(segs)

# pad_to_width/disp_width が未定義環境でも落ちないよう保険
def _disp_width_safe(s: str) -> int:
    if "disp_width" in globals():
        return globals()["disp_width"](s)
    return len(s)

def _pad_to_width_safe(s: str, w: int) -> str:
    # 可能なら表示幅ベースの pad_to_width を使う（日本語を含むときの枠ズレ防止）
    try:
        return pad_to_width(s, w)
    except Exception:
        # 最終手段（等幅・半角のみ前提）。日本語が混ざるとズレる可能性があります。
        return s.ljust(w)

# -----------------------------------------------------------------------------
# サブセクション: 画像ユーティリティ
# -----------------------------------------------------------------------------
def parse_color(color: str) -> Tuple[int,int,int]:
    if color.startswith("#"): color=color[1:]
    if len(color)==6 and all(c in "0123456789abcdefABCDEF" for c in color):
        return (int(color[0:2],16), int(color[2:4],16), int(color[4:6],16))
    return Image.new("RGB",(1,1),color).getpixel((0,0))

# -----------------------------------------------------------------------------
# Zip 画像サポート（zip://<abs_zip_path>::<member>）
#   - 既存コードをなるべく壊さず、Path 互換として「文字列キー」を扱います。
#   - PIL へは BytesIO 経由で渡します（ZipExtFile は seek 非対応のことがあるため）。
# -----------------------------------------------------------------------------
_ZIP_MEMBER_STAT_CACHE: Dict[tuple, tuple] = {}
_ZIP_MEMBER_STAT_LOCK = threading.Lock()

def _is_zip_key(x: Any) -> bool:
    try:
        s = str(x)
    except Exception:
        return False
    return s.startswith(ZIP_KEY_PREFIX) and (ZIP_KEY_SEP in s)

def _zip_parse(key: str) -> Tuple[str, str]:
    s = key[len(ZIP_KEY_PREFIX):]
    zpath, member = s.split(ZIP_KEY_SEP, 1)
    return zpath, member

def _zip_key(zip_path: Path, member: str) -> str:
    z = os.path.normcase(os.path.abspath(str(zip_path)))
    return f"{ZIP_KEY_PREFIX}{z}{ZIP_KEY_SEP}{member}"


# -----------------------------------------------------------------------------
# 7z / rar 画像サポート（7z://<abs_7z_path>::<member> / rar://<abs_rar_path>::<member>）
#   - ZIPと違い、基本的にファイル名はUnicodeで扱える前提です。
#   - 依存が無い場合（py7zr/rarfile）やエラー時は、そのアーカイブをスキップします。
#   - dHashキャッシュ用の署名は「アーカイブ本体のmtime/size + member名のCRC」でベストエフォートに作ります。
# -----------------------------------------------------------------------------
_ARCH_MEMBER_STAT_CACHE = {}
_ARCH_MEMBER_STAT_LOCK = threading.Lock()

def _is_7z_key(x: Any) -> bool:
    try:
        s = str(x)
    except Exception:
        return False
    return s.startswith(SEVENZ_KEY_PREFIX) and (ZIP_KEY_SEP in s)

def _7z_parse(key: str) -> Tuple[str, str]:
    s = key[len(SEVENZ_KEY_PREFIX):]
    apath, member = s.split(ZIP_KEY_SEP, 1)
    return apath, member

def _7z_key(archive_path: Path, member: str) -> str:
    a = os.path.normcase(os.path.abspath(str(archive_path)))
    return f"{SEVENZ_KEY_PREFIX}{a}{ZIP_KEY_SEP}{member}"

def _is_rar_key(x: Any) -> bool:
    try:
        s = str(x)
    except Exception:
        return False
    return s.startswith(RAR_KEY_PREFIX) and (ZIP_KEY_SEP in s)

def _rar_parse(key: str) -> Tuple[str, str]:
    s = key[len(RAR_KEY_PREFIX):]
    apath, member = s.split(ZIP_KEY_SEP, 1)
    return apath, member

def _rar_key(archive_path: Path, member: str) -> str:
    a = os.path.normcase(os.path.abspath(str(archive_path)))
    return f"{RAR_KEY_PREFIX}{a}{ZIP_KEY_SEP}{member}"

def _arch_member_stat(key: str) -> Tuple[object, object]:
    """7z/rar 用の (mtime_ns, size_like) を作る（ZIPほど厳密ではないベストエフォート）。"""
    try:
        if _is_7z_key(key):
            apath, member = _7z_parse(key)
        elif _is_rar_key(key):
            apath, member = _rar_parse(key)
        else:
            return None, None

        st = os.stat(apath)
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
        asize = int(st.st_size)
        ck = (os.path.normcase(apath), member, int(mtime_ns), asize)
        with _ARCH_MEMBER_STAT_LOCK:
            hit = _ARCH_MEMBER_STAT_CACHE.get(ck)
        if hit is not None:
            return hit[0], hit[1]

        try:
            crc = zlib.crc32(member.encode("utf-8", errors="ignore")) & 0xffffffff
        except Exception:
            crc = 0
        mix_size = (asize << 32) ^ int(crc)
        out = (int(mtime_ns), int(mix_size))
        with _ARCH_MEMBER_STAT_LOCK:
            _ARCH_MEMBER_STAT_CACHE[ck] = out
            if len(_ARCH_MEMBER_STAT_CACHE) > 10000:
                _ARCH_MEMBER_STAT_CACHE.clear()
        return out[0], out[1]
    except Exception:
        return None, None


# Zip 内のファイル名が Windows 由来(CP932)で UTF-8 フラグ無しの場合、
# Python の zipfile は歴史的経緯で CP437 として解釈するため、文字化けします。
# 例: 'カナボード' -> 'âJâiâ{ü[âh'
# ここでは「表示用の正しい Unicode 名」と「zipfile が要求する内部名（文字化け名）」を相互変換します。

_ZIP_MEMBER_RESOLVE_CACHE: Dict[tuple, str] = {}
_ZIP_MEMBER_RESOLVE_LOCK = threading.Lock()

def _looks_japanese(s: str) -> bool:
    try:
        for ch in s:
            o = ord(ch)
            # ひらがな・カタカナ・CJK 統合漢字（ざっくり）
            if (0x3040 <= o <= 0x30FF) or (0x3400 <= o <= 0x4DBF) or (0x4E00 <= o <= 0x9FFF):
                return True
    except Exception as e:
        _warn_exc_once(e)
        pass
    return False

def _zip_member_display_name(name: str) -> str:
    """zipfile が返す member 名（UTF-8 フラグ無しだと CP437 解釈で文字化け）を、
    可能なら CP932 として復元した表示用名に変換する。
    """
    if not name:
        return name

    # 既に日本語っぽいなら、そのまま（UTF-8 フラグ有り等）
    if _looks_japanese(name):
        return name

    # CP437 として再エンコードできる＝zip 内の「生バイト」を復元できる
    try:
        raw = name.encode("cp437")
    except Exception:
        return name

    # Windows で作られた zip の多くは CP932（Shift_JIS 拡張）
    try:
        dec = raw.decode("cp932")
    except Exception:
        return name

    # 日本語が出てきたら復元成功とみなす
    if dec and (dec != name) and _looks_japanese(dec):
        return dec
    return name

def _zip_member_guess_internal(display_name: str) -> str:
    """表示用名（Unicode）から、zipfile が要求する内部名（文字化け名）を推測する。
    CP932 バイト列を CP437 として解釈したと仮定して戻す。
    """
    if not display_name:
        return display_name
    try:
        raw = display_name.encode("cp932")
        return raw.decode("cp437")
    except Exception:
        return display_name

def _zip_resolve_member_internal(zpath: str, member_display: str, zf: Optional[zipfile.ZipFile] = None) -> str:
    """zip:// キーに入っている member 名（表示用）を、zipfile で確実に参照できる内部名へ解決する。
    できる限り O(1)（直接→推測）で解決し、ダメなら infolist を走査して拾う。
    """
    try:
        st = os.stat(zpath)
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
        zsize = int(st.st_size)
        ck = (os.path.normcase(zpath), member_display, int(mtime_ns), zsize)
    except Exception:
        ck = (os.path.normcase(zpath), member_display, None, None)

    with _ZIP_MEMBER_RESOLVE_LOCK:
        hit = _ZIP_MEMBER_RESOLVE_CACHE.get(ck)
    if hit is not None:
        return hit

    def _resolve_in_zf(_zf: zipfile.ZipFile) -> str:
        # 1) そのまま
        try:
            _zf.getinfo(member_display)
            return member_display
        except Exception as e:
            _warn_exc_once(e)
            pass
        # 2) CP932→CP437 推測
        guess = _zip_member_guess_internal(member_display)
        if guess != member_display:
            try:
                _zf.getinfo(guess)
                return guess
            except Exception as e:
                _warn_exc_once(e)
                pass
        # 3) 最後の手段：走査（表示名で一致するものを探す）
        try:
            for info in _zf.infolist():
                try:
                    if _zip_member_display_name(getattr(info, "filename", "")) == member_display:
                        return getattr(info, "filename", member_display)
                except Exception as e:
                    _kana_silent_exc('core:L4147', e)
                    continue
        except Exception as e:
            _warn_exc_once(e)
            pass
        return member_display

    try:
        if zf is None:
            with zipfile.ZipFile(zpath) as zf2:
                internal = _resolve_in_zf(zf2)
        else:
            internal = _resolve_in_zf(zf)

        with _ZIP_MEMBER_RESOLVE_LOCK:
            _ZIP_MEMBER_RESOLVE_CACHE[ck] = internal
            if len(_ZIP_MEMBER_RESOLVE_CACHE) > 20000:
                _ZIP_MEMBER_RESOLVE_CACHE.clear()
        return internal
    except Exception:
        return member_display

def _zip_member_stat(key: str):
    # dHash キャッシュ用の (mtime_ns, size_like) を作る
    # zip:// キーの member 部は「表示用名」を保持し、ここで内部名へ解決してから参照する。
    try:
        zpath, member_display = _zip_parse(key)
        st = os.stat(zpath)
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
        zsize = int(st.st_size)
        ck = (zpath, member_display, int(mtime_ns), zsize)
        with _ZIP_MEMBER_STAT_LOCK:
            hit = _ZIP_MEMBER_STAT_CACHE.get(ck)
        if hit is not None:
            return hit[0], hit[1]

        with zipfile.ZipFile(zpath) as zf:
            member_internal = _zip_resolve_member_internal(zpath, member_display, zf=zf)
            info = zf.getinfo(member_internal)
            crc = int(getattr(info, "CRC", 0))
            fsz = int(getattr(info, "file_size", 0))

        mix_size = (zsize << 32) ^ (crc & 0xffffffff) ^ (fsz & 0xffffffff)
        out = (int(mtime_ns), int(mix_size))
        with _ZIP_MEMBER_STAT_LOCK:
            _ZIP_MEMBER_STAT_CACHE[ck] = out
            if len(_ZIP_MEMBER_STAT_CACHE) > 10000:
                _ZIP_MEMBER_STAT_CACHE.clear()
        return out[0], out[1]
    except Exception:
        return None, None

def _imgref_mtime(p: ImageRef) -> float:
    try:
        s = str(p)
        if _is_zip_key(s):
            zpath, _ = _zip_parse(s)
            return float(os.path.getmtime(zpath))
        if _is_7z_key(s):
            apath, _ = _7z_parse(s)
            return float(os.path.getmtime(apath))
        if _is_rar_key(s):
            apath, _ = _rar_parse(s)
            return float(os.path.getmtime(apath))
        return float(Path(s).stat().st_mtime)
    except Exception:
        try:
            return float(os.path.getmtime(str(p)))
        except Exception:
            return 0.0

def _imgref_size(p: ImageRef) -> int:
    try:
        s = str(p)
        if _is_zip_key(s):
            _m, _sz = _zip_member_stat(s)
            return int(_sz or 0)
        if _is_7z_key(s) or _is_rar_key(s):
            _m, _sz = _arch_member_stat(s)
            return int(_sz or 0)
        return int(Path(s).stat().st_size)
    except Exception:
        try:
            return int(os.path.getsize(str(p)))
        except Exception:
            return 0

def _imgref_display(p: ImageRef) -> str:
    s = str(p)
    if _is_zip_key(s) or _is_7z_key(s) or _is_rar_key(s):
        return s
    # 動画抽出フレーム（キャッシュ）なら、元の動画情報を優先表示（used_images 等の見やすさ用）
    try:
        v = _VIDEO_FRAME_SRC.get(str(p))
        if v:
            return v
    except Exception as e:
        _kana_silent_exc('core:L4243', e)
        pass
    try:
        return str(Path(s).resolve())
    except Exception:
        return s

# -----------------------------------------------------------------------------
# 画像読み込みの安全ガード（巨大画像／ZIP爆弾 対策）
#   - MAX_IMAGE_PIXELS_LIMIT: 画像の最大ピクセル数（これを超える画像はスキップ）
#   - ZIP_MEMBER_MAX_BYTES  : ZIP内メンバーの展開後サイズ上限（bytes）
#   - ZIP_MEMBER_MAX_RATIO  : ZIPの展開比上限（file_size／compress_size）
# -----------------------------------------------------------------------------
try: MAX_IMAGE_PIXELS_LIMIT
except NameError: MAX_IMAGE_PIXELS_LIMIT = 200_000_000  # 200MP（4K/8Kは余裕、異常に巨大な画像を避ける）
try: ZIP_MEMBER_MAX_BYTES
except NameError: ZIP_MEMBER_MAX_BYTES = 256 * 1024 * 1024  # 256MB（展開後）
try: ZIP_MEMBER_MAX_RATIO
except NameError: ZIP_MEMBER_MAX_RATIO = 300  # 展開比が異常に高いもの（ZIP爆弾対策）

def open_image_safe(p: ImageRef, draft_to: Optional[Tuple[int,int]] = None, force_mode: Optional[str] = None) -> Image.Image:
    """Path または zip:// / 7z:// / rar:// キーから PIL.Image を安全に開く。

    安全のために以下を行います：
    - 元の Image.open() 由来オブジェクト（ファイルハンドル）を必ず close
    - exif_transpose/convert で別オブジェクトになっても、load()+copy() で完全に分離
    - 巨大画像（MAX_IMAGE_PIXELS_LIMIT）や ZIP 爆弾疑い（ZIP_MEMBER_MAX_BYTES／ZIP_MEMBER_MAX_RATIO）はスキップ
    """
    s = str(p)

    def _postprocess_and_detach(im0: Image.Image) -> Image.Image:
        # JPEG などは draft で“だいたい目標サイズ”に近い解像度でデコードさせると速い
        # ※ draft は load() 前にしか効かないので、ここ（detach 前）で適用する
        if draft_to is not None:
            try:
                fmt = str(getattr(im0, "format", "")).upper()
                if fmt in ("JPEG", "JPG"):
                    im0.draft(force_mode or "RGB", (max(1, int(draft_to[0])), max(1, int(draft_to[1]))))
            except Exception as e:
                _warn_exc_once(e)
                pass

        im = ImageOps.exif_transpose(im0)
        if force_mode:
            if im.mode != force_mode:
                im = im.convert(force_mode)
        else:
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")

        try:
            max_px = int(globals().get("MAX_IMAGE_PIXELS_LIMIT", MAX_IMAGE_PIXELS_LIMIT))
        except Exception:
            max_px = int(MAX_IMAGE_PIXELS_LIMIT)

        if max_px and max_px > 0:
            try:
                w, h = im.size
                if int(w) * int(h) > int(max_px):
                    raise ValueError(f"Image too large: {w}x{h} pixels (limit={max_px})")
            except Exception as e:
                # サイズ取得が失敗した場合も、念のためスキップ側に倒す
                raise ValueError(f"Image size check failed: {e}") from e

        # load() でデータを読み切り、copy() で完全にメモリに分離（ハンドルリーク防止）
        im.load()
        return im.copy()

    if _is_zip_key(s):
        zpath, member = _zip_parse(s)
        bio = None  # type: Optional[io.BytesIO]
        im0 = None  # type: Optional[Image.Image]
        try:
            with zipfile.ZipFile(zpath) as zf:
                member_internal = _zip_resolve_member_internal(zpath, member, zf=zf)
                # ZIP爆弾対策：サイズ・展開比でガード
                try:
                    zi = zf.getinfo(member_internal)
                    max_bytes = int(globals().get("ZIP_MEMBER_MAX_BYTES", ZIP_MEMBER_MAX_BYTES))
                    if max_bytes and max_bytes > 0 and int(zi.file_size) > int(max_bytes):
                        raise ValueError(f"ZIP member too large: {zi.file_size} bytes (limit={max_bytes})")
                    max_ratio = int(globals().get("ZIP_MEMBER_MAX_RATIO", ZIP_MEMBER_MAX_RATIO))
                    if max_ratio and max_ratio > 0 and int(zi.compress_size) > 0:
                        ratio = float(zi.file_size) / float(zi.compress_size)
                        if ratio > float(max_ratio):
                            raise ValueError(f"ZIP expand ratio too high: {ratio:.1f} (limit={max_ratio})")
                except Exception as e:
                    # ガード判定自体が失敗した場合は、読む前にスキップ
                    raise
                data = zf.read(member_internal)

            bio = io.BytesIO(data)
            im0 = Image.open(bio)
            return _postprocess_and_detach(im0)
        finally:
            if im0 is not None:
                try:
                    im0.close()
                except Exception as e:
                    _warn_exc_once(e)
                    pass
            if bio is not None:
                try:
                    bio.close()
                except Exception as e:
                    _warn_exc_once(e)
                    pass

    elif _is_7z_key(s):
        apath, member = _7z_parse(s)
        bio = None  # type: Optional[io.BytesIO]
        im0 = None  # type: Optional[Image.Image]
        try:
            try:
                import py7zr  # type: ignore
            except Exception as e:
                raise ImportError('py7zr not installed') from e

            with py7zr.SevenZipFile(apath, mode='r') as zf:
                data_map = zf.read([member])
                obj = None
                try:
                    obj = data_map.get(member)
                except Exception:
                    obj = None
                if obj is None:
                    try:
                        # 互換: dictの先頭を使う
                        obj = next(iter(data_map.values()))
                    except Exception:
                        obj = None
                if obj is None:
                    raise FileNotFoundError(f'7z member not found: {member}')
                if hasattr(obj, 'read'):
                    data = obj.read()
                else:
                    data = bytes(obj)

            # サイズガード（展開後）
            try:
                max_bytes = int(globals().get('ZIP_MEMBER_MAX_BYTES', ZIP_MEMBER_MAX_BYTES))
            except Exception:
                max_bytes = int(ZIP_MEMBER_MAX_BYTES)
            if max_bytes and max_bytes > 0 and len(data) > int(max_bytes):
                raise ValueError(f'Archive member too large: {len(data)} bytes (limit={max_bytes})')

            bio = io.BytesIO(data)
            im0 = Image.open(bio)
            return _postprocess_and_detach(im0)
        finally:
            if im0 is not None:
                try:
                    im0.close()
                except Exception as e:
                    _warn_exc_once(e)
                    pass
            if bio is not None:
                try:
                    bio.close()
                except Exception as e:
                    _warn_exc_once(e)
                    pass

    elif _is_rar_key(s):
        apath, member = _rar_parse(s)
        bio = None  # type: Optional[io.BytesIO]
        im0 = None  # type: Optional[Image.Image]
        try:
            try:
                import rarfile  # type: ignore
            except Exception as e:
                raise ImportError('rarfile not installed') from e

            with rarfile.RarFile(apath) as rf:
                try:
                    info = rf.getinfo(member)
                    try:
                        max_bytes = int(globals().get('ZIP_MEMBER_MAX_BYTES', ZIP_MEMBER_MAX_BYTES))
                    except Exception:
                        max_bytes = int(ZIP_MEMBER_MAX_BYTES)
                    if max_bytes and max_bytes > 0 and int(getattr(info, 'file_size', 0) or 0) > int(max_bytes):
                        raise ValueError(f'Archive member too large: {int(getattr(info, "file_size", 0) or 0)} bytes (limit={max_bytes})')
                except Exception:
                    # info取得やガードが失敗した場合でも、読み込み自体は試みる（後段でlen(data)でもガード）
                    pass
                data = rf.read(member)

            try:
                max_bytes = int(globals().get('ZIP_MEMBER_MAX_BYTES', ZIP_MEMBER_MAX_BYTES))
            except Exception:
                max_bytes = int(ZIP_MEMBER_MAX_BYTES)
            if max_bytes and max_bytes > 0 and len(data) > int(max_bytes):
                raise ValueError(f'Archive member too large: {len(data)} bytes (limit={max_bytes})')

            bio = io.BytesIO(data)
            im0 = Image.open(bio)
            return _postprocess_and_detach(im0)
        finally:
            if im0 is not None:
                try:
                    im0.close()
                except Exception as e:
                    _warn_exc_once(e)
                    pass
            if bio is not None:
                try:
                    bio.close()
                except Exception as e:
                    _warn_exc_once(e)
                    pass

    else:
        im0 = Image.open(s)

        try:
            return _postprocess_and_detach(im0)
        finally:
            try:
                im0.close()
            except Exception as e:
                _warn_exc_once(e)
                pass
def resize_into_cell(img: Image.Image, cw:int, ch:int, mode:str) -> Image.Image:
    iw, ih = img.size
    if mode == "fit":
        s = min(cw/iw, ch/ih) if iw and ih else 1.0
        tw, th = max(1, int(iw*s)), max(1, int(ih*s))
        return hq_resize(img, (tw, th))

    # fill の場合：フォーカス検出（Face→Person→Saliency）が有効なら、中央クロップより先に試す。
    # ※ 画像を貼るすべての経路で「なるべく主題を残す」ため、ここに集約します。
    try:
        if bool(globals().get("FACE_FOCUS_ENABLE", True)) and bool(globals().get("FACE_FOCUS_FORCE_ALL_MODES", True)):
            sp = getattr(img, "filename", None)
            return _cover_rect_face_focus(img, int(cw), int(ch), src_path=sp)
    except Exception as e:
        _kana_silent_exc('core:L4478', e)
        pass
    s = max(cw/iw, ch/ih) if iw and ih else 1.0
    tw, th = max(1, int(math.ceil(iw*s))), max(1, int(math.ceil(ih*s)))
    tmp = hq_resize(img, (tw, th))
    x0 = max(0, (tmp.size[0] - cw) // 2); y0 = max(0, (tmp.size[1] - ch) // 2)
    return tmp.crop((x0, y0, x0 + cw, y0 + ch))

def paste_cell(canvas: Image.Image, mask: Image.Image, im: Image.Image,
               x:int, y:int, w:int, h:int, mode:str):
    if mode=="fit":
        rez=resize_into_cell(im,w,h,"fit")
        rx=x+(w-rez.size[0])//2; ry=y+(h-rez.size[1])//2
        canvas.paste(rez,(rx,ry))
        mask.paste(255, (rx, ry, rx + rez.size[0], ry + rez.size[1]))
    else:
        rez=resize_into_cell(im,w,h,"fill")
        canvas.paste(rez,(x,y))
        mask.paste(255, (x, y, x + w, y + h))

def hq_resize(img: Image.Image, size: tuple[int,int]) -> Image.Image:
    """大幅縮小に強い高品質リサイズ（CPU）：
    Pillow の `reducing_gap` を優先利用し、JPEG は draft/reduce による高速化も期待します。
    古い Pillow では reducing_gap が無い場合があるため、その際は従来の段階縮小にフォールバックします。
    """
    tw, th = max(1, int(size[0])), max(1, int(size[1]))
    iw, ih = img.size
    if iw == tw and ih == th:
        return img

    # 非 HQ：速度優先（reducing_gap が使える場合は 2.0 を採用）
    if RESAMPLE_MODE != "hq":
        try:
            return img.resize((tw, th), Resampling.LANCZOS, reducing_gap=2.0)
        except TypeError:
            return img.resize((tw, th), Resampling.LANCZOS)

    # HQ：品質を保ちつつ高速化（3.0 以上は fair resampling とほぼ同等）
    try:
        return img.resize((tw, th), Resampling.LANCZOS, reducing_gap=3.0)
    except TypeError:
        # 旧 Pillow フォールバック：2倍刻みで近づける → 最終 LANCZOS
        cur = img
        while iw // 2 >= tw * 1.1 and ih // 2 >= th * 1.1:
            iw //= 2
            ih //= 2
            cur = cur.resize((max(1, iw), max(1, ih)), Image.BOX)
        return cur.resize((tw, th), Resampling.LANCZOS)


# -----------------------------------------------------------------------------
# 動画フレーム抽出（候補画像に混ぜる）
# -----------------------------------------------------------------------------
_VIDEO_FRAME_SRC: Dict[str, str] = {}  # 抽出フレーム（ファイル）→表示用の元情報
_VIDEO_FRAME_META: Dict[str, Tuple[str, int]] = {}  # 抽出フレーム（ファイル）→ (元動画名lower, 推定/実測timestamp_ms)
_VIDEO_WARNED_NO_CV2 = False

def _video_cache_dir() -> Path:
    try:
        p = Path(str(globals().get("VIDEO_FRAME_CACHE_DIR", ""))).expanduser()
        if not str(p):
            raise ValueError("empty")
    except Exception:
        p = Path(tempfile.gettempdir()) / "kana_wallpaper_video_frames_cache"

    # 容量肥大化対策：起動時に動画フレームキャッシュを毎回削除（任意）
    try:
        if bool(globals().get("VIDEO_FRAME_CACHE_CLEAR_ON_START", False)) and not bool(globals().get("_VIDEO_CACHE_CLEARED", False)):
            cd = p
            # 安全チェック：一時フォルダ配下 or 既定名っぽいフォルダ名のみ削除対象にする
            safe = False
            try:
                cd_res = cd.resolve()
                tmp_res = Path(tempfile.gettempdir()).resolve()
                if (tmp_res in cd_res.parents) or ("kana_wallpaper_video_frames_cache" in cd_res.name):
                    safe = True
            except Exception:
                safe = ("kana_wallpaper_video_frames_cache" in str(cd))

            if safe or bool(globals().get("VIDEO_FRAME_CACHE_CLEAR_FORCE", False)):
                try:
                    shutil.rmtree(str(cd), ignore_errors=True)
                except Exception as e:
                    _kana_silent_exc('core:L4560', e)
                    pass
            else:
                pass
            globals()["_VIDEO_CACHE_CLEARED"] = True
    except Exception:
        globals()["_VIDEO_CACHE_CLEARED"] = True

    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        _kana_silent_exc('core:L4570', e)
        pass
    return p


def _video_frames_cache_cleanup_on_exit() -> None:
    """処理終了後（プロセス終了時）に動画フレームキャッシュを削除（任意）"""
    try:
        if not bool(globals().get("VIDEO_FRAME_CACHE_CLEAR_ON_END", False)):
            return
        if bool(globals().get("_VIDEO_CACHE_CLEARED_END", False)):
            return

        try:
            cd = Path(str(globals().get("VIDEO_FRAME_CACHE_DIR", ""))).expanduser()
            if not str(cd):
                raise ValueError("empty")
        except Exception:
            cd = Path(tempfile.gettempdir()) / "kana_wallpaper_video_frames_cache"

        # 安全チェック：一時フォルダ配下 or 既定名っぽいフォルダ名のみ削除対象にする
        safe = False
        try:
            cd_res = cd.resolve()
            tmp_res = Path(tempfile.gettempdir()).resolve()
            if (tmp_res in cd_res.parents) or ("kana_wallpaper_video_frames_cache" in cd_res.name):
                safe = True
        except Exception:
            try:
                safe = ("kana_wallpaper_video_frames_cache" in str(cd))
            except Exception:
                safe = False

        if safe or bool(globals().get("VIDEO_FRAME_CACHE_CLEAR_FORCE", False)):
            try:
                shutil.rmtree(str(cd), ignore_errors=True)
            except Exception as e:
                _kana_silent_exc('core:L4606', e)
                pass
        globals()["_VIDEO_CACHE_CLEARED_END"] = True
    except Exception as e:
        _kana_silent_exc('core:L4609', e)
        pass
# プロセス終了時に（必要なら）キャッシュ削除
try:
    import atexit
    atexit.register(_video_frames_cache_cleanup_on_exit)
except Exception as e:
    _kana_silent_exc('core:L4617', e)
    pass
def _video_cache_key(vp: Path) -> str:
    """
    動画フレームキャッシュ用のキー。

    - ファイル名順（name_asc/name_desc）で並べたときに「元動画名 → 時間順」になりやすいよう、
      ベース名に動画の stem（ファイル名）を含めます。
    - 衝突回避のため、パス/mtime/size から作る短いハッシュも付与します。
    """
    # ファイルの更新でキャッシュが衝突しにくいように mtime/size を混ぜる
    try:
        st = vp.stat()
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
        sz = int(st.st_size)
    except Exception:
        mtime_ns, sz = 0, 0

    try:
        stem = vp.stem
    except Exception:
        stem = "video"

    # Windows の禁止文字を避けつつ、読みやすさも少し残す
    try:
        stem = re.sub(r'[\\/:*?"<>|]+', "_", str(stem))
        stem = re.sub(r"\s+", "_", stem).strip("_")
    except Exception:
        stem = "video"

    if not stem:
        stem = "video"
    if len(stem) > 60:
        stem = stem[:60]

    s = f"{str(vp.resolve())}|{mtime_ns}|{sz}"
    h = hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()
    return f"{stem}_{h[:10]}"


def _video_rng_for_file(vp: Path) -> random.Random:
    # 再現性：SHUFFLE_SEED が数値ならそれをベースに、動画ごとに固定乱数列を作る
    ss = globals().get("SHUFFLE_SEED", None)
    seed_base = None
    if isinstance(ss, int):
        seed_base = ss
    else:
        try:
            if isinstance(ss, str) and ss.strip().isdigit():
                seed_base = int(ss.strip())
        except Exception:
            seed_base = None
    if seed_base is None:
        # 既定は「実行ごとに変わる乱数」を使う（同一実行内では動画ごとに固定）
        rs = globals().get("_RUN_SEED_USED", None)
        if isinstance(rs, int):
            seed_base = rs
        else:
            seed_base = int(hashlib.sha1(str(vp).encode("utf-8", "ignore")).hexdigest()[:8], 16)
    seed = int(hashlib.sha1((str(vp) + "|" + str(seed_base)).encode("utf-8", "ignore")).hexdigest()[:8], 16)
    return random.Random(seed)

# --- 動画の表示アスペクト（SAR/DAR）補正（任意） ------------------------------------------
# OpenCV の VideoCapture はコンテナの SAR/DAR（再生時の表示アス比）を反映しない場合があります。
# そのため、ffprobe が利用できる環境では、動画ストリームの sample_aspect_ratio / display_aspect_ratio
# を読み取り、抽出フレームを「正しい表示アス比（正方画素）」に補正してからキャッシュへ保存します。
_VIDEO_DAR_CACHE: Dict[str, float] = {}
_VIDEO_WARNED_NO_FFPROBE = False
_VIDEO_FFPROBE_SHOWN = False

def _parse_ratio_str(s: str) -> Tuple[int, int]:
    try:
        s = (s or "").strip()
        if not s or s.upper() == "N/A":
            return (0, 0)
        if ":" in s:
            a, b = s.split(":", 1)
        elif "/" in s:
            a, b = s.split("/", 1)
        else:
            # 単一数値は分子扱い（分母=1）
            return (int(float(s)), 1)
        num = int(a)
        den = int(b)
        if den == 0:
            return (0, 0)
        return (num, den)
    except Exception:
        return (0, 0)

def _probe_video_display_aspect(vp: Path) -> Optional[float]:
    """動画 vp の「表示アスペクト比（DAR）」を推定して返す（例: 1.777...）。"""
    global _VIDEO_WARNED_NO_FFPROBE
    key = str(vp)
    if key in _VIDEO_DAR_CACHE:
        return _VIDEO_DAR_CACHE[key]

    if not bool(globals().get("VIDEO_ASPECT_FROM_CONTAINER", True)):
        return None

    try:
        ffprobe = globals().get("VIDEO_FFPROBE_PATH", None)
        if ffprobe:
            ffprobe = str(ffprobe).strip().strip('"')
            if not os.path.exists(ffprobe):
                ffprobe = None
        if not ffprobe:
            ffprobe = shutil.which("ffprobe") or shutil.which("ffprobe.exe")

        if not ffprobe:
            if not _VIDEO_WARNED_NO_FFPROBE:
                _VIDEO_WARNED_NO_FFPROBE = True
                try:
                    note(_lang("ffprobe が見つからないため、動画のSAR/DAR（表示アス比）補正は行いません","ffprobe not found; SAR/DAR (display aspect ratio) correction will be skipped"))
                except Exception as e:
                    _kana_silent_exc('core:L4732', e)
                    pass
            return None

        global _VIDEO_FFPROBE_SHOWN
        if (not _VIDEO_FFPROBE_SHOWN) and bool(globals().get("VERBOSE", False)) and bool(globals().get("VIDEO_DIAG_SHOW_FFPROBE_LINE", False)):
            _VIDEO_FFPROBE_SHOWN = True
            try:
                note(f"ffprobe: {ffprobe}")
            except Exception as e:
                _kana_silent_exc('core:L4741', e)
                pass
        cmd = [
            ffprobe,
            "-v", "error",
            "-select_streams", "v",
            "-show_entries",
            "stream=index,width,height,display_width,display_height,"
            "sample_aspect_ratio,display_aspect_ratio,disposition",
            "-of", "json",
            str(vp),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if res.returncode != 0:
            if bool(globals().get("VERBOSE", False)):
                try:
                    msg = (res.stderr or "").strip()
                    if msg:
                        note("ffprobe エラー（DAR取得）: " + msg.splitlines()[0])
                except Exception as e:
                    _kana_silent_exc('core:L4761', e)
                    pass
            return None

        data = json.loads(res.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return None

        # 添付サムネ（attached_pic）を避けつつ、解像度が大きい映像ストリームを優先
        best = None
        best_area = -1
        for st in streams:
            try:
                w = int(st.get("width") or 0)
                h = int(st.get("height") or 0)
            except Exception:
                w, h = 0, 0
            if w <= 0 or h <= 0:
                continue
            disp = st.get("disposition") or {}
            try:
                if int(disp.get("attached_pic") or 0) == 1:
                    continue
            except Exception as e:
                _kana_silent_exc('core:L4785', e)
                pass
            area = w * h
            if area > best_area:
                best = st
                best_area = area
        if best is None:
            best = streams[0] or {}

        try:
            w = int(best.get("width") or 0)
            h = int(best.get("height") or 0)
        except Exception:
            w, h = 0, 0

        # まず display_width/display_height（Matroska 等で有効なことがある）を優先
        try:
            dw = int(best.get("display_width") or 0)
            dh = int(best.get("display_height") or 0)
        except Exception:
            dw, dh = 0, 0

        sar_s = str(best.get("sample_aspect_ratio") or "")
        dar_s = str(best.get("display_aspect_ratio") or "")

        ratio: Optional[float] = None

        if dw > 0 and dh > 0:
            ratio = float(dw) / float(dh)
        else:
            sar_num, sar_den = _parse_ratio_str(sar_s)
            if w > 0 and h > 0 and sar_num > 0 and sar_den > 0:
                # DAR = (w * SAR) / h
                ratio = (float(w) * float(sar_num)) / (float(h) * float(sar_den))
            else:
                dar_num, dar_den = _parse_ratio_str(dar_s)
                if dar_num > 0 and dar_den > 0:
                    ratio = float(dar_num) / float(dar_den)

        if ratio is None:
            return None
        if ratio <= 0.05 or ratio >= 20.0:
            return None

        _VIDEO_DAR_CACHE[key] = float(ratio)
        return _VIDEO_DAR_CACHE[key]
    except Exception:
        return None

def _frame_score_from_bgr(frame_bgr, mode: str) -> float:
    # best_* 用の簡易スコア。opencv の ndarray を想定。
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bright = float(np.mean(gray))
        # シャープさ：ラプラシアン分散（大きいほど細部がある）
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharp = float(lap.var())
        if mode == "best_bright":
            return bright
        if mode == "best_sharp":
            return sharp
        # best_combo（明るさとシャープさを混ぜる）
        return bright * 0.35 + sharp * 0.65
    except Exception:
        return 0.0

def _extract_video_frames(vp: Path, max_frames: int) -> List[Path]:
    # 動画 vp からフレームを抽出してキャッシュへ保存し、ファイルパスを返す
    global _VIDEO_WARNED_NO_CV2
    if max_frames <= 0:
        return []

    if not bool(globals().get("VIDEO_SCAN_ENABLE", False)):
        return []

    # opencv が無ければスキップ
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        if not _VIDEO_WARNED_NO_CV2:
            _VIDEO_WARNED_NO_CV2 = True
            note(_lang("動画フレーム抽出をスキップしました（opencv-python が見つかりません）。","Video frame extraction skipped (opencv-python not found)."))
        return []

    try:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            return []
    except Exception:
        return []

    mode = str(globals().get("VIDEO_FRAME_SELECT_MODE", "random")).strip().lower()
    candidates = int(globals().get("VIDEO_FRAME_SCORE_CANDIDATES", 16) or 16)
    max_dim = int(globals().get("VIDEO_FRAME_MAX_DIM", 1280) or 1280)
    rng = _video_rng_for_file(vp)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)


    vp_name_l = vp.name.lower()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    # 注: 一部のコンテナ/コーデックでは OpenCV の CAP_PROP_FRAME_COUNT が不正になることがある。
    # その場合、フレーム番号ベースの seek だと動画の後半に到達できず、タイムラインが前半に偏る。
    # AVI_RATIO (0.0-1.0) を使って「動画全体に対する相対位置」で seek するフォールバックを用意する。
    frame_count_raw = int(frame_count)
    _use_ratio_seek = False
    _avi_ratio_prop = getattr(cv2, "CAP_PROP_POS_AVI_RATIO", None)
    if _avi_ratio_prop is not None:
        end_ms_probe = 0.0
        try:
            cap_probe = cv2.VideoCapture(str(vp))
            cap_probe.set(_avi_ratio_prop, 0.999)
            ok_probe, fr_probe = cap_probe.read()
            if ok_probe and (fr_probe is not None):
                end_ms_probe = float(cap_probe.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            cap_probe.release()
        except Exception:
            end_ms_probe = 0.0
        if end_ms_probe > 0.0:
            # フレーム数ヒント（fps が取れる場合のみ）
            if fps > 0.0:
                frame_hint = int(round((end_ms_probe / 1000.0) * fps))
            else:
                frame_hint = 0
            if frame_hint > frame_count_raw:
                frame_count = frame_hint
            # 推定終端時刻が大きく乖離していれば、frame_count の信頼性が低いと判断
            if (frame_count_raw <= 0) or (fps <= 0.0):
                _use_ratio_seek = True
            else:
                t_est_end = (max(frame_count_raw - 1, 0) / fps) * 1000.0
                if (t_est_end > 0.0) and (end_ms_probe > max(5000.0, t_est_end * 1.35)):
                    _use_ratio_seek = True
            if _use_ratio_seek and VERBOSE:
                print(f"  • NOTE: OpenCV frame_count unreliable? raw={frame_count_raw} → hint={frame_count} (end≈{end_ms_probe/1000.0:.2f}s) => seek=AVI_RATIO")

    # コンテナ情報（SAR/DAR）に基づく表示アス比補正（任意）
    dar = _probe_video_display_aspect(vp)

    dar_applied = False
    if dar is not None and bool(globals().get("VERBOSE", False)):
        try:
            if bool(globals().get("VERBOSE", False)) and bool(globals().get("VIDEO_DAR_LOG", False)):
                note(f"動画DAR: {dar:.6f}  {vp.name}")
        except Exception as e:
            _kana_silent_exc('core:L4932', e)
            pass
    def _seek_read(frame_index: int):
        """指定フレームへ seek して 1 枚読む（失敗時は複数のフォールバックを試す）。"""
        fi0 = max(0, int(frame_index))

        def _try_read(setter):
            try:
                setter()
            except Exception:
                return None
            ok, fr = cap.read()
            if ok and fr is not None:
                return fr
            return None

        # 1) まずは既定の seek（AVI_RATIO を使う設定ならそれを優先）
        if _use_ratio_seek and (_avi_ratio_prop is not None) and (frame_count > 1):
            ratio0 = float(fi0) / float(frame_count - 1)
            fr = _try_read(lambda: cap.set(_avi_ratio_prop, max(0.0, min(1.0, ratio0))))
        else:
            fr = _try_read(lambda: cap.set(cv2.CAP_PROP_POS_FRAMES, fi0))
        if fr is not None:
            return fr

        # 2) OpenCV の動画によっては「末尾/後半」の seek が不安定になるので、少し手前へ戻して再試行
        for back in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144):
            bfi = max(0, fi0 - int(back))
            fr = _try_read(lambda bfi=bfi: cap.set(cv2.CAP_PROP_POS_FRAMES, bfi))
            if fr is not None:
                return fr

        # 3) まだだめなら AVI_RATIO を試す（未使用だった場合の保険）
        if (_avi_ratio_prop is not None) and (frame_count > 1):
            ratio1 = float(fi0) / float(frame_count - 1)
            fr = _try_read(lambda: cap.set(_avi_ratio_prop, max(0.0, min(1.0, ratio1))))
            if fr is not None:
                return fr


        # 4) それでもだめなら「キャプチャを開き直して1回だけ」読んでみる（特定のAVI対策）
        try:
            cap2 = cv2.VideoCapture(str(vp))
            if cap2 is not None and cap2.isOpened():
                if _use_ratio_seek and (_avi_ratio_prop is not None) and (frame_count > 1):
                    ratio2 = float(fi0) / float(frame_count - 1)
                    try:
                        cap2.set(_avi_ratio_prop, max(0.0, min(1.0, ratio2)))
                    except Exception:
                        cap2.set(cv2.CAP_PROP_POS_FRAMES, fi0)
                else:
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, fi0)

                ok2, fr2 = cap2.read()
                try:
                    cap2.release()
                except Exception as e:
                    _kana_silent_exc('core:L4990', e)
                    pass
                if ok2 and fr2 is not None:
                    return fr2
        except Exception as e:
            _kana_silent_exc('core:L4994', e)
            pass
        return None

    # 取り出したいフレーム index を決める
    idxs: List[int] = []
    if frame_count > 0:
        if mode == "uniform":
            for i in range(max_frames):
                t = (i + 0.5) / float(max_frames)
                idxs.append(min(frame_count - 1, max(0, int(t * frame_count))))
        elif mode in ("scene", "scene_best"):
            # シーン切替を「動画全体から満遍なく」拾う（後半欠けを防ぐ）
            # 1) 動画全体を均等サンプリングして差分(diff)を測る
            # 2) max_frames 個の時間ビンに分け、各ビンで diff 最大の点を代表として採用
            # 3) scene_best は代表点の近傍から明るさ/シャープネスで1枚厳選
            try:
                scan_mult = float(globals().get("VIDEO_SCENE_ANALYZE_MULT", 3.0) or 3.0)
            except Exception:
                scan_mult = 3.0
            scan_mult = max(1.0, min(12.0, scan_mult))

            try:
                thr = float(globals().get("VIDEO_SCENE_THRESHOLD", 0.35) or 0.35)
            except Exception:
                thr = 0.35
            try:
                offset_sec = float(globals().get("VIDEO_SCENE_OFFSET_SEC", 0.4) or 0.4)
            except Exception:
                offset_sec = 0.4
            try:
                best_win_sec = float(globals().get("VIDEO_SCENE_BEST_WINDOW_SEC", 0.8) or 0.8)
            except Exception:
                best_win_sec = 0.8

            fps_est = float(fps) if fps > 0 else 30.0
            offset_frames = int(max(0, round(offset_sec * fps_est)))
            best_win_frames = int(max(0, round(best_win_sec * fps_est)))

            def _sig_from_frame(fr_bgr):
                try:
                    # ざっくり特徴（低解像度 + HSV + エッジ）を作る
                    sm = cv2.resize(fr_bgr, (96, 54), interpolation=cv2.INTER_AREA)
                    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
                    h = hsv[..., 0].astype(np.float32)
                    s = hsv[..., 1].astype(np.float32)
                    v = hsv[..., 2].astype(np.float32)
                    gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
                    mag = cv2.magnitude(gx, gy)
                    # 軽い正規化
                    mag = np.clip(mag, 0.0, 255.0)
                    # 合成（H は循環なので弱め、S/V/edge を強め）
                    sig = (0.10 * (h / 180.0) + 0.35 * (s / 255.0) + 0.35 * (v / 255.0) + 0.20 * (mag / 255.0))
                    return (sig * 255.0).astype(np.uint8)
                except Exception:
                    return None

            def _bright_sharp(fr_bgr):
                try:
                    g = cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2GRAY)
                    luma = float(np.mean(g)) / 255.0
                    sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())
                    return luma, sharp
                except Exception:
                    return 0.5, 0.0

            def _pick_evenly(arr, k):
                if not arr:
                    return []
                if k <= 1:
                    return [arr[len(arr) // 2]]
                if len(arr) <= k:
                    return arr[:]
                idxs2 = [int(round(i * (len(arr) - 1) / float(k - 1))) for i in range(k)]
                out = []
                last = None
                for ii in idxs2:
                    v = arr[ii]
                    if last is None or v != last:
                        out.append(v)
                        last = v
                return out

            def _uniform_points(k):
                if k <= 0:
                    return []
                if frame_count <= 1:
                    return [0] * k
                return [
                    min(frame_count - 1, max(0, int(round((i + 0.5) * (frame_count - 1) / float(k)))))
                    for i in range(k)
                ]

            # サンプリング数（max_frames の数倍だけ調査）
            probe_n = int(max_frames * scan_mult)
            probe_n = max(max_frames, probe_n)
            # 上限（重すぎ防止）
            fc_limit = int(frame_count) if int(frame_count) > 0 else int(probe_n)
            probe_n = min(fc_limit, probe_n, max_frames * 16, 2400)
            probe_n = max(2, int(probe_n))

            # scene/scene_best は「全体調査（probe）」を先に行う
            # 旧: 均等サンプル点へ都度 seek して read
            # 新: シークだらけを避け、順次読み（read + grab）で軽量に走査する
            #     - POS_MSEC / AVI_RATIO が取れる場合は、それを使って進捗も表示する
            try:
                scan_fps_req = float(globals().get("VIDEO_SCENE_SCAN_FPS", 3) or 3)
            except Exception:
                scan_fps_req = 3.0
            scan_fps_req = max(0.2, min(12.0, scan_fps_req))

            # 走査の目標FPS（probe_n を満たす程度まで引き上げる）
            duration_sec = 0.0
            try:
                if float(end_ms_probe) > 0.0:
                    duration_sec = float(end_ms_probe) / 1000.0
            except Exception:
                duration_sec = 0.0
            if duration_sec <= 0.0 and (fps_est > 0.0) and (int(frame_count) > 0):
                duration_sec = float(int(frame_count)) / float(fps_est)
            if duration_sec <= 0.0:
                # 推定できない場合は「probe_n / scan_fps_req」程度の長さを仮定（極端な値を避ける）
                duration_sec = max(1.0, float(probe_n) / max(0.2, scan_fps_req))

            need_fps = float(probe_n) / max(1e-6, duration_sec)
            scan_fps_eff = max(scan_fps_req, need_fps)
            scan_fps_eff = min(12.0, max(0.2, scan_fps_eff))

            # 進捗: タイムライン比で表示（可能なら POS_MSEC / end_ms_probe を使う）
            PROG_TOTAL = 1000
            last_prog = -1

            samples = []  # (frame_index, pos_msec, diff)
            prev_sig = None

            # 順次走査（read + grab）
            try:
                note(_lang("シーン解析: 順次スキャン（analyze 進捗を表示）", "Scene analyze: sequential scan (shows analyze progress)"))
            except Exception as e:
                _kana_silent_exc('core:L5134', e)
                pass
            try:
                cap_scan = cv2.VideoCapture(str(vp))
            except Exception:
                cap_scan = None

            if cap_scan is not None and bool(getattr(cap_scan, "isOpened", lambda: False)()):
                try:
                    fps_scan = float(cap_scan.get(cv2.CAP_PROP_FPS) or 0.0)
                except Exception:
                    fps_scan = 0.0
                if fps_scan <= 0.0:
                    fps_scan = float(fps_est) if fps_est > 0.0 else 30.0

                step = int(max(1, round(fps_scan / max(0.2, scan_fps_eff))))
                # 極端な step を避ける（probe_n が多いときは1に近づく）
                step = int(max(1, min(240, step)))

                # まず1フレーム読んで、以降は grab でスキップする
                while True:
                    ok, fr = cap_scan.read()
                    if not ok or fr is None:
                        break

                    # frame index / time
                    try:
                        pos_frames = int(cap_scan.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                        fi = int(max(0, pos_frames - 1))
                    except Exception:
                        # POS_FRAMES が取れない場合のフォールバック
                        fi = int(samples[-1][0] + step) if samples else 0

                    try:
                        t_ms = float(cap_scan.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                    except Exception:
                        t_ms = 0.0

                    sig = _sig_from_frame(fr)
                    if sig is not None:
                        if prev_sig is None:
                            diff = 0.0
                        else:
                            try:
                                diff = float(np.mean(np.abs(sig - prev_sig))) / 255.0
                            except Exception:
                                diff = 0.0
                        samples.append((int(fi), float(t_ms), float(diff)))
                        prev_sig = sig

                    # analyze 進捗（タイムライン比が取れるならそれを優先）
                    prog = None
                    try:
                        if float(end_ms_probe) > 0.0 and t_ms > 0.0:
                            prog = int(round(PROG_TOTAL * min(1.0, max(0.0, float(t_ms) / float(end_ms_probe)))))
                    except Exception:
                        prog = None
                    if prog is None:
                        # POS_MSEC が使えない場合はフレーム比（推定）
                        try:
                            denom = float(max(1, int(frame_count) - 1))
                            prog = int(round(PROG_TOTAL * min(1.0, max(0.0, float(fi) / denom))))
                        except Exception:
                            prog = None
                    if prog is None:
                        prog = int(min(PROG_TOTAL, (samples[-1][0] // max(1, step)) if samples else 0))

                    if prog != last_prog:
                        last_prog = prog
                        bar(int(max(0, min(PROG_TOTAL, prog))), PROG_TOTAL, prefix="analyze")

                    # スキップ（grab は decode せず軽い）
                    for _ in range(step - 1):
                        if not cap_scan.grab():
                            break
                try:
                    cap_scan.release()
                except Exception as e:
                    _kana_silent_exc('core:L5212', e)
                    pass
            else:
                # 開けない場合は旧方式へフォールバック（seek/read）
                pass

            # analyze を完了表示
            try:
                bar(PROG_TOTAL, PROG_TOTAL, prefix="analyze", final=True)
            except Exception as e:
                _kana_silent_exc('core:L5221', e)
                pass
            # frame_count が不正な場合、順次走査の最終フレームで補正（後半欠け対策）
            try:
                if samples:
                    last_fi = int(max(s[0] for s in samples))
                    if last_fi + 1 > int(frame_count):
                        frame_count = int(last_fi + 1)
            except Exception as e:
                _kana_silent_exc('core:L5230', e)
                pass
            # 順次走査で十分なサンプルが取れない場合は、均等 seek/read へフォールバック
            if len(samples) < 2 and int(frame_count) > 1:
                # 均等サンプリング index
                if probe_n <= 2:
                    probe_idxs = [0, int(frame_count) - 1]
                else:
                    probe_idxs = [
                        min(int(frame_count) - 1, max(0, int(round(i * (int(frame_count) - 1) / float(probe_n - 1)))))
                        for i in range(probe_n)
                    ]
                samples = []
                prev_sig = None
                for fi in probe_idxs:
                    fr = _seek_read(int(fi))
                    if fr is None:
                        continue
                    sig = _sig_from_frame(fr)
                    if sig is None:
                        continue
                    if prev_sig is None:
                        diff = 0.0
                    else:
                        try:
                            diff = float(np.mean(np.abs(sig - prev_sig))) / 255.0
                        except Exception:
                            diff = 0.0
                    samples.append((int(fi), 0.0, float(diff)))
                    prev_sig = sig

            # samples が取れなければ uniform にフォールバック
            if not samples:
                idxs = _uniform_points(max_frames)
            else:
                # 各ビンの「最大 diff」を代表点にする
                best_by_bin = [None] * max_frames  # (diff, fi)
                denom = float(max(1, frame_count - 1))
                for fi, t_ms, diff in samples:
                    # POS_MSEC が使えるなら時間比で、無理ならフレーム比でビン分け
                    try:
                        if (float(end_ms_probe) > 0.0) and (float(t_ms) > 0.0):
                            r = float(t_ms) / float(end_ms_probe)
                        else:
                            r = float(fi) / denom
                    except Exception:
                        r = float(fi) / denom
                    r = min(1.0, max(0.0, r))
                    bi = int(r * max_frames)
                    if bi >= max_frames:
                        bi = max_frames - 1
                    cur = best_by_bin[bi]
                    if (cur is None) or (diff > cur[0]):
                        best_by_bin[bi] = (diff, fi)

                cand = []
                uni = _uniform_points(max_frames)
                for bi in range(max_frames):
                    if best_by_bin[bi] is not None:
                        fi = int(best_by_bin[bi][1])
                        # diff があまりにも低い時は uniform へ寄せる（変化が無い区間対策）
                        if float(best_by_bin[bi][0]) < thr:
                            fi = uni[bi]
                    else:
                        fi = uni[bi]
                    fi = int(min(frame_count - 1, max(0, fi + offset_frames)))
                    cand.append(fi)

                # 重複を落としつつ時間順へ
                cand = sorted(set(cand))

                # 端（0/末尾）を保険で入れる
                if frame_count > 1:
                    cand.append(0)
                    cand.append(frame_count - 1)
                    cand = sorted(set(cand))

                # 目標枚数へ調整（足りないなら uniform で埋める / 多いなら evenly で間引く）
                if len(cand) < max_frames:
                    for fi in uni:
                        if fi not in cand:
                            cand.append(fi)
                            if len(cand) >= max_frames:
                                break
                    cand = sorted(set(cand))
                if len(cand) > max_frames:
                    cand = _pick_evenly(cand, max_frames)

                if mode == "scene_best":
                    # 代表点の近傍から「明るさ/シャープネス」の良い1枚を採用
                    q_lmin = float(globals().get("VIDEO_QUALITY_LUMA_MIN", 0.07) or 0.07)
                    q_lmax = float(globals().get("VIDEO_QUALITY_LUMA_MAX", 0.93) or 0.93)
                    q_smin = float(globals().get("VIDEO_QUALITY_SHARP_MIN", 6.0) or 6.0)
                    q_w = float(globals().get("VIDEO_QUALITY_WEIGHT", 0.55) or 0.55)

                    # scene_best は「代表点の近傍」を追加で評価するため、
                    # scene より抽出前の待ちが増えることがあります（seek/read が多い）。
                    # 進捗（best）を出して“いま何してるか”見えるようにします。
                    try:
                        eval_pts = int(globals().get("VIDEO_SCENE_BEST_EVAL_POINTS", 5) or 5)
                    except Exception:
                        eval_pts = 5
                    eval_pts = int(max(1, min(5, eval_pts)))

                    # フレーム評価のキャッシュ（同じ ii を複数回 seek しないため）
                    q_cache = {}
                    try:
                        if bool(globals().get("VERBOSE", False)):
                            note(_lang("scene_best: 品質厳選（明るさ/シャープ）", "scene_best: quality selection (luma/sharp)"))
                            bar(0, max(1, len(cand)), prefix="best")
                    except Exception as e:
                        _kana_silent_exc('core:L5341', e)
                        pass
                    picked = []
                    for _bi, base_idx in enumerate(cand, 1):
                        end = min(frame_count - 1, int(base_idx) + best_win_frames)
                        if end < int(base_idx):
                            end = int(base_idx)

                        if end == int(base_idx):
                            eval_idxs = [int(base_idx)]
                        else:
                            span = max(1, end - int(base_idx))
                            if eval_pts <= 3:
                                eval_idxs = [
                                    int(base_idx),
                                    min(frame_count - 1, int(base_idx) + max(1, span // 2)),
                                    end,
                                ]
                            else:
                                eval_idxs = [
                                    int(base_idx),
                                    min(frame_count - 1, int(base_idx) + max(1, span // 4)),
                                    min(frame_count - 1, int(base_idx) + max(1, span // 2)),
                                    min(frame_count - 1, int(base_idx) + max(1, span * 3 // 4)),
                                    end,
                                ]
                        eval_idxs = sorted(set(int(x) for x in eval_idxs))

                        best_i = None
                        best_sc = -1e9
                        for ii in eval_idxs:
                            ii = int(ii)
                            if ii in q_cache:
                                _q = q_cache.get(ii)
                                if _q is None:
                                    continue
                                luma, sharp = _q
                            else:
                                fr2 = _seek_read(ii)
                                if fr2 is None:
                                    q_cache[ii] = None
                                    continue
                                luma, sharp = _bright_sharp(fr2)
                                q_cache[ii] = (float(luma), float(sharp))
                            if not (q_lmin <= luma <= q_lmax):
                                continue
                            if sharp < q_smin:
                                continue

                            sc = (q_w * luma) + ((1.0 - q_w) * (sharp / (sharp + 30.0)))
                            if sc > best_sc:
                                best_sc = sc
                                best_i = int(ii)
                        picked.append(best_i if best_i is not None else int(base_idx))
                        try:
                            if bool(globals().get("VERBOSE", False)):
                                bar(_bi, max(1, len(cand)), prefix="best")
                        except Exception as e:
                            _kana_silent_exc('core:L5399', e)
                            pass
                    cand = sorted(set(picked))

                    try:
                        if bool(globals().get("VERBOSE", False)):
                            bar(max(1, len(cand)), max(1, len(cand)), prefix="best", final=True)
                    except Exception as e:
                        _kana_silent_exc('core:L5407', e)
                        pass
                    # 念のため端を入れる
                    if frame_count > 1:
                        cand.append(0)
                        cand.append(frame_count - 1)
                        cand = sorted(set(cand))

                    if len(cand) < max_frames:
                        for fi in uni:
                            if fi not in cand:
                                cand.append(fi)
                                if len(cand) >= max_frames:
                                    break
                        cand = sorted(set(cand))
                    if len(cand) > max_frames:
                        cand = _pick_evenly(cand, max_frames)

                idxs = cand

            # 最終保険：0 枚は絶対に返さない
            if not idxs:
                idxs = _uniform_points(max_frames)

        elif mode.startswith("best_"):
            cnum = min(frame_count, max(candidates, max_frames * 4))
            try:
                cand_idxs = rng.sample(range(frame_count), cnum)
            except Exception:
                cand_idxs = [rng.randrange(0, frame_count) for _ in range(cnum)]
            scored: List[Tuple[float, int]] = []
            for fi in cand_idxs:
                fr = _seek_read(fi)
                if fr is None:
                    continue
                sc = _frame_score_from_bgr(fr, mode)
                scored.append((sc, int(fi)))
            scored.sort(key=lambda x: x[0], reverse=True)
            idxs = [fi for _sc, fi in scored[:max_frames]]
            if not idxs:
                idxs = [rng.randrange(0, frame_count) for _ in range(max_frames)]
        else:
            try:
                idxs = rng.sample(range(frame_count), min(max_frames, frame_count))
            except Exception:
                idxs = [rng.randrange(0, frame_count) for _ in range(max_frames)]
    else:
        # frame_count が取れない場合：先頭から一定間隔で読む（上限を強めに）
        idxs = []
        step = 30
        cur = 0
        for _ in range(max_frames):
            idxs.append(cur)
            cur += step


    # 動画内のフレーム index は常に時間順に整列して安定させる
    try:
        idxs = sorted(int(x) for x in idxs)
    except Exception:
        try:
            idxs = sorted(idxs)
        except Exception as e:
            _kana_silent_exc('core:L5470', e)
            pass
    cache_dir = _video_cache_dir()
    base = _video_cache_key(vp)
    # DAR補正の有無/値が変わったときに古いキャッシュを誤用しないよう、キーに DAR を混ぜる
    if dar is not None and bool(globals().get("VIDEO_ASPECT_FROM_CONTAINER", True)):
        try:
            base = f"{base}_dar{int(round(float(dar) * 1000.0))}"
        except Exception as e:
            _kana_silent_exc('core:L5479', e)
            pass
    saved: List[Path] = []
    # 保存形式（png/jpg）を選択（デフォ png）
    try:
        _vfmt = str(globals().get("VIDEO_FRAME_CACHE_FORMAT", "png") or "png").strip().lower()
        _vfmt = _vfmt.lstrip(".")
    except Exception:
        _vfmt = "png"
    if _vfmt in ("jpg", "jpeg"):
        _vext = "jpg"
    else:
        _vext = "png"
    try:
        _jpg_q = int(globals().get("VIDEO_FRAME_JPEG_QUALITY", 92))
    except Exception:
        _jpg_q = 92
    try:
        _png_c = int(globals().get("VIDEO_FRAME_PNG_COMPRESSION", 3))
    except Exception:
        _png_c = 3


    # 進捗バー（video）を「抽出開始時」に1回だけ表示して、0/total を見せます。
    # これにより、シーン解析→抽出の間に画面が止まって見えるのを避けます。
    try:
        if bool(globals().get('VERBOSE', False)):
            _t = int(globals().get('_VIDEO_EXTRACT_TOTAL', 0) or 0)
            if _t > 0 and not bool(globals().get('_VIDEO_EXTRACT_PROGRESS_STARTED', False)):
                _d0 = int(globals().get('_VIDEO_EXTRACT_DONE', 0) or 0)
                bar(_d0, _t, prefix="video")
                globals()['_VIDEO_EXTRACT_PROGRESS_STARTED'] = True
    except Exception as e:
        _kana_silent_exc('core:L5512', e)
        pass
    for fi in idxs:
        out_path = cache_dir / f"{base}_{int(fi):08d}.{_vext}"
        try:
            if out_path.exists() and out_path.stat().st_size > 0:
                saved.append(out_path)
                # 抽出進捗（フレーム枚数）を更新（バーは1本、prefixに動画番号）
                try:
                    _d = int(globals().get('_VIDEO_EXTRACT_DONE', 0) or 0) + 1
                    globals()['_VIDEO_EXTRACT_DONE'] = _d
                    _t = int(globals().get('_VIDEO_EXTRACT_TOTAL', 0) or 0)
                    _vn = int(globals().get('_VIDEO_EXTRACT_VN', 1) or 1)
                    _vi = int(globals().get('_VIDEO_EXTRACT_VIDX', 1) or 1)
                    if bool(globals().get('VERBOSE', False)) and (_t > 0):
                        bar(_d, _t, prefix="video")
                except Exception as e:
                    _kana_silent_exc('core:L5530', e)
                    pass
                _VIDEO_FRAME_SRC[str(out_path)] = f"{VIDEO_KEY_PREFIX}{str(vp)}{VIDEO_KEY_SEP}frame={int(fi)}"
                # 並び替え用メタ（元動画名＋推定タイムスタンプ）
                try:
                    t_ms_est = int(round(float(fi) * 1000.0 / fps)) if fps > 0 else int(fi)
                except Exception:
                    t_ms_est = int(fi)
                try:
                    _VIDEO_FRAME_META[str(out_path)] = (vp_name_l, int(t_ms_est))
                except Exception as e:
                    _kana_silent_exc('core:L5540', e)
                    pass
                continue
        except Exception as e:
            _kana_silent_exc('core:L5543', e)
            pass
        fr = _seek_read(int(fi))
        if fr is None:
            continue

        # 抽出位置のタイムスタンプ(ms)（取得できる場合）
        try:
            t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        except Exception:
            t_ms = 0.0


        try:
            h, w = fr.shape[:2]

            # SAR/DAR（再生時の表示アス比）に基づく補正（ffprobeが使える場合のみ）
            # 例）720x480 の NTSC 映像で DAR=16:9 の場合、幅を 854 付近へ補正して正方画素化する
            if dar and h > 0:
                try:
                    cur = float(w) / float(h)
                    if cur > 0 and abs(cur - float(dar)) / max(1e-9, float(dar)) > 0.01:
                        tw = max(1, int(round(float(h) * float(dar))))
                        if tw != w:
                            interp = cv2.INTER_AREA if tw < w else cv2.INTER_LINEAR
                            fr = cv2.resize(fr, (tw, h), interpolation=interp)
                            dar_applied = True
                            h, w = fr.shape[:2]
                except Exception as e:
                    _kana_silent_exc('core:L5572', e)
                    pass
            m = max(h, w)
            if max_dim > 0 and m > max_dim:
                scale = float(max_dim) / float(m)
                nw = max(1, int(w * scale))
                nh = max(1, int(h * scale))
                fr = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA)
        except Exception as e:
            _kana_silent_exc('core:L5581', e)
            pass
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _imwrite_params = ([int(cv2.IMWRITE_PNG_COMPRESSION), int(_png_c)] if _vext == "png" else [int(cv2.IMWRITE_JPEG_QUALITY), int(_jpg_q)])
            _saved_ok = False
            try:
                # まずは従来どおり（結果互換優先）
                _saved_ok = bool(cv2.imwrite(str(out_path), fr, _imwrite_params))
            except Exception as e:
                _kana_silent_exc('core:L5588_imwrite', e)
                _saved_ok = False

            # Windows環境で日本語パスを含むと cv2.imwrite が失敗することがあるため、
            # 動画フレーム保存だけ Unicode 耐性のある経路で救済する（結果は同一内容の画像）。
            _need_unicode_fallback = (not _saved_ok)
            if not _need_unicode_fallback:
                try:
                    _need_unicode_fallback = (not out_path.exists()) or (out_path.stat().st_size <= 0)
                except Exception:
                    _need_unicode_fallback = True
            if _need_unicode_fallback:
                try:
                    _ext_dot = ".png" if _vext == "png" else ".jpg"
                    _enc_ok, _enc_buf = cv2.imencode(_ext_dot, fr, _imwrite_params)
                    if bool(_enc_ok):
                        with open(out_path, "wb") as _wf:
                            _wf.write(_enc_buf.tobytes())
                        _saved_ok = True
                except Exception as e:
                    _kana_silent_exc('core:L5602_imencode_fallback', e)
                    _saved_ok = False
            if out_path.exists() and out_path.stat().st_size > 0:
                saved.append(out_path)
                # 抽出進捗（フレーム枚数）を更新（バーは1本、prefixに動画番号）
                try:
                    _d = int(globals().get('_VIDEO_EXTRACT_DONE', 0) or 0) + 1
                    globals()['_VIDEO_EXTRACT_DONE'] = _d
                    _t = int(globals().get('_VIDEO_EXTRACT_TOTAL', 0) or 0)
                    _vn = int(globals().get('_VIDEO_EXTRACT_VN', 1) or 1)
                    _vi = int(globals().get('_VIDEO_EXTRACT_VIDX', 1) or 1)
                    if bool(globals().get('VERBOSE', False)) and (_t > 0):
                        bar(_d, _t, prefix="video")
                except Exception as e:
                    _kana_silent_exc('core:L5598', e)
                    pass
                _VIDEO_FRAME_SRC[str(out_path)] = f"{VIDEO_KEY_PREFIX}{str(vp)}{VIDEO_KEY_SEP}frame={int(fi)}"
                # 並び替え用メタ（元動画名＋タイムスタンプ）
                try:
                    t_ms_use = int(round(t_ms)) if (t_ms and t_ms > 0) else (int(round(float(fi) * 1000.0 / fps)) if fps > 0 else int(fi))
                except Exception:
                    t_ms_use = int(fi)
                try:
                    _VIDEO_FRAME_META[str(out_path)] = (vp_name_l, int(t_ms_use))
                except Exception as e:
                    _kana_silent_exc('core:L5608', e)
                    pass
        except Exception as e:
            _kana_silent_exc('core:L5610', e)
            continue
    if dar is not None and bool(globals().get("VERBOSE", False)) and bool(globals().get("VIDEO_DAR_LOG", False)):
        try:
            note(f"DAR補正: {'適用' if dar_applied else '不要'}  ({vp.name})")
        except Exception as e:
            _kana_silent_exc('core:L5616', e)
            pass
    try:
        cap.release()
    except Exception as e:
        _kana_silent_exc('core:L5621', e)
        pass
    return saved


def collect_images(paths: Sequence[ImageRef], recursive: bool=True) -> List[ImageRef]:
    out: List[ImageRef] = []
    seen = 0
    zip_files = 0
    zip_entries = 0
    sevenz_files = 0
    sevenz_entries = 0
    rar_files = 0
    rar_entries = 0
    video_files = 0
    video_frames = 0

    video_pending: List[Path] = []
    # スキャン対象の総数を先に数える（進捗表示の分母を安定させる）
    total_scan = 0
    total_scan_known = False
    if bool(globals().get("VERBOSE", False)) and bool(globals().get("SCAN_PRECOUNT_ENABLE", True)):
        try:
            max_cap = int(globals().get("SCAN_PRECOUNT_MAX", 500_000) or 0)
        except Exception:
            max_cap = 500_000
        try:
            for _p in paths:
                try:
                    if isinstance(_p, str) and (_p.startswith(ZIP_KEY_PREFIX) or _p.startswith(SEVENZ_KEY_PREFIX) or _p.startswith(RAR_KEY_PREFIX)):
                        # 既に zip:// 形式ならカウントしない（ここではファイル走査のみ）
                        continue
                    pp = Path(str(_p))
                    if pp.is_dir():
                        for _, __, files in os.walk(pp):
                            total_scan += len(files)
                            if max_cap > 0 and total_scan >= max_cap:
                                total_scan = 0
                                total_scan_known = False
                                raise StopIteration
                    else:
                        # ファイル/存在しないパスでも 1 として数える（表示目的）
                        total_scan += 1
                        if max_cap > 0 and total_scan >= max_cap:
                            total_scan = 0
                            total_scan_known = False
                            raise StopIteration
                except StopIteration:
                    raise
                except Exception:
                    # 変なパスは無視
                    continue
            if total_scan > 0:
                total_scan_known = True
        except StopIteration:
            pass
        except Exception:
            total_scan = 0
            total_scan_known = False

    def _iter_zip_members(zp: Path):
        nonlocal zip_entries
        try:
            with zipfile.ZipFile(zp) as zf:
                try:
                    members = zf.namelist()
                except Exception:
                    members = []
                if ZIP_SCAN_MAX_MEMBERS_PER_ZIP and len(members) > int(ZIP_SCAN_MAX_MEMBERS_PER_ZIP):
                    members = members[: int(ZIP_SCAN_MAX_MEMBERS_PER_ZIP)]

                for name in members:
                    if not name or name.endswith("/"):
                        continue
                    if ZIP_SCAN_SKIP_HIDDEN:
                        # __MACOSX/ や 隠しファイルっぽいものをざっくり除外
                        if name.startswith("__MACOSX/") or "/." in name:
                            continue
                    lower = name.lower()
                    if any(lower.endswith(ext) for ext in IMAGE_EXTS):
                        out.append(_zip_key(zp, _zip_member_display_name(name)))
                        zip_entries += 1
        except Exception:
            return

    def _iter_7z_members(zp: Path):
        nonlocal sevenz_entries
        try:
            try:
                import py7zr  # type: ignore
            except Exception as e:
                _warn_exc_once(e)
                return
            with py7zr.SevenZipFile(str(zp), mode="r") as zf:
                names = []
                try:
                    names = list(zf.getnames())
                except Exception:
                    try:
                        infos = zf.list()
                        names = [getattr(i, "filename", "") for i in infos]
                    except Exception:
                        names = []
                if ZIP_SCAN_MAX_MEMBERS_PER_ZIP and len(names) > int(ZIP_SCAN_MAX_MEMBERS_PER_ZIP):
                    names = names[: int(ZIP_SCAN_MAX_MEMBERS_PER_ZIP)]
                for name in names:
                    if not name or str(name).endswith("/"):
                        continue
                    if ZIP_SCAN_SKIP_HIDDEN:
                        n2 = str(name)
                        if n2.startswith("__MACOSX/") or "/." in n2:
                            continue
                    lower = str(name).lower()
                    if any(lower.endswith(ext) for ext in IMAGE_EXTS):
                        out.append(_7z_key(zp, str(name)))
                        sevenz_entries += 1
        except Exception as e:
            _warn_exc_once(e)
            return

    def _iter_rar_members(zp: Path):
        nonlocal rar_entries
        try:
            try:
                import rarfile  # type: ignore
            except Exception as e:
                _warn_exc_once(e)
                return
            with rarfile.RarFile(str(zp)) as rf:
                try:
                    infos = rf.infolist()
                except Exception:
                    infos = []
                if ZIP_SCAN_MAX_MEMBERS_PER_ZIP and len(infos) > int(ZIP_SCAN_MAX_MEMBERS_PER_ZIP):
                    infos = infos[: int(ZIP_SCAN_MAX_MEMBERS_PER_ZIP)]
                for info in infos:
                    try:
                        name = getattr(info, "filename", "")
                    except Exception:
                        name = ""
                    if not name or str(name).endswith("/"):
                        continue
                    if ZIP_SCAN_SKIP_HIDDEN:
                        if str(name).startswith("__MACOSX/") or "/." in str(name):
                            continue
                    lower = str(name).lower()
                    if any(lower.endswith(ext) for ext in IMAGE_EXTS):
                        out.append(_rar_key(zp, str(name)))
                        rar_entries += 1
        except Exception as e:
            _warn_exc_once(e)
            return

    def _add_video_frames(vp: Path, max_per: int) -> None:
        """vp から max_per 枚までフレーム抽出して out に追加する（失敗時は黙ってスキップ）"""
        nonlocal video_files, video_frames
        try:
            max_per = int(max_per)
        except Exception:
            max_per = 0
        if max_per <= 0:
            return
        frames = _extract_video_frames(vp, max_per)
        if frames:
            # 「触った」動画数は、実際に抽出が試みられたものだけ数える
            video_files += 1
            for fp in frames:
                out.append(fp)
                video_frames += 1

    # 動画の抽出枚数（固定／自動）
    try:
        _vpp_raw = globals().get("VIDEO_FRAMES_PER_VIDEO", 0)
        _vpp = int(_vpp_raw) if _vpp_raw is not None else 0
    except Exception:
        _vpp = 0
    _video_auto = bool(globals().get("VIDEO_SCAN_ENABLE", False)) and (_vpp <= 0)

    for p in paths:
        if not isinstance(p, Path):
            p = Path(p)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            for fp in it:
                if fp.is_file():
                    suf = fp.suffix.lower()
                    if suf in IMAGE_EXTS:
                        out.append(fp)
                    elif bool(globals().get("VIDEO_SCAN_ENABLE", False)) and (suf in VIDEO_SCAN_EXTS):
                        video_pending.append(fp) if _video_auto else _add_video_frames(fp, _vpp)
                    elif bool(globals().get("ZIP_SCAN_ENABLE", False)) and (suf in ZIP_SCAN_EXTS):
                        if zip_files < int(ZIP_SCAN_MAX_ZIPS):
                            zip_files += 1
                            _iter_zip_members(fp)
                    elif bool(globals().get("SEVENZ_SCAN_ENABLE", False)) and (suf in SEVENZ_SCAN_EXTS):
                        if sevenz_files < int(ZIP_SCAN_MAX_ZIPS):
                            sevenz_files += 1
                            _iter_7z_members(fp)
                    elif bool(globals().get("RAR_SCAN_ENABLE", False)) and (suf in RAR_SCAN_EXTS):
                        if rar_files < int(ZIP_SCAN_MAX_ZIPS):
                            rar_files += 1
                            _iter_rar_members(fp)
                seen += 1
                if VERBOSE:
                    bar(seen, total_scan if total_scan_known else 0, prefix="scan   ")
        elif p.is_file():
            suf = p.suffix.lower()
            if suf in IMAGE_EXTS:
                out.append(p)
            elif bool(globals().get("VIDEO_SCAN_ENABLE", False)) and (suf in VIDEO_SCAN_EXTS):
                video_pending.append(p) if _video_auto else _add_video_frames(p, _vpp)
            elif bool(globals().get("ZIP_SCAN_ENABLE", False)) and (suf in ZIP_SCAN_EXTS):
                if zip_files < int(ZIP_SCAN_MAX_ZIPS):
                    zip_files += 1
                    _iter_zip_members(p)
            elif bool(globals().get("SEVENZ_SCAN_ENABLE", False)) and (suf in SEVENZ_SCAN_EXTS):
                if sevenz_files < int(ZIP_SCAN_MAX_ZIPS):
                    sevenz_files += 1
                    _iter_7z_members(p)
            elif bool(globals().get("RAR_SCAN_ENABLE", False)) and (suf in RAR_SCAN_EXTS):
                if rar_files < int(ZIP_SCAN_MAX_ZIPS):
                    rar_files += 1
                    _iter_rar_members(p)
            seen += 1

    # スキャン表示は「走査したファイル数」を基準にする（抽出フレームで分母が跳ねないように）
    if VERBOSE:
        if total_scan_known and total_scan > 0:
            bar(seen, total_scan, prefix="scan   ", final=True)
        else:
            bar(seen, seen, prefix="scan   ", final=True)

    # スキャン完了の枠表示は省略（進捗バーの最終行で完了が分かるため）

    # 動画フレーム（自動配分）の抽出はスキャン後にまとめて行う
    if bool(globals().get("VIDEO_SCAN_ENABLE", False)) and _video_auto and video_pending:
        try:
            import math as _math
            # 目標枚数の推定（grid は ROWS×COLS を優先）
            try:
                _rows = int(globals().get("ROWS", 0) or 0)
                _cols = int(globals().get("COLS", 0) or 0)
            except Exception:
                _rows, _cols = 0, 0
            try:
                _count = int(globals().get("COUNT", 0) or 0)
            except Exception:
                _count = 0
            _target = 0
            if _rows > 0 and _cols > 0:
                _target = _rows * _cols
            if _count > _target:
                _target = _count

            # 既に候補が十分ある場合も、最低枚数は抽出する
            try:
                _minpv = int(globals().get("VIDEO_FRAMES_AUTO_MIN_PER_VIDEO", 1) or 1)
            except Exception:
                _minpv = 1
            try:
                _margin = float(globals().get("VIDEO_FRAMES_AUTO_MARGIN", 1.15) or 1.15)
            except Exception:
                _margin = 1.15
            try:
                _cap = int(globals().get("VIDEO_FRAMES_PER_VIDEO_CAP", 0) or 0)
            except Exception:
                _cap = 0

            nvid = len(video_pending)
            # 追加で必要な枚数（不足分だけ増やす。0 でも最低 _minpv）
            need = max(0, (_target - len(out))) if _target > 0 else 0
            # dHash などで間引かれても不足しにくいように少し多めに
            base = max(1, int(_math.ceil((max(1, need) * _margin) / max(1, nvid)))) if _target > 0 else _minpv
            per_video = max(_minpv, base)
            if _cap > 0:
                per_video = min(per_video, _cap)

            # 動画ファイルは名前順で処理して安定させる
            try:
                video_pending.sort(key=lambda p: str(getattr(p, "name", p)).lower())
            except Exception as e:
                _kana_silent_exc('core:L5902', e)
                pass
            banner(_lang("動画フレーム抽出（自動配分）","Video frame extraction (auto allocation)"))
            note(f"videos: {nvid} / target≈{_target} / per_video={per_video} (need={need})")

            # 進捗は「動画本数」ではなく「抽出フレーム枚数」で更新する（進捗バーは1本）
            _total_frames = int(max(0, per_video * nvid))
            globals()["_VIDEO_EXTRACT_TOTAL"] = _total_frames
            globals()["_VIDEO_EXTRACT_DONE"] = 0
            globals()["_VIDEO_EXTRACT_VN"] = nvid
            globals()["_VIDEO_EXTRACT_VIDX"] = 1
            globals()["_VIDEO_EXTRACT_PROGRESS_STARTED"] = False
            for i, vp in enumerate(video_pending, 1):
                globals()["_VIDEO_EXTRACT_VIDX"] = i
                _add_video_frames(vp, per_video)

            if VERBOSE:
                # 最終行を確定（例: 75/75）
                _done = int(globals().get("_VIDEO_EXTRACT_DONE", 0) or 0)
                bar(_done, max(1, _total_frames), prefix="video", final=True)
        except Exception as e:
            _warn_exc_once(e)
            pass
    # サマリ（1行・安定フォーマット）
    try:
        zip_on = bool(globals().get("ZIP_SCAN_ENABLE", False))
    except Exception:
        zip_on = False
    try:
        sevenz_on = bool(globals().get("SEVENZ_SCAN_ENABLE", False))
    except Exception:
        sevenz_on = False
    try:
        rar_on = bool(globals().get("RAR_SCAN_ENABLE", False))
    except Exception:
        rar_on = False
    try:
        vid_on = bool(globals().get("VIDEO_SCAN_ENABLE", False))
    except Exception:
        vid_on = False
    parts = [f"Candidates: {len(out)}"]
    # Archives summary: show only formats that actually contributed entries.
    _arch = _fmt_archives_summary(zip_entries, zip_files, sevenz_entries, sevenz_files, rar_entries, rar_files)
    if _arch:
        parts.append(f"Archives: {_arch}")
    # Video summary: show only when video files exist in inputs.
    try:
        _vfound = int(len(video_pending))
    except Exception:
        _vfound = 0
    if _vfound > 0:
        parts.append(f"Video: found={_vfound}, frames={video_frames}")
    # Always print scan summary line (guard against progress-bar overwrite)
    try:
        _ensure_newline_if_bar_active()
    except Exception as e:
        _kana_silent_exc('core:L5958', e)
        pass
    note(" | ".join(parts))
    return out

def reorder_global_spectral_hilbert(paths: list[Path], objective: str = "min") -> list[Path]:
    """全画像を1列に並べる前処理。色→2D→ヒルベルト順。
       objective="max" の場合は蛇行反転で“バラけ”を少し増やす。"""
    n = len(paths)
    if n <= 1: return paths

    # 進捗：特徴量→射影→ランク
    banner(_lang("前処理: Global order（spectral→hilbert）","Preprocess: Global order (spectral→hilbert)"))
    vecs = []
    if VERBOSE: bar(0, n, prefix="feat   ", final=False)
    for i, p in enumerate(paths, 1):
        vecs.append(_avg_lab_vector(p))
        if VERBOSE: bar(i, n, prefix="feat   ", final=(i == n))

    if np is not None:
        X = np.array(vecs, dtype=float)
        X = X - X.mean(axis=0, keepdims=True)
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        P = X @ VT[:2].T
        mins = P.min(axis=0); maxs = P.max(axis=0)
        rng = np.where(maxs > mins, maxs - mins, 1.0)
        Q = (P - mins) / rng
        xs = (Q[:, 0] * 1023.999).astype(int).tolist()
        ys = (Q[:, 1] * 1023.999).astype(int).tolist()
        if VERBOSE: bar(1, 1, prefix="pca    ", final=True)
    else:
        Ls = [v[0] for v in vecs]; As = [v[1] for v in vecs]
        loL, hiL = min(Ls), max(Ls); loA, hiA = min(As), max(As)
        dL = (hiL - loL) if hiL > loL else 1.0
        dA = (hiA - loA) if hiA > loA else 1.0
        xs = [int(((L - loL) / dL) * 1023.999) for L in Ls]
        ys = [int(((A - loA) / dA) * 1023.999) for A in As]
        if VERBOSE: bar(1, 1, prefix="project", final=True)

    ranks = []
    if VERBOSE: bar(0, n, prefix="rank   ", final=False)
    for i in range(n):
        ranks.append((_hilbert_index(xs[i], ys[i], order=10), i))
        if VERBOSE: bar(i + 1, n, prefix="rank   ", final=((i + 1) == n))
    ranks.sort(key=lambda x: x[0])
    order = [i for _, i in ranks]

    if objective == "max":
        # ジグザグ反転で隣接色差を少し上げる
        half = len(order) // 2
        order = order[:half] + list(reversed(order[half:]))

    return [paths[i] for i in order]


def _rotate_order_move_max_jump_to_end(paths: list[Path]) -> list[Path]:
    """
    Lab で並べ替えた順序を回転し、最大の隣接ジャンプが「最後↔最初」の境界に来るようにします。

        1D の色順を 2D の空間充填曲線（ヒルベルト等）へ割り当てるとき、途中で「途切れ（break）」が見えるのを軽減できることがあります。
    """
    n = len(paths)
    if n < 4:
        return paths
    try:
        vecs = [_avg_lab_vector(p) for p in paths]
    except Exception:
        return paths
    max_i = 0
    max_d = -1.0
    for i in range(n - 1):
        dL = vecs[i][0] - vecs[i + 1][0]
        dA = vecs[i][1] - vecs[i + 1][1]
        dB = vecs[i][2] - vecs[i + 1][2]
        d = dL * dL + dA * dA + dB * dB
        if d > max_d:
            max_d = d
            max_i = i
    rot = max_i + 1
    if rot <= 0 or rot >= n:
        return paths
    return paths[rot:] + paths[:rot]

def reorder_global_anneal(paths: list[Path], objective: str = "max", iters: int = 20000, reheats: int = 1, seed: int | str = 0) -> list[Path]:
    """全画像を1列の並びで最適化（ΣΔ色）。_optimize_sequence を再利用。

    表示/UIの統一のため、iters は「総ステップ数」として扱います（= 進捗は iters/iters）。
    reheats は「再スタート回数（試行回数）」として扱い、総ステップ iters を各試行へ均等配分します。
    """
    n = len(paths)
    if n <= 2:
        return list(paths)

    # seed を解決（"random" も許容）
    try:
        base_seed = secrets.randbits(32) if seed == "random" else int(seed)
    except Exception:
        base_seed = secrets.randbits(32)

    obj = str(objective).lower().strip()
    if obj not in ("min", "max"):
        obj = "max"

    rh = max(1, int(reheats))
    total_steps = int(max(1, int(iters)))

    # iters を均等配分（total==iters を保証）
    base = total_steps // rh
    rem = total_steps % rh

    # 特徴量（平均 Lab）
    vecs = [_avg_lab_vector(p) for p in paths]

    # ベース順（入力の順）を基準に改善量を出す
    base_order = list(range(n))
    init_sum = _seq_adj_sum(base_order, vecs)

    best_order = list(base_order)
    best_sum = float(init_sum)
    accepted_total = 0

    try:
        banner(_lang("前処理: Global order（anneal/hill）", "Preprocess: Global order (anneal/hill)"))
        note(f"Objective: {obj} / steps={int(total_steps)} reheats={int(rh)}")
    except Exception as e:
        _warn_exc_once(e)
        pass

    # 進捗は iters/iters で統一（reheats>1 でも分母は増やさない）
    prog = {"i": 0, "n": max(1, int(total_steps)), "prefix": "anneal"}

    for r in range(rh):
        # reheats で seed を少しずつずらす
        try:
            s = (int(base_seed) + int(r) * 10007) & 0xFFFFFFFF
        except Exception:
            s = secrets.randbits(32)
        rng = random.Random(s)

        run_steps = base + (1 if r < rem else 0)
        if run_steps <= 0:
            continue

        order = list(base_order)

        order, ini, fin, imp, acc = _optimize_sequence(order, vecs, int(run_steps), obj, rng, prog)
        accepted_total += int(acc)

        # 目的に応じて「より良い」方を採用
        try:
            fin_val = float(fin)
        except Exception:
            fin_val = float(_seq_adj_sum(order, vecs))

        better = (fin_val > best_sum) if (obj == "max") else (fin_val < best_sum)
        if better or (r == 0 and rh == 1):
            best_order = list(order)
            best_sum = float(fin_val)

    # 統一表示（mosaic の anneal と同じ形式）
    try:
        _note_opt_improve_sumdelta(float(init_sum), float(best_sum), obj, int(accepted_total), int(total_steps))
    except Exception:
        pass

    return [paths[i] for i in best_order]


# -----------------------------------------------------------------------------
# サブセクション: 簡易“美選抜”と重複除去
# -----------------------------------------------------------------------------
def grayscale_small(im: Image.Image, size=(192,192)) -> Image.Image:
    g=im.convert("L")
    if max(im.size) > max(size):
        g = g.resize(size, Resampling.LANCZOS)
    return g

def dhash64(img: Image.Image, hash_size=8) -> int:
    g = img.convert("L").resize((hash_size+1, hash_size))
    bits=0; i=0
    for y in range(hash_size):
        for x in range(hash_size):
            if g.getpixel((x,y)) < g.getpixel((x+1,y)):
                bits |= (1<<i)
            i+=1
    return bits

def hamming(a:int,b:int)->int: return (a^b).bit_count()


# -----------------------------------------------------------------------------
# サブセクション: dHash（近似ハッシュ）: キャッシュ & 先読み
# -----------------------------------------------------------------------------

# =============================================================================
# セクション: 永続キャッシュ（dHash／Aesthetic／Lab／Face など）
# =============================================================================
_DHASH_CACHE = {}            # { norm_abs_path: {"mtime_ns":int, "size":int, "dhash":int, "t":float}, ... }
_DHASH_CACHE_DIRTY = False
_DHASH_CACHE_LOADED = False
_DHASH_CACHE_LOCK = threading.Lock()

_DHASH_PREFETCH_EXEC = None  # ThreadPoolExecutor
_DHASH_PREFETCH_FUTURES = {} # { norm_abs_path: Future }

def _dhash_norm_path(p: Any) -> str:
    # キャッシュのキーとして使うパス文字列（OS差を吸収）
    s = str(p)
    if _is_zip_key(s):
        try:
            zpath, member = _zip_parse(s)
            z_norm = os.path.normcase(os.path.abspath(zpath))
            return f"{ZIP_KEY_PREFIX}{z_norm}{ZIP_KEY_SEP}{member}"
        except Exception:
            return s
    if _is_7z_key(s):
        try:
            apath, member = _7z_parse(s)
            a_norm = os.path.normcase(os.path.abspath(apath))
            return f"{SEVENZ_KEY_PREFIX}{a_norm}{ZIP_KEY_SEP}{member}"
        except Exception:
            return s
    if _is_rar_key(s):
        try:
            apath, member = _rar_parse(s)
            a_norm = os.path.normcase(os.path.abspath(apath))
            return f"{RAR_KEY_PREFIX}{a_norm}{ZIP_KEY_SEP}{member}"
        except Exception:
            return s
    try:
        return os.path.normcase(os.path.abspath(s))
    except Exception:
        return s

def _is_video_frame_cache_path(path_str: str) -> bool:
    """
    動画抽出フレーム（VIDEO_FRAME_CACHE_DIR 配下）は、永続dHashキャッシュへ保存しない。
    - 動画フレームは一時キャッシュを消す運用が多く、永続キャッシュに残すと死んだパスが増えやすい
    """
    try:
        if not path_str:
            return False
        if _is_zip_key(path_str) or _is_7z_key(path_str) or _is_rar_key(path_str):
            return False
    except Exception:
        return False

    # 生成済みのフレームを追跡している場合は優先判定
    try:
        if str(path_str) in _VIDEO_FRAME_SRC:
            return True
    except Exception as e:
        _kana_silent_exc('core:L6144', e)
        pass
    try:
        cache_dir = str(globals().get("VIDEO_FRAME_CACHE_DIR", "") or "")
    except Exception:
        cache_dir = ""
    if not cache_dir:
        return False

    try:
        cd = Path(cache_dir).expanduser()
        try:
            cd = cd.resolve()
        except Exception as e:
            _kana_silent_exc('core:L6158', e)
            pass
        p = Path(str(path_str))
        try:
            p = p.resolve()
        except Exception as e:
            _kana_silent_exc('core:L6163', e)
            pass
        cd_s = os.path.normcase(str(cd))
        p_s = os.path.normcase(str(p))
        if not cd_s:
            return False
        # ファイルがキャッシュディレクトリ配下にあるか
        return p_s.startswith(cd_s + os.sep)
    except Exception:
        return False


def _dhash_cache_path() -> Path:
    # 永続キャッシュの保存先。DHASH_CACHE_FILE が空なら、この .py と同じ場所に作る。
    try:
        cf = str(globals().get("DHASH_CACHE_FILE", "") or "")
    except Exception:
        cf = ""
    if cf:
        try:
            return Path(cf).expanduser()
        except Exception:
            return Path(cf)
    try:
        # 例: kana_wallpaper_unified_final.py → kana_wallpaper_unified_final.dhash_cache.json
        return Path(__file__).resolve().with_suffix(".dhash_cache.json")
    except Exception:
        return Path.cwd() / "kana_wallpaper.dhash_cache.json"

def _dhash_stat(path_str: str) -> Tuple[Optional[int], Optional[int]]:
    # (mtime_ns, size) を返す。失敗時は (None, None)。
    try:
        if _is_zip_key(path_str):
            return _zip_member_stat(str(path_str))
        if _is_7z_key(path_str) or _is_rar_key(path_str):
            return _arch_member_stat(str(path_str))
    except Exception as e:
        _warn_exc_once(e)
        pass
    try:
        st = Path(path_str).stat()
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
        return int(mtime_ns), int(st.st_size)
    except Exception:
        try:
            import os as _os
            st = _os.stat(path_str)
            mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
            return int(mtime_ns), int(st.st_size)
        except Exception:
            return None, None

def _dhash_cache_load_once() -> None:
    # キャッシュを 1 回だけ読み込む（失敗しても黙って続行）。
    global _DHASH_CACHE_LOADED, _DHASH_CACHE
    if _DHASH_CACHE_LOADED:
        return
    _DHASH_CACHE_LOADED = True
    if not bool(globals().get("DHASH_CACHE_ENABLE", True)):
        return
    path = _dhash_cache_path()
    try:
        if not Path(path).exists():
            # キャッシュファイルが無い場合でも、次回以降のために空のキャッシュを作成できるようにしておく。
            # （DHASH_CACHE_DIRTY を立てて、終了時に _dhash_cache_save() がファイルを書き出します）
            global _DHASH_CACHE_DIRTY
            _DHASH_CACHE = {}
            _DHASH_CACHE_DIRTY = True
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _DHASH_CACHE = data

            # 動画フレームキャッシュ由来のエントリは永続dHashキャッシュに残さない
            # （キャッシュ削除運用や一時ファイルのため、死んだパスが増えるのを防ぐ）
            try:
                removed = 0
                for k in list(_DHASH_CACHE.keys()):
                    if _is_video_frame_cache_path(str(k)):
                        _DHASH_CACHE.pop(k, None)
                        removed += 1
                if removed:
                    _DHASH_CACHE_DIRTY = True
            except Exception as e:
                _kana_silent_exc('core:L6247', e)
                pass
    except Exception:
        return

def _dhash_cache_prune_unlocked() -> None:
    # 上限を超えたら古いものから間引く（ロック獲得済み前提）
    try:
        limit = int(globals().get("DHASH_CACHE_MAX", 200_000))
    except Exception:
        limit = 200_000
    if limit <= 0:
        _DHASH_CACHE.clear()
        return
    n = len(_DHASH_CACHE)
    if n <= limit:
        return
    items = []
    for k, v in _DHASH_CACHE.items():
        t = 0.0
        if isinstance(v, dict):
            try:
                t = float(v.get("t", 0.0))
            except Exception:
                t = 0.0
        items.append((t, k))
    items.sort()
    drop = n - limit
    for _, k in items[:drop]:
        try:
            _DHASH_CACHE.pop(k, None)
        except Exception as e:
            _warn_exc_once(e)
            pass
def _dhash_cache_save() -> None:
    # 必要ならキャッシュを保存（終了時に自動実行）。
    global _DHASH_CACHE_DIRTY
    if not bool(globals().get("DHASH_CACHE_ENABLE", True)):
        return
    if not _DHASH_CACHE_DIRTY:
        return
    path = Path(_dhash_cache_path())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        _warn_exc_once(e)
        pass
    try:
        with _DHASH_CACHE_LOCK:
            _dhash_cache_prune_unlocked()
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(_DHASH_CACHE, ensure_ascii=False), encoding="utf-8")
            try:
                tmp.replace(path)
            except Exception:
                # Windows で replace が失敗する場合のフォールバック
                path.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")
                try:
                    tmp.unlink()
                except Exception as e:
                    _warn_exc_once(e)
                    pass
            _DHASH_CACHE_DIRTY = False
    except Exception:
        return

# 終了時にキャッシュを保存
try:
    atexit.register(_dhash_cache_save)
except Exception as e:
    _warn_exc_once(e)
    pass
def _dhash_compute_from_path(path_str: str) -> Optional[int]:
    # ファイルから dHash(64bit) を計算。失敗時は None。
    try:
        with open_image_safe(path_str) as im:
            return dhash64(grayscale_small(im), 8)
    except Exception:
        return None

def _dhash_prefetch_init():
    # 先読み用のスレッドプールを初期化
    global _DHASH_PREFETCH_EXEC
    if _DHASH_PREFETCH_EXEC is not None:
        return
    if not bool(globals().get("DHASH_PREFETCH_ENABLE", True)):
        return
    try:
        workers = int(globals().get("DHASH_PREFETCH_WORKERS", 4))
    except Exception:
        workers = 4
    workers = max(1, workers)
    try:
        _DHASH_PREFETCH_EXEC = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dhashpref")
    except Exception:
        _DHASH_PREFETCH_EXEC = None

def dhash_prefetch_paths(paths: Sequence[str], limit: Optional[int] = None) -> None:
    # paths の dHash を先読み（キャッシュに無いものだけ）
    if not bool(globals().get("DHASH_PREFETCH_ENABLE", True)):
        return
    _dhash_cache_load_once()
    _dhash_prefetch_init()
    if _DHASH_PREFETCH_EXEC is None:
        return
    try:
        ahead = int(globals().get("DHASH_PREFETCH_AHEAD", 0))
    except Exception:
        ahead = 0
    # DHASH_PREFETCH_AHEAD == 0 は「自動」。COUNT（または ROWS*COLS）と重複しきい値から安全な先読み量を決めます。
    if ahead <= 0:
        try:
            cnt = int(globals().get("COUNT", 0) or 0)
        except Exception:
            cnt = 0
        if cnt <= 0:
            try:
                cnt = int(globals().get("ROWS", 0) or 0) * int(globals().get("COLS", 0) or 0)
            except Exception:
                cnt = 0
        if cnt <= 0:
            cnt = 140
        try:
            thr = int(globals().get("DEDUP_DHASH_THRESHOLD", globals().get("DEDUPE_HAMMING", 4)))  # 旧設定/互換のため、閾値は複数キーから取得（未設定時は4）
        except Exception:
            thr = 4
        # しきい値が大きいほど（除去が強いほど）スキャンが伸びやすいので、先読みを少し増やします。
        if thr <= 6:
            factor = 3.5
        elif thr <= 12:
            factor = 4.5
        else:
            factor = 5.5
        try:
            ahead = int(max(200, min(2000, round(cnt * factor))))
        except Exception:
            ahead = 600
    if limit is None:
        limit = ahead
    else:
        try:
            limit = int(limit)
        except Exception:
            limit = ahead
    n = 0
    for p in paths:
        if n >= limit:
            break
        key = _dhash_norm_path(p)
        with _DHASH_CACHE_LOCK:
            ent = _DHASH_CACHE.get(key)
            if isinstance(ent, dict):
                mns, sz = _dhash_stat(key)
                try:
                    if mns is not None and sz is not None and int(ent.get("mtime_ns", -1)) == int(mns) and int(ent.get("size", -1)) == int(sz):
                        continue
                except Exception as e:
                    _warn_exc_once(e)
                    pass
            if key in _DHASH_PREFETCH_FUTURES:
                continue
            try:
                _DHASH_PREFETCH_FUTURES[key] = _DHASH_PREFETCH_EXEC.submit(_dhash_compute_from_path, key)
            except Exception as e:
                _warn_exc_once(e)
                pass
        n += 1

def dhash64_for_path_cached(p: Union[str, Path]) -> Optional[int]:
    # path の dHash を取得（永続キャッシュ + 先読みを利用）
    _dhash_cache_load_once()
    key = _dhash_norm_path(p)

    # まず永続キャッシュを確認
    if bool(globals().get("DHASH_CACHE_ENABLE", True)):
        mns, sz = _dhash_stat(key)
        with _DHASH_CACHE_LOCK:
            ent = _DHASH_CACHE.get(key)
            if isinstance(ent, dict) and mns is not None and sz is not None:
                try:
                    if int(ent.get("mtime_ns", -1)) == int(mns) and int(ent.get("size", -1)) == int(sz):
                        return int(ent.get("dhash"))
                except Exception as e:
                    _warn_exc_once(e)
                    pass
    # 先読みの結果があればそれを使う（無ければ計算）
    fut = None
    with _DHASH_CACHE_LOCK:
        fut = _DHASH_PREFETCH_FUTURES.get(key)
    if fut is not None:
        try:
            h = fut.result()
        except Exception:
            h = None
        with _DHASH_CACHE_LOCK:
            _DHASH_PREFETCH_FUTURES.pop(key, None)
    else:
        h = _dhash_compute_from_path(key)

    # 保存
    if bool(globals().get("DHASH_CACHE_ENABLE", True)) and h is not None and (not _is_video_frame_cache_path(key)):
        mns, sz = _dhash_stat(key)
        with _DHASH_CACHE_LOCK:
            try:
                _DHASH_CACHE[key] = {"mtime_ns": int(mns or 0), "size": int(sz or 0), "dhash": int(h), "t": float(time.time())}
                global _DHASH_CACHE_DIRTY
                _DHASH_CACHE_DIRTY = True
            except Exception as e:
                _warn_exc_once(e)
                pass
    return h


def _aes_base_score_from_image(im) -> float:
    # aesthetic のベーススコア（ランダム要素を除く）
    g = grayscale_small(im)
    st = ImageStat.Stat(g)
    mean = float(st.mean[0])
    std = float(st.stddev[0])
    sharp = float(ImageStat.Stat(g.filter(ImageFilter.FIND_EDGES)).mean[0])
    return 0.45 * sharp + 0.35 * std + 0.2 * (1.0 - abs(mean - 128.0) / 128.0)  # 簡易『情報量/見栄え』スコア：シャープさ/コントラスト/明るさ偏りの重み


def _aes_cache_get(p):
    # dhash 永続キャッシュを流用して aesthetic のベーススコアを保存/再利用する
    if not bool(globals().get("AESTHETIC_CACHE_ENABLE", True)):
        return None
    if not bool(globals().get("DHASH_CACHE_ENABLE", True)):
        return None

    _dhash_cache_load_once()
    key = _dhash_norm_path(p)
    if _is_video_frame_cache_path(key):
        return

    mns, sz = _dhash_stat(key)
    if mns is None or sz is None:
        return None

    with _DHASH_CACHE_LOCK:
        ent = _DHASH_CACHE.get(key)
        if isinstance(ent, dict):
            try:
                if int(ent.get("mtime_ns", -1)) == int(mns) and int(ent.get("size", -1)) == int(sz) and "aes" in ent:
                    return float(ent.get("aes"))
            except Exception:
                return None
    return None


def _aes_cache_put(p, base) -> None:
    if base is None:
        return
    if not bool(globals().get("AESTHETIC_CACHE_ENABLE", True)):
        return
    if not bool(globals().get("DHASH_CACHE_ENABLE", True)):
        return

    _dhash_cache_load_once()
    key = _dhash_norm_path(p)
    if _is_video_frame_cache_path(key):
        return

    mns, sz = _dhash_stat(key)
    if mns is None or sz is None:
        return

    global _DHASH_CACHE_DIRTY
    with _DHASH_CACHE_LOCK:
        ent = _DHASH_CACHE.get(key)
        if not isinstance(ent, dict):
            ent = {}
        ent["mtime_ns"] = int(mns)
        ent["size"] = int(sz)
        ent["aes"] = float(base)
        ent["t"] = float(time.time())
        _DHASH_CACHE[key] = ent
        _DHASH_CACHE_DIRTY = True


def _lab_cache_get(p):
    # dhash 永続キャッシュを流用して平均Lab（L,a,b）を保存/再利用する
    if not bool(globals().get("LAB_CACHE_ENABLE", True)):
        return None
    if not bool(globals().get("DHASH_CACHE_ENABLE", True)):
        return None

    _dhash_cache_load_once()
    key = _dhash_norm_path(p)
    mns, sz = _dhash_stat(key)
    if mns is None or sz is None:
        return None

    with _DHASH_CACHE_LOCK:
        ent = _DHASH_CACHE.get(key)
        if isinstance(ent, dict):
            try:
                if int(ent.get("mtime_ns", -1)) == int(mns) and int(ent.get("size", -1)) == int(sz) and "lab" in ent:
                    v = ent.get("lab")
                    if isinstance(v, (list, tuple)) and len(v) == 3:
                        return (float(v[0]), float(v[1]), float(v[2]))
            except Exception:
                return None
    return None


def _lab_cache_put(p, vec) -> None:
    if vec is None:
        return
    if not bool(globals().get("LAB_CACHE_ENABLE", True)):
        return
    if not bool(globals().get("DHASH_CACHE_ENABLE", True)):
        return

    _dhash_cache_load_once()
    key = _dhash_norm_path(p)
    mns, sz = _dhash_stat(key)
    if mns is None or sz is None:
        return

    global _DHASH_CACHE_DIRTY
    with _DHASH_CACHE_LOCK:
        ent = _DHASH_CACHE.get(key)
        if not isinstance(ent, dict):
            ent = {}
        ent["mtime_ns"] = int(mns)
        ent["size"] = int(sz)
        ent["lab"] = [float(vec[0]), float(vec[1]), float(vec[2])]
        ent["t"] = float(time.time())
        _DHASH_CACHE[key] = ent
        _DHASH_CACHE_DIRTY = True


def _face_cache_signature() -> str:
    g = globals()
    # v2: AIバックエンド/モデル/しきい値も含める（切替時にキャッシュが混ざらないように）
    try:
        backend = str(g.get("FACE_FOCUS_AI_BACKEND", "")).lower().strip()
    except Exception:
        backend = ""
    try:
        ai_on = 1 if bool(g.get("FACE_FOCUS_AI_ENABLE", False)) else 0
    except Exception:
        ai_on = 0

    def _bn(p):
        # パスは長いので basename で十分（同名が衝突する場合は手動で無効化/削除）
        try:
            import os
            return os.path.basename(str(p))
        except Exception:
            return str(p)

    parts = [
        "v2",
        f"strict={1 if bool(g.get('FACE_FOCUS_STRICT_EYES', True)) else 0}",
        f"allow_low={1 if bool(g.get('FACE_FOCUS_ALLOW_LOW', True)) else 0}",
        f"ratio_min={float(g.get('FACE_FOCUS_FACE_RATIO_MIN', 0.65)):.4f}",
        f"ratio_max={float(g.get('FACE_FOCUS_FACE_RATIO_MAX', 1.60)):.4f}",
        f"eye_min={int(g.get('FACE_FOCUS_EYE_MIN', 1))}",
        f"min_eye_dist={float(g.get('FACE_FOCUS_MIN_EYE_DIST_FRAC', 0.06)):.4f}",
        f"use_profile={1 if bool(g.get('FACE_FOCUS_USE_PROFILE', True)) else 0}",
        f"use_upper={1 if bool(g.get('FACE_FOCUS_USE_UPPER', False)) else 0}",
        f"use_person={1 if bool(g.get('FACE_FOCUS_USE_PERSON', True)) else 0}",
        f"hog_person={1 if bool(g.get('PERSON_FOCUS_HOG_ENABLE', True)) else 0}",
        f"use_saliency={1 if bool(g.get('FACE_FOCUS_USE_SALIENCY', True)) else 0}",
        f"ai_on={ai_on}",
        f"ai_backend={backend}",
        # YOLO
        f"yolo_model={_bn(g.get('FACE_FOCUS_YOLO_MODEL', ''))}",
        f"yolo_conf={float(g.get('FACE_FOCUS_YOLO_CONF', 0.25)):.3f}",
        f"yolo_imgsz={int(g.get('FACE_FOCUS_YOLO_IMGSZ', 1536))}",
        # YuNet
        f"yunet_model={_bn(g.get('FACE_FOCUS_YUNET_MODEL', ''))}",
        f"yunet_score={float(g.get('FACE_FOCUS_YUNET_SCORE', 0.60)):.3f}",
        # AnimeFace cascade
        f"af_xml={_bn(g.get('FACE_FOCUS_ANIMEFACE_CASCADE', ''))}",
        f"af_sf={float(g.get('FACE_FOCUS_ANIMEFACE_SCALE_FACTOR', 1.10)):.3f}",
        f"af_mn={int(g.get('FACE_FOCUS_ANIMEFACE_MIN_NEIGHBORS', 3))}",
        f"af_ms={int(g.get('FACE_FOCUS_ANIMEFACE_MIN_SIZE', 24))}",
    ]
    return "|".join(parts)


def _face_cache_get(p):
    if not bool(globals().get("FACE_CACHE_ENABLE", True)):
        return None
    if not bool(globals().get("DHASH_CACHE_ENABLE", True)):
        return None

    _dhash_cache_load_once()
    key = _dhash_norm_path(p)
    mns, sz = _dhash_stat(key)
    if mns is None or sz is None:
        return None
    sig = _face_cache_signature()

    with _DHASH_CACHE_LOCK:
        ent = _DHASH_CACHE.get(key)
        if not isinstance(ent, dict):
            return None
        try:
            if int(ent.get("mtime_ns", -1)) != int(mns) or int(ent.get("size", -1)) != int(sz):
                return None
            if ent.get("face_sig") != sig:
                return None
        except Exception:
            return None

        out = {"face": None, "upper": None, "person": None, "saliency": None}
        fv = ent.get("face", None)
        uv = ent.get("upper", None)
        pv = ent.get("person", None)
        sv = ent.get("saliency", None)

        if isinstance(fv, (list, tuple)) and len(fv) == 5 and isinstance(fv[0], str) and (not (bool(globals().get("FACE_CACHE_DISABLE_AI", True)) and str(fv[0]).lower() == "ai")):
            try:
                out["face"] = (str(fv[0]), int(fv[1]), int(fv[2]), int(fv[3]), int(fv[4]))
            except Exception:
                out["face"] = None

        if isinstance(uv, (list, tuple)) and len(uv) == 4:
            try:
                out["upper"] = (int(uv[0]), int(uv[1]), int(uv[2]), int(uv[3]))
            except Exception:
                out["upper"] = None

        if isinstance(pv, (list, tuple)) and len(pv) == 4:
            try:
                out["person"] = (int(pv[0]), int(pv[1]), int(pv[2]), int(pv[3]))
            except Exception:
                out["person"] = None

        if isinstance(sv, (list, tuple)) and len(sv) == 2:
            try:
                out["saliency"] = (float(sv[0]), float(sv[1]))
            except Exception:
                out["saliency"] = None

        return out


def _face_cache_put(p, face, upper, person=None, saliency=None) -> None:
    if not bool(globals().get("FACE_CACHE_ENABLE", True)):
        return
    if not bool(globals().get("DHASH_CACHE_ENABLE", True)):
        return

    _dhash_cache_load_once()
    key = _dhash_norm_path(p)
    mns, sz = _dhash_stat(key)
    if mns is None or sz is None:
        return
    sig = _face_cache_signature()

    f_json = None
    if isinstance(face, tuple) and len(face) == 5:
        try:
            f_json = [str(face[0]), int(face[1]), int(face[2]), int(face[3]), int(face[4])]
        except Exception:
            f_json = None

    try:
        if bool(globals().get("FACE_CACHE_DISABLE_AI", True)) and isinstance(face, tuple) and len(face) == 5:
            if str(face[0]).lower() == "ai":
                f_json = None
    except Exception as e:
        _kana_silent_exc('core:L6713', e)
        pass
    u_json = None
    if isinstance(upper, tuple) and len(upper) == 4:
        try:
            u_json = [int(upper[0]), int(upper[1]), int(upper[2]), int(upper[3])]
        except Exception:
            u_json = None

    p_json = None
    if isinstance(person, tuple) and len(person) == 4:
        try:
            p_json = [int(person[0]), int(person[1]), int(person[2]), int(person[3])]
        except Exception:
            p_json = None

    s_json = None
    if isinstance(saliency, tuple) and len(saliency) == 2:
        try:
            s_json = [float(saliency[0]), float(saliency[1])]
        except Exception:
            s_json = None

    global _DHASH_CACHE_DIRTY
    with _DHASH_CACHE_LOCK:
        ent = _DHASH_CACHE.get(key)
        if not isinstance(ent, dict):
            ent = {}
        ent["mtime_ns"] = int(mns)
        ent["size"] = int(sz)
        ent["face_sig"] = sig
        ent["face"] = f_json
        ent["upper"] = u_json
        ent["person"] = p_json
        ent["saliency"] = s_json
        ent["t"] = float(time.time())
        _DHASH_CACHE[key] = ent
        _DHASH_CACHE_DIRTY = True


# -----------------------------------------------------------------------------
# タイル描画メモキャッシュ（メモリ上・レイアウト共通）
#   - 同じ画像を複数回使う場合の描画を高速化する
#       （topped-up／wrap／extend／重複など）
#   - タイル描画専用で JPEG の "draft" デコードも有効化する。
# -----------------------------------------------------------------------------
from collections import OrderedDict as _KanaOrderedDict
import threading as _kana_threading

_TILE_MEMCACHE_LOCK = _kana_threading.Lock()
_TILE_MEMCACHE = _KanaOrderedDict()  # key(str) -> PIL.Image (tile)
_TILE_MEMCACHE_BYTES = 0

def _tile_memcache_limits() -> tuple[int, int]:
    """タイル描画のメモリキャッシュ上限を (max_items, max_bytes) で返します。0 はキャッシュ無効です。"""
    g = globals()
    try:
        max_items = int(max(0, int(g.get("TILE_MEMCACHE_MAX_ITEMS", 512))))
    except Exception:
        max_items = 512
    try:
        max_bytes = int(max(0, int(g.get("TILE_MEMCACHE_MAX_BYTES", 256 * 1024 * 1024))))
    except Exception:
        max_bytes = 256 * 1024 * 1024
    return max_items, max_bytes

def _tile_memcache_enabled() -> bool:
    return bool(globals().get("TILE_MEMCACHE_ENABLE", True))

def _tile_memcache_est_bytes(img: Image.Image) -> int:
    try:
        bpp = 4 if img.mode in ("RGBA", "LA") else 3
        return int(img.size[0]) * int(img.size[1]) * bpp
    except Exception:
        return 0

def _tile_memcache_get(key: str) -> Optional[Image.Image]:
    if not _tile_memcache_enabled():
        return None
    max_items, max_bytes = _tile_memcache_limits()
    if max_items <= 0 or max_bytes <= 0:
        return None
    with _TILE_MEMCACHE_LOCK:
        img = _TILE_MEMCACHE.get(key)
        if img is not None:
            try:
                _TILE_MEMCACHE.move_to_end(key)
            except Exception as e:
                _warn_exc_once(e)
                pass
        return img

def _tile_memcache_put(key: str, img: Image.Image) -> None:
    if not _tile_memcache_enabled():
        return
    max_items, max_bytes = _tile_memcache_limits()
    if max_items <= 0 or max_bytes <= 0:
        return
    est = _tile_memcache_est_bytes(img)
    if est <= 0:
        return

    global _TILE_MEMCACHE_BYTES
    with _TILE_MEMCACHE_LOCK:
        if key in _TILE_MEMCACHE:
            # 既存を更新
            try:
                _TILE_MEMCACHE.move_to_end(key)
            except Exception as e:
                _warn_exc_once(e)
                pass
            return

        _TILE_MEMCACHE[key] = img
        _TILE_MEMCACHE_BYTES += est

        # 件数で追い出し
        while len(_TILE_MEMCACHE) > max_items:
            try:
                k, v = _TILE_MEMCACHE.popitem(last=False)
                _TILE_MEMCACHE_BYTES -= _tile_memcache_est_bytes(v)
            except Exception as e:
                _kana_silent_exc('core:L6835', e)
                break
        # バイト数で追い出し
        while _TILE_MEMCACHE_BYTES > max_bytes and len(_TILE_MEMCACHE) > 0:
            try:
                k, v = _TILE_MEMCACHE.popitem(last=False)
                _TILE_MEMCACHE_BYTES -= _tile_memcache_est_bytes(v)
            except Exception as e:
                _kana_silent_exc('core:L6843', e)
                break
def _tile_cache_key(p: Path, cw: int, ch: int, mode: str, use_face_focus: bool) -> str:
    base = _dhash_norm_path(p)
    key = f"{base}|{int(cw)}x{int(ch)}|{str(mode)}"
    if use_face_focus:
        # 顔フォーカスは検出器設定に依存
        try:
            key += f"|ff=1|sig={_face_cache_signature()}"
        except Exception:
            key += "|ff=1"
    return key

def _tile_render_cached(p: Path, cw: int, ch: int, mode: str, use_face_focus: bool=False) -> Image.Image:
    """
    描画（draw）段階で使うタイル画像を生成し、必要に応じて顔フォーカスを適用してメモリキャッシュします。

        mode: "fill" または "fit"
    """
    # すべてのモードで毎回フォーカス検出を使いたい場合、use_face_focus が False でもここで有効化します。
    use_ff = bool(use_face_focus)
    try:
        if str(mode) != "fit" and bool(globals().get("FACE_FOCUS_ENABLE", True)) and bool(globals().get("FACE_FOCUS_FORCE_ALL_MODES", True)):
            use_ff = True
    except Exception as e:
        _kana_silent_exc('core:L6868', e)
        pass
    key = _tile_cache_key(p, cw, ch, mode, use_ff)
    hit = _tile_memcache_get(key)
    if hit is not None:
        return hit

    # 描画
    with open_image_safe(p, draft_to=(max(1, int(cw)), max(1, int(ch))), force_mode="RGB") as rgb:
        if use_ff and str(mode) != "fit":
            tile = _cover_rect_face_focus(rgb, int(cw), int(ch), src_path=p)
        else:
            tile = resize_into_cell(rgb, int(cw), int(ch), "fit" if str(mode) == "fit" else "fill")

    _tile_memcache_put(key, tile)
    return tile


def _aes_compute_base_score(p):
    try:
        with open_image_safe(p) as im:
            return _aes_base_score_from_image(im)
    except Exception:
        return None


def score_and_pick(paths: List[Path], count:int, seed:int) -> List[Path]:
    rnd = random.Random(seed)
    feats = []

    banner(_lang("特徴量抽出","Feature extraction"))

    # まずはキャッシュ（kana_wallpaper.dhash_cache.json）から高速に復元
    base_scores = [None] * len(paths)
    missing = []
    for i, p in enumerate(paths):
        sc = _aes_cache_get(p)
        if sc is None:
            missing.append((i, p))
        else:
            base_scores[i] = sc

    # キャッシュに無い分だけ計算（初回や更新分のみ重くなる）
    if missing:
        try:
            workers = int(globals().get("AESTHETIC_WORKERS", 0))
        except Exception:
            workers = 0
        if workers <= 0:
            workers = min(32, (os.cpu_count() or 8))

        # 少数なら逐次、量が多ければスレッドで並列化
        if workers <= 1 or len(missing) < 64:
            for j, (i, p) in enumerate(missing, 1):
                sc = _aes_compute_base_score(p)
                base_scores[i] = sc
                if sc is not None:
                    _aes_cache_put(p, sc)
                if VERBOSE:
                    bar(j, len(missing), prefix="aesthetic ", final=(j == len(missing)))
        else:
            try:
                from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    fut_map = {ex.submit(_aes_compute_base_score, p): (i, p) for (i, p) in missing}
                    done = 0
                    for fut in as_completed(fut_map):
                        i, p = fut_map[fut]
                        try:
                            sc = fut.result()
                        except Exception:
                            sc = None
                        base_scores[i] = sc
                        if sc is not None:
                            _aes_cache_put(p, sc)
                        done += 1
                        if VERBOSE:
                            bar(done, len(missing), prefix="aesthetic ", final=(done == len(missing)))
            except Exception:
                # 並列が失敗したらフォールバック
                for j, (i, p) in enumerate(missing, 1):
                    sc = _aes_compute_base_score(p)
                    base_scores[i] = sc
                    if sc is not None:
                        _aes_cache_put(p, sc)
                    if VERBOSE:
                        bar(j, len(missing), prefix="aesthetic ", final=(j == len(missing)))

    # スコア + 既存の軽いランダム要素で並べ替え
    for i, p in enumerate(paths, 1):
        sc = base_scores[i - 1]
        if sc is None:
            continue
        feats.append({"path": p, "score": float(sc) + rnd.random() * 0.01})
        if VERBOSE:
            bar(i, len(paths), prefix="rank ", final=(i == len(paths)))

    feats.sort(key=lambda x: x["score"], reverse=True)

    uniq = []
    hashes = []
    for i, f in enumerate(feats, 1):
        h = dhash64_for_path_cached(f["path"])
        # 近似重複の判定は、統一閾値 DEDUP_DHASH_THRESHOLD を使用します。
        # 未定義の場合は旧設定 DEDUPE_HAMMING にフォールバックします。
        # 環境変数/設定ファイルの誤設定に備え、念のため int 化します。
        thr = int(globals().get("DEDUP_DHASH_THRESHOLD", DEDUPE_HAMMING))
        if h is not None and any(hamming(h, hv) <= thr for hv in hashes):
            pass
        else:
            uniq.append(f)
            if h is not None:
                hashes.append(h)
        if VERBOSE:
            bar(i, len(feats), prefix="dedup ", final=(i == len(feats)))

    return [f["path"] for f in uniq[:count]]


# -----------------------------------------------------------------------------
# 抽出モードに応じてリストを並べ替える共通関数
#   recent     : 更新日時が新しい順 (mtime降順)
#   oldest     : 更新日時が古い順 (mtime昇順)
#   name_asc   : ファイル名昇順 (大文字小文字無視)
#   name_desc  : ファイル名降順
#   その他     : 並べ替えなし

# =============================================================================
# セクション: 画像抽出（SELECT_MODE）と近似重複排除
# =============================================================================
def sort_by_select_mode(paths: list) -> list:
    """
    SELECT_MODE に応じてパスのリストを並び替える。
    引数のリストを破壊せず、新しいソート済みリストを返す。
    ソート基準は SELECT_MODE グローバル変数の値によって決まる。
    """
    try:
        mode = str(globals().get("SELECT_MODE", "")).lower()
    except Exception:
        return paths
    # ファイル名をソートする際は lower() した完全パス文字列で安定ソート
    def _key_name(p):
        try:
            s = str(p)
            s_l = s.lower()
            try:
                meta = _VIDEO_FRAME_META.get(s)
            except Exception:
                meta = None
            if meta:
                try:
                    vname, t_ms = meta
                    return f"{vname}__{int(t_ms):012d}__{s_l}"
                except Exception as e:
                    _kana_silent_exc('core:L7022', e)
                    pass
            return s_l
        except Exception:
            try:
                return str(p)
            except Exception:
                return ""
    # 更新日時を取得する。取得失敗時は epoch=0 で一番古い扱い
    def _key_mtime(p):
        return _imgref_mtime(p)
    # recent／newest／mtime／modified → 降順
    if mode in ("recent", "newest", "mtime", "modified"):
        return sorted(paths, key=_key_mtime, reverse=True)
    # oldest／older／mtime_asc → 昇順
    if mode in ("oldest", "older", "mtime_asc"):
        return sorted(paths, key=_key_mtime)
    # name_desc／filename_desc → 降順
    if mode in ("name_desc", "filename_desc"):
        return sorted(paths, key=_key_name, reverse=True)
    # name_asc／name／filename／filename_asc → 昇順
    if mode in ("name_asc", "name", "filename", "filename_asc"):
        return sorted(paths, key=_key_name)
    # デフォルト: 並べ替え無し
    return paths

# -----------------------------------------------------------------------------
# 更新順・ソート順抽出関数
#  pick_recent(paths, count, dedupe): 最近のファイルから優先的に選ぶサンプラー
#       更新日時が新しい順に並べ、必要枚数だけ選択。
#       dedupe=True なら近似重複(dHash)を除去しながら選択します。
#  pick_sorted_generic(paths, count, dedupe): 任意のキーでソートして先頭から選ぶ汎用サンプラー
#       sort_by_select_mode() に従って全体を並べ替え、必要枚数だけ選択。
#       dedupe=True なら近似重複(dHash)を除去しながら選択します。


def _all_video_frames_only(paths: list) -> bool:
    """すべてのパスが『動画から抽出したフレーム（キャッシュ画像）』なら True を返します。"""
    if not paths:
        return False
    for p in paths:
        try:
            if _VIDEO_FRAME_META.get(str(p)) is None:
                return False
        except Exception:
            return False
    return True


def _pick_video_frames_timeline_spread(paths: list, count: int, order: str = "asc", dedupe: bool = True) -> list:
    """動画から抽出したフレームを、タイムライン全体から満遍なく選びます。

    使うのは次の条件を満たす場合のみ：
      - LAYOUT_STYLE == 'grid'
      - GRID_VIDEO_TIMELINE in ('asc','desc')
      - all candidates are video frames

    It prevents the tail part from being starved when dedupe early-stops.
    """
    order = (order or "asc").strip().lower()
    thr = int(globals().get("DEDUP_DHASH_THRESHOLD", 4))
    scope = str(globals().get("GRID_VIDEO_TIMELINE_DEDUP_SCOPE", "local") or "local").strip().lower()
    win = int(globals().get("GRID_VIDEO_TIMELINE_DEDUP_WINDOW", 24) or 24)
    if win < 1:
        win = 1
    force_pick = bool(globals().get("GRID_VIDEO_TIMELINE_FORCE_PICK", True))
    force_pick_count = 0

    if not paths or count <= 0:
        return []

    # 記録したメタ情報で (video, timestamp, path) の順にソートします。
    def _k(p):
        meta = _VIDEO_FRAME_META.get(str(p))
        if meta:
            vname, t_ms = meta
            try:
                t_ms_i = int(t_ms)
            except Exception:
                t_ms_i = 0
            if order == "desc":
                return (str(vname), -t_ms_i, str(p).lower())
            return (str(vname), t_ms_i, str(p).lower())
        # フォールバック（通常はmetaがあるので起きない想定）
        return ("", 0, str(p).lower())

    sorted_paths = sorted(list(paths), key=_k)
    n = len(sorted_paths)
    if n <= count:
        return sorted_paths

    # 遅延を減らすため dHash を先読みします。
    if dedupe:
        dhash_prefetch_paths(sorted_paths)

    # ローカルdedupe：直近window枚だけ比較し、後半が欠けるのを防ぎます。
    hashes_all = []
    hashes_recent = []

    def _is_dup(h) -> bool:
        if h is None:
            return False
        if scope == "global":
            return any(hamming(h, hv) <= thr for hv in hashes_all)
        # local
        return any(hamming(h, hv) <= thr for hv in hashes_recent)

    picked = []
    picked_set = set()
    scanned = 0
    skipped = 0
    topped_up = 0

    # タイムライン全体を bins 個に分割
    bins = []
    for i in range(count):
        lo = int(i * n / count)
        hi = int((i + 1) * n / count) - 1
        if hi < lo:
            hi = lo
        bins.append((lo, hi))

    def _iter_center_out(lo: int, hi: int):
        mid = (lo + hi) // 2
        yield mid
        for d in range(1, (hi - lo) + 1):
            a = mid + d
            b = mid - d
            if a <= hi:
                yield a
            if b >= lo:
                yield b

    # bin ごとに1枚ずつ選ぶ
    for lo, hi in bins:
        chosen = None
        chosen_h = None

        for idx in _iter_center_out(lo, hi):
            p = sorted_paths[idx]
            if p in picked_set:
                continue
            scanned += 1
            h = dhash64_for_path_cached(p) if dedupe else None
            if dedupe and _is_dup(h):
                skipped += 1
                continue
            chosen = p
            chosen_h = h
            break

        # dedupeで見つからなくても、このbinから1枚は選びます（カバレッジ優先）
        if chosen is None and force_pick:
            for idx in _iter_center_out(lo, hi):
                p = sorted_paths[idx]
                if p in picked_set:
                    continue
                scanned += 1
                chosen = p
                chosen_h = dhash64_for_path_cached(p) if dedupe else None
                force_pick_count += 1
                break

        if chosen is not None:
            picked.append(chosen)
            picked_set.add(chosen)
            if dedupe and chosen_h is not None:
                hashes_all.append(chosen_h)
                hashes_recent.append(chosen_h)
                if len(hashes_recent) > win:
                    hashes_recent[:] = hashes_recent[-win:]

    # bin側で足りない場合は top-up で補充（まれ）
    if len(picked) < count:
        for p in sorted_paths:
            if len(picked) >= count:
                break
            if p in picked_set:
                continue
            scanned += 1
            h = dhash64_for_path_cached(p) if dedupe else None
            if dedupe and _is_dup(h):
                skipped += 1
                continue
            picked.append(p)
            picked_set.add(p)
            topped_up += 1
            if dedupe and h is not None:
                hashes_all.append(h)
                hashes_recent.append(h)
                if len(hashes_recent) > win:
                    hashes_recent[:] = hashes_recent[-win:]

    # 最後に strict な順序を強制
    picked = sorted(picked, key=_k)

    # ログ（他モードと同じフォーマット）
    if VERBOSE and dedupe:
        _x = (float(scanned) / float(n)) if n else 0.0
        _x_suffix = f", x{_x:.2f}" if (n and scanned > n) else ""
        _fp_suffix = f", force_pick={force_pick_count}"
        _scope_suffix = f", scope={scope}"
        if scope != "global":
            _scope_suffix += f", win={win}"
        note(
            f"Near-duplicate filtering: picked {min(len(picked), count)}/{count}, "
            f"scanned {scanned}/{n}, skipped {skipped}, topped up {topped_up} (thr={thr}{_scope_suffix}{_x_suffix}{_fp_suffix})"
        )

    return picked[:count]


def pick_recent(paths: list, count: int, dedupe: bool = True) -> list:
    """更新日時が新しい順に抽出する。

    paths: 画像パスのリスト
    count: 抽出枚数
    dedupe: True なら dHash による近似重複除去
    戻り値: 選抜したパスのリスト
    """
    # ソート: 新しい順（mtime降順）
    sorted_paths = sorted(paths, key=_imgref_mtime, reverse=True)
    if not dedupe:
        return sorted_paths[:count]

    uniq: list = []
    hashes: list = []
    thr = int(globals().get("DEDUP_DHASH_THRESHOLD", 4))
    skipped = 0
    scanned = 0

    # dedupの枠（banner）は、進捗バー表示時のみ出します。
    # 先読み: ループ中の待ち時間を減らすため、dHash を前もって計算します。
    dhash_prefetch_paths(sorted_paths)

    # 近似重複除去の走査バーは、分母が大きいと 0% に見えやすく混乱しがちなので既定OFF。
    # 必要なら SHOW_RANDOM_DEDUP_PROGRESS=True で「選抜枚数/必要枚数」のバーを表示します。
    show_bar = VERBOSE and bool(globals().get("SHOW_RANDOM_DEDUP_PROGRESS", False))

    if show_bar:
        banner(_lang("類似除去 (dHash)", "Near-duplicate removal (dHash)"))
    n = len(sorted_paths)
    for i, p in enumerate(sorted_paths, 1):
        scanned = i
        h = dhash64_for_path_cached(p)
        is_dup = (h is not None and any(hamming(h, hv) <= thr for hv in hashes))
        if is_dup:
            skipped += 1
            continue

        uniq.append(p)
        if h is not None:
            hashes.append(h)

        if show_bar:
            bar(min(len(uniq), count), max(1, count), prefix="dedup ", final=False)

        if len(uniq) >= count:
            break

    if show_bar:
        bar(min(len(uniq), count), max(1, count), prefix="dedup ", final=True)

    # 表示まとめ（モードに関係なく同形式）
    if VERBOSE:
        picked = min(len(uniq), count)
        note(
            f"Near-duplicate filtering: picked {picked}/{count}, "
            f"scanned {scanned}/{n}, skipped {skipped}, topped up 0 (thr={thr})"
        )

    return uniq[:count]

def pick_sorted_generic(paths: list, count: int, dedupe: bool = True) -> list:
    """SELECT_MODE に応じて並べ替えて抽出する。

    sort_by_select_mode() に従って paths を並べ替え、先頭 count 枚を返す。
    dedupe=True なら dHash で近似重複を除去しながら選択。
    """
    sorted_paths = sort_by_select_mode(list(paths))

    # Grid動画タイムライン：全区間から満遍なく拾う（後半欠け防止）
    try:
        _gvt = str(globals().get('GRID_VIDEO_TIMELINE', 'off') or 'off').strip().lower()
        _style = str(globals().get('LAYOUT_STYLE', '') or '').strip().lower()
        _spread = bool(globals().get('GRID_VIDEO_TIMELINE_SPREAD', True))
        if _style == 'grid' and _spread and _gvt in ('asc', 'desc') and _all_video_frames_only(list(paths)):
            return _pick_video_frames_timeline_spread(list(paths), count, order=_gvt, dedupe=dedupe)
    except Exception as e:
        _warn_exc_once(e)
        pass

    if not dedupe:
        return sorted_paths[:count]

    uniq: list = []
    hashes: list = []
    thr = int(globals().get("DEDUP_DHASH_THRESHOLD", 4))
    skipped = 0
    scanned = 0

    # dedupの枠（banner）は、進捗バー表示時のみ出します。
    # 先読み: ループ中の待ち時間を減らすため、dHash を前もって計算します。
    dhash_prefetch_paths(sorted_paths)

    # 近似重複除去の走査バーは、分母が大きいと 0% に見えやすく混乱しがちなので既定OFF。
    # 必要なら SHOW_RANDOM_DEDUP_PROGRESS=True で「選抜枚数/必要枚数」のバーを表示します。
    show_bar = VERBOSE and bool(globals().get("SHOW_RANDOM_DEDUP_PROGRESS", False))

    if show_bar:
        banner(_lang("類似除去 (dHash)", "Near-duplicate removal (dHash)"))
    n = len(sorted_paths)
    for i, p in enumerate(sorted_paths, 1):
        scanned = i
        h = dhash64_for_path_cached(p)
        is_dup = (h is not None and any(hamming(h, hv) <= thr for hv in hashes))
        if is_dup:
            skipped += 1
            continue

        uniq.append(p)
        if h is not None:
            hashes.append(h)

        if show_bar:
            bar(min(len(uniq), count), max(1, count), prefix="dedup ", final=False)

        if len(uniq) >= count:
            break

    if show_bar:
        bar(min(len(uniq), count), max(1, count), prefix="dedup ", final=True)

    # 表示まとめ（モードに関係なく同形式）
    if VERBOSE:
        picked = min(len(uniq), count)
        note(
            f"Near-duplicate filtering: picked {picked}/{count}, "
            f"scanned {scanned}/{n}, skipped {skipped}, topped up 0 (thr={thr})"
        )

    return uniq[:count]

def pick_random_dedup(shuffled_paths: list, count: int) -> list:
    """シャッフル済みリストから、dHash で近似重複を避けつつ count 枚を選ぶ。

    - 入力は「既にランダム化されている」ことを前提に、順番は維持します。
    - 近似重複が多くて count 枚に届かない場合は、不足分を「できるだけ遠い順」で補充します。
      （それでも候補総数が足りない場合は、取れた分だけ返します）
    """
    thr = int(globals().get("DEDUP_DHASH_THRESHOLD", 4))

    # 近似重複除去の走査バーは、分母が大きいと 0% に見えやすく混乱しがちなので既定OFF。
    # 必要なら SHOW_RANDOM_DEDUP_PROGRESS=True で「選抜枚数/必要枚数」のバーを表示します。
    show_bar = VERBOSE and bool(globals().get("SHOW_RANDOM_DEDUP_PROGRESS", False))

    # 先読みで体感速度を上げる（キャッシュ未計算の dHash を先に作る）
    dhash_prefetch_paths(shuffled_paths)

    def _pkey(x) -> str:
        try:
            return str(x)
        except Exception:
            return repr(x)

    uniq: list = []
    hashes: list = []
    n = len(shuffled_paths)
    scanned = 0
    skipped = 0

    # --- まずは近似重複を避けながら順に採用 ---
    for i, p in enumerate(shuffled_paths, 1):
        scanned = i
        h = dhash64_for_path_cached(p)

        # 近似（Hamming距離 <= thr）なら除外
        if h is not None and any(hamming(h, hv) <= thr for hv in hashes):
            skipped += 1
            continue

        uniq.append(p)
        if h is not None:
            hashes.append(h)

        if show_bar:
            # 「候補全体」ではなく「必要枚数に対してどれだけ選べたか」を表示する
            bar(min(len(uniq), count), max(1, count), prefix="dedup ", final=False)

        if len(uniq) >= count:
            break

    # --- 足りない場合は topped-up（補充） ---
    topped_up = 0
    topped_up_strict = 0
    topped_up_relaxed = 0
    strict_timeout = False
    if len(uniq) < count:
        need = count - len(uniq)

        sel_keys = {_pkey(x) for x in uniq}
        pool = [p for p in shuffled_paths if _pkey(p) not in sel_keys]

        sel_hashes = [h for h in hashes if h is not None]

        # 候補の『今の集合に対する最小距離』を持たせる（greedy farthest-first）
        cand = []
        for p in pool:
            h = dhash64_for_path_cached(p)
            if h is None or not sel_hashes:
                mind = -1
            else:
                md = 1_000_000
                for sh in sel_hashes:
                    d = hamming(h, sh)
                    if d < md:
                        md = d
                        if md <= 0:
                            break
                mind = int(md) if md != 1_000_000 else -1
            cand.append([p, h, mind])

        # strict補充: 可能な限り thr を満たす（mind > thr）候補で埋める
        enable_strict = bool(globals().get('TOPUP_STRICT_DEDUP_ENABLE', True))
        timeout_sec = float(globals().get('TOPUP_STRICT_DEDUP_TIMEOUT_SEC', 5.0))
        # Trueにすると、strictで埋まらない場合は『遠い順』で緩く補充して count まで埋めます
        relax_if_fail = bool(globals().get('TOPUP_STRICT_DEDUP_RELAX_IF_TIMEOUT', True))
        deadline = time.time() + max(0.0, timeout_sec)

        def _pick_best(require_over_thr: bool) -> int:
            best_idx = -1
            best_score = -1_000_000
            for j in range(len(cand)):
                sc = cand[j][2]
                if require_over_thr and sc >= 0 and sc <= thr:
                    continue
                if sc > best_score:
                    best_score = sc
                    best_idx = j
            return best_idx

        # 1) strict で埋める（timeoutあり）
        while need > 0 and cand and enable_strict:
            if timeout_sec > 0.0 and time.time() > deadline:
                strict_timeout = True
                break
            idx = _pick_best(require_over_thr=True)
            if idx < 0:
                break
            p, h, _ = cand.pop(idx)
            uniq.append(p)
            topped_up += 1
            topped_up_strict += 1
            need -= 1

            if h is not None:
                sel_hashes.append(h)
                # 追加した1枚に対してだけ更新（O(N)）
                for row in cand:
                    hh = row[1]
                    if hh is None:
                        continue
                    d = hamming(hh, h)
                    if row[2] < 0 or d < row[2]:
                        row[2] = d

            if show_bar:
                bar(min(len(uniq), count), max(1, count), prefix='dedup ', final=False)

        # 2) strictで足りない場合の扱い
        if need > 0:
            if (not enable_strict) or relax_if_fail:
                # 緩く補充（thr 条件は満たさない可能性あり）
                while need > 0 and cand:
                    idx = _pick_best(require_over_thr=False)
                    if idx < 0:
                        break
                    p, h, _ = cand.pop(idx)
                    uniq.append(p)
                    topped_up += 1
                    topped_up_relaxed += 1
                    need -= 1

                    if h is not None:
                        sel_hashes.append(h)
                        for row in cand:
                            hh = row[1]
                            if hh is None:
                                continue
                            d = hamming(hh, h)
                            if row[2] < 0 or d < row[2]:
                                row[2] = d

                    if show_bar:
                        bar(min(len(uniq), count), max(1, count), prefix='dedup ', final=False)
            else:
                # strictのみ（埋まらない場合はここで打ち切り）
                pass
    # topped_up が発生した場合、選んだ集合は「何を選ぶか」は良くても「並び」が偶然で似た画像が隣り合うことがあります。
    # そこで、隣接がなるべく離れるように並び順だけ分散（選択結果は変えない）します。
    spread_applied = False
    picked = uniq[:count]
    if topped_up > 0 and SPREAD_RANDOM_WHEN_TOPPED_UP and len(picked) >= 3:
        try:
            items = [(p, dhash64_for_path_cached(p)) for p in picked]

            # 初手：平均距離が最大のもの（なければ先頭）
            best0 = 0
            best0_score = -1.0
            hs = [h for _, h in items if h is not None]
            if hs:
                for i0, (_, h0) in enumerate(items):
                    if h0 is None:
                        continue
                    s = 0
                    c = 0
                    for _, h1 in items:
                        if h1 is None or h1 == h0:
                            continue
                        s += hamming(h0, h1)
                        c += 1
                    sc = (s / c) if c else -1.0
                    if sc > best0_score:
                        best0_score = sc
                        best0 = i0

            order = []
            used = [False] * len(items)
            order.append(items[best0][0])
            used[best0] = True

            last_hashes = []
            h0 = items[best0][1]
            if h0 is not None:
                last_hashes.append(h0)

            # 直近2枚に対しての最小距離を最大化（ローカルに散らす）
            while len(order) < len(items):
                best_i = -1
                best_score = -1
                tail = [h for h in last_hashes[-2:] if h is not None]
                for i1, (p1, h1) in enumerate(items):
                    if used[i1]:
                        continue
                    if h1 is None or not tail:
                        sc = 32  # 比較不能は中程度扱いで混ぜる
                    else:
                        sc = min(hamming(h1, th) for th in tail)
                    if sc > best_score:
                        best_score = sc
                        best_i = i1

                if best_i < 0:
                    break
                order.append(items[best_i][0])
                used[best_i] = True
                if items[best_i][1] is not None:
                    last_hashes.append(items[best_i][1])

            # うまく組めなかった分は元順で補完（保険）
            if len(order) != len(items):
                rest = [items[i][0] for i in range(len(items)) if not used[i]]
                order.extend(rest)

            picked = order
            spread_applied = True
        except Exception:
            # 並び分散に失敗しても生成自体は続行
            picked = uniq[:count]
            spread_applied = False
    else:
        picked = uniq[:count]

    # 表示まとめ（モードに関係なく同形式）
    if show_bar:
        bar(min(len(uniq), count), max(1, count), prefix="dedup ", final=True)

    if VERBOSE:
        note(
            f"Near-duplicate filtering: picked {min(len(uniq), count)}/{count}, "
            f"scanned {scanned}/{n}, skipped {skipped}, topped up {topped_up} (thr={thr})"
            + (f", strict {topped_up_strict}" if topped_up_strict else "")
            + (f", relaxed {topped_up_relaxed}" if topped_up_relaxed else "")
            + (", timeout" if strict_timeout else "")
            + (", spread order" if spread_applied else "")
        )

    return picked


# -----------------------------------------------------------------------------
# サブセクション: 色ベクトル（LAB 近似）と距離
# -----------------------------------------------------------------------------
def _avg_lab_vector(p: Path) -> Tuple[float,float,float]:
    # プロセス内キャッシュ（最速）
    if p in _AVG_LAB_CACHE:
        return _AVG_LAB_CACHE[p]

    # 永続キャッシュ（dhash キャッシュを流用）
    try:
        v = _lab_cache_get(p)
        if v is not None:
            _AVG_LAB_CACHE[p] = v
            return v
    except Exception as e:
        _warn_exc_once(e)
        pass
    # 実計算
    try:
        with open_image_safe(p) as im:
            sm = im.resize((32,32), Image.BILINEAR)
            try:
                lab = sm.convert("LAB")
                L,a,b = ImageStat.Stat(lab).mean
            except Exception:
                r,g,b = ImageStat.Stat(sm).mean[:3]
                L = 0.2126*r + 0.7152*g + 0.0722*b
                a = 0.5*(r - g)
                b = 0.5*(b - g)
            vec = (float(L), float(a), float(b))
            _AVG_LAB_CACHE[p] = vec
            try:
                _lab_cache_put(p, vec)
            except Exception as e:
                _warn_exc_once(e)
                pass
            return vec
    except Exception:
        return (50.0, 0.0, 0.0)


def _vec_dist(u:Tuple[float,float,float], v:Tuple[float,float,float])->float:
    dx=u[0]-v[0]; dy=u[1]-v[1]; dz=u[2]-v[2]
    return math.sqrt(dx*dx+dy*dy+dz*dz)

# -----------------------------------------------------------------------------
# サブセクション: Grid 近傍色差：従来ヒルクライム
# -----------------------------------------------------------------------------
def _grid_edges(rows:int, cols:int):    # レイアウト情報（1回だけ）を表示
    global _PRINTED_LAYOUT_ONCE
    if not _PRINTED_LAYOUT_ONCE:
        try:
            note(f"LAYOUT: grid | ROWS×COLS: {rows}×{cols}")
        except Exception as e:
            _warn_exc_once(e)
            pass
        _PRINTED_LAYOUT_ONCE = True

    edges=[]; neigh=[set() for _ in range(rows*cols)]
    def idx(r,c): return r*cols+c
    for r in range(rows):
        for c in range(cols):
            i=idx(r,c)
            if c+1<cols:
                j=idx(r,c+1); edges.append((i,j)); neigh[i].add(j); neigh[j].add(i)
            if r+1<rows:
                j=idx(r+1,c); edges.append((i,j)); neigh[i].add(j); neigh[j].add(i)
    neigh=[sorted(list(s)) for s in neigh]
    return edges, neigh


# =============================================================================
# セクション: 最適化（grid/hex/mosaic の近傍目的関数・anneal）
# =============================================================================
def optimize_grid_neighbors(paths: List[Path], rows:int, cols:int, iters:int=1500, seed:int=0, objective:str="max"):
    n = min(len(paths), rows*cols)
    paths = list(paths[:n])
    vecs = [_avg_lab_vector(p) for p in paths]
    edges, neigh = _grid_edges(rows, cols)
    order = list(range(n))

    def sumdiff() -> float:
        s=0.0
        for i,j in edges:
            if i<n and j<n: s += _vec_dist(vecs[order[i]], vecs[order[j]])
        return s

    def cost(value: float) -> float:
        return -value if objective=="max" else value

    curr_sum = sumdiff()
    best_cost = cost(curr_sum)
    init_sum = curr_sum
    accepted = 0
    rnd = random.Random(seed if seed!="random" else secrets.randbits(32))

    def local_delta_sum(a:int, b:int, s_now:float) -> float:
        if a==b: return s_now
        for k in neigh[a]:
            if k<n and k!=b: s_now -= _vec_dist(vecs[order[a]], vecs[order[k]])
        for k in neigh[b]:
            if k<n and k!=a: s_now -= _vec_dist(vecs[order[b]], vecs[order[k]])
        order[a], order[b] = order[b], order[a]
        for k in neigh[a]:
            if k<n and k!=b: s_now += _vec_dist(vecs[order[a]], vecs[order[k]])
        for k in neigh[b]:
            if k<n and k!=a: s_now += _vec_dist(vecs[order[b]], vecs[order[k]])
        return s_now

    # 最適化のバナー/目的を表示します（UI_LANG='en' の場合は英語）。
        banner("Optimize: Grid neighbor color (hill)")
        obj_label = "maximize (diversify)" if objective == "max" else "minimize (similarize)"
        note(f"Objective: {obj_label} / iterations {iters}")
    for t in range(iters):
        a = rnd.randrange(n); b = rnd.randrange(n)
        new_sum  = local_delta_sum(a, b, curr_sum)
        new_cost = cost(new_sum)
        if new_cost <= best_cost:
            curr_sum  = new_sum; best_cost = new_cost; accepted += 1
        else:
            order[a], order[b] = order[b], order[a]
        if VERBOSE: bar(t+1, iters, prefix="hill", final=(t+1==iters))

    imp = ((curr_sum-init_sum)/init_sum*100.0) if objective=="max" and init_sum>0 else \
          ((init_sum-curr_sum)/init_sum*100.0) if objective=="min" and init_sum>0 else 0.0
    if str(globals().get('UI_LANG','')).lower() == 'en':
        note(f"ΣΔcolor: {init_sum:.1f} → {curr_sum:.1f} ({imp:+.1f}%) / accepted {accepted}/{iters}")
    else:
        note(f"ΣΔColor: {init_sum:.1f} → {curr_sum:.1f} ({imp:+.1f}%) / accepted {accepted}/{iters}")

    new_paths = [paths[i] for i in order]
    summary = {"grid_neighbor":{"objective":objective,"initial":init_sum,"final":curr_sum,"improved_pct":imp,"accepted":accepted,"iters":iters}}
    return new_paths, summary

# -----------------------------------------------------------------------------
# サブセクション: Grid 近傍色差：anneal（Simulated Annealing）
# -----------------------------------------------------------------------------
def optimize_grid_neighbors_anneal(
    paths: Sequence[Path],
    rows: int,
    cols: int,
    steps: int = 20000,
    T0: float = 1.0,
    Tend: float = 1e-3,
    reheats: int = 2,
    seed: Union[int, str] = "random",
    objective: str = "min",
) -> Tuple[List[Path], Dict[str, Any]]:
    """
    Grid: annealで Σ(隣接セルの色差) を最適化。
    steps  : 総ステップ（大きいほど強い）
    T0/Tend: 温度（指数冷却）
    reheats: 再加熱回数（局所脱出用）
    objective: "max"=バラけ／"min"=似る
    """
    n = min(len(paths), rows * cols)
    paths = list(paths[:n])
    vecs = [_avg_lab_vector(p) for p in paths]
    edges, neigh = _grid_edges(rows, cols)

    # 有効範囲の無向エッジだけに絞る（表示/再計算用）
    edges_n = [(i, j) for (i, j) in edges if i < n and j < n]

    rng = random.Random(seed if seed != "random" else secrets.randbits(32))
    order = list(range(n))  # 位置 -> 画像index

    def sumdiff_for(ord_list):
        s = 0.0
        for i, j in edges_n:
            s += _vec_dist(vecs[ord_list[i]], vecs[ord_list[j]])
        return s

    def sumdiff():
        return sumdiff_for(order)

    def to_cost(value):
        return -value if objective == "max" else value

    best_order = order[:]
    init_sum = sumdiff()
    curr_sum = init_sum
    best_sum = init_sum
    curr_cost = to_cost(curr_sum)
    best_cost = curr_cost

    def local_swap_delta(a, b, s_now):
        if a == b:
            return s_now
        ea = [k for k in neigh[a] if k < n and k != b]
        eb = [k for k in neigh[b] if k < n and k != a]
        for k in ea:
            s_now -= _vec_dist(vecs[order[a]], vecs[order[k]])
        for k in eb:
            s_now -= _vec_dist(vecs[order[b]], vecs[order[k]])
        order[a], order[b] = order[b], order[a]
        for k in ea:
            s_now += _vec_dist(vecs[order[a]], vecs[order[k]])
        for k in eb:
            s_now += _vec_dist(vecs[order[b]], vecs[order[k]])
        return s_now

    def local_insert_delta(a, b, s_now):
        if a == b:
            return s_now
        val = order.pop(a)
        order.insert(b, val)
        # insert は正確さ優先でフル再計算
        return sumdiff()

    # Grid anneal最適化のバナー/目的を表示します（UI_LANG='en' の場合は英語）。
        banner("Optimize: Grid anneal")
        obj_label = "maximize (diversify)" if objective == "max" else "minimize (similarize)"
        note(f"Objective: {obj_label} / steps={int(steps)} reheats={int(reheats)}")

    steps = int(max(1, steps))
    reheats = int(max(0, reheats))

    # remainder を配って total==steps を保証
    phases = reheats + 1
    base = steps // phases
    rem = steps % phases

    done = 0
    accepted = 0

    for ph in range(phases):
        phase_steps = base + (1 if ph < rem else 0)
        t = 0
        while t < phase_steps:
            frac = t / max(1, phase_steps - 1)
            T = float(T0) * ((float(Tend) / float(T0)) ** frac)

            if rng.random() < 0.8:
                a = rng.randrange(n)
                b = rng.randrange(n)
                new_sum = local_swap_delta(a, b, curr_sum)
            else:
                a = rng.randrange(n)
                b = rng.randrange(n)
                new_sum = local_insert_delta(a, b, curr_sum)

            new_cost = to_cost(new_sum)
            d = new_cost - curr_cost
            if d <= 0 or rng.random() < math.exp(-d / max(1e-12, T)):
                curr_sum = new_sum
                curr_cost = new_cost
                accepted += 1
                if curr_cost <= best_cost:
                    best_cost = curr_cost
                    best_sum = curr_sum
                    best_order = order[:]
            else:
                # revert: best をベースに戻す（安定志向）
                order = best_order[:]
                curr_sum = sumdiff_for(order)
                curr_cost = to_cost(curr_sum)

            t += 1
            done += 1
            if VERBOSE:
                bar(done, steps, prefix="anneal", final=False)

        # 再加熱：ベストから小乱択
        if ph < phases - 1:
            order = best_order[:]
            for _ in range(max(1, n // 20)):
                i = rng.randrange(n)
                j = rng.randrange(n)
                order[i], order[j] = order[j], order[i]
            curr_sum = sumdiff_for(order)
            curr_cost = to_cost(curr_sum)

    if VERBOSE:
        bar(steps, steps, prefix="anneal", final=True)

    # 真値を再計算して「best vs final」を選ぶ（丸め/増分誤差を吸収）
    final_sum_true = sumdiff_for(order)
    best_sum_true = sumdiff_for(best_order)

    chosen_order = list(best_order)
    chosen_sum = float(best_sum_true)
    if str(objective).lower() == "max":
        if final_sum_true > chosen_sum:
            chosen_order = list(order)
            chosen_sum = float(final_sum_true)
    else:
        if final_sum_true < chosen_sum:
            chosen_order = list(order)
            chosen_sum = float(final_sum_true)

    # Mosaic と同じ形式で改善量を表示
    _note_opt_improve_sumdelta(init_sum, chosen_sum, str(objective).lower(), accepted, steps)

    new_paths = [paths[i] for i in chosen_order]
    summary = {
        "grid_neighbor_anneal": {
            "objective": str(objective).lower(),
            "init_sum": float(init_sum),
            "best_sum": float(chosen_sum),
            "final_sum": float(final_sum_true),
            "steps": int(steps),
            "accepted": int(accepted),
        }
    }
    return new_paths, summary
# -----------------------------------------------------------------------------
# サブセクション: Grid：Checkerboard 市松＋貪欲
# -----------------------------------------------------------------------------
def _median_split_by_L(vecs: List[Tuple[float,float,float]]):
    Ls = [v[0] for v in vecs]
    med = sorted(Ls)[len(Ls)//2] if Ls else 50.0
    A = [i for i,v in enumerate(vecs) if v[0]>=med]
    B = [i for i,v in enumerate(vecs) if v[0]< med]
    return A, B

def optimize_grid_checkerboard(paths: List[Path], rows:int, cols:int, seed:int=0, objective:str="max"):
    """
    明暗2分割→市松模様に割当→貪欲に隣接差を稼ぐ。
    objective="max" 推奨（"min" の場合は“似せる”方向）。
    """
    n = min(len(paths), rows*cols)
    paths = list(paths[:n])

    # 市松（checkerboard）は準備計算（Labベクトル計算）が重い場合があるため、
    # 先にバナーを出して「いま何をしているか」が分かるようにします。
    banner(_lang("最適化: Grid checkerboard", "Optimize: Grid checkerboard"))
# 平均Labベクトルを計算（キャッシュが無い初回はここが重くなりがち）
    vecs = []
    for i, p in enumerate(paths, 1):
        vecs.append(_avg_lab_vector(p))
        if VERBOSE:
            bar(i, n, prefix="feat", final=(i == n))

    A, B = _median_split_by_L(vecs)
    rng = random.Random(seed if seed!="random" else secrets.randbits(32))

    order = [None]*(rows*cols)
    A_list = [i for i in A]; B_list = [i for i in B]
    # 乱数器の内部状態に依存しない“ハッシュシャッフル”で初期順序を作る（seed固定なら再現可能）
    hash_shuffle_inplace(A_list, seed, salt="grid_checker_A")
    hash_shuffle_inplace(B_list, seed, salt="grid_checker_B")

    def neighbor_positions(r,c):
        pos=[]
        if c-1>=0: pos.append((r,c-1))
        if r-1>=0: pos.append((r-1,c))
        return pos

    def score_idx(idx, r, c):
        s=0.0
        for rr,cc in neighbor_positions(r,c):
            k = rr*cols+cc
            if order[k] is not None:
                s += _vec_dist(vecs[idx], vecs[order[k]])
        return s
    total = rows*cols; done = 0

    for r in range(rows):
        for c in range(cols):
            wantA = ((r+c)&1)==0
            cand_pool = A_list if wantA and A_list else B_list if not wantA and B_list else (A_list or B_list)
            if not cand_pool: break

            best_i = None; best_sc = -1e18 if objective=="max" else 1e18
            K = min(16, len(cand_pool))  # プルーニング
            for idx in cand_pool[:K]:
                sc = score_idx(idx, r, c)
                if (objective=="max" and sc>best_sc) or (objective=="min" and sc<best_sc):
                    best_sc = sc; best_i = idx
            if best_i is None: best_i = cand_pool[0]
            order[r*cols+c] = best_i
            cand_pool.remove(best_i)

            done += 1
            if VERBOSE: bar(done, total, prefix="assign", final=(done==total))

    new_paths = [paths[i] for i in order if i is not None]
    summary = {"grid_checkerboard":{"objective":objective,"filled":len(new_paths),"rows":rows,"cols":cols}}
    return new_paths, summary

# -----------------------------------------------------------------------------
# サブセクション: Grid：Spectral→Hilbert（PCAが無ければLAB簡易2D）
# -----------------------------------------------------------------------------
def _hilbert_index(xi:int, yi:int, order:int=10) -> int:
    """2D→Hilbert 曲線インデックス（整数）。order=10 → 1024x1024 グリッド。"""
    # 参考：bit操作の簡易実装
    index = 0
    n = 1 << order
    x = xi; y = yi
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        index += s * s * ((3 * rx) ^ ry)
        # 回転（Hilbert 曲線の回転処理）
        if ry == 0:
            if rx == 1:
                x = n-1 - x
                y = n-1 - y
            x, y = y, x
        s >>= 1
    return index


def _hex_neighbor_graph(centers, step_x, step_y, max_deg=6, slack=0.30):
    """hex の可視タイル中心座標から 6近傍（近距離）グラフを作る。
    orient の偶奇分岐に依存せず、幾何的距離で近傍を決めるため安全。
    """
    import math
    n = len(centers)
    if n <= 1:
        return [], [[] for _ in range(n)]

    dmax = max(float(step_x), float(step_y)) * (1.0 + float(slack))
    dmax2 = dmax * dmax
    cell = max(8.0, dmax)

    bins = {}
    for i, (cx, cy) in enumerate(centers):
        key = (int(cx // cell), int(cy // cell))
        bins.setdefault(key, []).append(i)

    cand = [[] for _ in range(n)]
    for i, (cx, cy) in enumerate(centers):
        ix, iy = int(cx // cell), int(cy // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in bins.get((ix + dx, iy + dy), []):
                    if j == i:
                        continue
                    x2, y2 = centers[j]
                    dd = (cx - x2) * (cx - x2) + (cy - y2) * (cy - y2)
                    if dd <= dmax2:
                        cand[i].append((dd, j))

    directed = [[] for _ in range(n)]
    for i in range(n):
        cand[i].sort(key=lambda t: t[0])
        for dd, j in cand[i][:max(1, int(max_deg))]:
            directed[i].append(j)

    edges_set = set()
    neigh = [[] for _ in range(n)]
    for i in range(n):
        for j in directed[i]:
            a, b = (i, j) if i < j else (j, i)
            edges_set.add((a, b))
    edges = sorted(edges_set)
    for a, b in edges:
        neigh[a].append(b)
        neigh[b].append(a)
    return edges, neigh


def _hex_bipartite_colors(neigh):
    """neigh (list[list[int]]) を 2色に塗る。失敗したら None を返す。"""
    n = len(neigh)
    if n <= 0:
        return []
    col = [None] * n
    for s in range(n):
        if col[s] is not None:
            continue
        col[s] = 0
        q = [s]
        qi = 0
        while qi < len(q):
            u = q[qi]; qi += 1
            cu = col[u]
            for v in neigh[u]:
                if v < 0 or v >= n:
                    continue
                if col[v] is None:
                    col[v] = 1 - cu
                    q.append(v)
                elif col[v] == cu:
                    return None
    return col


def _pca1_direction(vecs, iters=24):
    """3次元ベクトル群 vecs から PCA1（主方向）を power-iteration で推定。"""
    n = len(vecs)
    if n <= 0:
        return (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    mx = sum(v[0] for v in vecs) / n
    my = sum(v[1] for v in vecs) / n
    mz = sum(v[2] for v in vecs) / n

    c00 = c01 = c02 = c11 = c12 = c22 = 0.0
    for x, y, z in vecs:
        dx = x - mx
        dy = y - my
        dz = z - mz
        c00 += dx * dx
        c01 += dx * dy
        c02 += dx * dz
        c11 += dy * dy
        c12 += dy * dz
        c22 += dz * dz
    inv = 1.0 / max(1, n)
    c00 *= inv; c01 *= inv; c02 *= inv; c11 *= inv; c12 *= inv; c22 *= inv

    vx, vy, vz = 1.0, 1.0, 1.0
    for _ in range(max(1, int(iters))):
        nx = c00 * vx + c01 * vy + c02 * vz
        ny = c01 * vx + c11 * vy + c12 * vz
        nz = c02 * vx + c12 * vy + c22 * vz
        norm = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
        vx, vy, vz = nx / norm, ny / norm, nz / norm

    return (vx, vy, vz), (mx, my, mz)


def reorder_global_spectral_diagonal(paths, objective="min", diag_dir: str = "tl_br"):
    """全画像を1列に並べる前処理（hex用にも使う）: PCA→2D→対角スイープ。
       objective="max" の場合は端から交互に取って“バラけ”を増やす。"""
    n = len(paths)
    if n <= 1:
        return list(paths)

    try:
        banner(_lang("前処理: Hex/Global（スペクトル→対角）","Preprocess: Hex global (spectral → diagonal)"))
    except Exception as e:
        _warn_exc_once(e)
        pass
    vecs = []
    if VERBOSE:
        try:
            bar(0, n, prefix="feat   ", final=False)
        except Exception as e:
            _warn_exc_once(e)
            pass
    for i, p in enumerate(paths, 1):
        try:
            v = _avg_lab_vector(p)
        except Exception:
            v = (0.0, 0.0, 0.0)
        vecs.append(v)
        if VERBOSE:
            try:
                bar(i, n, prefix="feat   ", final=(i == n))
            except Exception as e:
                _warn_exc_once(e)
                pass
    # PC1（第1主成分）
    (d1x, d1y, d1z), (mx, my, mz) = _pca1_direction(vecs, iters=28)

    # PC2（簡易 deflation: vec を PC1 へ射影→除去→再度 PCA1 を当てて PC2 とみなす）
    resid = []
    for x, y, z in vecs:
        dx, dy, dz = x - mx, y - my, z - mz
        proj = dx * d1x + dy * d1y + dz * d1z
        rx = dx - proj * d1x
        ry = dy - proj * d1y
        rz = dz - proj * d1z
        resid.append((rx, ry, rz))
    (d2x, d2y, d2z), _ = _pca1_direction(resid, iters=28)

    uv = []
    for i, (x, y, z) in enumerate(vecs):
        dx, dy, dz = x - mx, y - my, z - mz
        u = dx * d1x + dy * d1y + dz * d1z
        v = dx * d2x + dy * d2y + dz * d2z
        uv.append((u, v, i))

    # 0..1 に正規化
    us = [t[0] for t in uv]; vs = [t[1] for t in uv]
    umin, umax = min(us), max(us)
    vmin, vmax = min(vs), max(vs)
    du = (umax - umin) if (umax != umin) else 1.0
    dv = (vmax - vmin) if (vmax != vmin) else 1.0

    d = (diag_dir or "tl_br").strip().lower()

    keys = []
    for u, v, i in uv:
        uu = (u - umin) / du
        vv = (v - vmin) / dv
        # 対角: dir に応じたスイープ
        if d in ("br_tl", "reverse_tl_br", "tl_br_rev", "rev_tl_br"):
            k0 = -(uu + vv); k1 = uu
        elif d in ("tr_bl", "tr2bl", "topright_bottomleft"):
            k0 = (vv - uu); k1 = uu
        elif d in ("bl_tr", "bl2tr", "bottomleft_topright"):
            k0 = (uu - vv); k1 = uu
        else:
            k0 = (uu + vv); k1 = uu
        keys.append((k0, k1, i))

    keys.sort(key=lambda t: (t[0], t[1]))
    idxs = [i for _, _, i in keys]

    if str(objective).lower() == "max":
        # 端から交互（low/high/low/high...）で“市松っぽさ”を少し足す
        lo = 0
        hi = len(idxs) - 1
        zig = []
        toggle = True
        while lo <= hi:
            if toggle:
                zig.append(idxs[lo]); lo += 1
            else:
                zig.append(idxs[hi]); hi -= 1
            toggle = not toggle
        idxs = zig

    return [paths[i] for i in idxs]


def reorder_hex_checkerboard_seed(paths, centers, step_x, step_y, seed="random"):
    """hex用: 6近傍の2彩（市松）を作る初期並び。
       PCA1 で上下に割って、隣接が“高コントラスト”になりやすい配置を作る。"""
    npos = len(centers)
    nimg = len(paths)
    if npos <= 1 or nimg <= 1:
        return list(paths)

    n = min(npos, nimg)

    # 近傍グラフ（checkerboard 用は緩めすぎると三角ができるので slack 小さめ）
    try:
        edges, neigh = _hex_neighbor_graph(centers, step_x, step_y, max_deg=6, slack=0.12)
    except Exception:
        edges, neigh = [], [[] for _ in range(npos)]

    neigh_use = [ [v for v in neigh[i] if v < n] for i in range(min(npos, n)) ]
    colors = _hex_bipartite_colors(neigh_use)
    if colors is None:
        # フォールバック: 描画順の偶奇
        colors = [i & 1 for i in range(n)]

    idx0 = [i for i, c in enumerate(colors[:n]) if c == 0]
    idx1 = [i for i, c in enumerate(colors[:n]) if c == 1]
    n0, n1 = len(idx0), len(idx1)
    if n0 == 0 or n1 == 0:
        colors = [i & 1 for i in range(n)]
        idx0 = [i for i in range(n) if (i & 1) == 0]
        idx1 = [i for i in range(n) if (i & 1) == 1]
        n0, n1 = len(idx0), len(idx1)

    # 特徴量
    try:
                banner(_lang("前処理: Hex checkerboard seed", "Preprocess: Hex checkerboard seed"))
    except Exception as e:
        _warn_exc_once(e)
        pass
    vecs = []
    if VERBOSE:
        try:
            bar(0, n, prefix="feat   ", final=False)
        except Exception as e:
            _warn_exc_once(e)
            pass
    for i, p in enumerate(paths[:n], 1):
        try:
            v = _avg_lab_vector(p)
        except Exception:
            v = (0.0, 0.0, 0.0)
        vecs.append(v)
        if VERBOSE:
            try:
                bar(i, n, prefix="feat   ", final=(i == n))
            except Exception as e:
                _warn_exc_once(e)
                pass
    (dx, dy, dz), (mx, my, mz) = _pca1_direction(vecs, iters=28)
    scores = []
    for i, (x, y, z) in enumerate(vecs):
        sx = x - mx
        sy = y - my
        sz = z - mz
        s = sx * dx + sy * dy + sz * dz
        scores.append((s, i))

    scores.sort(key=lambda t: t[0])
    low = [i for _, i in scores[:n1]]
    high = [i for _, i in scores[-n0:]]

    # ほどよくランダムに（同じ側が偏って並ぶのを防ぐ）
    try:
        rnd = random.Random(seed if seed != "random" else secrets.randbits(32))
    except Exception:
        rnd = random.Random(secrets.randbits(32))
    rnd.shuffle(low)
    rnd.shuffle(high)

    order = [None] * n
    hi_p = 0
    lo_p = 0
    for pos in range(n):
        if colors[pos] == 0:
            order[pos] = high[hi_p]; hi_p += 1
        else:
            order[pos] = low[lo_p]; lo_p += 1

    first = [paths[i] for i in order if i is not None]
    rest = list(paths[n:])
    return first + rest

def _kana_hex_collect_visible_centers(orient, S, step_x, step_y, margin, width, height, extend, r_used, c_used):
    """描画で画像を消費する順序（可視タイル順）に合わせて中心座標を列挙する。"""
    centers = []
    S = float(S)
    if str(orient).lower() == "row-shift":
        half = S / 2.0
        min_r, max_r = -int(extend), int(r_used) + int(extend)
        min_c, max_c = -int(extend), int(c_used) + int(extend)
        for r in range(min_r, max_r):
            shift = half if (r % 2 != 0) else 0.0
            y = float(margin) + float(int(round(r * float(step_y))))
            for c in range(min_c, max_c):
                x = float(margin) + float(int(round(shift + c * float(step_x))))
                if x + S <= 0 or y + S <= 0 or x >= float(width) or y >= float(height):
                    continue
                centers.append((x + S / 2.0, y + S / 2.0))
    else:
        half_v = float(step_y) / 2.0
        min_c, max_c = -int(extend), int(c_used) + int(extend)
        min_r, max_r = -int(extend), int(r_used) + int(extend)
        for c in range(min_c, max_c):
            shift_y = half_v if (c % 2 != 0) else 0.0
            x = float(margin) + float(int(round(c * float(step_x))))
            for r in range(min_r, max_r):
                y = float(margin) + float(int(round(shift_y + r * float(step_y))))
                if x + S <= 0 or y + S <= 0 or x >= float(width) or y >= float(height):
                    continue
                centers.append((x + S / 2.0, y + S / 2.0))
    return centers


def optimize_hex_neighbors_anneal(
    paths: Sequence[Path],
    edges: List[Tuple[int, int]],
    neigh: List[List[int]],
    steps: int = 40000,
    T0: float = 1.0,
    Tend: float = 1e-3,
    reheats: int = 2,
    seed: Union[int, str] = "random",
    objective: str = "min",
) -> Tuple[List[Path], Dict[str, Any]]:
    """Hex向け: 6近傍の隣接コスト（ΣΔcolor）を局所最適化する（anneal）。
    objective:
      - "min" : 近い色を近く（グラデ向き）
      - "max" : 近い色を離す（散らし向き）

    補足:
      - T0/Tend は温度スケジュール（指数冷却）の開始/終了値です。
        T0 を上げると探索が広がり、Tend を下げると終盤が締まります。
      - reheats は再加熱回数です（局所解からの脱出用。0〜3程度が目安）。
    """
    import random, secrets, math
    obj = str(objective).lower()

    n = min(len(paths), len(neigh))
    if n <= 1:
        return list(paths), {"hex_neighbor_anneal": {"skipped": True, "reason": "n<=1"}}

    paths = list(paths[:n])
    vecs = [_avg_lab_vector(p) for p in paths]

    # 有効範囲の無向エッジだけに絞る（表示/再計算用）
    edges_n = [(i, j) for (i, j) in edges if i < n and j < n]

    rng = random.Random(seed if seed != "random" else secrets.randbits(32))
    order = list(range(n))

    def sumdiff_for(ord_list):
        s = 0.0
        for i, j in edges_n:
            s += _vec_dist(vecs[ord_list[i]], vecs[ord_list[j]])
        return s

    def to_cost(value):
        v = float(value)
        return -v if obj == "max" else v

    best_order = order[:]
    init_sum = sumdiff_for(order)
    curr_sum = init_sum
    curr_cost = to_cost(curr_sum)
    best_sum = curr_sum
    best_cost = curr_cost

    def local_swap_delta(a, b, s_now):
        if a == b:
            return s_now
        ea = [k for k in neigh[a] if k < n and k != b]
        eb = [k for k in neigh[b] if k < n and k != a]
        for k in ea:
            s_now -= _vec_dist(vecs[order[a]], vecs[order[k]])
        for k in eb:
            s_now -= _vec_dist(vecs[order[b]], vecs[order[k]])
        order[a], order[b] = order[b], order[a]
        for k in ea:
            s_now += _vec_dist(vecs[order[a]], vecs[order[k]])
        for k in eb:
            s_now += _vec_dist(vecs[order[b]], vecs[order[k]])
        return s_now

    try:
        banner(_lang("最適化: Hex 6-neighbor annealing", "Optimize: Hex 6-neighbor anneal"))
        obj_label = "maximize (diversify)" if obj == "max" else "minimize (similarize)"
        note(f"Objective: {obj_label} / steps={int(steps)} reheats={int(reheats)}")
    except Exception as e:
        _warn_exc_once(e)
        pass
    steps = int(max(1, steps))
    reheats = int(max(0, reheats))

    phases = reheats + 1
    base = steps // phases
    rem = steps % phases

    done = 0
    accepted = 0

    for ph in range(phases):
        phase_steps = base + (1 if ph < rem else 0)
        t = 0
        while t < phase_steps:
            frac = t / max(1, phase_steps - 1)
            T = float(T0) * ((float(Tend) / float(T0)) ** frac)

            a = rng.randrange(n)
            b = rng.randrange(n)
            new_sum = local_swap_delta(a, b, curr_sum)
            new_cost = to_cost(new_sum)
            d = new_cost - curr_cost
            if d <= 0 or rng.random() < math.exp(-d / max(1e-12, T)):
                curr_sum = new_sum
                curr_cost = new_cost
                accepted += 1
                if curr_cost <= best_cost:
                    best_cost = curr_cost
                    best_sum = curr_sum
                    best_order = order[:]
            else:
                # revert: best をベースに戻す（安定志向）
                order = best_order[:]
                curr_sum = sumdiff_for(order)
                curr_cost = to_cost(curr_sum)

            t += 1
            done += 1
            if VERBOSE:
                bar(min(done, steps), steps, prefix="anneal ", final=False)

        if ph < phases - 1:
            order = best_order[:]
            for _ in range(max(1, n // 20)):
                i = rng.randrange(n)
                j = rng.randrange(n)
                order[i], order[j] = order[j], order[i]
            curr_sum = sumdiff_for(order)
            curr_cost = to_cost(curr_sum)

    if VERBOSE:
        bar(steps, steps, prefix="anneal ", final=True)

    final_sum_true = sumdiff_for(order)
    best_sum_true = sumdiff_for(best_order)

    chosen_order = list(best_order)
    chosen_sum = float(best_sum_true)
    if obj == "max":
        if final_sum_true > chosen_sum:
            chosen_order = list(order)
            chosen_sum = float(final_sum_true)
    else:
        if final_sum_true < chosen_sum:
            chosen_order = list(order)
            chosen_sum = float(final_sum_true)

    # Mosaic と同じ形式で改善量を表示
    _note_opt_improve_sumdelta(init_sum, chosen_sum, obj, accepted, steps)

    new_paths = [paths[i] for i in chosen_order]
    summary = {
        "hex_neighbor_anneal": {
            "objective": obj,
            "init_sum": float(init_sum),
            "best_sum": float(chosen_sum),
            "final_sum": float(final_sum_true),
            "steps": int(steps),
            "accepted": int(accepted),
        }
    }
    return new_paths, summary


def _mosaic_pos_order_diagonal(centers: Sequence[Tuple[float, float]], diag_dir: str = "tl_br") -> List[int]:
    """タイル中心を「対角スイープ順」に並べたときのインデックス順を返します。"""
    n = len(centers)
    if n <= 1:
        return list(range(n))
    d = (diag_dir or "tl_br").strip().lower()
    # x：左→右、y：上→下
    # tl_br : ↘（x+y が増える方向）
    # br_tl : ↖（x+y が減る方向）
    # tr_bl : ↙（y-x が増える方向）
    # bl_tr : ↗（x-y が増える方向）
    if d in ("br_tl", "reverse_tl_br", "tl_br_rev", "rev_tl_br"):
        key = lambda i: (-(centers[i][0] + centers[i][1]), centers[i][0])
    elif d in ("tr_bl", "tr2bl", "topright_bottomleft"):
        key = lambda i: ((centers[i][1] - centers[i][0]), centers[i][0])
    elif d in ("bl_tr", "bl2tr", "bottomleft_topright"):
        key = lambda i: ((centers[i][0] - centers[i][1]), centers[i][0])
    else:
        key = lambda i: ((centers[i][0] + centers[i][1]), centers[i][0])
    return sorted(range(n), key=key)


def _mosaic_pos_order_hilbert_continuous(centers: Sequence[Tuple[float, float]], order: int = 10) -> List[int]:
    """
    タイル中心座標を正規化し、(x,y)→ヒルベルトインデックス（連続マッピング）で並べた順のインデックスを返します。

        これは元の（素朴な）実装です。モザイクはタイル幅が可変で穴もあるため、ヒルベルトのセルが未使用になりやすく、
        その未使用セルを飛び越えることで、並びの途中に「途切れ」が見えることがあります。
    """
    n = len(centers)
    if n <= 1:
        return list(range(n))
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    dx = max(1e-9, (maxx - minx))
    dy = max(1e-9, (maxy - miny))
    grid = (1 << int(order)) - 1

    def hkey(i):
        xn = (centers[i][0] - minx) / dx
        yn = (centers[i][1] - miny) / dy
        xi = int(max(0, min(grid, round(xn * grid))))
        yi = int(max(0, min(grid, round(yn * grid))))
        return _hilbert_index(xi, yi, order=int(order))

    return sorted(range(n), key=hkey)


def _mosaic_pos_order_hilbert(centers: Sequence[Tuple[float, float]], order: int = 10) -> List[int]:
    """
    モザイク向けに「ヒルベルトっぽい」走査で並べたインデックス順を返します。

        grid/hex と違い、モザイクのタイル中心は完全な格子ではありません（幅が可変・空白がある等）。
        連続座標 (x,y)→HilbertIndex で直接ソートすると、未使用セルを飛び越える影響で画像の途中に「途切れ」が出やすくなります。
        それを抑えるため、まず y/x 位置から粗い行・列の格子を推定してスナップ（grid に近い扱い）し、その格子上でヒルベルトを適用します。

        行クラスタリングに失敗した場合は、連続マッピング方式にフォールバックします。
    """
    n = len(centers)
    if n <= 1:
        return list(range(n))

    # ---- Y でタイルを行にクラスタリング（同じモザイク行の中心は近いYを持つ）
    idx_y = sorted(range(n), key=lambda i: centers[i][1])
    ys = [centers[i][1] for i in idx_y]
    diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    pos_diffs = sorted([d for d in diffs if d > 1e-6])
    if pos_diffs:
        med = pos_diffs[len(pos_diffs) // 2]
        row_tol = max(1.0, med * 0.5)
    else:
        row_tol = 5.0

    rows = []
    cur = [idx_y[0]]
    for k in range(1, len(idx_y)):
        if (centers[idx_y[k]][1] - centers[idx_y[k - 1]][1]) > row_tol:
            rows.append(cur)
            cur = [idx_y[k]]
        else:
            cur.append(idx_y[k])
    rows.append(cur)

    # 複数行を作れなければ、元のマッピングに戻す。
    if len(rows) < 2:
        return _mosaic_pos_order_hilbert_continuous(centers, order=order)

    # ---- 各タイルに粗い格子上の整数 (col,row) を割り当てる
    max_cols = max(len(r) for r in rows) if rows else 1
    max_cols = max(1, max_cols)
    coords = {}

    for r_i, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda i: centers[i][0])
        m = len(row_sorted)
        for c_rank, i in enumerate(row_sorted):
            if max_cols <= 1 or m <= 1:
                cx = 0
            else:
                # 各行のランクを [0, max_cols-1] に広げ、穴（欠け）が支配しないようにする。
                cx = int(round(c_rank * (max_cols - 1) / (m - 1)))
            coords[i] = (cx, int(r_i))

    # ---- 格子サイズからHilbert順を決める（矩形を2の累乗の正方形に埋め込む）
    max_dim = max(max_cols, len(rows))
    p = int(max(1, math.ceil(math.log2(max_dim)))) if max_dim > 1 else 1

    def hkey(i):
        x, y = coords.get(i, (0, 0))
        # 同じスナップ格子セルに複数タイルが入った場合でも順序が安定するよう、二次タイブレークを入れる。
        return (_hilbert_index(int(x), int(y), order=p), int(y), int(x))

    return sorted(range(n), key=hkey)


def _mosaic_knn_neighbor_graph(centers: Sequence[Tuple[float, float]], k: int = 8) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    """タイル中心点から、kNN の近傍リストと無向エッジリストを構築します。"""
    n = len(centers)
    k = int(k) if k is not None else 8
    k = max(1, min(k, max(1, n - 1)))
    neigh = [[] for _ in range(n)]
    # 総当たり（nは壁紙用途ならだいたい十分小さい）
    for i in range(n):
        xi, yi = centers[i]
        dists = []
        for j in range(n):
            if i == j:
                continue
            xj, yj = centers[j]
            dx = xi - xj
            dy = yi - yj
            dists.append((dx * dx + dy * dy, j))
        dists.sort(key=lambda t: t[0])
        neigh[i] = [j for _, j in dists[:k]]

    edges = set()
    for i in range(n):
        for j in neigh[i]:
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    edges = sorted(edges)
    return edges, neigh


def _mosaic_checkerboard_seed_paths(paths: Sequence[Path], centers: Sequence[Tuple[float, float]], k: int = 8, img_order: Optional[List[int]] = None) -> List[Path]:
    """チェッカーボード風の初期化：kNN グラフを 2 色に彩色し、色ごとにジグザグで割り当てます。"""
    n = min(len(paths), len(centers))
    if n <= 1:
        return list(paths[:n])

    edges, neigh = _mosaic_knn_neighbor_graph(centers[:n], k=max(3, int(k)))
    colors = [None] * n

    node_order = sorted(range(n), key=lambda i: (centers[i][0] + centers[i][1], centers[i][0]))
    for s in node_order:
        if colors[s] is not None:
            continue
        colors[s] = 0
        stack = [s]
        while stack:
            i = stack.pop()
            ci = colors[i] if colors[i] is not None else 0
            for j in neigh[i]:
                if j >= n:
                    continue
                if colors[j] is None:
                    colors[j] = 1 - ci
                    stack.append(j)

    # 余りノード
    for i in range(n):
        if colors[i] is None:
            colors[i] = 0

    g0 = [i for i in range(n) if colors[i] == 0]
    g1 = [i for i in range(n) if colors[i] == 1]

    # scatterでは斜めスイープ順を避ける（グラデーションに見えてしまう）。
    # 代わりに OPT_SEED 駆動の安定ハッシュシャッフルで色を散らす。
    _scatter_seed = globals().get("OPT_SEED", "random") or "random"
    try:
        g0 = hash_shuffle(g0, _scatter_seed, salt="mosaic_scatter_pos0")
        g1 = hash_shuffle(g1, _scatter_seed, salt="mosaic_scatter_pos1")
    except Exception:
        # フォールバック：斜めっぽい順
        g0.sort(key=lambda i: (centers[i][0] + centers[i][1], centers[i][0]))
        g1.sort(key=lambda i: (centers[i][0] + centers[i][1], centers[i][0]))

    if img_order is None:
        img_order = list(paths[:n])
    else:
        img_order = list(img_order[:n])

    # ジグザグの両端（低、高、低、高...）
    zig = []
    a = 0
    b = n - 1
    while a <= b:
        zig.append(img_order[a])
        if a != b:
            zig.append(img_order[b])
        a += 1
        b -= 1

    list0 = zig[0::2]
    list1 = zig[1::2]

    # 各グループ内をシャッフル（低/高の分離は保ちつつ、全体が大きなグラデになり過ぎるのを防ぐ）
    try:
        list0 = hash_shuffle(list0, _scatter_seed, salt="mosaic_scatter_img0")
        list1 = hash_shuffle(list1, _scatter_seed, salt="mosaic_scatter_img1")
    except Exception as e:
        _warn_exc_once(e)
        pass
    assigned = [None] * n
    for idx, ti in enumerate(g0):
        assigned[ti] = list0[idx % len(list0)]
    for idx, ti in enumerate(g1):
        assigned[ti] = list1[idx % len(list1)]
    # None を埋める（欠損補完）
    for i in range(n):
        if assigned[i] is None:
            assigned[i] = img_order[i % len(img_order)]
    return assigned


def optimize_mosaic_neighbors_anneal(
    paths: Sequence[Path],
    edges: List[Tuple[int, int]],
    neigh: List[List[int]],
    steps: int = 40000,
    T0: float = 1.0,
    Tend: float = 1e-3,
    reheats: int = 2,
    seed: Union[int, str] = "random",
    objective: str = "min",
) -> Tuple[List[Path], Dict[str, Any]]:
    """Mosaic向け: k近傍の隣接コスト（ΣΔcolor）を局所最適化する（anneal）。
    objective:
      - "min" : 近い色を近く（グラデ向き）
      - "max" : 近い色を離す（散らし向き）

    補足:
      - T0/Tend は温度スケジュール（指数冷却）の開始/終了値です。
        T0 を上げると探索が広がり、Tend を下げると終盤が締まります。
      - reheats は再加熱回数です（局所解からの脱出用。0〜3程度が目安）。
    """
    import random, secrets, math
    obj = str(objective).lower()
    n = min(len(paths), len(neigh))
    if n <= 1:
        return list(paths), {"mosaic_neighbor_anneal": {"skipped": True, "reason": "n<=1"}}

    paths = list(paths[:n])
    vecs = [_avg_lab_vector(p) for p in paths]

    # --- 辺/近傍リストを正規化（以下を満たすようにする）：
    #   - sumdiff() は無向エッジ（重複なし）を走査
    #   - local_swap_delta() は無向近傍リストを使い、関係する全エッジの差分を更新
    edges_n = [(i, j) for (i, j) in edges if i < n and j < n]

    neigh_u = [set() for _ in range(n)]
    for i, j in edges_n:
        neigh_u[i].add(j)
        neigh_u[j].add(i)
    neigh_u = [sorted(s) for s in neigh_u]

    rng = random.Random(seed if seed != "random" else secrets.randbits(32))
    order = list(range(n))

    def sumdiff_for(ord_list):
        s = 0.0
        for i, j in edges_n:
            s += _vec_dist(vecs[ord_list[i]], vecs[ord_list[j]])
        return s

    def to_cost(value):
        v = float(value)
        # maximize の場合は (-value) を最小化する
        return -v if obj == "max" else v

    def local_swap_delta(a, b, s_now):
        """無向近傍を使って、位置 a と b を入れ替えたときの ΣΔcolor の差分更新量を計算します。"""
        if a == b:
            return s_now
        # 入れ替え相手を除いた近傍（a-b 辺は swap しても不変）
        ea = [k for k in neigh_u[a] if k != b]
        eb = [k for k in neigh_u[b] if k != a]

        va = vecs[order[a]]
        vb = vecs[order[b]]

        for k in ea:
            vk = vecs[order[k]]
            s_now -= _vec_dist(va, vk)
            s_now += _vec_dist(vb, vk)
        for k in eb:
            vk = vecs[order[k]]
            s_now -= _vec_dist(vb, vk)
            s_now += _vec_dist(va, vk)
        return s_now

    init_sum = sumdiff_for(order)
    curr_sum = init_sum
    curr_cost = to_cost(curr_sum)

    best_order = list(order)
    best_sum = curr_sum
    best_cost = curr_cost

    try:
        banner(_lang("最適化: Mosaic k-neighbor anneal", "Optimize: Mosaic k-neighbor annealing"))
        obj_label = _lang("maximize（diversify）", "maximize (diversify)") if obj == "max" else _lang("minimize（similarize）", "minimize (similarize)")
        note(f"Objective: {obj_label} / steps={int(steps)} reheats={int(reheats)}")
    except Exception as e:
        _warn_exc_once(e)
        pass
    steps = int(max(1, steps))
    reheats = int(max(0, reheats))

    # 端数を配分して、総反復回数が steps になるようにする
    phases = reheats + 1
    base = steps // phases
    rem = steps % phases

    accepted = 0
    done = 0

    for ph in range(phases):
        phase_steps = base + (1 if ph < rem else 0)
        t = 0
        while t < phase_steps:
            frac = t / max(1, phase_steps - 1)
            T = float(T0) * ((float(Tend) / float(T0)) ** frac)

            a = rng.randrange(n)
            b = rng.randrange(n)
            new_sum = local_swap_delta(a, b, curr_sum)
            new_cost = to_cost(new_sum)
            d = new_cost - curr_cost
            if d <= 0 or rng.random() < math.exp(-d / max(1e-12, T)):
                order[a], order[b] = order[b], order[a]
                curr_sum = new_sum
                curr_cost = new_cost
                accepted += 1
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_sum = curr_sum
                    best_order = list(order)

            t += 1
            done += 1
            # 進捗バー
            try:
                bar(done, steps, prefix="anneal", final=False)
            except Exception as e:
                _warn_exc_once(e)
                pass
        # フェーズ間で再加熱（真の総和を再計算してドリフトを避ける）
        if ph < phases - 1:
            curr_sum = sumdiff_for(order)
            curr_cost = to_cost(curr_sum)

    # 真値を再計算（丸めの錯覚／累積ドリフトを回避）
    try:
        bar(steps, steps, prefix="anneal", final=True)
        final_sum_true = sumdiff_for(order)
        best_sum_true = sumdiff_for(best_order)

        # best_order と current order のうち、本当に良い方を選ぶ
        chosen_order = list(best_order)
        chosen_sum = best_sum_true
        if obj == "max":
            if final_sum_true > chosen_sum:
                chosen_order = list(order)
                chosen_sum = final_sum_true
        else:
            if final_sum_true < chosen_sum:
                chosen_order = list(order)
                chosen_sum = final_sum_true

        # より高精度でログ出力（統一）
        _note_opt_improve_sumdelta(init_sum, chosen_sum, obj, accepted, steps)

    except Exception:
        chosen_order = list(best_order)
        chosen_sum = float(best_sum)
        final_sum_true = float(curr_sum)

    new_paths = [paths[i] for i in chosen_order]
    summary = {
        "mosaic_neighbor_anneal": {
            "objective": obj,
            "init_sum": float(init_sum),
            "best_sum": float(chosen_sum),
            "final_sum": float(final_sum_true),
            "steps": int(steps),
            "accepted": int(accepted),
        }
    }
    return new_paths, summary


def _mosaic_enhance_active():
    """
    Mosaic の post-pack assignment（拡張割り当て）を実行すべきとき True を返します。

        ランチャー/本体のバージョン差に強い判定にしてあり、profile が文字列で指定されている場合や、
        旧仕様の int profile からでも有効扱いにできます。

        注意:
          - ARRANGE_FULL_SHUFFLE が有効なときは、意図的に Mosaic の post-pack assignment（および局所最適化）を無効化します。
            「フルシャッフル＝グラデや並び制約なし」を厳密に守るためです。
    """
    # 入力順保持（順序を壊す処理を抑止）: Mosaic の POST 割り当てを無効化します。
    try:
        if bool(globals().get('PRESERVE_INPUT_ORDER', False)):
            return False
    except Exception as e:
        _warn_exc_once(e)
        pass

    # フルシャッフルは Mosaic の post-pack 割り当て（グラデ/散らし）を上書きし、出力をランダムに保つ。
    try:
        if bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
            try:
                prof0 = str(globals().get("MOSAIC_ENHANCE_PROFILE", "")).strip().lower()
            except Exception:
                prof0 = ""
            if prof0 and prof0 not in ("off", "none", "random"):
                # Remember that Mosaic POST assignment was disabled by full shuffle.
                globals()["_MOSAIC_POST_DISABLED_BY_FULLSHUFFLE"] = True
            return False
    except Exception as e:
        _warn_exc_once(e)
        pass
    try:
        prof = str(globals().get("MOSAIC_ENHANCE_PROFILE", "")).strip().lower()
    except Exception:
        prof = ""
    if prof and prof not in ("off", "none", "random"):
        return True
    try:
        p = globals().get("MOSAIC_PROFILE", None)
        if isinstance(p, int) and p in (1, 2, 3):  # diagonal / hilbert / scatter
            return True
    except Exception as e:
        _warn_exc_once(e)
        pass
    return bool(globals().get("MOSAIC_ENHANCE_ENABLE", False))


def _mosaic_post_assign_paths(paths: Sequence[Path], centers: Sequence[Tuple[float, float]]) -> List[Path]:
    """Mosaic の拡張割り当て（post-pack assignment）＋必要なら局所最適化を適用した後の描画順で paths を返します。"""
    n = min(len(paths), len(centers))
    if n <= 1:
        return list(paths[:n])

    prof = str(globals().get("MOSAIC_ENHANCE_PROFILE", "hilbert")).strip().lower()
    diag_dir = str(globals().get("MOSAIC_DIAGONAL_DIRECTION", "tl_br")).strip().lower()

    # 旧トークン（tlbr/brtl/...）も受け入れて、tl_br 形式へ正規化する
    try:
        diag_dir = str(diag_dir).replace("-", "_")
    except Exception as e:
        _warn_exc_once(e)
        pass
    _diag_map = {"tlbr": "tl_br", "brtl": "br_tl", "trbl": "tr_bl", "bltr": "bl_tr"}
    diag_dir = _diag_map.get(diag_dir, diag_dir)

    pos_h_order = int(globals().get("MOSAIC_POS_HILBERT_ORDER", 10))
    local_enable = bool(globals().get("MOSAIC_LOCAL_OPT_ENABLE", True))
    steps = int(globals().get("MOSAIC_LOCAL_OPT_STEPS", 40000))
    reheats = int(globals().get("MOSAIC_LOCAL_OPT_REHEATS", 2))
    k = int(globals().get("MOSAIC_LOCAL_OPT_K", 8))
    seed = globals().get("OPT_SEED", "random") or "random"

    base_paths = list(paths[:n])

    if prof in ("off", "none", "random"):
        return base_paths

    # 色特徴空間で順序をなだらかにする
    try:
        # 注：
        #   画像側の順序は 2D-Hilbert(特徴空間) を使うと、分布次第で途中に“大ジャンプ”が挟まり
        #   キャンバス上で『真ん中で途切れる』ように見えることがある。
        #   そこで、まずは通常の spectral->Hilbert 順で並べ、
        #   その後 “最大ジャンプ” を末尾へ回す回転を入れて、途切れを端へ追いやる。
        img_order = reorder_global_spectral_hilbert(list(base_paths), objective="min")
        if prof == "hilbert":
            img_order = _rotate_order_move_max_jump_to_end(list(img_order))
    except Exception:
        img_order = list(base_paths)

    if prof in ("scatter", "checker", "checkerboard", "scatter-checker"):
        objective = "max"
        banner(_lang("前処理: Mosaic Checkerboard（scatter）", "Preprocess: Mosaic checkerboard-ish (scatter)"))
        note(_lang(f"seed assign: checkerboard(scatter) | k={k} | n={n}",
                  f"seed assign: checkerboard(scatter) | k={k} | n={n}"))
        assigned = _mosaic_checkerboard_seed_paths(base_paths, centers[:n], k=k, img_order=img_order)
    else:
        objective = "min"
        if prof in ("diagonal", "diag", "grad-diagonal", "gradient-diagonal"):
            banner(_lang("前処理: Mosaic 対角グラデ", "Preprocess: Mosaic diagonal gradient"))
            note(_lang(f"pos order: diagonal({diag_dir}) | n={n}",
                      f"pos order: diagonal({diag_dir}) | n={n}"))
            pos_order = _mosaic_pos_order_diagonal(centers[:n], diag_dir=diag_dir)
        else:
            banner(_lang("前処理: Mosaic ヒルベルトグラデ", "Preprocess: Mosaic Hilbert gradient"))
            note(_lang(f"pos order: hilbert(order={pos_h_order}) | n={n}",
                      f"pos order: hilbert(order={pos_h_order}) | n={n}"))
            pos_order = _mosaic_pos_order_hilbert(centers[:n], order=pos_h_order)

        assigned = [None] * n
        for rank, ti in enumerate(pos_order):
            assigned[ti] = img_order[rank]
        for i in range(n):
            if assigned[i] is None:
                assigned[i] = img_order[i % len(img_order)]

    if local_enable and n >= 3 and steps >= 1:
        edges, neigh = _mosaic_knn_neighbor_graph(centers[:n], k=max(3, k))
        try:
            assigned, _ = optimize_mosaic_neighbors_anneal(
                assigned, edges, neigh,
                steps=steps, reheats=reheats, seed=seed, objective=objective
            )
        except Exception as e:
            _warn_exc_once(e)
            pass
    return assigned


def _kana_hex_apply_global_and_local_opt(images, _vis_needed, orient, S, step_x, step_y, margin, width, height, extend, r_used, c_used):
    """hex 用: グローバル並び（任意）→ 6近傍局所最適化（任意）を適用して images を返す。"""
    try:
        if bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
            return images
    except Exception as e:
        _warn_exc_once(e)
        pass
    # ---- 全体順序 ----
    try:
        hgo = str(globals().get("HEX_GLOBAL_ORDER", "inherit")).lower()
    except Exception:
        hgo = "inherit"
    if hgo == "inherit":
        try:
            hgo = str(globals().get("MOSAIC_GLOBAL_ORDER", "none")).lower()
        except Exception:
            hgo = "none"

    # ---- ランチャー互換（Hexの簡易配置：diagonal/hilbert/scatter） ----
    # ランチャー側は HEX_GLOBAL_ORDER を直接セットしない場合があります。
    # その場合でも「対角グラデ／ヒルベルト／散らし」が効くように、ここで global order を推定します。
    # 優先順位: HEX_ENHANCE_PROFILE(文字列) -> HEX_PROFILE/HEX_GRAD_PROFILE(文字列) -> HEX_GRAD_PROFILE/HEX_PROFILE(数値1/2/3)
    if hgo in ("inherit", "none", "", "off"):
        prof = ""
        try:
            prof = str(globals().get("HEX_ENHANCE_PROFILE", "")).lower()
        except Exception:
            prof = ""
        if not prof:
            # 旧互換: HEX_PROFILE／HEX_GRAD_PROFILE に文字列が入る場合
            try:
                prof = str(globals().get("HEX_PROFILE", globals().get("HEX_GRAD_PROFILE", ""))).lower()
            except Exception:
                prof = ""
        if prof in ("diagonal", "diag", "spectral-diagonal"):
            hgo = "spectral-diagonal"
        elif prof in ("hilbert", "spectral-hilbert", "spectral"):
            hgo = "spectral-hilbert"
        elif prof in ("scatter", "checkerboard", "cb", "hex-checkerboard"):
            hgo = "checkerboard"
        else:
            # 旧互換: 1=diagonal 2=hilbert 3=scatter
            try:
                _p = int(globals().get("HEX_GRAD_PROFILE", globals().get("HEX_PROFILE", 0)))
            except Exception:
                _p = 0
            if _p == 1:
                hgo = "spectral-diagonal"
            elif _p == 2:
                hgo = "spectral-hilbert"
            elif _p == 3:
                hgo = "checkerboard"
    # Hex のログ（通常は短く、必要なら HEX_DEBUG_LOG=True で詳細も出す）
    try:
        if not globals().get("_HEX_ORDER_DEBUG_ONCE", False):
            if bool(globals().get("VERBOSE", False)):
                note(
                    f"Hex: global_order={globals().get('HEX_GLOBAL_ORDER', None)}, "
                    f"profile={globals().get('HEX_PROFILE', None)}, "
                    f"enhance={globals().get('HEX_ENHANCE_PROFILE', None)}, "
                    f"full_shuffle={globals().get('ARRANGE_FULL_SHUFFLE', None)}"
                )
                if bool(globals().get("HEX_DEBUG_LOG", False)):
                    note(
                        "HexDBG: "
                        + f"hgo={hgo} "
                        + f"enhance={globals().get('HEX_ENHANCE_PROFILE', None)} "
                        + f"profile={globals().get('HEX_PROFILE', None)} "
                        + f"grad_profile={globals().get('HEX_GRAD_PROFILE', None)} "
                        + f"global_order={globals().get('HEX_GLOBAL_ORDER', None)} "
                        + f"diag_dir={globals().get('HEX_DIAG_DIR', None)} "
                        + f"diag_dir2={globals().get('HEX_DIAGONAL_DIRECTION', None)} "
                        + f"full_shuffle={globals().get('ARRANGE_FULL_SHUFFLE', None)}"
                    )
            globals()["_HEX_ORDER_DEBUG_ONCE"] = True
    except Exception as e:
        _kana_silent_exc('core:L9106', e)
        pass
    try:
        hobj = str(globals().get("HEX_GLOBAL_OBJECTIVE", "min")).lower()
    except Exception:
        hobj = "min"
    try:
        hiters = int(globals().get("HEX_GLOBAL_ITERS", 20000))
    except Exception:
        hiters = 20000

    if hgo in ("checkerboard", "cb", "hex-checkerboard"):
        try:
            _seed = globals().get("OPT_SEED", "random")
            centers_cb = _kana_hex_collect_visible_centers(orient, S, step_x, step_y, margin, width, height, extend, r_used, c_used)
            if centers_cb:
                # 画像が不足する場合はここで明示的に巡回して揃える
                if len(images) < len(centers_cb) and len(images) > 0:
                    rep = (len(centers_cb) + len(images) - 1) // len(images)
                    images = (list(images) * rep)[:len(centers_cb)]
                images = reorder_hex_checkerboard_seed(list(images), centers_cb, step_x, step_y, seed=_seed)
        except Exception as e:
            _warn_exc_once(e)
            pass
    elif hgo in ("spectral-diagonal", "diag", "diagonal"):
        # スペクトル順 + タイル中心での対角割り当て
        # 方向は HEX_DIAG_DIR で選択：tlbr／brtl／trbl／bltr
        try:
            diag = str(globals().get("HEX_DIAGONAL_DIRECTION", globals().get("HEX_DIAG_DIR", "tlbr"))).lower()
            diag = diag.replace("_", "").replace("-", "")
            # 数値指定（1〜4）も受ける: 1 tlbr／2 brtl／3 trbl／4 bltr
            if diag in ("1", "tlbr"):
                diag = "tlbr"
            elif diag in ("2", "brtl"):
                diag = "brtl"
            elif diag in ("3", "trbl"):
                diag = "trbl"
            elif diag in ("4", "bltr"):
                diag = "bltr"
        except Exception:
            diag = "tlbr"
        try:
            # 注: Hexの対角グラデを安定させるため、可視中心点を「描画ループと同じ順序」で構築します
            # （内部関数の順序に依存すると、imagesの消費順とズレてグラデが崩れることがあります）
            centers_d = []
            cr_d = []  # (c, r) in draw-loop order
            try:
                if str(orient).lower() == "row-shift":
                    half_shift = float(S) / 2.0
                    min_r, max_r = -extend, int(r_used) + extend
                    min_c, max_c = -extend, int(c_used) + extend
                    for r in range(min_r, max_r):
                        shift = half_shift if (r % 2 != 0) else 0.0
                        y = float(margin) + float(int(round(r * float(step_y))))
                        for c in range(min_c, max_c):
                            x = float(margin) + float(int(round(shift + c * float(step_x))))
                            if not (x + S <= 0 or y + S <= 0 or x >= width or y >= height):
                                centers_d.append((x + float(S) / 2.0, y + float(S) / 2.0))
                                cr_d.append((c, r))
                else:  # col-shift
                    half_v = float(step_y) / 2.0
                    min_c, max_c = -extend, int(c_used) + extend
                    min_r, max_r = -extend, int(r_used) + extend
                    for c in range(min_c, max_c):
                        shift_y = half_v if (c % 2 != 0) else 0.0
                        x = float(margin) + float(int(round(c * float(step_x))))
                        for r in range(min_r, max_r):
                            y = float(margin) + float(int(round(shift_y + r * float(step_y))))
                            if not (x + S <= 0 or y + S <= 0 or x >= width or y >= height):
                                centers_d.append((x + float(S) / 2.0, y + float(S) / 2.0))
                                cr_d.append((c, r))
            except Exception:
                centers_d = _kana_hex_collect_visible_centers(
                    orient=orient,
                    S=S,
                    step_x=step_x,
                    step_y=step_y,
                    margin=margin,
                    width=W,
                    height=H,
                    extend=extend,
                    r_used=r_used,
                    c_used=c_used,
                    )
                cr_d = []

            try:
                note(f"HexDBG: diag_branch diag={diag} centers={len(centers_d)} images={len(images)}")
            except Exception as e:
                _kana_silent_exc('core:L9195', e)
                pass
            if centers_d:
                if len(images) > 0 and len(images) < len(centers_d):
                    rep = (len(centers_d) + len(images) - 1) // len(images)
                    images = (list(images) * rep)[:len(centers_d)]

                use_n = min(len(images), len(centers_d))
                # 対角グラデを“見た目で分かりやすく”するため、色空間の主成分(1軸)で 1D ソートします（より滑らか）
                try:
                    _imgs0 = list(images[:use_n])
                    vecs = [_avg_lab_vector(p) for p in _imgs0]
                    mx = sum(v[0] for v in vecs) / float(len(vecs))
                    my = sum(v[1] for v in vecs) / float(len(vecs))
                    mz = sum(v[2] for v in vecs) / float(len(vecs))
                    centered = [(v[0]-mx, v[1]-my, v[2]-mz) for v in vecs]
                    (d1x, d1y, d1z), _ = _pca1_direction(centered, iters=28)
                    # 見た目の対角グラデを分かりやすくするため、平均Labの L（明度）で単純ソートします（暗→明）
                    order = sorted(range(len(vecs)), key=lambda i: vecs[i][0])
                    sorted_imgs = [_imgs0[i] for i in order]
                except Exception:
                    # フォールバック（従来）
                    sorted_imgs = reorder_global_spectral_diagonal(list(images[:use_n]), objective=hobj)

                # 位置スコア: 推定グリッド座標(c,r)で対角スコア化（hexのずれを吸収）
                if cr_d:
                    cs = [v[0] for v in cr_d[:use_n]]
                    rs = [v[1] for v in cr_d[:use_n]]
                    min_c, max_c = min(cs), max(cs)
                    min_r, max_r = min(rs), max(rs)
                    dc = (max_c - min_c) if max_c > min_c else 1.0
                    dr = (max_r - min_r) if max_r > min_r else 1.0
                    cn = [(c - min_c) / dc for c in cs]
                    rn = [(r - min_r) / dr for r in rs]
                else:
                    # フォールバック: 物理座標
                    xs = [c[0] for c in centers_d[:use_n]]
                    ys = [c[1] for c in centers_d[:use_n]]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    dx = (max_x - min_x) if max_x > min_x else 1.0
                    dy = (max_y - min_y) if max_y > min_y else 1.0
                    cn = [(x - min_x) / dx for x in xs]
                    rn = [(y - min_y) / dy for y in ys]

                def score_cr(c, r, mode):
                    if mode == "tlbr":
                        return c + r
                    if mode == "trbl":
                        return (1.0 - c) + r
                    if mode == "bltr":
                        return c + (1.0 - r)
                    if mode == "brtl":
                        return (1.0 - c) + (1.0 - r)
                    return c + r

                scores = [score_cr(c, r, diag) for c, r in zip(cn, rn)]
                pos_order = sorted(range(use_n), key=lambda i: scores[i])

                assigned = [None] * use_n
                for k, pi in enumerate(pos_order):
                    assigned[pi] = sorted_imgs[k]

                images = list(images)
                images[:use_n] = assigned
            else:
                images = reorder_global_spectral_diagonal(list(images), objective=hobj)
        except Exception as e:
            _warn_exc_once(e)
            pass
    elif hgo in ("spectral-hilbert", "spectral", "hilbert"):
        try:
            images = reorder_global_spectral_hilbert(list(images), objective=hobj)
        except Exception as e:
            _warn_exc_once(e)
            pass
    elif hgo in ("anneal", "sa", "simulated-annealing"):
        try:
            _seed = globals().get("OPT_SEED", "random")
            images = reorder_global_anneal(list(images), objective=hobj, iters=hiters, seed=_seed)
        except Exception as e:
            _warn_exc_once(e)
            pass
    # ---- 局所最適化（6近傍） ----
    try:
        # 外部JSON/ランチャ用の簡易スイッチ（HEX_OPTIMIZER）で上書きできるようにする
        opt = str(globals().get("HEX_OPTIMIZER", "inherit") or "inherit").strip().lower()
        if opt in ("inherit", ""):
            do_local = bool(globals().get("HEX_LOCAL_OPT_ENABLE", False))
        else:
            do_local = opt not in ("off", "none", "false", "0", "disable", "disabled", "no")
    except Exception:
        do_local = bool(globals().get("HEX_LOCAL_OPT_ENABLE", False))

    if not do_local:
        return images

    centers = _kana_hex_collect_visible_centers(orient, S, step_x, step_y, margin, width, height, extend, r_used, c_used)
    if not centers:
        return images

    edges, neigh = _hex_neighbor_graph(centers, step_x, step_y, max_deg=int(globals().get("HEX_LOCAL_OPT_MAX_DEG", 6)))

    try:
        steps = int(globals().get("HEX_LOCAL_OPT_STEPS", 20000))
    except Exception:
        steps = 20000
    try:
        reheats = int(globals().get("HEX_LOCAL_OPT_REHEATS", 4))
    except Exception:
        reheats = 2
    try:
        objective = str(globals().get("HEX_LOCAL_OPT_OBJECTIVE", "inherit")).lower()
    except Exception:
        objective = "inherit"
    if objective in ("inherit", "global"):
        try:
            objective = str(globals().get("HEX_GLOBAL_OBJECTIVE", "min")).lower()
        except Exception:
            objective = "min"
    if objective not in ("min", "max"):
        objective = "min"
    try:
        seed = globals().get("HEX_LOCAL_OPT_SEED", None)
    except Exception:
        seed = None
    if seed is None:
        seed = globals().get("OPT_SEED", "random")
    try:
        T0 = float(globals().get("HEX_LOCAL_OPT_T0", 1.0))
        Tend = float(globals().get("HEX_LOCAL_OPT_TEND", 1e-3))
    except Exception:
        T0, Tend = 1.0, 1e-3

    use_n = min(len(images), len(centers))
    if use_n <= 3 or not edges or steps <= 0:
        return images

    # 足りない場合は、局所最適化の対象が可視タイル数に追いつくように拡張（wrap 依存を減らす）
    if use_n < len(centers) and len(images) > 0:
        rep = (len(centers) + len(images) - 1) // len(images)
        images = (list(images) * rep)[:len(centers)]
        use_n = min(len(images), len(centers))

    try:
        new_first, _ = optimize_hex_neighbors_anneal(images[:use_n], edges, neigh,
                                                     steps=steps, T0=T0, Tend=Tend,
                                                     reheats=reheats, seed=seed, objective=objective)
        images = list(images)
        images[:use_n] = new_first
    except Exception as e:
        _warn_exc_once(e)
        pass
    return images

def optimize_grid_spectral_hilbert(paths: List[Path], rows:int, cols:int, objective:str="min"):
    """
    色ベクトル→2D射影（numpy があれば PCA、無ければ LAB の (L,a) 簡易）→Hilbert順→格子充填。
    objective="min": 滑らか／"max": 逆順・蛇行でバラけ。
    """
    n = min(len(paths), rows*cols)
    paths = list(paths[:n])

    banner(_lang("最適化: Grid spectral→hilbert","Optimize: Grid spectral→hilbert"))

    # 色ベクトル抽出（進捗あり）
    vecs = []
    if VERBOSE and n>0: bar(0, n, prefix="feat   ", final=False)
    for i,p in enumerate(paths,1):
        vecs.append(_avg_lab_vector(p))
        if VERBOSE: bar(i, n, prefix="feat   ", final=(i==n))

    # 2D へ射影
    if np is not None:
        # PCA（主成分分析）
        if VERBOSE: bar(1, 1, prefix="pca    ", final=True)
        X = np.array(vecs, dtype=float)
        X = X - X.mean(axis=0, keepdims=True)
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        P = X @ VT[:2].T  # (n,2)
        mins = P.min(axis=0); maxs = P.max(axis=0); rng = np.where(maxs>mins, maxs-mins, 1.0)
        Q = (P - mins) / rng
        xs = (Q[:,0]*1023.999).astype(int).tolist()
        ys = (Q[:,1]*1023.999).astype(int).tolist()
    else:
        # 簡易：LABの(L,a) を 0..1 正規化
        Ls = [v[0] for v in vecs]; As=[v[1] for v in vecs]
        loL,hiL=min(Ls),max(Ls); loA,hiA=min(As),max(As)
        dL = (hiL-loL) if hiL>loL else 1.0
        dA = (hiA-loA) if hiA>loA else 1.0
        xs = [int(((L-loL)/dL)*1023.999) for L in Ls]
        ys = [int(((A-loA)/dA)*1023.999) for A in As]
        if VERBOSE: bar(1, 1, prefix="project", final=True)

    # Hilbert 索引化（進捗あり）
    ranks=[]
    if VERBOSE and n>0: bar(0, n, prefix="rank   ", final=False)
    for i in range(n):
        ranks.append((_hilbert_index(xs[i], ys[i], order=10), i))
        if VERBOSE: bar(i+1, n, prefix="rank   ", final=((i+1)==n))

    ranks.sort(key=lambda x:x[0])
    order = [i for _,i in ranks]

    # objective="max" の場合は蛇行で“バラけ”を少し増やす
    if objective=="max":
        grid = [order[r*cols:(r+1)*cols] for r in range(rows)]
        for r in range(rows):
            if r%2==1:
                grid[r].reverse()
        order = [x for row in grid for x in row]

    new_paths = [paths[i] for i in order]
    summary = {"grid_spectral_hilbert":{"objective":objective,"rows":rows,"cols":cols,"numpy": (np is not None)}}
    return new_paths, summary

# -----------------------------------------------------------------------------
# サブセクション: Mosaic バランス最適化（強化版）
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# サブセクション: Grid：Spectral→Diagonal Sweep（PCA→2D→対角スイープ）
# -----------------------------------------------------------------------------
def optimize_grid_spectral_diagonal(paths, rows:int, cols:int, objective:str="min",
                                    diagonal:str="tlbr", zigzag:bool=True):
    try:
        banner(_lang("最適化: Grid spectral→diagonal sweep", "Optimize: Grid spectral → diagonal sweep"))
    except Exception as e:
        _warn_exc_once(e)
        pass
    vecs = []
    try:
        _VERBOSE = bool(VERBOSE)
    except Exception:
        _VERBOSE = False
    if _VERBOSE and len(paths)>0:
        try: bar(0, len(paths), prefix="feat   ", final=False)
        except Exception: pass
    for i, p in enumerate(paths, 1):
        try:
            vecs.append(_avg_lab_vector(p))
        except Exception:
            try:
                vecs.append(_avg_rgb_vector(p))
            except Exception:
                vecs.append((0.5,0.5,0.5))
        if _VERBOSE:
            try: bar(i, len(paths), prefix="feat   ", final=(i==len(paths)))
            except Exception: pass
    try: _np = np
    except Exception: _np = None
    if _np is not None:
        if _VERBOSE:
            try: bar(1,1,prefix="pca    ", final=True)
            except Exception: pass
        X = _np.array(vecs, dtype=float)
        X = X - X.mean(axis=0, keepdims=True)
        U, S, VT = _np.linalg.svd(X, full_matrices=False)
        P = X @ VT[:2].T
        mins = P.min(axis=0); maxs = P.max(axis=0); rng = _np.where(maxs>mins, maxs-mins, 1.0)
        Q = (P - mins) / rng
        xs = Q[:,0].tolist(); ys = Q[:,1].tolist()
    else:
        Ls = [v[0] for v in vecs]; As = [v[1] for v in vecs]
        loL, hiL = min(Ls), max(Ls); loA, hiA = min(As), max(As)
        dL = (hiL - loL) if hiL>loL else 1.0
        dA = (hiA - loA) if hiA>loA else 1.0
        xs = [(L - loL)/dL for L in Ls]
        ys = [(A - loA)/dA for A in As]
    def score_xy(x, y, mode):
        if mode == "tlbr":   return x + y
        if mode == "trbl":   return (1.0 - x) + y
        if mode == "bltr":   return x + (1.0 - y)
        if mode == "brtl":   return (1.0 - x) + (1.0 - y)
        return x + y
    if isinstance(diagonal, str) and diagonal.lower()=="random":
        import random
        diagonal = random.choice(["tlbr","trbl","bltr","brtl"])
    svals = [score_xy(x, y, diagonal) for x, y in zip(xs, ys)]
    order = sorted(range(len(paths)), key=lambda i: svals[i], reverse=(objective=="max"))
    sorted_paths = [paths[i] for i in order]
    diag_coords = []
    for d in range(rows + cols - 1):
        band = []
        r_min = max(0, d - (cols - 1))
        r_max = min(rows - 1, d)
        for r in range(r_min, r_max + 1):
            c = d - r
            if 0 <= c < cols:
                band.append((r, c))
        if zigzag and (d % 2 == 1):
            band.reverse()
        diag_coords.extend(band)
    grid = [None]*(rows*cols)
    for k, (r, c) in enumerate(diag_coords):
        if k>=len(sorted_paths): break
        grid[r*cols + c] = sorted_paths[k]
    new_paths = [p for p in grid if p is not None]
    return new_paths, {"grid_spectral_diagonal":{"rows":rows,"cols":cols,"diagonal":diagonal,"zigzag":zigzag}}


def _row_width(row, gutter):
    if not row: return 0
    return sum(w for _,_,w in row) + gutter*(len(row)-1)

def _col_height(col, gutter):
    if not col: return 0
    return sum(h for _,_,h in col) + gutter*(len(col)-1)

def optimize_rows_hillclimb(rows, W, gutter, iters=1500, show_progress=None, restarts_left=2):
    OVERFLOW_W = 1_000_000.0
    sp = VERBOSE if show_progress is None else show_progress
    # 無限再スタートを防ぐ（restarts_left が尽きたら打ち切り）
    if restarts_left <= 0:
        return rows
    STDDEV_W   = 0.50
    rows = [(list(r), h) for (r,h) in rows]
    widths = [_row_width(r, gutter) for r,_ in rows]
    def score(ww):
        if not ww: return 0.0
        overflow = sum(max(0.0, w - W) for w in ww)
        mx = max(ww); mean = sum(ww)/len(ww)
        var = sum((w-mean)*(w-mean) for w in ww)/len(ww)
        return OVERFLOW_W*overflow + mx + STDDEV_W*math.sqrt(var)
    best = score(widths); initial = best
    accepted = 0; noimp = 0
    rnd = random.Random(OPT_SEED if OPT_SEED!="random" else secrets.randbits(32))
    def delta_remove_len(L): return gutter if L>=2 else 0
    def delta_add_len(L):    return gutter if L>=1 else 0
    if sp: banner(_lang("最適化: Mosaic balance（行）","Optimize: Mosaic balance (rows)"))
    for t in range(iters):
        if not rows: break
        hi = max(range(len(rows)), key=lambda i: widths[i])
        lo = min(range(len(rows)), key=lambda i: widths[i])
        if hi==lo or not rows[hi][0]:
            noimp += 1
            if sp: bar(t+1, iters, prefix="opt-bal", final=(t+1==iters))
            if BALANCE_EARLY_STOP and noimp>max(100, iters//5): break
            continue
        # 移動（長い行→短い行へ要素を移す）
        cand_idx = rnd.sample(range(len(rows[hi][0])), k=min(4, len(rows[hi][0])))
        cur_best = best; best_move=None
        for idx in cand_idx:
            item = rows[hi][0][idx]
            dw_hi = item[2] + delta_remove_len(len(rows[hi][0]))
            dw_lo = item[2] + delta_add_len(len(rows[lo][0]))
            tmp = widths[:]; tmp[hi]=widths[hi]-dw_hi; tmp[lo]=widths[lo]+dw_lo
            sc = score(tmp)
            if sc <= cur_best - 1e-9:
                cur_best=sc; best_move=("move", idx, tmp[hi], tmp[lo], sc)
        if best_move:
            _, idx, new_hi_w, new_lo_w, sc = best_move
            item = rows[hi][0].pop(idx); rows[lo][0].append(item)
            widths[hi]=new_hi_w; widths[lo]=new_lo_w
            best = sc; accepted += 1; noimp = 0
        else:
            # 交換（行同士で 1 要素ずつ入れ替え）
            if rows[lo][0]:
                i = rnd.randrange(len(rows[hi][0])); j = rnd.randrange(len(rows[lo][0]))
                a = rows[hi][0][i]; b = rows[lo][0][j]
                new_hi_w = widths[hi] - a[2] + b[2]
                new_lo_w = widths[lo] - b[2] + a[2]
                tmp = widths[:]; tmp[hi]=new_hi_w; tmp[lo]=new_lo_w
                sc = score(tmp)
                if sc <= best - 1e-9:
                    rows[hi][0][i], rows[lo][0][j] = b, a
                    widths[hi]=new_hi_w; widths[lo]=new_lo_w
                    best = sc; accepted += 1; noimp=0
                else:
                    noimp += 1
            else:
                noimp += 1
        if sp: bar(t+1, iters, prefix="opt-bal", final=(t+1==iters))
        if BALANCE_EARLY_STOP and noimp>max(200, iters//4):
            if sp: note("  (early stop: rows balance stalled)")
            break
    if BALANCE_RESTART_ON_STALL and accepted < max(3, len(rows)//6):
        rnd = random.Random(OPT_SEED if OPT_SEED!="random" else secrets.randbits(32))
        for _ in range(min(10, sum(len(r) for r,_ in rows))):
            hi = rnd.randrange(len(rows)); lo = rnd.randrange(len(rows))
            if hi==lo or not rows[hi][0] or not rows[lo][0]: continue
            i = rnd.randrange(len(rows[hi][0])); j = rnd.randrange(len(rows[lo][0]))
            rows[hi][0][i], rows[lo][0][j] = rows[lo][0][j], rows[hi][0][i]
        return optimize_rows_hillclimb(rows, W, gutter, iters=max(200, iters//3), show_progress=show_progress, restarts_left=restarts_left-1)
    imp = ((initial-best)/initial*100.0) if initial>0 else 0.0
    note(f"Row balance: {initial:.1f} → {best:.1f} ({imp:+.1f}%) / accepted {accepted}/{t+1}")
    return [(rows[i][0], rows[i][1]) for i in range(len(rows))]

def optimize_cols_hillclimb(cols, H, gutter, iters=1500, show_progress=None, restarts_left=2):
    OVERFLOW_W = 1_000_000.0
    sp = VERBOSE if show_progress is None else show_progress
    # 無限再スタートを防ぐ（restarts_left が尽きたら打ち切り）
    if restarts_left <= 0:
        return cols
    STDDEV_W   = 0.50
    cols=[list(c) for c in cols]
    heights=[_col_height(c,gutter) for c in cols]
    def score(hh):
        if not hh: return 0.0
        overflow = sum(max(0.0, h - H) for h in hh)
        mx = max(hh); mean = sum(hh)/len(hh)
        var = sum((h-mean)*(h-mean) for h in hh)/len(hh)
        return OVERFLOW_W*overflow + mx + STDDEV_W*math.sqrt(var)
    best = score(heights); initial = best
    accepted = 0; noimp = 0
    rnd = random.Random(OPT_SEED if OPT_SEED!="random" else secrets.randbits(32))
    def delta_remove_len(L): return gutter if L>=2 else 0
    def delta_add_len(L):    return gutter if L>=1 else 0
    if sp: banner(_lang("最適化: Mosaic balance（列）","Optimize: Mosaic balance (cols)"))
    for t in range(iters):
        if not cols: break
        hi = max(range(len(cols)), key=lambda i: heights[i])
        lo = min(range(len(cols)), key=lambda i: heights[i])
        if hi==lo or not cols[hi]:
            noimp += 1
            if sp: bar(t+1, iters, prefix="opt-bal", final=(t+1==iters))
            if BALANCE_EARLY_STOP and noimp>max(100, iters//5): break
            continue
        # 移動（長い列→短い列へ要素を移す）
        cand_idx = rnd.sample(range(len(cols[hi])), k=min(4, len(cols[hi])))
        best_move=None; cur_best=best
        for idx in cand_idx:
            item = cols[hi][idx]
            dh_hi = item[2] + delta_remove_len(len(cols[hi]))
            dh_lo = item[2] + delta_add_len(len(cols[lo]))
            tmp = heights[:]; tmp[hi]=heights[hi]-dh_hi; tmp[lo]=heights[lo]+dh_lo
            sc = score(tmp)
            if sc <= cur_best - 1e-9:
                cur_best=sc; best_move=("move", idx, tmp[hi], tmp[lo], sc)
        if best_move:
            _, idx, new_hi_h, new_lo_h, sc = best_move
            item = cols[hi].pop(idx); cols[lo].append(item)
            heights[hi]=new_hi_h; heights[lo]=new_lo_h
            best = sc; accepted += 1; noimp = 0
        else:
            # 交換（列同士で 1 要素ずつ入れ替え）
            if cols[lo]:
                i = rnd.randrange(len(cols[hi])); j = rnd.randrange(len(cols[lo]))
                a = cols[hi][i]; b = cols[lo][j]
                new_hi_h = heights[hi] - a[2] + b[2]
                new_lo_h = heights[lo] - b[2] + a[2]
                tmp = heights[:]; tmp[hi]=new_hi_h; tmp[lo]=new_lo_h
                sc = score(tmp)
                if sc <= best - 1e-9:
                    cols[hi][i], cols[lo][j] = b, a
                    heights[hi]=new_hi_h; heights[lo]=new_lo_h
                    best = sc; accepted += 1; noimp = 0
                else:
                    noimp += 1
            else:
                noimp += 1
        if sp: bar(t+1, iters, prefix="opt-bal", final=(t+1==iters))
        if BALANCE_EARLY_STOP and noimp>max(200, iters//4):
            if sp: note("  (early stop: cols balance stalled)")
            break
    if BALANCE_RESTART_ON_STALL and accepted < max(3, len(cols)//6):
        rnd = random.Random(OPT_SEED if OPT_SEED!="random" else secrets.randbits(32))
        for _ in range(min(10, sum(len(c) for c in cols))):
            hi = rnd.randrange(len(cols)); lo = rnd.randrange(len(cols))
            if hi==lo or not cols[hi] or not cols[lo]: continue
            i = rnd.randrange(len(cols[hi])); j = rnd.randrange(len(cols[lo]))
            cols[hi][i], cols[lo][j] = cols[lo][j], cols[hi][i]
        return optimize_cols_hillclimb(cols, H, gutter, iters=max(200, iters//3), show_progress=show_progress, restarts_left=restarts_left-1)
    imp = ((initial-best)/initial*100.0) if initial>0 else 0.0
    note(f"列バランス: {initial:.1f} → {best:.1f} ({imp:+.1f}%) / 採用 {accepted}/{t+1}")
    return cols

# -----------------------------------------------------------------------------
# サブセクション: Mosaic 色差ヒルクライム（swap／2opt）
# -----------------------------------------------------------------------------
def _seq_adj_sum(order: List[int], vecs: List[Tuple[float,float,float]]) -> float:
    if len(order)<2: return 0.0
    s=0.0
    for i in range(len(order)-1):
        s += _vec_dist(vecs[order[i]], vecs[order[i+1]])
    return s

def _local_contrib(order: List[int], vecs, pos: int) -> float:
    n=len(order); s=0.0
    if 0<=pos-1<n and 0<=pos<n:   s+=_vec_dist(vecs[order[pos-1]], vecs[order[pos]])
    if 0<=pos<n   and 0<=pos+1<n: s+=_vec_dist(vecs[order[pos]],   vecs[order[pos+1]])
    return s

def _optimize_sequence(order: List[int], vecs: List[Tuple[float,float,float]],
                       iters: int, objective: str, rng: random.Random,
                       prog: dict|None):
    curr = _seq_adj_sum(order, vecs)
    best = -curr if objective=="max" else curr
    init = curr; accepted=0
    for _ in range(iters):
        a=rng.randrange(len(order)); b=rng.randrange(len(order))
        if a==b:
            if prog:
                prog["i"]+=1
                if VERBOSE: bar(prog["i"], prog["n"], prefix=prog.get("prefix","opt-col"), final=(prog["i"]==prog["n"]))
            continue
        if a>b: a,b=b,a
        old=_local_contrib(order, vecs, a)+_local_contrib(order, vecs, b)
        if b==a+1: old -= _vec_dist(vecs[order[a]], vecs[order[b]])
        order[a], order[b] = order[b], order[a]
        new=_local_contrib(order, vecs, a)+_local_contrib(order, vecs, b)
        if b==a+1: new -= _vec_dist(vecs[order[a]], vecs[order[b]])
        new_sum = curr - old + new
        new_cost = -new_sum if objective=="max" else new_sum
        if new_cost <= best:
            curr=new_sum; best=new_cost; accepted+=1
        else:
            order[a], order[b] = order[b], order[a]
        if prog:
            prog["i"]+=1
            if VERBOSE: bar(prog["i"], prog["n"], prefix=prog.get("prefix","opt-col"), final=(prog["i"]==prog["n"]))
    imp = ((curr-init)/init*100.0) if objective=="max" and init>0 else \
          ((init-curr)/init*100.0) if objective=="min" and init>0 else 0.0
    return order, init, curr, imp, accepted

def _optimize_sequence_2opt(order, vecs, iters, objective, rng, prog):
    def seq_sum(ordr): return _seq_adj_sum(ordr, vecs)
    curr = seq_sum(order)
    best_cost = -curr if objective=="max" else curr
    accepted = 0
    n = len(order)
    for _ in range(iters):
        if n < 4:
            if prog:
                prog["i"]+=1
                if VERBOSE: bar(prog["i"], prog["n"], prefix=prog.get("prefix","opt-col"), final=(prog["i"]==prog["n"]))
            continue
        a = rng.randrange(0, n-2)
        b = rng.randrange(a+2, n)
        new = order[:a+1] + list(reversed(order[a+1:b])) + order[b:]
        new_sum = _seq_adj_sum(new, vecs)
        new_cost = -new_sum if objective=="max" else new_sum
        if new_cost <= best_cost:
            order = new; curr = new_sum; best_cost = new_cost; accepted += 1
        if prog:
            prog["i"]+=1
            if VERBOSE: bar(prog["i"], prog["n"], prefix=prog.get("prefix","opt-col"), final=(prog["i"]==prog["n"]))
    return order, curr, accepted

def optimize_rows_color_neighbors(rows, objective="max", iters_per_line=200, seed=0):
    rng = random.Random(seed if seed!="random" else secrets.randbits(32))
    tot_iters = sum(max(0, iters_per_line if len(r)>1 else 0) for r,_ in rows)
    # 2opt も回すなら加算
    if MOSAIC_SEQ_ALGO in ("2opt","swap+2opt"):
        tot_iters += sum(max(0, iters_per_line if len(r)>1 else 0) for r,_ in rows)
    prog={"i":0,"n":max(1,tot_iters),"prefix":"opt-row"}
    # Mosaic の行内並び最適化のバナー/目的を表示します（UI_LANG='en' の場合は英語）。
    if str(globals().get('UI_LANG','')).lower() == 'en':
        banner("Optimize: Mosaic color diff (row order)")
        obj_label = "maximize (diversify)" if objective == "max" else "minimize (similarize)"
        note(f"Objective: {obj_label} / per row {iters_per_line}")
    else:
        banner(_lang("最適化: Mosaic color diff（行の並び）","Optimize: Mosaic color diff (row order)"))
        note(f"Objective: {'maximize (spread)' if objective=='max' else 'minimize (similar)' } / per row {iters_per_line}")
    # 画像ごとの平均LABベクトルをまとめて確保
    lab_cache = {}
    for r,_ in rows:
        for (p,_,_) in r:
            if p not in lab_cache:
                lab_cache[p] = _avg_lab_vector(p)
    total_init=0.0; total_final=0.0; total_acc=0
    for ridx,(row,h) in enumerate(rows):
        if len(row)<2: continue
        vecs=[lab_cache[p] for (p,_,_) in row]
        order=list(range(len(row)))
        order, ini, fin, imp, acc = _optimize_sequence(order, vecs, iters_per_line, objective, rng, prog)
        if MOSAIC_SEQ_ALGO in ("2opt","swap+2opt"):
            order, fin2, acc2 = _optimize_sequence_2opt(order, vecs, iters_per_line, objective, rng, prog)
            acc += acc2; fin=fin2
        rows[ridx]=([row[i] for i in order], h)
        total_init += ini; total_final += fin; total_acc += acc
    if VERBOSE: bar(prog["n"], prog["n"], prefix=prog.get("prefix","opt"), final=True)
    imp=((total_final-total_init)/total_init*100.0) if objective=="max" and total_init>0 else \
        ((total_init-total_final)/total_init*100.0) if objective=="min" and total_init>0 else 0.0
    # 行最適化の合計結果（ΣΔ色と採用数）を UI_LANG に応じて表示します
    if str(globals().get('UI_LANG','')).lower() == 'en':
        note(f"ΣΔcolor(row): {total_init:.1f} → {total_final:.1f} ({imp:+.1f}%) / accepted {total_acc}")
    else:
        note(f"ΣΔColor(row): {total_init:.1f} → {total_final:.1f} ({imp:+.1f}%) / accepted {total_acc}")
    return rows, {"rows_adj_initial":total_init,"rows_adj_final":total_final,"rows_adj_imp_pct":imp,"objective":objective}

def optimize_cols_color_neighbors(cols, objective="max", iters_per_line=200, seed=0):
    rng = random.Random(seed if seed!="random" else secrets.randbits(32))
    tot_iters = sum(max(0, iters_per_line if len(c)>1 else 0) for c in cols)
    if MOSAIC_SEQ_ALGO in ("2opt","swap+2opt"):
        tot_iters += sum(max(0, iters_per_line if len(c)>1 else 0) for c in cols)
    prog={"i":0,"n":max(1,tot_iters),"prefix":"opt-col"}
    # Mosaic の列内並び最適化のバナー/目的を表示します（UI_LANG='en' の場合は英語）。
    if str(globals().get('UI_LANG','')).lower() == 'en':
        banner("Optimize: Mosaic color diff (column order)")
        obj_label = "maximize (diversify)" if objective == "max" else "minimize (similarize)"
        note(f"Objective: {obj_label} / per column {iters_per_line}")
    else:
        banner(_lang("最適化: Mosaic color diff（列の並び）","Optimize: Mosaic color diff (col order)"))
        note(f"Objective: {'maximize (spread)' if objective=='max' else 'minimize (similar)' } / per col {iters_per_line}")
    total_init=0.0; total_final=0.0; total_acc=0
    for cidx,col in enumerate(cols):
        if len(col)<2: continue
        vecs=[_avg_lab_vector(p) for (p,_,_) in col]
        order=list(range(len(col)))
        order, ini, fin, imp, acc = _optimize_sequence(order, vecs, iters_per_line, objective, rng, prog)
        if MOSAIC_SEQ_ALGO in ("2opt","swap+2opt"):
            order, fin2, acc2 = _optimize_sequence_2opt(order, vecs, iters_per_line, objective, rng, prog)
            acc += acc2; fin=fin2
        cols[cidx]=[col[i] for i in order]
        total_init += ini; total_final += fin; total_acc += acc
    if VERBOSE: bar(prog["n"], prog["n"], prefix=prog.get("prefix","opt"), final=True)
    imp=((total_final-total_init)/total_init*100.0) if objective=="max" and total_init>0 else \
        ((total_init-total_final)/total_init*100.0) if objective=="min" and total_init>0 else 0.0
    # 列最適化の合計結果（ΣΔ色と採用数）を UI_LANG に応じて表示します
    if str(globals().get('UI_LANG','')).lower() == 'en':
        note(f"ΣΔcolor(column): {total_init:.1f} → {total_final:.1f} ({imp:+.1f}%) / accepted {total_acc}")
    else:
        note(f"ΣΔColor(col): {total_init:.1f} → {total_final:.1f} ({imp:+.1f}%) / accepted {total_acc}")
    return cols, {"cols_adj_initial":total_init,"cols_adj_final":total_final,"cols_adj_imp_pct":imp,"objective":objective}

# -----------------------------------------------------------------------------
# サブセクション: レイアウト（grid／mosaic）
# -----------------------------------------------------------------------------
def compute_grid(n:int, width:int, height:int, rows:Optional[int], cols:Optional[int]):
    if rows is None and cols is None:
        aspect = width/height; cols=max(1,int(round(math.sqrt(n*aspect)))); rows=int(math.ceil(n/cols))
    elif rows is None:
        cols=max(1,cols); rows=int(math.ceil(n/cols))
    elif cols is None:
        rows=max(1,rows); cols=int(math.ceil(n/rows))
    else:
        rows=max(1,rows); cols=max(1,cols)
    return rows, cols

def compute_cell_sizes(width:int, height:int, rows:int, cols:int, margin:int, gutter:int):
    total_w = width - margin*2 - gutter*(cols-1)
    total_h = height - margin*2 - gutter*(rows-1)
    base_w, rem_w = divmod(total_w, cols)
    base_h, rem_h = divmod(total_h, rows)
    col_w = [base_w + (1 if i<rem_w else 0) for i in range(cols)]
    row_h = [base_h + (1 if i<rem_h else 0) for i in range(rows)]
    return row_h, col_w


# =============================================================================
# セクション: レイアウト生成（grid／mosaic／hex）
# =============================================================================

# =============================================================================
# セクション: レイアウト: Grid
# - レイアウト生成の本体（配置/合成/マスクなど）
# =============================================================================

def layout_grid(images: List[Path], width:int, height:int, margin:int, gutter:int,
                rows:Optional[int], cols:Optional[int], mode:str, bg_rgb:Tuple[int,int,int]):
    """ROWS×COLS の等間隔グリッドに画像を並べるレイアウト関数。

    images: 使用する画像ファイルパスのリスト
    width, height: 出力キャンバスのサイズ
    margin, gutter: 外枠の余白・セル間のすき間
    rows, cols: グリッドの行数・列数
    mode: リサイズモード (cover／contain など)
    bg_rgb: 背景色 (R, G, B)
    """
    # レイアウト情報（1回だけ）を表示 (grid)
    global _PRINTED_LAYOUT_ONCE
    if not _PRINTED_LAYOUT_ONCE:
        try:
            note(f"LAYOUT: grid | ROWS×COLS: {rows}×{cols}")
        except Exception as e:
            _warn_exc_once(e)
            pass
        _PRINTED_LAYOUT_ONCE = True

    rows, cols = compute_grid(len(images), width, height, rows, cols)
    # face-focus のデバッグカウンタは、Grid 描画の開始時にリセットします。
    # このレイアウト分だけ数えるためで、hex 描画と同じ「描画前に 0 クリア」の挙動に合わせています。
    if globals().get("FACE_FOCUS_ENABLE", True) and globals().get("GRID_FACE_FOCUS_ENABLE", False):
        try:
            global _FDBG, _FDBG2
            _FDBG = {"cv2": None, "frontal":0, "profile":0, "anime":0, "ai":0, "upper":0, "person":0, "saliency":0, "center":0,
                     "reject_pos":0, "reject_ratio":0, "errors":0}
            _FDBG2 = {"eyes_ok":0, "eyes_ng":0, "low_reject":0,
                      "anime_face_ok":0, "anime_face_ng":0,
                      "anime_eyes_ok":0, "anime_eyes_ng":0, "ai_face_ok":0, "ai_face_ng":0}
        except Exception as e:
            _warn_exc_once(e)
            pass
    _preserve = bool(globals().get("PRESERVE_INPUT_ORDER", False))
    # 完全シャッフル（ARRANGE_FULL_SHUFFLE）が有効なら、レイアウト前に images を全体シャッフルします。
    # OPT_SEED が固定値なら再現性あり（"random" なら毎回変化）。
    # この関数では、完全シャッフル時は tempo 並び替え（_tempo_apply）は行いません。
    if images and (not _preserve) and bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
        try:
            _seed = globals().get("OPT_SEED", None)
            # 乱数器の内部状態に依存しない“ハッシュシャッフル”で、最終集合を一度だけ並べ替える
            hash_shuffle_inplace(images, _seed, salt="grid_fullshuffle")
        except Exception as e:
            _warn_exc_once(e)
            pass
        # Full shuffle の状態表示（重複しないよう英語で1回だけ）
        try:
            if bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
                if bool(globals().get("_MOSAIC_POST_DISABLED_BY_FULLSHUFFLE", False)):
                    note("Full shuffle: ON (Mosaic POST assignment disabled)")
                else:
                    note("Full shuffle: ON")
        except Exception as e:
            _warn_exc_once(e)
            pass
    layout_info={}
    # 完全シャッフルが無効な場合のみ、近傍色差の最適化（選択した方式）を適用します。
    if (not _preserve) and (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and GRID_NEIGHBOR_OBJECTIVE in ("min","max"):
        seed = OPT_SEED if OPT_SEED!="random" else secrets.randbits(32)
        if GRID_OPTIMIZER == "anneal":
            images, summ = optimize_grid_neighbors_anneal(
                images, rows, cols,
                steps=GRID_ANNEAL_STEPS, T0=GRID_ANNEAL_T0, Tend=GRID_ANNEAL_TEND,
                reheats=GRID_ANNEAL_REHEATS, seed=seed, objective=GRID_NEIGHBOR_OBJECTIVE
            )
        elif GRID_OPTIMIZER == "checkerboard":
            images, summ = optimize_grid_checkerboard(
                images, rows, cols, seed=seed, objective=GRID_NEIGHBOR_OBJECTIVE
            )
        elif GRID_OPTIMIZER == "spectral-hilbert":
            images, summ = optimize_grid_spectral_hilbert(
                images, rows, cols, objective=GRID_NEIGHBOR_OBJECTIVE
            )
        elif GRID_OPTIMIZER == "spectral-diagonal":
            try: diag = GRID_DIAGONAL_DIRECTION
            except NameError: diag = "random"
            try: zz = bool(GRID_DIAGONAL_ZIGZAG)
            except NameError: zz = True
            try: note(f"layout: spectral-diagonal | diag={diag} | zigzag={zz}")
            except Exception: pass
            images, summ = optimize_grid_spectral_diagonal(
                images, rows, cols, objective=GRID_NEIGHBOR_OBJECTIVE,
                diagonal=diag, zigzag=zz
            )
        else:
            images, summ = optimize_grid_neighbors(
                images, rows, cols, iters=OPT_ITERS, seed=seed, objective=GRID_NEIGHBOR_OBJECTIVE
            )
        layout_info.update(summ)

        # Tune のときは、ベース並び（spectral/checkerboard 等）の後に
        # 追加でanneal(anneal)を回して、近傍の色差目的をさらに詰めます。
        # （従来は GRID_OPTIMIZER == "anneal" のときしか走らず、Tune が効かないケースがありました）
        if bool(globals().get("GRID_ANNEAL_ENABLE", False)) and GRID_OPTIMIZER != "anneal":
            try:
                if int(GRID_ANNEAL_STEPS) > 0:
                    images, summ2 = optimize_grid_neighbors_anneal(
                        images, rows, cols,
                        steps=GRID_ANNEAL_STEPS, T0=GRID_ANNEAL_T0, Tend=GRID_ANNEAL_TEND,
                        reheats=GRID_ANNEAL_REHEATS, seed=seed, objective=GRID_NEIGHBOR_OBJECTIVE
                    )
                    layout_info.update(summ2)
            except Exception as e:
                _warn_exc_once(e)
                pass
    row_h, col_w = compute_cell_sizes(width, height, rows, cols, margin, gutter)
    # （完全シャッフルが無効な場合）シャッフル/色最適化の後に tempo 並び替えを適用します。
    if (not _preserve) and (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and images:
        try:
            images = _tempo_apply(images)
        except Exception as e:
            _warn_exc_once(e)
            pass
    canvas=Image.new("RGB",(width,height),bg_rgb); mask=Image.new("L",(width,height),0)
    banner(_lang("処理中: Grid","Rendering: Grid"))
    total=min(len(images), rows*cols); done=0
    # --- draw prefetch（CPU）：ジョブを組んでから、スレッドでタイル描画します ---
    jobs = []  # (path, x, y, w, h)
    y = margin
    idx = 0
    for r in range(rows):
        x = margin
        for c in range(cols):
            if idx >= len(images):
                break
            w, h = col_w[c], row_h[r]
            try:
                p = images[idx]
            except Exception:
                p = images[idx]
            jobs.append((p, x, y, int(w), int(h)))
            x += w + gutter
            idx += 1
        y += row_h[r] + gutter

    _pf_ahead = int(max(0, int(globals().get('DRAW_PREFETCH_AHEAD', 16))))

    _pf_ahead = _effective_draw_prefetch_ahead(width, height, _pf_ahead)
    _pf_workers = int(max(1, int(globals().get('DRAW_PREFETCH_WORKERS', 0) or (os.cpu_count() or 4))))
    _pf_on = bool(globals().get('DRAW_PREFETCH_ENABLE', True)) and (_pf_ahead > 0)

    _grid_use_ff = (mode == 'fill' and bool(globals().get('GRID_FACE_FOCUS_ENABLE', False))
                    and bool(globals().get('FACE_FOCUS_ENABLE', True)))

    def _grid_render(job):
        p, _x, _y, w, h = job
        if _grid_use_ff:
            tile = _tile_render_cached(p, w, h, 'fill', use_face_focus=True)
            return 'fill_ff', tile
        if mode == 'fit':
            tile = _tile_render_cached(p, w, h, 'fit', use_face_focus=False)
            return 'fit', tile
        tile = _tile_render_cached(p, w, h, 'fill', use_face_focus=False)
        return 'fill', tile

    done = 0
    _pf_backend = str(globals().get('DRAW_PREFETCH_BACKEND', ('process' if os.name == 'nt' else 'thread'))).lower()
    _pf_use_mp = _pf_backend in ('process', 'mp', 'multiprocess', 'proc', 'processpool', 'process_pool')

    if _pf_on and jobs:
        if _pf_use_mp:
            try:
                _pf_items = [(job[0], job[3], job[4], mode, _grid_use_ff) for job in jobs]
                _pf_stream = prefetch_ordered_mp_safe(_pf_items, _pf_worker_grid_render, ahead=_pf_ahead, max_workers=_pf_workers)

                def _wrap_pf():
                    for i, (_item, _res, _exc) in enumerate(_pf_stream):
                        yield jobs[i], _res, _exc

                _it = _wrap_pf()
            except Exception as _e_pf:
                print(f"[WARN] process prefetch unavailable; fallback to thread. reason={_e_pf}")
                _it = prefetch_ordered_safe(jobs, _grid_render, ahead=_pf_ahead, max_workers=_pf_workers)
        else:
            _it = prefetch_ordered_safe(jobs, _grid_render, ahead=_pf_ahead, max_workers=_pf_workers)
    else:
        _it = ((job, _grid_render(job), None) for job in jobs)

    for job, out, exc in _it:
        p, x, y, w, h = job
        try:
            if exc is not None:
                raise exc
            kind, tile = out
            if kind == 'fit':
                rx = x + (w - tile.size[0]) // 2
                ry = y + (h - tile.size[1]) // 2
                canvas.paste(tile, (rx, ry))
                mask.paste(255, (rx, ry, rx + tile.size[0], ry + tile.size[1]))
            else:
                canvas.paste(tile, (x, y))
                mask.paste(255, (x, y, x + w, y + h))
        except Exception as e:
            # フォールバック（旧パス）
            try:
                with open_image_safe(p) as im:
                    paste_cell(canvas, mask, im, x, y, w, h, mode)
            except Exception as e2:
                print(f"[WARN] {p}: {e2}")
        done = min(done + 1, total)
        if VERBOSE:
            bar(done, max(1, total), prefix='draw   ', final=(done == total))

    # --- /draw prefetch（CPU） ---
    # 画像が 0 枚のときでも進捗バーを確実に閉じる（未定義変数参照を避ける）
    if total == 0:
        bar(done, 1, prefix="draw   ", final=True)
    # face-focus のデバッグ（grid+fill のとき、FACE_FOCUS_DEBUG=True なら検出カウンタを表示）
    try:
        if (mode == "fill" and globals().get("GRID_FACE_FOCUS_ENABLE", False)
                and globals().get("FACE_FOCUS_ENABLE", True)
                and globals().get("FACE_FOCUS_DEBUG", False)):
            _note_face_focus_stats(_FDBG, _FDBG2)
    except Exception as e:
        _warn_exc_once(e)
        pass
    return canvas, mask, layout_info, rows, cols

def _load_ars(paths: List[Path]) -> List[Tuple[Path,float]]:
    out=[]
    for p in paths:
        try:
            with open_image_safe(p) as im:
                ar = im.width/max(1,im.height)
        except Exception:
            ar=1.0
        out.append((p,ar))
    return out


# =============================================================================
# セクション: レイアウト: Mosaic (uniform-height)
# - レイアウト生成の本体（配置/合成/マスクなど）
# =============================================================================

def layout_mosaic_uniform_height(paths: List[Path], width: int, height: int, margin: int, gutter: int,
                                 bg_rgb: Tuple[int, int, int]):
    """行の高さを一定にし、横方向へ詰めていくモード（隙間少なめ・サイズ感均一）"""
    # レイアウト情報（1回だけ）を表示
    global _PRINTED_LAYOUT_ONCE
    if not _PRINTED_LAYOUT_ONCE:
        try:
            uh_assign = globals().get('MOSAIC_UH_ASSIGN', globals().get('MOSAIC_UW_ASSIGN', '(n/a)'))
            uh_order = globals().get('MOSAIC_UH_ROW_ORDER', None)
            if not uh_order:
                uh_order = 'avgLAB' if bool(globals().get('MOSAIC_UH_ORDER_ROWS', False)) else 'none'
            _mprof = str(globals().get("MOSAIC_ENHANCE_PROFILE", "off")).strip()
            _mpost = ""
            if _mosaic_enhance_active() and _mprof and _mprof.lower() not in ("off", "none", "random"):
                _mpost = f" | POST: {_mprof}"
            note(f"LAYOUT: mosaic-uniform-height | ASSIGN: {uh_assign} | ORDER: {uh_order}{_mpost}")
        except Exception as e:
            _warn_exc_once(e)
            pass
        _PRINTED_LAYOUT_ONCE = True
    _preserve = bool(globals().get('PRESERVE_INPUT_ORDER', False))
    # 完全シャッフル（ARRANGE_FULL_SHUFFLE）が有効なら、Mosaic の処理前に paths を全体シャッフルします。
    # OPT_SEED が固定値なら再現性あり（"random" なら毎回変化）。
    if bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
        try:
            _seed = globals().get("OPT_SEED", None)
            paths = hash_shuffle(list(paths), _seed, salt="mosaic_uh_fullshuffle")
        except Exception as e:
            _warn_exc_once(e)
            pass
        # Full shuffle の状態表示（重複しないよう英語で1回だけ）
        try:
            if bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
                if bool(globals().get("_MOSAIC_POST_DISABLED_BY_FULLSHUFFLE", False)):
                    note("Full shuffle: ON (Mosaic POST assignment disabled)")
                else:
                    note("Full shuffle: ON")
        except Exception as e:
            _warn_exc_once(e)
            pass
    # まずグローバル順序の並べ替えを適用（この後のパックに効かせるため）。
    # 全シャッフルが有効な場合はスキップします。
    if (not _preserve) and (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and (not bool(globals().get("MOSAIC_ENHANCE_ENABLE", False))):
        if MOSAIC_GLOBAL_ORDER == "spectral-hilbert":
            paths = reorder_global_spectral_hilbert(list(paths), objective=MOSAIC_GLOBAL_OBJECTIVE)
        elif MOSAIC_GLOBAL_ORDER == "anneal":
            paths = reorder_global_anneal(
                list(paths), objective=MOSAIC_GLOBAL_OBJECTIVE,
                iters=MOSAIC_GLOBAL_ITERS, seed=OPT_SEED
            )

    # （完全シャッフルが無効な場合）グローバル並び替えの後、行詰め込み前に tempo 並び替えを適用します。
    if (not _preserve) and (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and paths:
        try:
            paths = _tempo_apply(paths)
        except Exception as e:
            _warn_exc_once(e)
            pass
    W = width - margin*2
    H = height - margin*2
    ars = _load_ars(paths)

    # --- 固定行数を使用する場合のパッキング ---
    # MOSAIC_USE_ROWS_COLS が有効で ROWS が正の場合、指定された行数に
    # 合わせて行の高さを計算し、すべての画像を行方向に順次詰め込む。
    # そうでない場合は従来の二分探索により行の高さを決定します。
    # ROWS に正の値が設定されている場合や MOSAIC_USE_ROWS_COLS が有効な場合、
    # 固定行数モードとみなす。ROWS は 0 または負の場合に無視されます。
    try:
        desired_rows = int(globals().get("ROWS", 0))
    except Exception:
        desired_rows = 0
    # 行数が指定されている場合は常に固定行数モードとして扱う。
    use_fixed = bool(desired_rows > 0 or globals().get("MOSAIC_USE_ROWS_COLS", False))
    rows = []
    total_h = 0
    if use_fixed and desired_rows > 0:
        # 行数（ROWS）を目安にしつつ、横幅(W)に収まる最大の行高さを探索します。
        # （UHでも UW と同じ選択肢で扱えるよう、割り当てポリシーを用意）
        raw_h = H - gutter * max(0, desired_rows - 1)
        base_h_max = max(1, raw_h // max(1, desired_rows))
        # JUSTIFY_MIN_ROW_H／JUSTIFY_MAX_ROW_H の範囲に制限
        try:
            min_h = max(1, int(globals().get('JUSTIFY_MIN_ROW_H', JUSTIFY_MIN_ROW_H)))
        except Exception:
            min_h = 1
        try:
            max_h = max(1, int(globals().get('JUSTIFY_MAX_ROW_H', JUSTIFY_MAX_ROW_H)))
        except Exception:
            max_h = base_h_max
        hi = max(min_h, min(max_h, base_h_max))
        lo_h = min_h

        # UH の割り当てポリシー（名前は UW と揃える）
        try:
            _uh_policy = str(globals().get('MOSAIC_UH_ASSIGN', MOSAIC_UH_ASSIGN)).strip().lower()
        except Exception:
            _uh_policy = 'packed'
        if not _uh_policy:
            _uh_policy = 'packed'

        def _pack_fixed(h: int):
            # desired_rows 個の“行”へ割り当てて、各行の幅が W を超えないかをチェック。
            rows_bins: List[List[Tuple[Path, float, int]]] = [[] for _ in range(desired_rows)]
            widths = [0] * desired_rows
            cur_row = 0

            def _preferred_row(k: int) -> int:
                if desired_rows <= 1:
                    return 0
                if _uh_policy == 'roundrobin':
                    return k % desired_rows
                if _uh_policy == 'snake':
                    period = 2 * desired_rows - 2
                    m = k % period
                    return m if m < desired_rows else (period - m)
                return 0

            for k, (p_i, ar_i) in enumerate(ars):
                wj = max(1, int(round(ar_i * h)))

                if _uh_policy in ('packed', 'sequential', 'pack'):
                    # 既存挙動（左→右に詰め、収まらなければ次の行）
                    while True:
                        if cur_row >= desired_rows:
                            return None
                        add = wj if not rows_bins[cur_row] else (wj + gutter)
                        if rows_bins[cur_row] and widths[cur_row] + add > W:
                            cur_row += 1
                            continue
                        if widths[cur_row] + add > W:
                            # 空行なのに入らない（画像が極端に横長など）
                            return None
                        rows_bins[cur_row].append((p_i, ar_i, wj))
                        widths[cur_row] += add
                        break
                    continue

                # それ以外: 行候補を決めて、入る行に割り当てる
                if _uh_policy == 'minheight':
                    # UH では minwidth 相当（現在もっとも短い行へ）
                    candidates = sorted(range(desired_rows), key=lambda j: widths[j])
                else:
                    pref = _preferred_row(k)
                    candidates = list(range(pref, desired_rows)) + list(range(0, pref))

                placed = False
                for j in candidates:
                    add = wj if not rows_bins[j] else (wj + gutter)
                    if widths[j] + add <= W:
                        rows_bins[j].append((p_i, ar_i, wj))
                        widths[j] += add
                        placed = True
                        break
                if not placed:
                    return None

            rows_local = [(row_list, h) for row_list in rows_bins if row_list]
            total_height = len(rows_local) * h + gutter * max(0, len(rows_local) - 1)
            return rows_local, total_height

        # 最大の h を二分探索（割り当てが成立する範囲で最大化）
        best_rows = None
        best_total_h = 0
        best_h = 1
        lo2, hi2 = lo_h, hi
        while lo2 <= hi2:
            mid = (lo2 + hi2) // 2
            res = _pack_fixed(mid)
            if res is not None:
                best_rows, best_total_h = res
                best_h = mid
                lo2 = mid + 1
            else:
                hi2 = mid - 1

        if best_rows is None:
            # 最終手段（1pxで試す）。それでも無理なら空。
            res = _pack_fixed(1)
            if res is None:
                rows = []
                total_h = 0
                best_h = 1
            else:
                rows, total_h = res
                best_h = 1
        else:
            rows = best_rows
            total_h = best_total_h

        # gapless 拡張や後続処理で参照される row_h (lo) を設定
        lo = best_h
    else:
        # --- 従来のパック関数と二分探索 ---
        def pack(h: int):
            rows_local = []
            row_local = []
            cur = 0
            for p, ar in ars:
                wj = max(1, int(round(ar * h)))
                add = wj if not row_local else (gutter + wj)
                if row_local and cur + add > W:
                    rows_local.append((row_local, h))
                    row_local = []
                    cur = 0
                    add = wj
                row_local.append((p, ar, wj))
                cur += add
            if row_local:
                rows_local.append((row_local, h))
            total_height = len(rows_local) * h + gutter * max(0, len(rows_local) - 1)
            return rows_local, total_height
        lo = max(1, JUSTIFY_MIN_ROW_H)
        hi = max(1, min(JUSTIFY_MAX_ROW_H, H))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            _, tot = pack(mid)
            if tot <= H:
                lo = mid
            else:
                hi = mid - 1
        rows, total_h = pack(lo)

    layout_info = {}

    # フルシャッフルが無効な場合のみ、行バランス調整や近傍色最適化を適用します。
    if (not _preserve) and (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and (not bool(globals().get("MOSAIC_ENHANCE_ENABLE", False))):
        if MOSAIC_BALANCE_ENABLE:
            rows = optimize_rows_hillclimb(rows, W, gutter, iters=OPT_ITERS, show_progress=False)
            layout_info["balance"]="rows"

        if MOSAIC_NEIGHBOR_OBJECTIVE in ("min","max"):
            rows, color_sum = optimize_rows_color_neighbors(
                rows,
                objective=MOSAIC_NEIGHBOR_OBJECTIVE,
                iters_per_line=MOSAIC_NEIGHBOR_ITERS_PER_LINE,
                seed=OPT_SEED
            )
            layout_info.update({"rows_color":color_sum})

    # 行順の整列（UH）。フルシャッフルが有効な場合はスキップ。
    try:
        if not bool(globals().get('ARRANGE_FULL_SHUFFLE', False)):
            uh_order = globals().get('MOSAIC_UH_ROW_ORDER', None)
            if not uh_order:
                uh_order = 'avgLAB' if bool(globals().get('MOSAIC_UH_ORDER_ROWS', False)) else 'none'
            uh_order_l = str(uh_order).strip().lower()
            if (not _preserve) and uh_order_l and uh_order_l not in ('none', 'off', 'false', '0'):
                _rank = {p: i for i, p in enumerate(paths)}

                def _row_key_first(row_list):
                    return min((_rank.get(p, 10**9) for (p, _ar, _wj) in row_list), default=10**9)

                def _row_key_avg(row_list):
                    rs = [_rank.get(p, 10**9) for (p, _ar, _wj) in row_list]
                    return sum(rs) / max(1, len(rs))

                def _row_key_avg_lab(row_list):
                    n = max(1, len(row_list))
                    sL = sa = sb = 0.0
                    for (p, _ar, _wj) in row_list:
                        L, a, b = _avg_lab_vector(p)
                        sL += L; sa += a; sb += b
                    return (sa / n, sb / n, sL / n)

                if uh_order_l in ('first-rank', 'first_rank', 'firstrank'):
                    rows.sort(key=lambda t: _row_key_first(t[0]))
                    layout_info['rows_order'] = 'first-rank'
                elif uh_order_l in ('avg-rank', 'avg_rank', 'avgrank'):
                    rows.sort(key=lambda t: _row_key_avg(t[0]))
                    layout_info['rows_order'] = 'avg-rank'
                else:
                    # 'avgLAB'（デフォルト）
                    rows.sort(key=lambda t: _row_key_avg_lab(t[0]))
                    layout_info['rows_order'] = 'avgLAB'
    except Exception as e:
        _warn_exc_once(e)
        pass
    # -----------------------------------------------------------------------------
    # テンポ並べ替え（post／blend）
    # 事前（pre）でテンポを適用していても、その後の近傍最適化や平均LABソートで
    # 「速い/遅いの交互」が崩れることがあります。
    # stage が post／blend のときは、現在の行順を一度フラット化してテンポ並べ替えを再適用し、
    # その順番で行を再構築して最終結果のテンポ感を守ります。
    try:
        if ((not _preserve) and (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False)) and
            globals().get("ARRANGE_TEMPO_ENABLE", False))):
            stage = str(globals().get("ARRANGE_TEMPO_STAGE", "pre")).lower()
            if stage in ("post", "blend"):
                # 参照用にアスペクト比を辞書化
                ars_dict = {p: ar for (p, ar) in ars}
                # 現在の行順のままパスをフラット化
                flat_paths: List[Path] = []
                for (row_list, _h) in rows:
                    for (p_i, _ar_i, _wj_i) in row_list:
                        flat_paths.append(p_i)
                # フラット列にテンポ並べ替えを適用
                try:
                    reordered = _tempo_apply(list(flat_paths))
                except Exception:
                    reordered = flat_paths
                # 並べ替え後の順番で行を再構築（タイル幅も再計算）
                new_rows = []
                idx_fp = 0
                for (row_list, rh) in rows:
                    new_row: List[Tuple[Path, float, int]] = []
                    for _ in row_list:
                        p_new = reordered[idx_fp]; idx_fp += 1
                        ar_new = ars_dict.get(p_new, 1.0)
                        # 行高さとアスペクト比からタイル幅を再計算
                        wj_new = max(1, int(round(ar_new * rh)))
                        new_row.append((p_new, ar_new, wj_new))
                    new_rows.append((new_row, rh))
                rows = new_rows
                # 以後の参照の整合性のため ars も順序を合わせる（必須ではない）
                try:
                    ars = [(p, ars_dict.get(p, 1.0)) for p in reordered]
                except Exception as e:
                    _warn_exc_once(e)
                    pass
    except Exception:
        # post テンポ適用に失敗しても致命にしない（現状の rows をそのまま使う）
        pass

    # ギャップレスモザイク拡張: 縦方向のスペースを埋めるため行の高さを調整し、横方向に拡張します
    if (not _preserve) and globals().get("MOSAIC_GAPLESS_EXTEND", False):
        try:
            # justify 検索で基本の行高さを決定します
            row_h = lo
            num_rows = len(rows)
            # 使用済み画像を除外したグローバルスキャンプールから供給リストを作成します。
            # ギャップレスモードでは各行を延長するために追加のタイルが必要となる場合があります。
            # KANA_SCAN_ALL から供給リストを作成し、現在の SELECT_MODE に従ってソートし、
            # ランダムモードの場合はオプションでシャッフルします。
            all_images = (globals().get("KANA_SCAN_ALL", []) or [])
            used_images_set: set = set()
            for (_r, _hh) in rows:
                for (p_used, _, _) in _r:
                    used_images_set.add(p_used)
            # すでに初期行で使われた画像は供給リストから除外します
            supply_paths = [p_sp for p_sp in all_images if p_sp not in used_images_set]
            # 未使用が無い場合は元の候補（ars）にフォールバックします
            if not supply_paths:
                supply_paths = [p0 for (p0, _) in ars] if ars else []
            # パス表記ゆれを正規化して重複（同一ファイル）を除去します
            try:
                import os as _supply_os
                def _supply_norm(p):
                    try:
                        return _supply_os.path.normcase(_supply_os.path.normpath(p))
                    except Exception:
                        return p
                _seen_norm: set = set()
                _uniq_supply: list = []
                for _p in supply_paths:
                    _k = _supply_norm(_p)
                    if _k not in _seen_norm:
                        _uniq_supply.append(_p); _seen_norm.add(_k)
                supply_paths = _uniq_supply
            except Exception as e:
                _warn_exc_once(e)
                pass
            # 供給リストを SELECT_MODE に従って並べ替えます。
            # モードが 'recent' や 'oldest' の場合、追加画像も初期選択と同じ順序に沿います。
            # SELECT_MODE が 'random' の場合は、バイアスを避けるために供給リストをシャッフルします。
            try:
                supply_paths = sort_by_select_mode(list(supply_paths))
            except Exception as e:
                _warn_exc_once(e)
                pass
            try:
                _mode_now = str(globals().get("SELECT_MODE", "")).lower()
            except Exception:
                _mode_now = ""
            # ランダム選択が有効な場合は、供給リストをシャッフルします。
            # SHUFFLE_SEED に数値を指定すると決定的な乱数種が用いられ、
            # 指定が無い場合は Python のデフォルトの RNG が使用されます。
            if _mode_now == "random":
                try:
                    _seed = globals().get("SHUFFLE_SEED", None)
                    supply_paths = hash_shuffle(list(supply_paths), _seed, salt="gapless_supply_uh")
                except Exception as e:
                    _warn_exc_once(e)
                    pass
            # 供給リストを順番に取り出すイテレータです。末尾に到達したら停止します（繰り返し許可なら先頭に戻ります）。
            supply_idx = 0
            supply_ar_cache: dict = {}
            def _next_supply():
                nonlocal supply_idx
                if not supply_paths:
                    return None
                # 初回だけ初期化（closure の状態を保持）
                if not hasattr(_next_supply, "_inited"):
                    _next_supply._inited = True  # type: ignore
                    try:
                        thr = globals().get("MOSAIC_GAPLESS_SUPPLY_DEDUPE_HAMMING", None)
                        thr = int(thr) if thr is not None else 0
                    except Exception:
                        thr = 8
                    _next_supply._thr = int(thr)  # type: ignore
                    _next_supply._used = []  # type: ignore
                    _next_supply._supplied = []  # type: ignore
                    if bool(globals().get("MOSAIC_GAPLESS_SUPPLY_DEDUPE", True)):
                        try:
                            for _p in list(used_images_set):
                                _h = dhash64_for_path_cached(_p)
                                if _h is not None:
                                    _next_supply._used.append(int(_h))  # type: ignore
                        except Exception as e:
                            _kana_silent_exc('core:L10479', e)
                            pass
                max_tries = max(1, int(len(supply_paths)))
                tries = 0
                while tries < max_tries:
                    tries += 1
                    if supply_idx >= len(supply_paths):
                        if bool(globals().get('MOSAIC_GAPLESS_ALLOW_REPEAT', False)):
                            supply_idx = 0
                        else:
                            return None
                    psp = supply_paths[supply_idx]
                    supply_idx += 1
                    # 近似重複スキップ（同じ絵/ほぼ同じ絵を避ける）
                    if bool(globals().get("MOSAIC_GAPLESS_SUPPLY_DEDUPE", True)):
                        try:
                            _h = dhash64_for_path_cached(psp)
                            if _h is not None:
                                _h = int(_h)
                                _thr = int(getattr(_next_supply, "_thr", 8))  # type: ignore
                                _skip = False
                                for _uh in getattr(_next_supply, "_used", []):  # type: ignore
                                    if hamming(_h, int(_uh)) <= _thr:
                                        _skip = True; break
                                if not _skip:
                                    for _sh in getattr(_next_supply, "_supplied", []):  # type: ignore
                                        if hamming(_h, int(_sh)) <= _thr:
                                            _skip = True; break
                                if _skip:
                                    continue
                                getattr(_next_supply, "_supplied", []).append(_h)  # type: ignore
                        except Exception as e:
                            _kana_silent_exc('core:L10511', e)
                            pass
                    ar_sp = supply_ar_cache.get(psp)
                    if ar_sp is None:
                        try:
                            with open_image_safe(psp) as im_sp:
                                iw_sp, ih_sp = im_sp.size
                                ar_sp = iw_sp / float(ih_sp) if ih_sp > 0 else 1.0
                        except Exception:
                            ar_sp = 1.0
                        supply_ar_cache[psp] = ar_sp
                    return (psp, ar_sp)
                return None
            # 行の総高さと余剰スペースを計算する
            total_h = num_rows * row_h + gutter * max(0, num_rows - 1)
            extra_space = H - total_h
            # 縦方向スペースをちょうど埋めるよう各行の高さを決定する
            row_heights = []
            if extra_space > 0 and num_rows > 0:
                base_add = extra_space // num_rows
                extra_rem = extra_space % num_rows
                for i in range(num_rows):
                    rh = row_h + base_add + (1 if i < extra_rem else 0)
                    row_heights.append(rh)
            else:
                row_heights = [row_h] * num_rows
            # 調整後の各行でタイルの最大幅を計算します。
            # この値を使って保守的な横方向のオーバーシュートを決定します。
            # オーバーシュートが小さすぎると、切り抜き時の丸め誤差で端に細い隙間が残る可能性があるため、
            # 最も幅の広いタイルに対して余裕を持たせます。
            max_orig_w_global = 0
            for idx_r, (orig_row, _orig_h) in enumerate(rows):
                rhh = row_heights[idx_r]
                for (_, ar, _wj0) in orig_row:
                    new_wj = max(1, int(round(ar * rhh)))
                    if new_wj > max_orig_w_global:
                        max_orig_w_global = new_wj
            # ギャップレス用に行を「横方向に余分に描いてから中央で切り抜く」ための準備を行います。
            # まず各行を新しい行高で組み直し、その後「全行で共通の target_w」を決めます。
            # 各行の幅が target_w に達するまで供給画像を追加し、行ごとの差で縦の継ぎ目が出ないようにします。
            ext_rows = []
            initial_row_widths = []
            # 各行を構築し、初期の幅を記録します
            for idx_r, (orig_row, _orig_h) in enumerate(rows):
                rhh = row_heights[idx_r]
                row_tiles: List[Tuple[Path, float, int]] = []
                cur_w = 0
                for (p, ar, _wj) in orig_row:
                    new_wj = max(1, int(round(ar * rhh)))
                    row_tiles.append((p, ar, new_wj))
                    cur_w += new_wj if not row_tiles[:-1] else (new_wj + gutter)
                initial_row_widths.append(cur_w)
                ext_rows.append((row_tiles, rhh))
            # 全行共通の目標幅 target_w を決めます（後で中央クロップするためのオーバーシュート）。
            # 切り抜き時の丸め誤差で隙間が出にくいよう、W + 3*最大タイル幅 以上を確保します。
            max_row_width = max(initial_row_widths) if initial_row_widths else 0
            target_w = max(max_row_width, W + 3 * max_orig_w_global)
            # 各行の幅が target_w に達するまで横方向にタイルを追加します
            for idx_r, (row_tiles, rhh) in enumerate(ext_rows):
                cur_w = initial_row_widths[idx_r]
                # 初期幅に誤ったガター計算が含まれている場合は、正しく再計算します
                # row_tiles と gutter を使って現在の幅を再計算し、数え間違いを防ぎます
                cur_w = sum(w_j for (_, _, w_j) in row_tiles) + gutter * max(0, len(row_tiles) - 1)
                while cur_w < target_w:
                    nxt = _next_supply()
                    if nxt is None:
                        break
                    p_sup, ar_sup = nxt
                    wj_sup = max(1, int(round(ar_sup * rhh)))
                    row_tiles.append((p_sup, ar_sup, wj_sup))
                    cur_w += (wj_sup + gutter)
                ext_rows[idx_r] = (row_tiles, rhh)
            # 各行の描画Y座標を決定します（縦方向は余りピクセル分を配分済み）
            y_positions = []
            y_cur = margin
            for ridx, (_, rhh) in enumerate(ext_rows):
                y_positions.append(y_cur)
                if ridx < len(ext_rows) - 1:
                    y_cur += rhh + gutter
            # 拡張キャンバスとマスクを用意します
            canvas_ext = Image.new("RGB", (width, height), bg_rgb)
            mask_ext = Image.new("L", (width, height), 0)
            total_draw = sum(len(rr) for (rr, _) in ext_rows)
            done_cnt = 0
            # 全行共通の横オフセットで描画します（行ごとにズレると縦の継ぎ目が出るため）。
            # extra_w = target_w - W を左右に均等配分するイメージで中央に寄せます。
            global_extra_w = max(target_w - W, 0)
            x_off_global = margin - (global_extra_w // 2)
            # --- gaplessモード用 モザイク拡張割り当て（post-pack） ---
            # 注：gaplessモードは以前、通常のpost-packブロックより前でreturnしてしまい、
            #     MOSAIC_ENHANCE_PROFILE／ローカル最適化パラメータが効いていないように見えていた。
            #     ここでもpost-pack割り当てを適用して効果が見えるようにする。
            try:
                if _mosaic_enhance_active():
                    visible_tiles = []
                    visible_paths = []
                    visible_centers = []
                    for _ridx, (_rrow, _rhh) in enumerate(ext_rows):
                        _y_cur = y_positions[_ridx]
                        _x_cur = x_off_global
                        for _cidx, (_p_t, _ar_t, _wj_t) in enumerate(_rrow):
                            # 可視領域（viewport）に交差するタイルだけ対象にする。
                            if not (_x_cur + _wj_t <= margin or _x_cur >= margin + W or _y_cur + _rhh <= margin or _y_cur >= margin + H):
                                visible_tiles.append((_ridx, _cidx))
                                visible_paths.append(_p_t)
                                visible_centers.append((_x_cur + _wj_t / 2.0, _y_cur + _rhh / 2.0))
                            _x_cur += _wj_t + gutter

                    if len(visible_paths) > 1 and len(visible_centers) == len(visible_paths):
                        try:
                            banner(_lang("前処理: Mosaic post-pack割当", "Preprocess: Mosaic post-pack assignment"))
                        except Exception as e:
                            _warn_exc_once(e)
                            pass
                        _assigned = _mosaic_post_assign_paths(visible_paths, visible_centers)
                        # 書き戻し（アス比/幅を再計算して各タイルのアス比を維持。クロップなし・黒帯なし）
                        _ar_local_cache = {}
                        for (_ridx, _cidx), _new_p in zip(visible_tiles, _assigned):
                            try:
                                _row_ref, _rhh_ref = ext_rows[_ridx]
                                ar_new = None
                                try:
                                    ar_new = supply_ar_cache.get(_new_p)
                                except Exception:
                                    ar_new = None
                                if ar_new is None:
                                    ar_new = _ar_local_cache.get(_new_p)
                                if ar_new is None:
                                    try:
                                        with open_image_safe(_new_p) as _im_tmp:
                                            _iw_tmp, _ih_tmp = _im_tmp.size
                                        ar_new = _iw_tmp / float(_ih_tmp) if _ih_tmp > 0 else 1.0
                                    except Exception:
                                        ar_new = 1.0
                                    _ar_local_cache[_new_p] = ar_new
                                new_wj = max(1, int(round(ar_new * _rhh_ref)))
                                _row_ref[_cidx] = (_new_p, ar_new, new_wj)
                            except Exception as e:
                                _warn_exc_once(e)
                                pass
                        # 再割り当てで幅が変わった場合、各行がtarget_wに届くようtopping upで補充（gapless）
                        try:
                            if '_next_supply' in locals():
                                for _rr_idx, (_rr_row, _rr_hh) in enumerate(ext_rows):
                                    _cur_w2 = 0
                                    for _jj, (_pp2, _aa2, _ww2) in enumerate(_rr_row):
                                        _cur_w2 += _ww2 if _jj == 0 else (_ww2 + gutter)
                                    _cap = int(target_w / max(8, int(round(_rr_hh * 0.25)))) + 32
                                    while _cur_w2 < target_w and len(_rr_row) < _cap:
                                        _nxt = _next_supply()
                                        if not _nxt:
                                            break
                                        _psp, _ar_sp = _nxt
                                        _w_sp = max(1, int(round(_ar_sp * _rr_hh)))
                                        _rr_row.append((_psp, _ar_sp, _w_sp))
                                        _cur_w2 += _w_sp + gutter
                        except Exception as e:
                            _warn_exc_once(e)
                            pass
                    else:
                        if bool(globals().get("MOSAIC_POST_DEBUG", False)):
                            print(f"[MOSAIC_POST_DEBUG] UH gapless: skip post-pack (tiles={len(visible_paths)} centers={len(visible_centers)})")
            except Exception as _pp_ex:
                if bool(globals().get("MOSAIC_POST_DEBUG", False)):
                    print(f"[MOSAIC_POST_DEBUG] UH gapless: post-pack error: {_pp_ex}")
            # --- gapless用post-packここまで ---
            # --- draw prefetch（CPU）（mosaic-uniform-height） ---
            _pf_ahead = int(max(0, int(globals().get('DRAW_PREFETCH_AHEAD', 16))))
            _pf_ahead = _effective_draw_prefetch_ahead(width, height, _pf_ahead)
            _pf_workers = int(max(1, int(globals().get('DRAW_PREFETCH_WORKERS', 0) or (os.cpu_count() or 4))))
            _pf_on = bool(globals().get('DRAW_PREFETCH_ENABLE', True)) and (_pf_ahead > 0)

            steps = []      # ('skip', None) or ('draw', meta)
            draw_items = [] # (path, w, h)

            for ridx, (rrow, rhh) in enumerate(ext_rows):
                y_cur = y_positions[ridx]
                # 画面外（上）ならスキップ
                if y_cur + rhh <= margin:
                    for _ in rrow:
                        steps.append(('skip', None))
                    continue
                # 画面外（下）に到達したら終了
                if y_cur >= margin + H:
                    break
                x_cur = x_off_global
                for (p_t, ar_t, wj_t) in rrow:
                    # 画面外（左）ならスキップ
                    if x_cur + wj_t <= margin:
                        x_cur += wj_t + gutter
                        steps.append(('skip', None))
                        continue
                    # 画面外（右）ならスキップ（以降も右端外のため、x_cur は進めない）
                    if x_cur >= margin + W:
                        steps.append(('skip', None))
                        continue
                    # クリッピング量
                    l_clip = max(margin - x_cur, 0)
                    r_clip = max((x_cur + wj_t) - (margin + W), 0)
                    v_w = int(wj_t - l_clip - r_clip)
                    t_clip = max(margin - y_cur, 0)
                    b_clip = max((y_cur + rhh) - (margin + H), 0)
                    v_h = int(rhh - t_clip - b_clip)

                    if v_w > 0 and v_h > 0:
                        nx = int(x_cur + l_clip)
                        ny = int(y_cur + t_clip)
                        meta = (p_t, int(wj_t), int(rhh), int(l_clip), int(t_clip), int(v_w), int(v_h), nx, ny)
                        steps.append(('draw', meta))
                        draw_items.append((p_t, int(wj_t), int(rhh)))
                    else:
                        steps.append(('skip', None))

                    x_cur += wj_t + gutter

            def _mosaic_uh_render(item):
                p_t, wj_t, rhh = item
                with open_image_safe(p_t, draft_to=(max(1, int(wj_t)), max(1, int(rhh))), force_mode='RGB') as im_tt:
                    return hq_resize(im_tt, (max(1, int(wj_t)), max(1, int(rhh))))

            _pf_backend = str(globals().get('DRAW_PREFETCH_BACKEND', ('process' if os.name == 'nt' else 'thread'))).lower()
            _pf_use_mp = _pf_backend in ('process', 'mp', 'multiprocess', 'proc', 'processpool', 'process_pool')

            if _pf_on and draw_items:
                try:
                    if _pf_use_mp:
                        _draw_it = iter(prefetch_ordered_mp_safe(draw_items, _pf_worker_mosaic_uh_render, ahead=_pf_ahead, max_workers=_pf_workers))
                    else:
                        _draw_it = iter(prefetch_ordered_safe(draw_items, _mosaic_uh_render, ahead=_pf_ahead, max_workers=_pf_workers))
                except Exception as _e_pf:
                    print(f"[WARN] mosaic-UH process prefetch unavailable; fallback to thread. reason={_e_pf}")
                    _draw_it = iter(prefetch_ordered_safe(draw_items, _mosaic_uh_render, ahead=_pf_ahead, max_workers=_pf_workers))
            else:
                _pf_on = False
                _draw_it = None


            for kind, meta in steps:
                if kind != 'draw':
                    done_cnt += 1
                    if VERBOSE:
                        bar(done_cnt, max(1, total_draw), prefix='draw   ', final=False)
                    continue

                p_t, wj_t, rhh, l_clip, t_clip, v_w, v_h, nx, ny = meta
                try:
                    if _pf_on:
                        _item, rez, exc = next(_draw_it)
                        if exc is not None:
                            raise exc
                    else:
                        rez = _mosaic_uh_render((p_t, wj_t, rhh))

                    if l_clip != 0 or t_clip != 0 or v_w != wj_t or v_h != rhh:
                        rez = rez.crop((int(l_clip), int(t_clip), int(l_clip + v_w), int(t_clip + v_h)))
                    canvas_ext.paste(rez, (int(nx), int(ny)))
                    mask_ext.paste(255, (int(nx), int(ny), int(nx + v_w), int(ny + v_h)))
                except Exception as ex_draw:
                    print(f"[WARN] {p_t}: {ex_draw}")

                done_cnt += 1
                if VERBOSE:
                    bar(done_cnt, max(1, total_draw), prefix='draw   ', final=False)

            # --- /draw prefetch（CPU）（mosaic-uniform-height） ---
            if total_draw == 0:
                # 1枚も描けなかった場合でも final=True で1回だけ表示
                bar(done_cnt, 1, prefix="draw   ", final=True)
            else:
                # 100% 表示で完了させる
                bar(max(done_cnt, total_draw), max(1, total_draw), prefix="draw   ", final=True)
            # 自動補正が有効な場合：マスクから残った隙間（1px相当のズレ）を検出し、
            # 必要ならクロップ位置を微調整して埋めます（UH は縦方向のズレ検出を主に扱います）。
            if globals().get("MOSAIC_AUTO_INTERPOLATE", False):
                try:
                    # 縦方向の隙間検出（axis="vertical"）
                    shift_x, _shift_y = _detect_gap_shift(mask_ext, margin, W, H, axis="vertical")
                    if shift_x != 0:
                        # 拡張キャンバス上でのクロップ座標を計算
                        sx = margin + shift_x
                        sy = margin
                        ex = sx + W
                        ey = sy + H
                        # 拡張キャンバスの範囲内にクランプ
                        sx_clamped = max(0, min(canvas_ext.width, sx))
                        sy_clamped = max(0, min(canvas_ext.height, sy))
                        ex_clamped = max(0, min(canvas_ext.width, ex))
                        ey_clamped = max(0, min(canvas_ext.height, ey))
                        sub_canvas = canvas_ext.crop((int(sx_clamped), int(sy_clamped), int(ex_clamped), int(ey_clamped)))
                        sub_mask = mask_ext.crop((int(sx_clamped), int(sy_clamped), int(ex_clamped), int(ey_clamped)))
                        # 最終キャンバスへの貼り付け位置（負の shift は右、正の shift は左に寄せる）
                        paste_x = margin + int(sx_clamped - sx)
                        paste_y = margin
                        # 新しいキャンバス/マスクに貼り直す
                        new_canvas = Image.new("RGB", (width, height), bg_rgb)
                        new_mask = Image.new("L", (width, height), 0)
                        new_canvas.paste(sub_canvas, (int(paste_x), int(paste_y)))
                        new_mask.paste(sub_mask, (int(paste_x), int(paste_y)))
                        canvas_ext = new_canvas
                        mask_ext = new_mask
                        layout_info["interpolate_shift_x"] = shift_x
                except Exception as e:
                    _warn_exc_once(e)
                    pass
            # gapless をマークして返します
            layout_info["gapless"] = True
            return canvas_ext, mask_ext, layout_info
        except Exception as gap_err:
            # ギャップレス拡張に失敗したら警告を出して、通常描画にフォールバックします
            print(f"[WARN] gapless mosaic extension failed: {gap_err}")

    # --- モザイク拡張割り当て（post-pack） ---
    try:
        # 描画順（貼り付け順）でのタイル中心座標
        _m_centers = []
        if rows:
            num_rows = len(rows)
            extra_space = max(0, H - total_h)
            base_gutter = gutter
            extra_rem = 0
            if extra_space > 0 and num_rows > 1:
                base_gutter = gutter + (extra_space // (num_rows - 1))
                extra_rem = extra_space % (num_rows - 1)

            y0 = margin
            if extra_space <= 0:
                y0 = margin + max(0, (H - total_h) // 2)

            y = y0
            for row_i, (row, h) in enumerate(rows):
                row_w = sum(w for (_p, _ar, w) in row) + gutter * (len(row) - 1)
                # 行を拡張（drawと同じ）
                if bool(globals().get("MOSAIC_UH_EXPAND_ROW", False)) and row_w < W and len(row) > 1:
                    ws = [w for (_p, _ar, w) in row]
                    sum_ws = sum(ws)
                    if sum_ws > 0:
                        scale = (W - gutter * (len(row) - 1)) / float(sum_ws)
                        new_ws = [max(1, int(round(w * scale))) for w in ws]
                        diff = (W - gutter * (len(row) - 1)) - sum(new_ws)
                        if diff != 0 and len(new_ws) > 0:
                            new_ws[-1] = max(1, new_ws[-1] + diff)
                    else:
                        new_ws = [w for (_p, _ar, w) in row]

                    x = margin
                    for idx, (_p, _ar, _wj) in enumerate(row):
                        wj = int(new_ws[idx])
                        _m_centers.append((x + wj / 2.0, y + h / 2.0))
                        x += wj + gutter
                else:
                    x = margin + max(0, (W - row_w) // 2)
                    for _p, _ar, wj in row:
                        _m_centers.append((x + wj / 2.0, y + h / 2.0))
                        x += wj + gutter

                if extra_space > 0 and num_rows > 1:
                    y += h + base_gutter
                    if row_i < extra_rem:
                        y += 1
                else:
                    y += h + gutter

        tile_paths_draw = [p for (row, _h) in rows for (p, _ar, _w) in row]
        if _mosaic_enhance_active() and len(tile_paths_draw) > 1 and len(_m_centers) == len(tile_paths_draw):
            try:
                banner(_lang("前処理: Mosaic post-pack割当", "Preprocess: Mosaic post-pack assignment"))
            except Exception as e:
                _warn_exc_once(e)
                pass
            paths_for_draw = _mosaic_post_assign_paths(tile_paths_draw, _m_centers)
        else:
            # 元の描画順
            paths_for_draw = tile_paths_draw
    except Exception:
        paths_for_draw = [p for (row, _h) in rows for (p, _ar, _w) in row]

    draw_idx = 0
    canvas = Image.new("RGB", (width, height), bg_rgb)
    mask = Image.new("L", (width, height), 0)
    banner(_lang("処理中: Mosaic / Uniform Height","Rendering: Mosaic / Uniform Height"))
    total = len(paths_for_draw); done = 0
    # 余った縦スペースを計算し、行間へ配分する準備をします
    extra_space = H - total_h
    num_rows = len(rows)
    if extra_space > 0 and num_rows > 1:
        base_gutter = gutter + extra_space // (num_rows - 1)
        extra_rem = extra_space % (num_rows - 1)
        y = margin
    else:
        y = margin + max(0, (H - total_h) // 2)
    for row_index, (row, h) in enumerate(rows):
        # 現在の行の総幅を計算し、余りスペースを求めます。
        row_w = sum(w for _, _, w in row) + gutter * max(0, len(row) - 1)
        leftover_w = W - row_w
        # 横方向に余剰幅がある場合、各タイルの幅を少しずつ拡大して行幅をぴったり埋めます。
        # 余り幅はタイル間で均等に分配します。
        # このスケーリングは gapless モードが無効な場合（MOSAIC_GAPLESS_EXTEND が False）のみに適用されます。
        if leftover_w > 0 and len(row) > 0 and not globals().get("MOSAIC_GAPLESS_EXTEND", False):
            base_extra = leftover_w // len(row)
            extra_rem = leftover_w % len(row)
            # 各タイルの新しい幅をあらかじめ計算します
            new_ws: List[int] = []
            for idx, (_, _ar, wj) in enumerate(row):
                extra = base_extra + (1 if idx < extra_rem else 0)
                new_ws.append(wj + extra)
            # 幅をぴったり埋めるので中央寄せはせず、左マージンから開始します
            x = margin
            # 拡大後の幅でタイルを描画します
            for idx, (p, ar, wj) in enumerate(row):
                p = paths_for_draw[draw_idx]
                draw_idx += 1
                nw = new_ws[idx]
                try:
                    with open_image_safe(p) as im:
                        rez = _fit_rect_no_crop_no_upscale(im, max(1, nw), max(1, h))
                        canvas.paste(rez, (x, y))
                        mask.paste(255, (x, y, x + nw, y + h))
                except Exception as e:
                    print(f"[WARN] {p}: {e}")
                x += nw + gutter
                done += 1
                if VERBOSE:
                    bar(done, max(1, total), prefix="draw   ", final=(done == total))
        else:
            # 拡大不要な場合は従来通り中央寄せします
            x = margin + max(0, (W - row_w) // 2)
            for p, ar, wj in row:
                p = paths_for_draw[draw_idx]
                draw_idx += 1
                try:
                    with open_image_safe(p) as im:
                        # モザイクはクロップ/ズームしない（タイルに収めて貼る）
                        rez = _fit_rect_no_crop_no_upscale(im, max(1, wj), max(1, h))
                        canvas.paste(rez, (x, y))
                        mask.paste(255, (x, y, x + wj, y + h))
                except Exception as e:
                    print(f"[WARN] {p}: {e}")
                x += wj + gutter
                done += 1
                if VERBOSE:
                    bar(done, max(1, total), prefix="draw   ", final=(done == total))
        # 次の行の描画Y座標へ進めます
        if extra_space > 0 and num_rows > 1:
            if row_index < num_rows - 1:
                y += h + base_gutter
                if row_index < extra_rem:
                    y += 1
        else:
            y += h + gutter
    # この関数では hex 描画用の min_r/max_r/min_c/max_c は定義されません。
    # 画像が 0 枚のときはそれらを参照せず、進捗バーだけ最終表示します。
    if total == 0:
        bar(done, 1, prefix="draw   ", final=True)
    return canvas, mask, layout_info

def layout_mosaic_uniform_width(paths: List[Path], width: int, height: int, margin: int, gutter: int,
                                bg_rgb: Tuple[int, int, int]):
    """列の幅を一定にし、縦方向へ積むモード（Masonry 風）"""
    # テンポ並べ替えはこのあと paths に直接適用します。


    # レイアウト情報（1回だけ）を表示
    # UH/UW 共通フォーマット: "LAYOUT_STYLE: <style> | MOSAIC_UW_ASSIGN: <assign>"
    global _PRINTED_LAYOUT_ONCE
    if not _PRINTED_LAYOUT_ONCE:
        try:
            uw = globals().get('MOSAIC_UW_ASSIGN', '(n/a)')
            _mprof = str(globals().get("MOSAIC_ENHANCE_PROFILE", "off")).strip()
            _mpost = ""
            if _mosaic_enhance_active() and _mprof and _mprof.lower() not in ("off", "none", "random"):
                _mpost = f" | POST: {_mprof}"
            note(f"LAYOUT: mosaic-uniform-width | ASSIGN: {uw}{_mpost}")
        except Exception as e:
            _warn_exc_once(e)
            pass
        _PRINTED_LAYOUT_ONCE = True
    _preserve = bool(globals().get('PRESERVE_INPUT_ORDER', False))
    # フルシャッフルが有効な場合は、全体ソート等の前に paths を完全にシャッフルします。
    # OPT_SEED が指定されていて "random" でないときは決定的（再現性のある）順序になります。
    if bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
        try:
            _seed = globals().get("OPT_SEED", None)
            paths = hash_shuffle(list(paths), _seed, salt="mosaic_uw_fullshuffle")
        except Exception as e:
            _warn_exc_once(e)
            pass
        # Full shuffle の状態表示（重複しないよう英語で1回だけ）
        try:
            if bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
                if bool(globals().get("_MOSAIC_POST_DISABLED_BY_FULLSHUFFLE", False)):
                    note("Full shuffle: ON (Mosaic POST assignment disabled)")
                else:
                    note("Full shuffle: ON")
        except Exception as e:
            _warn_exc_once(e)
            pass
    # グローバル順序を先に適用（全体の色の流れを先に決める）
    if (not _preserve) and (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and (not bool(globals().get("MOSAIC_ENHANCE_ENABLE", False))):
        if MOSAIC_GLOBAL_ORDER == "spectral-hilbert":
            paths = reorder_global_spectral_hilbert(list(paths), objective=MOSAIC_GLOBAL_OBJECTIVE)
        elif MOSAIC_GLOBAL_ORDER == "anneal":
            paths = reorder_global_anneal(list(paths), objective=MOSAIC_GLOBAL_OBJECTIVE,
                                          iters=MOSAIC_GLOBAL_ITERS, seed=OPT_SEED)

    # テンポ並べ替え（有効時）: グローバル順序のあと、列詰め（pack）前に paths を入れ替えます。
    # フルシャッフル時はランダム性を優先するためスキップします。
    if (not _preserve) and (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and paths:
        try:
            paths = _tempo_apply(paths)
        except Exception as e:
            _warn_exc_once(e)
            pass
    W = width - margin * 2
    H = height - margin * 2
    ars = _load_ars(paths)

    # 列幅固定モード: MOSAIC_USE_ROWS_COLS が有効かつ COLS が正の場合、
    # 事前に計算した列幅で固定数の列を生成する。そうでない場合は従来の二分探索で列幅を決定する。
    # COLS に正の値が設定されている場合や MOSAIC_USE_ROWS_COLS が有効な場合は
    # 固定列数モードとして扱います。COLS が 0 以下の場合は無視されます。
    try:
        desired_cols = int(globals().get("COLS", 0))
    except Exception:
        desired_cols = 0
    use_fixed = bool(desired_cols > 0 or globals().get("MOSAIC_USE_ROWS_COLS", False))
    cols: List[list] = []
    w = 0
    C = 0
    if use_fixed and desired_cols > 0:
        # ガター分を除いた有効幅を計算し、列幅を求める
        raw_w = W - gutter * max(0, desired_cols - 1)
        base_w = max(1, raw_w // max(1, desired_cols))
        # JUSTIFY_MIN_COL_W と JUSTIFY_MAX_COL_W の範囲に制限
        try:
            min_w = max(1, min(W, globals().get('JUSTIFY_MIN_COL_W', JUSTIFY_MIN_ROW_H)))
        except Exception:
            min_w = 1
        try:
            max_w_val = max(min_w, min(W, globals().get('JUSTIFY_MAX_COL_W', JUSTIFY_MAX_ROW_H)))
        except Exception:
            max_w_val = base_w
        base_w = max(min_w, min(max_w_val, base_w))
        # 固定列数で画像を縦に積み上げるローカル関数
        def pack_cols_fixed(w_fix: int, C_fix: int) -> Tuple[List[list], int, int]:
            cols_fix: List[list] = [[] for _ in range(C_fix)]
            heights_fix: List[int] = [0] * C_fix
            policy_fix = globals().get('MOSAIC_UW_ASSIGN', 'minheight')
            for k, (p, ar) in enumerate(ars):
                # 縦サイズをアスペクト比から計算
                h_fix = max(1, int(round(w_fix / max(1e-6, ar))))
                # 割り当て列を決定
                if policy_fix == 'minheight':
                    j = min(range(C_fix), key=lambda i: heights_fix[i])
                elif policy_fix == 'roundrobin':
                    j = k % C_fix if C_fix > 0 else 0
                else:  # snake
                    m = k % (2 * C_fix if C_fix > 0 else 1)
                    j = m if m < C_fix else 2 * C_fix - 1 - m
                heights_fix[j] += h_fix if not cols_fix[j] else h_fix + gutter
                cols_fix[j].append((p, w_fix, h_fix))
            return cols_fix, max(heights_fix) if heights_fix else 0, C_fix
        # 固定幅で列を構成
        cols, mh, C = pack_cols_fixed(base_w, desired_cols)
        w = base_w
    else:
        # --- 従来の pack_cols 関数と二分探索 ---
        def pack_cols(w_val: int) -> Tuple[List[list], int, int]:
            """指定幅で画像を縦に積み上げ、列構成と高さ・列数を返す"""
            C_val = max(1, min(len(ars), (W + gutter) // (w_val + gutter)))
            cols_local: List[list] = [[] for _ in range(C_val)]
            heights = [0] * C_val
            policy = globals().get('MOSAIC_UW_ASSIGN', 'minheight')
            for k, (p, ar) in enumerate(ars):
                h_val = max(1, int(round(w_val / max(1e-6, ar))))
                # 割り当て列を決定
                if policy == "minheight":
                    j = min(range(C_val), key=lambda i: heights[i])
                elif policy == "roundrobin":
                    j = k % C_val if C_val > 0 else 0
                else:
                    m = k % (2 * C_val if C_val > 0 else 1)
                    j = m if m < C_val else 2 * C_val - 1 - m
                heights[j] += h_val if not cols_local[j] else h_val + gutter
                cols_local[j].append((p, w_val, h_val))
            return cols_local, max(heights) if heights else 0, C_val
        # 二分探索で列幅を決定: JUSTIFY_MIN_COL_W ～ JUSTIFY_MAX_COL_W の範囲
        min_w = max(1, min(W, globals().get('JUSTIFY_MIN_COL_W', JUSTIFY_MIN_ROW_H)))
        max_w_val = max(min_w, min(W, globals().get('JUSTIFY_MAX_COL_W', JUSTIFY_MAX_ROW_H)))
        lo = min_w
        hi = max_w_val
        best: Optional[Tuple[List[list], int, int, int]] = None
        while lo <= hi:
            mid = (lo + hi) // 2
            cols_mid, mh_mid, C_mid = pack_cols(mid)
            if mh_mid <= H:
                best = (cols_mid, mid, C_mid, mh_mid)
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            # 下限をさらに緩めて全画像が収まる幅を探す
            fallback_w = min_w
            cols_local, mh_val, C_val = pack_cols(fallback_w)
            # 高さが収まらない場合は幅を順次下げる
            if mh_val > H:
                for w_try in range(fallback_w - 1, 0, -1):
                    cols_try, mh_try, C_try = pack_cols(w_try)
                    if mh_try <= H:
                        cols_local, C_val, mh_val, fallback_w = cols_try, C_try, mh_try, w_try
                        break
            cols = cols_local
            w = fallback_w
            C = C_val
        else:
            cols, w, C, _ = best

    # メタ情報を初期化（以後は update を使う）
    layout_info = {}

    # フルシャッフルが無効な場合のみ、列バランス調整・近傍色最適化・最終列順の整列を行います。
    if (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and (not bool(globals().get("MOSAIC_ENHANCE_ENABLE", False))):
        if MOSAIC_BALANCE_ENABLE:
            cols = optimize_cols_hillclimb(cols, H, gutter, iters=OPT_ITERS, show_progress=False)
            layout_info["balance"] = "cols"

        if MOSAIC_NEIGHBOR_OBJECTIVE in ("min", "max"):
            cols, color_sum = optimize_cols_color_neighbors(
                cols,
                objective=MOSAIC_NEIGHBOR_OBJECTIVE,
                iters_per_line=MOSAIC_NEIGHBOR_ITERS_PER_LINE,
                seed=OPT_SEED
            )
            layout_info.update({"cols_color": color_sum})

        # 最終的に列順を再度整列して横方向のグラデーションを強調する
        try:
            _rank = {p: i for i, p in enumerate(paths)}
            def _col_key_first(col):
                return min(_rank.get(p, 0) for (p, _, _) in col) if col else 0
            def _col_key_avg(col):
                n = max(1, len(col)); return sum(_rank.get(p, 0) for (p, _, _) in col) / n
            def _col_key_avglab(col):
                n = max(1, len(col)); sL = sa = sb = 0.0
                for (p, _, _) in col:
                    L, a, b = _avg_lab_vector(p)
                    sL += L; sa += a; sb += b
                return (sa / max(1, n), sb / max(1, n), sL / max(1, n))
            uw_order = globals().get('MOSAIC_UW_COL_ORDER', 'first-rank')
            # 最終列順の整列はフルシャッフル無効時のみ行う
            if (not _preserve) and (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and (not bool(globals().get("MOSAIC_ENHANCE_ENABLE", False))):
                if uw_order and uw_order != 'none':
                    if uw_order == 'first-rank':
                        cols.sort(key=_col_key_first)
                    elif uw_order == 'avg-rank':
                        cols.sort(key=_col_key_avg)
                    elif uw_order == 'avgLAB':
                        cols.sort(key=_col_key_avglab)
                    layout_info['cols_order_after'] = uw_order
        except Exception as e:
            _warn_exc_once(e)
            pass
    # -----------------------------------------------------------------------------
    # テンポ並べ替え（post／blend）
    # 列バランス調整や最終整列でテンポ（速い/遅いの交互）が崩れることがあるため、
    # stage が post／blend のときは列内パスをフラット化してテンポ並べ替えを再適用し、
    # 固定列幅 w に合わせて高さを再計算しつつ列を再構築します。
    try:
        if ((not _preserve) and (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False)) and
            globals().get("ARRANGE_TEMPO_ENABLE", False))):
            stage = str(globals().get("ARRANGE_TEMPO_STAGE", "pre")).lower()
            if stage in ("post", "blend"):
                # 参照用にアスペクト比を辞書化
                ars_dict = {p: ar for (p, ar) in ars}
                # 列順のままパスをフラット化
                flat_paths: List[Path] = []
                for col in cols:
                    for (p_i, _w_i, _h_i) in col:
                        flat_paths.append(p_i)
                # フラット列にテンポ並べ替えを適用
                try:
                    reordered = _tempo_apply(list(flat_paths))
                except Exception:
                    reordered = flat_paths
                # 並べ替え後の順番で列を再構築（高さも再計算）
                new_cols: List[list] = []
                idx_fp = 0
                for col in cols:
                    new_col: List[Tuple[Path, int, int]] = []
                    for _ in col:
                        p_new = reordered[idx_fp]; idx_fp += 1
                        ar_new = ars_dict.get(p_new, 1.0)
                        # 固定列幅 w とアスペクト比から高さを計算
                        h_new = max(1, int(round(w / max(1e-6, ar_new))))
                        new_col.append((p_new, w, h_new))
                    new_cols.append(new_col)
                cols = new_cols
                # 以後の参照の整合性のため ars も順序を合わせる（必須ではない）
                try:
                    ars = [(p, ars_dict.get(p, 1.0)) for p in reordered]
                except Exception as e:
                    _warn_exc_once(e)
                    pass
    except Exception:
        # post テンポ適用に失敗しても致命にしない（現状の cols をそのまま使う）
        pass
    # 描画処理：全ての画像をキャンバス内に収めて貼り付ける
    # --- ギャップレス拡張（Uniform Width） ---
    # MOSAIC_GAPLESS_EXTEND が True のとき、表示領域の外側まで縦横にオーバーフィルして描画し、
    # 最後に中央クロップして隙間を消します。追加画像は KANA_SCAN_ALL から取り出し、
    # 供給が尽きるまでは既存画像の重複を避けます。失敗した場合は通常描画へフォールバックします。
    if (not _preserve) and globals().get("MOSAIC_GAPLESS_EXTEND", False):
        try:
            # 初期列で使用されていない画像から供給リストを作成します。このリストは縦方向のオーバーフィルにのみ使用され、列の横方向拡張には使いません。
            all_images = (globals().get("KANA_SCAN_ALL", []) or [])
            used_images_set: set = set()
            for _col in cols:
                for (_p_used, _, _) in _col:
                    used_images_set.add(_p_used)
            # KANA_SCAN_ALL から使用済み画像を除外して供給リストを作成します。
            # 未使用の画像がない場合は、読み込んだアスペクト比リスト "ars" を使用します。
            supply_paths = [p for p in all_images if p not in used_images_set]
            if not supply_paths:
                supply_paths = [p for (p, _) in ars] if ars else []
            # パスの大文字小文字を無視した正規化で重複を除去します。
            # シンボリックリンクや大文字小文字の違いによって同じファイルが複数回出現するのを防ぎます。
            try:
                import os as _supply_os
                def _supply_norm(p):
                    try:
                        return _supply_os.path.normcase(_supply_os.path.normpath(p))
                    except Exception:
                        return p
                _seen_norm: set = set()
                _uniq_supply: list = []
                for _p in supply_paths:
                    _k = _supply_norm(_p)
                    if _k not in _seen_norm:
                        _uniq_supply.append(_p); _seen_norm.add(_k)
                supply_paths = _uniq_supply
            except Exception as e:
                _warn_exc_once(e)
                pass
            # 追加画像（供給リスト）も SELECT_MODE に沿って並べ替えます。
            # random のときは、ソート後にシャッフルしてバイアスを抑えます。
            try:
                supply_paths = sort_by_select_mode(list(supply_paths))
            except Exception as e:
                _warn_exc_once(e)
                pass
            try:
                _mode_now = str(globals().get("SELECT_MODE", "")).lower()
            except Exception:
                _mode_now = ""
            if _mode_now == "random":
                try:
                    _seed = globals().get("SHUFFLE_SEED", None)
                    supply_paths = hash_shuffle(list(supply_paths), _seed, salt="gapless_supply_uw")
                except Exception as e:
                    _warn_exc_once(e)
                    pass
            # 供給リストを順番に返すイテレータです（尽きたら停止（繰り返し許可なら循環））。
            # アスペクト比はキャッシュして、毎回の IO を避けます。
            supply_idx = 0
            supply_ar_cache: dict = {}
            def _next_supply() -> Optional[Tuple[Path, float]]:
                nonlocal supply_idx
                if not supply_paths:
                    return None
                # 初回だけ初期化（closure の状態を保持）
                if not hasattr(_next_supply, "_inited"):
                    _next_supply._inited = True  # type: ignore
                    try:
                        thr = globals().get("MOSAIC_GAPLESS_SUPPLY_DEDUPE_HAMMING", None)
                        thr = int(thr) if thr is not None else 0
                    except Exception:
                        thr = 8
                    _next_supply._thr = int(thr)  # type: ignore
                    _next_supply._used = []  # type: ignore
                    _next_supply._supplied = []  # type: ignore
                    if bool(globals().get("MOSAIC_GAPLESS_SUPPLY_DEDUPE", True)):
                        try:
                            for _p in list(used_images_set):
                                _h = dhash64_for_path_cached(_p)
                                if _h is not None:
                                    _next_supply._used.append(int(_h))  # type: ignore
                        except Exception as e:
                            _kana_silent_exc('core:L11297', e)
                            pass
                max_tries = max(1, int(len(supply_paths)))
                tries = 0
                while tries < max_tries:
                    tries += 1
                    if supply_idx >= len(supply_paths):
                        if bool(globals().get('MOSAIC_GAPLESS_ALLOW_REPEAT', False)):
                            supply_idx = 0
                        else:
                            return None
                    psp = supply_paths[supply_idx]
                    supply_idx += 1
                    # 近似重複スキップ（同じ絵/ほぼ同じ絵を避ける）
                    if bool(globals().get("MOSAIC_GAPLESS_SUPPLY_DEDUPE", True)):
                        try:
                            _h = dhash64_for_path_cached(psp)
                            if _h is not None:
                                _h = int(_h)
                                _thr = int(getattr(_next_supply, "_thr", 8))  # type: ignore
                                _skip = False
                                for _uh in getattr(_next_supply, "_used", []):  # type: ignore
                                    if hamming(_h, int(_uh)) <= _thr:
                                        _skip = True; break
                                if not _skip:
                                    for _sh in getattr(_next_supply, "_supplied", []):  # type: ignore
                                        if hamming(_h, int(_sh)) <= _thr:
                                            _skip = True; break
                                if _skip:
                                    continue
                                getattr(_next_supply, "_supplied", []).append(_h)  # type: ignore
                        except Exception as e:
                            _kana_silent_exc('core:L11329', e)
                            pass
                    ar_sp = supply_ar_cache.get(psp)
                    if ar_sp is None:
                        try:
                            with open_image_safe(psp) as im_sp:
                                iw_sp, ih_sp = im_sp.size
                                ar_sp = iw_sp / float(ih_sp) if ih_sp > 0 else 1.0
                        except Exception:
                            ar_sp = 1.0
                        supply_ar_cache[psp] = ar_sp
                    return (psp, ar_sp)
                return None
            # 既存列をコピーして ext_cols として編集します（各要素は (path, width, height)）。
            ext_cols = [list(col) for col in cols]
            def _col_height(c: list) -> int:
                return sum(hv for (_, _, hv) in c) + gutter * max(0, len(c) - 1)
            # 各列の総高さを計算します
            col_heights = [_col_height(c) for c in ext_cols]
            # 縦方向の目標高さ（オーバーフィル）を決めます。
            # 「最大列高」と「H + 3*w」の大きい方を採用し、クロップ時の丸め誤差で
            # 薄い横スジ（隙間）が出る確率を下げます。
            min_target_h = H + 3 * w
            target_h = max(col_heights + [min_target_h]) if col_heights else min_target_h
            # 各列が target_h に達するまで、供給画像を縦に追加します
            for idx_col, c in enumerate(ext_cols):
                ch = _col_height(c)
                while ch < target_h:
                    nxt = _next_supply()
                    if nxt is None:
                        break
                    p_sup, ar_sup = nxt
                    # 固定列幅 w とアスペクト比から高さを算出します
                    h_sup = max(1, int(round(w / max(1e-6, ar_sup))))
                    if not c:
                        c.append((p_sup, w, h_sup))
                        ch += h_sup
                    else:
                        c.append((p_sup, w, h_sup))
                        ch += h_sup + gutter
                ext_cols[idx_col] = c
            # 1周追加した後、全列を同じ総高さに揃えてクロップを安定させます
            ext_height_total = max([_col_height(c) for c in ext_cols]) if ext_cols else 0
            # 揃えた高さも最低 H + 3*w を確保します
            if ext_height_total < H + 3 * w:
                ext_height_total = H + 3 * w
                for idx_col, c in enumerate(ext_cols):
                    ch = _col_height(c)
                    while ch < ext_height_total:
                        nxt = _next_supply()
                        if nxt is None:
                            break
                        p_sup, ar_sup = nxt
                        h_sup = max(1, int(round(w / max(1e-6, ar_sup))))
                        if not c:
                            c.append((p_sup, w, h_sup))
                            ch += h_sup
                        else:
                            c.append((p_sup, w, h_sup))
                            ch += h_sup + gutter
                    ext_cols[idx_col] = c
                # 追加後の最大高さを再計算します
                ext_height_total = max([_col_height(c) for c in ext_cols]) if ext_cols else 0
            # 拡張後の総幅を計算します（この時点では元の列数なので基本は元の幅です）
            ext_cols_count = len(ext_cols)
            ext_width_total = ext_cols_count * w + gutter * max(0, ext_cols_count - 1)
            # 固定列数モードの場合は横方向のオーバーフィルを行わず、列数を厳守する。
            # そうでない場合は、横方向にもオーバーフィルを行って余分な列を追加し、
            # その後クロップすることで隙間を解消します。
            if not (use_fixed and desired_cols > 0):
                horiz_target = W + 3 * w
                if ext_width_total < horiz_target:
                    while ext_width_total < horiz_target:
                        new_col: list = []
                        ch_new = 0
                        # 新しい列を、現在の拡張高さ ext_height_total まで供給画像で埋めます
                        while ch_new < ext_height_total:
                            nxt = _next_supply()
                            if not nxt:
                                break
                            p_sup, ar_sup = nxt
                            h_sup = max(1, int(round(w / max(1e-6, ar_sup))))
                            if not new_col:
                                new_col.append((p_sup, w, h_sup))
                                ch_new += h_sup
                            else:
                                new_col.append((p_sup, w, h_sup))
                                ch_new += h_sup + gutter
                        ext_cols.append(new_col)
                        ext_cols_count += 1
                        ext_width_total = ext_cols_count * w + gutter * max(0, ext_cols_count - 1)
                    # 追加列を足した後、全列の総高さをもう一度揃えます
                    ext_height_total = max([_col_height(c) for c in ext_cols]) if ext_cols else 0
                    for idx_col, c in enumerate(ext_cols):
                        ch = _col_height(c)
                        while ch < ext_height_total:
                            nxt = _next_supply()
                            if not nxt:
                                break
                            p_sup, ar_sup = nxt
                            h_sup = max(1, int(round(w / max(1e-6, ar_sup))))
                            if not c:
                                c.append((p_sup, w, h_sup))
                                ch += h_sup
                            else:
                                c.append((p_sup, w, h_sup))
                                ch += h_sup + gutter
                        ext_cols[idx_col] = c
                    ext_cols_count = len(ext_cols)
                    ext_width_total = ext_cols_count * w + gutter * max(0, ext_cols_count - 1)
            # 横方向に拡張した場合も含め、余剰分（extra_h/extra_w）からクロップのオフセットを決めます。
            # 基本は中央に寄せる（中心クロップ）イメージです。
            extra_h = ext_height_total - H
            y_off = margin - (extra_h // 2)
            extra_w = ext_width_total - W
            x_off = margin - (extra_w // 2)
            # x_off を「列幅+ガター」のグリッドに揃えます。
            # これにより左端/右端に半端に見切れた列が出るのを抑えます。
            grid = w + gutter
            try:
                rem = (margin - x_off) % grid
            except Exception:
                rem = 0
            x_off += rem
            x_positions: List[int] = []
            for j in range(ext_cols_count):
                x_positions.append(x_off + j * (w + gutter))
            # オーバーフィルした全体モザイクを一度大きな画像として作ります
            mosaic_w = ext_cols_count * w + gutter * max(0, ext_cols_count - 1)
            mosaic_h = ext_height_total
            mosaic_img = Image.new('RGB', (mosaic_w, mosaic_h), bg_rgb)
            mosaic_mask = Image.new('L', (mosaic_w, mosaic_h), 0)
            total_draw = sum(len(c) for c in ext_cols)
            done_cnt = 0
            # --- gaplessモード用 モザイク拡張割り当て（post-pack, Uniform Width） ---
            try:
                if _mosaic_enhance_active():
                    _flat_paths = []
                    _flat_centers = []
                    for _jdx, _ext_col in enumerate(ext_cols):
                        _x_cur = _jdx * (w + gutter)
                        _y_cur = 0
                        for _p_t, _wj_t, _h_t in _ext_col:
                            _flat_paths.append(_p_t)
                            _flat_centers.append((_x_cur + _wj_t / 2.0, _y_cur + _h_t / 2.0))
                            _y_cur += _h_t + gutter

                    if len(_flat_paths) > 1 and len(_flat_centers) == len(_flat_paths):
                        try:
                            banner(_lang("前処理: Mosaic post-pack割当", "Preprocess: Mosaic post-pack assignment"))
                        except Exception as e:
                            _warn_exc_once(e)
                            pass
                        _assigned = _mosaic_post_assign_paths(_flat_paths, _flat_centers)
                        # ext_colsへ書き戻し（アス比から高さ再計算。列幅維持。クロップなし・黒帯なし）
                        _ar_local_cache = {}
                        _k = 0
                        for _jdx, _ext_col in enumerate(ext_cols):
                            for _i in range(len(_ext_col)):
                                try:
                                    _p_old, _wj_old, _h_old = _ext_col[_i]
                                    _new_p = _assigned[_k]
                                    ar_new = None
                                    try:
                                        ar_new = supply_ar_cache.get(_new_p)
                                    except Exception:
                                        ar_new = None
                                    if ar_new is None:
                                        ar_new = _ar_local_cache.get(_new_p)
                                    if ar_new is None:
                                        try:
                                            with open_image_safe(_new_p) as _im_tmp:
                                                _iw_tmp, _ih_tmp = _im_tmp.size
                                            ar_new = _iw_tmp / float(_ih_tmp) if _ih_tmp > 0 else 1.0
                                        except Exception:
                                            ar_new = 1.0
                                        _ar_local_cache[_new_p] = ar_new
                                    _new_h = max(1, int(round(_wj_old / max(ar_new, 1e-6))))
                                    _ext_col[_i] = (_new_p, _wj_old, _new_h)
                                except Exception as e:
                                    _warn_exc_once(e)
                                    pass
                                _k += 1
                        # 再割り当てで高さが変わった場合、各列がtarget_hに届くようtopping upで補充（gapless）
                        try:
                            if '_next_supply' in locals():
                                for _cc_idx, _cc_col in enumerate(ext_cols):
                                    _cur_h2 = 0
                                    for _jj, (_pp2, _ww2, _hh2) in enumerate(_cc_col):
                                        _cur_h2 += _hh2 if _jj == 0 else (_hh2 + gutter)
                                    _cap = int(target_h / max(8, int(round(w * 0.25)))) + 32
                                    while _cur_h2 < target_h and len(_cc_col) < _cap:
                                        _nxt = _next_supply()
                                        if not _nxt:
                                            break
                                        _psp, _ar_sp = _nxt
                                        _h_sp = max(1, int(round(w / max(_ar_sp, 1e-6))))
                                        _cc_col.append((_psp, int(w), _h_sp))
                                        _cur_h2 += _h_sp + gutter
                        except Exception as e:
                            _warn_exc_once(e)
                            pass
                    else:
                        if bool(globals().get("MOSAIC_POST_DEBUG", False)):
                            print(f"[MOSAIC_POST_DEBUG] UW gapless: skip post-pack (tiles={len(_flat_paths)} centers={len(_flat_centers)})")
            except Exception as _pp_ex:
                if bool(globals().get("MOSAIC_POST_DEBUG", False)):
                    print(f"[MOSAIC_POST_DEBUG] UW gapless: post-pack error: {_pp_ex}")
            # --- UW gapless用post-packここまで ---

            # --- draw prefetch（CPU）（mosaic-uniform-width） ---
            _pf_ahead = int(max(0, int(globals().get('DRAW_PREFETCH_AHEAD', 16))))
            _pf_ahead = _effective_draw_prefetch_ahead(width, height, _pf_ahead)
            _pf_workers = int(max(1, int(globals().get('DRAW_PREFETCH_WORKERS', 0) or (os.cpu_count() or 4))))
            _pf_on = bool(globals().get('DRAW_PREFETCH_ENABLE', True)) and (_pf_ahead > 0)

            steps = []      # (p_t, wj_t, h_t, x_cur, y_cur)
            draw_items = [] # (p_t, wj_t, h_t)

            for jdx, ext_col in enumerate(ext_cols):
                x_cur = jdx * (w + gutter)
                y_cur_col = 0
                for (p_t, wj_t, h_t) in ext_col:
                    steps.append((p_t, int(wj_t), int(h_t), int(x_cur), int(y_cur_col)))
                    draw_items.append((p_t, int(wj_t), int(h_t)))
                    y_cur_col += h_t + gutter

            def _mosaic_uw_render(item):
                p_t, wj_t, h_t = item
                with open_image_safe(p_t, draft_to=(max(1, int(wj_t)), max(1, int(h_t))), force_mode='RGB') as im_tt:
                    return hq_resize(im_tt, (max(1, int(wj_t)), max(1, int(h_t))))

            _pf_backend = str(globals().get('DRAW_PREFETCH_BACKEND', ('process' if os.name == 'nt' else 'thread'))).lower()
            _pf_use_mp = _pf_backend in ('process', 'mp', 'multiprocess', 'proc', 'processpool', 'process_pool')

            if _pf_on and draw_items:
                try:
                    if _pf_use_mp:
                        _draw_it = iter(prefetch_ordered_mp_safe(draw_items, _pf_worker_mosaic_uw_render, ahead=_pf_ahead, max_workers=_pf_workers))
                    else:
                        _draw_it = iter(prefetch_ordered_safe(draw_items, _mosaic_uw_render, ahead=_pf_ahead, max_workers=_pf_workers))
                except Exception as _e_pf:
                    print(f"[WARN] mosaic-UW process prefetch unavailable; fallback to thread. reason={_e_pf}")
                    _draw_it = iter(prefetch_ordered_safe(draw_items, _mosaic_uw_render, ahead=_pf_ahead, max_workers=_pf_workers))
            else:
                _pf_on = False
                _draw_it = None


            for (p_t, wj_t, h_t, x_cur, y_cur_col) in steps:
                try:
                    if _pf_on:
                        _item, rez, exc = next(_draw_it)
                        if exc is not None:
                            raise exc
                    else:
                        rez = _mosaic_uw_render((p_t, wj_t, h_t))
                    mosaic_img.paste(rez, (int(x_cur), int(y_cur_col)))
                    mosaic_mask.paste(255, (int(x_cur), int(y_cur_col), int(x_cur + wj_t), int(y_cur_col + h_t)))
                except Exception as ex_draw:
                    print(f"[WARN] {p_t}: {ex_draw}")
                done_cnt += 1
                if VERBOSE:
                    bar(done_cnt, max(1, total_draw), prefix='draw   ', final=False)

            # --- /draw prefetch（CPU）（mosaic-uniform-width） ---
            # 進捗バーを必ず 100% で締めます
            if total_draw == 0:
                bar(done_cnt, 1, prefix='draw   ', final=True)
            else:
                bar(max(done_cnt, total_draw), max(1, total_draw), prefix='draw   ', final=True)
            # 生成したモザイクを横幅 W に合わせてリサイズし、縦は中央クロップします
            try:
                if mosaic_w > 0:
                    scale_x = float(W) / float(mosaic_w)
                else:
                    scale_x = 1.0
                scaled_h = max(1, int(round(mosaic_h * scale_x)))
                try:
                    resized_img = mosaic_img.resize((W, scaled_h), Resampling.LANCZOS)
                except Exception:
                    resized_img = mosaic_img.resize((W, scaled_h), Image.LANCZOS)
                resized_mask = mosaic_mask.resize((W, scaled_h), Resampling.NEAREST)
                extra_v = scaled_h - H
                if extra_v > 0:
                    y_crop = extra_v // 2
                else:
                    y_crop = 0
                sub_img = resized_img.crop((0, y_crop, W, y_crop + H))
                sub_mask = resized_mask.crop((0, y_crop, W, y_crop + H))
                canvas_ext = Image.new('RGB', (width, height), bg_rgb)
                mask_ext = Image.new('L', (width, height), 0)
                canvas_ext.paste(sub_img, (margin, margin))
                mask_ext.paste(sub_mask, (margin, margin))
                layout_info['gapless_scale_fill'] = True
            except Exception:
                # リサイズに失敗した場合は、（必要なら中央寄せしつつ）そのまま貼り付けます
                canvas_ext = Image.new('RGB', (width, height), bg_rgb)
                mask_ext = Image.new('L', (width, height), 0)
                if mosaic_w < W:
                    x_pad = (W - mosaic_w) // 2
                else:
                    x_pad = 0
                crop_y = 0
                if mosaic_h > H:
                    crop_y = (mosaic_h - H) // 2
                sub_img = mosaic_img.crop((0, crop_y, mosaic_w, crop_y + min(H, mosaic_h)))
                sub_mask = mosaic_mask.crop((0, crop_y, mosaic_w, crop_y + min(H, mosaic_h)))
                canvas_ext.paste(sub_img, (margin + x_pad, margin))
                mask_ext.paste(sub_mask, (margin + x_pad, margin))
            layout_info['gapless'] = True
            return canvas_ext, mask_ext, layout_info
        except Exception as gapless_err:
            # 失敗した場合は警告を出して、通常描画にフォールバックします
            print(f"[WARN] gapless mosaic uniform-width extension failed: {gapless_err}")

    # --- モザイク拡張割り当て（post-pack） ---
    try:
        _m_centers = []
        if cols:
            total_w = C * w + gutter * (C - 1)
            x0 = margin + max(0, (W - total_w) // 2)
            x = x0
            for col in cols:
                col_h = sum(h for (_p, _wj, h) in col) + gutter * (len(col) - 1)
                extra_space = max(0, H - col_h)
                base_extra = 0
                extra_rem = 0
                if extra_space > 0 and len(col) > 1 and not bool(globals().get("MOSAIC_UW_GAPLESS_EXTEND", False)):
                    base_extra = extra_space // (len(col) - 1)
                    extra_rem = extra_space % (len(col) - 1)
                    y0 = margin
                else:
                    y0 = margin + max(0, (H - col_h) // 2)

                y = y0
                for i, (_p, _wj, h) in enumerate(col):
                    _m_centers.append((x + w / 2.0, y + h / 2.0))
                    if extra_space > 0 and len(col) > 1 and not bool(globals().get("MOSAIC_UW_GAPLESS_EXTEND", False)):
                        y += h + gutter + base_extra
                        if i < extra_rem:
                            y += 1
                    else:
                        y += h + gutter
                x += w + gutter

        tile_paths_draw = [p for col in cols for (p, _wj, _h) in col]
        if _mosaic_enhance_active() and len(tile_paths_draw) > 1 and len(_m_centers) == len(tile_paths_draw):
            try:
                banner(_lang("前処理: Mosaic post-pack割当", "Preprocess: Mosaic post-pack assignment"))
            except Exception as e:
                _warn_exc_once(e)
                pass
            paths_for_draw = _mosaic_post_assign_paths(tile_paths_draw, _m_centers)
        else:
            paths_for_draw = tile_paths_draw
    except Exception:
        paths_for_draw = [p for col in cols for (p, _wj, _h) in col]

    draw_idx = 0
    canvas = Image.new("RGB", (width, height), bg_rgb)
    mask = Image.new("L", (width, height), 0)
    banner(_lang("処理中: Mosaic / Uniform Width","Rendering: Mosaic / Uniform Width"))
    total = sum(len(col) for col in cols)  # 実際に描画する枚数
    done = 0
    # 計算された列幅と列数から総横幅を求め、左右に余白を均等に配置する
    total_w = C * w + gutter * max(0, C - 1)
    x = margin + max(0, (W - total_w) // 2)
    # 列ごとに描画
    for j in range(C):
            col = cols[j]
            # 各列の合計高さを算出
            col_h = sum(h for _, _, h in col) + gutter * max(0, len(col) - 1)
            # 残り高さ
            extra_space = H - col_h
            num_images = len(col)
            # 縦方向に余りがある場合（かつギャップレス拡張が無効な場合）は、
            # 各タイルの高さを少しずつ増やして列の高さを埋めます
            if extra_space > 0 and num_images > 0 and not globals().get("MOSAIC_GAPLESS_EXTEND", False):
                base_extra = extra_space // num_images
                extra_rem = extra_space % num_images
                new_heights: List[int] = []
                for idx, (_, _, h) in enumerate(col):
                    extra = base_extra + (1 if idx < extra_rem else 0)
                    new_heights.append(h + extra)
                # 列は上端（margin）から積み上げます
                y = margin
                for idx, (p, wj, h) in enumerate(col):
                    nh = new_heights[idx]
                    try:
                        with open_image_safe(p) as im:
                            rez = _fit_rect_no_crop_no_upscale(im, max(1, wj), max(1, nh))
                            canvas.paste(rez, (x, y))
                            mask.paste(255, (x, y, x + wj, y + nh))
                    except Exception as e:
                        print(f"[WARN] {p}: {e}")
                    y += nh + gutter
                    done += 1
                    if VERBOSE:
                        bar(done, max(1, total), prefix="draw   ", final=(done == total))
            else:
                # 既存動作: 画像が複数なら余りをガターに配分し、1枚なら中央寄せします。
                # extra_space <= 0 のケースや、ギャップレス拡張が有効なケースもここで扱います。
                if extra_space > 0 and num_images > 1 and not globals().get("MOSAIC_GAPLESS_EXTEND", False):
                    gaps = num_images - 1
                    base_gutter = gutter + extra_space // gaps
                    extra_rem = extra_space % gaps
                    y = margin
                    for idx, (p, wj, h) in enumerate(col):
                        try:
                            with open_image_safe(p) as im:
                                # モザイクはクロップ/ズームしない（タイルに収めて貼る）
                                rez = _fit_rect_no_crop_no_upscale(im, max(1, wj), max(1, h))
                                canvas.paste(rez, (x, y))
                                mask.paste(255, (x, y, x + wj, y + h))
                        except Exception as e:
                            print(f"[WARN] {p}: {e}")
                        # 次の画像の y 位置を更新
                        if idx < num_images - 1:
                            y += h + base_gutter
                            # 余りピクセルを最初の extra_rem 個の隙間に 1 ピクセルずつ追加
                            if idx < extra_rem:
                                y += 1
                        done += 1
                        if VERBOSE:
                            bar(done, max(1, total), prefix="draw   ", final=(done == total))
                else:
                    # 余りが無い、または画像が 1 枚以下の場合は中央揃えで描画
                    y = margin + max(0, (H - col_h) // 2)
                    for p, wj, h in col:
                        p = paths_for_draw[draw_idx]
                        draw_idx += 1
                        try:
                            with open_image_safe(p) as im:
                                rez = _fit_rect_no_crop_no_upscale(im, max(1, wj), max(1, h))
                                canvas.paste(rez, (x, y))
                                mask.paste(255, (x, y, x + wj, y + h))
                        except Exception as e:
                            print(f"[WARN] {p}: {e}")
                        y += h + gutter
                        done += 1
                        if VERBOSE:
                            bar(done, max(1, total), prefix="draw   ", final=(done == total))
            # 次の列の x 位置を更新
            x += w + gutter
    # 画像が無い場合はプログレスバーを一度だけ更新
    if total == 0:
        bar(done, 1, prefix="draw   ", final=True)
    return canvas, mask, layout_info

# -----------------------------------------------------------------------------
# サブセクション: 明るさ補正（背景マスク込み／None安全）
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# サブセクション: Quilt レイアウト: BSP（ギロチン）分割で大小ブロックを敷き詰め（隙間なし）
# -----------------------------------------------------------------------------
def _quilt_aspect(w: int, h: int) -> float:
    """縦横比（必ず 1 以上）を返す。"""
    if w <= 0 or h <= 0:
        return 1e9
    r = float(w) / float(h)
    return r if r >= 1.0 else (1.0 / r)


def _quilt_ar_group(ratio_w_over_h: float, eps: float) -> int:
    """画像/タイルの縦横比からグループIDを返す。

    - 2: 横長（landscape）
    - 1: 縦長（portrait）
    - 0: 中立（ほぼ正方形）

    eps は「どれくらい縦/横に寄ったら分類するか」の閾値。
    例: eps=0.12 なら、w/h が 1.12 超を横長、1/1.12 未満を縦長。
    """
    try:
        r = float(ratio_w_over_h)
    except Exception:
        r = 1.0
    try:
        e = float(eps)
    except Exception:
        e = 0.12
    e = max(0.0, min(0.9, e))
    hi = 1.0 + e
    lo = 1.0 / hi
    if r >= hi:
        return 2
    if r <= lo:
        return 1
    return 0


def _quilt_read_size_quick(p: Path) -> Tuple[int, int]:
    """画像サイズをなるべく軽く取得する（デコードは最小限）。"""
    try:
        with Image.open(p) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        # 壊れた画像などは中立扱い
        return 1, 1


def _quilt_assign_by_aspect(
    paths: List[Path],
    rects: List[Tuple[int, int, int, int]],
    *,
    eps: float,
) -> Tuple[List[Path], List[int], dict]:
    """タイルの縦横比に合わせて、画像の縦横比が近いものをなるべく割り当てる。

    返り値:
      (new_paths, tile_groups, summary)
    """
    n = min(len(paths), len(rects))
    if n <= 1:
        return list(paths), [0] * len(paths), {"quilt_aspect_match": {"skipped": True, "reason": "n<=1"}}

    # タイル側グループ
    tile_groups: List[int] = []
    tile_ar: List[float] = []
    for (x, y, w, h) in rects[:n]:
        ww = max(1, int(w)); hh = max(1, int(h))
        r = float(ww) / float(hh)
        tile_ar.append(r)
        tile_groups.append(_quilt_ar_group(r, eps))

    # 画像側グループ
    img_wh_cache: Dict[Path, Tuple[int, int]] = {}
    img_ar: Dict[Path, float] = {}
    img_group: Dict[Path, int] = {}
    for p in paths[:n]:
        if p not in img_wh_cache:
            w, h = _quilt_read_size_quick(p)
            img_wh_cache[p] = (w, h)
            r = float(w) / float(max(1, h))
            img_ar[p] = r
            img_group[p] = _quilt_ar_group(r, eps)

    # タイルindexをグループ別に
    idx_by_g = {0: [], 1: [], 2: []}
    for i, g in enumerate(tile_groups):
        idx_by_g.setdefault(g, []).append(i)

    # 画像をグループ別に
    imgs_by_g = {0: [], 1: [], 2: []}
    for p in paths[:n]:
        imgs_by_g.setdefault(img_group.get(p, 0), []).append(p)

    # グループ内は縦横比の近い順で揃える
    # portrait: w/h が小さいほど縦長、landscape: 大きいほど横長
    imgs_by_g[1].sort(key=lambda p: img_ar.get(p, 1.0))
    imgs_by_g[2].sort(key=lambda p: img_ar.get(p, 1.0), reverse=True)
    imgs_by_g[0].sort(key=lambda p: abs(img_ar.get(p, 1.0) - 1.0))

    # タイル側も同様に
    idx_by_g[1].sort(key=lambda i: tile_ar[i])
    idx_by_g[2].sort(key=lambda i: tile_ar[i], reverse=True)
    idx_by_g[0].sort(key=lambda i: abs(tile_ar[i] - 1.0))

    assigned = [None] * n
    used = set()

    def pop_first(lst: List[Path]) -> Optional[Path]:
        while lst:
            p = lst.pop(0)
            if p not in used:
                return p
        return None

    # まず portrait/landscape を優先で埋める
    match_cnt = 0
    for g in (1, 2):
        for i in idx_by_g.get(g, []):
            p = pop_first(imgs_by_g[g])
            if p is None:
                # 中立→反対向きの順で妥協
                p = pop_first(imgs_by_g[0])
            if p is None:
                p = pop_first(imgs_by_g[3-g])  # 1<->2
            if p is None:
                continue
            assigned[i] = p
            used.add(p)
            if img_group.get(p, 0) == g:
                match_cnt += 1

    # 残り（中立タイル等）を埋める
    rest_imgs = []
    for gg in (0, 1, 2):
        rest_imgs.extend([p for p in imgs_by_g.get(gg, []) if p not in used])
    rest_iter = iter(rest_imgs)
    rest_assigned = 0
    for i in range(n):
        if assigned[i] is not None:
            continue
        try:
            p = next(rest_iter)
        except StopIteration:
            p = None
        if p is None:
            break
        assigned[i] = p
        used.add(p)
        rest_assigned += 1
        if img_group.get(p, 0) == tile_groups[i]:
            match_cnt += 1

    # 念のため None を除去しつつ、元の長さに揃える
    out_paths = []
    for i in range(n):
        if assigned[i] is None:
            out_paths.append(paths[i])
        else:
            out_paths.append(assigned[i])

    summ = {
        "quilt_aspect_match": {
            "enabled": True,
            "eps": float(eps),
            "tiles": int(n),
            "tile_portrait": int(len(idx_by_g.get(1, []))),
            "tile_landscape": int(len(idx_by_g.get(2, []))),
            "img_portrait": int(sum(1 for p in paths[:n] if img_group.get(p, 0) == 1)),
            "img_landscape": int(sum(1 for p in paths[:n] if img_group.get(p, 0) == 2)),
            "matched": int(match_cnt),
            "matched_pct": float(match_cnt) / float(max(1, n)) * 100.0,
        }
    }
    return out_paths, tile_groups, summ


def _quilt_range_int(v) -> Tuple[float, float]:
    """(low, high) を float で返す。異常値でも落ちないように丸める。"""
    try:
        low, high = v
        low = float(low); high = float(high)
    except Exception:
        low, high = 0.35, 0.65
    if low > high:
        low, high = high, low
    # 0〜1 にクリップ
    low = max(0.0, min(1.0, low))
    high = max(0.0, min(1.0, high))
    # 極端すぎると分割不能になりやすいので少し救済
    if high - low < 0.05:
        mid = (low + high) * 0.5
        low = max(0.0, mid - 0.05)
        high = min(1.0, mid + 0.05)
    return low, high


def _quilt_split_range_vertical(w: int, h: int, gap: int, min_short: int, max_aspect: float, frac_range: Tuple[float, float]) -> Optional[Tuple[int, int]]:
    """縦分割（左右）で、左の幅 left_w の取り得る整数範囲 (lo,hi) を返す。無理なら None。"""
    if max_aspect <= 0:
        max_aspect = 3.0
    if min_short < 1:
        min_short = 1
    if gap < 0:
        gap = 0
    # gap 分を除いた有効幅
    if w <= gap + 2 * min_short:
        return None
    avail = w - gap
    low_f, high_f = frac_range
    # 縦横比制約（left_w は [ceil(h/maxA), floor(h*maxA)]）
    a_min = int(math.ceil(float(h) / float(max_aspect)))
    a_max = int(math.floor(float(h) * float(max_aspect)))
    # 率による制約
    f_min = int(math.ceil(avail * low_f))
    f_max = int(math.floor(avail * high_f))
    lo = max(min_short, a_min, avail - a_max, f_min)
    hi = min(avail - min_short, a_max, avail - a_min, f_max)
    if lo > hi:
        return None
    return lo, hi


def _quilt_split_range_horizontal(w: int, h: int, gap: int, min_short: int, max_aspect: float, frac_range: Tuple[float, float]) -> Optional[Tuple[int, int]]:
    """横分割（上下）で、上の高さ top_h の取り得る整数範囲 (lo,hi) を返す。無理なら None。"""
    if max_aspect <= 0:
        max_aspect = 3.0
    if min_short < 1:
        min_short = 1
    if gap < 0:
        gap = 0
    if h <= gap + 2 * min_short:
        return None
    avail = h - gap
    low_f, high_f = frac_range
    a_min = int(math.ceil(float(w) / float(max_aspect)))
    a_max = int(math.floor(float(w) * float(max_aspect)))
    f_min = int(math.ceil(avail * low_f))
    f_max = int(math.floor(avail * high_f))
    lo = max(min_short, a_min, avail - a_max, f_min)
    hi = min(avail - min_short, a_max, avail - a_min, f_max)
    if lo > hi:
        return None
    return lo, hi


def _quilt_sample_t(rng: random.Random, split_style: str) -> float:
    """分割位置の 0..1 パラメータ t を生成する。

    split_style:
      - classic : 中央寄り（三角分布）
      - mixed   : 中央寄り＋たまに端寄り
      - extreme : 端寄り多め
      - uniform : 一様
    """
    s = str(split_style or 'classic').strip().lower()
    if s == 'uniform':
        return float(rng.random())
    if s == 'extreme':
        try:
            return float(rng.betavariate(0.7, 0.7))
        except Exception:
            return float(rng.random())
    if s == 'mixed':
        # 7割は中央寄り、3割は端寄り
        try:
            if rng.random() < 0.70:
                return float(rng.betavariate(6.0, 6.0))
            return float(rng.betavariate(0.7, 0.7))
        except Exception:
            return float(rng.random())
    # classic（既定）：中央寄り（三角分布相当）
    return float((rng.random() + rng.random()) * 0.5)


def _quilt_ar_bin_len(edge_len: int, bin_w: int) -> int:
    """分割線長のビン（同じ長さの線が増えすぎないようにするため）。"""
    bw = int(bin_w) if isinstance(bin_w, int) else 8
    if bw <= 0:
        bw = 8
    return int(max(0, int(edge_len) // bw))


def _quilt_ar_bin_area_log(area: float, log_bin: float) -> int:
    """面積のログビン。面積が近いタイルが増えすぎないようにするため。"""
    try:
        a = float(area)
    except Exception:
        a = 1.0
    if a <= 1.0:
        a = 1.0
    try:
        b = float(log_bin)
    except Exception:
        b = 0.18
    if b <= 1e-9:
        b = 0.18
    try:
        return int(math.floor(math.log(a) / b))
    except Exception:
        return 0


def _quilt_try_split_rect(rect: Tuple[int, int, int, int], rng: random.Random, gap: int, min_short: int, max_aspect: float,
                         frac_range: Tuple[float, float], *, split_style: str = 'classic', force_vertical: Optional[bool] = None) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    """rect を 2 分割して子rectを返す。無理なら None。

    force_vertical:
      - True  : 縦分割を優先
      - False : 横分割を優先
      - None  : 既存ロジックで自動
    """
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return None
    # まず分割方向の希望を決める（細長い方向に切りやすくする）
    if force_vertical is None:
        prefer_vertical = None
        if w > h * 1.25:
            prefer_vertical = True
        elif h > w * 1.25:
            prefer_vertical = False
        else:
            prefer_vertical = (rng.random() < 0.5)
    else:
        prefer_vertical = bool(force_vertical)

    # 2回試す（希望方向→反対方向）
    for attempt in range(2):
        vertical = prefer_vertical if attempt == 0 else (not prefer_vertical)
        if vertical:
            r = _quilt_split_range_vertical(w, h, gap, min_short, max_aspect, frac_range)
            if r is None:
                continue
            lo, hi = r
            # 分割位置（t）を split_style に従って選ぶ
            t = _quilt_sample_t(rng, split_style)
            left_w = int(round(lo + t * (hi - lo)))
            left_w = max(lo, min(hi, left_w))
            right_w = (w - gap) - left_w
            r1 = (x, y, left_w, h)
            r2 = (x + left_w + gap, y, right_w, h)
            # 最終ガード（念のため）
            if min(r1[2], r1[3]) < min_short or min(r2[2], r2[3]) < min_short:
                continue
            if _quilt_aspect(r1[2], r1[3]) > max_aspect or _quilt_aspect(r2[2], r2[3]) > max_aspect:
                continue
            return r1, r2
        else:
            r = _quilt_split_range_horizontal(w, h, gap, min_short, max_aspect, frac_range)
            if r is None:
                continue
            lo, hi = r
            t = _quilt_sample_t(rng, split_style)
            top_h = int(round(lo + t * (hi - lo)))
            top_h = max(lo, min(hi, top_h))
            bot_h = (h - gap) - top_h
            r1 = (x, y, w, top_h)
            r2 = (x, y + top_h + gap, w, bot_h)
            if min(r1[2], r1[3]) < min_short or min(r2[2], r2[3]) < min_short:
                continue
            if _quilt_aspect(r1[2], r1[3]) > max_aspect or _quilt_aspect(r2[2], r2[3]) > max_aspect:
                continue
            return r1, r2
    return None


def _quilt_generate_rects_bsp(x0: int, y0: int, w0: int, h0: int, target_tiles: int, rng: random.Random,
                              gap: int, min_short: int, max_aspect: float, frac_range: Tuple[float, float], stop_prob: float,
                              *, split_style: str = 'classic', multi_split_prob: float = 0.0, pick_style: str = 'topk',
                              split_orient_mode: str = 'auto',
                              antirepeat_enable: bool = False,
                              antirepeat_tries: int = 24,
                              antirepeat_len_bin: int = 8,
                              antirepeat_area_log_bin: float = 0.18,
                              antirepeat_w_len: float = 1.0,
                              antirepeat_w_area: float = 0.6,
                              antirepeat_p_len: float = 1.2,
                              antirepeat_p_area: float = 1.1) -> List[Tuple[int, int, int, int]]:
    """BSP 分割で矩形リストを生成する。隙間は gap で表現し、穴は作らない。"""
    if target_tiles < 1:
        target_tiles = 1
    rects: List[Tuple[int, int, int, int]] = [(x0, y0, w0, h0)]
    frozen: set = set()
    # 失敗で無限ループしないように上限
    max_iter = max(200, target_tiles * 80)
    it = 0

    # 分割方向モード（実験用）
    som = str(split_orient_mode or 'auto').strip().lower()
    if som in ('alternate', 'alt', 'vh', 'v-h', 'v_h'):
        som = 'alternate_vh'
    if som in ('hv', 'h-v', 'h_v'):
        som = 'alternate_hv'
    split_count = 0

    # anti-repeat: 分割線長とタイル面積の“重複”を抑えてランダム感を増やす
    ar_en = bool(antirepeat_enable)
    try:
        ar_tries = int(antirepeat_tries)
    except Exception:
        ar_tries = 24
    ar_tries = max(1, min(200, ar_tries))
    try:
        ar_len_bin = int(antirepeat_len_bin)
    except Exception:
        ar_len_bin = 8
    ar_len_bin = max(1, ar_len_bin)
    try:
        ar_area_bin = float(antirepeat_area_log_bin)
    except Exception:
        ar_area_bin = 0.18
    if ar_area_bin <= 1e-9:
        ar_area_bin = 0.18
    try:
        ar_w_len = float(antirepeat_w_len)
    except Exception:
        ar_w_len = 1.0
    try:
        ar_w_area = float(antirepeat_w_area)
    except Exception:
        ar_w_area = 0.6
    try:
        ar_p_len = float(antirepeat_p_len)
    except Exception:
        ar_p_len = 1.2
    try:
        ar_p_area = float(antirepeat_p_area)
    except Exception:
        ar_p_area = 1.1

    len_counts: Dict[int, int] = {}
    area_counts: Dict[int, int] = {}
    if ar_en:
        # 現在の leaf タイル面積ヒストグラム
        b0 = _quilt_ar_bin_area_log(float(w0) * float(h0), ar_area_bin)
        area_counts[b0] = 1

    def _pow(v: int, p: float) -> float:
        try:
            return float(max(0, int(v))) ** float(p)
        except Exception:
            return float(max(0, int(v)))

    def _score_delta_for_split(parent: Tuple[int, int, int, int], child1: Tuple[int, int, int, int], child2: Tuple[int, int, int, int], vertical: bool) -> Tuple[float, int, int, int, int]:
        """anti-repeat の差分コストを計算し、同時に更新用のビンを返す。"""
        px, py, pw, ph = parent
        a_parent = float(max(1, int(pw)) * max(1, int(ph)))
        a1 = float(max(1, int(child1[2])) * max(1, int(child1[3])))
        a2 = float(max(1, int(child2[2])) * max(1, int(child2[3])))
        b_parent = _quilt_ar_bin_area_log(a_parent, ar_area_bin)
        b1 = _quilt_ar_bin_area_log(a1, ar_area_bin)
        b2 = _quilt_ar_bin_area_log(a2, ar_area_bin)
        edge_len = int(ph) if vertical else int(pw)
        b_len = _quilt_ar_bin_len(edge_len, ar_len_bin)

        # length cost（同じ長さの線が増えるのを嫌う）
        c0 = int(len_counts.get(b_len, 0))
        len_old = _pow(c0, ar_p_len)
        len_new = _pow(c0 + 1, ar_p_len)
        d_len = (len_new - len_old)

        # area cost（近い面積のタイルが増えるのを嫌う）
        bins = {b_parent, b1, b2}
        old_cost = 0.0
        new_cost = 0.0
        # 影響するビンだけ計算（重なりも正しく扱う）
        for b in bins:
            c = int(area_counts.get(b, 0))
            old_cost += _pow(c, ar_p_area)
            # new counts
            c2v = c
            if b == b_parent:
                c2v -= 1
            if b == b1:
                c2v += 1
            if b == b2:
                c2v += 1
            new_cost += _pow(c2v, ar_p_area)
        d_area = (new_cost - old_cost)

        score = float(ar_w_len) * float(d_len) + float(ar_w_area) * float(d_area)
        return score, b_len, b_parent, b1, b2

    def _apply_split_hist(b_len: int, b_parent: int, b1: int, b2: int) -> None:
        """anti-repeat のヒストグラムを更新（採用後に呼ぶ）。"""
        len_counts[b_len] = int(len_counts.get(b_len, 0)) + 1
        area_counts[b_parent] = int(area_counts.get(b_parent, 0)) - 1
        if area_counts[b_parent] <= 0:
            try:
                del area_counts[b_parent]
            except Exception as e:
                _kana_silent_exc('core:L12276', e)
                pass
        area_counts[b1] = int(area_counts.get(b1, 0)) + 1
        area_counts[b2] = int(area_counts.get(b2, 0)) + 1

    def _choose_split_antirepeat(parent: Tuple[int, int, int, int], force_vertical: Optional[bool]) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], bool, int, int, int, int]]:
        """anti-repeat 付きで分割候補を複数試し、最も“重複が少ない”候補を選ぶ。"""
        x, y, w, h = parent
        if w <= 0 or h <= 0:
            return None

        # まず分割方向の希望を決める（細長い方向に切りやすくする）
        if force_vertical is None:
            if w > h * 1.25:
                prefer_vertical = True
            elif h > w * 1.25:
                prefer_vertical = False
            else:
                prefer_vertical = (rng.random() < 0.5)
        else:
            prefer_vertical = bool(force_vertical)

        best = None
        best_score = None
        # 2回試す（希望方向→反対方向）
        for attempt in range(2):
            vertical = prefer_vertical if attempt == 0 else (not prefer_vertical)
            if vertical:
                rr = _quilt_split_range_vertical(w, h, gap, min_short, max_aspect, frac_range)
                if rr is None:
                    continue
                lo, hi = rr
                tried = set()
                for _ in range(ar_tries):
                    t = _quilt_sample_t(rng, split_style)
                    cut = int(round(lo + t * (hi - lo)))
                    cut = max(lo, min(hi, cut))
                    if cut in tried:
                        continue
                    tried.add(cut)
                    left_w = int(cut)
                    right_w = (w - gap) - left_w
                    c1 = (x, y, left_w, h)
                    c2 = (x + left_w + gap, y, right_w, h)
                    # 念のためガード
                    if min(c1[2], c1[3]) < min_short or min(c2[2], c2[3]) < min_short:
                        continue
                    if _quilt_aspect(c1[2], c1[3]) > max_aspect or _quilt_aspect(c2[2], c2[3]) > max_aspect:
                        continue
                    sc, b_len, b_parent, b1, b2 = _score_delta_for_split(parent, c1, c2, True)
                    if (best_score is None) or (sc < best_score) or ((sc == best_score) and (rng.random() < 0.5)):
                        best_score = sc
                        best = (c1, c2, True, b_len, b_parent, b1, b2)
            else:
                rr = _quilt_split_range_horizontal(w, h, gap, min_short, max_aspect, frac_range)
                if rr is None:
                    continue
                lo, hi = rr
                tried = set()
                for _ in range(ar_tries):
                    t = _quilt_sample_t(rng, split_style)
                    cut = int(round(lo + t * (hi - lo)))
                    cut = max(lo, min(hi, cut))
                    if cut in tried:
                        continue
                    tried.add(cut)
                    top_h = int(cut)
                    bot_h = (h - gap) - top_h
                    c1 = (x, y, w, top_h)
                    c2 = (x, y + top_h + gap, w, bot_h)
                    if min(c1[2], c1[3]) < min_short or min(c2[2], c2[3]) < min_short:
                        continue
                    if _quilt_aspect(c1[2], c1[3]) > max_aspect or _quilt_aspect(c2[2], c2[3]) > max_aspect:
                        continue
                    sc, b_len, b_parent, b1, b2 = _score_delta_for_split(parent, c1, c2, False)
                    if (best_score is None) or (sc < best_score) or ((sc == best_score) and (rng.random() < 0.5)):
                        best_score = sc
                        best = (c1, c2, False, b_len, b_parent, b1, b2)
            # 片方で良い候補が見つかったら、もう一方も試すが極端に遅くしないため軽くブレイクして良い
            #（ここは“試し”のための実装。必要なら後で調整）
        return best
    while len(rects) < target_tiles and it < max_iter:
        it += 1
        # 分割候補（凍結されていない矩形）を選ぶ
        ps = str(pick_style or 'topk').strip().lower()
        idx = None
        if ps in ('area_weighted', 'weighted', 'area'):
            # 面積で重み付けして選ぶ（大きいほど選ばれやすいが固定になりにくい）
            cand = []
            total = 0.0
            for i, rr in enumerate(rects):
                if i in frozen:
                    continue
                ww = int(rr[2]); hh = int(rr[3])
                if ww <= 0 or hh <= 0:
                    continue
                # これ以上分割できそうにないものは候補から外す（無駄なトライを減らす）
                if (ww < 2 * min_short + gap) and (hh < 2 * min_short + gap):
                    continue
                a = float(ww * hh)
                wgt = a ** 1.15
                cand.append((i, wgt))
                total += wgt
            if cand and total > 0.0:
                r0 = rng.random() * total
                acc = 0.0
                for i, wgt in cand:
                    acc += wgt
                    if acc >= r0:
                        idx = i
                        break
                if idx is None:
                    idx = cand[-1][0]
        if idx is None:
            # 既存：面積上位からランダム
            candidates = [(rr[2] * rr[3], i) for i, rr in enumerate(rects) if i not in frozen]
            if not candidates:
                break
            candidates.sort(reverse=True)
            top = candidates[:min(6, len(candidates))]
            idx = rng.choice([i for _, i in top])

        r = rects[idx]

        # 途中停止（ただしデフォは 0）
        try:
            sp = float(stop_prob)
        except Exception:
            sp = 0.0
        # 早すぎる凍結は枚数到達を阻害しやすいので、ある程度割れてから適用
        if sp > 0.0 and rng.random() < sp and len(rects) >= max(2, int(target_tiles * 0.35)):
            frozen.add(idx)
            continue

        _force = None
        if som == 'alternate_vh':
            _force = True if (split_count % 2 == 0) else False
        elif som == 'alternate_hv':
            _force = True if (split_count % 2 == 1) else False
        ar_meta = None
        if ar_en:
            ar_meta = _choose_split_antirepeat(r, _force)
        if ar_meta is None:
            split = _quilt_try_split_rect(r, rng, gap, min_short, max_aspect, frac_range,
                                          split_style=split_style, force_vertical=_force)
            if split is None:
                frozen.add(idx)
                continue
            r1, r2 = split
        else:
            r1, r2, _vertical, _b_len, _b_parent, _b1, _b2 = ar_meta
            _apply_split_hist(_b_len, _b_parent, _b1, _b2)
        # 置換（pop は index がずれて frozen が破綻するので、代入＋append にする）
        rects[idx] = r1
        rects.append(r2)
        split_count += 1

        # たまに「割った直後にもう一度割る」＝線が増えて複雑に見える
        try:
            msp = float(multi_split_prob)
        except Exception:
            msp = 0.0
        if msp > 0.0 and rng.random() < msp and len(rects) < target_tiles:
            j = (len(rects) - 1) if rng.random() < 0.65 else idx
            # 直前と違う向きに切ると“切り刻み感”が増えやすい。
            # alternate モードの場合は、交互規則を優先して試す。
            _force2 = None
            if som == 'alternate_vh':
                _force2 = True if (split_count % 2 == 0) else False
            elif som == 'alternate_hv':
                _force2 = True if (split_count % 2 == 1) else False
            else:
                _force2 = True if rng.random() < 0.5 else False
            ar_meta2 = None
            if ar_en:
                try:
                    ar_meta2 = _choose_split_antirepeat(rects[j], _force2)
                except Exception:
                    ar_meta2 = None
            if ar_meta2 is None:
                split2 = _quilt_try_split_rect(rects[j], rng, gap, min_short, max_aspect, frac_range,
                                               split_style=split_style, force_vertical=_force2)
                if split2 is not None and len(rects) < target_tiles:
                    rr1, rr2 = split2
                    rects[j] = rr1
                    rects.append(rr2)
                    split_count += 1
            else:
                rr1, rr2, _vertical2, _b_len2, _b_parent2, _b12, _b22 = ar_meta2
                if len(rects) < target_tiles:
                    _apply_split_hist(_b_len2, _b_parent2, _b12, _b22)
                    rects[j] = rr1
                    rects.append(rr2)
                    split_count += 1
    # 見た目＆再現性のため、上から左→右の順にソート
    rects.sort(key=lambda t: (t[1], t[0]))
    return rects


# -----------------------------------------------------------------------------
# サブセクション: Quilt：タイル順（diagonal/hilbert/scatter/as-is）と近傍anneal（anneal）
# -----------------------------------------------------------------------------
def _quilt_rect_center(r: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = r
    return (float(x) + float(w) * 0.5, float(y) + float(h) * 0.5)


def _quilt_slot_order(rects: List[Tuple[int, int, int, int]], profile: str, diag_dir: str, *, hilbert_bits: int = 8) -> List[int]:
    """rects の順序（スロット順）を決める。

    - profile:
        - "as_is"    : 現状維持（入力順）
        - "scatter"  : ざっくりばらけ（Hilbert順→偶奇インタリーブ）
        - "diagonal" : 対角方向（diag_dir）
        - "hilbert"  : Hilbert 曲線（空間的連続）
    """
    n = len(rects)
    if n <= 1:
        return list(range(n))
    p = str(profile or '').strip().lower()
    if p in ('as-is', 'as_is', 'asis'):
        return list(range(n))

    centers = [_quilt_rect_center(r) for r in rects]
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max(1e-9, max_x - min_x)
    dy = max(1e-9, max_y - min_y)
    qn = (1 << int(max(2, min(12, hilbert_bits)))) - 1

    def qxy(i: int) -> Tuple[int, int]:
        cx, cy = centers[i]
        xi = int(round((cx - min_x) / dx * qn))
        yi = int(round((cy - min_y) / dy * qn))
        if xi < 0:
            xi = 0
        elif xi > qn:
            xi = qn
        if yi < 0:
            yi = 0
        elif yi > qn:
            yi = qn
        return xi, yi

    if p == 'hilbert':
        order = list(range(n))
        order.sort(key=lambda i: _hilbert_index(*qxy(i), order=int(max(2, min(12, hilbert_bits)))))
        return order

    if p == 'scatter':
        # まず Hilbert で近接を作り、偶奇でインタリーブして“ばらけ”を作る
        base = list(range(n))
        base.sort(key=lambda i: _hilbert_index(*qxy(i), order=int(max(2, min(12, hilbert_bits)))))
        ev = base[::2]
        od = base[1::2]
        # 端まで散るように、odd は逆順で混ぜる
        od.reverse()
        out = []
        for a, b in zip(ev, od):
            out.append(a)
            out.append(b)
        out.extend(ev[len(od):])
        out.extend(od[len(ev):])
        return out

    # diagonal
    d = str(diag_dir or 'tl_br').strip().lower()
    # 方向ごとにキーを変える
    def key_diag(i: int):
        cx, cy = centers[i]
        if d in ('tl_br', 'tlbr', 'tl-br'):
            return (cx + cy, cy, cx)
        if d in ('tr_bl', 'trbl', 'tr-bl'):
            return ((-cx) + cy, cy, -cx)
        if d in ('bl_tr', 'bltr', 'bl-tr'):
            return (cx + (-cy), -cy, cx)
        if d in ('br_tl', 'brtl', 'br-tl'):
            return ((-cx) + (-cy), -cy, -cx)
        return (cx + cy, cy, cx)

    order = list(range(n))
    order.sort(key=key_diag)
    return order


def _quilt_neighbor_graph_rects(rects: List[Tuple[int, int, int, int]], gap: int, *, tol: int = 2) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    """Quilt の矩形から隣接グラフ（共有辺に近いもの）を作る。

    gap が 0 でなくても、edge-to-edge の距離が gap 前後で、かつ垂直/水平の投影が重なるものを近傍とする。
    """
    n = len(rects)
    if n <= 1:
        return [], [[] for _ in range(n)]

    # 右/下端を事前計算
    boxes = []
    for (x, y, w, h) in rects:
        x1 = int(x)
        y1 = int(y)
        x2 = int(x) + int(w)
        y2 = int(y) + int(h)
        boxes.append((x1, y1, x2, y2))

    neigh: List[List[int]] = [[] for _ in range(n)]
    edges: List[Tuple[int, int]] = []
    g = int(max(0, gap))
    thr = int(g + int(max(0, tol)))

    def _overlap_1d(a1: int, a2: int, b1: int, b2: int) -> int:
        return min(a2, b2) - max(a1, b1)

    # O(n^2) だが n<=200 程度なら十分速い
    for i in range(n):
        ax1, ay1, ax2, ay2 = boxes[i]
        for j in range(i + 1, n):
            bx1, by1, bx2, by2 = boxes[j]

            # 横方向の近接（左右）
            dx1 = abs(ax2 - bx1)
            dx2 = abs(bx2 - ax1)
            if (dx1 <= thr) or (dx2 <= thr):
                oy = _overlap_1d(ay1, ay2, by1, by2)
                if oy > 0:
                    neigh[i].append(j)
                    neigh[j].append(i)
                    edges.append((i, j))
                    continue

            # 縦方向の近接（上下）
            dy1 = abs(ay2 - by1)
            dy2 = abs(by2 - ay1)
            if (dy1 <= thr) or (dy2 <= thr):
                ox = _overlap_1d(ax1, ax2, bx1, bx2)
                if ox > 0:
                    neigh[i].append(j)
                    neigh[j].append(i)
                    edges.append((i, j))

    return edges, neigh


def optimize_quilt_neighbors_anneal(
    paths: List[Path],
    edges: List[Tuple[int, int]],
    neigh: List[List[int]],
    *,
    steps: int,
    T0: float,
    Tend: float,
    reheats: int,
    seed: int,
    objective: str,
    groups: Optional[List[int]] = None,
) -> Tuple[List[Path], dict]:
    """Quilt の近傍色差を swap でanneal最適化する。

    objective:
      - "min" : 隣接色差を最小化（滑らか）
      - "max" : 隣接色差を最大化（ばらけ）
    """
    n = len(paths)
    if n <= 1 or not edges:
        return list(paths), {"quilt_neighbor_anneal": {"skipped": True, "reason": "n<=1 or no edges"}}

    obj = str(objective or 'min').strip().lower()
    if obj not in ('min', 'max'):
        obj = 'min'

    # LAB キャッシュ
    lab_cache = {}
    for p in paths:
        if p not in lab_cache:
            lab_cache[p] = _avg_lab_vector(p)

    def dist(pa: Path, pb: Path) -> float:
        return _vec_dist(lab_cache[pa], lab_cache[pb])

    # 現在割当
    assigned = list(paths)

    # 初期スコア（ΣΔ）
    s0 = 0.0
    for (i, j) in edges:
        s0 += dist(assigned[i], assigned[j])

    # cost を最小化する形に統一
    def score_to_cost(s: float) -> float:
        return -s if obj == 'max' else s

    curr_score = float(s0)
    curr_cost = score_to_cost(curr_score)
    best_score = float(curr_score)
    best_cost = float(curr_cost)
    best_assigned = list(assigned)

    rnd = random.Random(int(seed) if isinstance(seed, int) else 0)

    # グループ制約（任意）：同じグループ内だけで swap する
    # 例: Quilt のタイル形状（縦長/横長/中立）に合わせた割り当てを崩さないため
    group_bins = None
    group_keys = []
    group_w = []
    group_total = 0
    if groups is not None and isinstance(groups, list) and len(groups) == n:
        try:
            bins: Dict[int, List[int]] = {}
            for idx, g in enumerate(groups):
                gg = int(g) if isinstance(g, int) else int(str(g))
                bins.setdefault(gg, []).append(idx)
            # swap 可能な（2要素以上）グループのみ残す
            bins = {k: v for k, v in bins.items() if len(v) >= 2}
            if bins:
                group_bins = bins
                group_keys = list(group_bins.keys())
                group_w = [len(group_bins[k]) for k in group_keys]
                group_total = int(sum(group_w))
        except Exception:
            group_bins = None
    rehs = int(max(0, int(reheats)))
    total_steps = int(max(1, int(steps)))
    phases = rehs + 1
    # Grid の anneal と同じ解釈：steps は「総ステップ数」。phases（=reheats+1）に均等配分します。
    base = total_steps // phases
    rem = total_steps % phases

    # 進捗表示
    # 表示名は他の anneal 系（mosaic/hex 等）に合わせ、"anneal" 表現は使わない
    banner(_lang("最適化: Quilt neighbor anneal", "Optimize: Quilt neighbor anneal"))
    try:
        note(f"Objective: {obj} | steps={total_steps} | reheats={rehs} (phases={phases})")
    except Exception as e:
        _kana_silent_exc('core:L12708', e)
        pass
    # i/j に関係するエッジ集合を作る
    def affected_edges(i: int, j: int) -> List[Tuple[int, int]]:
        es = set()
        for a in (i, j):
            for b in neigh[a]:
                u, v = (a, b) if a < b else (b, a)
                es.add((u, v))
        return list(es)

    done = 0
    for ph in range(phases):
        phase_steps = base + (1 if ph < rem else 0)
        # 再加熱ごとに温度を戻す
        t0 = float(T0)
        t1 = float(Tend)
        if t0 <= 0:
            t0 = 1.0
        if t1 <= 0:
            t1 = 1e-3

        for s in range(phase_steps):
            if group_bins is not None and group_total > 0:
                # サイズで重み付けしてグループを選ぶ
                r = rnd.randrange(group_total)
                acc = 0
                g_sel = group_keys[0]
                for k, wgt in zip(group_keys, group_w):
                    acc += int(wgt)
                    if acc > r:
                        g_sel = k
                        break
                lst = group_bins.get(g_sel, None)
                if not lst:
                    # 念のためフォールバック
                    i = rnd.randrange(n)
                    j = rnd.randrange(n)
                else:
                    i = lst[rnd.randrange(len(lst))]
                    j = lst[rnd.randrange(len(lst))]
            else:
                i = rnd.randrange(n)
                j = rnd.randrange(n)
            if i == j:
                done += 1
                if VERBOSE:
                    bar(done, total_steps, prefix="anneal", final=(done == total_steps))
                continue
            if i > j:
                i, j = j, i

            egs = affected_edges(i, j)
            old_part = 0.0
            for (u, v) in egs:
                old_part += dist(assigned[u], assigned[v])

            # swap
            assigned[i], assigned[j] = assigned[j], assigned[i]

            new_part = 0.0
            for (u, v) in egs:
                new_part += dist(assigned[u], assigned[v])

            new_score = curr_score - old_part + new_part
            new_cost = score_to_cost(new_score)
            dcost = new_cost - curr_cost

            # 温度（線形）
            t = t0 + (t1 - t0) * (float(s) / float(max(1, phase_steps - 1)))
            accept = False
            if dcost <= 0:
                accept = True
            else:
                # exp(-dcost/T)
                try:
                    prob = math.exp(-dcost / max(1e-12, t))
                    accept = (rnd.random() < prob)
                except Exception:
                    accept = False

            if accept:
                curr_score = float(new_score)
                curr_cost = float(new_cost)
                if curr_cost < best_cost:
                    best_cost = float(curr_cost)
                    best_score = float(curr_score)
                    best_assigned = list(assigned)
            else:
                # revert
                assigned[i], assigned[j] = assigned[j], assigned[i]

            done += 1
            if VERBOSE:
                bar(done, total_steps, prefix="anneal", final=(done == total_steps))

    # best を返す
    assigned = best_assigned
    sf = float(best_score)
    imp = 0.0
    if obj == 'max' and s0 > 0:
        imp = (sf - s0) / s0 * 100.0
    elif obj == 'min' and s0 > 0:
        imp = (s0 - sf) / s0 * 100.0
    try:
        note(f"ΣΔColor(quilt): {s0:.1f} → {sf:.1f} ({imp:+.1f}%)")
    except Exception as e:
        _kana_silent_exc('core:L12815', e)
        pass
    summary = {
        "quilt_neighbor_anneal": {
            "objective": obj,
            "edges": int(len(edges)),
            "initial": float(s0),
            "final": float(sf),
            "imp_pct": float(imp),
            "steps": int(total_steps),
            "reheats": int(rehs),
        }
    }
    return assigned, summary



# =============================================================================
# セクション: レイアウト: Quilt
# - レイアウト生成の本体（配置/合成/マスクなど）
# =============================================================================

def layout_quilt_bsp(paths: List[Path], width: int, height: int, margin: int, gutter: int,
                     bg_rgb: Tuple[int, int, int], *, rng: Optional[random.Random] = None):
    """Quilt（Mondrian）: BSP 分割で大小ブロックを敷き詰めて描画する。"""
    global _PRINTED_LAYOUT_ONCE
    if rng is None:
        rng = random.Random()

    # 設定（無ければ既定）
    try:
        max_tiles = int(globals().get('QUILT_MAX_TILES', 0) or 0)
    except Exception:
        max_tiles = 0
    try:
        min_short = int(globals().get('QUILT_MIN_SHORT', 220) or 220)
    except Exception:
        min_short = 220
    try:
        max_aspect = float(globals().get('QUILT_MAX_ASPECT', 3.0) or 3.0)
    except Exception:
        max_aspect = 3.0
    frac_range = _quilt_range_int(globals().get('QUILT_SPLIT_RANGE', (0.12, 0.88)))
    try:
        stop_prob = float(globals().get('QUILT_STOP_PROB', 0.0) or 0.0)
    except Exception:
        stop_prob = 0.0
    split_style = str(globals().get('QUILT_SPLIT_STYLE', 'classic') or 'classic').strip()
    try:
        multi_split_prob = float(globals().get('QUILT_MULTI_SPLIT_PROB', 0.0) or 0.0)
    except Exception:
        multi_split_prob = 0.0
    pick_style = str(globals().get('QUILT_PICK_STYLE', 'topk') or 'topk').strip()
    split_orient_mode = str(globals().get('QUILT_SPLIT_ORIENT_MODE', 'auto') or 'auto').strip().lower()

    # anti-repeat（分割線/面積の重複抑制）
    try:
        ar_en = bool(globals().get('QUILT_ANTIREPEAT_ENABLE', False))
    except Exception:
        ar_en = False
    try:
        ar_tries = int(globals().get('QUILT_ANTIREPEAT_TRIES', 24) or 24)
    except Exception:
        ar_tries = 24
    try:
        ar_len_bin = int(globals().get('QUILT_ANTIREPEAT_LEN_BIN', 8) or 8)
    except Exception:
        ar_len_bin = 8
    try:
        ar_area_log_bin = float(globals().get('QUILT_ANTIREPEAT_AREA_LOG_BIN', 0.18) or 0.18)
    except Exception:
        ar_area_log_bin = 0.18
    try:
        ar_w_len = float(globals().get('QUILT_ANTIREPEAT_W_LEN', 1.0) or 1.0)
    except Exception:
        ar_w_len = 1.0
    try:
        ar_w_area = float(globals().get('QUILT_ANTIREPEAT_W_AREA', 0.6) or 0.6)
    except Exception:
        ar_w_area = 0.6
    try:
        ar_p_len = float(globals().get('QUILT_ANTIREPEAT_P_LEN', 1.2) or 1.2)
    except Exception:
        ar_p_len = 1.2
    try:
        ar_p_area = float(globals().get('QUILT_ANTIREPEAT_P_AREA', 1.1) or 1.1)
    except Exception:
        ar_p_area = 1.1

    # 描画モード（Fit/Fill）
    mode = str(globals().get('MODE', 'fill')).strip().lower()
    if mode not in ('fit', 'fill'):
        mode = 'fill'

    # face/person/saliency focus の統計（_FDBG/_FDBG2）は、Quilt 描画の開始時にリセットします。
    # これにより、Quilt 1回分の検出内訳（frontal/profile/person/saliency/center 等）を正確に表示できます。
    try:
        if (mode != 'fit') and bool(globals().get('FACE_FOCUS_ENABLE', True)):
            global _FDBG, _FDBG2
            _FDBG = {"cv2": None, "frontal":0, "profile":0, "anime":0, "ai":0, "upper":0, "person":0, "saliency":0, "center":0,
                     "reject_pos":0, "reject_ratio":0, "errors":0}
            _FDBG2 = {"eyes_ok":0, "eyes_ng":0, "low_reject":0,
                      "anime_face_ok":0, "anime_face_ng":0,
                      "anime_eyes_ok":0, "anime_eyes_ng":0, "ai_face_ok":0, "ai_face_ng":0}
    except Exception as e:
        _kana_silent_exc('core:L12914', e)
        pass
    total_imgs = len(paths)
    target_tiles = total_imgs if max_tiles <= 0 else min(total_imgs, max_tiles)
    target_tiles = max(0, int(target_tiles))

    # ルート領域（整数ピクセルで扱う）
    x0 = int(margin); y0 = int(margin)
    w0 = int(max(1, width - 2 * int(margin)))
    h0 = int(max(1, height - 2 * int(margin)))
    gap = int(max(0, gutter))

    # ルートが制約を満たさない場合は、制約側を少し救済
    if min(w0, h0) < min_short:
        min_short = max(1, min(w0, h0))
    if _quilt_aspect(w0, h0) > max_aspect:
        # 画面全体が極端に細長いことは稀だが、念のため上限を緩めて落ちないようにする
        max_aspect = max(_quilt_aspect(w0, h0), max_aspect)

    rects = _quilt_generate_rects_bsp(
        x0, y0, w0, h0,
        max(1, target_tiles) if target_tiles > 0 else 1,
        rng,
        gap=gap,
        min_short=min_short,
        max_aspect=max_aspect,
        frac_range=frac_range,
        stop_prob=stop_prob,
        split_style=split_style,
        multi_split_prob=multi_split_prob,
        pick_style=pick_style,
        split_orient_mode=split_orient_mode,
        antirepeat_enable=ar_en,
        antirepeat_tries=ar_tries,
        antirepeat_len_bin=ar_len_bin,
        antirepeat_area_log_bin=ar_area_log_bin,
        antirepeat_w_len=ar_w_len,
        antirepeat_w_area=ar_w_area,
        antirepeat_p_len=ar_p_len,
        antirepeat_p_area=ar_p_area,
    )

    # 画像が少ない場合は rects を詰める。多い場合は paths を詰める。
    if target_tiles <= 0:
        rects = []
    use_n = min(len(paths), len(rects))
    use_paths = paths[:use_n]
    use_rects = rects[:use_n]

    layout_info = {
        'style': 'quilt',
        'tiles': int(len(use_rects)),
        'gap': int(gap),
        'min_short': int(min_short),
        'max_aspect': float(max_aspect),
        'split_range': (float(frac_range[0]), float(frac_range[1])),
        'split_style': str(split_style),
        'stop_prob': float(stop_prob),
        'multi_split_prob': float(multi_split_prob),
        'pick_style': str(pick_style),
        'split_orient_mode': str(split_orient_mode),
        'antirepeat_enable': bool(ar_en),
        'antirepeat_tries': int(ar_tries),
        'antirepeat_len_bin': int(ar_len_bin),
        'antirepeat_area_log_bin': float(ar_area_log_bin),
        'antirepeat_w_len': float(ar_w_len),
        'antirepeat_w_area': float(ar_w_area),
    }

    # --- Quilt: タイル縦横比に合わせて、画像の縦横比もなるべく合わせる（任意） ---
    tile_groups = None
    try:
        _am_en = bool(globals().get('QUILT_ASPECT_MATCH_ENABLE', False))
    except Exception:
        _am_en = False
    if _am_en and use_n >= 2:
        try:
            _am_eps = float(globals().get('QUILT_ASPECT_MATCH_EPS', 0.12) or 0.12)
        except Exception:
            _am_eps = 0.12
        try:
            use_paths, tile_groups, _am_summ = _quilt_assign_by_aspect(list(use_paths), list(use_rects), eps=_am_eps)
            try:
                layout_info.update(_am_summ)
            except Exception as e:
                _kana_silent_exc('core:L12999', e)
                pass
        except Exception as e:
            _warn_exc_once(e)
            tile_groups = None

    # --- Quilt: 配置順（place）＋近傍最適化（anneal） ---
    try:
        _q_enh = bool(globals().get('QUILT_ENHANCE_ENABLE', False))
    except Exception:
        _q_enh = False
    try:
        _q_profile = str(globals().get('QUILT_ENHANCE_PROFILE', 'hilbert') or 'hilbert').strip().lower()
    except Exception:
        _q_profile = 'hilbert'
    try:
        _q_diag = str(globals().get('QUILT_DIAGONAL_DIRECTION', 'tl_br') or 'tl_br').strip().lower()
    except Exception:
        _q_diag = 'tl_br'
    try:
        _q_obj = str(globals().get('QUILT_NEIGHBOR_OBJECTIVE', 'min') or 'min').strip().lower()
    except Exception:
        _q_obj = 'min'
    if _q_obj not in ('min', 'max'):
        _q_obj = 'min'
    try:
        _q_opt = str(globals().get('QUILT_OPTIMIZER', 'none') or 'none').strip().lower()
    except Exception:
        _q_opt = 'none'
    try:
        _q_anneal_en = bool(globals().get('QUILT_ANNEAL_ENABLE', False))
    except Exception:
        _q_anneal_en = False
    _q_do_anneal = _q_anneal_en or (_q_opt in ('anneal', 'sa', 'simulated-anneal', 'simulated_anneal', 'simulated-annealing'))

    if _q_enh and use_n >= 2:
        # まず rect の順序（スロット順）を決める
        try:
            _slot = _quilt_slot_order(list(use_rects), _q_profile, _q_diag, hilbert_bits=8)
            if len(_slot) == len(use_rects):
                use_rects = [use_rects[i] for i in _slot]
        except Exception as e:
            _warn_exc_once(e)

        # 次に画像側の初期順序（目的に合わせて）
        try:
            if _q_profile in ('diagonal', 'hilbert') and _q_obj == 'min':
                use_paths = reorder_global_spectral_hilbert(list(use_paths), objective='min')
            elif _q_profile == 'scatter' and _q_obj == 'max':
                # まずは軽いシャッフルで“ばらけ”の初期状態を作る
                _seed = globals().get('OPT_SEED', 'random')
                _seed_i = _seed_to_int(_seed)
                _sv = (_seed_i if isinstance(_seed_i, int) else globals().get('_RUN_SEED_USED', secrets.randbits(32)))
                hash_shuffle_inplace(use_paths, _sv, salt='quilt_scatter')
        except Exception as e:
            _warn_exc_once(e)

        # 最後に anneal（任意）：矩形タイルの隣接グラフで swap 最適化
        if _q_do_anneal and use_n >= 3 and _q_obj in ('min', 'max'):
            try:
                _steps = int(globals().get('QUILT_ANNEAL_STEPS', 0) or 0)
            except Exception:
                _steps = 0
            try:
                _t0 = float(globals().get('QUILT_ANNEAL_T0', 1.0) or 1.0)
            except Exception:
                _t0 = 1.0
            try:
                _t1 = float(globals().get('QUILT_ANNEAL_TEND', 1e-3) or 1e-3)
            except Exception:
                _t1 = 1e-3
            try:
                _reh = int(globals().get('QUILT_ANNEAL_REHEATS', 0) or 0)
            except Exception:
                _reh = 0
            # OPT_SEED を優先（固定なら再現性あり）
            _seed = globals().get('OPT_SEED', 'random')
            _seed_i = _seed_to_int(_seed)
            _seed_i = int(_seed_i) if isinstance(_seed_i, int) else int(globals().get('_RUN_SEED_USED', secrets.randbits(32)))
            if _steps > 0:
                edges, neigh = _quilt_neighbor_graph_rects(list(use_rects), gap)
                use_paths, summ = optimize_quilt_neighbors_anneal(
                    list(use_paths), edges, neigh,
                    steps=_steps, T0=_t0, Tend=_t1,
                    reheats=_reh, seed=_seed_i, objective=_q_obj,
                    groups=tile_groups,
                )
                layout_info.update(summ)

    # 表示用（アスペクト相性）
    am_disp = 'off'
    try:
        if _am_en and isinstance(layout_info.get('quilt_aspect_match'), dict):
            _d = layout_info.get('quilt_aspect_match')
            _pct = float(_d.get('matched_pct', 0.0) or 0.0)
            am_disp = f"on({_pct:.1f}%)"
    except Exception:
        am_disp = 'on'

    if not _PRINTED_LAYOUT_ONCE:
        try:
            note(
                f"Layout: quilt tiles={len(use_rects)} gap={gap} min_short={min_short} max_aspect={max_aspect} "
                f"split_style={split_style} orient={split_orient_mode} stop_prob={stop_prob} multi_split={multi_split_prob} pick={pick_style} "
                f"antirepeat={'on' if ar_en else 'off'}(tries={ar_tries},lb={ar_len_bin},ab={ar_area_log_bin}) "
                f"aspect_match={am_disp}"
            )
        except Exception as e:
            _kana_silent_exc('core:L13106', e)
            pass
        _PRINTED_LAYOUT_ONCE = True

    canvas = Image.new('RGB', (width, height), bg_rgb)
    mask = Image.new('L', (width, height), 0)

    done = 0
    # layout_info は上で作成済み

    # quilt: fill のときはタイルごとに face-focus を適用（任意）
    _quilt_use_ff = (mode == 'fill'
                     and bool(globals().get('QUILT_FACE_FOCUS_ENABLE', True))
                     and bool(globals().get('FACE_FOCUS_ENABLE', True)))

    # --- draw prefetch（CPU）：ジョブを組んでから、スレッド/プロセスでタイル描画します ---
    jobs = []  # (path, x, y, w, h)
    for p, (x, y, w, h) in zip(use_paths, use_rects):
        jobs.append((p, int(x), int(y), max(1, int(w)), max(1, int(h))))

    _pf_ahead = int(max(0, int(globals().get('DRAW_PREFETCH_AHEAD', 16))))
    _pf_ahead = _effective_draw_prefetch_ahead(width, height, _pf_ahead)
    _pf_workers = int(max(1, int(globals().get('DRAW_PREFETCH_WORKERS', 0) or (os.cpu_count() or 4))))
    _pf_on = bool(globals().get('DRAW_PREFETCH_ENABLE', True)) and (_pf_ahead > 0)

    def _quilt_render(job):
        p, _x, _y, w, h = job
        # quilt でも grid と同じキャッシュ描画パスを使う
        if _quilt_use_ff:
            tile = _tile_render_cached(p, w, h, 'fill', use_face_focus=True)
            return 'fill_ff', tile
        if mode == 'fit':
            tile = _tile_render_cached(p, w, h, 'fit', use_face_focus=False)
            return 'fit', tile
        tile = _tile_render_cached(p, w, h, 'fill', use_face_focus=False)
        return 'fill', tile

    _pf_backend = str(globals().get('DRAW_PREFETCH_BACKEND', ('process' if os.name == 'nt' else 'thread'))).lower()
    _pf_use_mp = _pf_backend in ('process', 'mp', 'multiprocess', 'proc', 'processpool', 'process_pool')

    if _pf_on and jobs:
        if _pf_use_mp:
            try:
                # 既存の grid 用ワーカーを流用（(path,w,h,mode,use_ff)）
                _pf_items = [(job[0], job[3], job[4], mode, _quilt_use_ff) for job in jobs]
                _pf_stream = prefetch_ordered_mp_safe(_pf_items, _pf_worker_grid_render, ahead=_pf_ahead, max_workers=_pf_workers)

                def _wrap_pf():
                    for i, (_item, _res, _exc) in enumerate(_pf_stream):
                        yield jobs[i], _res, _exc

                _it = _wrap_pf()
            except Exception as _e_pf:
                print(f"[WARN] process prefetch unavailable; fallback to thread. reason={_e_pf}")
                _it = prefetch_ordered_safe(jobs, _quilt_render, ahead=_pf_ahead, max_workers=_pf_workers)
        else:
            _it = prefetch_ordered_safe(jobs, _quilt_render, ahead=_pf_ahead, max_workers=_pf_workers)
    else:
        _it = ((job, _quilt_render(job), None) for job in jobs)

    done = 0
    for job, out, exc in _it:
        p, x, y, w, h = job
        try:
            if exc is not None:
                raise exc
            kind, tile = out
            if kind == 'fit':
                rx = x + (w - tile.size[0]) // 2
                ry = y + (h - tile.size[1]) // 2
                canvas.paste(tile, (rx, ry))
                mask.paste(255, (rx, ry, rx + tile.size[0], ry + tile.size[1]))
            else:
                canvas.paste(tile, (x, y))
                mask.paste(255, (x, y, x + w, y + h))
        except Exception:
            # フォールバック（旧パス）
            try:
                with open_image_safe(p, draft_to=(w, h)) as im:
                    paste_cell(canvas, mask, im, x, y, w, h, mode)
            except Exception as e2:
                _warn_exc_once(e2)
                pass

        done = min(done + 1, max(1, use_n))
        try:
            bar(done, max(1, use_n), prefix='draw   ', final=(done == use_n))
        except Exception as e:
            _kana_silent_exc('core:L13193', e)
            pass
    # --- /draw prefetch（CPU） ---

    if use_n == 0:
        try:
            bar(1, 1, prefix='draw   ', final=True)
        except Exception as e:
            _kana_silent_exc('core:L13201', e)
            pass
    # Quilt の face/person/saliency focus 統計を表示（DEBUG 有効時）
    # ※ grid/hex と同様に、1回の実行分の内訳だけを出します。
    try:
        if (mode != 'fit') and bool(globals().get('FACE_FOCUS_ENABLE', True)) and bool(globals().get('FACE_FOCUS_DEBUG', True)):
            _note_face_focus_stats(_FDBG, _FDBG2)
    except Exception as e:
        _kana_silent_exc('core:L13209', e)
        pass
    return canvas, mask, layout_info


def _poly_area_xy(pts: List[Tuple[float, float]]) -> float:
    """多角形面積（符号付き）。"""
    if not pts or len(pts) < 3:
        return 0.0
    a = 0.0
    x0, y0 = pts[-1]
    for x1, y1 in pts:
        a += (x0 * y1 - x1 * y0)
        x0, y0 = x1, y1
    return 0.5 * a


def _poly_angles_deg(pts: List[Tuple[float, float]]) -> List[float]:
    """多角形（点列順）の各頂点の内角（度）を返す。
    Voronoi facet は基本的に凸多角形の想定。
    """
    if not pts or len(pts) < 3:
        return []
    angs: List[float] = []
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[(i - 1) % n]
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        ux, uy = (x0 - x1), (y0 - y1)
        vx, vy = (x2 - x1), (y2 - y1)
        nu = math.hypot(ux, uy)
        nv = math.hypot(vx, vy)
        if nu <= 1e-9 or nv <= 1e-9:
            angs.append(180.0)
            continue
        dot = (ux * vx + uy * vy) / (nu * nv)
        # 数値誤差対策
        dot = max(-1.0, min(1.0, dot))
        ang = math.degrees(math.acos(dot))
        angs.append(float(ang))
    return angs


def _poly_simplify_max_angle(pts: List[Tuple[float, float]], max_angle_deg: float, min_vertices: int) -> List[Tuple[float, float]]:
    """内角が max_angle_deg を超える（≒ほぼ一直線）頂点を間引いて、形をスッキリさせる。
    1回で大量に削ると崩れるので、最大角の頂点を1つずつ削る（安定優先）。
    """
    if not pts or len(pts) < 3:
        return pts
    max_angle_deg = float(max_angle_deg)
    min_vertices = int(max(3, min_vertices))
    # 実質フィルタ無効
    if max_angle_deg >= 179.999:
        return pts
    out = list(pts)
    # 反復上限（安全策）
    for _ in range(512):
        if len(out) <= min_vertices:
            break
        angs = _poly_angles_deg(out)
        if not angs:
            break
        mx = max(angs)
        if mx <= max_angle_deg:
            break
        # 最大角の頂点を1つ削除
        idx = angs.index(mx)
        out.pop(idx)
    return out


# -----------------------------------------------------------------------------
# stained-glass: polygon helpers (module-level)
#   ※ v133 で _poly_enforce_constraints() が参照する補助関数が
#     誤ってローカルスコープ内（layout 関数内）に置かれていたため、
#     ここでモジュールスコープに定義して NameError を解消します。
# -----------------------------------------------------------------------------

def _poly_sanitize(pts, eps=1e-6, bbox=None):
    """頂点列を軽く正規化する。

    - 連続重複点の除去
    - NaN/inf の排除
    - bbox=(x0,y0,x1,y1) が与えられた場合のみ、その範囲にクランプ

    ※ 以前は 0..1 へのクランプをしていたが、stained-glass はピクセル座標を扱うため
       ここで 0..1 に潰すと面積が消えて Voronoi が全滅する（WarpGrid へ落ちる）ので禁止。
    """
    if not pts:
        return []

    x0 = y0 = x1 = y1 = None
    if bbox is not None:
        x0, y0, x1, y1 = bbox

    cleaned = []
    last = None
    for p in pts:
        try:
            x = float(p[0])
            y = float(p[1])
        except Exception as e:
            _kana_silent_exc('core:L13312', e)
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue

        if bbox is not None:
            # bbox にクランプ
            if x < x0:
                x = x0
            elif x > x1:
                x = x1
            if y < y0:
                y = y0
            elif y > y1:
                y = y1

        if last is None:
            cleaned.append((x, y))
            last = (x, y)
            continue

        if abs(x - last[0]) <= eps and abs(y - last[1]) <= eps:
            continue
        cleaned.append((x, y))
        last = (x, y)

    # 先頭と末尾の重複も落とす
    if len(cleaned) >= 2:
        if abs(cleaned[0][0] - cleaned[-1][0]) <= eps and abs(cleaned[0][1] - cleaned[-1][1]) <= eps:
            cleaned.pop()

    return cleaned


def _poly_sort_ccw(pts):
    """頂点を重心の周りで CCW に並べ替える（Voronoi セル用の簡易版）。"""
    if not pts or len(pts) <= 2:
        return list(pts) if pts else []

    # ローカル import で依存関係を増やさない
    import math as _math

    cx = sum(p[0] for p in pts) / float(len(pts))
    cy = sum(p[1] for p in pts) / float(len(pts))
    ordered = sorted(pts, key=lambda p: _math.atan2(p[1] - cy, p[0] - cx))

    # CCW になっているかの簡易チェック（符号付き面積）
    area2 = 0.0
    for i in range(len(ordered)):
        x1, y1 = ordered[i]
        x2, y2 = ordered[(i + 1) % len(ordered)]
        area2 += (x1 * y2 - x2 * y1)
    if area2 < 0.0:
        ordered.reverse()

    return ordered

def _clip_poly_rect(poly, x0, y0, x1, y1):
    """ポリゴンを矩形でクリップ（Sutherland–Hodgman）。
    注意: これは“押し込み(clamp)”ではなく、境界との交点を作って正しく切り取る。
    端に接するセルで角度制約が破綻しないようにするための重要処理。
    """
    if not poly or len(poly) < 3:
        return []

    x0 = float(x0); y0 = float(y0); x1 = float(x1); y1 = float(y1)

    def inside(p, edge):
        x, y = p
        if edge == 0:   # left
            return x >= x0
        if edge == 1:   # right
            return x <= x1
        if edge == 2:   # top
            return y >= y0
        return y <= y1  # bottom

    def intersect(a, b, edge):
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay

        if edge == 0:  # x = x0
            if abs(dx) < 1e-12:
                return (x0, ay)
            t = (x0 - ax) / dx
            return (x0, ay + t * dy)
        if edge == 1:  # x = x1
            if abs(dx) < 1e-12:
                return (x1, ay)
            t = (x1 - ax) / dx
            return (x1, ay + t * dy)
        if edge == 2:  # y = y0
            if abs(dy) < 1e-12:
                return (ax, y0)
            t = (y0 - ay) / dy
            return (ax + t * dx, y0)
        # y = y1
        if abs(dy) < 1e-12:
            return (ax, y1)
        t = (y1 - ay) / dy
        return (ax + t * dx, y1)

    out = [(float(x), float(y)) for (x, y) in poly]
    for edge in (0, 1, 2, 3):
        if not out:
            break
        inp = out
        out = []
        prev = inp[-1]
        prev_in = inside(prev, edge)
        for cur in inp:
            cur_in = inside(cur, edge)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur, edge))
                out.append(cur)
            else:
                if prev_in:
                    out.append(intersect(prev, cur, edge))
            prev, prev_in = cur, cur_in

    # 量子化/交点生成で同一点が連続しやすいので軽く間引く（順序は維持）
    if len(out) >= 2:
        dedup = [out[0]]
        for p in out[1:]:
            if abs(p[0] - dedup[-1][0]) < 1e-9 and abs(p[1] - dedup[-1][1]) < 1e-9:
                continue
            dedup.append(p)
        out = dedup
    if len(out) >= 3 and abs(out[0][0] - out[-1][0]) < 1e-9 and abs(out[0][1] - out[-1][1]) < 1e-9:
        out = out[:-1]

    return out


def _poly_convex_hull_ccw(pts):
    """凸包（Monotonic chain, CCW）。Voronoi facet の頂点順が崩れても自己交差を避けやすくする。
    ※制約（最大角度など）はこの後で別途かける。ここは“順序安定化”が目的。
    """
    if not pts or len(pts) < 3:
        return list(pts) if pts else []

    # 極小だけ丸めて重複除去
    uniq = sorted(set((round(float(x), 6), round(float(y), 6)) for x, y in pts))
    if len(uniq) < 3:
        return []

    pts2 = [(float(x), float(y)) for x, y in uniq]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts2:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts2):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return hull

def _poly_enforce_constraints(
    pts: List[Tuple[float, float]],
    min_vertices: int,
    max_vertices: int,
    max_corner_angle_deg: float,
    bbox: Tuple[float, float, float, float],
    lead_width: int = 0,
) -> List[Tuple[float, float]]:
    """
    stained-glass 用ポリゴン制約（※タイル整合性優先）

    重要:
      - 近傍セルとのタイル整合性を壊さないため、頂点を「内側へ曲げる」等の変形は行わない。
      - 代わりに「境界上の点の追加（辺の分割）」と「境界上でほぼ一直線な点の削除」に限定する。
      - max_corner_angle は “ほぼ一直線な点(数値誤差/追加点)” を落とす用途として扱う。
        ※Voronoi 由来の本来の角（セルのコーナー）を削る/曲げると境界が崩れ、黒線/隙間の原因になるため。

    これにより:
      - 謎の黒線（境界が二重に出る/画像貼りが境界に従わない）を優先的に抑える。
      - 厳密な max_corner_angle を常に満たすことは保証できない（タイル整合性を優先）。
    """
    if not pts:
        return []

    x0, y0, x1, y1 = bbox

    # 入力の整形
    pts = _poly_sanitize(pts, bbox=(x0, y0, x1, y1))
    if len(pts) < 3:
        return pts

    min_vertices = int(max(3, min_vertices))
    max_vertices = int(max_vertices)
    if max_vertices <= 0:
        # 0=無制限（上限を課さない）
        max_vertices = 0
    else:
        max_vertices = int(max(3, max_vertices))
        if max_vertices < min_vertices:
            max_vertices = min_vertices

    x0, y0, x1, y1 = bbox

    def _clamp01(p: Tuple[float, float]) -> Tuple[float, float]:
        return (min(max(p[0], x0), x1), min(max(p[1], y0), y1))

    pts = [_clamp01(p) for p in pts]
    pts = _poly_sort_ccw(pts)

    # 点→線分距離^2（小さいほど線分上）
    def _p2seg2(p, a, b):
        px, py = p
        ax, ay = a
        bx, by = b
        vx, vy = (bx - ax), (by - ay)
        wx, wy = (px - ax), (py - ay)
        vv = vx * vx + vy * vy
        if vv <= 1e-12:
            dx, dy = (px - ax), (py - ay)
            return dx * dx + dy * dy
        t = (wx * vx + wy * vy) / vv
        if t <= 0.0:
            dx, dy = (px - ax), (py - ay)
            return dx * dx + dy * dy
        if t >= 1.0:
            dx, dy = (px - bx), (py - by)
            return dx * dx + dy * dy
        projx = ax + t * vx
        projy = ay + t * vy
        dx, dy = (px - projx), (py - projy)
        return dx * dx + dy * dy

    # ほぼ一直線の点だけを削除（境界線を壊さない）
    # lead_width があると線が太くなるので、許容を少しだけ広げる
    eps_base = max(0.75, float(lead_width) * 0.35)  # px

    # max_corner_angle_deg を“できるだけ寄せる”ための軽量クリーニング
    # max_vertices が無制限(0)でも、クリップ交点などで増えた「ほぼ一直線」点だけを落とします。
    # ※線分上の点を除去するだけなので、タイル整合性を壊しにくい方針。
    try:
        _ang_thr = float(max_corner_angle_deg)
    except Exception:
        _ang_thr = 179.5
    # 安全側クランプ（180 は計算誤差で不安定なので少し手前）
    _ang_thr = max(0.0, min(179.99, _ang_thr))

    def _remove_nearly_straight_points(pts_in: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        pts2 = list(pts_in)
        if len(pts2) <= 3:
            return pts2
        eps = eps_base
        eps2 = float(eps) * float(eps)
        guard = 0
        while len(pts2) > min_vertices and len(pts2) > 3 and guard < 20000:
            guard += 1
            angs = _poly_angles_deg(pts2)
            n = len(pts2)
            removed = False
            for i in range(n):
                a = float(angs[i])
                if a <= _ang_thr:
                    continue
                p_prev = pts2[(i - 1) % n]
                p = pts2[i]
                p_next = pts2[(i + 1) % n]
                d2 = _p2seg2(p, p_prev, p_next)
                if d2 <= eps2:
                    pts2.pop(i)
                    removed = True
                    break
            if not removed:
                break
        return pts2

    pts = _remove_nearly_straight_points(pts)


    def _reduce_vertices_to_limit(pts_in: List[Tuple[float, float]], limit: int) -> List[Tuple[float, float]]:
        """max_vertices を“なるべく境界を壊さず”に厳守する。

        方針:
          1) まずは「ほぼ一直線」かつ「線分からのズレが小さい」点だけを落とす。
             → 近傍セルとの整合性を壊しにくい。
          2) それでも落とし切れない場合は、許容(eps)を段階的に広げて同方針で削る。
          3) 最終手段として、面積寄与が極小の点を落とす（発生は稀）。
        """
        pts2 = list(pts_in)
        if limit <= 0 or len(pts2) <= limit:
            return pts2
        if len(pts2) <= 3:
            return pts2

        ang_thr = float(max_corner_angle_deg)

        def _best_candidate(pts_cur: List[Tuple[float, float]], eps2: float, strict_straight_only: bool) -> int:
            angs = _poly_angles_deg(pts_cur)
            n = len(pts_cur)
            best_key = None
            best_i = -1
            for i in range(n):
                a = float(angs[i])
                if strict_straight_only and a <= ang_thr:
                    continue
                p_prev = pts_cur[(i - 1) % n]
                p = pts_cur[i]
                p_next = pts_cur[(i + 1) % n]
                d2 = _p2seg2(p, p_prev, p_next)
                if d2 > eps2:
                    continue
                # 直線(角度大) & ずれ小 を優先して落とす
                penalty = 0 if a > ang_thr else 1
                key = (penalty, d2, -a)
                if best_key is None or key < best_key:
                    best_key = key
                    best_i = i
            return best_i

        # まずは “直線っぽい点” を中心に、eps を段階的に広げながら落とす
        eps_seq = [eps_base, eps_base * 1.5, eps_base * 2.0, eps_base * 3.0, eps_base * 4.0, 6.0, 8.0]
        guard = 0
        for eps in eps_seq:
            eps2 = float(eps) * float(eps)
            while len(pts2) > limit and len(pts2) > 3 and guard < 20000:
                guard += 1
                idx = _best_candidate(pts2, eps2, strict_straight_only=True)
                if idx < 0:
                    break
                pts2.pop(idx)
            if len(pts2) <= limit:
                return pts2
        # それ以上はここでは削らずに返します（隙間を作りにくくするため）
        return pts2

    # まず max_vertices を超える場合は、境界を壊しにくい点から優先して落とす
    if max_vertices > 0 and len(pts) > max_vertices:
        pts = _reduce_vertices_to_limit(pts, max_vertices)


    # min_vertices に足りない場合は、境界上で辺を分割して点を追加（内側へは動かさない）
    if len(pts) < min_vertices:
        guard = 0
        while len(pts) < min_vertices and guard < 5000:
            guard += 1
            n = len(pts)
            # 最長辺を分割
            best_i = 0
            best_d2 = -1.0
            for i in range(n):
                a = pts[i]
                b = pts[(i + 1) % n]
                dx, dy = (b[0] - a[0]), (b[1] - a[1])
                d2 = dx * dx + dy * dy
                if d2 > best_d2:
                    best_d2 = d2
                    best_i = i
            a = pts[best_i]
            b = pts[(best_i + 1) % n]
            mid = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)  # 境界上
            pts.insert(best_i + 1, _clamp01(mid))

    # 追加後に max_vertices を超えていたら、境界を壊しにくい点から優先して落とす
    if max_vertices > 0 and len(pts) > max_vertices:
        pts = _reduce_vertices_to_limit(pts, max_vertices)

    pts = _poly_sanitize(pts, bbox=(x0, y0, x1, y1))
    if len(pts) >= 3:
        pts = _poly_sort_ccw(pts)
    return pts


# =============================================================================
# StainedGlass: グローバルメッシュ簡略化（角数制約の緩和・隙間抑制）
# =============================================================================

def _sg_global_mesh_simplify_polys(
    polys: List[List[Tuple[float, float]]],
    bbox: Tuple[float, float, float, float],
    min_vertices: int,
    max_vertices: int,
    lead_width: int = 0,
) -> List[List[Tuple[float, float]]]:
    """ステンドグラス用: メッシュ全体（隣接セル共有）で角数上限を満たすための簡略化（グローバル処理）。

    ねらい:
      - セル単体で頂点を削ると、隣セルとの境界がズレて「黒い隙間」になりやすい。
      - ここでは “頂点” をメッシュ共有として扱い、頂点の統合（edge collapse）を全セルに一括反映する。
      - inpaint / WarpGrid への自動切替を使わず、幾何学的なタイル整合性を保ったまま減らす。

    注意:
      - max_vertices が極端に小さい（例: 4）場合、Voronoi 由来の形状を大きく崩さずに“必ず”満たすのは難しい。
      - その場合でも「隙間を作らない」ことを優先し、達成不能なら達成できた範囲で止める。
    """
    try:
        minv = int(max(3, int(min_vertices)))
    except Exception:
        minv = 3
    try:
        maxv = int(max_vertices)
    except Exception:
        maxv = 0
    if maxv <= 0:
        return polys  # 無制限

    # 量子化: 隣接セルの共有頂点を同一視する（float誤差の吸収）
    #  - ここが大きすぎると別頂点まで潰すので控えめに。
    eps = float(max(1e-3, min(0.25, 0.02 + float(max(0, lead_width)) * 0.01)))  # px相当
    inv = 1.0 / eps

    def _sg_is_finite_f(v: float) -> bool:
        # math.isfinite を使わずに NaN/Inf を弾く（環境差を吸収）
        try:
            if v != v:  # NaN
                return False
            if v == float("inf") or v == float("-inf"):
                return False
            # 極端な値も除外（丸めの例外対策）
            if v > 1e308 or v < -1e308:
                return False
            return True
        except Exception:
            return False

    def qkey(p: Tuple[float, float]) -> Tuple[int, int]:
        try:
            fx = float(p[0])
            fy = float(p[1])
        except Exception:
            fx, fy = 0.0, 0.0
        if not (_sg_is_finite_f(fx) and _sg_is_finite_f(fy)):
            fx, fy = 0.0, 0.0
        # 量子化キー
        return (int(round(fx * inv)), int(round(fy * inv)))

    # vertices
    verts: List[Tuple[float, float]] = []
    vmap: Dict[Tuple[int, int], int] = {}

    def vid(p: Tuple[float, float]) -> int:
        k = qkey(p)
        i = vmap.get(k)
        if i is None:
            i = len(verts)
            try:
                fx = float(p[0])
                fy = float(p[1])
            except Exception:
                fx, fy = 0.0, 0.0
            if not (_sg_is_finite_f(fx) and _sg_is_finite_f(fy)):
                fx, fy = 0.0, 0.0
            verts.append((fx, fy))
            vmap[k] = i
        return i

    faces: List[List[int]] = []
    for poly in polys:
        if not poly or len(poly) < 3:
            faces.append([])
            continue
        ids = [vid(p) for p in poly]
        # 連続重複を除去
        cleaned: List[int] = []
        for i in ids:
            if not cleaned or cleaned[-1] != i:
                cleaned.append(i)
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
            cleaned.pop()
        if len(cleaned) < 3:
            faces.append([])
            continue
        faces.append(cleaned)

    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    def poly_area(face: List[int], override_p: Optional[Dict[int, Tuple[float, float]]] = None) -> float:
        if not face or len(face) < 3:
            return 0.0
        s = 0.0
        n = len(face)
        for i in range(n):
            a = face[i]
            b = face[(i + 1) % n]
            if override_p is None:
                ax, ay = verts[a]
                bx, by = verts[b]
            else:
                # override_p.get(key, verts[key]) だと default が先に評価され、
                # 仮ID（new_id=len(verts)）で IndexError になり得るため、明示分岐にする
                if a in override_p:
                    ax, ay = override_p[a]
                else:
                    ax, ay = verts[a]
                if b in override_p:
                    bx, by = override_p[b]
                else:
                    bx, by = verts[b]
            s += ax * by - ay * bx
        return 0.5 * s

    def is_simple_polygon(face: List[int], override_p: Optional[Dict[int, Tuple[float, float]]] = None) -> bool:
        """自己交差チェック（小さな多角形想定でO(n^2)）。"""
        if not face or len(face) < 3:
            return False
        n = len(face)

        def pt(i: int) -> Tuple[float, float]:
            if override_p is None:
                return verts[i]
            # override_p.get(i, verts[i]) だと default が先に評価され、
            # 仮ID（new_id=len(verts)）で IndexError になり得る
            if i in override_p:
                return override_p[i]
            return verts[i]

        def seg_intersect(a, b, c, d) -> bool:
            # 端点共有は交差扱いにしない（隣接エッジは除外して呼ぶ）
            def orient(p, q, r) -> float:
                return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

            def on_seg(p, q, r) -> bool:
                return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and
                        min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)

            o1 = orient(a, b, c)
            o2 = orient(a, b, d)
            o3 = orient(c, d, a)
            o4 = orient(c, d, b)

            if (o1 == 0.0 and on_seg(a, c, b)) or (o2 == 0.0 and on_seg(a, d, b)) or \
               (o3 == 0.0 and on_seg(c, a, d)) or (o4 == 0.0 and on_seg(c, b, d)):
                return True

            return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

        # エッジ列
        pts = [pt(i) for i in face]
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            # bbox外へ大きく出るのは拒否（微小誤差は許容）
            if (a[0] < x0 - 2.0) or (a[0] > x1 + 2.0) or (a[1] < y0 - 2.0) or (a[1] > y1 + 2.0):
                return False
            if (b[0] < x0 - 2.0) or (b[0] > x1 + 2.0) or (b[1] < y0 - 2.0) or (b[1] > y1 + 2.0):
                return False

        for i in range(n):
            a1 = pts[i]
            a2 = pts[(i + 1) % n]
            for j in range(i + 1, n):
                # 隣接/同一エッジはスキップ
                if j == i:
                    continue
                if (j + 1) % n == i:
                    continue
                if (i + 1) % n == j:
                    continue
                b1 = pts[j]
                b2 = pts[(j + 1) % n]
                # 端点共有はOK
                if a1 == b1 or a1 == b2 or a2 == b1 or a2 == b2:
                    continue
                if seg_intersect(a1, a2, b1, b2):
                    return False
        return True

    def cleanup_face(face: List[int]) -> List[int]:
        if not face:
            return face
        out: List[int] = []
        for v in face:
            if not out or out[-1] != v:
                out.append(v)
        if len(out) >= 2 and out[0] == out[-1]:
            out.pop()
        return out

    # edge adjacency
    def build_edge_faces(faces_cur: List[List[int]]) -> Dict[Tuple[int, int], List[int]]:
        ef: Dict[Tuple[int, int], List[int]] = {}
        for fi, f in enumerate(faces_cur):
            if not f or len(f) < 3:
                continue
            n = len(f)
            for i in range(n):
                a = f[i]
                b = f[(i + 1) % n]
                if a == b:
                    continue
                k = (a, b) if a < b else (b, a)
                ef.setdefault(k, []).append(fi)
        return ef

    # Stage A: global collinear cleanup (degree-2 vertices)
    changed = True
    guard_a = 0
    while changed and guard_a < 20:
        guard_a += 1
        changed = False
        neigh: Dict[int, set] = {}
        occ: Dict[int, List[Tuple[int, int]]] = {}
        for fi, f in enumerate(faces):
            if not f or len(f) < 3:
                continue
            n = len(f)
            for i, v in enumerate(f):
                occ.setdefault(v, []).append((fi, i))
                v_prev = f[(i - 1) % n]
                v_next = f[(i + 1) % n]
                neigh.setdefault(v, set()).add(v_prev)
                neigh.setdefault(v, set()).add(v_next)

        to_remove: List[int] = []
        for v, ns in neigh.items():
            if len(ns) != 2:
                continue
            a, b = list(ns)
            vx, vy = verts[v]
            ax, ay = verts[a]
            bx, by = verts[b]
            px, py = vx, vy
            vx1, vy1 = (bx - ax), (by - ay)
            wx, wy = (px - ax), (py - ay)
            vv = vx1 * vx1 + vy1 * vy1
            if vv <= 1e-12:
                continue
            t = (wx * vx1 + wy * vy1) / vv
            if t < 0.0 or t > 1.0:
                continue
            projx = ax + t * vx1
            projy = ay + t * vy1
            dx, dy = (px - projx), (py - projy)
            d2 = dx * dx + dy * dy
            if d2 <= (eps * 2.5) ** 2:
                to_remove.append(v)

        if not to_remove:
            break

        for v in to_remove:
            if v not in occ:
                continue
            for fi, _ in occ[v]:
                f = faces[fi]
                if not f or len(f) < 4:
                    continue
                faces[fi] = [x for x in f if x != v]
                faces[fi] = cleanup_face(faces[fi])
            changed = True

    # Stage B: edge collapse to reduce vertex count (global)
    max_collapses = int(max(0, globals().get("STAINED_GLASS_GLOBAL_SIMPLIFY_MAX_COLLAPSES", 800)))
    collapses = 0

    def face_has_duplicate(face: List[int]) -> bool:
        return len(set(face)) != len(face)

    def try_collapse(edge: Tuple[int, int], ef: Dict[Tuple[int, int], List[int]]) -> Optional[Tuple[int, Dict[int, int], Dict[int, Tuple[float, float]]]]:
        nonlocal verts
        a, b = edge
        ax, ay = verts[a]
        bx, by = verts[b]
        px, py = (ax + bx) * 0.5, (ay + by) * 0.5
        px = float(min(max(px, x0), x1))
        py = float(min(max(py, y0), y1))

        new_id = len(verts)  # 仮ID（override_pで扱う）
        override_p = {new_id: (px, py)}

        affected: List[int] = []
        for fi, f in enumerate(faces):
            if not f or len(f) < 3:
                continue
            if (a in f) or (b in f):
                affected.append(fi)

        for fi in affected:
            f = faces[fi]
            tmp = [new_id if (v == a or v == b) else v for v in f]
            tmp = cleanup_face(tmp)
            if len(tmp) < 3:
                return None
            if len(tmp) < minv:
                return None
            if face_has_duplicate(tmp):
                return None
            ar = abs(poly_area(tmp, override_p=override_p))
            if ar <= 2.0:
                return None
            if not is_simple_polygon(tmp, override_p=override_p):
                return None

        rep = {a: new_id, b: new_id}
        return new_id, rep, override_p

    def apply_collapse(new_id: int, rep: Dict[int, int], new_pos: Tuple[float, float]) -> None:
        nonlocal verts, faces
        verts.append((float(new_pos[0]), float(new_pos[1])))
        a_old, b_old = list(rep.keys())
        for fi, f in enumerate(faces):
            if not f:
                continue
            if (a_old in f) or (b_old in f):
                tmp = [new_id if (v == a_old or v == b_old) else v for v in f]
                tmp = cleanup_face(tmp)
                faces[fi] = tmp

    def current_bad_faces() -> List[int]:
        return [fi for fi, f in enumerate(faces) if f and len(f) > maxv]

    while collapses < max_collapses:
        bad = current_bad_faces()
        if not bad:
            break

        ef = build_edge_faces(faces)

        cand: List[Tuple[float, Tuple[int, int]]] = []
        for fi in bad:
            f = faces[fi]
            n = len(f)
            for i in range(n):
                a = f[i]
                b = f[(i + 1) % n]
                if a == b:
                    continue
                k = (a, b) if a < b else (b, a)
                ax, ay = verts[k[0]]
                bx, by = verts[k[1]]
                dx, dy = (ax - bx), (ay - by)
                l2 = dx * dx + dy * dy
                boundary_pen = 5.0 if len(ef.get(k, [])) <= 1 else 0.0
                cand.append((l2 + boundary_pen, k))

        if not cand:
            break
        cand.sort(key=lambda t: t[0])

        applied = False
        for _, e in cand[:200]:
            res = try_collapse(e, ef)
            if res is None:
                continue
            new_id, rep, override_p = res
            px, py = override_p[new_id]
            apply_collapse(new_id, rep, (px, py))
            collapses += 1
            applied = True
            break

        if not applied:
            break

    out_polys: List[List[Tuple[float, float]]] = []
    for f in faces:
        if not f or len(f) < 3:
            out_polys.append([])
            continue
        pts = [verts[i] for i in f]
        pts = _poly_sanitize(pts, bbox=bbox)
        if len(pts) >= 3:
            try:
                if _poly_area_xy(pts) < 0.0:
                    pts = list(reversed(pts))
            except Exception:
                pass
        out_polys.append(pts)

    return out_polys


def _voronoi_facets_python(pts, bbox):
    """Voronoi セル（凸多角形）を Python だけで生成する。

    pts: [(x,y), ...]  ※ピクセル座標
    bbox: (x0,y0,x1,y1) ※セルをこの矩形でクリップ

    OpenCV(Subdiv2D) が不安定な環境でも確実に多角形が得られるようにするための保険。
    n が 30〜100 程度なら十分実用。
    """
    x0, y0, x1, y1 = bbox
    x0 = float(x0)
    y0 = float(y0)
    x1 = float(x1)
    y1 = float(y1)

    base_poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    # 半平面 ax*x + ay*y <= b でクリップ（Sutherland–Hodgman）
    def _inside(p, ax, ay, b, tol=1e-7):
        return (ax * p[0] + ay * p[1]) <= (b + tol)

    def _intersect(p1, p2, ax, ay, b):
        # 線分 p1->p2 と境界 ax*x + ay*y = b の交点
        x1_, y1_ = p1
        x2_, y2_ = p2
        d1 = ax * x1_ + ay * y1_ - b
        d2 = ax * x2_ + ay * y2_ - b
        denom = (d2 - d1)
        if abs(denom) < 1e-12:
            return p2  # ほぼ平行: 端点を返す（後段 sanitize で整える）
        t = d1 / (d1 - d2)  # 0..1
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        return (x1_ + (x2_ - x1_) * t, y1_ + (y2_ - y1_) * t)

    def _clip_halfplane(poly, ax, ay, b):
        if not poly:
            return []
        out = []
        prev = poly[-1]
        prev_in = _inside(prev, ax, ay, b)
        for cur in poly:
            cur_in = _inside(cur, ax, ay, b)
            if cur_in:
                if not prev_in:
                    out.append(_intersect(prev, cur, ax, ay, b))
                out.append(cur)
            elif prev_in:
                out.append(_intersect(prev, cur, ax, ay, b))
            prev = cur
            prev_in = cur_in
        return out

    facets = []
    centers = []

    pts_f = [(float(p[0]), float(p[1])) for p in pts]

    for i, (pix, piy) in enumerate(pts_f):
        poly = base_poly[:]
        for j, (pjx, pjy) in enumerate(pts_f):
            if j == i:
                continue
            dx = pjx - pix
            dy = pjy - piy
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue  # 同一点は無視

            # 近い側（pi）を残す半平面
            ax = 2.0 * dx
            ay = 2.0 * dy
            b = (pjx * pjx + pjy * pjy) - (pix * pix + piy * piy)

            poly = _clip_halfplane(poly, ax, ay, b)
            if len(poly) < 3:
                poly = []
                break

        if poly:
            poly = _poly_sanitize(poly, bbox=bbox)
            if len(poly) >= 3:
                poly = _poly_sort_ccw(poly)

        if not poly or len(poly) < 3:
            facets.append([])
            centers.append((pix, piy))
            continue

        facets.append(poly)
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        centers.append((cx, cy))

    return facets, centers


def _poly_convex_hull_ccw(pts: List[Tuple[float, float]], eps: float = 1e-6) -> List[Tuple[float, float]]:
    """凸包（CCW）を返す。Voronoiセルは本来凸なので、頂点順の乱れや自己交差の回避に使う。"""
    pts = _poly_sanitize(pts, eps=eps)
    if len(pts) <= 2:
        return pts

    # 点を一意化してソート（x, y）
    uniq = sorted(set((float(x), float(y)) for (x, y) in pts))
    if len(uniq) <= 2:
        return [(uniq[0][0], uniq[0][1]), (uniq[-1][0], uniq[-1][1])]

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in uniq:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= eps:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(uniq):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= eps:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    # 念のため CCW ソート
    hull2 = _poly_sort_ccw([(x, y) for (x, y) in hull])
    return hull2


def _poly_reduce_to_max_vertices(pts: List[Tuple[float, float]], max_vertices: int, min_vertices: int) -> List[Tuple[float, float]]:
    """角数が多すぎる多角形を、形の崩れが小さい頂点（=内角が大きい=ほぼ一直線）から間引いて上限へ寄せる。"""
    if not pts or len(pts) < 3:
        return pts
    min_vertices = int(max(3, min_vertices))
    max_vertices = int(max_vertices)
    if max_vertices <= 0:
        return pts
    # 矛盾回避
    if max_vertices < min_vertices:
        max_vertices = min_vertices
    out = list(pts)
    for _ in range(2048):
        if len(out) <= max_vertices:
            break
        if len(out) <= min_vertices:
            break
        angs = _poly_angles_deg(out)
        if not angs:
            break
        idx = angs.index(max(angs))
        try:
            out.pop(idx)
        except Exception as e:
            _kana_silent_exc('core:L13776', e)
            break
    return out


def _stained_glass_facets_warpgrid(n: int, x0: float, y0: float, w0: float, h0: float,
                                  rng: "random.Random", jitter: float):
    """Voronoi が不安定な環境向けの安全フォールバック（WarpGrid）。

    - (nx+1)x(ny+1) の格子点を少しだけ揺らして、セル（四角形）でタイル分割します。
    - 完全なタイル分割になるので、欠け/空セルが起こりにくいのが利点です。
    """
    n = int(max(0, n))
    if n <= 0:
        return [], []

    aspect = (float(w0) / max(1.0, float(h0)))
    nx = int(round(math.sqrt(max(1.0, n) * aspect)))
    nx = max(1, nx)
    ny = int(math.ceil(n / nx))
    ny = max(1, ny)

    dx = float(w0) / nx
    dy = float(h0) / ny

    jitter = float(max(0.0, min(1.0, jitter)))
    amp = min(dx, dy) * 0.35 * jitter

    nodes = []
    for iy in range(ny + 1):
        row = []
        for ix in range(nx + 1):
            bx = float(x0 + ix * dx)
            by = float(y0 + iy * dy)
            if (0 < ix < nx) and (0 < iy < ny) and amp > 0.0:
                jx = (rng.random() - 0.5) * 2.0 * amp
                jy = (rng.random() - 0.5) * 2.0 * amp
                bx = bx + jx
                by = by + jy
            bx = float(min(max(bx, x0), x0 + w0))
            by = float(min(max(by, y0), y0 + h0))
            row.append((bx, by))
        nodes.append(row)

    facets = []
    centers = []
    for iy in range(ny):
        for ix in range(nx):
            p00 = nodes[iy][ix]
            p10 = nodes[iy][ix + 1]
            p11 = nodes[iy + 1][ix + 1]
            p01 = nodes[iy + 1][ix]
            poly = [p00, p10, p11, p01]
            facets.append(poly)
            cx = (p00[0] + p10[0] + p11[0] + p01[0]) / 4.0
            cy = (p00[1] + p10[1] + p11[1] + p01[1]) / 4.0
            centers.append((cx, cy))

    return facets, centers



# =============================================================================
# セクション: レイアウト: StainedGlass (Voronoi)
# - レイアウト生成の本体（配置/合成/マスクなど）
# =============================================================================

def layout_stained_glass_voronoi(paths: List[Path], width: int, height: int, margin: int, gutter: int,
                                 bg_rgb: Tuple[int, int, int], *, rng: Optional[random.Random] = None):
    """ステンドグラス（Voronoi）: 多角形パネルに画像を貼る。

    実装の方針:
    - OpenCV の Subdiv2D で Voronoi facet を作る
    - facet の外接矩形に画像を fill（cover）で貼り、facet 形状マスクで切り抜く
    - 最後に鉛線（境界線）を重ねる
    """
    if rng is None:
        rng = random.Random(0)

    # パフォーマンス計測（StainedGlass）
    SG_PERF["yolo_sec"] = 0.0
    SG_PERF["yolo_calls"] = 0

    n = int(len(paths) or 0)
    banner(_lang("処理中: StainedGlass", "Rendering: StainedGlass"))
    canvas = Image.new("RGB", (int(width), int(height)), tuple(map(int, bg_rgb)))
    mask = Image.new("L", (int(width), int(height)), 0)

    # Lead（境界線）設定（先に確定して、マスクの“重なり”補正にも使う）
    try:
        lead_w = int(STAINED_GLASS_LEAD_WIDTH)
    except Exception:
        lead_w = 6
    lead_w = int(max(0, lead_w))
    try:
        lead_rgb = tuple(int(x) for x in STAINED_GLASS_LEAD_RGB)  # type: ignore
        if len(lead_rgb) != 3:
            lead_rgb = (0, 0, 0)
    except Exception:
        lead_rgb = (0, 0, 0)

    # 文字列指定があれば優先（例: "#000000"）
    try:
        _lead_color = globals().get("STAINED_GLASS_LEAD_COLOR", None)
        if isinstance(_lead_color, str) and _lead_color.strip():
            lead_rgb = tuple(parse_color(_lead_color))
    except Exception as e:
        _kana_silent_exc('core:L13875', e)
        pass
    try:
        lead_a = float(STAINED_GLASS_LEAD_ALPHA)
    except Exception:
        lead_a = 0.85
    lead_a = float(max(0.0, min(1.0, lead_a)))

    # lead の描画スタイル
    #   - outer : 外側にだけ線を引く（線が太く見えにくい／内側に黒縁が食い込みにくい）
    #   - center: 内外にまたがる線（従来の太めの境界）
    try:
        lead_style = str(STAINED_GLASS_LEAD_STYLE).strip().lower()
    except Exception:
        lead_style = "outer"
    if lead_style not in ("outer", "center"):
        lead_style = "outer"


    # Lead 描画は“線”ではなく“各パネルの縁マスク”を合成する（線の継ぎ目の抜けを抑える）
    lead_overlay = None
    lead_rgba = None
    if lead_w > 0 and lead_a > 0.0:
        lead_overlay = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
        lead_rgba = (int(lead_rgb[0]), int(lead_rgb[1]), int(lead_rgb[2]), int(255 * lead_a))

    if n <= 0:
        return canvas, mask, {"Layout": "StainedGlass(Voronoi)", "tiles": 0}

    # 描画領域（外側 margin を避ける）
    x0 = int(max(0, margin))
    y0 = int(max(0, margin))
    w0 = int(max(1, width - margin * 2))
    h0 = int(max(1, height - margin * 2))
    # 念のため、はみ出しを避ける
    w0 = min(w0, int(width) - x0)
    h0 = min(h0, int(height) - y0)

    # seed 点（ランダム + エッジ優遇）
    # 目的:
    #  - Voronoi は seed の配置に強く依存。格子だと“gridっぽい”単調な窓割りになりやすい
    #  - 一方で完全ランダムだと「端のピースが巨大化」しやすいので、端にも点を置きやすくする
    area = float(w0) * float(h0)
    cell = math.sqrt(max(1.0, area / max(1.0, float(n))))
    edge_band = max(6.0, cell * 0.45)

    # jitter(0..1): 大きいほど近接を許して“荒さ”を出す（=窓割りのバリエーションが増える）
    #   ※起点は STAINED_GLASS_POINT_JITTER（外部設定/グローバル）で調整
    try:
        jitter = float(globals().get("STAINED_GLASS_POINT_JITTER", 0.55))
    except Exception:
        jitter = 0.55

    _jit = float(max(0.0, min(1.0, float(jitter))))
    # internal constraints (read once per run)
    #   - min_vertices: 3（三角形まで許容）
    #   - max_vertices: 無制限（0 で無効）
    _minv_cfg = 3
    _maxv_cfg = 0
    try:
        _maxang_cfg = float(globals().get("STAINED_GLASS_MAX_CORNER_ANGLE_DEG", 179.5))
    except Exception:
        _maxang_cfg = 179.5
    # one-line info for debug
    try:
        note(_lang(f"stained-glass cfg: maxang={_maxang_cfg:.1f}° | jitter={_jit:.2f}",
                  f"stained-glass cfg: maxang={_maxang_cfg:.1f}° | jitter={_jit:.2f}"))
    except Exception as e:
        _kana_silent_exc('core:L13949', e)
        pass
    min_dist = cell * (0.50 - 0.25 * _jit)  # 0.25..0.50 * cell
    min_dist = max(2.0, float(min_dist))
    min_dist2 = float(min_dist) * float(min_dist)

    pts = []
    seen = set()

    def _clamp_pt(px, py):
        px = float(min(max(px, x0 + 1), x0 + w0 - 2))
        py = float(min(max(py, y0 + 1), y0 + h0 - 2))
        return px, py

    def _add_pt(px, py, check_dist=True):
        px, py = _clamp_pt(px, py)
        key = (int(round(px)), int(round(py)))
        if key in seen:
            return False
        if check_dist and pts:
            for qx, qy in pts:
                dx_ = qx - px
                dy_ = qy - py
                if (dx_ * dx_ + dy_ * dy_) < min_dist2:
                    return False
        seen.add(key)
        pts.append((float(key[0]), float(key[1])))
        return True

    # 四隅アンカー（端の巨大ピース化を防ぐ）
    corner_margin = max(8.0, cell * 0.30)
    jcorner = corner_margin * (0.10 + 0.40 * _jit)
    corners = [
        (x0 + corner_margin, y0 + corner_margin),
        (x0 + w0 - corner_margin, y0 + corner_margin),
        (x0 + w0 - corner_margin, y0 + h0 - corner_margin),
        (x0 + corner_margin, y0 + h0 - corner_margin),
    ]
    for cx, cy in corners:
        _add_pt(
            cx + (rng.random() - 0.5) * jcorner,
            cy + (rng.random() - 0.5) * jcorner,
            check_dist=False,
        )

    # 残りをサンプリング（内側 + エッジ帯を混ぜる）
    edge_prob = 0.30 + 0.10 * (1.0 - _jit)  # jitter 小さいと端を厚めに
    tries = 0
    max_tries = int(max(4000, n * 1200))
    while len(pts) < n and tries < max_tries:
        tries += 1

        if rng.random() < edge_prob:
            side = rng.randint(0, 3)
            if side == 0:  # left
                px = x0 + 1 + rng.random() * edge_band
                py = y0 + 1 + rng.random() * max(1.0, (h0 - 3))
            elif side == 1:  # right
                px = x0 + w0 - 2 - rng.random() * edge_band
                py = y0 + 1 + rng.random() * max(1.0, (h0 - 3))
            elif side == 2:  # top
                px = x0 + 1 + rng.random() * max(1.0, (w0 - 3))
                py = y0 + 1 + rng.random() * edge_band
            else:  # bottom
                px = x0 + 1 + rng.random() * max(1.0, (w0 - 3))
                py = y0 + h0 - 2 - rng.random() * edge_band
        else:
            px = x0 + 1 + rng.random() * max(1.0, (w0 - 3))
            py = y0 + 1 + rng.random() * max(1.0, (h0 - 3))

        # ちょい足しの揺らし（cell 比）
        j = cell * 0.20 * _jit
        if j > 0:
            px += (rng.random() - 0.5) * j
            py += (rng.random() - 0.5) * j

        if _add_pt(px, py, check_dist=True):
            continue

        # jitter が大きいときは、近接でも少しだけ許す（窓割りの“粗さ”を増やす）
        if _jit >= 0.70 and rng.random() < 0.18:
            _add_pt(px, py, check_dist=False)

    # どうしても不足したら、軽い jittered-grid で補充（最終安全網）
    if len(pts) < n:
        aspect = (float(w0) / max(1.0, float(h0)))
        nx = int(round(math.sqrt(max(1.0, float(n)) * aspect)))
        nx = max(1, nx)
        ny = int(math.ceil(float(n) / float(nx)))
        ny = max(1, ny)
        dx = float(w0) / float(nx)
        dy = float(h0) / float(ny)

        for iy in range(ny):
            for ix in range(nx):
                if len(pts) >= n:
                    break
                base_x = ix + 0.5
                base_y = iy + 0.5
                jx = (rng.random() - 0.5) * 0.85
                jy = (rng.random() - 0.5) * 0.85
                px = x0 + (base_x + jx) * dx
                py = y0 + (base_y + jy) * dy
                _add_pt(px, py, check_dist=False)
            if len(pts) >= n:
                break

    # 念のため float 化（cv2 が int を嫌う環境がある）
    pts = [(float(x), float(y)) for x, y in pts]
    # Voronoi の点を正規化しておく（後段での数値誤差を抑える）
    facets = None
    centers = None
    _used_backend = None
    _vor_backend = str(globals().get("STAINED_GLASS_VORONOI_BACKEND", "auto")).strip().lower()

    if _vor_backend in ("auto", "opencv"):
        try:
            import cv2  # type: ignore
            rect = (int(x0), int(y0), int(w0), int(h0))
            subdiv = cv2.Subdiv2D(rect)
            for px, py in pts:
                subdiv.insert((float(px), float(py)))
            facets, centers = subdiv.getVoronoiFacetList([])
            _used_backend = "opencv"
        except Exception:
            facets, centers = None, None

    # OpenCV が失敗/空の場合は Python 実装（依存なし）へ（まずはこちらを試す）

    if not facets:
        try:
            facets, centers = _voronoi_facets_python(pts, (float(x0), float(y0), float(x0 + w0), float(y0 + h0)))
            _used_backend = "python"
        except Exception:
            facets, centers = None, None

    # それでも失敗した場合は最終フォールバック: WarpGrid（依存なしで安定）
    if not facets:
        try:
            try:
                note(_lang("[WARN] stained-glass: Voronoi failed -> WarpGrid fallback", "[WARN] stained-glass: Voronoi failed -> WarpGrid fallback"))
            except Exception as e:
                _kana_silent_exc('core:L14090', e)
                pass
            facets, centers = _stained_glass_facets_warpgrid(n, float(x0), float(y0), float(w0), float(h0), rng, jitter)
            _used_backend = "warpgrid"
        except Exception:
            facets, centers = None, None

    # facet 整形
    def _poly_sort_ccw(pts):
        """重心まわりの角度で CCW ソート。頂点順の崩れによるマスク欠け（黒い線）を抑える。"""
        if len(pts) <= 2:
            return pts
        cx = sum(p[0] for p in pts) / float(len(pts))
        cy = sum(p[1] for p in pts) / float(len(pts))
        pts2 = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        # 連続重複を落とす（極小の丸めで同一点になることがある）
        out = []
        for p in pts2:
            if not out:
                out.append(p)
                continue
            if abs(p[0] - out[-1][0]) < 1e-6 and abs(p[1] - out[-1][1]) < 1e-6:
                continue
            out.append(p)
        if len(out) >= 3 and abs(out[0][0] - out[-1][0]) < 1e-6 and abs(out[0][1] - out[-1][1]) < 1e-6:
            out.pop()
        return out

    items = []


    # ピースの縫い目が“抜け”やすいので、境界線がある場合はマスクを少し膨張させて重ねます。
    # （最後に lead を描くので、重なりは基本的に見えません）
    # seam-guard:
    #  - Voronoi の頂点は float → int 量子化されるため、稀に 1px の“抜け”が出ます
    #  - 各パネルのマスクを少し膨張させて重ね、隙間を作りにくくします（Lead があれば線で隠れます）
    overlap_px = 2
    # だいたい “線の太さ” に見合うくらいの重ね
    if lead_style == "outer":
        overlap_px = int(min(4, max(1, math.ceil(lead_w * 0.50))))
    else:
        overlap_px = int(min(6, max(2, math.ceil(lead_w * 0.75))))

    def _build_items(_facets, _centers):
        _items = []
        for i, f in enumerate(_facets):
            try:
                poly = [(float(p[0]), float(p[1])) for p in f]
            except Exception as e:
                _kana_silent_exc('core:L14139', e)
                continue
            if len(poly) < 3:
                continue

            # 端で頂点を「押し込む(clamp)」と形が壊れて角度制約が破綻しやすいので、
            # 矩形クリップ（交点を作って切り取る）で正しく切り取ります。
            # さらに、facet の頂点順が崩れている場合に備えて凸包で順序を安定化します。
            x1 = float(x0 + w0 - 1e-3)
            y1 = float(y0 + h0 - 1e-3)

            clamped = _poly_convex_hull_ccw(poly)
            if len(clamped) < 3:
                continue
            clamped = _clip_poly_rect(clamped, float(x0), float(y0), x1, y1)
            if len(clamped) < 3:
                continue
                # 角数/角度の制約（ステンドグラス向け）
            minv = _minv_cfg
            maxang = _maxang_cfg

            # できるだけ制約を守るように整形（不足時は辺分割、直線角は間引き/軽微な内側曲げ）
            clamped = _poly_enforce_constraints(
                clamped,
                min_vertices=minv,
                max_vertices=0,
                max_corner_angle_deg=maxang,
                bbox=(x0, y0, x0 + w0, y0 + h0),
                lead_width=lead_w,
            )


            # enforce 後に bbox 外へ微小にはみ出すことがあるため、もう一度クリップして安全側へ寄せます。
            clamped = _clip_poly_rect(clamped, float(x0), float(y0), x1, y1)
            # 重要: clip は交点を追加して“頂点数が増える”ことがあります（特に画面端）。
            #       max_vertices / 角度制約を最終形に反映するため、ここで制約を再適用します。
            clamped = _poly_enforce_constraints(
                clamped,
                min_vertices=minv,
                max_vertices=0,
                max_corner_angle_deg=maxang,
                bbox=(x0, y0, x1, y1),
                lead_width=lead_w,
            )
            if len(clamped) < minv:
                continue

            # クリップは順序を保持するので並べ替えは不要。向きだけ CCW に揃えます。
            try:
                if _poly_area_xy(clamped) < 0.0:
                    clamped = list(reversed(clamped))
            except Exception as e:
                _kana_silent_exc('core:L14191', e)
                pass
            if len(clamped) < minv:
                continue

            # 退化防止
            area = abs(_poly_area_xy(clamped))
            if area <= 1.0:
                continue
            try:
                cx, cy = _centers[i]
                cx = float(min(max(float(cx), x0), x0 + w0))
                cy = float(min(max(float(cy), y0), y0 + h0))
            except Exception:
                # 重心近似
                cx = sum(p[0] for p in clamped) / len(clamped)
                cy = sum(p[1] for p in clamped) / len(clamped)
            _items.append({"poly": clamped, "center": (cx, cy), "area": area})
        return _items

    items = _build_items(facets, centers)

    # OpenCV Voronoi は「一部 facet が空」になりやすく、黒フレーム（未充填領域）になり得る。
    # items が不足したら Python Voronoi を優先して再試行する。
    if len(items) < n:
        try:
            facets_py, centers_py = _voronoi_facets_python(pts, bbox)
            items_py = _build_items(facets_py, centers_py)
            # 置き換え条件: より多く確保できた（できれば n 個）
            if len(items_py) > len(items):
                facets, centers, items = facets_py, centers_py, items_py
                _used_backend = "python"
        except Exception as e:
            _kana_silent_exc('core:L14223', e)
            pass
    # ─────────────────────────────────────────────────────────────
    # StainedGlass: Global mesh simplification (experimental)
    #  - セル単体の硬い角数制約で隙間（黒い抜け）が出る問題に対して、
    #    メッシュ全体で頂点統合（edge collapse）を行い、タイル整合性を保ちつつ角数を減らす。
    # ─────────────────────────────────────────────────────────────
    try:
        _sg_global = bool(globals().get("STAINED_GLASS_GLOBAL_MESH_SIMPLIFY", True))
    except Exception:
        _sg_global = True

    if _sg_global and (_used_backend in ("opencv", "python")):
        try:
            if _maxv_cfg > 0 and len(items) >= 3:
                _bbox_sg = (float(x0), float(y0), float(x0 + w0), float(y0 + h0))
                _polys = [it.get("poly", []) for it in items]
                _polys2 = _sg_global_mesh_simplify_polys(
                    _polys,
                    bbox=_bbox_sg,
                    min_vertices=_minv_cfg,
                    max_vertices=_maxv_cfg,
                    lead_width=lead_w,
                )
                if _polys2 and len(_polys2) == len(items):
                    for it, p2 in zip(items, _polys2):
                        if p2 and len(p2) >= 3:
                            it["poly"] = p2
        except Exception as e:
            _kana_silent_exc("core:SG_GLOBAL_SIMPLIFY", e)
            pass
    if len(items) < n:
        # Voronoi の退化で items が空になる環境があるため、WarpGrid / RectGrid で継続します。
        # 重要: ここで grid レイアウトに落とすと anneal 等が走ってしまうため、
        #       stained-glass の“タイル”として最後まで描画します。
        try:
            try:
                note(_lang(f"[WARN] stained-glass: Voronoi incomplete ({len(items)}/{n}) -> WarpGrid fallback", f"[WARN] stained-glass: Voronoi incomplete ({len(items)}/{n}) -> WarpGrid fallback"))
            except Exception as e:
                _kana_silent_exc('core:L14233', e)
                pass
            # 1) WarpGrid 生成（必ず矩形セルでタイル分割できる）
            try:
                facets, centers = _stained_glass_facets_warpgrid(
                    n, float(x0), float(y0), float(w0), float(h0), rng, jitter
                )
                _used_backend = "warpgrid"
            except Exception:
                facets, centers = [], []
                _used_backend = "warpgrid_err"

            # WarpGrid は“矩形セル”なので、制約処理を通さず超安全に items 化します。
            _items2 = []
            for i, f in enumerate(facets or []):
                try:
                    poly = [(float(p[0]), float(p[1])) for p in f]
                except Exception as e:
                    _kana_silent_exc('core:L14251', e)
                    continue
                if len(poly) < 3:
                    continue
                clamped = []
                for xx, yy in poly:
                    xx = float(min(max(xx, x0), x0 + w0))
                    yy = float(min(max(yy, y0), y0 + h0))
                    clamped.append((xx, yy))
                try:
                    clamped = _poly_sort_ccw(clamped)
                except Exception as e:
                    _kana_silent_exc('core:L14262', e)
                    pass
                area = abs(_poly_area_xy(clamped))
                if area <= 1.0:
                    continue
                try:
                    cx, cy = centers[i]
                    cx = float(min(max(float(cx), x0), x0 + w0))
                    cy = float(min(max(float(cy), y0), y0 + h0))
                except Exception:
                    cx = sum(p[0] for p in clamped) / len(clamped)
                    cy = sum(p[1] for p in clamped) / len(clamped)
                _items2.append({"poly": clamped, "center": (cx, cy), "area": area})

            items = _items2

            # 2) それでも空なら、RectGrid（揺らし無しの確実矩形）
            if not items:
                try:
                    try:
                        note(_lang("[WARN] stained-glass: WarpGrid empty -> RectGrid fallback", "[WARN] stained-glass: WarpGrid empty -> RectGrid fallback"))
                    except Exception as e:
                        _kana_silent_exc('core:L14283', e)
                        pass
                    aspect = (float(w0) / max(1.0, float(h0)))
                    nx = int(round(math.sqrt(max(1.0, float(n)) * aspect)))
                    nx = max(1, nx)
                    ny = int(math.ceil(float(n) / float(nx)))
                    ny = max(1, ny)
                    dx = float(w0) / float(nx)
                    dy = float(h0) / float(ny)
                    _items3 = []
                    k = 0
                    for iy in range(ny):
                        for ix in range(nx):
                            if k >= n:
                                break
                            xa = float(x0 + ix * dx)
                            xb = float(x0 + (ix + 1) * dx)
                            ya = float(y0 + iy * dy)
                            yb = float(y0 + (iy + 1) * dy)
                            poly = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
                            try:
                                poly = _poly_sort_ccw(poly)
                            except Exception as e:
                                _kana_silent_exc('core:L14305', e)
                                pass
                            cx = (xa + xb) * 0.5
                            cy = (ya + yb) * 0.5
                            area = abs(_poly_area_xy(poly))
                            _items3.append({"poly": poly, "center": (cx, cy), "area": area})
                            k += 1
                        if k >= n:
                            break
                    items = _items3
                    _used_backend = "rectgrid"
                except Exception as e:
                    _kana_silent_exc('core:L14316', e)
                    pass
        except Exception:
            # ここで例外を落とすと最悪何も描けないので、items は空のまま先へ（下で救済）
            pass

    if not items:
        # 本当に最後の救済: 1 枚だけ全画面パネルとして扱う（grid には落とさない）
        items = [{
            "poly": [(float(x0), float(y0)), (float(x0 + w0), float(y0)), (float(x0 + w0), float(y0 + h0)), (float(x0), float(y0 + h0))],
            "center": (float(x0 + w0 * 0.5), float(y0 + h0 * 0.5)),
            "area": abs(_poly_area_xy([(float(x0), float(y0)), (float(x0 + w0), float(y0)), (float(x0 + w0), float(y0 + h0)), (float(x0), float(y0 + h0))]))
        }]
        _used_backend = "single_panel"
    # 並び順（画像割当の順序）
    order = str(STAINED_GLASS_ORDER).strip().lower()
    if order == "hilbert":
        bits = int(max(4, min(10, int(STAINED_GLASS_HILBERT_BITS))))
        ngrid = (1 << bits) - 1
        def _key(it):
            cx, cy = it["center"]
            xi = int(round(((cx - x0) / max(1.0, w0)) * ngrid))
            yi = int(round(((cy - y0) / max(1.0, h0)) * ngrid))
            xi = max(0, min(ngrid, xi))
            yi = max(0, min(ngrid, yi))
            return _hilbert_index(xi, yi, order=bits)
        items.sort(key=_key)
    elif order == "scan":
        items.sort(key=lambda it: (it["center"][1], it["center"][0]))
    elif order == "diag":
        items.sort(key=lambda it: (it["center"][0] + it["center"][1]))
    elif order == "spiral":
        cx0 = x0 + w0 * 0.5
        cy0 = y0 + h0 * 0.5
        def _key(it):
            cx, cy = it["center"]
            ang = math.atan2(cy - cy0, cx - cx0)
            rad = (cx - cx0) ** 2 + (cy - cy0) ** 2
            return (ang, rad)
        items.sort(key=_key)
    else:
        rng.shuffle(items)
    use_ff = bool(globals().get("STAINED_GLASS_FACE_FOCUS_ENABLE", True)) and bool(globals().get("FACE_FOCUS_ENABLE", True))

    # facet 数が画像より多い場合は切り詰め、少ない場合は面積大きい順で補う
    if len(items) > n:
        items = items[:n]
    elif len(items) < n:
        # 面積の大きい順で複製（見た目は悪化するが 0枚よりマシ）
        items.sort(key=lambda it: it["area"], reverse=True)
        _base = int(len(items))
        if _base <= 0:
            # 万一 items が空なら、1枚パネルで救済（ここで落とさない）
            _poly0 = [(float(x0), float(y0)), (float(x0 + w0), float(y0)), (float(x0 + w0), float(y0 + h0)), (float(x0), float(y0 + h0))]
            items = [{
                "poly": _poly0,
                "center": (float(x0 + w0 * 0.5), float(y0 + h0 * 0.5)),
                "area": abs(_poly_area_xy(_poly0)),
            }]
            _base = 1
        while len(items) < n:
            items.append(items[len(items) % _base])

    use_priority = bool(globals().get("STAINED_GLASS_FACE_PRIORITY_ENABLE", True)) and use_ff
    use_facefit = bool(globals().get("STAINED_GLASS_FACE_FIT_ENABLE", True)) and use_ff

    try:
        _max_tries = int(globals().get("STAINED_GLASS_FACE_FIT_MAX_TRIES", 10))
    except Exception:
        _max_tries = 10
    _max_tries = int(max(1, _max_tries))

    try:
        _min_short = int(globals().get("STAINED_GLASS_FACE_FIT_MIN_SHORT_SIDE", 96))
    except Exception:
        _min_short = 96
    _min_short = int(max(8, _min_short))

    try:
        _min_fill = float(globals().get("STAINED_GLASS_FACE_FIT_MIN_FILL_RATIO", 0.45))
    except Exception:
        _min_fill = 0.45
    _min_fill = float(max(0.0, min(1.0, _min_fill)))

    # 安全マージン: None の場合は lead_w から自動推定
    # ※ステンドグラスは多角形マスクで「目」だけ欠けやすいので、既定は少し強めに寄せる
    _sm = globals().get("STAINED_GLASS_FACE_FIT_SAFE_MARGIN_PX", None)
    try:
        if _sm is None:
            # lead_w による自動推定（既定を強めに）
            _safe_margin = int(max(8, round(float(lead_w) * 1.05)))
        else:
            _safe_margin = int(_sm)
    except Exception:
        _safe_margin = int(max(8, round(float(lead_w) * 1.05)))
    _safe_margin = int(max(0, _safe_margin))

    try:
        _eye_y_frac = float(globals().get("STAINED_GLASS_FACE_FIT_EYE_Y_FRAC", 0.30))
    except Exception:
        _eye_y_frac = 0.35
    _eye_y_frac = float(max(0.05, min(0.80, _eye_y_frac)))

    try:
        _eye_dx_frac = float(globals().get("STAINED_GLASS_FACE_FIT_EYE_SPREAD_X_FRAC", 0.08))
    except Exception:
        _eye_dx_frac = 0.06
    _eye_dx_frac = float(max(0.0, min(0.30, _eye_dx_frac)))

    try:
        _eye_dy_frac = float(globals().get("STAINED_GLASS_FACE_FIT_EYE_SPREAD_Y_FRAC", 0.05))
    except Exception:
        _eye_dy_frac = 0.04
    _eye_dy_frac = float(max(0.0, min(0.30, _eye_dy_frac)))

    try:
        _eye_spread_scale = float(globals().get("STAINED_GLASS_FACE_FIT_EYE_SPREAD_SCALE", 1.0))
    except Exception:
        _eye_spread_scale = 1.0
    _eye_spread_scale = float(max(0.50, min(2.50, _eye_spread_scale)))


    try:
        _pt_ok_ratio = float(globals().get("STAINED_GLASS_FACE_FIT_POINT_OK_RATIO", globals().get("STAINED_GLASS_FACE_FIT_OK_RATIO", 0.85)))
    except Exception:
        _pt_ok_ratio = 0.85
    _pt_ok_ratio = float(max(0.05, min(1.0, _pt_ok_ratio)))

    # 点サンプルの密度:
    #  - "lite" : 目の中心 + 十字（速い／過剰な弾きが減る）
    #  - "full" : 目の中心 + 十字 + 斜め（厳しめ／欠けを減らすが遅くなりがち）
    try:
        _pts_mode = str(globals().get("STAINED_GLASS_FACE_FIT_POINTS_MODE", "lite")).strip().lower()
    except Exception:
        _pts_mode = "lite"
    if _pts_mode not in ("lite", "full"):
        _pts_mode = "lite"

    # 1パネル内で目チェック失敗が多い場合の“早めの妥協”（過剰ループ抑制）
    # 0 なら無効
    try:
        _failfast_eye = int(globals().get("STAINED_GLASS_FACE_FIT_FAILFAST_FAIL_EYE", 4))
    except Exception:
        _failfast_eye = 4
    _failfast_eye = int(max(0, min(999, _failfast_eye)))


    try:
        _strict_band = bool(globals().get("STAINED_GLASS_FACE_FIT_STRICT_EYE_BAND", False))
    except Exception:
        _strict_band = False

    # strict では “目の安全域” を少し強める（検出誤差・ワープ誤差の吸収）
    if _strict_band:
        try:
            _eye_spread_scale = float(max(_eye_spread_scale, 1.35))
        except Exception:
            _eye_spread_scale = 1.35

    # face-fit の有効状態を 1 行だけ表示（必要なら STAINED_GLASS_FACE_FIT_STATUS_LOG=False で抑止）
    if use_ff and bool(globals().get("STAINED_GLASS_FACE_FIT_STATUS_LOG", True)):
        try:
            print(
                f"[StainedGlass] face_focus=ON facefit={'ON' if use_facefit else 'OFF'} strict={'ON' if _strict_band else 'OFF'} "
                f"tries={_max_tries} safe_margin={_safe_margin}px "
                f"eye_y={_eye_y_frac:.2f} spread=({ _eye_dx_frac:.2f},{ _eye_dy_frac:.2f})x{_eye_spread_scale:.2f} pts={_pts_mode} ff_eye={_failfast_eye} "
                f"ok_ratio={_pt_ok_ratio:.2f} thin=(min_short={_min_short}, min_fill={_min_fill:.2f})"
            )
        except Exception as e:
            _kana_silent_exc('core:L14436', e)
            pass
    # face-cache（dhash cache）で顔bboxを引けるなら、パネル内に「目」が入っているか判定してリトライ
    from collections import deque


    # 画像サイズ（w,h）の軽量キャッシュ（ヘッダ読みのみ）
    _sg_wh_cache = {}  # key: norm_path -> (w,h)

    def _sg_get_wh(pp):
        try:
            k = _dhash_norm_path(pp)
        except Exception:
            k = str(pp)
        wh = _sg_wh_cache.get(k, None)
        if isinstance(wh, tuple) and len(wh) == 2:
            return wh
        try:
            _im = open_image_safe(pp)  # 画像キー（zip:// 等）も可
            try:
                wh = (int(_im.size[0]), int(_im.size[1]))
            finally:
                try:
                    _im.close()
                except Exception:
                    pass
        except Exception:
            wh = (0, 0)
        _sg_wh_cache[k] = wh
        return wh

    # stained-glass 用: 顔bbox（元画像座標）のインメモリキャッシュ
    # - FACE_CACHE_DISABLE_AI=True（既定）でも、同一実行内で同じ画像を何度もAI解析しないためのもの
    _SG_NOFACE = object()
    _sg_face_raw_cache = {}  # key: norm_path -> (name,x,y,w,h) or _SG_NOFACE

    def _sg_get_face_raw(pp):
        """顔bbox（元画像座標）を返す。無ければ None。"""
        try:
            k = _dhash_norm_path(pp)
        except Exception:
            k = str(pp)

        v = _sg_face_raw_cache.get(k, None)
        if v is _SG_NOFACE:
            return None
        if isinstance(v, (list, tuple)) and len(v) == 5:
            return v

        # 永続キャッシュ（AI結果）を使うのは、明示的に許可された場合のみ
        # ※FACE_CACHE_DISABLE_AI=True のときは読み出しもしない（精度重視）
        try:
            if not bool(globals().get('FACE_CACHE_DISABLE_AI', True)):
                ent = _face_cache_get(pp)
                face = ent.get('face', None) if isinstance(ent, dict) else None
                if isinstance(face, (list, tuple)) and len(face) == 5:
                    _sg_face_raw_cache[k] = face
                    return face
        except Exception:
            pass

        def _detect_face_on_draft(_maxdim: int):
            try:
                ow, oh = _sg_get_wh(pp)
                if not (isinstance(ow, int) and isinstance(oh, int) and ow > 0 and oh > 0):
                    return None
                _maxdim = int(max(256, min(2048, int(_maxdim))))
                _im = open_image_safe(pp, draft_to=(_maxdim, _maxdim))
                if _im is None:
                    return None
                _im = _im.convert('RGB')
                w2, h2 = _im.size
                if (w2 > _maxdim) or (h2 > _maxdim):
                    _im.thumbnail((_maxdim, _maxdim), resample=Image.Resampling.LANCZOS)

                # src_path=None にして永続キャッシュ系の副作用を避ける
                cand = _get_focus_candidates(_im, None)
                f2 = cand.get('face', None) if isinstance(cand, dict) else None
                if not (isinstance(f2, (list, tuple)) and len(f2) == 5):
                    return None

                name, fx, fy, fw, fh = f2
                # draft_to/thumbnail した座標 → 元画像座標へ
                sx = float(ow) / float(max(1, _im.size[0]))
                sy = float(oh) / float(max(1, _im.size[1]))
                return (name, float(fx) * sx, float(fy) * sy, float(fw) * sx, float(fh) * sy)
            except Exception:
                return None

        # オンデマンド解析（軽量化のため縮小して検出 → 元画像座標へスケール）
        try:
            maxdim = int(globals().get('STAINED_GLASS_FACE_DETECT_MAX_DIM', 512))
            face = _detect_face_on_draft(maxdim)

            # 1回目で取れなかった場合のみ、任意の2段目（高解像度）で再挑戦
            # 例: STAINED_GLASS_FACE_DETECT_MAX_DIM2=768
            try:
                maxdim2 = int(globals().get('STAINED_GLASS_FACE_DETECT_MAX_DIM2', 0))
            except Exception:
                maxdim2 = 0
            if face is None and isinstance(maxdim2, int) and maxdim2 > maxdim:
                face = _detect_face_on_draft(maxdim2)

            if isinstance(face, (list, tuple)) and len(face) == 5:
                _sg_face_raw_cache[k] = face
                try:
                    # AIキャッシュが許可されているなら、dHashキャッシュにも保存（任意）
                    if not bool(globals().get('FACE_CACHE_DISABLE_AI', True)):
                        _face_cache_put(pp, face, None, None, None)
                except Exception:
                    pass
                return face

            _sg_face_raw_cache[k] = _SG_NOFACE
            return None
        except Exception:
            _sg_face_raw_cache[k] = _SG_NOFACE
            return None

    def _sg_face_box_in_tile(pp, cw, ch):

        # 返り値: (x0,y0,x1,y1) in tile coords, or None
        try:
            face = _sg_get_face_raw(pp)
            if not (isinstance(face, (list, tuple)) and len(face) == 5):
                return None

            _, fx, fy, fw, fh = face
            ow, oh = _sg_get_wh(pp)
            if not (isinstance(ow, int) and isinstance(oh, int) and ow > 0 and oh > 0):
                return None

            x0 = int(round((ow - cw) * 0.5))
            y0 = int(round((oh - ch) * 0.5))

            # 顔中心に寄せる（ただし端には寄せすぎない）
            cx = float(fx) + float(fw) * 0.5
            cy = float(fy) + float(fh) * 0.5
            x0 = int(round(cx - cw * 0.5))
            y0 = int(round(cy - ch * 0.5))

            x0 = max(0, min(int(ow - cw), x0))
            y0 = max(0, min(int(oh - ch), y0))

            fx0 = float(fx)
            fy0 = float(fy)
            fx1 = float(fx) + float(fw)
            fy1 = float(fy) + float(fh)

            # face box in tile coords
            return (fx0 - float(x0), fy0 - float(y0), fx1 - float(x0), fy1 - float(y0))
        except Exception:
            return None

    def _sg_face_points(face_box, eye_y_frac=None, points_mode=None):
        # face_box: (x0,y0,x1,y1) in tile coords
        # eye_y_frac: 目の高さ（bbox比）。None の場合は既定（_eye_y_frac）を使う
        fx0, fy0, fx1, fy1 = face_box
        fw = max(1.0, float(fx1 - fx0))
        fh = max(1.0, float(fy1 - fy0))

        try:
            eyf = float(_eye_y_frac if eye_y_frac is None else eye_y_frac)
        except Exception:
            eyf = float(_eye_y_frac)
        # points_mode: None の場合は既定（_pts_mode）を使う
        try:
            pm = str(_pts_mode if points_mode is None else points_mode).strip().lower()
        except Exception:
            pm = "lite"
        if pm not in ("lite", "full"):
            pm = "lite"


        # 目の中心（左右）: bbox 比で推定
        ex1 = float(fx0) + fw * 0.33
        ex2 = float(fx0) + fw * 0.67
        ey = float(fy0) + fh * eyf

        # 目の“広がり”も点サンプルして、目が少しでも欠けるなら弾く
        dx = fw * float(_eye_dx_frac) * float(_eye_spread_scale)
        dy = fh * float(_eye_dy_frac) * float(_eye_spread_scale)

        cx = float(fx0) + fw * 0.50
        cy = float(fy0) + fh * 0.50

        pts = []

        # 左右の目（中心 + 十字 + 斜め）
        for ex in (ex1, ex2):
            pts.append((ex, ey))
            if dx > 0.0:
                pts.append((ex - dx, ey))
                pts.append((ex + dx, ey))
            if dy > 0.0:
                pts.append((ex, ey - dy))
                pts.append((ex, ey + dy))
            if pm == "full" and dx > 0.0 and dy > 0.0:
                pts.append((ex - dx, ey - dy))
                pts.append((ex + dx, ey - dy))
                pts.append((ex - dx, ey + dy))
                pts.append((ex + dx, ey + dy))

        # 顔中心も入れておく（極端な欠けを避ける）
        pts.append((cx, cy))
        return pts

    def _sg_mask_has(mimg, x, y):
        try:
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if xi < 0 or yi < 0 or xi >= int(mimg.size[0]) or yi >= int(mimg.size[1]):
                return False
            return int(mimg.getpixel((xi, yi))) >= 128
        except Exception:
            return False

    def _sg_face_ok(pmask0, face_box, points_mode=None, pt_ok_ratio=None):
        """face_box（目推定点）がパネル内に収まるかを判定します。

        - face_box が None（検出失敗/非顔）は “OK扱い” で進めます（過剰リトライ抑制）。
        - safe_margin が有効な場合は、MinFilter でパネル領域を縮めたマスクで判定します。
        - strict_eye_band が有効な場合は、目位置の上下ずらし（±alt）も同時に満たす必要があります。
        """
        # face_box が無い（非顔・検出失敗）は“OK扱い”で進める（過剰リトライを防ぐ）
        if face_box is None:
            return True

        mchk = pmask0
        if _safe_margin > 0:
            try:
                # 小さすぎるパネルで崩壊しないように上限を設ける
                # （既定 safe_margin は強めだが、極小パネルでは過剰に縮めない）
                mm = int(min(_safe_margin, max(0, (min(pmask0.size) // 8))))
                if mm > 0:
                    # MinFilter: 白領域（=パネル）が縮む → “安全域”の内側判定になる
                    mchk = pmask0.filter(ImageFilter.MinFilter(size=mm * 2 + 1))
            except Exception:
                mchk = pmask0

        def _check_pts(_pts):
            try:
                if not _pts:
                    return True
                ok = 0
                for (px, py) in _pts:
                    if _sg_mask_has(mchk, px, py):
                        ok += 1
                need = int(math.ceil(float(len(_pts)) * float(_pt_ok_ratio if pt_ok_ratio is None else pt_ok_ratio)))
                need = max(1, min(len(_pts), need))
                return ok >= need
            except Exception:
                return True

        # strict 判定: 目付近の“縦レンジ”（eye_y ± alt）も安全域に入っていることを要求
        try:
            _strict_band = bool(globals().get("STAINED_GLASS_FACE_FIT_STRICT_EYE_BAND", False))
        except Exception:
            _strict_band = False

        # 目位置の推定が画像によってズレることがあるため、少し上下にずらした判定も使います
        try:
            _alt = float(globals().get("STAINED_GLASS_FACE_FIT_EYE_Y_ALT_DELTA", 0.04))
        except Exception:
            _alt = 0.04
        _alt = float(max(0.0, min(0.20, _alt)))

        # まずは既定の目位置で判定
        if _check_pts(_sg_face_points(face_box, points_mode=points_mode)):
            if not _strict_band:
                return True
            if _alt <= 0.0:
                return True
            if not _check_pts(_sg_face_points(face_box, eye_y_frac=float(_eye_y_frac) - _alt, points_mode=points_mode)):
                return False
            if not _check_pts(_sg_face_points(face_box, eye_y_frac=float(_eye_y_frac) + _alt, points_mode=points_mode)):
                return False
            return True

        # strict band は“誤採用”を減らすために保守的に倒す（上下ずらしの救済は使わない）
        if _strict_band:
            return False

        # 非 strict: 上下にずらした判定も試す（救済）
        if _alt > 0.0:
            if _check_pts(_sg_face_points(face_box, eye_y_frac=float(_eye_y_frac) - _alt, points_mode=points_mode)):
                return True
            if _check_pts(_sg_face_points(face_box, eye_y_frac=float(_eye_y_frac) + _alt, points_mode=points_mode)):
                return True

        return False


    # -----------------------------------------------------------------------------
    # StainedGlass: 並び（グラデ / 散らし）
    # -----------------------------------------------------------------------------
    # - stained-glass でも他レイアウトと同様に「全体の色流れ（グラデ）」や「散らし（バラけ）」を作れるようにします。
    # - 画像順: 既存の “global reorder” を流用（spectral→Hilbert/Diagonal/anneal）
    # - パネル順: パネル中心座標で diag / hilbert / checker / random に並べ替え
    # - stained-glass でも他レイアウトと同様に「全体の色流れ（グラデ）」や「散らし（バラけ）」を作れるようにします。
    # - 画像順: 既存の “global reorder” を流用（spectral→Hilbert/Diagonal/anneal）
    # - パネル順: パネル中心座標で diag / hilbert / checker / random に並べ替え
    try:
        _ord_enable = bool(globals().get("STAINED_GLASS_ORDER_ENABLE", False))
    except Exception:
        _ord_enable = False
    try:
        _ord_mode = str(globals().get("STAINED_GLASS_ORDER_MODE", "grad")).strip().lower()
    except Exception:
        _ord_mode = "grad"
    if _ord_mode not in ("grad", "scatter", "random"):
        _ord_mode = "grad"

    try:
        _panel_order = str(globals().get("STAINED_GLASS_PANEL_ORDER", "auto")).strip().lower()
    except Exception:
        _panel_order = "auto"
    try:
        _img_order = str(globals().get("STAINED_GLASS_IMAGE_ORDER", "spectral_hilbert")).strip().lower()
    except Exception:
        _img_order = "spectral_hilbert"

    try:
        _obj = str(globals().get("STAINED_GLASS_ORDER_OBJECTIVE", "auto")).strip().lower()
    except Exception:
        _obj = "auto"
    if _obj == "auto":
        _obj = "min" if _ord_mode == "grad" else "max"
    if _obj not in ("min", "max"):
        _obj = "min"

    try:
        _hb = int(globals().get("STAINED_GLASS_ORDER_HILBERT_BITS", 6))
    except Exception:
        _hb = 6
    _hb = int(max(3, min(10, _hb)))

    # 対角方向（tl_br / tr_bl / bl_tr / br_tl）
    try:
        _diag_dir = str(
            globals().get(
                "STAINED_GLASS_DIAG_DIR",
                globals().get(
                    "STAINED_GLASS_DIAGONAL_DIRECTION",
                    globals().get(
                        "STAINED_GLASS_PANEL_DIAG_DIR",
                        globals().get("DIAG_DIR", "tl_br"),
                    ),
                ),
            )
        ).strip().lower()
    except Exception:
        _diag_dir = "tl_br"

    # anneal params for stained-glass image order (used only when STAINED_GLASS_IMAGE_ORDER == "anneal")
    try:
        _anneal_steps = int(
            globals().get(
                "STAINED_GLASS_ORDER_ANNEAL_STEPS",
                globals().get("STAINED_GLASS_ANNEAL_STEPS", 20000),
            )
        )
    except Exception:
        _anneal_steps = 20000
    _anneal_steps = int(max(1000, min(200000, _anneal_steps)))

    try:
        _anneal_reheats = int(
            globals().get(
                "STAINED_GLASS_ORDER_ANNEAL_REHEATS",
                globals().get("STAINED_GLASS_ANNEAL_REHEATS", 1),
            )
        )
    except Exception:
        _anneal_reheats = 1
    _anneal_reheats = int(max(1, min(10, _anneal_reheats)))

    try:
        _anneal_seed = globals().get(
            "STAINED_GLASS_ORDER_ANNEAL_SEED",
            globals().get("STAINED_GLASS_ANNEAL_SEED", globals().get("OPT_SEED", "random")),
        )
    except Exception:
        _anneal_seed = "random"


    if _panel_order == "auto":
        if _ord_mode == "scatter":
            _panel_order = "checker"
        elif _ord_mode == "random":
            _panel_order = "random"
        else:
            _panel_order = "hilbert"

    if _ord_enable and items and paths:
        # 1) 画像順（paths）
        _paths2 = list(paths)
        try:
            if _img_order in ("spectral_hilbert", "hilbert", "h"):
                _paths2 = reorder_global_spectral_hilbert(_paths2, objective=_obj)
            elif _img_order in ("spectral_diagonal", "diagonal", "d"):
                _paths2 = reorder_global_spectral_diagonal(_paths2, objective=_obj, diag_dir=_diag_dir)
            elif _img_order in ("anneal", "a"):
                _paths2 = reorder_global_anneal(_paths2, objective=_obj, iters=_anneal_steps, reheats=_anneal_reheats, seed=_anneal_seed)
            elif _img_order in ("shuffle", "random", "rand"):
                rng.shuffle(_paths2)
            else:
                # none / unknown: keep original order
                pass
        except Exception as e:
            _kana_silent_exc('core:SG_ORDER:img', e)
            pass
        paths = _paths2

        # 2) パネル順（items）
        def _panel_key_diag(itm):
            cx, cy = itm.get("center", (0.0, 0.0))
            try:
                u = (float(cx) - float(x0)) / float(max(1.0, float(w0)))
                v = (float(cy) - float(y0)) / float(max(1.0, float(h0)))
            except Exception:
                u = v = 0.0
            return (u + v, u)

        def _panel_key_xy(itm):
            cx, cy = itm.get("center", (0.0, 0.0))
            return (cy, cx)

        def _panel_key_hilbert(itm):
            cx, cy = itm.get("center", (0.0, 0.0))
            try:
                u = (float(cx) - float(x0)) / float(max(1.0, float(w0)))
                v = (float(cy) - float(y0)) / float(max(1.0, float(h0)))
            except Exception:
                u = v = 0.0
            u = float(max(0.0, min(1.0, u)))
            v = float(max(0.0, min(1.0, v)))
            n2 = 1 << int(_hb)
            xi = int(max(0, min(n2 - 1, int(round(u * float(n2 - 1))))))
            yi = int(max(0, min(n2 - 1, int(round(v * float(n2 - 1))))))
            return _hilbert_index(xi, yi, order=int(_hb))

        def _panel_key_checker(itm):
            cx, cy = itm.get("center", (0.0, 0.0))
            try:
                u = (float(cx) - float(x0)) / float(max(1.0, float(w0)))
                v = (float(cy) - float(y0)) / float(max(1.0, float(h0)))
            except Exception:
                u = v = 0.0
            u = float(max(0.0, min(1.0, u)))
            v = float(max(0.0, min(1.0, v)))
            n2 = 1 << int(_hb)
            xi = int(max(0, min(n2 - 1, int(round(u * float(n2 - 1))))))
            yi = int(max(0, min(n2 - 1, int(round(v * float(n2 - 1))))))
            parity = (xi + yi) & 1
            h = _hilbert_index(xi, yi, order=int(_hb))
            return (parity, h)

        try:
            if _panel_order in ("diag", "diagonal"):
                try:
                    centers = [itm.get("center", (0.0, 0.0)) for itm in items]
                    idxs = _mosaic_pos_order_diagonal(centers, diag_dir=_diag_dir)
                    items = [items[i] for i in idxs]
                except Exception:
                    items = sorted(items, key=_panel_key_diag)
            elif _panel_order in ("xy", "scan", "scanline"):
                items = sorted(items, key=_panel_key_xy)
            elif _panel_order in ("hilbert", "h"):
                items = sorted(items, key=_panel_key_hilbert)
            elif _panel_order in ("checker", "cb", "check"):
                items = sorted(items, key=_panel_key_checker)
            elif _panel_order in ("random", "rand", "shuffle"):
                _tmp = list(items)
                rng.shuffle(_tmp)
                items = _tmp
            else:
                pass
        except Exception as e:
            _kana_silent_exc('core:SG_ORDER:panel', e)
            pass

        # 3) ログ
        try:
            note(_lang(
                f"[StainedGlass] order: panels={_panel_order} images={_img_order} obj={_obj} diag={_diag_dir}",
                f"[StainedGlass] order: panels={_panel_order} images={_img_order} obj={_obj} diag={_diag_dir}"
            ))
        except Exception:
            pass

        # 並び重視のときは「顔優先キュー」で順序が崩れるため、必要なら off 推奨
        try:
            if bool(globals().get("STAINED_GLASS_ORDER_DISABLE_FACE_PRIORITY", True)):
                use_priority = False
        except Exception:
            pass

# 優先割当: 顔あり画像は太いパネルへ、細いパネルは非顔画像へ寄せる
    face_q = deque()
    other_q = deque()
    if use_ff and use_priority:
        # 顔あり画像を「優先的」に割り当てる場合、候補の順番がランダムだと
        # “目チェックで落ちる顔” を何度も引いて無駄なリトライが増えがちです。
        # そこで、顔bboxが大きく（=目が欠けにくい）かつ中心寄り（=切れにくい）な順に並べて
        # なるべく一発で通りやすい顔から使います（必要なら False で無効化できます）。
        sort_enable = bool(globals().get("STAINED_GLASS_FACE_PRIORITY_SORT", True))
        center_penalty = float(globals().get("STAINED_GLASS_FACE_PRIORITY_CENTER_PENALTY", 0.15))
        face_list = []
        for _p in paths:
            try:
                face = _sg_get_face_raw(_p)
                if isinstance(face, (list, tuple)) and len(face) == 5:
                    if sort_enable:
                        ow, oh = _sg_get_wh(_p)
                        try:
                            _, fx, fy, fw, fh = face
                        except Exception:
                            fx = fy = fw = fh = 0
                        # score: 顔が大きいほど +、中心から離れるほど -（少しだけ）
                        try:
                            area_ratio = (float(fw) * float(fh)) / float(max(1, int(ow) * int(oh)))
                        except Exception:
                            area_ratio = 0.0
                        try:
                            cx = (float(fx) + float(fw) * 0.5) / float(max(1, int(ow)))
                            cy = (float(fy) + float(fh) * 0.5) / float(max(1, int(oh)))
                            dist = math.hypot(cx - 0.5, cy - 0.5)
                        except Exception:
                            dist = 0.5
                        score = float(area_ratio) - float(dist) * float(center_penalty)
                        face_list.append((score, _p))
                    else:
                        face_q.append(_p)
                else:
                    other_q.append(_p)
            except Exception:
                other_q.append(_p)

        if sort_enable:
            try:
                face_list.sort(key=lambda t: t[0], reverse=True)
            except Exception:
                pass
            for _, pp in face_list:
                face_q.append(pp)
    else:
        other_q = deque(list(paths))


    # face-fit 診断カウンタ（ログ用）
    # - “目が欠ける” の原因が「細すぎ」「判定が厳しすぎ」「顔検出が取れてない」など、どこに多いかを数で見える化する
    ff_stats = {
        "panels": 0,
        "thin_panels": 0,
        "tries": 0,
        "ok_face": 0,
        "assigned_face": 0,
        "no_face": 0,
        "fail_thin_face": 0,
        "fail_eye": 0,
        "fallback": 0,
        # thin パネルで face_focus を切って“顔を狙わない”描画に落とした回数（目欠け対策）
        "relaxed_thin": 0,
        # 最後の手段（フォールバック）で、目欠け回避のために face_focus を切って採用した回数
        "relaxed_fallback": 0,
    }

    # パネル貼り込み
    for it in items:
        poly = it["poly"]
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        bx0 = int(max(x0, math.floor(min(xs))))
        by0 = int(max(y0, math.floor(min(ys))))
        bx1 = int(min(x0 + w0, math.ceil(max(xs))))
        by1 = int(min(y0 + h0, math.ceil(max(ys))))
        bw = int(max(1, bx1 - bx0))
        bh = int(max(1, by1 - by0))
        if bw <= 1 or bh <= 1:
            continue

        # 多角形マスク（bbox内座標にシフト）
        # - pmask0: 判定用（2値。>=128 を白とみなす）
        # - pmask : 描画用（アンチエイリアス/フェザー適用）
        pts_local = [(float(px - bx0), float(py - by0)) for (px, py) in poly]

        # overlap: 鉛線幅ぶん、タイルマスクを“少しだけ”広げて継ぎ目の白抜けを減らす
        overlap_px = int(max(0, int(round(lead_w * float(globals().get("STAINED_GLASS_OVERLAP_FACTOR", 0.75))))))

        # パネルマスクを高解像度で描いて縮小（多角形境界のジャギー低減）
        try:
            _ss = int(globals().get("STAINED_GLASS_MASK_SUPERSAMPLE", 2))
        except Exception:
            _ss = 2
        _ss = int(max(1, min(4, _ss)))

        if _ss <= 1:
            pmask0 = Image.new("L", (bw, bh), 0)
            draw = ImageDraw.Draw(pmask0)
            try:
                draw.polygon(pts_local, fill=255)
            except Exception as e:
                _kana_silent_exc('core:L14685', e)
                continue
            if overlap_px > 0:
                try:
                    pmask = pmask0.filter(ImageFilter.MaxFilter(size=overlap_px * 2 + 1))
                except Exception:
                    pmask = pmask0
            else:
                pmask = pmask0
        else:
            # 高解像度マスク（2x/3x/4x）→縮小で AA
            try:
                pmask_hi = Image.new("L", (int(bw * _ss), int(bh * _ss)), 0)
                draw_hi = ImageDraw.Draw(pmask_hi)
                pts_hi = [(float(x) * float(_ss), float(y) * float(_ss)) for (x, y) in pts_local]
                draw_hi.polygon(pts_hi, fill=255)
            except Exception as e:
                _kana_silent_exc('core:L14685', e)
                continue

            # overlap は hi-res 側で先に行い、縮小で滑らかにする
            if overlap_px > 0:
                try:
                    ohi = int(max(1, int(round(float(overlap_px) * float(_ss)))))
                    pmask_hi = pmask_hi.filter(ImageFilter.MaxFilter(size=ohi * 2 + 1))
                except Exception:
                    pass

            try:
                # Pillow 9+ の Resampling
                pmask = pmask_hi.resize((bw, bh), resample=Image.Resampling.LANCZOS)  # type: ignore
            except Exception:
                pmask = pmask_hi.resize((bw, bh), resample=Image.LANCZOS)

            # 判定用は 2値へ（ロジックが大きく変わらないように）
            try:
                pmask0 = pmask.point(lambda v: 255 if int(v) >= 128 else 0)
            except Exception:
                pmask0 = Image.new("L", (bw, bh), 0)
                draw = ImageDraw.Draw(pmask0)
                try:
                    draw.polygon(pts_local, fill=255)
                except Exception as e:
                    _kana_silent_exc('core:L14685', e)
                    continue

        # 見た目の境界をさらに滑らかにする（パネル貼り込み用マスクのみ）
        # ※ 判定用の pmask0 はそのまま（ロジックが変わらないように）
        try:
            _feather = float(globals().get("STAINED_GLASS_MASK_FEATHER_PX", 0.0))
        except Exception:
            _feather = 0.0
        if _feather > 0.0:
            try:
                pmask = pmask.filter(ImageFilter.GaussianBlur(radius=float(_feather)))
            except Exception:
                pass

# 「細いパネル」判定（顔を避ける）
        try:
            hist = pmask0.histogram()
            poly_area = int(hist[255]) if len(hist) >= 256 else 0
        except Exception:
            poly_area = 0
        try:
            fill_ratio = float(poly_area) / float(max(1, bw * bh))
        except Exception:
            fill_ratio = 0.0
        is_thin = (int(min(bw, bh)) < int(_min_short)) or (float(fill_ratio) < float(_min_fill))
        # 細いパネルに「顔を狙って」貼ると、目だけ欠ける/細片になる確率が上がります。
        # そこで thin パネルの挙動をモード化します：
        #   - "reject"      : 顔あり画像は避ける（従来挙動、他の画像が十分ある前提）
        #   - "nofacefocus" : 顔があっても face_focus を切り、目チェックもスキップして“顔を狙わない”描画に落とす（既定）
        try:
            thin_mode = str(globals().get("STAINED_GLASS_FACE_FIT_THIN_MODE", "nofacefocus")).lower().strip()
        except Exception:
            thin_mode = "nofacefocus"
        if thin_mode not in ("reject", "avoid", "nofacefocus"):
            thin_mode = "nofacefocus"


        if use_ff and use_facefit:
            try:
                ff_stats["panels"] += 1
                if is_thin:
                    ff_stats["thin_panels"] += 1
            except Exception as e:
                _kana_silent_exc('core:L14715', e)
                pass
        # 画像選択（priority + face-fit retry）
        chosen_p = None
        tile = None
        _ff_fail_eye_local = 0  # 1パネル内の fail_eye カウンタ（failfast 用）


        if use_ff:
            if use_facefit:
                for _t in range(_max_tries):
                    cand_p = None
                    if is_thin:
                        if len(other_q) > 0:
                            cand_p = other_q.popleft()
                        elif len(face_q) > 0:
                            cand_p = face_q.popleft()
                    else:
                        if len(face_q) > 0:
                            cand_p = face_q.popleft()
                        elif len(other_q) > 0:
                            cand_p = other_q.popleft()

                    if cand_p is None:
                        break


                    if use_ff and use_facefit:
                        try:
                            ff_stats["tries"] += 1
                        except Exception as e:
                            _kana_silent_exc('core:L14744', e)
                            pass
                                        # まず face bbox（タイル座標）を推定：目チェックで弾く場合、レンダーを避けて高速化
                    fbox = _sg_face_box_in_tile(cand_p, bw, bh)

                    # 細いパネルで顔が出る場合の扱い（thin_mode）
                    #  - reject/avoid  : 顔ありは避ける（従来）
                    #  - nofacefocus   : 顔があっても face_focus を切り、目チェックもスキップして“顔を狙わない”描画に落とす
                    relaxed_thin = False
                    if is_thin and fbox is not None:
                        if thin_mode in ("reject", "avoid"):
                            if use_ff and use_facefit:
                                try:
                                    ff_stats["fail_thin_face"] += 1
                                except Exception as e:
                                    _kana_silent_exc('core:L14754', e)
                                    pass
                            face_q.append(cand_p)
                            tile = None
                            continue
                        # nofacefocus: face_focus を切って描画（目チェックもスキップ）
                        relaxed_thin = True
                        try:
                            tile = _tile_render_cached(cand_p, bw, bh, "fill", use_face_focus=False)
                        except Exception:
                            face_q.append(cand_p)
                            tile = None
                            continue
                        fbox = None
                    else:
                        # 目チェック（face_focus 前に判定して、弾くならレンダーしない）
                        if (fbox is not None) and (not _sg_face_ok(pmask0, fbox, points_mode=_pts_mode, pt_ok_ratio=_pt_ok_ratio)):
                            if use_ff and use_facefit:
                                try:
                                    ff_stats["fail_eye"] += 1
                                except Exception as e:
                                    _kana_silent_exc('core:L14765', e)
                                    pass
                            try:
                                _ff_fail_eye_local += 1
                            except Exception:
                                _ff_fail_eye_local = 1

                            # failfast: 失敗が多いパネルは、少しだけ判定を緩めて採用に寄せる（過剰ループ抑制）
                            if (_failfast_eye > 0) and (_ff_fail_eye_local >= _failfast_eye):
                                try:
                                    _rr = float(max(0.60, min(1.0, float(_pt_ok_ratio) - 0.20)))
                                except Exception:
                                    _rr = 0.70
                                if _sg_face_ok(pmask0, fbox, points_mode="lite", pt_ok_ratio=_rr):
                                    # “最後の手段”扱いとしてカウント（目欠け率は抑えつつ速度も優先）
                                    try:
                                        ff_stats["relaxed_fallback"] += 1
                                    except Exception:
                                        pass
                                else:
                                    face_q.append(cand_p)
                                    tile = None
                                    continue
                            else:
                                face_q.append(cand_p)
                                tile = None
                                continue

                        # ここまで来たら採用候補なのでレンダー
                        try:
                            tile = _tile_render_cached(cand_p, bw, bh, "fill", use_face_focus=True)
                        except Exception:
                            face_q.append(cand_p)
                            tile = None
                            continue
                    chosen_p = cand_p
                    break


                # どうしても決まらない場合のフォールバック
                # ※face-fit 有効時は、フォールバックでも「目が欠ける」面を極力避ける（検証をスキップしない）
                if tile is None:
                    if use_ff and use_facefit:
                        try:
                            ff_stats["fallback"] += 1
                        except Exception as e:
                            _kana_silent_exc('core:L14791', e)
                            pass
                    # フォールバック探索回数（既定は少し多め）
                    try:
                        _fb_max = int(globals().get("STAINED_GLASS_FACE_FIT_FALLBACK_TRIES", max(8, _max_tries * 2)))
                    except Exception:
                        _fb_max = max(8, _max_tries * 2)
                    _fb_max = int(max(1, _fb_max))

                    _picked = None
                    _picked_tile = None

                    for _fb in range(_fb_max):
                        cand_p = None
                        # まずは非顔（other）を優先し、それでも無ければ顔キューを使う
                        if len(other_q) > 0:
                            cand_p = other_q.popleft()
                        elif len(face_q) > 0:
                            cand_p = face_q.popleft()

                        if cand_p is None:
                            break

                        _timg = _tile_render_cached(cand_p, bw, bh, "fill", use_face_focus=True)
                        fbox_fb = _sg_face_box_in_tile(cand_p, bw, bh)

                        # 細いパネルで顔が出る場合の扱い（thin_mode）
                        if is_thin and fbox_fb is not None:
                            if thin_mode in ("reject", "avoid"):
                                face_q.append(cand_p)
                                continue
                            # nofacefocus: face_focus を切って採用（目チェックもスキップ）
                            try:
                                _timg = _tile_render_cached(cand_p, bw, bh, "fill", use_face_focus=False)
                            except Exception:
                                face_q.append(cand_p)
                                continue
                            fbox_fb = None

                        # 目が多角形マスク内に入っていなければ後ろに回す（=再探索）
                        if (fbox_fb is not None) and (not _sg_face_ok(pmask0, fbox_fb)):
                            face_q.append(cand_p)
                            continue

                        _picked = cand_p
                        _picked_tile = _timg
                        break

                    chosen_p = _picked
                    tile = _picked_tile

                    # それでも決まらない場合は “最後の手段” として1枚だけ取る
                    # ただし face-fit 有効時は、顔があるのに「目が欠ける」見た目になりやすいので
                    # 最後の最後で顔を引いた場合でも、目チェックに落ちるなら face_focus を切って採用します。
                    if tile is None:
                        picked_from_face = False
                        if len(other_q) > 0:
                            chosen_p = other_q.popleft()
                        elif len(face_q) > 0:
                            chosen_p = face_q.popleft()
                            picked_from_face = True

                        if chosen_p is not None:
                            if picked_from_face and use_ff and use_facefit:
                                try:
                                    fbox_lr = _sg_face_box_in_tile(chosen_p, bw, bh)
                                except Exception:
                                    fbox_lr = None

                                if (fbox_lr is not None) and (not _sg_face_ok(pmask0, fbox_lr)):
                                    try:
                                        ff_stats["relaxed_fallback"] += 1
                                    except Exception:
                                        pass
                                    # face_focus を切る（“顔を狙わない”）ことで目欠けの確率を下げる
                                    tile = _tile_render_cached(chosen_p, bw, bh, "fill", use_face_focus=False)
                                else:
                                    tile = _tile_render_cached(chosen_p, bw, bh, "fill", use_face_focus=True)
                            else:
                                tile = _tile_render_cached(chosen_p, bw, bh, "fill", use_face_focus=True)

                    if use_ff and use_facefit and (chosen_p is not None):
                        try:
                            fbox_fb2 = _sg_face_box_in_tile(chosen_p, bw, bh)
                            if fbox_fb2 is not None:
                                ff_stats["assigned_face"] += 1
                            else:
                                ff_stats["no_face"] += 1
                        except Exception as e:
                            _kana_silent_exc('core:L14851', e)
                            pass
            else:
                # priority のみ（リトライなし）
                if is_thin and len(other_q) > 0:
                    chosen_p = other_q.popleft()
                elif (not is_thin) and len(face_q) > 0:
                    chosen_p = face_q.popleft()
                elif len(other_q) > 0:
                    chosen_p = other_q.popleft()
                elif len(face_q) > 0:
                    chosen_p = face_q.popleft()

                if chosen_p is not None:
                    tile = _tile_render_cached(chosen_p, bw, bh, "fill", use_face_focus=True)
        else:
            # face-focus 無しで軽量に（draft_to で I/O も軽く）
            if len(other_q) <= 0:
                continue
            chosen_p = other_q.popleft()
            with open_image_safe(chosen_p, draft_to=(bw, bh), force_mode="RGB") as im:
                tile = resize_into_cell(im, bw, bh, "fill")

        if tile is None:
            continue

        # 貼り込み（mask / canvas）
        canvas.paste(tile, (bx0, by0), pmask)
        mask.paste(pmask0, (bx0, by0), pmask0)


    if use_ff and use_facefit and bool(globals().get("STAINED_GLASS_FACE_FIT_STATUS_LOG", True)):
        try:
            _p = int(ff_stats.get("panels", 0))
            _thin = int(ff_stats.get("thin_panels", 0))
            _tries = int(ff_stats.get("tries", 0))
            _okf = int(ff_stats.get("ok_face", 0))
            _ass = int(ff_stats.get("assigned_face", 0))
            _nf = int(ff_stats.get("no_face", 0))
            _ft = int(ff_stats.get("fail_thin_face", 0))
            _fe = int(ff_stats.get("fail_eye", 0))
            _fb = int(ff_stats.get("fallback", 0))
            _rt = int(ff_stats.get("relaxed_thin", 0))
            _rf = int(ff_stats.get("relaxed_fallback", 0))
            print(
                f"[StainedGlass] facefit_stats: panels={_p} thin={_thin} tries={_tries} ok_face={_okf} "
                f"assigned_face={_ass} fail_eye={_fe} fail_thin_face={_ft} fallback={_fb} no_face={_nf} "
                f"relaxed=(thin={_rt},fb={_rf}) | yolo={SG_PERF['yolo_sec']:.2f}s calls={SG_PERF['yolo_calls']}"
            )
        except Exception as e:
            _kana_silent_exc('core:L14895', e)
            pass
    # 鉛線（境界線）
    # ※ lead_w / lead_rgb / lead_a は上で確定済み（重複計算しない）

    # gap fill:
    # max_vertices 制約で頂点削減を強めると、隣接パネル同士の境界が“完全一致”しない場合があります。
    # その結果、背景色（黒など）が 1〜数十px の“割れ目”として見えることがあります。
    # ただし後処理（埋め）は好みが分かれるため、既定は OFF（無効）です。
    # ※STAINED_GLASS_GAP_FILL_ENABLE=True のときのみ実行します。
    try:
        _sg_gap_enable = bool(globals().get("STAINED_GLASS_GAP_FILL_ENABLE", False))
    except Exception:
        # 既定は OFF。例外時も安全側（無効）に倒します。
        _sg_gap_enable = False

    if _sg_gap_enable:
        try:
            import numpy as _np  # noqa: N812

            _arr = _np.array(canvas, dtype=_np.uint8)
            _um = _np.array(mask, dtype=_np.uint8) > 0  # union mask（パネルが塗られている領域）
            _rect = _np.zeros_like(_um, dtype=bool)
            _rect[int(y0):int(y0 + h0), int(x0):int(x0 + w0)] = True

            _holes = _rect & (~_um)
            if _holes.any():
                _method = str(globals().get("STAINED_GLASS_GAP_FILL_METHOD", "inpaint")).strip().lower()
                if _method not in ("propagate",):
                    _method = "propagate"

                # --- A) inpaint は使用しない（センパイ要望） ---
                # ここでは inpaint は実行せず、B) の 4近傍伝播のみを使う

                # --- B) 4近傍伝播（保険） ---
                if _method == "propagate":
                    _iters = int(globals().get("STAINED_GLASS_GAP_FILL_MAX_ITERS", 24))
                    _iters = int(max(4, min(96, _iters)))
                    for _ in range(_iters):
                        _holes = _rect & (~_um)
                        if not _holes.any():
                            break

                        # 上/下
                        _cand = _holes[1:, :] & _um[:-1, :]
                        if _cand.any():
                            _arr[1:, :][_cand] = _arr[:-1, :][_cand]
                            _um[1:, :][_cand] = True

                        _cand = _holes[:-1, :] & _um[1:, :]
                        if _cand.any():
                            _arr[:-1, :][_cand] = _arr[1:, :][_cand]
                            _um[:-1, :][_cand] = True

                        # 左/右
                        _cand = _holes[:, 1:] & _um[:, :-1]
                        if _cand.any():
                            _arr[:, 1:][_cand] = _arr[:, :-1][_cand]
                            _um[:, 1:][_cand] = True

                        _cand = _holes[:, :-1] & _um[:, 1:]
                        if _cand.any():
                            _arr[:, :-1][_cand] = _arr[:, 1:][_cand]
                            _um[:, :-1][_cand] = True

                    canvas = Image.fromarray(_arr, mode="RGB")
        except Exception as e:
            _kana_silent_exc('core:L14932', e)
            pass

    # Lead overlay
    if lead_w > 0 and lead_a > 0.0:
        try:
            # lead の生成方法:
            #   - "mask"  : 従来（各パネルの縁マスクを合成）
            #              ※共有境界が「両側のパネルで」描かれるため、lead_w=1 でも太く見えやすい
            #   - "edges" : 共有エッジを重複排除して描く（細線が作りやすい / ジャギーが減る）
            try:
                _lm = str(globals().get("STAINED_GLASS_LEAD_METHOD", "mask")).strip().lower()
            except Exception:
                _lm = "mask"
            if _lm not in ("mask", "edges"):
                _lm = "mask"

            # さらに、PIL の Max/MinFilter は環境によってサイズ制限があるため、3x3 を反復して膨張/収縮する。
            from PIL import ImageChops, ImageFilter

            def _clamp_int(v, lo, hi):
                try:
                    iv = int(v)
                except Exception:
                    iv = int(lo)
                return int(max(int(lo), min(int(hi), iv)))

            def _dilate3(im, steps):
                out = im
                n = _clamp_int(steps, 0, 64)  # 反復数の暴走を防ぐ
                for _ in range(n):
                    out = out.filter(ImageFilter.MaxFilter(3))
                return out

            def _erode3(im, steps):
                out = im
                n = _clamp_int(steps, 0, 64)  # 反復数の暴走を防ぐ
                for _ in range(n):
                    out = out.filter(ImageFilter.MinFilter(3))
                return out

            # 値の安全化（例外を未然に防ぐ）
            try:
                _la = float(lead_a)
            except Exception:
                _la = 0.0
            try:
                if not math.isfinite(_la):
                    _la = 0.0
            except Exception:
                _la = 0.0
            _la = 0.0 if _la < 0.0 else (1.0 if _la > 1.0 else _la)

            def _clamp_u8(x):
                try:
                    xi = int(x)
                except Exception:
                    return 0
                if xi < 0:
                    return 0
                if xi > 255:
                    return 255
                return xi

            try:
                _lr = (_clamp_u8(lead_rgb[0]), _clamp_u8(lead_rgb[1]), _clamp_u8(lead_rgb[2]))
            except Exception:
                _lr = (0, 0, 0)

            # 出力サイズ（lead の合成領域）。ここで out_w/out_h を確実に定義しておく
            try:
                out_w = int(canvas.size[0])
                out_h = int(canvas.size[1])
            except Exception:
                out_w = int(width)
                out_h = int(height)
            out_w = int(max(1, out_w))
            out_h = int(max(1, out_h))

            # lead_mask を作る
            lead_mask = None

            if _lm == "edges":
                # 共有エッジを重複排除して 1回だけ描く（=太さが二重になりにくい）
                try:
                    _ss = int(globals().get("STAINED_GLASS_LEAD_SUPERSAMPLE", 2))
                except Exception:
                    _ss = 2
                # 4K でのメモリ安全を優先して 1..2 に制限
                _ss = 1 if _ss < 1 else (2 if _ss > 2 else _ss)

                try:
                    _q = int(globals().get("STAINED_GLASS_LEAD_EDGE_QUANT", 4))
                except Exception:
                    _q = 4
                _q = int(max(1, min(16, _q)))

                edges = {}
                for it in items:
                    poly = it.get('poly')
                    if (not poly) or (len(poly) < 2):
                        continue
                    L = int(len(poly))
                    for i in range(L):
                        p1 = poly[i]
                        p2 = poly[(i + 1) % L]
                        try:
                            x1 = float(p1[0]); y1 = float(p1[1])
                            x2 = float(p2[0]); y2 = float(p2[1])
                        except Exception:
                            continue
                        # 量子化して“同一エッジ”を判定（向き違いも同一視）
                        try:
                            a = (int(round(x1 * _q)), int(round(y1 * _q)))
                            b = (int(round(x2 * _q)), int(round(y2 * _q)))
                        except Exception:
                            continue
                        if b < a:
                            a, b = b, a
                            x1, y1, x2, y2 = x2, y2, x1, y1
                        key = (a[0], a[1], b[0], b[1])
                        if key in edges:
                            continue
                        edges[key] = (x1, y1, x2, y2)

                # 高解像度で描いてから縮小（アンチエイリアス）
                try:
                    _m_w = int(out_w * _ss)
                    _m_h = int(out_h * _ss)
                    lead_hi = Image.new('L', (_m_w, _m_h), 0)
                    dr = ImageDraw.Draw(lead_hi)

                    w_hi = int(max(1, int(round(float(lead_w) * float(_ss)))))
                    for (x1, y1, x2, y2) in edges.values():
                        try:
                            dr.line((x1 * _ss, y1 * _ss, x2 * _ss, y2 * _ss), fill=255, width=w_hi)
                        except Exception:
                            continue

                    if _ss > 1:
                        try:
                            _resamp = Resampling.LANCZOS
                        except Exception:
                            try:
                                _resamp = Image.LANCZOS
                            except Exception:
                                _resamp = Image.BICUBIC
                        lead_mask = lead_hi.resize((out_w, out_h), _resamp)
                    else:
                        lead_mask = lead_hi
                except Exception:
                    # edges 方式が失敗したら従来方式へフォールバック
                    lead_mask = None
                    _lm = "mask"

            if lead_mask is None:
                # 従来方式（各ピースの縁マスクを合成）
                lead_mask = Image.new('L', (out_w, out_h), 0)
                pad = int(max(2, int(lead_w) * 2 + 4))  # ローカル領域の余白

                for it in items:
                    poly = it.get('poly')
                    if not poly:
                        continue
                    try:
                        xs = [p[0] for p in poly]
                        ys = [p[1] for p in poly]
                        x1 = max(0, int(min(xs)) - pad)
                        y1 = max(0, int(min(ys)) - pad)
                        x2 = min(out_w, int(max(xs)) + pad)
                        y2 = min(out_h, int(max(ys)) + pad)
                        bw = int(max(1, x2 - x1))
                        bh = int(max(1, y2 - y1))
                        pm = Image.new('L', (bw, bh), 0)
                        _poly_local = [(float(p[0]) - float(x1), float(p[1]) - float(y1)) for p in poly]
                        ImageDraw.Draw(pm).polygon(_poly_local, fill=255)

                        # lead_w px 相当の膨張/収縮
                        if lead_style == 'outer':
                            d = _dilate3(pm, lead_w)
                            edge = ImageChops.subtract(d, pm)
                        else:
                            d = _dilate3(pm, lead_w)
                            e = _erode3(pm, lead_w)
                            edge = ImageChops.subtract(d, e)

                        region = lead_mask.crop((x1, y1, x2, y2))
                        region = ImageChops.lighter(region, edge)
                        lead_mask.paste(region, (x1, y1))
                    except Exception:
                        continue

            # α を反映（0..255 -> 0..(255*lead_a)）
            _laf = float(_la)
            lut = [int(i * _laf) for i in range(256)]
            alpha = lead_mask.point(lut)

            # lead（鉛線）のジャギーを軽減（αを軽くぼかす）
            try:
                _ls = float(globals().get("STAINED_GLASS_LEAD_SMOOTH_PX", 0.0))
            except Exception:
                _ls = 0.0
            if _ls > 0.0:
                try:
                    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=float(_ls)))
                except Exception:
                    pass

            lead_rgba = Image.new('RGBA', (out_w, out_h), (_lr[0], _lr[1], _lr[2], 0))
            lead_rgba.putalpha(alpha)

            canvas_rgba = canvas.convert('RGBA')
            canvas = Image.alpha_composite(canvas_rgba, lead_rgba).convert('RGB')
        except Exception as e:
            _kana_silent_exc('core:lead_overlay', e)
            pass
    layout_name = "StainedGlass(Voronoi)"
    try:
        _b = str(_used_backend or "")
        if _b == "warpgrid":
            layout_name = "StainedGlass(WarpGrid)"
        elif _b == "rectgrid":
            layout_name = "StainedGlass(RectGrid)"
        elif _b == "single_panel":
            layout_name = "StainedGlass(SinglePanel)"
        elif _b == "warpgrid_err":
            layout_name = "StainedGlass(WarpGridErr)"
    except Exception as e:
        _kana_silent_exc('core:L14984', e)
        pass
    info = {
        "Layout": str(layout_name),
        "tiles": int(n),
        "order": str(order),
        "lead_w": int(lead_w),
        "face_focus": bool(use_ff),
        "facefit": bool(use_facefit),
        "backend": str(_used_backend or "unknown"),
    }
    return canvas, mask, info


def mean_luma_masked(img: Image.Image, mask: Optional[Image.Image], sample: int = 512) -> Optional[float]:
    """画像の平均輝度（0.0〜1.0）を計算します。

    - mask が指定されている場合は、mask の非ゼロ部分のみを対象にします。
    - 大きい画像は計算負荷を抑えるため、最大辺が sample を超える場合に縮小して計算します。
    - 対象ピクセルが 0 件の場合は None を返します。
    """
    # 輝度（L）計算用にグレースケールへ変換
    g = img.convert("L")
    m = None
    if mask is not None:
        # マスクのサイズが一致しない場合は最近傍でリサイズ（マスクなので補間しない）
        m = mask.resize(g.size, Resampling.NEAREST) if mask.size != g.size else mask
    # 大きい画像は縮小して計算コストを抑える
    if max(g.size) > sample:
        s = max(1, max(g.size) // sample)
        new_size = (max(1, g.size[0] // s), max(1, g.size[1] // s))
        g = g.resize(new_size, Resampling.BILINEAR)
        if m is not None:
            m = m.resize(new_size, Resampling.NEAREST)
    gl = g.load()
    w, h = g.size
    if m is not None:
        ml = m.load()
        S = 0.0
        N = 0
        for y in range(h):
            for x in range(w):
                if ml[x, y] > 0:
                    S += gl[x, y]
                    N += 1
    else:
        # マスクが無い場合は全ピクセルを対象にする
        S = 0.0
        N = w * h
        for y in range(h):
            for x in range(w):
                S += gl[x, y]
    return (S / (N * 255.0)) if N > 0 else None

# -----------------------------------------------------------------------------
# 残ったスキマ（ギャップ）の検出と補正
# モザイクのギャップレス描画では、わずかに大きめに描画してからトリミングすることで、余白をなくすようにしています。
# しかし、整数丸めや行・列ごとのオーバーシュート量の違いにより、
# まれに 1 ピクセル幅のスキマが残ってしまうことがあります。
# このヘルパーはマスク画像を走査し、完全に空の列（縦方向のストライプ）や
# 行（横方向のストライプ）が可視領域内に存在するかをチェックし、
# 見つかった場合は 1 ピクセル分のシフト量を提案してスキマを埋めます。
# 縦方向のギャップでは左右どちら側か、横方向では上下どちら側かに応じて、
# 適切な向きにシフト方向を決定します。ギャップが見つからない場合は (0, 0) を返します。
# この補正は MOSAIC_AUTO_INTERPOLATE が有効なときのみ適用されます。
# 実際の利用箇所はギャップレス描画パス内の呼び出し元を参照してください。
def _detect_gap_shift(mask: Image.Image, margin: int, inner_w: int, inner_h: int, axis: str) -> Tuple[int, int]:
    """残ったスキマ（空列/空行）を埋めるための 1px シフト量を返します。

    axis:
      - "vertical"   : 可視領域内の「完全に空の列」を探して左右どちらへ寄せるかを返す
      - "horizontal" : 可視領域内の「完全に空の行」を探して上下どちらへ寄せるかを返す

    見つからない場合は (0, 0) を返します。
    """
    try:
        mload = mask.load()
    except Exception:
        return (0, 0)
    shift_x = 0
    shift_y = 0
    # 可視領域内で「完全に空の列/行（=スキマ）」があるかを走査する
    if axis == "vertical":
        x_start = max(0, margin)
        x_end = min(mask.size[0], margin + inner_w)
        for x in range(x_start, x_end):
            full_zero = True
            for y in range(margin, margin + inner_h):
                if mload[x, y] != 0:
                    full_zero = False
                    break
            if full_zero:
                # スキマが左右どちら側にあるかで 1px の寄せ方向を決める
                midpoint = margin + inner_w // 2
                shift_x = -1 if x < midpoint else 1
                break
    elif axis == "horizontal":
        y_start = max(0, margin)
        y_end = min(mask.size[1], margin + inner_h)
        for y in range(y_start, y_end):
            full_zero = True
            for x in range(margin, margin + inner_w):
                if mload[x, y] != 0:
                    full_zero = False
                    break
            if full_zero:
                midpoint = margin + inner_h // 2
                shift_y = -1 if y < midpoint else 1
                break
    return (shift_x, shift_y)


# =============================================================================
# セクション: 後処理（ガンマ/ハレーション/ビネット/粒状など）
# =============================================================================
def apply_gamma(img: Image.Image, gamma:float) -> Image.Image:
    if abs(gamma-1.0)<1e-4: return img
    inv=1.0/gamma; lut=[min(255,max(0,int((i/255.0)**inv*255.0+0.5))) for i in range(256)]
    if img.mode=="RGBA":
        r,g,b,a=img.split(); r=r.point(lut); g=g.point(lut); b=b.point(lut); return Image.merge("RGBA",(r,g,b,a))
    r,g,b=img.split(); r=r.point(lut); g=g.point(lut); b=b.point(lut); return Image.merge("RGB",(r,g,b))


# ---- 3点セット: ヘルパー ----
def _apply_halation_bloom(img, intensity=0.25, radius=18, threshold=0.70, knee=0.08):  # intensity: ハレーション強度 / radius: ぼかし半径
    """ハレーション（Bloom）を適用します。

    旧方式は画像全体のぼかしを合成するため、状況によっては“全体がふわっと明るくなる”ことがありました。
    ここでは「明るい部分だけ」を抽出（bright-pass）してからぼかしを合成することで、
    “光ってほしいところだけ光る”品のあるハレーションにします。

    - intensity : 0.0〜1.0（強さ）
    - radius    : ぼかし半径（px）
    - threshold : 0.0〜1.0（明部抽出のしきい値。0.60〜0.80 付近が目安）
    - knee      : 0.0〜0.5（ソフトニー。しきい値付近の立ち上がりを滑らかにする）
    """
    try:
        from PIL import Image, ImageFilter, ImageChops
    except Exception:
        return img

    if intensity <= 0 or radius <= 0:
        return img

    t = max(0.0, min(1.0, float(threshold))) * 255.0
    # しきい値付近が急に切れないよう、ソフトニー（knee）を適用します。
    # knee は 0.0〜0.5 目安（値を上げるほど立ち上がりが滑らかになります）
    knee = max(0.0, min(0.5, float(knee))) * 255.0

    # RGBA の場合は Alpha を保持して RGB で処理
    if img.mode == 'RGBA':
        rgb = img.convert('RGB')
        a = img.split()[3]
        has_alpha = True
    else:
        rgb = img.convert('RGB') if img.mode != 'RGB' else img
        a = None
        has_alpha = False

    # 明部抽出マスク（L）を作る：
    #   y <= t-knee -> 0
    #   y >= t+knee -> 255
    #   それ以外は線形補間（soft-knee）
    y = rgb.convert('L')
    def _knee_curve(v):
        if v <= t - knee:
            return 0
        if v >= t + knee:
            return 255
        return int((v - (t - knee)) / (2.0 * knee) * 255.0 + 0.5)
    mask = y.point(_knee_curve)

    # 明部のみを取り出してからぼかす（bright-pass）
    black = Image.new('RGB', rgb.size, (0, 0, 0))
    bright = Image.composite(rgb, black, mask)
    blur = bright.filter(ImageFilter.GaussianBlur(radius=radius))

    # screen 合成（黒は影響ゼロなので、抽出部だけが自然に光る）
    inv_mul = ImageChops.multiply(ImageChops.invert(rgb), ImageChops.invert(blur))
    screen = ImageChops.invert(inv_mul)
    out_rgb = ImageChops.blend(rgb, screen, max(0.0, min(1.0, float(intensity))))

    if has_alpha:
        return Image.merge('RGBA', (*out_rgb.split(), a))
    # 元が RGB 以外なら近いモードへ戻す（可能なら）
    if img.mode != 'RGB':
        try:
            return out_rgb.convert(img.mode)
        except Exception:
            return out_rgb
    return out_rgb

def _apply_vignette(img, strength=0.15, roundness=0.5):  # strength: ビネット強度 / roundness: 形（0.5 で均等）
    """普通のビネット（周辺減光）を適用します。

    - strength: 0.0〜1.0（端の暗さ。端は概ね (1-strength) 倍）
      目安：0.05〜0.30 くらいが自然です。
    - roundness: 0.0〜1.0（0.5 で左右上下が均等）
      0.9 付近で『上下が強め』、0.1 付近で『左右が強め』の楕円になりやすいです。
    """
    if strength <= 0:
        return img
    from PIL import Image, ImageFilter, ImageChops
    import numpy as np

    w, h = img.size
    cx, cy = w * 0.5, h * 0.5

    # 形（縦横の効き方）
    rr = float(roundness)
    rx = max(1.0, cx) * (0.90 + 0.10 * rr)
    ry = max(1.0, cy) * (0.90 + 0.10 * (1.0 - rr))

    # 低解像度マスクを作って拡大（高速化）
    mw, mh = max(64, w // 6), max(64, h // 6)
    yy, xx = np.mgrid[0:mh, 0:mw]
    sx = (xx * w / mw - cx) / rx
    sy = (yy * h / mh - cy) / ry

    # 中心 r=0 → 1.0、周辺 r=1 → (1-strength) へ滑らかに落とす
    r = np.clip((sx * sx + sy * sy) ** 0.5, 0.0, 1.0)
    power = 2.2  # 大きいほど『周辺だけ』が暗くなる（普通のビネットは 2.0〜3.0 が目安）
    mask_small = 1.0 - float(strength) * (r ** power)

    mask = np.clip(mask_small * 255.0, 0.0, 255.0).astype('uint8')
    mimg = Image.fromarray(mask).convert('L').resize((w, h), Image.BICUBIC)

    # 境界をほんのり柔らかくする
    mimg = mimg.filter(ImageFilter.GaussianBlur(radius=min(w, h) * 0.02))

    base = img.convert('RGB') if img.mode != 'RGB' else img
    m3 = Image.merge('RGB', (mimg, mimg, mimg))
    return ImageChops.multiply(base, m3)

# グレースケール（白黒）エフェクトを適用する。
# ハレーションなどの効果の後に呼び出し、画像を白黒に変換してから再び RGB に戻します。
def _apply_bw_effect(img):
    """白黒（モノクロ）化エフェクトを適用します。

    Halation/Bloom の後段で、いったん L（グレースケール）にしてから RGB に戻します。
    失敗した場合は元画像を返します。
    """
    try:
        gray = img.convert("L")
        return gray.convert("RGB")
    except Exception:
        return img

# セピア調エフェクト。グレースケール化した画像に暖色系のトーンを重ね、
# 元の画像とブレンドしてレトロな雰囲気を出します。
def _apply_sepia(img, intensity=0.35):
    """セピア調エフェクトを適用します。

    intensity は元画像とセピア画像のブレンド比率です（大きいほどセピアが強い）。
    失敗した場合は元画像を返します。
    """
    try:
        gray = img.convert("L")
        from PIL import ImageOps
        sepia = ImageOps.colorize(gray, "#704214", "#C0A080")
        # intensity を [0, 1] に収めてからブレンド
        alpha = max(0.0, min(1.0, float(intensity)))
        return Image.blend(img, sepia.convert(img.mode), alpha)
    except Exception:
        return img

# フィルムグレインエフェクト。輝度（明るさ）だけに“加算型”ノイズを加えてフィルムの粒子感を再現します。
# 注: 背景（余白）を綺麗に保ちたい場合は content_mask を渡すと、コンテンツ領域のみに適用できます。
def _apply_grain(img, amount=0.05, content_mask=None):
    """フィルムグレイン（粒子）エフェクトを加えます。

    - amount: ノイズ強度（0.0〜1.0 推奨）。
      加算型（ゼロ平均）なので、ブレンド方式よりも“黒浮き”や“色の濁り”が起きにくい設計です。
    - content_mask: 背景（余白）を除外したいときのマスク（L 推奨）。指定時はコンテンツ領域のみに適用します。

    失敗した場合は元画像を返します。
    """
    try:
        from PIL import Image
        import numpy as np

        a = max(0.0, min(1.0, float(amount)))
        if a <= 0.0:
            return img

        # まず RGB（必要なら Alpha を退避）へ
        if img.mode == 'RGBA':
            rgb = img.convert('RGB')
            alpha_ch = img.split()[3]
            has_alpha = True
        else:
            rgb = img.convert('RGB') if img.mode != 'RGB' else img
            alpha_ch = None
            has_alpha = False

        w, h = rgb.size

        # 0 を中心としたノイズを作る（Pillow の effect_noise は平均 128 付近）
        # sigma=50 は std が約 49 程度。これを amount 倍して輝度へ加算します。
        noise = Image.effect_noise((w, h), 50.0)
        n = np.asarray(noise, dtype=np.float32) - 128.0

        # 輝度（Y）だけへ加算（色相・彩度を汚しにくい）
        y, cb, cr = rgb.convert('YCbCr').split()
        yarr = np.asarray(y, dtype=np.float32)
        yarr = np.clip(yarr + (n * a), 0.0, 255.0).astype('uint8')
        y2 = Image.fromarray(yarr).convert('L')

        out_rgb = Image.merge('YCbCr', (y2, cb, cr)).convert('RGB')
        if has_alpha:
            out = Image.merge('RGBA', (*out_rgb.split(), alpha_ch))
        else:
            out = out_rgb
            if img.mode != 'RGB':
                # 元画像が RGB 以外なら近いモードへ戻す
                try:
                    out = out.convert(img.mode)
                except Exception as e:
                    _kana_silent_exc('core:L15302', e)
                    pass
        # マスク指定があれば、コンテンツ領域だけに適用（余白は元のまま）
        if content_mask is not None:
            try:
                m = content_mask.convert('L') if content_mask.mode != 'L' else content_mask
                out = Image.composite(out, img, m)
            except Exception as e:
                _kana_silent_exc('core:L15310', e)
                pass
        return out
    except Exception:
        return img

# 彩度ブースト（ビブランス）エフェクト。
# 低彩度の色ほど持ち上げ、元から鮮やかな色は上げすぎない“vibrance”寄りの動きにします。


# クラリティ（局所コントラスト）エフェクト。
# 輝度（Y）成分に対して「高周波成分（ハイパス）」を適度に加えることで、
# 質感や立体感を出します。色相・彩度を汚しにくいよう YCbCr で処理します。
# 注: 背景（余白）を綺麗に保ちたい場合は content_mask を渡すと、コンテンツ領域のみに適用できます。
def _apply_clarity(img, amount=0.12, radius=2.0, content_mask=None):
    """クラリティ（局所コントラスト）を調整します。

    - amount: 0.0〜1.0（強さ）。0.08〜0.20 が自然な目安です。
    - radius: ぼかし半径（px）。1.5〜3.0 が使いやすい目安です。
    - content_mask: 背景（余白）を除外したいときのマスク（L 推奨）。指定時はコンテンツ領域のみに適用します。

    失敗した場合は元画像を返します。
    """
    try:
        from PIL import Image, ImageFilter
        import numpy as np

        a = max(0.0, min(1.0, float(amount)))
        if a <= 0.0:
            return img

        r = float(radius)
        if r <= 0.0:
            return img

        # まず RGB（必要なら Alpha を退避）へ
        if img.mode == 'RGBA':
            rgb = img.convert('RGB')
            alpha_ch = img.split()[3]
            has_alpha = True
        else:
            rgb = img.convert('RGB') if img.mode != 'RGB' else img
            alpha_ch = None
            has_alpha = False

        # 輝度（Y）を取り出してぼかし → ハイパス
        y, cb, cr = rgb.convert('YCbCr').split()
        y_blur = y.filter(ImageFilter.GaussianBlur(radius=r))

        yarr = np.asarray(y, dtype=np.float32)
        barr = np.asarray(y_blur, dtype=np.float32)
        hp = yarr - barr

        # 極端なギラつきを避けるため、ハイパス成分をやわらかく制限（tanh でソフトクリップ）
        # 係数 64 は “強さの感触” を安定させるための経験値です。
        hp = np.tanh(hp / 64.0) * 64.0

        # amount に応じて加算（1.25 は “クラリティっぽさ” の係数）
        y2 = np.clip(yarr + (hp * (a * 1.25)), 0.0, 255.0).astype('uint8')
        y2img = Image.fromarray(y2).convert('L')

        out_rgb = Image.merge('YCbCr', (y2img, cb, cr)).convert('RGB')
        if has_alpha:
            out = Image.merge('RGBA', (*out_rgb.split(), alpha_ch))
        else:
            out = out_rgb
            if img.mode != 'RGB':
                # 元画像が RGB 以外なら近いモードへ戻す
                try:
                    out = out.convert(img.mode)
                except Exception as e:
                    _kana_silent_exc('core:L15381', e)
                    pass
        # マスク指定があれば、コンテンツ領域だけに適用（余白は元のまま）
        if content_mask is not None:
            try:
                m = content_mask.convert('L') if content_mask.mode != 'L' else content_mask
                out = Image.composite(out, img, m)
            except Exception as e:
                _kana_silent_exc('core:L15389', e)
                pass
        return out
    except Exception:
        return img

def _apply_unsharp_mask(img, amount=0.35, radius=1.2, threshold=3, content_mask=None):
    """アンシャープマスク（輪郭強調）を適用します。

    - amount: 0.0〜1.0（強さ）。0.25〜0.55 が使いやすい目安です。
    - radius: 半径（px）。0.8〜1.8 が使いやすい目安です。
    - threshold: しきい値（0〜20）。大きいほどノイズを拾いにくくなります（2〜6 が目安）。
    - content_mask: 背景（余白）を除外したいときのマスク（L 推奨）。指定時はコンテンツ領域のみに適用します。

    失敗した場合は元画像を返します。
    """
    try:
        from PIL import Image, ImageFilter
        import numpy as np

        a = float(amount)
        if a <= 0.0:
            return img
        a = max(0.0, min(1.0, a))

        r = float(radius)
        if r <= 0.0:
            return img
        r = max(0.0, min(50.0, r))

        th = int(threshold)
        th = max(0, min(20, th))

        # Pillow の UnsharpMask の percent は 0〜500 程度が目安。
        # amount=0.35 で percent≈150 前後になるようにスケールします。
        percent = int(max(0, min(500, round(a * 430))))

        # まず RGB（必要なら Alpha を退避）へ
        if img.mode == 'RGBA':
            rgb = img.convert('RGB')
            alpha_ch = img.split()[3]
            has_alpha = True
        else:
            rgb = img.convert('RGB') if img.mode != 'RGB' else img
            alpha_ch = None
            has_alpha = False

        # 輝度（Y）だけにアンシャープをかける（色のにじみを抑える）
        y, cb, cr = rgb.convert('YCbCr').split()
        y2 = y.filter(ImageFilter.UnsharpMask(radius=r, percent=percent, threshold=th))

        out_rgb = Image.merge('YCbCr', (y2, cb, cr)).convert('RGB')
        if has_alpha:
            out = Image.merge('RGBA', (*out_rgb.split(), alpha_ch))
        else:
            out = out_rgb
            if img.mode != 'RGB':
                # 元画像が RGB 以外なら近いモードへ戻す
                try:
                    out = out.convert(img.mode)
                except Exception as e:
                    _kana_silent_exc('core:L15450', e)
                    pass
        # マスク指定があれば、コンテンツ領域だけに適用（余白は元のまま）
        if content_mask is not None:
            try:
                m = content_mask.convert('L') if content_mask.mode != 'L' else content_mask
                out = Image.composite(out, img, m)
            except Exception as e:
                _kana_silent_exc('core:L15458', e)
                pass
        return out
    except Exception:
        return img

def _apply_vibrance(img, factor=1.2):
    """ビブランス（彩度）を調整します。

    - factor: 1.0 で変化なし。1.20〜1.40 が使いやすい目安です。
      ※本実装は“低彩度ほど強めに効く”方式のため、単純な彩度倍率とは一致しません。

    失敗した場合は元画像を返します。
    """
    try:
        from PIL import Image
        import numpy as np

        f = float(factor)
        if abs(f - 1.0) < 1e-6:
            return img

        # まずは RGB（必要なら Alpha を退避）へ
        if img.mode == 'RGBA':
            rgb = img.convert('RGB')
            alpha_ch = img.split()[3]
            has_alpha = True
        else:
            rgb = img.convert('RGB') if img.mode != 'RGB' else img
            alpha_ch = None
            has_alpha = False

        # PIL の HSV を使って S（彩度）だけを操作する
        hsv = rgb.convert('HSV')
        h, s, v = hsv.split()

        s_arr = np.asarray(s, dtype=np.float32) / 255.0

        # “低彩度ほど強めに”効かせるスケール
        # 例: factor=1.3 のとき、Sが低いほど 1.3 に近づき、Sが高いほど 1.0 に近づく
        gamma = 1.6
        scale = 1.0 + (f - 1.0) * np.power(1.0 - s_arr, gamma)
        s2 = np.clip(s_arr * scale, 0.0, 1.0)

        s2_img = Image.fromarray((s2 * 255.0 + 0.5).astype(np.uint8)).convert('L')
        hsv2 = Image.merge('HSV', (h, s2_img, v))
        out_rgb = hsv2.convert('RGB')

        if has_alpha:
            out = Image.merge('RGBA', (*out_rgb.split(), alpha_ch))
        else:
            out = out_rgb
            if img.mode != 'RGB':
                # 元画像が RGB 以外なら近いモードへ戻す
                try:
                    out = out.convert(img.mode)
                except Exception as e:
                    _kana_silent_exc('core:L15515', e)
                    pass
        return out
    except Exception:
        # numpy が無い環境などでは従来の彩度倍率方式にフォールバック
        try:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Color(img)
            return enhancer.enhance(max(0.0, float(factor)))
        except Exception:
            return img


def _apply_shadow_highlight(img, shadow=0.22, highlight=0.18, content_mask=None):
    """Shadow/Highlight（暗部救済・白飛び抑え）を適用します。

    - shadow:   暗部を持ち上げる強さ（0.0〜1.0 目安）
    - highlight:明部を抑える強さ（0.0〜1.0 目安）
    - content_mask: 背景（余白）を除外したいときのマスク（L 推奨）。指定時はコンテンツ領域のみに適用します。
    """
    try:
        from PIL import Image
        import numpy as np

        s = max(0.0, min(1.0, float(shadow)))
        h = max(0.0, min(1.0, float(highlight)))
        if s <= 0.0 and h <= 0.0:
            return img

        # まず RGB（必要なら Alpha を退避）へ
        if img.mode == 'RGBA':
            rgb = img.convert('RGB')
            alpha_ch = img.split()[3]
            has_alpha = True
        else:
            rgb = img.convert('RGB') if img.mode != 'RGB' else img
            alpha_ch = None
            has_alpha = False

        # 輝度（Y）だけで処理（色相・彩度を崩しにくい）
        y, cb, cr = rgb.convert('YCbCr').split()
        yarr = np.asarray(y, dtype=np.float32) / 255.0

        # シャドウ/ハイライトの“効き方”を調整（値が大きいほど、より極端な領域だけに効く）
        g = 1.6
        sm = (1.0 - yarr) ** g
        hm = (yarr) ** g

        # 暗部は持ち上げ、明部は抑える（飽和しにくい形）
        y2 = yarr
        if s > 0.0:
            y2 = y2 + s * sm * (1.0 - y2)
        if h > 0.0:
            y2 = y2 - h * hm * y2

        y2 = np.clip(y2, 0.0, 1.0)
        y_img = Image.fromarray((y2 * 255.0).astype('uint8')).convert('L')
        out_rgb = Image.merge('YCbCr', (y_img, cb, cr)).convert('RGB')

        if has_alpha:
            out = Image.merge('RGBA', (*out_rgb.split(), alpha_ch))
        else:
            out = out_rgb
            if img.mode != 'RGB':
                try:
                    out = out.convert(img.mode)
                except Exception as e:
                    _kana_silent_exc('core:L15582', e)
                    pass
        # マスク指定があれば、コンテンツ領域だけに適用（余白は元のまま）
        if content_mask is not None:
            try:
                m = content_mask.convert('L') if content_mask.mode != 'L' else content_mask
                out = Image.composite(out, img, m)
            except Exception as e:
                _kana_silent_exc('core:L15590', e)
                pass
        return out
    except Exception:
        return img


def _apply_denoise(img, mode="light", strength=0.25, content_mask=None):
    """ノイズ除去（軽量）を適用します。

    - mode:
        - "off"   : 無効
        - "light" : 輝度だけを薄くスムージング（軽い / 破綻しにくい）
        - "median": 点ノイズ（塩胡椒）向け（少し丸くなります）
        - "edge"  : エッジ保護（低解像度で軽量バイラテラル）※やや重い
        - "heavy" : 強力（OpenCV があれば NLM / なければ強めバイラテラル）※重い
    - strength: 0.0〜1.0（効き具合）。目安：0.15〜0.40
    - content_mask: 背景（余白）を除外したいときのマスク（L 推奨）。指定時はコンテンツ領域のみに適用します。

    失敗した場合は元画像を返します。
    """
    try:
        from PIL import Image, ImageFilter
        import numpy as np

        m = str(mode).strip().lower()
        s = float(strength)

        if (not m) or m in ("off", "none", "0", "false") or s <= 0.0:
            return img

        # RGBA の場合は Alpha を保持して RGB で処理
        if img.mode == 'RGBA':
            rgb = img.convert('RGB')
            alpha_ch = img.split()[3]
            has_alpha = True
        else:
            rgb = img.convert('RGB') if img.mode != 'RGB' else img
            alpha_ch = None
            has_alpha = False

        # 輝度（Y）だけで処理（色相を汚しにくい）
        y, cb, cr = rgb.convert('YCbCr').split()

        if m == "median":
            # 強さに応じて 3 / 5 を切り替え（軽量）
            k = 3 if s < 0.60 else 5
            y2 = y.filter(ImageFilter.MedianFilter(size=k))
            y2 = Image.blend(y, y2, max(0.0, min(1.0, s)))
        elif m == "light":
            # 輝度の薄いスムージング（軽い）
            rad = 0.60 + 1.40 * max(0.0, min(1.0, s))  # 0.60〜2.00
            y_blur = y.filter(ImageFilter.GaussianBlur(radius=rad))
            y2 = Image.blend(y, y_blur, max(0.0, min(1.0, s)))
        elif m in ("heavy", "nlm"):
            # 強力版（できれば NLM、無ければ強めのエッジ保護フィルタ）
            w, h = y.size
            use_cv2 = False
            try:
                import cv2  # type: ignore
                use_cv2 = True
            except Exception:
                use_cv2 = False

            if use_cv2:
                # OpenCV の fastNlMeansDenoising を輝度だけに適用（強力。ただし環境依存）
                max_dim = 900  # 重いので適度に縮小してから処理
                scale = min(1.0, float(max_dim) / float(max(w, h)))
                mw = max(64, int(w * scale))
                mh = max(64, int(h * scale))
                y_small = y.resize((mw, mh), Image.BILINEAR)
                y_np = np.asarray(y_small, dtype=np.uint8)
                hval = int(5 + 25 * max(0.0, min(1.0, s)))  # 5〜30 目安
                hval = max(3, min(30, hval))
                try:
                    dst = cv2.fastNlMeansDenoising(y_np, None, h=hval, templateWindowSize=7, searchWindowSize=21)
                except Exception:
                    dst = cv2.fastNlMeansDenoising(y_np, None, h=hval)
                y_nlm = Image.fromarray(dst).convert('L').resize((w, h), Image.BICUBIC)
                y2 = Image.blend(y, y_nlm, max(0.0, min(1.0, s)))
            else:
                # OpenCV が無い場合のフォールバック：edge より少し高解像度＆（場合により）2パス
                max_dim = 420  # edge(320)より少し大きい
                scale = min(1.0, float(max_dim) / float(max(w, h)))
                mw = max(64, int(w * scale))
                mh = max(64, int(h * scale))

                y_small = y.resize((mw, mh), Image.BILINEAR)
                arr = np.asarray(y_small, dtype=np.float32)

                # パラメータ（強め）
                ss = max(0.0, min(1.0, s))
                sigma_s = 2.4 + 2.6 * ss   # 2.4〜5.0
                sigma_r = 20.0 + 60.0 * ss  # 20〜80
                passes = 2 if ss >= 0.35 else 1

                radw = int(max(1, round(sigma_s * 1.5)))
                pad = radw
                offs = list(range(-radw, radw + 1))
                xx, yy = np.meshgrid(offs, offs)
                spatial = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma_s * sigma_s)).astype(np.float32)

                for _ in range(passes):
                    arrp = np.pad(arr, pad, mode='edge')
                    center = arrp[pad:pad + mh, pad:pad + mw]
                    num = np.zeros_like(center, dtype=np.float32)
                    den = np.zeros_like(center, dtype=np.float32)
                    for jj, dy in enumerate(offs):
                        for ii, dx in enumerate(offs):
                            neigh = arrp[pad + dy:pad + dy + mh, pad + dx:pad + dx + mw]
                            range_w = np.exp(-((neigh - center) ** 2) / (2.0 * sigma_r * sigma_r)).astype(np.float32)
                            wgt = spatial[jj, ii] * range_w
                            num += wgt * neigh
                            den += wgt
                    arr = num / np.maximum(den, 1e-6)

                out = np.clip(arr, 0.0, 255.0).astype(np.uint8)
                y_small2 = Image.fromarray(out).convert('L')
                y_heavy = y_small2.resize((w, h), Image.BICUBIC)
                y2 = Image.blend(y, y_heavy, max(0.0, min(1.0, s)))

        elif m in ("edge", "bilateral"):
            # 低解像度で軽量バイラテラル（エッジ保護）
            w, h = y.size
            max_dim = 320  # 低解像度側の最大辺（速度優先）
            scale = min(1.0, float(max_dim) / float(max(w, h)))
            mw = max(64, int(w * scale))
            mh = max(64, int(h * scale))

            y_small = y.resize((mw, mh), Image.BILINEAR)
            arr = np.asarray(y_small, dtype=np.float32)

            # パラメータ（ほどよく）
            # sigma_s: 空間の広がり（小さいほどエッジを守りやすい）
            # sigma_r: 明るさ差の許容（小さいほどエッジを守りやすい）
            sigma_s = 2.0
            sigma_r = 18.0

            radw = int(max(1, round(sigma_s * 1.5)))  # だいたい 3〜4
            pad = radw
            arrp = np.pad(arr, pad, mode='edge')
            center = arrp[pad:pad + mh, pad:pad + mw]

            # 空間重み（固定）
            offs = list(range(-radw, radw + 1))
            xx, yy = np.meshgrid(offs, offs)
            spatial = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma_s * sigma_s)).astype(np.float32)

            num = np.zeros_like(center, dtype=np.float32)
            den = np.zeros_like(center, dtype=np.float32)

            for j, dy in enumerate(offs):
                for i, dx in enumerate(offs):
                    neigh = arrp[pad + dy:pad + dy + mh, pad + dx:pad + dx + mw]
                    # 画素差重み（エッジ保護）
                    range_w = np.exp(-((neigh - center) ** 2) / (2.0 * sigma_r * sigma_r)).astype(np.float32)
                    wgt = spatial[j, i] * range_w
                    num += wgt * neigh
                    den += wgt

            out = num / np.maximum(den, 1e-6)
            out = np.clip(out, 0.0, 255.0).astype(np.uint8)
            y_small2 = Image.fromarray(out).convert('L')

            # 元解像度へ戻してブレンド
            y_edge = y_small2.resize((w, h), Image.BICUBIC)
            y2 = Image.blend(y, y_edge, max(0.0, min(1.0, s)))
        else:
            return img

        out_rgb = Image.merge('YCbCr', (y2, cb, cr)).convert('RGB')
        if has_alpha:
            out = Image.merge('RGBA', (*out_rgb.split(), alpha_ch))
        else:
            out = out_rgb
            if img.mode != 'RGB':
                try:
                    out = out.convert(img.mode)
                except Exception as e:
                    _kana_silent_exc('core:L15769', e)
                    pass
        # マスク指定があれば、コンテンツ領域だけに適用（余白は元のまま）
        if content_mask is not None:
            try:
                mimg = content_mask.convert('L') if content_mask.mode != 'L' else content_mask
                out = Image.composite(out, img, mimg)
            except Exception as e:
                _kana_silent_exc('core:L15777', e)
                pass
        return out
    except Exception:
        return img

def _apply_dehaze(img, amount=0.10, radius=24, content_mask=None):
    """Dehaze（霞み抜き）を適用します。

    霧/霞/白っぽい眠さでコントラストが低い素材を、自然な範囲で“抜き”ます。
    Dark Channel Prior を軽量化した方式（低解像度で推定→復元）です。

    - amount: 0.0〜1.0（強さ）。目安：0.05〜0.20
    - radius: 解析半径（px）。大きいほど“大きな霞”に効きます（推奨 16〜40）
    - content_mask: 背景（余白）を除外したいときのマスク（L 推奨）。指定時はコンテンツ領域のみに適用します。
    """
    try:
        from PIL import Image, ImageFilter
        import numpy as np

        a = max(0.0, min(1.0, float(amount)))
        if a <= 0.0:
            return img

        # まず RGB（必要なら Alpha を退避）へ
        if img.mode == 'RGBA':
            rgb = img.convert('RGB')
            alpha_ch = img.split()[3]
            has_alpha = True
        else:
            rgb = img.convert('RGB') if img.mode != 'RGB' else img
            alpha_ch = None
            has_alpha = False

        w, h = rgb.size
        if w <= 0 or h <= 0:
            return img

        # 推定は低解像度で（高速化）
        ds = 6 if max(w, h) >= 2000 else 4
        sw = max(64, w // ds)
        sh = max(64, h // ds)
        small = rgb.resize((sw, sh), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0  # (sh, sw, 3)

        # ミニマムフィルタのカーネルサイズ（radius を縮小に合わせて調整）
        r = max(1, int(float(radius)))
        kk = max(1, int(r / ds))
        k = max(3, kk * 2 + 1)
        k = min(31, k)
        if (k % 2) == 0:
            k += 1

        # --- Atmospheric Light（A）推定 ---
        # 暗チャンネル（min(R,G,B)）を作って min-filter
        dark = np.min(arr, axis=2)
        dark_img = Image.fromarray(np.clip(dark * 255.0, 0.0, 255.0).astype('uint8')).convert('L')
        dark_min = dark_img.filter(ImageFilter.MinFilter(size=k))
        dark_min_arr = np.asarray(dark_min, dtype=np.float32) / 255.0

        flat = dark_min_arr.reshape(-1)
        n = flat.size
        top_n = max(1, int(n * 0.001))  # 上位 0.1%
        # 上位の候補から、RGB 輝度が最大の点を A として採用
        idxs = np.argpartition(flat, -top_n)[-top_n:]
        rgb_flat = arr.reshape(-1, 3)
        cand = rgb_flat[idxs]
        brightness = cand.sum(axis=1)
        A = cand[int(brightness.argmax())]
        A = np.clip(A, 0.05, 1.0)  # 0 除算回避

        # --- Transmission（t）推定（dark channel of I/A） ---
        norm = np.clip(arr / A, 0.0, 1.0)
        dark_norm = np.min(norm, axis=2)
        dark_norm_img = Image.fromarray(np.clip(dark_norm * 255.0, 0.0, 255.0).astype('uint8')).convert('L')
        dark_norm_min = dark_norm_img.filter(ImageFilter.MinFilter(size=k))
        dark_norm_min_arr = np.asarray(dark_norm_min, dtype=np.float32) / 255.0

        omega = 0.95
        t = 1.0 - omega * dark_norm_min_arr
        t = np.clip(t, 0.0, 1.0)

        # t をフル解像度へ
        t_img = Image.fromarray(np.clip(t * 255.0, 0.0, 255.0).astype('uint8')).convert('L')
        t_img = t_img.resize((w, h), Image.BILINEAR)
        t_full = (np.asarray(t_img, dtype=np.float32) / 255.0)

        # 復元（J = (I-A)/max(t,t0) + A）
        t0 = 0.10
        t_full = np.maximum(t_full, t0)

        full = (np.asarray(rgb, dtype=np.float32) / 255.0)
        J = (full - A) / t_full[..., None] + A
        J = np.clip(J, 0.0, 1.0)

        # 強度はブレンドで調整（安全）
        out_arr = full * (1.0 - a) + J * a
        out_rgb = Image.fromarray(np.clip(out_arr * 255.0, 0.0, 255.0).astype('uint8')).convert('RGB')

        if has_alpha:
            out = Image.merge('RGBA', (*out_rgb.split(), alpha_ch))
        else:
            out = out_rgb
            if img.mode != 'RGB':
                try:
                    out = out.convert(img.mode)
                except Exception as e:
                    _kana_silent_exc('core:L15884', e)
                    pass
        # マスク指定があれば、コンテンツ領域だけに適用（余白は元のまま）
        if content_mask is not None:
            try:
                m = content_mask.convert('L') if content_mask.mode != 'L' else content_mask
                out = Image.composite(out, img, m)
            except Exception as e:
                _kana_silent_exc('core:L15892', e)
                pass
        return out
    except Exception:
        return img

# --- LUT（.cube）カラーグレーディング --------------------------------
# 3D LUT（.cube）を読み込み、画像へ適用します（色の世界観を切り替える用途）。
# 実装方針：
# - 3D LUT の trilinear 補間（見た目優先）
# - 大きな画像でもメモリが爆発しにくいよう、縦方向に分割して処理
# - content_mask が渡された場合、コンテンツ領域のみに適用（余白はそのまま）
_LUT_CACHE = {}  # {path: {"mtime": float, "lut": dict}}

def _load_cube_3d_lut(file_path: str) -> dict:
    '''3D LUT（.cube）を読み込みます。失敗時は例外を投げます。'''
    import os
    import numpy as np

    fp = str(file_path or "").strip()
    if not fp:
        raise ValueError("LUT_FILE is empty")

    if not os.path.exists(fp):
        raise FileNotFoundError(fp)

    size = None
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    data = []

    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue

            parts = s.split()
            if not parts:
                continue

            key = parts[0].upper()
            if key == "TITLE":
                # TITLE "xxx" は無視
                continue
            if key == "LUT_3D_SIZE" and len(parts) >= 2:
                try:
                    size = int(parts[1])
                except Exception:
                    raise ValueError("Invalid LUT_3D_SIZE")
                continue
            if key == "DOMAIN_MIN" and len(parts) >= 4:
                try:
                    domain_min = [float(parts[1]), float(parts[2]), float(parts[3])]
                except Exception as e:
                    _kana_silent_exc('core:L15949', e)
                    pass
                continue
            if key == "DOMAIN_MAX" and len(parts) >= 4:
                try:
                    domain_max = [float(parts[1]), float(parts[2]), float(parts[3])]
                except Exception as e:
                    _kana_silent_exc('core:L15955', e)
                    pass
                continue

            # 3値（R G B）データ
            if len(parts) >= 3:
                try:
                    r = float(parts[0]); g = float(parts[1]); b = float(parts[2])
                except Exception as e:
                    _kana_silent_exc('core:L15963', e)
                    continue
                data.append((r, g, b))

    if size is None:
        raise ValueError("LUT_3D_SIZE not found")

    expected = size * size * size
    if len(data) != expected:
        raise ValueError(f"Invalid LUT data length: {len(data)} (expected {expected})")

    table = np.asarray(data, dtype=np.float32).reshape((size, size, size, 3))
    # 念のため範囲を丸める（.cube の中には 0..1 を超える値もあり得る）
    table = np.clip(table, 0.0, 1.0)

    return {
        "size": int(size),
        "table": table,  # shape: (R, G, B, 3)
        "domain_min": np.asarray(domain_min, dtype=np.float32),
        "domain_max": np.asarray(domain_max, dtype=np.float32),
    }

def _get_cached_cube_lut(file_path: str) -> dict:
    '''キャッシュ付きで 3D LUT を取得します。失敗時は例外を投げます。'''
    import os
    fp = str(file_path or "").strip()
    if not fp:
        raise ValueError("LUT_FILE is empty")
    mtime = os.path.getmtime(fp)
    ent = _LUT_CACHE.get(fp)
    if ent and abs(ent.get("mtime", -1.0) - mtime) < 1e-6:
        return ent["lut"]
    lut = _load_cube_3d_lut(fp)
    _LUT_CACHE[fp] = {"mtime": float(mtime), "lut": lut}
    return lut

def _apply_lut_cube(img, lut_file: str, strength: float = 0.30, content_mask=None):
    '''3D LUT（.cube）を画像へ適用します。失敗時は元画像を返します。'''
    try:
        from PIL import Image
        import numpy as np

        st = float(strength)
        st = max(0.0, min(1.0, st))
        if st <= 0.0:
            return img

        lut = _get_cached_cube_lut(lut_file)
        size = int(lut["size"])
        table = lut["table"]
        dmin = lut["domain_min"]
        dmax = lut["domain_max"]
        drange = np.maximum(dmax - dmin, 1e-6)

        # まず RGB（必要なら Alpha を退避）
        if img.mode == "RGBA":
            rgb = img.convert("RGB")
            alpha_ch = img.split()[3]
            has_alpha = True
        else:
            rgb = img.convert("RGB") if img.mode != "RGB" else img
            alpha_ch = None
            has_alpha = False

        w, h = rgb.size
        out = np.empty((h, w, 3), dtype=np.uint8)

        # メモリ節約のため縦方向に分割（目安：~60万ピクセル/チャンク）
        target_pixels = 600_000
        chunk_h = max(32, min(h, max(32, int(target_pixels // max(1, w)))))

        def _apply_chunk(arr_u8: np.ndarray) -> np.ndarray:
            # arr_u8: (ch, w, 3) uint8
            orig = arr_u8.astype(np.float32) / 255.0  # 0..1
            x = (orig - dmin) / drange
            x = np.clip(x, 0.0, 1.0)

            # 0..(size-1) にスケール
            idx = x * float(size - 1)
            r = idx[..., 0]; g = idx[..., 1]; b = idx[..., 2]

            r0 = np.floor(r).astype(np.int32)
            g0 = np.floor(g).astype(np.int32)
            b0 = np.floor(b).astype(np.int32)

            r1 = np.clip(r0 + 1, 0, size - 1)
            g1 = np.clip(g0 + 1, 0, size - 1)
            b1 = np.clip(b0 + 1, 0, size - 1)

            dr_ = (r - r0).astype(np.float32)
            dg_ = (g - g0).astype(np.float32)
            db_ = (b - b0).astype(np.float32)

            # 8頂点
            c000 = table[r0, g0, b0]
            c001 = table[r0, g0, b1]
            c010 = table[r0, g1, b0]
            c011 = table[r0, g1, b1]
            c100 = table[r1, g0, b0]
            c101 = table[r1, g0, b1]
            c110 = table[r1, g1, b0]
            c111 = table[r1, g1, b1]

            db3 = db_[..., None]
            dg3 = dg_[..., None]
            dr3 = dr_[..., None]

            c00 = c000 * (1.0 - db3) + c001 * db3
            c01 = c010 * (1.0 - db3) + c011 * db3
            c10 = c100 * (1.0 - db3) + c101 * db3
            c11 = c110 * (1.0 - db3) + c111 * db3

            c0 = c00 * (1.0 - dg3) + c01 * dg3
            c1 = c10 * (1.0 - dg3) + c11 * dg3

            mapped = c0 * (1.0 - dr3) + c1 * dr3  # 0..1

            # ブレンド（元画像は domain 変換前の orig）
            out_f = orig * (1.0 - st) + mapped * st
            out_u8 = np.clip(out_f * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
            return out_u8

        y0 = 0
        while y0 < h:
            y1 = min(h, y0 + chunk_h)
            crop = rgb.crop((0, y0, w, y1))
            arr = np.asarray(crop, dtype=np.uint8)
            out[y0:y1, :, :] = _apply_chunk(arr)
            y0 = y1

        out_rgb = Image.fromarray(out).convert('RGB')
        if has_alpha:
            out_img = Image.merge("RGBA", (*out_rgb.split(), alpha_ch))
        else:
            out_img = out_rgb
            if img.mode != "RGB":
                try:
                    out_img = out_img.convert(img.mode)
                except Exception as e:
                    _kana_silent_exc('core:L16101', e)
                    pass
        # content_mask があれば、コンテンツ領域だけ適用（余白は元のまま）
        if content_mask is not None:
            try:
                m = content_mask.convert("L") if content_mask.mode != "L" else content_mask
                out_img = Image.composite(out_img, img, m)
            except Exception as e:
                _kana_silent_exc('core:L16109', e)
                pass
        return out_img
    except Exception:
        return img


def _apply_split_tone(img, shadow_hue=220.0, shadow_strength=0.06,
                      highlight_hue=35.0, highlight_strength=0.05,
                      balance=0.0, content_mask=None):
    """スプリットトーン（影・ハイライトに色味）を適用します。

    - shadow_hue / highlight_hue: 色相（0〜360）
    - shadow_strength / highlight_strength: 強さ（0.0〜1.0）
    - balance: -1.0〜1.0（0.0 で中間。負→影寄り / 正→明部寄り）
    - content_mask: 背景（余白）を除外したいときのマスク（L 推奨）
    """
    try:
        from PIL import Image
        import numpy as np
        import colorsys

        shs = float(max(0.0, min(1.0, shadow_strength)))
        his = float(max(0.0, min(1.0, highlight_strength)))
        if shs <= 0.0 and his <= 0.0:
            return img

        # 元画像をRGBへ（必要ならαを保持）
        if img.mode == "RGBA":
            base_rgb = img.convert("RGB")
            alpha_ch = img.split()[3]
            has_alpha = True
        else:
            base_rgb = img.convert("RGB") if img.mode != "RGB" else img
            alpha_ch = None
            has_alpha = False

        w, h = base_rgb.size
        arr = np.asarray(base_rgb, dtype=np.float32) / 255.0  # (H,W,3)

        # 輝度（0..1）
        luma = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722

        # バランス：ピボット（中間点）を動かす
        b = float(max(-1.0, min(1.0, balance)))
        pivot = 0.5 + (b * 0.25)  # 0.25〜0.75 に制限して極端を避ける
        pivot = float(max(0.05, min(0.95, pivot)))

        # 影・明部の重み（滑らか：指数でソフト化）
        # shadow: luma < pivot のとき 0..1
        sw = np.clip((pivot - luma) / max(pivot, 1e-6), 0.0, 1.0) ** 1.6
        # highlight: luma > pivot のとき 0..1
        hw = np.clip((luma - pivot) / max(1.0 - pivot, 1e-6), 0.0, 1.0) ** 1.6

        sw = sw * shs
        hw = hw * his

        # 色相→RGB（s=1,v=1）を“薄くブレンド”して色味だけ乗せる
        sh = float(shadow_hue) % 360.0
        hh = float(highlight_hue) % 360.0
        sh_rgb = np.array(colorsys.hsv_to_rgb(sh / 360.0, 1.0, 1.0), dtype=np.float32)
        hh_rgb = np.array(colorsys.hsv_to_rgb(hh / 360.0, 1.0, 1.0), dtype=np.float32)

        out = arr * (1.0 - sw[..., None] - hw[..., None]) + sh_rgb[None, None, :] * sw[..., None] + hh_rgb[None, None, :] * hw[..., None]
        out = np.clip(out, 0.0, 1.0)
        out_img = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)).convert('RGB')

        if has_alpha:
            out_img = Image.merge("RGBA", (*out_img.split(), alpha_ch))

        # 背景（余白）を守る
        if content_mask is not None:
            try:
                m = content_mask.convert("L") if content_mask.mode != "L" else content_mask
                out_img = Image.composite(out_img, img, m)
            except Exception as e:
                _kana_silent_exc('core:L16185', e)
                pass
        # 元モードへ戻す（可能なら）
        if img.mode not in ("RGB", "RGBA"):
            try:
                out_img = out_img.convert(img.mode)
            except Exception as e:
                _kana_silent_exc('core:L16192', e)
                pass
        return out_img
    except Exception:
        return img

def _apply_tonecurve(img, mode="film", strength=0.35, content_mask=None):
    """トーンカーブ（階調）を適用します（輝度のみ）。

    - mode:
        - "film"     : 影をほんのり持ち上げ、ハイライトを丸める“フィルムっぽい”階調
        - "liftgamma": 影〜中間をやや持ち上げる（ふんわり系）
        - "custom"   : 現状は film ベース（将来拡張用）
    - strength: 0.0〜1.0（効き具合）。0.20〜0.45 あたりが上品です。
    - content_mask: 背景（余白）を除外したいときのマスク（L 推奨）。指定時はコンテンツ領域のみに適用します。

    失敗した場合は元画像を返します。
    """
    try:
        from PIL import Image

        m = str(mode).strip().lower() if mode is not None else "film"
        if m not in ("film", "liftgamma", "custom"):
            m = "film"

        s = float(strength)
        if s <= 0.0:
            return img
        s = max(0.0, min(1.0, s))

        def _curve_base(x: float) -> float:
            # x は 0.0〜1.0
            if m == "liftgamma":
                t = 0.25     # toe 開始
                sp = 0.85    # shoulder 開始
                lift = 0.06  # 影持ち上げ量（t地点の上げ）
                roll = 0.01  # ハイライト丸め量（sp地点の下げ）
                toe_g = 0.70
                sh_g = 0.90
            else:
                # film / custom（現状は film ベース）
                t = 0.20
                sp = 0.80
                lift = 0.03
                roll = 0.03
                toe_g = 0.75
                sh_g = 0.75

            ty = max(0.0, min(1.0, t + lift))
            sy = max(0.0, min(1.0, sp - roll))

            if x <= t:
                u = x / (t if t > 1e-6 else 1e-6)
                y = (u ** toe_g) * ty
            elif x >= sp:
                u = (1.0 - x) / ((1.0 - sp) if (1.0 - sp) > 1e-6 else 1e-6)
                y = 1.0 - (u ** sh_g) * (1.0 - sy)
            else:
                # 中間は線形で繋いで、極端なクセが出ないようにする
                k = (sy - ty) / (sp - t)
                y = ty + (x - t) * k

            return max(0.0, min(1.0, y))

        # 0〜255 の LUT を作る（identity と base curve を strength でブレンド）
        lut = []
        for i in range(256):
            x = i / 255.0
            yb = _curve_base(x)
            y = (1.0 - s) * x + s * yb
            lut.append(int(max(0, min(255, y * 255.0 + 0.5))))

        # まずは RGB（必要なら Alpha を退避）へ
        if img.mode == "RGBA":
            rgb = img.convert("RGB")
            alpha_ch = img.split()[3]
            has_alpha = True
        else:
            rgb = img.convert("RGB") if img.mode != "RGB" else img
            alpha_ch = None
            has_alpha = False

        y, cb, cr = rgb.convert("YCbCr").split()
        y2 = y.point(lut)
        out_rgb = Image.merge("YCbCr", (y2, cb, cr)).convert("RGB")

        if has_alpha:
            out = Image.merge("RGBA", (*out_rgb.split(), alpha_ch))
        else:
            out = out_rgb
            if img.mode != "RGB":
                try:
                    out = out.convert(img.mode)
                except Exception as e:
                    _kana_silent_exc('core:L16286', e)
                    pass
        # マスク指定があれば、コンテンツ領域だけに適用（余白は元のまま）
        if content_mask is not None:
            try:
                msk = content_mask.convert("L") if content_mask.mode != "L" else content_mask
                out = Image.composite(out, img, msk)
            except Exception as e:
                _kana_silent_exc('core:L16294', e)
                pass
        return out
    except Exception:
        return img

def _arrange_by_tempo(images, mode="alt"):
    if not images: return images
    scored = [(p, _estimate_busyness_fast(p)) for p in images]
    scored.sort(key=lambda x: x[1])
    n = len(scored)
    mid = n//2
    quiet = [p for p,_ in scored[:mid]]
    busy  = [p for p,_ in scored[mid:]][::-1]
    out = []
    if mode == "2:1":
        while quiet or busy:
            if busy:  out.append(busy.pop(0))
            if busy:  out.append(busy.pop(0))
            if quiet: out.append(quiet.pop(0))
    else:
        turn = 0
        while quiet or busy:
            if turn % 2 == 0:
                if busy:  out.append(busy.pop(0))
                elif quiet: out.append(quiet.pop(0))
            else:
                if quiet: out.append(quiet.pop(0))
                elif busy: out.append(busy.pop(0))
            turn += 1
    return out

# ---- テンポ整列用ヘルパー群 --------------------------------------------
def _estimate_busyness_fast(path):
    """近傍差分の平均＋ファイルサイズの微加点で “賑やかさ” を推定（軽量）。

    注意: open_image_safe() は Image.open を返すため、ここでは with で確実に close します。
    """
    try:
        from PIL import Image
        import numpy as np
        with open_image_safe(path) as _im0:
            im = _im0.convert("L")
            im.thumbnail((96, 96), Image.BILINEAR)
            arr = np.array(im, dtype=np.float32)
            try:
                im.close()
            except Exception as e:
                _warn_exc_once(e)
                pass
        gx = float(np.abs(arr[:, 1:] - arr[:, :-1]).mean()) if arr.shape[1] > 1 else 0.0
        gy = float(np.abs(arr[1:, :] - arr[:-1, :]).mean()) if arr.shape[0] > 1 else 0.0
        g = (gx + gy) * 0.5
        try:
            g += (_imgref_size(path) / (1024*1024)) * 0.2
        except Exception as e:
            _warn_exc_once(e)
            pass
        return float(g)
    except Exception:
        return 0.0


def _arrange_by_tempo(images, mode="alt"):
    """[pre/post 共通] “賑やか/静か”を交互(alt)または2:1で並べる。"""
    if not images: return images
    scored = [(p, _estimate_busyness_fast(p)) for p in images]
    scored.sort(key=lambda x: x[1])
    n = len(scored); mid = n//2
    quiet = [p for p,_ in scored[:mid]]
    busy  = [p for p,_ in scored[mid:]][::-1]
    out = []
    if mode == "2:1":
        while quiet or busy:
            if busy:  out.append(busy.pop(0))
            if busy:  out.append(busy.pop(0))
            if quiet: out.append(quiet.pop(0))
    else:
        turn = 0
        while quiet or busy:
            if turn % 2 == 0:
                if busy:  out.append(busy.pop(0))
                elif quiet: out.append(quiet.pop(0))
            else:
                if quiet: out.append(quiet.pop(0))
                elif busy: out.append(busy.pop(0))
            turn += 1
    return out

def _tempo_postpass(seq, mode="alt", k=3, score_fn=None):
    """[blend] 最終順序を大きく崩さず、窓幅k内で “交互” を満たすよう局所スワップ。"""
    if not seq: return seq
    if score_fn is None:
        score_fn = _estimate_busyness_fast
    s = [(p, score_fn(p)) for p in seq]
    med = sorted(x for _,x in s)[len(s)//2] if len(s)>2 else 0.0
    flags = [1 if x>=med else 0 for _,x in s]
    def want(turn):
        if mode=="2:1":
            return 1 if (turn%3 in (0,1)) else 0
        return 1 if (turn%2==0) else 0
    out = list(seq); f = list(flags); n = len(out); i = 0
    while i < n:
        if f[i] != want(i):
            j = i+1; lim = min(n, i+1+max(1,int(k)))
            found = -1
            while j < lim:
                if f[j] == want(i):
                    found = j; break
                j += 1
            if found != -1:
                out[i], out[found] = out[found], out[i]
                f[i],   f[found]   = f[found],   f[i]
        i += 1
    return out

def _tempo_apply(images):
    """描画直前（post/blend）用の安全フック。"""
    try:
        if not globals().get("ARRANGE_TEMPO_ENABLE", False):
            return images
        st = str(globals().get("ARRANGE_TEMPO_STAGE", "pre")).lower()
        if st == "post":
            return _arrange_by_tempo(images, globals().get("ARRANGE_TEMPO_MODE","alt"))
        if st == "blend":
            return _tempo_postpass(images, globals().get("ARRANGE_TEMPO_MODE","alt"),
                                   int(globals().get("ARRANGE_TEMPO_WINDOW",3)))
    except Exception as e:
        _warn_exc_once(e)
        pass
    return images

# -----------------------------------------------------------------------------
# 共有フェイス検出ヘルパー
# `_cover_square_face_focus` と `_cover_rect_face_focus` で重複していた OpenCV ベースの検出処理を
# ここに集約します。最良の顔（あれば）、上半身領域（任意）、サリエンシー注目点（任意）を返します。
# また、除外理由などの統計を `_FDBG`／`_FDBG2` に加算します。
def _get_focus_candidates(im: Image.Image, src_path=None) -> dict:
    """フォーカスクロップ用の候補（顔/上半身/サリエンシー）を検出して返します。

    戻り値（dict）:
      - face     : ("frontal"|"profile", x, y, w, h) または None
      - upper    : (x, y, w, h) または None（FACE_FOCUS_USE_UPPER=True のときのみ）
      - saliency : (cx, cy) または None（FACE_FOCUS_USE_SALIENCY=True のときのみ）

    検出中に `_FDBG`／`_FDBG2`（除外理由・目検証の成否など）の統計を更新します。
    ただし、実際にどの候補を採用したか（frontal/profile の採用カウントなど）は
    呼び出し側で更新します。

    OpenCV が使えない/失敗した場合は例外を投げず、各要素は None のまま返します。
    """
    # 戻り値を初期化
    result = {"face": None, "upper": None, "person": None, "saliency": None}
    # 永続キャッシュ（顔/上半身）を先に参照して、重いカスケード検出をできるだけ省略
    try:
        if src_path is not None and bool(globals().get("FACE_CACHE_ENABLE", True)) and bool(globals().get("DHASH_CACHE_ENABLE", True)):
            c = _face_cache_get(src_path)
            if isinstance(c, dict):
                if c.get("face") is not None:
                    result["face"] = c.get("face")
                if c.get("upper") is not None:
                    result["upper"] = c.get("upper")
                if c.get("person") is not None:
                    result["person"] = c.get("person")
                if c.get("saliency") is not None:
                    result["saliency"] = c.get("saliency")
    except Exception as e:
        _warn_exc_once(e)
        pass
    need_face = (result.get("face") is None)
    need_upper = bool(globals().get("FACE_FOCUS_USE_UPPER", False)) and (result.get("upper") is None)
    need_person = bool(globals().get("FACE_FOCUS_USE_PERSON", True)) and (result.get("person") is None)
    need_saliency = bool(globals().get("FACE_FOCUS_USE_SALIENCY", True)) and (result.get("saliency") is None)
    try:
        import numpy as _np  # type: ignore
        import cv2  # type: ignore
        # 解析を揃えるため RGB 化して ndarray へ
        rgb = _np.array(im.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # 検出が安定しやすいようコントラストを整える（ヒストグラム平坦化）
        gray = cv2.equalizeHist(gray)
        ih_cv, iw_cv = gray.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 検出された顔候補から「使えるもの」だけに絞り込む
        def pick_faces(casc) -> list:
            if not need_face or casc is None:
                return []
            # 顔検出の最小サイズ（短辺の何割まで許容するか）は、AI/アニメ絵で重要です。
            # 既定（0.08）だと顔が小さめの絵で検出できないことがあるため、
            # FACE_FOCUS_MIN_FACE_FRAC で上書きできるようにします。
            try:
                _min_frac = float(globals().get('FACE_FOCUS_MIN_FACE_FRAC', 0.08))
            except Exception:
                _min_frac = 0.08
            _min_side = int(min(gray.shape[0], gray.shape[1]))
            _min_px = max(24, int(_min_frac * _min_side))
            faces = casc.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=(_min_px, _min_px),
            )
            valid: list = []
            for (x, y, w, h) in faces if faces is not None else []:
                # アスペクト比が極端なものは除外
                ratio = w / float(h + 1e-6)
                # 顔候補の縦横比（w/h）。誤検出（極端に縦長/横長）を弾くためのチェック。
                # 許容範囲は FACE_FOCUS_FACE_RATIO_MIN／MAX で調整できます。
                rmin = float(globals().get("FACE_FOCUS_FACE_RATIO_MIN", 0.65))
                rmax = float(globals().get("FACE_FOCUS_FACE_RATIO_MAX", 1.60))
                if ratio < rmin or ratio > rmax:
                    _FDBG["reject_ratio"] += 1
                    continue
                # 画面の下側すぎる顔は除外（ALLOW_LOW=True のときは許可）
                cy = y + h / 2.0
                # 画面の下側で検出された「顔っぽいもの」は誤検出が多いので、上側だけ探索するための制限。
                # FACE_FOCUS_TOP_FRAC は「上から何割までを探索するか」の目安（0.70=上70%）。
                if (
                    not globals().get("FACE_FOCUS_ALLOW_LOW", False)
                    and (cy > float(globals().get("FACE_FOCUS_TOP_FRAC", 0.70)) * ih_cv)  # 顔中心が下に寄りすぎる場合はNG（上側のみ探索）
                ):
                    _FDBG2["low_reject"] += 1
                    continue
                # 目検証（スコアリング用）。
                # アニメ/AI絵では「顔はあるが目カスケードが外れる」ケースが多いため、
                # strict_eyes=False のときは“拒否”ではなく「目の数（eyes_cnt）」として使います。
                strict = bool(globals().get("FACE_FOCUS_STRICT_EYES", False))
                eyes_ok = True
                eyes_cnt = 0
                try:
                    eyes_ok, eyes_cnt = _kana_face_eye_verify(gray, (x, y, w, h))
                except Exception:
                    eyes_ok, eyes_cnt = (True, 0)
                if strict and not eyes_ok:
                    _FDBG2["eyes_ng"] += 1
                    continue
                if strict:
                    _FDBG2["eyes_ok"] += 1
                valid.append((x, y, w, h, int(eyes_cnt)))
            return valid


        # AI顔検出（YuNet）
        def _resolve_model_path(p: str) -> str:
            try:
                if os.path.isabs(p):
                    return p
                base = os.path.dirname(__file__)
                return os.path.join(base, p)
            except Exception:
                return p

        def pick_ai_faces_yunet() -> list:
            # 戻り値: [(x,y,w,h,eyes_cnt,conf)]（eyes_cnt は2固定扱い）

            # 1回だけAI状態を表示（モデルパス/FaceDetectorYN有無）
            try:
                if bool(globals().get("FACE_FOCUS_AI_ENABLE", False)) and (not bool(globals().get("_FACE_FOCUS_AI_STATUS_SHOWN", False))):
                    globals()["_FACE_FOCUS_AI_STATUS_SHOWN"] = True
                    model = str(globals().get("FACE_FOCUS_YUNET_MODEL", "face_detection_yunet_2023mar.onnx"))
                    model_path = _resolve_model_path(model)
                    has_yn = hasattr(cv2, "FaceDetectorYN")
                    note(f"AI: enabled=True | backend='{str(globals().get('FACE_FOCUS_AI_BACKEND','yunet'))}' | FaceDetectorYN={has_yn} | model='{model_path}' | exists={os.path.exists(model_path)}")
            except Exception as e:
                _kana_silent_exc('core:L16560', e)
                pass
            if (not need_face) or (not bool(globals().get("FACE_FOCUS_AI_ENABLE", False))):
                return []
            if not hasattr(cv2, "FaceDetectorYN"):
                try:
                    _FDBG2["ai_face_ng"] += 1
                except Exception as e:
                    _kana_silent_exc('core:L16567', e)
                    pass
                if bool(globals().get("FACE_FOCUS_DEBUG", False)):
                    _warn_exc_once(RuntimeError("FACE_FOCUS_AI_ENABLE=True ですが、cv2.FaceDetectorYN が見つかりません（opencv-contrib-python が必要かもしれません）。"))
                return []
            try:
                model = str(globals().get("FACE_FOCUS_YUNET_MODEL", "face_detection_yunet_2023mar.onnx"))
                model_path = _resolve_model_path(model)
                if not os.path.exists(model_path):
                    try:
                        _FDBG2["ai_face_ng"] += 1
                    except Exception as e:
                        _kana_silent_exc('core:L16578', e)
                        pass
                    if bool(globals().get("FACE_FOCUS_DEBUG", False)):
                        _warn_exc_once(FileNotFoundError(f"YuNetモデルが見つかりません: {model_path}"))
                    return []

                score = float(globals().get("FACE_FOCUS_YUNET_SCORE", 0.60))
                nms = float(globals().get("FACE_FOCUS_YUNET_NMS", 0.30))
                topk = int(globals().get("FACE_FOCUS_YUNET_TOPK", 50))

                fd = cv2.FaceDetectorYN.create(model_path, "", (iw_cv, ih_cv), score, nms, topk)
                fd.setInputSize((iw_cv, ih_cv))
                _, faces = fd.detect(bgr)

                if faces is None or len(faces) == 0:
                    try:
                        _FDBG2["ai_face_ng"] += 1
                    except Exception as e:
                        _kana_silent_exc('core:L16595', e)
                        pass
                    return []

                out = []
                for row in faces:
                    x, y, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    conf = float(row[-1]) if len(row) >= 15 else 0.0
                    if w < 8 or h < 8:
                        continue
                    out.append((int(x), int(y), int(w), int(h), 2, conf))

                if out:
                    try:
                        _FDBG2["ai_face_ok"] += 1
                    except Exception as e:
                        _kana_silent_exc('core:L16610', e)
                        pass
                else:
                    try:
                        _FDBG2["ai_face_ng"] += 1
                    except Exception as e:
                        _kana_silent_exc('core:L16615', e)
                        pass
                return out
            except Exception as e:
                try:
                    _FDBG2["ai_face_ng"] += 1
                except Exception as e:
                    _kana_silent_exc('core:L16621', e)
                    pass
                _warn_exc_once(e)
                return []


        # AI顔検出（YOLOv8 アニメ顔）
        def pick_ai_faces_yolo() -> list:
            # 戻り値: [(x,y,w,h,eyes_cnt,conf)]
            if (not need_face) or (not bool(globals().get("FACE_FOCUS_AI_ENABLE", False))):
                return []

            # （センパイ案）Ultralytics の冗長出力を抑制（import前に設定）
            try:
                import os as _os, logging as _logging
                if not bool(globals().get("FACE_FOCUS_DEBUG", False)):
                    _os.environ["YOLO_VERBOSE"] = "False"
                    _logging.getLogger("ultralytics").setLevel(_logging.WARNING)
            except Exception as e:
                _kana_silent_exc('core:L16639', e)
                pass
            try:
                from ultralytics import YOLO  # type: ignore
                try:
                    from ultralytics.utils import SETTINGS  # type: ignore
                    if not bool(globals().get("FACE_FOCUS_DEBUG", False)):
                        try:
                            SETTINGS.update({"verbose": False})
                        except Exception as e:
                            _kana_silent_exc('core:L16649', e)
                            pass
                except Exception as e:
                    _kana_silent_exc('core:L16651', e)
                    pass
            except Exception as e:
                _warn_exc_once(e)
                try:
                    _FDBG2["ai_face_ng"] += 1
                except Exception as e:
                    _kana_silent_exc('core:L16657', e)
                    pass
                return []

            model_path = str(globals().get("FACE_FOCUS_YOLO_MODEL", "yolov8x6_animeface.pt"))
            model_path = _resolve_model_path(model_path)
            if not os.path.exists(model_path):
                try:
                    _FDBG2["ai_face_ng"] += 1
                except Exception as e:
                    _kana_silent_exc('core:L16666', e)
                    pass
                return []

            conf = float(globals().get("FACE_FOCUS_YOLO_CONF", 0.25))
            iou = float(globals().get("FACE_FOCUS_YOLO_IOU", 0.45))
            imgsz = int(globals().get("FACE_FOCUS_YOLO_IMGSZ", 1536))
            device = str(globals().get("FACE_FOCUS_YOLO_DEVICE", ""))
            maxdet = int(globals().get("FACE_FOCUS_YOLO_MAXDET", 10))

            cache_key = "_FACE_FOCUS_YOLO_MODEL_OBJ"
            yolo_obj = globals().get(cache_key, None)
            cache_path_key = cache_key + "_PATH"
            cached_path = globals().get(cache_path_key, None)
            if (yolo_obj is None) or (cached_path != model_path):
                lock_key = "_FACE_FOCUS_YOLO_MODEL_LOCK"
                yolo_lock = globals().get(lock_key, None)
                if yolo_lock is None:
                    import threading
                    yolo_lock = threading.Lock()
                    globals()[lock_key] = yolo_lock
                with yolo_lock:
                    yolo_obj = YOLO(model_path, task="detect")
                    globals()[cache_key] = yolo_obj
                    globals()[cache_path_key] = model_path

            src_img = bgr
            if src_img is None:
                try:
                    src_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                except Exception:
                    src_img = gray

            # imgsz が入力より大きい場合、アップスケール推論を避けて速度を優先（特に draft_to で縮小した入力に効く）
            try:
                if isinstance(src_img, np.ndarray):
                    _h, _w = int(src_img.shape[0]), int(src_img.shape[1])
                    _m = max(_h, _w)
                    if _m > 0 and int(imgsz) > _m:
                        imgsz = int(_m)
            except Exception:
                pass

            try:
                t0 = time.perf_counter()
                res = yolo_obj.predict(src_img, conf=conf, iou=iou, imgsz=imgsz, device=device, max_det=maxdet, verbose=False)
                dt = time.perf_counter() - t0
                SG_PERF["yolo_sec"] += float(dt)
                SG_PERF["yolo_calls"] += 1
            except Exception as e:
                _warn_exc_once(e)
                try:
                    _FDBG2["ai_face_ng"] += 1
                except Exception as e:
                    _kana_silent_exc('core:L16705', e)
                    pass
                return []

            if not res:
                try:
                    _FDBG2["ai_face_ng"] += 1
                except Exception as e:
                    _kana_silent_exc('core:L16712', e)
                    pass
                return []

            try:
                r0 = res[0]
                boxes = getattr(r0, "boxes", None)
                if boxes is None:
                    raise ValueError("no boxes")
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') and (boxes.conf is not None) else None
                out = []
                for i in range(int(xyxy.shape[0])):
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    x = int(max(0, round(x1)))
                    y = int(max(0, round(y1)))
                    w = int(max(1, round(x2 - x1)))
                    h = int(max(1, round(y2 - y1)))
                    c = float(confs[i]) if confs is not None else 0.0
                    out.append((x, y, w, h, 2, c))
                if out:
                    try:
                        _FDBG2["ai_face_ok"] += 1
                    except Exception as e:
                        _kana_silent_exc('core:L16735', e)
                        pass
                else:
                    try:
                        _FDBG2["ai_face_ng"] += 1
                    except Exception as e:
                        _kana_silent_exc('core:L16740', e)
                        pass
                return out
            except Exception as e:
                _warn_exc_once(e)
                try:
                    _FDBG2["ai_face_ng"] += 1
                except Exception as e:
                    _kana_silent_exc('core:L16747', e)
                    pass
                return []
        def pick_ai_faces_animeface_cascade() -> list:
            # 戻り値: [(x,y,w,h,eyes_cnt,conf)]
            if (not need_face) or (not bool(globals().get("FACE_FOCUS_AI_ENABLE", False))):
                return []
            try:
                cas_name = str(globals().get("FACE_FOCUS_ANIMEFACE_CASCADE", "lbpcascade_animeface.xml"))
                cas_path = _resolve_model_path(cas_name)
                if not os.path.exists(cas_path):
                    try:
                        _FDBG2["ai_face_ng"] += 1
                    except Exception as e:
                        _kana_silent_exc('core:L16760', e)
                        pass
                    return []

                cache_key = "_FACE_FOCUS_ANIMEFACE_CASCADE_OBJ"
                cas_obj = globals().get(cache_key, None)
                cas_path_cached = globals().get(cache_key + "_PATH", None)
                if (cas_obj is None) or (cas_path_cached != cas_path):
                    cas_obj = cv2.CascadeClassifier(cas_path)
                    globals()[cache_key] = cas_obj
                    globals()[cache_key + "_PATH"] = cas_path

                sf = float(globals().get("FACE_FOCUS_ANIMEFACE_SCALE_FACTOR", 1.10))
                mn = int(globals().get("FACE_FOCUS_ANIMEFACE_MIN_NEIGHBORS", 3))
                ms = int(globals().get("FACE_FOCUS_ANIMEFACE_MIN_SIZE", 24))

                faces = cas_obj.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=(ms, ms))
                if faces is None or len(faces) == 0:
                    try:
                        _FDBG2["ai_face_ng"] += 1
                    except Exception as e:
                        _kana_silent_exc('core:L16780', e)
                        pass
                    return []

                out = []
                for (x, y, w, h) in faces:
                    out.append((int(x), int(y), int(w), int(h), 2, 0.0))

                try:
                    _FDBG2["ai_face_ok"] += 1
                except Exception as e:
                    _kana_silent_exc('core:L16790', e)
                    pass
                return out
            except Exception as e:
                try:
                    _FDBG2["ai_face_ng"] += 1
                except Exception as e:
                    _kana_silent_exc('core:L16796', e)
                    pass
                _warn_exc_once(e)
                return []


        def pick_ai_faces_ai() -> list:


            # 1回だけAI状態を表示（バックエンド/モデル/依存関係）

            try:

                if bool(globals().get("FACE_FOCUS_AI_ENABLE", False)) and (not bool(globals().get("_FACE_FOCUS_AI_STATUS_SHOWN", False))):

                    globals()["_FACE_FOCUS_AI_STATUS_SHOWN"] = True

                    backend = str(globals().get("FACE_FOCUS_AI_BACKEND", "yunet")).lower().strip()

                    if backend in ("yolov8_animeface", "yolo", "yolov8", "animeface_yolo"):

                        yolo_model = str(globals().get("FACE_FOCUS_YOLO_MODEL", "yolov8x6_animeface.pt"))

                        yolo_path = _resolve_model_path(yolo_model)

                        try:

                            # ultralytics を import するとバナーが出ることがあるため、メタデータから取得（副作用なし）

                            import importlib.metadata as _im

                            ultra_ver = _im.version('ultralytics')

                            has_ultra = True

                        except Exception:

                            has_ultra = False

                            ultra_ver = 'n/a'

                        # 進捗バー更新中は保留（prefetch の見た目崩れ防止）

                        if bool(globals().get("_BAR_ACTIVE", False)):

                            globals()["_AI_STATUS_PENDING"] = f"AI(YOLO): enabled=True | backend='{backend}' | ultralytics={has_ultra}({ultra_ver}) | model='{yolo_path}' | exists={os.path.exists(yolo_path)}"

                        else:

                            note(f"AI(YOLO): enabled=True | backend='{backend}' | ultralytics={has_ultra}({ultra_ver}) | model='{yolo_path}' | exists={os.path.exists(yolo_path)}")
                    elif backend in ("animeface_cascade", "animeface", "lbpcascade_animeface"):
                        cas_name = str(globals().get("FACE_FOCUS_ANIMEFACE_CASCADE", "lbpcascade_animeface.xml"))
                        cas_path = _resolve_model_path(cas_name)
                        if bool(globals().get("_BAR_ACTIVE", False)):
                            globals()["_AI_STATUS_PENDING"] = f"AI(AnimeFaceCascade): enabled=True | backend='{backend}' | cascade='{cas_path}' | exists={os.path.exists(cas_path)}"
                        else:
                            note(f"AI(AnimeFaceCascade): enabled=True | backend='{backend}' | cascade='{cas_path}' | exists={os.path.exists(cas_path)}")
                    else:

                        model = str(globals().get("FACE_FOCUS_YUNET_MODEL", "face_detection_yunet_2023mar.onnx"))

                        model_path = _resolve_model_path(model)

                        has_yn = hasattr(cv2, "FaceDetectorYN")

                        # 進捗バー更新中は保留（prefetch の見た目崩れ防止）

                        if bool(globals().get("_BAR_ACTIVE", False)):

                            globals()["_AI_STATUS_PENDING"] = f"AI(YuNet): enabled=True | backend='{backend}' | FaceDetectorYN={has_yn} | model='{model_path}' | exists={os.path.exists(model_path)}"

                        else:

                            note(f"AI(YuNet): enabled=True | backend='{backend}' | FaceDetectorYN={has_yn} | model='{model_path}' | exists={os.path.exists(model_path)}")
            except Exception:

                pass
            # YuNet / YOLO の切り替え
            backend = str(globals().get("FACE_FOCUS_AI_BACKEND", "yunet")).lower().strip()
            if backend in ("yolov8_animeface", "yolo", "yolov8", "animeface_yolo"):
                return pick_ai_faces_yolo()
            if backend in ("animeface_cascade", "animeface", "lbpcascade_animeface"):
                return pick_ai_faces_animeface_cascade()
            # 既定: YuNet
            return pick_ai_faces_yunet()

        faces_all: list = []
        # 正面顔
        try:
            cas_fr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = pick_faces(cas_fr)
            faces_all += [("frontal",) + t for t in faces]
        except Exception as e:
            _warn_exc_once(e)
            pass
        # 横顔
        if globals().get("FACE_FOCUS_USE_PROFILE", True):
            try:
                cas_pr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
                faces = pick_faces(cas_pr)
                faces_all += [("profile",) + t for t in faces]
            except Exception as e:
                _warn_exc_once(e)
                pass

        # 採用：Haar だけで「顔」を確定しない（誤爆対策）。
        # - 候補が複数ある場合は、まず「上側（center_yが小さい）」を最優先で選びます。
        # - そのうえで中央寄り（center_x）・目の数（eyes）・面積（area）で安定化します。
        # - アニメ/AI絵では Haar が“たまに”誤爆するため、必要に応じて eye-pair 由来の擬似 face を比較します。
        face_selected = False
        best_haar = None  # ("frontal"|"profile", x, y, w, h, eyes_cnt)
        faces_eyes = []
        faces_loose = []


        # AI（YuNet）: 追加候補として faces_all に混ぜる（精度優先なら ALWAYS=True）
        if bool(globals().get("FACE_FOCUS_AI_ENABLE", False)) and bool(globals().get("FACE_FOCUS_AI_ALWAYS", False)):
            try:
                ai_faces = pick_ai_faces_ai()
                faces_all += [("ai", int(x), int(y), int(w), int(h), int(eyes)) for (x, y, w, h, eyes, _conf) in ai_faces]
            except Exception as e:
                _warn_exc_once(e)
                pass

        def _face_validator(t, require_eyes: bool = True, y_max_override=None):
            # t = (kind, x, y, w, h, eyes_cnt)
            if not bool(globals().get("FACE_FOCUS_VALIDATOR_ENABLE", True)):
                return True
            try:
                kind = str(t[0])
                x, y, w, h = float(t[1]), float(t[2]), float(t[3]), float(t[4])
                eyes = int(t[5]) if len(t) >= 6 else 0
                cy = y + h * 0.5
                y_norm = cy / max(1.0, float(ih_cv))
                if y_max_override is None:
                    y_max = float(globals().get("FACE_FOCUS_VALIDATOR_MAX_Y", 0.70))
                else:
                    y_max = float(y_max_override)
                eye_min = int(globals().get("FACE_FOCUS_VALIDATOR_EYE_MIN", 1))
                if y_norm > y_max:
                    return False
                # 形状（アスペクト比）で雑な誤検出を少し抑える
                try:
                    ar = float(w) / max(1.0, float(h))
                    ar_min = float(globals().get("FACE_FOCUS_VALIDATOR_AR_MIN", 0.45))
                    ar_max = float(globals().get("FACE_FOCUS_VALIDATOR_AR_MAX", 1.70))
                    if ar < ar_min or ar > ar_max:
                        return False
                except Exception as e:
                    _kana_silent_exc('core:L16944', e)
                    pass
                # アニメ候補は eye-pair 由来のため、verify が外れても棄却しない（位置で制御）
                if require_eyes and (kind != "anime") and (eyes < eye_min):
                    return False
                return True
            except Exception:
                return True

        if faces_all:
            try:
                eye_min = int(globals().get("FACE_FOCUS_EYE_MIN", 1) or 1)
            except Exception:
                eye_min = 1

            # 目が取れている候補を優先（誤検出: 胸/背景 を弾く）
            faces_eyes = [t for t in faces_all if (len(t) >= 6 and int(t[5]) >= eye_min and _face_validator(t, require_eyes=True))]
            # 目検出が外れた場合の救済（アニメ等）：位置だけで候補を残す（上側に限定）
            try:
                y_max_loose = float(globals().get("FACE_FOCUS_VALIDATOR_MAX_Y_LOOSE", 0.62))
            except Exception:
                y_max_loose = 0.62
            faces_loose = [t for t in faces_all if _face_validator(t, require_eyes=False, y_max_override=y_max_loose)]

            def _cand_score(t, prefer_anime: bool):
                # t = (kind, x, y, w, h, eyes_cnt)
                kind, x, y, w, h = t[0], int(t[1]), int(t[2]), int(t[3]), int(t[4])
                eyes_cnt = int(t[5]) if len(t) >= 6 else 0
                cx = x + w * 0.5
                cy = y + h * 0.5
                y_norm = cy / max(1.0, float(ih_cv))
                x_norm = abs(cx - float(iw_cv) * 0.5) / max(1.0, float(iw_cv))
                area = float(w * h)

                # kind 優先度（AI顔を最優先、次に anime/frontal）
                if kind == "ai":
                    kind_pri = -1
                elif kind == "anime" and prefer_anime:
                    kind_pri = 0
                elif kind == "frontal":
                    kind_pri = 1
                else:
                    kind_pri = 2

                # 重要度: 上側(y) → 中央寄り(x) → 目の数 → 面積
                # 最後に bbox を入れておくと、検出器の返却順が揺れても tie-break が安定します。
                return (kind_pri, y_norm, x_norm, -eyes_cnt, -area, y, x, h, w)

            if faces_eyes:
                faces_eyes.sort(key=lambda t: _cand_score(t, prefer_anime=False))
                best_haar = faces_eyes[0]
            elif faces_loose:
                faces_loose.sort(key=lambda t: _cand_score(t, prefer_anime=False))
                best_haar = faces_loose[0]

        def _is_suspicious_haar(t):
            # Haar が“怪しい”ときはアニメ顔推定も比較する
            try:
                thr_y = float(globals().get("FACE_FOCUS_HAAR_SUSPICIOUS_Y", 0.62) or 0.62)
            except Exception:
                thr_y = 0.62
            try:
                kind, x, y, w, h = t[0], float(t[1]), float(t[2]), float(t[3]), float(t[4])
            except Exception:
                return True
            cy = y + h * 0.5
            y_norm = cy / max(1.0, float(ih_cv))
            # 横顔は誤爆が増えやすいので怪しい判定に寄せる
            if kind != "frontal":
                return True
            return (y_norm > thr_y)

        # アニメ/AI絵向け: eye-pair から擬似 face bbox を推定（必要に応じて Haar と比較）
        anime_cand = None  # ("anime", x, y, w, h, eyes_cnt)
        prefer_anime = bool(globals().get("FACE_FOCUS_ANIME_FACE_PREFER", True))
        if need_face and bool(globals().get("FACE_FOCUS_ANIME_FACE_ENABLE", True)):
            do_anime = bool(globals().get("FACE_FOCUS_ANIME_FACE_ALWAYS", False))
            if (not do_anime) and (best_haar is None):
                do_anime = True
            if (not do_anime) and (best_haar is not None) and _is_suspicious_haar(best_haar):
                do_anime = True

            if do_anime:
                try:
                    topf = float(globals().get("FACE_FOCUS_TOP_FRAC", 0.70) or 0.70)
                except Exception:
                    topf = 0.70
                try:
                    md = int(globals().get("FACE_FOCUS_ANIME_FACE_MAX_DIM", 720) or 720)
                except Exception:
                    md = 720

                af = _kana_anime_face_detect(gray, top_frac=topf, max_dim=md)
                if af is None:
                    try:
                        _FDBG2["anime_face_ng"] += 1
                    except Exception as e:
                        _kana_silent_exc('core:L17040', e)
                        pass
                else:
                    ax, ay, aw, ah = af
                    try:
                        ratio = float(aw) / float(ah + 1e-6)
                        rmin = float(globals().get("FACE_FOCUS_FACE_RATIO_MIN", 0.65))
                        rmax = float(globals().get("FACE_FOCUS_FACE_RATIO_MAX", 1.60))
                        if ratio < rmin or ratio > rmax:
                            try:
                                _FDBG["reject_ratio"] += 1
                            except Exception as e:
                                _kana_silent_exc('core:L17051', e)
                                pass
                        else:
                            cy = ay + ah / 2.0
                            if (not globals().get("FACE_FOCUS_ALLOW_LOW", False)) and (cy > topf * ih_cv):
                                try:
                                    _FDBG2["low_reject"] += 1
                                except Exception as e:
                                    _kana_silent_exc('core:L17058', e)
                                    pass
                            else:
                                # 目の数をスコアリングに使う（取れない場合は0）
                                eyes_cnt = 0
                                try:
                                    _, eyes_cnt = _kana_face_eye_verify(gray, (int(ax), int(ay), int(aw), int(ah)))
                                except Exception:
                                    eyes_cnt = 0
                                anime_cand = ("anime", int(ax), int(ay), int(aw), int(ah), int(eyes_cnt))
                                try:
                                    _FDBG2["anime_face_ok"] += 1
                                except Exception as e:
                                    _kana_silent_exc('core:L17070', e)
                                    pass
                    except Exception:
                        # 失敗しても落とさない
                        pass

        # Haar と anime の比較 → 最終採用
        picked = None
        if best_haar is not None:
            picked = best_haar
        if anime_cand is not None and _face_validator(anime_cand, require_eyes=False):
            if picked is None:
                picked = anime_cand
            else:
                try:
                    if _cand_score(anime_cand, prefer_anime=prefer_anime) < _cand_score(picked, prefer_anime=prefer_anime):
                        picked = anime_cand
                except Exception:
                    # 例外時は anime を優先（不安定対策）
                    picked = anime_cand

        if picked is not None:
            k, x, y, w, h = picked[0], int(picked[1]), int(picked[2]), int(picked[3]), int(picked[4])
            result["face"] = (k, int(x), int(y), int(w), int(h))
            face_selected = True

        # 詳細デバッグ（kind / center_y / eyes）: 既定は少数画像だけ
        try:
            if bool(globals().get("FACE_FOCUS_DEBUG_DETAIL", False)):
                lim = int(globals().get("FACE_FOCUS_DEBUG_DETAIL_LIMIT", 3) or 3)
                shown = int(globals().get("_FACE_FOCUS_DEBUG_DETAIL_COUNT", 0) or 0)
                if (lim <= 0) or (shown < lim):
                    globals()["_FACE_FOCUS_DEBUG_DETAIL_COUNT"] = shown + 1
                    maxc = int(globals().get("FACE_FOCUS_DEBUG_DETAIL_MAX_CANDIDATES", 8) or 8)

                    name = "<memory>"
                    try:
                        if src_path:
                            name = os.path.basename(str(src_path))
                    except Exception as e:
                        _kana_silent_exc('core:L17109', e)
                        pass
                    def _fmt_one(t):
                        kind, x, y, w, h = t[0], float(t[1]), float(t[2]), float(t[3]), float(t[4])
                        eyes = int(t[5]) if len(t) >= 6 else 0
                        cy = y + h * 0.5
                        y_norm = cy / max(1.0, float(ih_cv))
                        return f"{kind:7s} center_y={y_norm:.3f} eyes={eyes} bbox=({int(x)},{int(y)},{int(w)},{int(h)})"

                    print(f"[face-debug] {name}")
                    if faces_eyes:
                        for t in faces_eyes[:maxc]:
                            print("  cand:", _fmt_one(t))
                    if anime_cand is not None:
                        print("  cand:", _fmt_one(anime_cand))
                    if picked is not None:
                        print("  -> pick:", _fmt_one(picked))
        except Exception as e:
            _kana_silent_exc('core:L17127', e)
            pass
# 上半身検出（任意）
        if globals().get("FACE_FOCUS_USE_UPPER", False) and need_upper:
            try:
                ub = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
                ubx = ub.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    flags=cv2.CASCADE_SCALE_IMAGE,
                    minSize=(max(32, int(0.15 * min(gray.shape))),) * 2,  # 上半身検出の最小サイズ（短辺の15%程度、最小32px）
                )
                if ubx is not None and len(ubx) > 0:
                    # 面積最大の候補を採用
                    x, y, w, h = max(ubx, key=lambda b: b[2] * b[3])
                    result["upper"] = (x, y, w, h)
            except Exception as e:
                _warn_exc_once(e)
                pass

        # Person-focus（任意）：顔が無い／小さいときに「人物（全体/上半身）」へ寄せます。
        # まず上半身カスケードの結果を流用し、無ければ（写真向け）HOG 人物検出を試します。
        if bool(globals().get("FACE_FOCUS_USE_PERSON", True)) and need_person:
            try:
                if result.get("upper") is not None:
                    result["person"] = result.get("upper")
                elif bool(globals().get("PERSON_FOCUS_HOG_ENABLE", True)):
                    # 高速化のため縮小してからHOG検出（検出座標は元解像度へ戻す）
                    _max_dim = int(globals().get("PERSON_FOCUS_HOG_MAX_DIM", 640) or 640)
                    sc = 1.0
                    if max(iw_cv, ih_cv) > max(64, _max_dim):
                        sc = float(_max_dim) / float(max(iw_cv, ih_cv))
                    if sc < 1.0:
                        g2 = cv2.resize(gray, (int(iw_cv * sc), int(ih_cv * sc)), interpolation=cv2.INTER_AREA)
                    else:
                        g2 = gray

                    # HOG 人物検出器は初期化コストがあるため、グローバルに遅延初期化します。
                    global _HOG_PERSON
                    try:
                        _HOG_PERSON
                    except NameError:
                        _HOG_PERSON = None
                    if _HOG_PERSON is None:
                        hog = cv2.HOGDescriptor()
                        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                        _HOG_PERSON = hog
                    rects, _weights = _HOG_PERSON.detectMultiScale(
                        g2,
                        winStride=(8, 8),
                        padding=(8, 8),
                        scale=1.05,
                    )
                    if rects is not None and len(rects) > 0:
                        # 面積最大の人物を採用
                        x, y, w, h = max(rects, key=lambda b: b[2] * b[3])
                        if sc < 1.0:
                            x = int(x / sc)
                            y = int(y / sc)
                            w = int(w / sc)
                            h = int(h / sc)
                        result["person"] = (int(x), int(y), int(w), int(h))
            except Exception as e:
                _kana_silent_exc('core:L17190', e)
                pass
        # Saliency-focus（任意）：最後の保険として「目立つ点」を推定します。
        if bool(globals().get("FACE_FOCUS_USE_SALIENCY", True)) and need_saliency:
            try:
                # 高速化のため縮小してから処理
                small_w = 480
                scale = small_w / float(im.size[0]) if im.size[0] > small_w else 1.0
                sm = cv2.resize(gray, (int(im.size[0] * scale), int(im.size[1] * scale)), interpolation=cv2.INTER_AREA)
                lap = cv2.Laplacian(sm, cv2.CV_32F)
                lap = cv2.convertScaleAbs(lap)
                win = int(max(16, (max(im.size) * scale) * 0.9))
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
                heat = cv2.filter2D(lap, -1, k)
                ylimit = int(heat.shape[0] * float(globals().get("FACE_FOCUS_TOP_FRAC", 0.70)))  # サリエンシー探索の上限（下側を無視する割合）
                # 顔検出と同じく、サリエンシー探索も下側を無視して誤爆（手や小物）を減らします。
                _, _, _, maxloc = cv2.minMaxLoc(heat[:ylimit, :])
                cx = maxloc[0] / max(1e-6, scale)
                cy = maxloc[1] / max(1e-6, scale)
                result["saliency"] = (cx, cy)
            except Exception:
                # 検出に失敗しても落とさない
                pass
        try:
            if src_path is not None and (need_face or need_upper or need_person or need_saliency):
                _face_cache_put(src_path, result.get("face"), result.get("upper"), result.get("person"), result.get("saliency"))
        except Exception as e:
            _warn_exc_once(e)
            pass
    except Exception:
        # OpenCV 未導入など、想定外でも落とさず None のまま返す
        pass
    return result

def adjust_brightness_with_mask(canvas: Image.Image, content_mask: Image.Image) -> tuple[Image.Image, dict]:
    mode=BRIGHTNESS_MODE
    method=AUTO_METHOD
    target=AUTO_TARGET_MEAN
    info={"mode":mode,"method":None,"target":target,"gain":None,"gamma":None,"original_mean":None,"final_mean":None}

    orig = mean_luma_masked(canvas, content_mask)
    info["original_mean"]=orig

    if orig is None or mode=="off":
        info["method"] = "off"
        info["final_mean"] = orig
        return canvas, info

    if mode=="manual":
        adj = apply_gamma(canvas, MANUAL_GAMMA)
        adj = ImageEnhance.Brightness(adj).enhance(MANUAL_GAIN)
        out = Image.composite(adj, canvas, content_mask)
        info.update({"method":"manual","gain":MANUAL_GAIN,"gamma":MANUAL_GAMMA,
                     "final_mean":mean_luma_masked(out, content_mask)})
        return out, info

    if method=="gamma":
        # 生のガンマ値を計算（apply_gamma は x**(1/gamma) なので、その向きを意識する）
        gamma_raw = max(AUTO_GAMMA_MIN, min(AUTO_GAMMA_MAX,
                        math.log(max(1e-4,orig))/math.log(max(1e-4,target))))
        # 1.0 に寄せて効きすぎを抑える（AUTO_GAMMA_SOFTEN が大きいほど弱まる）
        s = max(0.0, min(1.0, AUTO_GAMMA_SOFTEN))
        gamma = 1.0 + (gamma_raw - 1.0) * (1.0 - s)
        gamma = max(AUTO_GAMMA_MIN, min(AUTO_GAMMA_MAX, gamma))
        adj = apply_gamma(canvas, gamma)
        out = Image.composite(adj, canvas, content_mask)
        info.update({"method":"gamma","gamma":gamma,"gamma_raw":gamma_raw,"gamma_soften":s,
                     "final_mean":mean_luma_masked(out, content_mask)})
        return out, info

    if method=="gain":
        gain = max(AUTO_GAIN_MIN, min(AUTO_GAIN_MAX, target/max(1e-4,orig)))
        adj = ImageEnhance.Brightness(canvas).enhance(gain)
        out = Image.composite(adj, canvas, content_mask)
        info.update({"method":"gain","gain":gain,
                     "final_mean":mean_luma_masked(out, content_mask)})
        return out, info

    # hybrid（ガンマ → ゲイン の2段階補正）
    gamma_raw = max(AUTO_GAMMA_MIN, min(AUTO_GAMMA_MAX,
                    math.log(max(1e-4,orig))/math.log(max(1e-4,target))))
    s = max(0.0, min(1.0, AUTO_GAMMA_SOFTEN))
    gamma = 1.0 + (gamma_raw - 1.0) * (1.0 - s)
    gamma = max(AUTO_GAMMA_MIN, min(AUTO_GAMMA_MAX, gamma))
    tmp = apply_gamma(canvas, gamma)
    newm = mean_luma_masked(tmp, content_mask) or orig
    gain = max(AUTO_GAIN_MIN, min(AUTO_GAIN_MAX, target/max(1e-4,newm)))
    adj = ImageEnhance.Brightness(tmp).enhance(gain)
    out = Image.composite(adj, canvas, content_mask)
    info.update({"method":"hybrid","gamma":gamma,"gamma_raw":gamma_raw,"gamma_soften":s,"gain":gain,
                 "final_mean":mean_luma_masked(out, content_mask)})
    return out, info

# -----------------------------------------------------------------------------
# サブセクション: 壁紙設定・保存・ログ
# -----------------------------------------------------------------------------
def choose_output_path(dirpath: Path, basename: str, fmt: str) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    suffix = ".png" if fmt.lower().startswith("png") else ".jpg"
    return (dirpath / basename).with_suffix(suffix)

def set_wallpaper(path: Path, style: str = "Fill"):
    """Windows の壁紙を設定します。

    - Windows 以外では何もせず終了します（winreg が無い環境での例外を防ぐため）。
    - Windows ではレジストリのスタイルを更新後、SystemParametersInfoW で反映します。

    引数:
        path: 壁紙に設定する画像ファイルのパス
        style: 表示スタイル（"Fill"／"Fit"／"Stretch"／"Center"）
    """
    # Windows 以外では何もしない
    import sys
    if not sys.platform.startswith("win"):
        # 非 Windows 環境向けの案内メッセージ（_lang によりローカライズ可）
        try:
            msg = _lang("[INFO] 壁紙設定は Windows 環境でのみサポートされています",
                        "[INFO] Wallpaper setting is only supported on Windows")
            print(C("93", msg))
        except Exception as e:
            _warn_exc_once(e)
            pass
        return
    try:
        import ctypes, winreg  # type: ignore
        style_map = {
            "Fill": ("10", "0"),
            "Fit": ("6", "0"),
            "Stretch": ("2", "0"),
            "Center": ("0", "0"),
        }
        wp_style, tile = style_map.get(style, ("10", "0"))
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Control Panel\Desktop", 0, winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, "WallpaperStyle", 0, winreg.REG_SZ, wp_style)
        winreg.SetValueEx(key, "TileWallpaper", 0, winreg.REG_SZ, tile)
        winreg.CloseKey(key)
        # SystemParametersInfoW で壁紙を反映
        ctypes.windll.user32.SystemParametersInfoW(20, 0, str(path), 0x01 | 0x02)
    except Exception as e:
        # 失敗した場合は警告を出す（色付き出力が使えれば使う）
        try:
            msg = _lang(f"[WARN] 壁紙設定に失敗: {e}", f"[WARN] Failed to set wallpaper: {e}")
            print(C("91;1", msg))
        except Exception:
            # 色付きが使えない場合は通常の print へフォールバック
            try:
                msg = _lang(f"[WARN] 壁紙設定に失敗: {e}", f"[WARN] Failed to set wallpaper: {e}")
                print(msg)
            except Exception as e:
                _warn_exc_once(e)
                pass
def remove_artifacts(paths: Sequence[Path]):
    for p in paths:
        try:
            if p and p.exists(): p.unlink()
        except Exception as e:
            _warn_exc_once(e)
            pass
def write_used_lists(imgs: Sequence[ImageRef], rows:int, cols:int, seed:int, target:Sequence[ImageRef],
                     layout_info=None, brightness_info=None,
                     log_dir: Path = LOG_SAVE_DIR):
    if not SAVE_ARTIFACTS:
        if DELETE_OLD_WHEN_DISABLED:
            remove_artifacts([log_dir/"kana_wallpaper_used_images.csv",
                              log_dir/"kana_wallpaper_used_images.txt",
                              log_dir/"kana_wallpaper_meta.json"])
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    LIST_CSV  = log_dir / "kana_wallpaper_used_images.csv"
    LIST_TXT  = log_dir / "kana_wallpaper_used_images.txt"
    META_JSON = log_dir / "kana_wallpaper_meta.json"

    def file_info(p: ImageRef):
        try:
            sz = _imgref_size(p)
            mt = _imgref_mtime(p)
            return sz, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mt))
        except Exception:
            return None, None

    with open(LIST_CSV,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f); w.writerow(["index","row","col","abs_path","size_bytes","mtime_local"])
        for i,p in enumerate(imgs,1):
            r = (i-1)//max(1,cols)+1 if cols else 1
            c = (i-1)%max(1,cols)+1 if cols else i
            size,mt=file_info(p)
            w.writerow([i,r,c,_imgref_display(p),size,mt])

    with open(LIST_TXT,"w",encoding="utf-8",newline="\r\n") as f:
        f.write("Kana Wallpaper - Used Images List (Aggressive Optimizers Pack)\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Targets:\n")
        for tr in target: f.write(f"  - {tr}\n")
        f.write(f"Seed: {seed}\nLayoutStyle: {LAYOUT_STYLE} (random={RANDOM_LAYOUT_CANDIDATES})\n")
        if layout_info: f.write(f"LayoutInfo: {layout_info}\n")
        if brightness_info: f.write(f"Brightness: {brightness_info}\n")
        f.write("-"*60+"\n")
        for i,p in enumerate(imgs,1): f.write(f"[{i:03d}] {_imgref_display(p)}\n")

    meta={"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
          "width":WIDTH, "height":HEIGHT, "gutter":GUTTER, "margin":MARGIN,
          "format":FORMAT, "bg_color":BG_COLOR, "seed":seed, "count":len(imgs),
          "layout_style":LAYOUT_STYLE, "random_candidates":RANDOM_LAYOUT_CANDIDATES,
          "grid":{"rows":ROWS,"cols":COLS,"mode":MODE,"optimizer":GRID_OPTIMIZER,"objective":GRID_NEIGHBOR_OBJECTIVE},
          "mosaic":{"row_h_min":JUSTIFY_MIN_ROW_H,"row_h_max":JUSTIFY_MAX_ROW_H,
                    "balance":MOSAIC_BALANCE_ENABLE,"neighbor_obj":MOSAIC_NEIGHBOR_OBJECTIVE,"seq_algo":MOSAIC_SEQ_ALGO},
          "brightness":brightness_info or {}, "layout":layout_info or {},
          "targets":[str(t) for t in target]}
    with open(META_JSON,"w",encoding="utf-8") as f:
        json.dump(meta,f,ensure_ascii=False,indent=2)

# -----------------------------------------------------------------------------
# サブセクション: メイン
# -----------------------------------------------------------------------------
_ALLOWED = {"grid","mosaic-uniform-height","mosaic-uniform-width","hex","quilt"}

def choose_random_layout(rng: random.Random, candidates: Sequence[str]) -> str:
    valid=[c for c in candidates if str(c) in _ALLOWED]
    if not valid: valid=["mosaic-uniform-height","mosaic-uniform-width","grid","hex"]
    return rng.choice(valid)


# =============================================================================
# セクション: エントリーポイント（メイン）
# =============================================================================

# -----------------------------------------------------------------------------
# サブセクション: 実行前チェック
# -----------------------------------------------------------------------------

def _abort_no_images(reason: str) -> None:
    """画像が1枚も見つからない場合に、安全に中断します（壁紙の更新もしません）。"""
    print("")
    print("⚠ 画像が見つかりませんでした。処理を中止します。")
    if reason:
        print(f"  - {reason}")
    print("  - 入力フォルダ/サブフォルダ/ZIP/動画抽出設定、またはフォルダの中身を確認してください。")
    print("")

_LOGGER = None
_LOGGER_READY = False

def main():
    # 外部設定（ランチャー書き出し）を適用
    # - 本体を直接実行（__main__）したときのみ適用する
    # - ランチャーから import して実行する場合は、ランチャー側の設定を優先する
    if __name__ == "__main__":
        _apply_external_launcher_config()

    """コマンドライン引数や既定フォルダを解釈し、壁紙生成処理を一通り実行するエントリーポイント。"""
    argv = sys.argv[1:]
    # ヘルプは最優先で表示（他の表示・処理をしない）
    if any(a in ('-h', '--help', '-help') for a in argv):
        print(HELP_TEXT)
        return

    init_console(); banner("Kana Wallpaper - Unified FINAL")

    # KANA: 例外の握りつぶしを見える化（プロセス終了時に1回だけ表示）
    try:
        _kana_register_silent_exc_atexit(note)
    except Exception:
        pass

    roots: List[str] = []
    global SAVE_ARTIFACTS, SAVE_IMAGE, IMAGE_SAVE_DIR, IMAGE_BASENAME, LOG_SAVE_DIR, RECURSIVE
    next_is=None
    for a in sys.argv[1:]:
        if a in ("-h","--help","-help"):
            print(HELP_TEXT)
            return
        elif a in ("--logs","--records"):     SAVE_ARTIFACTS=True
        elif a in ("--no-image",):            SAVE_IMAGE=False
        elif a in ("--image",):               SAVE_IMAGE=True
        elif a=="--img-dir":  next_is="img_dir"
        elif a=="--log-dir":  next_is="log_dir"
        elif a=="--img-name": next_is="img_name"
        elif a in ("--top-only","--no-recursive"):
            # サブフォルダを辿らない（トップ階層のみ）
            RECURSIVE = False
        elif a in ("--recursive",):
            # サブフォルダも含めて走査
            RECURSIVE = True
        elif a.startswith("-"): pass
        else: roots.append(a)

    # 対象フォルダの決定（D&D or 既定フォルダ）
    if roots:
        targets=[Path(a).resolve() for a in roots]
        note("D&D/CLI:");
        for t in targets: note(f" - {t}")
    else:
        # 既定フォルダは「スクリプトと同じ場所」基準で解決（CWD依存で見失わない）
        _base = Path(__file__).resolve().parent
        targets=[]
        for s in DEFAULT_TARGET_DIRS:
            try:
                p = Path(s)
                if not p.is_absolute():
                    p = (_base / p)
                targets.append(p.resolve())
            except Exception:
                # 失敗したらそのまま文字列として扱う（後続で exists() が False になる）
                targets.append(Path(s))

        note(_lang("ダブルクリック（既定の複数フォルダ）:","Double-click input (default folders):"))
        for t in targets: note(f" - {t}")

    exists=[t for t in targets if t.exists()]
    if not exists:
        # 対象フォルダが 1 つも見つからない場合はエラー終了
        err_jp = f"[ERROR] フォルダが見つかりません: {', '.join(map(str, targets))}"
        err_en = f"[ERROR] Folder not found: {', '.join(map(str, targets))}"
        print(C("91;1", _lang(err_jp, err_en))); sys.exit(1)
    # 走査設定（サブフォルダ含む/含まない）を表示
    try:
        if globals().get("UI_LANG", "ja") == "en":
            note("Scan includes subfolders" if RECURSIVE else "Scan excludes subfolders")
        else:
            note("サブフォルダを含めて走査します" if RECURSIVE else "サブフォルダは対象外で走査します")
    except Exception as e:
        _warn_exc_once(e)
        pass
    # シード（動画フレーム抽出など、走査中の乱数にも使う）
        # シード（動画フレーム抽出など、走査中の乱数にも使う）
    _resolve_seed_aliases()
    _ss = globals().get("SHUFFLE_SEED", None)
    _ssi = _seed_to_int(_ss)
    seed_used = (_ssi if isinstance(_ssi, int) else secrets.randbits(64))
    # 動画フレーム抽出の乱数を「実行ごとに変える」ため、走査前に保存しておく
    globals()["_RUN_SEED_USED"] = seed_used
    _note_config_summary(seed_used)
    # スキャン
    all_imgs=collect_images(exists, recursive=RECURSIVE)

    # スキャン結果が0枚なら中断（黒い壁紙を生成しない）

    if not all_imgs:

        _abort_no_images('スキャン結果が0枚でした')

        return

    # hex の枚数不足を補うため、走査した全プールを globals() に保持しておく（重複除去付き）
    try:
        import os as _kana_os
        def _kana_norm(p):
            try:
                return _kana_os.path.normcase(_kana_os.path.normpath(p))
            except Exception:
                return p
        _seen = set(); _uniq = []
        for _p in all_imgs:
            _k = _kana_norm(_p)
            if _k not in _seen:
                _uniq.append(_p); _seen.add(_k)
        globals()["KANA_SCAN_ALL"] = _uniq
    except Exception:
        globals()["KANA_SCAN_ALL"] = list(dict.fromkeys(all_imgs))

    # シード
    # シード
    rng = random.Random(seed_used)
    # 抽出
    mode = str(SELECT_MODE).lower()

    _dedup_all = bool(globals().get('SELECT_DEDUP_ALWAYS', False))

    # --- Grid動画タイムライン：SELECT_MODEに関係なく全区間から満遍なく選ぶ ---
    # GRID_VIDEO_TIMELINE=asc/desc を選んだ場合、動画フレームの時系列カバレッジを最優先します。
    # （SELECT_MODE が random/aesthetic/recent/oldest 等でも同様）
    _picked_by_grid_timeline = False
    try:
        _gvt0 = str(globals().get('GRID_VIDEO_TIMELINE', 'off') or 'off').strip().lower()
    except Exception:
        _gvt0 = 'off'
    try:
        _style0 = str(globals().get('LAYOUT_STYLE', '') or '').strip().lower()
    except Exception:
        _style0 = ''
    try:
        _spread0 = bool(globals().get('GRID_VIDEO_TIMELINE_SPREAD', True))
    except Exception:
        _spread0 = True

    if _style0 == 'grid' and _gvt0 in ('asc', 'desc') and _spread0 and _all_video_frames_only(list(all_imgs)):
        try:
            banner(_lang('動画タイムライン抽出完了', 'Video timeline selection complete'))
            note(f'Grid video timeline (spread): {_gvt0}')
        except Exception as e:
            _kana_silent_exc('core:L17576', e)
            pass
        # 後半欠け防止のため、ローカルdedupe付きのタイムライン分散ピッカーを使います。
        picked_paths = _pick_video_frames_timeline_spread(list(all_imgs), COUNT, order=_gvt0, dedupe=bool(_dedup_all))
        _picked_by_grid_timeline = True

    if not _picked_by_grid_timeline:
        if mode == "aesthetic":
            picked_paths = score_and_pick(all_imgs, COUNT, seed=seed_used)
            banner(_lang("美選抜抽出完了","Aesthetic selection complete"))
        elif mode in ("recent", "newest", "mtime", "modified"):
            # 更新日時が新しい順に抽出
            picked_paths = pick_recent(all_imgs, COUNT, dedupe=(_dedup_all or bool(globals().get("SELECT_RECENT_DEDUP", True))))
            banner(_lang("更新順抽出完了","Recent selection complete"))
        elif mode in (
            "oldest", "older", "mtime_asc",
            "name", "filename", "name_asc", "filename_asc",
            "name_desc", "filename_desc"
        ):
            # ソート抽出（古い順・名前順）
            picked_paths = pick_sorted_generic(all_imgs, COUNT, dedupe=(_dedup_all or bool(globals().get("SELECT_SORT_DEDUP", True))))
            banner(_lang("並び替え抽出完了","Sorted selection complete"))
        else:
            # ランダム抽出
            _shuf = hash_shuffle(all_imgs, seed_used, salt="select_random")
            if _dedup_all or bool(globals().get("SELECT_RANDOM_DEDUP", False)):

                banner(_lang("ランダム抽出（近似重複除去）","Random selection (dedupe)"))
                picked_paths = pick_random_dedup(_shuf, COUNT)
            else:
                picked_paths = _shuf[:COUNT]
            banner(_lang("ランダム抽出完了","Random selection complete"))


    # Selection summary (single line; avoid per-branch duplicates)
    try:
        note(f"Picked: {len(picked_paths)}")
    except Exception as e:
        _kana_silent_exc('core:L17613', e)
        pass
        # 選択結果が0枚なら中断（黒い壁紙を生成しない）

        if not picked_paths:

            _abort_no_images('選択結果が0枚でした')

            return

    # --- Grid-only video timeline ordering (optional) ---
    try:
        _gvt = str(globals().get("GRID_VIDEO_TIMELINE", "off") or "off").strip().lower()
    except Exception:
        _gvt = "off"
    try:
        _style = str(globals().get("LAYOUT_STYLE", "")).strip().lower()
    except Exception:
        _style = ""
    if _style == "grid" and _gvt in ("asc", "desc"):
        try:
            if _all_video_frames_only(list(picked_paths)):
                try:
                    note(f"Grid video timeline (strict): {_gvt}")
                except Exception as e:
                    _kana_silent_exc('core:L17639', e)
                    pass
                def _k(_p):
                    meta = _VIDEO_FRAME_META.get(str(_p))
                    if meta:
                        vname, t_ms = meta
                        try:
                            t_ms_i = int(t_ms)
                        except Exception:
                            t_ms_i = 0
                        if _gvt == "desc":
                            return (str(vname), -t_ms_i, str(_p).lower())
                        return (str(vname), t_ms_i, str(_p).lower())
                    return ("", 0, str(_p).lower())
                picked_paths = sorted(list(picked_paths), key=_k)
        except Exception as e:
            _warn_exc_once(e)
            pass


    # --- Grid timeline: force strict order preservation ---
    # When GRID_VIDEO_TIMELINE is active (asc/desc) and the picked set is video frames only,
    # we must not run any reordering steps after this point (grid optimizer / tempo / shuffle),
    # otherwise the visual order will no longer be chronological.
    if _style == "grid" and _gvt in ("asc", "desc"):
        try:
            if _all_video_frames_only(list(picked_paths)):
                # Keep ordering stable for the whole run
                try:
                    globals()["PRESERVE_INPUT_ORDER"] = True
                except Exception as e:
                    _kana_silent_exc('core:L17669', e)
                    pass
                try:
                    globals()["ARRANGE_FULL_SHUFFLE"] = False
                except Exception as e:
                    _kana_silent_exc('core:L17673', e)
                    pass
                # Belt & suspenders: explicitly disable grid neighbor optimization
                try:
                    globals()["GRID_NEIGHBOR_OBJECTIVE"] = "off"
                except Exception as e:
                    _kana_silent_exc('core:L17678', e)
                    pass
        except Exception as e:
            _warn_exc_once(e)
            pass
    # --- Tempo（pre）: 入力順の事前整列 ---
    try:
        if (not bool(globals().get('PRESERVE_INPUT_ORDER', False))) and ARRANGE_TEMPO_ENABLE and str(ARRANGE_TEMPO_STAGE).lower() == "pre":
            picked_paths = _arrange_by_tempo(picked_paths, ARRANGE_TEMPO_MODE)
    except Exception as e:
        _warn_exc_once(e)
        pass
    # レイアウト選択
    bg=(0,0,0) if BG_COLOR in ("#000","#000000") else parse_color(BG_COLOR)
    layout_info={"style":LAYOUT_STYLE}
    style = LAYOUT_STYLE.lower()
    chosen = None
    if style == "random":
        chosen = choose_random_layout(rng, RANDOM_LAYOUT_CANDIDATES)

    # --- Layout dispatch (small step toward plugin/strategy style) ---
    def _run_layout_grid(_paths):
        # --- Tempo 配置（賑やか/静かの交互） ---
        _paths2 = _paths
        try:
            if (not bool(globals().get('PRESERVE_INPUT_ORDER', False))) and ARRANGE_TEMPO_ENABLE:
                _paths2 = _arrange_by_tempo(_paths2, ARRANGE_TEMPO_MODE)
        except Exception as e:
            _warn_exc_once(e)
            _paths2 = _paths
        canvas, mask, info, r, c = layout_grid(_paths2, WIDTH, HEIGHT, MARGIN, GUTTER, ROWS, COLS, MODE, bg)
        return canvas, mask, info, r, c, _paths2

    def _run_layout_hex(_paths):
        # 六角レイアウトを選択した場合は、ランダムモードでもラッパが動作するように一時的に
        # KANA_FORCE_HEX を "on" に設定します。戻すときは元の値を復元します。
        _old_force_hex = globals().get("KANA_FORCE_HEX", "off")
        try:
            globals()["KANA_FORCE_HEX"] = "on"
            canvas, mask, info, r, c = layout_grid(_paths, WIDTH, HEIGHT, MARGIN, GUTTER, ROWS, COLS, MODE, bg)
        finally:
            globals()["KANA_FORCE_HEX"] = _old_force_hex
        return canvas, mask, info, r, c, _paths

    def _run_layout_mosaic_uh(_paths):
        canvas, mask, info = layout_mosaic_uniform_height(_paths, WIDTH, HEIGHT, MARGIN, GUTTER, bg)
        return canvas, mask, info, 0, 0, _paths

    def _run_layout_mosaic_uw(_paths):
        canvas, mask, info = layout_mosaic_uniform_width(_paths, WIDTH, HEIGHT, MARGIN, GUTTER, bg)
        return canvas, mask, info, 0, 0, _paths


    def _run_layout_quilt(_paths):
        canvas, mask, info = layout_quilt_bsp(_paths, WIDTH, HEIGHT, MARGIN, GUTTER, bg, rng=rng)
        return canvas, mask, info, 0, 0, _paths

    def _run_layout_stained_glass(_paths):
        canvas, mask, info = layout_stained_glass_voronoi(_paths, WIDTH, HEIGHT, MARGIN, GUTTER, bg, rng=rng)
        return canvas, mask, info, 0, 0, _paths

    _LAYOUT_REGISTRY = {
        "grid": _run_layout_grid,
        "hex": _run_layout_hex,
        "mosaic-uniform-height": _run_layout_mosaic_uh,
        "mosaic-uniform-width": _run_layout_mosaic_uw,
        "quilt": _run_layout_quilt,
        "stained-glass": _run_layout_stained_glass,
    }

    effective_style = chosen if style == "random" else style
    start=time.perf_counter()
    runner = _LAYOUT_REGISTRY.get(effective_style, _run_layout_mosaic_uw)
    canvas, mask, info, r, c, picked_paths = runner(picked_paths)
    rows_used, cols_used = r, c
    layout_info.update(info)

    # 明るさ調整（背景マスク込み）
    effects_enable = bool(globals().get("EFFECTS_ENABLE", True))
    # 注: EFFECTS_ENABLE=False のときは、ここから下の“エフェクト群”を一括でスキップします。

    # エフェクト適用用のマスク（ステンドグラスのみ切替可能）
    fx_mask = mask
    try:
        if effects_enable and str(effective_style) == "stained-glass":
            mode = str(globals().get("STAINED_GLASS_EFFECTS_APPLY_MODE", "global") or "global").strip().lower()
            include_lead = bool(globals().get("STAINED_GLASS_EFFECTS_INCLUDE_LEAD", True))
            if mode in ("global", "full", "all"):
                fx_mask = Image.new("L", canvas.size, 255)
            else:
                fx_mask = mask
                # mask 系のみ: 鉛線も含める（境界での段差を減らす狙い）
                if include_lead and mode.startswith("mask"):
                    try:
                        lw = int(globals().get("STAINED_GLASS_LEAD_WIDTH", 6))
                        pad = max(1, int(round(lw * 0.75)))
                        k = pad * 2 + 1
                        fx_mask = fx_mask.filter(ImageFilter.MaxFilter(size=k))
                    except Exception as e:
                        _kana_silent_exc('core:L17776', e)
                        pass
                if mode in ("mask_feather", "feather", "soft"):
                    try:
                        lw = int(globals().get("STAINED_GLASS_LEAD_WIDTH", 6))
                        r = max(1, int(round(lw * 0.75)))
                        fx_mask = fx_mask.filter(ImageFilter.GaussianBlur(radius=r))
                    except Exception as e:
                        _kana_silent_exc('core:L17783', e)
                        pass
    except Exception:
        fx_mask = mask

    pre = mean_luma_masked(canvas, fx_mask)    # --- 各種エフェクト適用前の明るさ ---

    # --- 明るさ調整（エフェクト前） ---
    # オート明るさ/マニュアル明るさを使う場合、まず先に明るさを整えてから
    # Halation や彩度・トーンなどの“味付け”を行う方が、狙い通りになりやすいです。
    if effects_enable:
        canvas, binfo = adjust_brightness_with_mask(canvas, fx_mask)
    else:
        # エフェクト全体OFF：明るさ調整や後段エフェクトをスキップ
        binfo = {
            "method": "off",
            "target": globals().get("AUTO_TARGET_MEAN", 0.50),
            "gain": None,
            "gamma": None,
            "final_mean": pre,
            "original_mean": pre,
        }


    # --- Shadow/Highlight（明るさ調整の後 / Clarity の前） ---
    try:
        if effects_enable and bool(globals().get("SHADOWHIGHLIGHT_ENABLE", False)):
            sa = float(globals().get("SHADOW_AMOUNT", 0.22))
            ha = float(globals().get("HIGHLIGHT_AMOUNT", 0.18))
            canvas = _apply_shadow_highlight(canvas, sa, ha, content_mask=fx_mask)
    except Exception as e:
        _warn_exc_once(e)
        pass


    # --- ノイズ除去（Shadow/Highlight の後 / Dehaze の前） ---
    try:
        if effects_enable:
            nmode = str(globals().get("DENOISE_MODE", "off")).strip().lower()
            if nmode and nmode not in ("off", "none", "0", "false"):
                nst = float(globals().get("DENOISE_STRENGTH", 0.25))
                canvas = _apply_denoise(canvas, mode=nmode, strength=nst, content_mask=fx_mask)
    except Exception as e:
        _warn_exc_once(e)
        pass

    # --- Dehaze（Shadow/Highlight の後 / Clarity の前） ---
    try:
        if effects_enable and bool(globals().get("DEHAZE_ENABLE", False)):
            da = float(globals().get("DEHAZE_AMOUNT", 0.10))
            dr = int(globals().get("DEHAZE_RADIUS", 24))
            canvas = _apply_dehaze(canvas, da, dr, content_mask=fx_mask)
    except Exception as e:
        _warn_exc_once(e)
        pass

# --- Clarity（局所コントラスト：ハレーション前） ---
    # ハレーションの前段で局所コントラストを整えると、
    # “光は柔らかいのに輪郭は締まる”方向に寄せやすいです。
    try:
        if effects_enable and bool(globals().get("CLARITY_ENABLE", False)):
            camount = globals().get("CLARITY_AMOUNT", 0.12)
            crad = globals().get("CLARITY_RADIUS", 2.0)
            canvas = _apply_clarity(canvas, camount, crad, content_mask=fx_mask)
    except Exception as e:
        _warn_exc_once(e)
        pass
    # --- アンシャープ（輪郭の締め：クラリティ後 / ハレーション前） ---
    # クリアさ（クラリティ）の後に、輪郭を少しだけ締めます。
    try:
        if effects_enable and bool(globals().get("UNSHARP_ENABLE", False)):
            uamt = globals().get("UNSHARP_AMOUNT", 0.35)
            urad = globals().get("UNSHARP_RADIUS", 1.2)
            uth = globals().get("UNSHARP_THRESHOLD", 3)
            canvas = _apply_unsharp_mask(canvas, uamt, urad, uth, content_mask=fx_mask)
    except Exception as e:
        _warn_exc_once(e)
        pass

    # --- Bloom/Halation エフェクト（有効な場合のみ） ---
    try:
        # ハレーション処理は有効なときのみ行います。失敗してもそのまま進行します。
        if effects_enable and bool(globals().get("HALATION_ENABLE", False)):
            canvas = _apply_halation_bloom(canvas, HALATION_INTENSITY, HALATION_RADIUS, threshold=HALATION_THRESHOLD, knee=HALATION_KNEE)
    except Exception as e:
        _warn_exc_once(e)
        pass

    # --- 独立したアートエフェクト群（粒状以外） ---
    # ここから下のエフェクトはハレーションの有無に関係なく適用できます。各フラグを組み合わせて自由に調整してください。
    try:
        if effects_enable:
            # 白黒（グレースケール）エフェクト
            if bool(globals().get("BW_EFFECT_ENABLE", False)):
                canvas = _apply_bw_effect(canvas)
            # セピアエフェクト
            if bool(globals().get("SEPIA_ENABLE", False)):
                intensity = globals().get("SEPIA_INTENSITY", 0.35)
                canvas = _apply_sepia(canvas, intensity)
            # 彩度ブースト（ビブランス）エフェクト
            if bool(globals().get("VIBRANCE_ENABLE", False)):
                vfact = globals().get("VIBRANCE_FACTOR", 1.2)
                canvas = _apply_vibrance(canvas, vfact)
            # スプリットトーン（影・ハイライトに色味）
            if bool(globals().get("SPLIT_TONE_ENABLE", False)):
                shh = globals().get("SPLIT_TONE_SHADOW_HUE", 220.0)
                shs = globals().get("SPLIT_TONE_SHADOW_STRENGTH", 0.06)
                hih = globals().get("SPLIT_TONE_HIGHLIGHT_HUE", 35.0)
                his = globals().get("SPLIT_TONE_HIGHLIGHT_STRENGTH", 0.05)
                bal = globals().get("SPLIT_TONE_BALANCE", 0.0)
                canvas = _apply_split_tone(canvas, shh, shs, hih, his, bal, content_mask=fx_mask)
            # トーンカーブ（階調）エフェクト（グレインの前）
            if bool(globals().get("TONECURVE_ENABLE", False)):
                tmode = globals().get("TONECURVE_MODE", "film")
                tstr = globals().get("TONECURVE_STRENGTH", 0.35)
                canvas = _apply_tonecurve(canvas, mode=tmode, strength=tstr, content_mask=fx_mask)

            # LUT（.cube）カラーグレーディング（トーンカーブの後 / グレインの前）
            if bool(globals().get("LUT_ENABLE", False)):
                lpath = str(globals().get("LUT_FILE", "")).strip()
                if lpath:
                    lstr = float(globals().get("LUT_STRENGTH", 0.30))
                    canvas = _apply_lut_cube(canvas, lpath, lstr, content_mask=fx_mask)
    except Exception as e:
        _warn_exc_once(e)
        pass

    # --- フィルムグレイン（最後に質感を足す） ---
    try:
        if effects_enable and bool(globals().get("GRAIN_ENABLE", False)):
            amount = globals().get("GRAIN_AMOUNT", 0.05)
            canvas = _apply_grain(canvas, amount, content_mask=fx_mask)
    except Exception as e:
        _warn_exc_once(e)
        pass

    # --- Vignette（明るさ調整の後） ---
    try:
        _ = canvas
        if effects_enable and VIGNETTE_ENABLE:
            canvas = _apply_vignette(canvas, VIGNETTE_STRENGTH, VIGNETTE_ROUND)
    except NameError:
        pass

    if pre is not None: binfo["original_mean"]=pre

    banner(_lang("明るさ 調整","Brightness adjustment"))
    if not effects_enable:
        note(_lang('EFFECTS_ENABLE=off：エフェクトはスキップしました。','EFFECTS_ENABLE=off: effects were skipped.'))
    mb = binfo.get("original_mean"); ma = binfo.get("final_mean"); tgt = binfo.get("target")
    method = binfo.get("method") or BRIGHTNESS_MODE
    if mb is None:
        note(_lang("調整: なし（対象領域が検出できませんでした）","Adjustment: none (no target region detected)"))
    else:
        note(f"平均: {fmt_num(mb)} → {fmt_num(ma)}（目標 {fmt_num(tgt,2)}）")
        if binfo.get("gain")  is not None:  note(f"Gain: ×{fmt_num(binfo['gain'])}")
        if binfo.get('gamma') is not None:
            if binfo.get('gamma_raw') is not None:
                note(f"Gamma(raw→soft): x{fmt_num(binfo['gamma_raw'])} → x{fmt_num(binfo['gamma'])}")
            else:
                note(f"Gamma: x{fmt_num(binfo['gamma'])}")
        note(f"モード/メソッド: {BRIGHTNESS_MODE} / {method}")

    # 保存
    out_path = choose_output_path(Path(IMAGE_SAVE_DIR), IMAGE_BASENAME, FORMAT)
    save_kwargs = {"compress_level":6} if FORMAT.lower().startswith("png") else {"quality":95,"optimize":True,"progressive":True,"subsampling":1}
    canvas.save(out_path, **save_kwargs)
    # まとめ（英語固定で統一）
    note(f"Done: {time.perf_counter()-start:.2f}s | output={out_path}")

    # 使用リスト
    write_used_lists(picked_paths, rows_used, cols_used, seed_used, targets, layout_info, binfo, log_dir=Path(LOG_SAVE_DIR))

    # 壁紙更新（コンソール表示はローカライズ対応）
    try:
        tag = _mode_tag_for_console()  # Fill/Fit/Uniform Height/Uniform Width を安全に取得

        if APPLY_WALLPAPER:
            if out_path.exists() and out_path.stat().st_size > 0:
                set_wallpaper(out_path, style="Fill")
            print(C("92;1", "\n" + _lang(
                f"壁紙を更新しました（{tag}）。お楽しみください！",
                f"Wallpaper updated ({tag}). Enjoy!"
            )))
        else:
            print(C("90", "\n" + _lang(
                f"壁紙更新はOFFです（{tag}）。画像は保存しました。",
                f"Wallpaper update is OFF ({tag}). Image saved."
            )))
    except Exception as e:
        print(C("91;1", _lang(
            f"[警告] 壁紙設定に失敗: {e}",
            f"[WARN] Failed to set wallpaper: {e}"
        )))

    # 保存しない設定なら削除
    if not SAVE_IMAGE:
        try:
            out_path.unlink(missing_ok=True)
            note(_lang(
                "SAVE_IMAGE=False：作成画像は削除しました。",
                "SAVE_IMAGE=False: output image was deleted."
            ))
        except Exception as e:
            print(C("91;1", _lang(
                f"[警告] 画像削除に失敗: {e}",
                f"[WARN] Failed to delete image: {e}"
            )))


# --- Unsharp Mask（輪郭：ハレーション前） ---
# --- LAYOUT エイリアス（gridsafe 有効時は無効化） ---
try:
    _KANA_HEX_ALIAS = False
    _KANA_ORIG_LAYOUT_NAME = str(globals().get('LAYOUT_STYLE',''))
except Exception:
    _KANA_HEX_ALIAS = False
except Exception as e:
    _warn_exc_once(e)
    pass
# 【KANA修正】ラッパーが二重に適用されないように無効化しました。
_SQRT3_2 = (3 ** 0.5) * 0.5
_HEX_NAMES = ("hex", "honeycomb", "hex-tight")

def _hexmask_square(S:int):
    from PIL import Image, ImageDraw, ImageFilter
    h = int(round(_SQRT3_2 * S))
    oy = (S - h)//2
    pts = [(0.25*S, oy + 0),
           (0.75*S, oy + 0),
           (1.00*S, oy + h/2.0),
           (0.75*S, oy + h),
           (0.25*S, oy + h),
           (0.00*S, oy + h/2.0)]
    m = Image.new("L", (S,S), 0)
    ImageDraw.Draw(m).polygon([(int(x),int(y)) for x,y in pts], fill=255)
    d = int(max(0, min(2, float(globals().get("HEX_TIGHT_DILATE", 1)))))
    if d > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=d*0.6))
        m = m.point(lambda v: 255 if v >= 128 else 0, mode="1").convert("L")
    return m

# --- フェイスフォーカス用クロップ処理（OpenCV Haar があれば使用） ---
def _paste_with_clip(dest, src, mask, bx, by):
    """src を mask で dest に貼り付ける（部分的に画面外でも安全）。

    速度重視: タイルが完全に画面内に収まる場合は crop を避けて高速に貼ります。
    画面端のみ従来どおり crop して安全にクリップします。
    """
    from PIL import Image
    W, H = dest.size
    S = src.size[0]
    # Fast path: 完全に画面内なら crop を避ける
    if 0 <= bx and 0 <= by and (bx + S) <= W and (by + S) <= H:
        dest.paste(src, (bx, by), mask)
        return True

    x0 = max(0, bx); y0 = max(0, by)
    x1 = min(W, bx + S); y1 = min(H, by + S)
    if x0 >= x1 or y0 >= y1:
        return False
    cx0 = x0 - bx; cy0 = y0 - by
    cx1 = cx0 + (x1 - x0); cy1 = cy0 + (y1 - y0)
    dest.paste(src.crop((cx0, cy0, cx1, cy1)), (x0, y0), mask.crop((cx0, cy0, cx1, cy1)))
    return True

def _hexmask_square(S:int):
    from PIL import Image, ImageDraw, ImageFilter
    h = int(round(_SQRT3_2 * S))
    oy = (S - h)//2
    pts = [(0.25*S, oy + 0),
           (0.75*S, oy + 0),
           (1.00*S, oy + h/2.0),
           (0.75*S, oy + h),
           (0.25*S, oy + h),
           (0.00*S, oy + h/2.0)]
    m = Image.new("L", (S,S), 0)
    ImageDraw.Draw(m).polygon([(int(x),int(y)) for x,y in pts], fill=255)
    d = int(max(0, min(2, float(globals().get("HEX_TIGHT_DILATE", 1)))))
    if d > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=d*0.6))
        m = m.point(lambda v: 255 if v >= 128 else 0, mode="1").convert("L")
    return m

def _paste_with_clip(dest, src, mask, bx, by):
    """(bx,by) にタイルを貼り付ける。画面外はクリップし、完全内側は高速パス。

    戻り値: 実際に貼り付けたら True（交差なしなら False）
    """
    try:
        dw, dh = dest.size
        sw, sh = src.size
        x0, y0 = int(bx), int(by)
        x1, y1 = x0 + sw, y0 + sh

        # 完全にキャンバス内なら、crop なしでそのまま paste（高速）
        if x0 >= 0 and y0 >= 0 and x1 <= dw and y1 <= dh:
            dest.paste(src, (x0, y0), mask)
            return True

        # 交差領域を計算
        ix0, iy0 = max(0, x0), max(0, y0)
        ix1, iy1 = min(dw, x1), min(dh, y1)
        if ix0 >= ix1 or iy0 >= iy1:
            return False

        # src/mask 側の crop 領域
        cx0, cy0 = ix0 - x0, iy0 - y0
        cx1, cy1 = cx0 + (ix1 - ix0), cy0 + (iy1 - iy0)
        dest.paste(src.crop((cx0, cy0, cx1, cy1)), (ix0, iy0), mask.crop((cx0, cy0, cx1, cy1)))
        return True
    except Exception:
        return False

def _kana_anime_eye_pair_boxes(gray2: Any) -> Optional[Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]]:
    """アニメ/AI絵向け: 目（2つ）の候補をエッジベースで探します。

    入力:
      gray2: 2D グレースケール（numpy ndarray 相当）
    戻り値:
      (box1, box2) または None
        box = (x, y, w, h)
    """
    try:
        import cv2  # type: ignore
        H, W = gray2.shape[:2]
        if H <= 0 or W <= 0:
            return None

        # エッジ抽出（軽量）
        g = cv2.GaussianBlur(gray2, (0, 0), 1.0)
        e = cv2.Canny(g, 50, 150)
        e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        cnts = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]

        min_dim = min(H, W)
        minw = max(8, int(float(globals().get('FACE_FOCUS_ANIME_EYE_MIN_FRAC', 0.08)) * float(min_dim)))
        minh = max(6, int(0.65 * minw))
        maxw = int(float(globals().get('FACE_FOCUS_ANIME_EYE_MAX_W_FRAC', 0.60)) * float(W))
        maxh = int(float(globals().get('FACE_FOCUS_ANIME_EYE_MAX_H_FRAC', 0.45)) * float(H))

        boxes: List[Tuple[int,int,int,int]] = []
        img_area = float(H * W)
        for c in contours or []:
            x, y, w, h = cv2.boundingRect(c)
            if w < minw or h < minh:
                continue
            if w > maxw or h > maxh:
                continue
            # 目っぽい横長（アニメ目はかなり横長になりやすい）
            r = w / float(h + 1e-6)
            if r < 0.9 or r > 7.0:
                continue
            if (w * h) < max(36.0, 0.003 * img_area):
                continue
            boxes.append((int(x), int(y), int(w), int(h)))

        if len(boxes) < 2:
            return None

        # 大きい候補から上位だけ評価（重すぎ防止）
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        boxes = boxes[:30]

        best = None
        dy_frac = float(globals().get('FACE_FOCUS_ANIME_EYE_PAIR_MAX_DY_FRAC', 0.35))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                b1 = boxes[i]
                b2 = boxes[j]
                if b1[0] > b2[0]:
                    b1, b2 = b2, b1
                x1, y1, w1, h1 = b1
                x2, y2, w2, h2 = b2

                # 重なり/近すぎを除外
                if x1 + w1 > x2:
                    continue

                c1x, c1y = x1 + w1 * 0.5, y1 + h1 * 0.5
                c2x, c2y = x2 + w2 * 0.5, y2 + h2 * 0.5
                if abs(c1y - c2y) > dy_frac * max(h1, h2):
                    continue

                # サイズ類似
                if max(w1, w2) / float(max(1, min(w1, w2))) > 2.2:
                    continue
                if max(h1, h2) / float(max(1, min(h1, h2))) > 2.2:
                    continue

                dx = c2x - c1x
                avgw = (w1 + w2) * 0.5
                if dx < 0.55 * avgw or dx > 7.0 * avgw:
                    continue

                # スコア（面積重視 + 縦ズレ減点）
                score = (w1 * h1 + w2 * h2) - 2.0 * abs(c1y - c2y)
                if best is None or score > best[0]:
                    best = (score, b1, b2)

        if best is None:
            return None
        return best[1], best[2]
    except Exception:
        return None


def _kana_anime_eye_pair_boxes_haar(gray2: Any) -> Optional[Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]]:
    """アニメ/AI絵向け: Haarの eye カスケードで「左右の目ペア」を推定します。

    目的:
      - エッジベース（Canny+contour）の目ペアが取りにくい絵（淡い色、ハイライト多め等）でも
        目カスケードの誤検出をスコアリングで抑えつつ、ペアを作る。

    戻り値:
      - (box1, box2) または None
      - box = (x, y, w, h)
    """
    try:
        import cv2  # type: ignore
        H, W = gray2.shape[:2]
        if H <= 0 or W <= 0:
            return None

        # 目の候補サイズ（ROIの短辺に対する割合）
        try:
            min_frac = float(globals().get('FACE_FOCUS_ANIME_EYE_MIN_FRAC', 0.08))
        except Exception:
            min_frac = 0.08
        min_dim = min(H, W)
        minw = max(8, int(min_frac * float(min_dim)))
        minh = max(6, int(0.65 * float(minw)))

        # カスケードの近傍数（小さめにして拾いやすくし、後段でスコアリング）
        try:
            neigh = int(globals().get('FACE_FOCUS_ANIME_EYE_HAAR_NEIGHBORS', 2) or 2)
        except Exception:
            neigh = 2

        boxes = []
        for cas_name in ('haarcascade_eye.xml', 'haarcascade_eye_tree_eyeglasses.xml'):
            try:
                cas = cv2.CascadeClassifier(cv2.data.haarcascades + cas_name)
                det = cas.detectMultiScale(
                    gray2,
                    scaleFactor=1.1,
                    minNeighbors=neigh,
                    flags=cv2.CASCADE_SCALE_IMAGE,
                    minSize=(minw, minh),
                )
                if det is None:
                    continue
                for (x, y, w, h) in det:
                    # 極端な縦横比を除外
                    r = float(w) / float(h + 1e-6)
                    if r < 0.35 or r > 3.5:
                        continue
                    # 画面上側優先（顔は上に出やすい）
                    if (y + 0.5 * h) > float(globals().get('FACE_FOCUS_TOP_FRAC', 0.70) or 0.70) * float(H):
                        continue
                    boxes.append((int(x), int(y), int(w), int(h)))
            except Exception as e:
                _kana_silent_exc('core:L18239', e)
                continue
        if len(boxes) < 2:
            return None

        # 重複除去（ざっくり）
        boxes.sort(key=lambda b: (b[0], b[1], -b[2]*b[3]))
        uniq = []
        for b in boxes:
            x, y, w, h = b
            keep = True
            for (x2, y2, w2, h2) in uniq[-8:]:
                if abs(x-x2) < 6 and abs(y-y2) < 6 and abs(w-w2) < 8 and abs(h-h2) < 8:
                    keep = False
                    break
            if keep:
                uniq.append(b)
        boxes = uniq
        if len(boxes) < 2:
            return None

        # ベストペア選択
        best = None
        for i in range(len(boxes)):
            x1, y1, w1, h1 = boxes[i]
            c1x, c1y = x1 + w1 * 0.5, y1 + h1 * 0.5
            for j in range(i+1, len(boxes)):
                x2, y2, w2, h2 = boxes[j]
                c2x, c2y = x2 + w2 * 0.5, y2 + h2 * 0.5

                # 左右判定
                if c2x <= c1x:
                    continue

                dy = abs(c2y - c1y)
                avg_h = 0.5 * (h1 + h2)
                if dy > 0.45 * avg_h:
                    continue

                # サイズ近似
                size_pen = abs(w1 - w2) / max(w1, w2) + abs(h1 - h2) / max(h1, h2)
                if size_pen > 1.2:
                    continue

                dist = (c2x - c1x)
                avg_w = 0.5 * (w1 + w2)
                # 近すぎ・遠すぎを避ける
                if dist < 0.6 * avg_w:
                    continue
                if dist > 10.0 * avg_w:
                    continue

                # 期待距離（経験則）。外れすぎをペナルティ。
                exp = 2.4 * avg_w
                dist_pen = abs(dist - exp) / max(1e-6, exp)

                # 上側ほど加点
                y_norm = 0.5 * (c1y + c2y) / max(1.0, float(H))
                pos_bonus = (1.0 - y_norm) * 0.25

                area_bonus = (w1*h1 + w2*h2) / max(1.0, float(H*W))
                score = pos_bonus + 0.20 * area_bonus - (1.6 * size_pen + 1.0 * dist_pen + 0.35 * (dy / max(1e-6, avg_h)))

                if best is None or score > best[0]:
                    best = (score, (x1, y1, w1, h1), (x2, y2, w2, h2))

        if best is None:
            return None
        _, b1, b2 = best
        # ensure left-to-right
        if (b2[0] + b2[2]*0.5) < (b1[0] + b1[2]*0.5):
            b1, b2 = b2, b1
        return b1, b2
    except Exception:
        return None


def _kana_anime_face_detect(gray: Any, top_frac: float, max_dim: int) -> Optional[Tuple[int,int,int,int]]:
    """アニメ/AI絵向け: eye-pair から擬似 face bbox を推定します。

    - Haar の face 検出が失敗したときの救済用
    - 誤爆を減らすため「上側(top_frac)のみ」を探索します
    """
    try:
        import cv2  # type: ignore
        H0, W0 = gray.shape[:2]
        if H0 <= 0 or W0 <= 0:
            return None
        # 縮小して探索
        sc = 1.0
        md = int(max_dim) if isinstance(max_dim, int) and max_dim > 0 else 720
        if max(H0, W0) > md:
            sc = float(md) / float(max(H0, W0))
        if sc < 1.0:
            g = cv2.resize(gray, (int(W0 * sc), int(H0 * sc)), interpolation=cv2.INTER_AREA)
        else:
            g = gray
        H, W = g.shape[:2]
        ylim = int(max(1, min(H, round(H * float(top_frac)))))
        roi = g[:ylim, :]

        # まず Haar eye ペア（より直接的）を試し、取れなければエッジベースへフォールバック
        pair = None
        if bool(globals().get('FACE_FOCUS_ANIME_EYE_HAAR_ENABLE', True)):
            pair = _kana_anime_eye_pair_boxes_haar(roi)
        if pair is None:
            pair = _kana_anime_eye_pair_boxes(roi)
        if pair is None:
            return None
        (x1, y1, w1, h1), (x2, y2, w2, h2) = pair
        c1x, c1y = x1 + w1 * 0.5, y1 + h1 * 0.5
        c2x, c2y = x2 + w2 * 0.5, y2 + h2 * 0.5
        mx = (c1x + c2x) * 0.5
        my = (c1y + c2y) * 0.5
        eye_span = max(1.0, (c2x - c1x))

        # 目間距離から顔サイズを推定（経験則）
        face_w = eye_span / 0.55
        face_h = face_w * 1.35
        x0 = mx - face_w * 0.5
        y0 = my - face_h * 0.35
        # clamp
        x0 = max(0.0, min(float(W - 1), x0))
        y0 = max(0.0, min(float(H - 1), y0))
        x1f = max(0.0, min(float(W), x0 + face_w))
        y1f = max(0.0, min(float(H), y0 + face_h))
        bw = max(1.0, x1f - x0)
        bh = max(1.0, y1f - y0)

        # 元解像度へ
        x = int(round(x0 / sc))
        y = int(round(y0 / sc))
        w = int(round(bw / sc))
        h = int(round(bh / sc))
        # 範囲内
        x = max(0, min(W0 - 1, x))
        y = max(0, min(H0 - 1, y))
        w = max(1, min(W0 - x, w))
        h = max(1, min(H0 - y, h))
        return (int(x), int(y), int(w), int(h))
    except Exception:
        return None


def _kana_face_eye_verify(gray: Any, rect: Tuple[int, int, int, int]) -> Tuple[bool, int]:
    """顔矩形内で目が検出できるかの簡易チェックです。

    戻り値:
      - eyes_ok    : 目が所定数以上検出できたか
      - eyes_count : 検出された目の数（簡易）
    """
    import cv2
    global _FDBG2
    x,y,w,h = rect
    roi = gray[max(0,y):y+h, max(0,x):x+w]
    if roi.size == 0: return False, 0
    eh = max(1, int(0.6 * roi.shape[0]))
    # 目は顔矩形の上側に出やすいので、上60%だけを対象にして高速化＆誤検出低減します。
    roi2 = roi[0:eh,:]
    eyes = 0
    try:
        c1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        e1 = c1.detectMultiScale(roi2, scaleFactor=1.1, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE,
                                 minSize=(max(10,int(0.12*min(roi2.shape))),)*2)
        # 目検出の最小サイズ：ROI短辺の約12%（最小10px）。小さすぎる誤検出を避けるためです。
        eyes += 0 if e1 is None else len(e1)
    except Exception: pass
    try:
        c2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        e2 = c2.detectMultiScale(roi2, scaleFactor=1.1, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE,
                                 minSize=(max(10,int(0.12*min(roi2.shape))),)*2)
        # 眼鏡あり用カスケードでも同じ閾値を使います（過剰検出を抑えるため）。
        eyes += 0 if e2 is None else len(e2)
    except Exception: pass

    # アニメ/AI絵向けの目検出（Haar が外れる場合の救済）
    try:
        eye_min = int(globals().get("FACE_FOCUS_EYE_MIN", 1) or 1)
    except Exception:
        eye_min = 1

    if (eyes < eye_min) and bool(globals().get("FACE_FOCUS_ANIME_EYES_ENABLE", True)):
        try:
            # まず Haar eye ペアを試し、ダメならエッジベースへ
            pair = None
            if bool(globals().get('FACE_FOCUS_ANIME_EYE_HAAR_ENABLE', True)):
                pair = _kana_anime_eye_pair_boxes_haar(roi2)
            if pair is None:
                pair = _kana_anime_eye_pair_boxes(roi2)
            if pair is not None:
                try:
                    _FDBG2["anime_eyes_ok"] += 1
                except Exception as e:
                    _kana_silent_exc('core:L18432', e)
                    pass
                # ペアが見つかったら「2目」扱い
                eyes = max(eyes, 2)
            else:
                try:
                    _FDBG2["anime_eyes_ng"] += 1
                except Exception as e:
                    _kana_silent_exc('core:L18439', e)
                    pass
        except Exception:
            try:
                _FDBG2["errors"] += 1
            except Exception as e:
                _kana_silent_exc('core:L18444', e)
                pass
    return (eyes >= eye_min), eyes

def _cover_square_face_focus(im, S: int, src_path=None):
    """
    S×S の正方形タイルを作るために、画像を「拡大（fill）→クロップ」します。
    顔（正面/横顔）・上半身・サリエンシー（目立つ領域）の候補を `_get_focus_candidates()` で取得し、
    候補があればその中心を基準にズーム倍率とクロップ位置を決めます。
    候補が無い／検出が無効／例外が起きた場合は、中央クロップ（必要ならバイアス込み）にフォールバックします。
    ※デバッグ用に `_FDBG`／`_FDBG2` のカウンタが更新されます。
    """
    from PIL import Image
    global _FDBG, _FDBG2
    iw, ih = im.size
    # ターゲット（S×S）を埋めるための基準倍率（fill）。ゼロ除算を避けるため空画像なら 1.0。
    base_sc = max(S / iw, S / ih) if iw > 0 and ih > 0 else 1.0

    def clamp_zoom(sc: float) -> float:
        """ズーム倍率を設定された最小/最大の範囲に収めます。"""
        zmin = float(globals().get("FACE_FOCUS_ZOOM_MIN", 1.0)) * base_sc
        zmax = float(globals().get("FACE_FOCUS_ZOOM_MAX", 2.0)) * base_sc
        return max(zmin, min(zmax, sc))

    def crop_center(im2: Image.Image, cx: float, cy: float) -> Image.Image:
        """中心 (cx, cy) を基準に、im2 から S×S を切り出します。"""
        tw, th = im2.size
        x0 = int(round(cx - S / 2.0))
        y0 = int(round(cy - S / 2.0))
        # 画像外を切り出さないように、開始座標を画像内に丸めます
        x0 = max(0, min(tw - S, x0))
        y0 = max(0, min(th - S, y0))
        return im2.crop((x0, y0, x0 + S, y0 + S))

    def crop_center_keep_box(im2, cx, cy, box):
        """中心 (cx,cy) を基準に S×S で切り出しつつ、box が窓内に収まるように微調整する。"""
        tw, th = im2.size
        fx0, fy0, fx1, fy1 = box

        x0 = int(round(cx - S / 2.0))
        y0 = int(round(cy - S / 2.0))

        # box が窓からはみ出す場合、まずは box を優先して窓を寄せる
        if (fx1 - fx0) <= S:
            if x0 > fx0:
                x0 = int(fx0)
            if (x0 + S) < fx1:
                x0 = int(fx1 - S)
        if (fy1 - fy0) <= S:
            if y0 > fy0:
                y0 = int(fy0)
            if (y0 + S) < fy1:
                y0 = int(fy1 - S)

        x0 = max(0, min(tw - S, x0))
        y0 = max(0, min(th - S, y0))
        return im2.crop((x0, y0, x0 + S, y0 + S))

    def crop_center_keep_hex_face(im2, cx, cy, box):
        """Hex（正六角形マスク）向け: 顔が上端・斜めカットに欠けにくい位置へ自動でクロップ窓を補正する。"""
        tw, th = im2.size
        fx0, fy0, fx1, fy1 = box

        # 目標（S に対する比率）。HEX セクションの設定（HEX_FACE_SAFE_*）で微調整できます
        top_min = float(HEX_FACE_SAFE_TOP) * S
        bottom_min = float(HEX_FACE_SAFE_BOTTOM) * S
        x_center_min = float(HEX_FACE_SAFE_XCENTER_MIN) * S
        x_center_max = float(HEX_FACE_SAFE_XCENTER_MAX) * S
        y_center_min = float(HEX_FACE_SAFE_YCENTER_MIN) * S
        y_center_max = float(HEX_FACE_SAFE_YCENTER_MAX) * S
        top_band = float(HEX_FACE_SAFE_TOP_BAND) * S
        side_min_top = float(HEX_FACE_SAFE_SIDE_TOP) * S
        side_max_top = S - side_min_top

        x0 = int(round(cx - S / 2.0))
        y0 = int(round(cy - S / 2.0))

        # 反復で少しずつ補正（過剰なオフセットを避ける）
        for _ in range(4):
            # まず box を窓内に入れる（基本）
            if (fx1 - fx0) <= S:
                if x0 > fx0:
                    x0 = int(fx0)
                if (x0 + S) < fx1:
                    x0 = int(fx1 - S)
            if (fy1 - fy0) <= S:
                if y0 > fy0:
                    y0 = int(fy0)
                if (y0 + S) < fy1:
                    y0 = int(fy1 - S)

            x0 = max(0, min(tw - S, x0))
            y0 = max(0, min(th - S, y0))

            rx0, ry0 = fx0 - x0, fy0 - y0
            rx1, ry1 = fx1 - x0, fy1 - y0

            # 1) 上下の余白（欠け防止）
            if ry0 < top_min:
                y0 -= int(round(top_min - ry0))
            if ry1 > (S - bottom_min):
                y0 += int(round(ry1 - (S - bottom_min)))

            # 2) 顔中心の高さを雑に安定化（bias の固定値より自然になりやすい）
            ryc = (ry0 + ry1) / 2.0
            if ryc < y_center_min:
                y0 -= int(round(y_center_min - ryc))
            elif ryc > y_center_max:
                y0 += int(round(ryc - y_center_max))

            # 3) 斜めカット領域対策（顔が左右に寄り過ぎると欠けやすい）
            rxc = (rx0 + rx1) / 2.0
            if rxc < x_center_min:
                x0 -= int(round(x_center_min - rxc))
            elif rxc > x_center_max:
                x0 += int(round(rxc - x_center_max))

            # 4) 顔が上部にあるときは、さらに左右の安全域を強める
            if ry0 < top_band:
                if rx0 < side_min_top:
                    x0 -= int(round(side_min_top - rx0))
                if rx1 > side_max_top:
                    x0 += int(round(rx1 - side_max_top))

        x0 = max(0, min(tw - S, x0))
        y0 = max(0, min(th - S, y0))
        return im2.crop((x0, y0, x0 + S, y0 + S))

    try:
        # 顔フォーカスが無効なら検出処理は行わずフォールバックへ
        if not bool(globals().get("FACE_FOCUS_ENABLE", True)):
            raise RuntimeError("face focus disabled")
        # 統一ヘルパ `_get_focus_candidates()` で候補（顔/上半身/サリエンシー）を取得します
        cand = _get_focus_candidates(im, src_path)
        face = cand.get("face")
        person = cand.get("person") if bool(globals().get("FACE_FOCUS_USE_PERSON", True)) else None
        upper = cand.get("upper") if bool(globals().get("FACE_FOCUS_USE_UPPER", False)) else None
        sal = cand.get("saliency") if bool(globals().get("FACE_FOCUS_USE_SALIENCY", True)) else None
        # 顔候補があれば使用（正面/frontal を優先、無ければ横顔/profile）
        if face:
            name, x, y, w, h = face  # type: ignore
            cx = x + w / 2.0
            cy = y + h / 2.0
            # 顔の縦サイズがタイル内で占める比率（大きいほどズームが強くなります）
            target = float(globals().get("FACE_FOCUS_RATIO", 0.42))
            # 顔の高さからズーム倍率を計算します。base_sc（fill）より小さくならないようにし、
            # 顔がタイル外に切れやすいケースを抑えます。
            sc = clamp_zoom(max(base_sc, (target * S) / max(1.0, float(h))))
            tw, th = max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))
            im2 = im.resize((tw, th), Image.LANCZOS)
            # 設定に応じて「顔を厳密に中央」または「バイアスで少しずらす」を行います
            if bool(globals().get("FACE_FOCUS_CENTER_FACE", False)):
                cx = cx * sc
                cy = cy * sc
            else:
                bias_x = float(globals().get("FACE_FOCUS_BIAS_X", 0.0)) * S
                bias_y = float(globals().get("FACE_FOCUS_BIAS_Y", -0.10)) * S
                cx = cx * sc + bias_x
                cy = cy * sc + bias_y
            # デバッグ: 正面/横顔の採用回数をカウント
            if name == "profile":
                _FDBG["profile"] += 1
            elif name == "anime":
                _FDBG["anime"] += 1
            elif name == "ai":
                _FDBG["ai"] += 1
            else:
                _FDBG["frontal"] += 1
            # 顔がクロップ外に押し出される事故を避けるため、顔bboxが必ず窓内に収まるよう補正します。
            try:
                box = (float(x) * sc, float(y) * sc, float(x + w) * sc, float(y + h) * sc)
            except Exception:
                box = (0.0, 0.0, 0.0, 0.0)
            if bool(globals().get("_KANA_HEX_WANT", False)):
                return crop_center_keep_hex_face(im2, cx, cy, box)
            return crop_center_keep_box(im2, cx, cy, box)
        # Person-focus: 顔が無い／小さいときは人物領域（上半身/全身）へ寄せる
        if person:
            x, y, w, h = person  # type: ignore
            cx = x + w / 2.0
            # HOG の bbox は頭が欠けることがあるため、y 位置を状況で補正する
            try:
                y_top_norm = float(y) / max(1.0, float(ih))
            except Exception:
                y_top_norm = 0.0
            if y_top_norm > float(globals().get("FACE_FOCUS_PERSON_TOP_Y_SUSPECT", 0.10)):
                # bbox の上が肩付近まで落ちている可能性 → 少し上へ寄せる
                cy = y - float(globals().get("FACE_FOCUS_PERSON_HEAD_UP_FRAC", 0.12)) * h
            else:
                # bbox が頭から取れている → 上側を狙う（頭〜胸）
                cy = y + float(globals().get("FACE_FOCUS_PERSON_HEAD_CY_FRAC", 0.18)) * h
            cy = float(max(0.0, min(float(ih), cy)))
            sc = clamp_zoom(base_sc * 1.10)
            tw, th = max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))
            im2 = im.resize((tw, th), Image.LANCZOS)
            _FDBG["person"] += 1
            if bool(globals().get("FACE_FOCUS_CENTER_FACE", False)):
                return crop_center(im2, cx * sc, cy * sc)
            else:
                bias_x = float(globals().get("FACE_FOCUS_BIAS_X", 0.0)) * S
                bias_y = float(globals().get("FACE_FOCUS_BIAS_Y", -0.10)) * S
                return crop_center(im2, cx * sc + bias_x, cy * sc + bias_y)
        # 旧来互換: 上半身候補のみがある場合
        if upper:
            x, y, w, h = upper  # type: ignore
            cx = x + w / 2.0
            # HOG の bbox は頭が欠けることがあるため、y 位置を状況で補正する
            try:
                y_top_norm = float(y) / max(1.0, float(ih))
            except Exception:
                y_top_norm = 0.0
            if y_top_norm > float(globals().get("FACE_FOCUS_PERSON_TOP_Y_SUSPECT", 0.10)):
                # bbox の上が肩付近まで落ちている可能性 → 少し上へ寄せる
                cy = y - float(globals().get("FACE_FOCUS_PERSON_HEAD_UP_FRAC", 0.12)) * h
            else:
                # bbox が頭から取れている → 上側を狙う（頭〜胸）
                cy = y + float(globals().get("FACE_FOCUS_PERSON_HEAD_CY_FRAC", 0.18)) * h
            cy = float(max(0.0, min(float(ih), cy)))
            sc = clamp_zoom(base_sc * 1.10)
            tw, th = max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))
            im2 = im.resize((tw, th), Image.LANCZOS)
            _FDBG["upper"] += 1
            if bool(globals().get("FACE_FOCUS_CENTER_FACE", False)):
                return crop_center(im2, cx * sc, cy * sc)
            else:
                bias_x = float(globals().get("FACE_FOCUS_BIAS_X", 0.0)) * S
                bias_y = float(globals().get("FACE_FOCUS_BIAS_Y", -0.10)) * S
                return crop_center(im2, cx * sc + bias_x, cy * sc + bias_y)
        # サリエンシー候補があればそれを基準にクロップ（軽くズーム）
        if sal:
            cx, cy = sal  # type: ignore
            sc = clamp_zoom(base_sc * 1.10)
            tw, th = max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))
            im2 = im.resize((tw, th), Image.LANCZOS)
            _FDBG["saliency"] += 1
            if bool(globals().get("FACE_FOCUS_CENTER_FACE", False)):
                return crop_center(im2, cx * sc, cy * sc)
            else:
                bias_x = float(globals().get("FACE_FOCUS_BIAS_X", 0.0)) * S
                bias_y = float(globals().get("FACE_FOCUS_BIAS_Y", -0.10)) * S
                return crop_center(im2, cx * sc + bias_x, cy * sc + bias_y)
        # 候補が無い場合は中央クロップ（必要ならバイアス込み）
        sc = clamp_zoom(base_sc)
        tw, th = max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))
        im2 = im.resize((tw, th), Image.LANCZOS)
        _FDBG["center"] += 1
        if bool(globals().get("FACE_FOCUS_CENTER_FACE", False)):
            return crop_center(im2, tw / 2.0, th / 2.0)
        else:
            bias_x = float(globals().get("FACE_FOCUS_BIAS_X", 0.0)) * S
            bias_y = float(globals().get("FACE_FOCUS_BIAS_Y", -0.10)) * S
            return crop_center(im2, tw / 2.0 + bias_x, th / 2.0 + bias_y)
    except Exception:
        # 例外時（OpenCV不在など）は、検出を使わずに中央クロップへフォールバックします
        try:
            iw, ih = im.size
            sc = max(S / float(iw), S / float(ih))
            tw, th = max(1, int(iw * sc)), max(1, int(ih * sc))
            im2 = im.resize((tw, th), Image.LANCZOS)
            x0 = max(0, (tw - S) // 2)
            y0 = max(0, (th - S) // 2)
            return im2.crop((x0, y0, x0 + S, y0 + S))
        except Exception:
            # 最終手段: とにかく S×S にリサイズして返します
            try:
                return im.resize((S, S), Image.LANCZOS)
            except Exception:
                return im

# --- グリッド埋め込み用の矩形フェイスフォーカスクロップ ---
def _cover_rect_face_focus(im: Image.Image, cw: int, ch: int, src_path=None) -> Image.Image:
    """
    cw×ch の矩形タイルを作るために、画像を「拡大（fill）→クロップ」します。
    顔（正面/横顔）・上半身・サリエンシーの候補を `_get_focus_candidates()` で取得し、
    候補があればその中心を基準にズーム倍率とクロップ位置を決めます。
    候補が無い／検出が無効／例外が起きた場合は、中央クロップ（必要ならバイアス込み）にフォールバックします。
    ※デバッグ用に `_FDBG`／`_FDBG2` のカウンタが更新されます。
    """
    from PIL import Image
    global _FDBG, _FDBG2
    iw, ih = im.size
    # 出力矩形（cw×ch）を覆うための基準倍率（fill）。空画像なら 1.0。
    base_sc = max(cw / iw, ch / ih) if iw > 0 and ih > 0 else 1.0

    def clamp_zoom(sc: float) -> float:
        """
        基準倍率 base_sc に対し、倍率を指定範囲に収めます。

            - 返す値は FACE_FOCUS_ZOOM_MIN ～ FACE_FOCUS_ZOOM_MAX の間にクランプされます
        """
        zmin = float(globals().get("FACE_FOCUS_ZOOM_MIN", 1.0)) * base_sc
        zmax = float(globals().get("FACE_FOCUS_ZOOM_MAX", 2.0)) * base_sc
        return max(zmin, min(zmax, sc))

    def crop_center(im2: Image.Image, cx: float, cy: float) -> Image.Image:
        """中心 (cx, cy) を基準に、im2 から cw×ch を切り出します。"""
        tw, th = im2.size
        x0 = int(round(cx - cw / 2.0))
        y0 = int(round(cy - ch / 2.0))
        x0 = max(0, min(tw - cw, x0))
        y0 = max(0, min(th - ch, y0))
        return im2.crop((x0, y0, x0 + cw, y0 + ch))

    def crop_center_keep_box(im2: Image.Image, cx: float, cy: float, box: Tuple[float,float,float,float]) -> Image.Image:
        """中心 (cx,cy) を基本にしつつ、指定の box が必ずクロップ内に収まるように微調整します。"""
        tw, th = im2.size
        fx0, fy0, fx1, fy1 = box
        # 初期窓
        x0 = int(round(cx - cw / 2.0))
        y0 = int(round(cy - ch / 2.0))
        # box が窓に収まるよう補正（box が窓より大きい場合は中央優先で諦める）
        if (fx1 - fx0) <= cw:
            if x0 > fx0:
                x0 = int(fx0)
            if (x0 + cw) < fx1:
                x0 = int(fx1 - cw)
        if (fy1 - fy0) <= ch:
            if y0 > fy0:
                y0 = int(fy0)
            if (y0 + ch) < fy1:
                y0 = int(fy1 - ch)
        # clamp
        x0 = max(0, min(tw - cw, x0))
        y0 = max(0, min(th - ch, y0))
        return im2.crop((x0, y0, x0 + cw, y0 + ch))

    try:
        # 顔フォーカスが無効な場合は検出せずフォールバックへ
        if not bool(globals().get("FACE_FOCUS_ENABLE", True)):
            raise RuntimeError("face focus disabled")
        if not bool(globals().get("FACE_FOCUS_FORCE_ALL_MODES", True)):
            if not (bool(globals().get("GRID_FACE_FOCUS_ENABLE", False))
                    or bool(globals().get("QUILT_FACE_FOCUS_ENABLE", False))):
                raise RuntimeError("face focus disabled for this layout")
        # 統一ヘルパ `_get_focus_candidates()` で候補（顔/上半身/サリエンシー）を取得します
        cand = _get_focus_candidates(im, src_path)
        face = cand.get("face")
        # Person-focus は、基本的に上半身（upper）を含みつつ、必要なら HOG 人物検出を使います。
        person = cand.get("person") if bool(globals().get("FACE_FOCUS_USE_PERSON", True)) else None
        upper = cand.get("upper") if bool(globals().get("FACE_FOCUS_USE_UPPER", False)) else None
        sal = cand.get("saliency") if bool(globals().get("FACE_FOCUS_USE_SALIENCY", True)) else None
        # 顔候補: 高さ/幅の両方から倍率を計算し、タイル比率を保ちながら切り取りすぎを避けます
        if face:
            name, x, y, w, h = face  # type: ignore
            cx = x + w / 2.0
            cy = y + h / 2.0
            target = float(globals().get("FACE_FOCUS_RATIO", 0.42))
            sc_h = (target * ch) / max(1.0, float(h))
            width_frac = float(globals().get("FACE_FOCUS_WIDTH_FRAC", 1.0))
            sc_w = (width_frac * cw) / max(1.0, float(w))
            sc = clamp_zoom(max(base_sc, min(sc_h, sc_w)))
            tw, th = max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))
            im2 = im.resize((tw, th), Image.LANCZOS)
            # 矩形タイルでは常にバイアスを適用し、顔位置を少し調整できるようにします
            bias_x = float(globals().get("FACE_FOCUS_BIAS_X", 0.0)) * cw
            bias_y = float(globals().get("FACE_FOCUS_BIAS_Y", -0.10)) * ch
            cx = cx * sc + bias_x
            cy = cy * sc + bias_y
            if name == "profile":
                _FDBG["profile"] += 1
            elif name == "anime":
                _FDBG["anime"] += 1
            elif name == "ai":
                _FDBG["ai"] += 1
            else:
                _FDBG["frontal"] += 1
            # 顔bboxがクロップ外に押し出されないように、顔領域が必ず窓内に入るよう補正
            try:
                box = (float(x) * sc, float(y) * sc, float(x + w) * sc, float(y + h) * sc)
            except Exception:
                box = (0.0, 0.0, 0.0, 0.0)
            return crop_center_keep_box(im2, cx, cy, box)
        # 上半身候補: 上半身が収まるよう倍率を調整し、縦位置は少し上寄せのヒューリスティックを使います
        if person:
            x, y, w, h = person  # type: ignore
            cx = x + w / 2.0
            # HOG の bbox は頭が欠けることがあるため、y 位置を状況で補正する
            try:
                y_top_norm = float(y) / max(1.0, float(ih))
            except Exception:
                y_top_norm = 0.0
            if y_top_norm > float(globals().get("FACE_FOCUS_PERSON_TOP_Y_SUSPECT", 0.10)):
                # bbox の上が肩付近まで落ちている可能性 → 少し上へ寄せる
                cy = y - float(globals().get("FACE_FOCUS_PERSON_HEAD_UP_FRAC", 0.12)) * h
            else:
                # bbox が頭から取れている → 上側を狙う（頭〜胸）
                cy = y + float(globals().get("FACE_FOCUS_PERSON_HEAD_CY_FRAC", 0.18)) * h
            cy = float(max(0.0, min(float(ih), cy)))
            width_frac = float(globals().get("FACE_FOCUS_WIDTH_FRAC", 1.0))
            sc_w = (width_frac * cw) / max(1.0, float(w))
            sc = clamp_zoom(min(base_sc * 1.10, sc_w))
            tw, th = max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))
            im2 = im.resize((tw, th), Image.LANCZOS)
            _FDBG["person"] += 1
            bias_x = float(globals().get("FACE_FOCUS_BIAS_X", 0.0)) * cw
            bias_y = float(globals().get("FACE_FOCUS_BIAS_Y", -0.10)) * ch
            return crop_center(im2, cx * sc + bias_x, cy * sc + bias_y)
        # 旧来互換: 上半身候補のみがある場合
        if upper:
            x, y, w, h = upper  # type: ignore
            cx = x + w / 2.0
            # HOG の bbox は頭が欠けることがあるため、y 位置を状況で補正する
            try:
                y_top_norm = float(y) / max(1.0, float(ih))
            except Exception:
                y_top_norm = 0.0
            if y_top_norm > float(globals().get("FACE_FOCUS_PERSON_TOP_Y_SUSPECT", 0.10)):
                # bbox の上が肩付近まで落ちている可能性 → 少し上へ寄せる
                cy = y - float(globals().get("FACE_FOCUS_PERSON_HEAD_UP_FRAC", 0.12)) * h
            else:
                # bbox が頭から取れている → 上側を狙う（頭〜胸）
                cy = y + float(globals().get("FACE_FOCUS_PERSON_HEAD_CY_FRAC", 0.18)) * h
            cy = float(max(0.0, min(float(ih), cy)))
            width_frac = float(globals().get("FACE_FOCUS_WIDTH_FRAC", 1.0))
            sc_w = (width_frac * cw) / max(1.0, float(w))
            sc = clamp_zoom(min(base_sc * 1.10, sc_w))
            tw, th = max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))
            im2 = im.resize((tw, th), Image.LANCZOS)
            _FDBG["upper"] += 1
            bias_x = float(globals().get("FACE_FOCUS_BIAS_X", 0.0)) * cw
            bias_y = float(globals().get("FACE_FOCUS_BIAS_Y", -0.10)) * ch
            return crop_center(im2, cx * sc + bias_x, cy * sc + bias_y)
        # サリエンシー候補: 目立つ点を基準に軽くズームしてクロップします
        if sal:
            cx, cy = sal  # type: ignore
            sc = clamp_zoom(base_sc * 1.10)
            tw, th = max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))
            im2 = im.resize((tw, th), Image.LANCZOS)
            _FDBG["saliency"] += 1
            if bool(globals().get("FACE_FOCUS_CENTER_FACE", False)):
                return crop_center(im2, cx * sc, cy * sc)
            else:
                bias_x = float(globals().get("FACE_FOCUS_BIAS_X", 0.0)) * cw
                bias_y = float(globals().get("FACE_FOCUS_BIAS_Y", -0.10)) * ch
                return crop_center(im2, cx * sc + bias_x, cy * sc + bias_y)
        # 候補が無い場合は中央クロップ（必要ならバイアス込み）
        sc = clamp_zoom(base_sc)
        tw, th = max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))
        im2 = im.resize((tw, th), Image.LANCZOS)
        _FDBG["center"] += 1
        if bool(globals().get("FACE_FOCUS_CENTER_FACE", False)):
            return crop_center(im2, tw / 2.0, th / 2.0)
        else:
            bias_x = float(globals().get("FACE_FOCUS_BIAS_X", 0.0)) * cw
            bias_y = float(globals().get("FACE_FOCUS_BIAS_Y", -0.10)) * ch
            return crop_center(im2, tw / 2.0 + bias_x, th / 2.0 + bias_y)
    except Exception:
        # 例外時は、従来に近い「リサイズ→中央クロップ」へフォールバックします
        try:
            sc = max(cw / float(im.size[0]), ch / float(im.size[1]))
            tw, th = max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))
            im2 = im.resize((tw, th), Image.LANCZOS)
            if bool(globals().get("FACE_FOCUS_CENTER_FACE", False)):
                return crop_center(im2, tw / 2.0, th / 2.0)
            else:
                bias_x = float(globals().get("FACE_FOCUS_BIAS_X", 0.0)) * cw
                bias_y = float(globals().get("FACE_FOCUS_BIAS_Y", -0.10)) * ch
                return crop_center(im2, tw / 2.0 + bias_x, th / 2.0 + bias_y)
        except Exception:
            try:
                return im.resize((cw, ch), Image.LANCZOS)
            except Exception:
                return im

# --- Hex 用の S パラメータ解決ロジック ---
def _solve_S_hex(width:int, height:int, margin:int, N:int, max_cols:int, gap:int, eps:int, orient:str):
    best=None
    for cols in range(1, max(1, min(max_cols, N)) + 1):
        rows = (N + cols - 1)//cols
        if orient == "row-shift":
            denom_w = (1.0 + 0.75*(cols-1) + 0.5)
            avail_w = width - 2*margin - max(0, cols-1)*(gap - eps)
            Sw = int(avail_w / max(1e-6, denom_w))
            denom_h = (1.0 + _SQRT3_2*(rows-1))
            avail_h = height - 2*margin - max(0, rows-1)*(gap - eps)
            Sh = int(avail_h / max(1e-6, denom_h))
        else:
            denom_w = (1.0 + 0.75*(cols-1))
            avail_w = width - 2*margin - max(0, cols-1)*(gap - eps)
            Sw = int(avail_w / max(1e-6, denom_w))
            denom_h = (1.0 + _SQRT3_2*(rows-1) + 0.5*_SQRT3_2)
            avail_h = height - 2*margin - max(0, rows-1)*(gap - eps)
            Sh = int(avail_h / max(1e-6, denom_h))
        S = min(Sw, Sh)
        if S < 8: continue
        key=(S, -(rows*cols - N))
        if (best is None) or (key > best[0]): best=(key,S,rows,cols)
    if best is None:
        rows = N; cols = 1
        denom_h = (1.0 + _SQRT3_2*(rows-1))
        if orient != "row-shift": denom_h += 0.5*_SQRT3_2
        avail_h = height - 2*margin - max(0, rows-1)*(gap - eps)
        S = max(8, int(avail_h/max(1e-6, denom_h)))
        return S, rows, cols
    _,S,rows,cols = best
    return S, rows, cols

# --- Hex レンダラ用ラッパー（grid レイアウト呼び出しをフック） ---
def _kana_hex_wrapper(fn):
    """六角形（hex）タイルレイアウトを組み立てるラッパ関数。

    LAYOUT_STYLE や KANA_FORCE_HEX などのグローバル設定を参照し、
    実行時に hex レンダリングを行うかどうかを切り替えるためのデコレータ。
    """
    def _wrapped(images, width, height, margin, gutter, rows, cols, mode, bg_rgb):
        # 実行時の LAYOUT_STYLE を見て hex を使うか判定（KANA_FORCE_HEX で強制も可能）
        ls_now = str(globals().get("LAYOUT_STYLE","")).lower()
        want_hex = (ls_now in _HEX_NAMES)  # gridsafe: ignore alias/force
        # hex が無効、かつスタイルが hex でもない場合は通常の grid レンダラへ。
        # ただし KANA_FORCE_HEX が on のときは、スタイルに関係なく hex を実行します。
        _force_hex = str(globals().get("KANA_FORCE_HEX", "")).lower() in ("on", "true", "1")
        if not HEX_TIGHT_ENABLE or (not want_hex and not _force_hex):
            return fn(images, width, height, margin, gutter, rows, cols, mode, bg_rgb)

        # 完全シャッフルが有効な場合は、処理の最初に一度だけ通知します（重複表示を避けるフラグあり）。
        try:
            if bool(globals().get("ARRANGE_FULL_SHUFFLE", False)) and not globals().get("_FULL_SHUFFLE_NOTE_DONE_HEX", False):
                if str(globals().get("UI_LANG", "")).lower() == 'en':
                    note("Full shuffle enabled")
                else:
                    note("Full shuffle enabled")
                globals()["_FULL_SHUFFLE_NOTE_DONE_HEX"] = True
        except Exception as e:
            _warn_exc_once(e)
            pass
        total = len(images)
        _gap_cfg = globals().get("HEX_TIGHT_GAP", None)
        gap = int(max(0, float(gutter if _gap_cfg is None else _gap_cfg)))
        eps = int(max(0, min(2, float(globals().get("HEX_TIGHT_SEAM_EPS", 0)))))
        orient = str(globals().get("HEX_TIGHT_ORIENT", "col-shift")).lower()
        extend = int(max(0, float(globals().get("HEX_TIGHT_EXTEND", 2))))

        S, r_used, c_used = _solve_S_hex(width, height, margin, total, int(globals().get("HEX_TIGHT_MAX_COLS",128)), gap, eps, orient)

        # 可視タイル数（画面内に配置されるタイル総数）
        try:
            _vis_needed = int(max(1, (r_used + 2*extend) * (c_used + 2*extend)))
        except Exception:
            _vis_needed = 1

        from PIL import Image
        canvas = Image.new("RGB", (width, height), HEX_TIGHT_BG if HEX_TIGHT_BG is not None else bg_rgb)
        mask_canvas = Image.new("L", (width, height), 0)
        try:
            note(
                f"Layout(hex {orient}): grid={r_used}x{c_used}, tile_size={S}, tiles={_vis_needed}, "
                f"gap={gap}, eps={eps}, extend={extend}"
            )
        except Exception as e:
            _kana_silent_exc('core:L18993', e)
            pass
        try:
            banner("Rendering (Hex / Honeycomb)" if str(globals().get("UI_LANG","")).lower()=="en" else "処理中: Hex / Honeycomb")
        except Exception: pass

        hexmask = _hexmask_square(S)
        # --- KANA: draw 高速化（タイル生成のメモリキャッシュ + mask 用白タイル再利用） ---
        from collections import OrderedDict
        try:
            _tile_cache_max = int(globals().get("HEX_TILE_MEMCACHE_MAX", 4096))
        except Exception:
            _tile_cache_max = 4096
        _tile_cache = OrderedDict()  # key(str path) -> PIL.Image (SxS RGB)
        _whiteL = Image.new("L", (S, S), 255)

        def _tile_cache_get(k: str):
            if _tile_cache_max < 0:
                return None
            v = _tile_cache.get(k)
            if v is not None:
                try:
                    _tile_cache.move_to_end(k)
                except Exception as e:
                    _warn_exc_once(e)
                    pass
            return v

        def _tile_cache_put(k: str, v):
            if _tile_cache_max < 0:
                return
            try:
                _tile_cache[k] = v
                _tile_cache.move_to_end(k)
                if _tile_cache_max > 0:
                    while len(_tile_cache) > _tile_cache_max:
                        _tile_cache.popitem(last=False)
            except Exception as e:
                _warn_exc_once(e)
                pass
        step_x = 0.75*S + (gap - eps)
        step_y = _SQRT3_2*S + (gap - eps)

        # --- FIX(hex): 右端/下端が空くことがあるため、必要なら extend を自動で増やして端まで敷き詰める ---
        # 例: step_x=0.75*S なので、cols と extend の組み合わせによっては右端に余白が残ることがあります。
        # ここでは「カバー範囲」を見て、足りない分だけ extend を増やします（安全側）。
        try:
            if bool(globals().get("HEX_TIGHT_AUTO_EXTEND", True)):
                import math
                _orig_extend = int(extend)
                # 横方向の最大シフト（row-shift は奇数行で +S/2）
                _shift_x = (S/2.0) if orient == "row-shift" else 0.0
                # 縦方向の最大シフト（col-shift は奇数列で +step_y/2）
                _shift_y = 0.0 if orient == "row-shift" else (step_y/2.0)

                # 右端カバー（最後の列インデックス = c_used+extend-1）
                _cov_w = margin + int(round(_shift_x + (c_used + extend - 1) * step_x)) + S
                if _cov_w < width:
                    _extra_c = int(math.ceil((width - _cov_w) / max(1e-6, step_x)))
                    extend += max(0, _extra_c)

                # 下端カバー（最後の行インデックス = r_used+extend-1）
                _cov_h = margin + int(round(_shift_y + (r_used + extend - 1) * step_y)) + S
                if _cov_h < height:
                    _extra_r = int(math.ceil((height - _cov_h) / max(1e-6, step_y)))
                    extend += max(0, _extra_r)

                # extend が増えた場合のみ、ログに一言出す（表示が荒れないように）
                if int(extend) != _orig_extend:
                    try:
                        if bool(globals().get("VERBOSE", False)):
                            note(f"Hex auto-extend: { _orig_extend } -> { int(extend) }")
                    except Exception as e:
                        _kana_silent_exc('core:L19065', e)
                        pass
        except Exception as e:
            _warn_exc_once(e)
            pass

        # フェイスフォーカスのデバッグカウンタをリセット
        global _FDBG, _FDBG2
        _FDBG = {"cv2": None, "frontal":0, "profile":0, "anime":0, "ai":0, "upper":0, "person":0, "saliency":0, "center":0,
                 "reject_pos":0, "reject_ratio":0, "errors":0}
        _FDBG2 = {"eyes_ok":0, "eyes_ng":0, "low_reject":0,
                  "anime_face_ok":0, "anime_face_ng":0,
                  "anime_eyes_ok":0, "anime_eyes_ng":0, "ai_face_ok":0, "ai_face_ng":0}

        idx = 0; done = 0
        # -----------------------------------------------------------------------------
        # 内部ヘルパ: 重複除去→不足分補充→（必要なら）シャッフル→tempo 配置
        # row-shift／col-shift どちらでも同じ前処理を行うため、共通化しています。
        def _hex_prepare_images_local(img_list: List[Path], vis_needed: int) -> List[Path]:
            """hex レイアウト用の画像リスト前処理。

            - 同一パスを（順序を保ったまま）重複除去
            - タイル数が足りない場合は、グローバルスキャン結果から未使用画像で補充
            - 設定に応じてシャッフル（完全シャッフル/補充後シャッフル）
            - 可能なら tempo 配置（忙しさ/静けさの並び）を適用

            返り値の順序が、そのままタイル消費順になります。
            """
            images_local = list(img_list)
            _preserve = bool(globals().get("PRESERVE_INPUT_ORDER", False))
            try:
                import os as _kana_os
                def _kana_norm(p):
                    try:
                        return _kana_os.path.normcase(_kana_os.path.normpath(p))
                    except Exception:
                        return p
                # 1) 重複パスを除去（元の順序は維持）
                seen_norm: set = set()
                unique: List[Path] = []
                for _p in images_local:
                    _k = _kana_norm(_p)
                    if _k not in seen_norm:
                        unique.append(_p)
                        seen_norm.add(_k)
                # 2) グローバルスキャン結果から、未使用の画像をプール化
                try:
                    pool: List[Path] = []
                    gscan = (globals().get("KANA_SCAN_ALL", []) or [])
                    gscan = sort_by_select_mode(gscan)
                    # SELECT_MODE=random のときは、プール側もシャッフルして偏りを減らします
                    try:
                        mode_now = str(globals().get("SELECT_MODE", "")).lower()
                    except Exception:
                        mode_now = ""
                    if (not _preserve) and mode_now == "random":
                        try:
                            _seed = globals().get("SHUFFLE_SEED", None)
                            gscan = hash_shuffle(gscan, _seed, salt="hex_topup_pool")
                        except Exception as e:
                            _warn_exc_once(e)
                            pass
                    for _p in gscan:
                        _k = _kana_norm(_p)
                        if _k not in seen_norm:
                            pool.append(_p)
                            seen_norm.add(_k)
                except Exception:
                    pool = []
                # 3) vis_needed に足りない場合は、プールから不足分を補充します
                topup_count = 0
                if len(unique) < vis_needed and pool:
                    need = min(len(pool), vis_needed - len(unique))
                    if (not _preserve) and globals().get("HEX_TOPUP_INTERLEAVE", False):
                        try:
                            base = list(unique)
                            extra = list(pool[:need])
                            merged: List[Path] = []
                            i = j = 0
                            while i < len(base) or j < len(extra):
                                if i < len(base):
                                    merged.append(base[i]); i += 1
                                if j < len(extra):
                                    merged.append(extra[j]); j += 1
                            unique = merged
                        except Exception:
                            unique.extend(pool[:need])
                    else:
                        unique.extend(pool[:need])
                    topup_count = need
                images_prepared = unique
                # 4) 完全シャッフルが有効なら、ここで一度だけシャッフルします（OPT_SEED を尊重）。
                try:
                    if (not _preserve) and bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
                        _seed = globals().get("OPT_SEED", None)
                        hash_shuffle_inplace(images_prepared, _seed, salt="hex_fullshuffle")
                        # 通知は何度も出さない（重複表示を避けるフラグ）
                        try:
                            if not globals().get("_FULL_SHUFFLE_NOTE_DONE_HEX", False):
                                if str(globals().get("UI_LANG", "")).lower() == 'en':
                                    note("Full shuffle enabled")
                                else:
                                    note("Full shuffle enabled")
                                globals()["_FULL_SHUFFLE_NOTE_DONE_HEX"] = True
                        except Exception as e:
                            _warn_exc_once(e)
                            pass
                except Exception as e:
                    _warn_exc_once(e)
                    pass
                # 5) 補充が発生した場合、必要ならもう一度シャッフルして位置の偏りを減らします
                if (not _preserve) and globals().get("HEX_TOPUP_SHUFFLE", True) and topup_count > 0:
                    try:
                        _seed = globals().get("SHUFFLE_SEED", "random")
                        hash_shuffle_inplace(images_prepared, _seed, salt="hex_topup_shuffle")
                    except Exception as e:
                        _warn_exc_once(e)
                        pass
                # 6) 完全シャッフル中でない場合のみ、tempo 配置（忙/静の並び）を適用します（失敗しても無視）。
                try:
                    if images_prepared and (not _preserve) and (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))):
                        images_prepared = _tempo_apply(images_prepared)
                except Exception as e:
                    _warn_exc_once(e)
                    pass
                # 7) 重複除去/補充の結果を、この呼び出しの中で一度だけログ表示します
                try:
                    total_loc = len(images_prepared)
                    wrap_state = "on" if total_loc < vis_needed else "off"
                    note(f"Dedupe(NORM): sources={total_loc} / tiles={vis_needed} | wrap={wrap_state}")
                except Exception as e:
                    _warn_exc_once(e)
                    pass
                return images_prepared
            except Exception:
                # 何か失敗したら、安全側に倒して元のリストをそのまま返します
                return list(img_list)

        if orient == "row-shift":
            half_shift = S/2.0
            min_r = -extend; max_r = r_used + extend
            min_c = -extend; max_c = c_used + extend
            # --- FIX(hex): _vis_needed/_cur を必ず初期化（row/col 両方で同じ） ---
            _cur = 0
            try:
                _vis_needed = 0
                for _r in range(min_r, max_r):
                    _shift = (S/2.0) if (_r % 2 != 0) else 0.0
                    _y = margin + int(round(_r*step_y))
                    for _c in range(min_c, max_c):
                        _x = margin + int(round(_shift + _c*step_x))
                        if _x + S <= 0 or _y + S <= 0 or _x >= width or _y >= height:
                            continue
                        _vis_needed += 1
                _vis_needed = max(1, int(_vis_needed))
            except Exception:
                try:
                    _vis_needed = int(max(1, (r_used + 2*extend) * (c_used + 2*extend)))
                except Exception:
                    _vis_needed = max(1, len(images) if images else 1)
            images = _hex_prepare_images_local(images, _vis_needed)
            total = len(images)
            try:
                images = _kana_hex_apply_global_and_local_opt(images, _vis_needed, orient, S, step_x, step_y, margin, width, height, extend, r_used, c_used)
            except Exception as e:
                _warn_exc_once(e)
                pass
            total = len(images)
            for r in range(min_r, max_r):
                shift = half_shift if (r % 2 != 0) else 0.0
                y = margin + int(round(r*step_y))
                for c in range(min_c, max_c):
                    x = margin + int(round(shift + c*step_x))
                    if total==0: continue
                    if _cur >= total:
                        if total < _vis_needed:
                            _cur = 0
                        else:
                            continue
                        continue
                    _did = False
                    # タイル生成は重いので、同一画像の再利用が起きるケースではメモリキャッシュを使う
                    _k = str(images[_cur])
                    tile = _tile_cache_get(_k)
                    _did = False
                    try:
                        if tile is None:
                            with open_image_safe(images[_cur]) as im:
                                im_rgb = im if im.mode == "RGB" else im.convert("RGB")
                                tile = _cover_square_face_focus(im_rgb, S, images[_cur])
                            _tile_cache_put(_k, tile)
                        _did = _paste_with_clip(canvas, tile, hexmask, x, y)
                        if _did:
                            _paste_with_clip(mask_canvas, _whiteL, hexmask, x, y)
                    except Exception:
                        _FDBG["errors"] += 1
                    if _did:
                        _cur += 1
                    done += 1
                    if bool(globals().get('UI_PROGRESS', True)) and ((done % int(max(1, int(globals().get('UI_PROGRESS_EVERY', 8)))) == 0) or (done >= (max_r-min_r)*(max_c-min_c))):
                        bar(done, (max_r-min_r)*(max_c-min_c), prefix="draw   ", final=False)
        else:
            half_v = step_y/2.0
            min_c = -extend; max_c = c_used + extend
            min_r = -extend; max_r = r_used + extend
            # --- KANA(hex 重複抑止): 可視タイル数を概算し、不足分を補充 ---
            _vis_needed = 0
            if orient == "row-shift":
                for _r in range(min_r, max_r):
                    _shift = (S/2.0) if (_r % 2 != 0) else 0.0
                    _y = margin + int(round(_r*step_y))
                    for _c in range(min_c, max_c):
                        _x = margin + int(round(_shift + _c*step_x))
                        if _x + S <= 0 or _y + S <= 0 or _x >= width or _y >= height: continue
                        _vis_needed += 1
            else:
                for _c in range(min_c, max_c):
                    _shift_y = (step_y/2.0) if (_c % 2 != 0) else 0.0
                    _x = margin + int(round(_c*step_x))
                    for _r in range(min_r, max_r):
                        _y = margin + int(round(_shift_y + _r*step_y))
                        if _x + S <= 0 or _y + S <= 0 or _x >= width or _y >= height: continue
                        _vis_needed += 1
            # --- KANA(hex 重複抑止 v2): パス正規化で重複除去→補充→ログ ---
            import os as _kana_os
            def _kana_norm(p):
                try:
                    return _kana_os.path.normcase(_kana_os.path.normpath(p))
                except Exception:
                    return p
            # 1) images をパス正規化で一意化
            _seen = set()
            _unique = []
            for _p in images:
                _k = _kana_norm(_p)
                if _k not in _seen:
                    _unique.append(_p); _seen.add(_k)
            # 2) 走査済み全プールから「まだ使っていない」候補を作る
            try:
                _pool = []
                # SELECT_MODE に応じて並べ替えされたプールを使用
                _gscan = (globals().get("KANA_SCAN_ALL", []) or [])
                _gscan = sort_by_select_mode(_gscan)
                # ランダムモードではプールをシャッフルしてトップアップ時の選択バイアスを減らす
                try:
                    _mode_now = str(globals().get("SELECT_MODE", "")).lower()
                except Exception:
                    _mode_now = ""
                if (not bool(globals().get("PRESERVE_INPUT_ORDER", False))) and _mode_now == "random":
                    try:
                        _seed = globals().get("SHUFFLE_SEED", None)
                        _gscan = hash_shuffle(_gscan, _seed, salt="hex_topup_pool")
                    except Exception as e:
                        _warn_exc_once(e)
                        pass
                for _p in _gscan:
                    _k = _kana_norm(_p)
                    if _k not in _seen:
                        _pool.append(_p); _seen.add(_k)
            except Exception:
                _pool = []
            # 3) 目安の表示タイル数に足りない分を、未使用プールから補充
            _topup_count = 0  # 補充した枚数
            if len(_unique) < _vis_needed and _pool:
                # 重複除去後のプールは既に sort_by_select_mode() により並べ替え済み
                _need = min(len(_pool), _vis_needed - len(_unique))
                # 補充分の入れ方: interleave（交互に混ぜる）か、末尾に追加
                if (not bool(globals().get("PRESERVE_INPUT_ORDER", False))) and globals().get("HEX_TOPUP_INTERLEAVE", False):
                    try:
                        base = list(_unique)
                        extra = list(_pool[:_need])
                        merged = []
                        i = j = 0
                        while i < len(base) or j < len(extra):
                            if i < len(base):
                                merged.append(base[i]); i += 1
                            if j < len(extra):
                                merged.append(extra[j]); j += 1
                        _unique = merged
                    except Exception:
                        # うまく混ぜられなければ末尾に追加
                        _unique.extend(_pool[:_need])
                else:
                    _unique.extend(_pool[:_need])
                _topup_count = _need
            images = _unique

            # 完全シャッフル: 重複除去＆補充後のリストを一度だけシャッフルします（OPT_SEED を尊重）。
            try:
                if (not bool(globals().get("PRESERVE_INPUT_ORDER", False))) and bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
                    _seed = globals().get("OPT_SEED", None)
                    hash_shuffle_inplace(images, _seed, salt="hex_fullshuffle")
                    # 通知は一度だけ（重複表示を避けるフラグ）
                    try:
                        if not globals().get("_FULL_SHUFFLE_NOTE_DONE_HEX", False):
                            if str(globals().get('UI_LANG','')).lower() == 'en':
                                note("Full shuffle enabled")
                            else:
                                note("Full shuffle enabled")
                            globals()["_FULL_SHUFFLE_NOTE_DONE_HEX"] = True
                    except Exception as e:
                        _warn_exc_once(e)
                        pass
            except Exception as e:
                _warn_exc_once(e)
                pass
            # オプション: 補充後に画像をシャッフルして順序の偏りを減らす
            if (not bool(globals().get("PRESERVE_INPUT_ORDER", False))) and globals().get("HEX_TOPUP_SHUFFLE", True) and _topup_count > 0:
                try:
                    _seed = globals().get("SHUFFLE_SEED", "random")
                    hash_shuffle_inplace(images, _seed, salt="hex_topup_shuffle")
                except Exception as e:
                    _warn_exc_once(e)
                    pass
            # tempo 配置（忙/静の並び）は最終リストに適用します（失敗しても無視）。
            if (not bool(globals().get("PRESERVE_INPUT_ORDER", False))) and images:
                try:
                    images = _tempo_apply(images)
                except Exception as e:
                    _warn_exc_once(e)
                    pass
            try:
                images = _kana_hex_apply_global_and_local_opt(images, _vis_needed, orient, S, step_x, step_y, margin, width, height, extend, r_used, c_used)
            except Exception as e:
                _warn_exc_once(e)
                pass
            total = len(images)
            try:
                wrap_state = "on" if total < _vis_needed else "off"
                note(f"Dedupe(NORM): sources={total} / tiles={_vis_needed} | wrap={wrap_state}")
            except Exception as e:
                _warn_exc_once(e)
                pass
            _cur = 0
            # --- draw prefetch (CPU) : hex tiles ---
            # hex の draw はタイル生成（open/decode + crop/face-focus）が重いので、
            # 先に（可視領域ぶんの）タイルを並列生成してキャッシュへ入れておきます。
            try:
                _pf_ahead = int(max(0, int(globals().get('DRAW_PREFETCH_AHEAD', 16))))
                _pf_ahead = _effective_draw_prefetch_ahead(width, height, _pf_ahead)
                _pf_workers = int(max(1, int(globals().get('DRAW_PREFETCH_WORKERS', 0) or (os.cpu_count() or 4))))
                _pf_on = bool(globals().get('DRAW_PREFETCH_ENABLE', True)) and (_pf_ahead > 0)
                _pf_backend = str(globals().get('DRAW_PREFETCH_BACKEND', ('process' if os.name == 'nt' else 'thread'))).lower()
                _pf_use_mp = _pf_backend in ('process','mp','multiprocess','proc','processpool','process_pool','processpoolexecutor')
            except Exception:
                _pf_ahead, _pf_workers, _pf_on, _pf_use_mp = 0, (os.cpu_count() or 4), False, False

            if _pf_on and total > 0:
                try:
                    _cw, _ch = canvas.size
                    def _hex_would_paste(_x, _y, _s, _w, _h):
                        return not (_x >= _w or _y >= _h or (_x + _s) <= 0 or (_y + _s) <= 0)

                    _need = []
                    _seen = set()
                    _tmp_cur = 0
                    _lim = int(min(total, int(_vis_needed)))

                    for c in range(min_c, max_c):
                        shift_y = half_v if (c % 2 != 0) else 0.0
                        x = margin + int(round(c*step_x))
                        for r in range(min_r, max_r):
                            y = margin + int(round(shift_y + r*step_y))
                            if _tmp_cur >= total:
                                if total < _vis_needed:
                                    _tmp_cur = 0
                                else:
                                    continue
                            if not _hex_would_paste(x, y, S, _cw, _ch):
                                continue
                            _p = images[_tmp_cur]
                            _k = str(_p)
                            if _k not in _seen and _tile_cache_get(_k) is None:
                                _need.append((_p, S))
                                _seen.add(_k)
                                if len(_need) >= _lim:
                                    break
                            _tmp_cur += 1
                        if len(_need) >= _lim:
                            break

                    if _need:
                        _pf_total = len(_need)
                        _pf_done = 0
                        try:
                            _lang = str(globals().get('UI_LANG', '')).lower()
                            banner('Prefetch: Hex tiles' if _lang == 'en' else '処理中: Hex tiles (prefetch)')
                        except Exception as e:
                            _warn_exc_once(e)
                            pass
                        _ui_prog = bool(globals().get('UI_PROGRESS', True))
                        _step = int(max(1, int(globals().get('UI_PROGRESS_EVERY', 8))))
                        if _ui_prog:
                            try:
                                bar(0, _pf_total, prefix='prefetch ', final=False)
                                import sys as _sys
                                _sys.stdout.flush()
                            except Exception as e:
                                _warn_exc_once(e)
                                pass
                        if _pf_use_mp:
                            with ProcessPoolExecutor(max_workers=_pf_workers) as _ex:
                                for _k, _tile in _ex.map(_pf_worker_hex_render, _need, chunksize=1):
                                    if _tile is not None:
                                        _tile_cache_put(_k, _tile)
                                    _pf_done += 1
                                    if _ui_prog and ((_pf_done % _step) == 0 or _pf_done == _pf_total):
                                        bar(_pf_done, _pf_total, prefix='prefetch ', final=(_pf_done == _pf_total))
                                        try:
                                            import sys as _sys
                                            _sys.stdout.flush()
                                        except Exception as e:
                                            _kana_silent_exc('core:L19476', e)
                                            pass
                        else:
                            with ThreadPoolExecutor(max_workers=_pf_workers) as _ex:
                                for _k, _tile in _ex.map(_pf_worker_hex_render, _need):
                                    if _tile is not None:
                                        _tile_cache_put(_k, _tile)
                                    _pf_done += 1
                                    if _ui_prog and ((_pf_done % _step) == 0 or _pf_done == _pf_total):
                                        bar(_pf_done, _pf_total, prefix='prefetch ', final=(_pf_done == _pf_total))
                                        try:
                                            import sys as _sys
                                            _sys.stdout.flush()
                                        except Exception as e:
                                            _kana_silent_exc('core:L19489', e)
                                            pass
                except Exception as e:
                    _warn_exc_once(e)
                    pass
            for c in range(min_c, max_c):
                shift_y = half_v if (c % 2 != 0) else 0.0
                x = margin + int(round(c*step_x))
                for r in range(min_r, max_r):
                    y = margin + int(round(shift_y + r*step_y))
                    if total==0: continue
                    if _cur >= total:
                        if total < _vis_needed:
                            _cur = 0
                        else:
                            continue
                        continue
                    _did = False
                    # タイル生成は重いので、同一画像の再利用が起きるケースではメモリキャッシュを使う
                    _k = str(images[_cur])
                    tile = _tile_cache_get(_k)
                    _did = False
                    try:
                        if tile is None:
                            with open_image_safe(images[_cur]) as im:
                                im_rgb = im if im.mode == "RGB" else im.convert("RGB")
                                tile = _cover_square_face_focus(im_rgb, S, images[_cur])
                            _tile_cache_put(_k, tile)
                        _did = _paste_with_clip(canvas, tile, hexmask, x, y)
                        if _did:
                            _paste_with_clip(mask_canvas, _whiteL, hexmask, x, y)
                    except Exception:
                        _FDBG["errors"] += 1
                    if _did:
                        _cur += 1
                    done += 1
                    if bool(globals().get('UI_PROGRESS', True)) and ((done % int(max(1, int(globals().get('UI_PROGRESS_EVERY', 8)))) == 0) or (done >= (max_r-min_r)*(max_c-min_c))):
                        bar(done, (max_r-min_r)*(max_c-min_c), prefix="draw   ", final=False)

        bar(max(done, (max_r-min_r)*(max_c-min_c)), (max_r-min_r)*(max_c-min_c), prefix="draw   ", final=True)
        if globals().get("FACE_FOCUS_DEBUG", True):
            try:
                cv2_state = _FDBG.get("cv2")
                _note_face_focus_stats(_FDBG, _FDBG2)
            except Exception as e:
                _kana_silent_exc('core:L19533', e)
                pass
        layout_info = {"style":"hex","rows":r_used,"cols":c_used,"S":S,
                       "step_x":step_x,"step_y":step_y,"orient":orient,"extend":extend,"gap":gap}
        return canvas, mask_canvas, layout_info, r_used, c_used
    return _wrapped

# --- 実行時ゲートとラッパーの組み込み ---
try:
    _old = layout_grid
    def _kana_hex_runtime_gate(fn):
        def _wrapped(images, width, height, margin, gutter, rows, cols, mode, bg_rgb):
            ls_now = str(globals().get("LAYOUT_STYLE","")).lower()
            want_hex = (ls_now in _HEX_NAMES)  # gridsafe: ignore alias/force
            globals()["_KANA_HEX_WANT"] = want_hex
            return fn(images, width, height, margin, gutter, rows, cols, mode, bg_rgb)
        return _wrapped
    if not getattr(_old, "_kana_hex_runtime_gate", False):
        layout_grid = _kana_hex_runtime_gate(layout_grid)
        setattr(layout_grid, "_kana_hex_runtime_gate", True)
except Exception as e:
    _warn_exc_once(e)
    pass
if "layout_grid" in globals() and callable(layout_grid):
    layout_grid = _kana_hex_wrapper(layout_grid)
# === KANA: コンソールメッセージ向け i18n ヘルパー ===

# 重複排除用のグローバルプールのプレースホルダ。どこかの処理段階で設定されていればレンダラ側で利用されます。
try:
    KANA_SCAN_ALL
except NameError:
    KANA_SCAN_ALL = None

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # argparse 等が SystemExit を投げることがあるので、そのまま伝播
        raise
    except Exception as e:
        try:
            import os as _os, time as _time, traceback as _traceback
            import tempfile as _tempfile
            _raw = str(globals().get("CRASH_LOG_FILE", "") or "").strip()
            if _raw:
                _log = _raw
            else:
                _base = str(globals().get("LOG_SAVE_DIR", "") or "").strip()
                if _base:
                    _log = _os.path.join(_base, "kana_wallpaper_crash.log")
                else:
                    _log = _os.path.join(_tempfile.gettempdir(), "kana_wallpaper_crash.log")
            with open(_log, "a", encoding="utf-8") as f:
                f.write("=== kana wallpaper crash ===\n")
                f.write("time: " + _time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("error: " + repr(e) + "\n")
                f.write(_traceback.format_exc() + "\n")
            try:
                print(f"[CRASH] 詳細ログ: {_log}")
            except Exception as e:
                _kana_silent_exc('core:L19584', e)
                pass
        except Exception as e:
            _kana_silent_exc('core:L19586', e)
            pass
        raise
