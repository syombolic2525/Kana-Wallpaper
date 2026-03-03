"""
Kana Wallpaper - Unified FINAL（ランチャー / launcher）

本体（kana_wallpaper_unified_final.py）を対話形式で操作するためのランチャーです。

できること
- レイアウト設定（grid/hex/mosaic/random など）を対話で選んで実行
- レイアウトプリセット管理（保存/リネーム/削除/並び替え）
  ※レイアウトプリセットは「並べ方のみ」を保存します（エフェクトは混ぜません）
- エフェクト設定（光/色味/ディテール/仕上げ/明るさ）をカテゴリで管理
- エフェクトプリセット管理（用途別に切り替え）
- LUT運用を快適化:
  - LUTフォルダ登録
  - サブフォルダ込みで .cube を一覧表示して番号で選択
  - 最近使ったLUTから選択

注意（個人情報）
- プリセット/前回設定/キャッシュには、画像やLUTのパスが保存される場合があります。
- これらは既定で「このランチャーと同じ場所/_kana_state」に保存されます。
"""


from __future__ import annotations

import json
import os
import random
import re
import sys
sys.dont_write_bytecode = True  # __pycache__ を作らない
import traceback
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
        _kana_silent_exc('launcher:L65', e)
        pass
    try:
        lim = int(globals().get("SILENT_EXC_SAMPLE_MAX", 5))
        if lim > 0 and len(_KANA_SILENT_EXC_SAMPLES) < lim:
            msg = str(e).replace("\n", " ").replace("\r", " ")
            if len(msg) > 200:
                msg = msg[:200] + "…"
            _KANA_SILENT_EXC_SAMPLES.append((where, e.__class__.__name__, msg))
    except Exception as e:
        _kana_silent_exc('launcher:L74', e)
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
        _kana_silent_exc('launcher:L96', e)
        pass
    try:
        head = f"Silent exceptions: {_KANA_SILENT_EXC_TOTAL}"
        if parts:
            head += " | top: " + " / ".join(parts)
        _note_func(head)
        if bool(globals().get("SILENT_EXC_VERBOSE", False)):
            for w, cls, msg in _KANA_SILENT_EXC_SAMPLES:
                _note_func(f"  - {w}: {cls}: {msg}")
    except Exception as e:
        _kana_silent_exc('launcher:L106', e)
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
        _kana_silent_exc('launcher:L118', e)
        pass
# === /KANA: Silent Exception Summary ===

import importlib.util

# =============================================================================
# よく編集する設定（ここだけ触ればOK）
# =============================================================================
# 表示言語（"ja" または "en"）
DEFAULT_LANG = "ja"

# UI 表示の既定（"ascii" / "unicode"）
# - 文字化けや幅ズレが気になる場合は "ascii" に戻してください。
LAUNCHER_UI_STYLE = "ascii"



# 見出しバナーの最小幅（0 でタイトルに合わせた最小幅 / 28 などで固定幅にできます）
LAUNCHER_BANNER_MIN_WIDTH = 0
# プリセット側に ui_style 等が保存されていても、ランチャーの見た目設定を優先して core に反映するか。
# True（既定）: LAUNCHER_UI_STYLE / LAUNCHER_UNICODE_BLING / LAUNCHER_PROGRESS_BAR_STYLE / LAUNCHER_PROGRESS_WIDTH を常に適用
# False: プリセットに保存された UI 設定を優先
LAUNCHER_FORCE_CORE_UI = True

# Unicode 表示の装飾（True で派手派手）
LAUNCHER_UNICODE_BLING = True

# ANSI エスケープで色付け（対応していない環境では自動的に無効化されます）
LAUNCHER_ANSI_COLOR = True

# 文字色の強さ（見やすさ優先）
#  - 'subtle' : ほどよい濃淡（既定）
#  - 'none'   : 色なし（ログをコピペする用途向け）
LAUNCHER_COLOR_STYLE = 'subtle'


# =============================================================================
# たまに編集する設定（必要なときだけ）
# =============================================================================
# 起動時の機能診断を表示する（環境差の原因切り分けに便利）
LAUNCHER_DIAG_ENABLE = True

# 進捗バー（None の場合は core の設定を上書きしない）
LAUNCHER_PROGRESS_BAR_STYLE = None   # 例: "segment" / "paint"
LAUNCHER_PROGRESS_WIDTH = None       # 例: 40

# 副産物の保存（使用画像リスト/メタ情報など）
# True : kana_wallpaper_used_images.csv / .txt / meta.json を保存します（LOG_SAVE_DIR）
# False: 保存しません
LAUNCHER_SAVE_ARTIFACTS = None
# =============================================================================
# 最適化パラメータ（ランチャー共通の範囲/既定）
# =============================================================================
# steps: 焼きなまし等の繰り返し回数（総回数は steps×reheats になる場合があります）
OPT_STEPS_MIN = 1000
OPT_STEPS_MAX = 200000
OPT_STEPS_DEFAULT = 20000

# reheats: リヒート回数（1〜10 で統一）
OPT_REHEATS_MIN = 1
OPT_REHEATS_MAX = 10
OPT_REHEATS_DEFAULT = 4

# k: 近傍（ばらけ最適化など）
OPT_K_MIN = 3
OPT_K_MAX = 24
OPT_K_DEFAULT = 8

# LUT フォルダの既定候補（未設定時）
try:
    LUT_DEFAULT_DIR_FALLBACK = str(Path.home() / 'Desktop' / 'LUTs')
except Exception:
    LUT_DEFAULT_DIR_FALLBACK = ''


# =============================================================================
# 保存先（プリセット/前回設定/LUT/キャッシュ/ログ等の「副産物」をまとめる）
# =============================================================================
# ✅ 基本：ここだけ触ればOK
#   - 既定は「このランチャーと同じフォルダ/_kana_state」にまとめます。
#   - ランチャーと同じフォルダにまとめたい場合は None にしてください（_kana_state を使用）。
#   - 別の場所にしたい場合は、例）STATE_DIR = r"D:\kana_state"
STATE_DIR: Optional[str] = None
STATE_DIR_BASENAME = "_kana_state"

# =============================================================================
# AIモデル保存ベース（YOLO/YuNet/AnimeFace）
# - モデルの重み（.pt）、YuNet（.onnx）、AnimeFaceカスケード（.xml）を置く場所です。
# ✅ 基本：ここだけ触ればOK
#   - None: STATE_DIR（=このランチャーの隣/_kana_state）の中に "models" を作って参照します（おすすめ）
#   - 例: r"D:\kana_state\models"  または  r".\_kana_state\models"（相対はこのランチャー基準）
# =============================================================================
MODEL_DIR: Optional[str] = None
MODEL_SUBDIR_NAME = "models"



# ✅ ランダムプリセット開始の挙動（GitHub運用向け）
#   - True : ランダムで選ばれたプリセット名を表示（デバッグ用）
#   - False: どれが選ばれたか表示しない（既定）
RANDOM_PRESET_REVEAL_NAME = False


# ✅ 連続出力（ランダムプリセット）中の表示
#   - True : 連続実行の各回で、選ばれたプリセット名を表示（既定）
#   - False: 連続実行中もプリセット名を表示しない
RANDOM_PRESET_REVEAL_NAME_DURING_CONTINUOUS = True

# ランダムプリセットの選択は SystemRandom を使い、seed設定と独立させる
_RANDOM_PRESET_RNG = random.SystemRandom()


# 並べ方プリセットのサニタイズ（ディスクは汚さない）
# - 旧データに混ざっていた UI/Effects キーは「適用時」に除去します。
# - 起動時にJSONを書き換えたり、マーカー用の副産物ファイルは作りません。

# クリーンアップ判定用（旧互換のエフェクト混入 / UI混入）
_LAYOUT_PRESET_UI_KEYS = {
    "ui_style", "unicode_bling",
    "progress_bar_style", "progress_width",
}
_LAYOUT_PRESET_EXTRA_KEYS = {
    "fx", "effect_cfg", "effect", "effects",
    "effect_preset", "effect_preset_name", "effect_preset_id",
}
_LAYOUT_PRESET_EFFECT_SNAKE_KEYS = {
    "effects_enable",
    "halation_enable", "halation_intensity", "halation_radius", "halation_threshold", "halation_knee",
    "grain_enable", "grain_amount",
    "clarity_enable", "clarity_amount", "clarity_radius",
    "unsharp_enable", "unsharp_amount", "unsharp_radius", "unsharp_threshold",
    "denoise_mode", "denoise_strength",
    "dehaze_enable", "dehaze_amount", "dehaze_radius",
    "shadowhighlight_enable", "shadow_amount", "highlight_amount",
    "tonecurve_enable", "tonecurve_mode", "tonecurve_strength",
    "lut_enable", "lut_file", "lut_strength",
    "split_tone_enable", "split_tone_shadow_hue", "split_tone_shadow_strength",
    "split_tone_highlight_hue", "split_tone_highlight_strength", "split_tone_balance",
    "vignette_enable", "vignette_strength", "vignette_round",
    "sepia_enable", "sepia_intensity",
    "vibrance_enable", "vibrance_factor",
    "bw_effect_enable",
    "brightness_mode", "auto_method", "auto_target_mean",
    "manual_gain", "manual_gamma",
}

# ✅ 個別上書き（ふだんは触らなくてOK）
#   どうしても「プリセットだけ別の場所にしたい」等がある人向け。
#   指定できるキー:
#     preset, lastrun, effect_preset, lut_library,
#     dhash_cache, log_file, launcher_error_log, launcher_export_json,
#     video_cache_dir
STATE_PATH_OVERRIDES: Dict[str, str] = {}
# 例）STATE_PATH_OVERRIDES["preset"] = r"D:\kana_state\kana_wallpaper_presets.json"
# 例）STATE_PATH_OVERRIDES["launcher_export_json"] = r"D:\kana_state\exports\"  # フォルダ指定もOK

# ✅ 既定のファイル名（STATE_DIR 配下に置く時の名前）
#   ※基本いじらなくてOK（名前を変えたい人だけ）
STATE_FILE_BASENAMES: Dict[str, str] = {
    "preset": "kana_wallpaper_presets.json",
    "lastrun": "kana_wallpaper_last_run.json",
    "effect_preset": "kana_wallpaper_effect_presets.json",
    "lut_library": "kana_wallpaper_lut_library.json",
    "dhash_cache": "kana_wallpaper.dhash_cache.json",
    "log_file": "kana_wallpaper.log",
    "launcher_error_log": "kana_wallpaper_launcher_error.log",
    "launcher_export_json": "kana_wallpaper_launcher_export.json",
    "video_cache_dir": "kana_wallpaper_video_frames_cache",
}


# =============================================================================
# ローカライズ
# =============================================================================

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "ja": {
        "title": "=== Kana Wallpaper Launcher ===",
        "core": "core: {core}",
        "presets": "presets: {path}",
        "cache": "cache(dHash): {path}",
        "mode": "起動方法",
        "mode_manual": "手動で設定",
        "mode_preset": "プリセットから開始",
        "mode_random_preset": "プリセットをランダムで開始",
        "mode_manage_presets": "プリセット管理",
        "mode_effects": "エフェクト設定",
        "mode_export_core": "コア用外部設定をエクスポート（JSON）",
        "mode_seed": "シード（再現性）",
        "seed_title": "シード（再現性）",
        "seed_random": "毎回ランダム",
        "seed_same": "前回と同じ（{seed}）",
        "seed_same_na": "前回と同じ（未取得）",
        "seed_specify": "数値で指定",
        "seed_value": "seed（整数）",
        "mode_last": "上記の設定で開始",
        "last_action": "前回の設定で",
        "last_summary": "{summary}",
        "effects_summary": "エフェクト: {summary}",
        "last_effects_summary": "前回エフェクト: {summary}",
        "summary_title": "サマリ",
        "layout": "レイアウトを選択",
        "layout_mosaic_h": "mosaic-uniform-height",
        "layout_mosaic_w": "mosaic-uniform-width",
        "layout_grid": "grid",
        "layout_hex": "hex",
        "layout_quilt": "quilt",
        "layout_random": "random",
        "layout_stained_glass": "stained-glass",
        "select_mode": "抽出モード（SELECT_MODE）",
        "shuffle": "完全シャッフル（ARRANGE_FULL_SHUFFLE）",
        "grid_video_timeline": "Grid: 動画タイムスタンプ順（順序）",
        "grid_video_timeline_asc": "asc（時系列）",
        "grid_video_timeline_desc": "desc（逆順）",
        "grid_video_timeline_off": "off（しない）",
        "zip": "アーカイブ内ファイルを読む（ZIP/7z/rar）",
        "zip_yes": "読む",
        "zip_no": "読まない",
        "video": "動画から抽出して使う（VIDEO_SCAN_ENABLE）",
        "video_yes": "する",
        "video_no": "しない",
        "video_mode": "動画抽出モード",
        "video_mode_fixed": "固定（動画1本ごとに最大枚数を指定）",
        "video_mode_auto": "自動（必要枚数と動画本数から抽出枚数を決定）",
        "video_frames_per_video": "動画1本あたり最大抽出枚数（VIDEO_FRAMES_PER_VIDEO）",
        "video_select_mode": "フレーム選別方法（VIDEO_FRAME_SELECT_MODE）",

        "video_sel_random": "random（ランダム）",
        "video_sel_uniform": "uniform（等間隔）",
        "video_sel_scene": "scene（シーン切替優先）",
        "video_sel_scene_best": "scene_best（シーン切替＋ベスト厳選）",
        "video_sel_best_bright": "best_bright（明るめ優先）",
        "video_sel_best_sharp": "best_sharp（シャープ優先）",
        "video_sel_best_combo": "best_combo（明るさ＋シャープ）",
        "on": "on",
        "off": "off",
        "rows": "ROWS（行数）",
        "cols": "COLS（列数）",
        "hex_count": "COUNT（枚数目安）",
        "quilt_count": "タイル枚数（QUILT）",
        "stained_glass_count": "ピース数（stained-glass）",
        "stained_glass_lead_width": "Lead width（境界線太さ）",
        "stained_glass_lead_alpha": "Lead alpha（境界線不透明度）",
        "stained_glass_max_corner_angle": "Max corner angle（度）",
        "stained_glass_effects_apply_mode": "Effects apply mode（stained-glass）",
        "stained_glass_apply_global": "global",
        "stained_glass_apply_mask": "mask",
        "stained_glass_apply_mask_feather": "mask_feather",
        "stained_glass_effects_include_lead": "Effects: include lead（maskのみ）",
        "stained_glass_face_focus_enable": "ステンドグラス：顔フォーカス（ピース選別）",
        "stained_glass_face_priority_enable": "ステンドグラス：顔優先度（配置）",
        "quilt_split_style": "分割スタイル（QUILT_SPLIT_STYLE）",
        "quilt_split_classic": "classic（中央寄り）",
        "quilt_split_mixed": "mixed（中央＋端）",
        "quilt_split_extreme": "extreme（端寄り多め）",
        "quilt_split_uniform": "uniform（一様）",
        "hex_orient": "HEX_TIGHT_ORIENT（行ずらし/列ずらし）",
        "hex_orient_col": "col-shift",
        "hex_orient_row": "row-shift",
        "arr_simple": "配置（簡易）",
        "arr_diag": "gradient（対角）",
        "arr_hilb": "gradient（ヒルベルト）",
        "arr_scatter": "scatter（ばらけ）",
        "arr_as_is": "as-is（順序を保つ）",
        "diag_dir": "対角方向",
        "diag_tlbr": "左上→右下（↘）",
        "diag_brtl": "右下→左上（↖）",
        "diag_trbl": "右上→左下（↙）",
        "diag_bltr": "左下→右上（↗）",
        "opt_extra": "最適化パラメータ（任意）",
        "opt_default": "デフォルト（高速・最適化なし）",
        "opt_tune": "調整する（最適化を実行：steps / 再加熱 / k）",
        "steps": f"最適化 steps（{OPT_STEPS_MIN}〜{OPT_STEPS_MAX}）",
        "reheats": f"再加熱回数（{OPT_REHEATS_MIN}〜{OPT_REHEATS_MAX}）",
        "k": f"近傍 k（{OPT_K_MIN}〜{OPT_K_MAX}）",
        "mosaic_est": "※推定: {other}={v} / ユニーク≈{cnt}",
        "preset_title": "プリセット",
        "preset_none": "プリセットがありません。手動設定に切り替えます。",
        "preset_pick": "プリセットを選択",
        "preset_random_pick": "ランダムプリセットを選択しました。",
        "preset_random_pick_named": "ランダムで選ばれました：{name}",
        "preset_action": "このプリセットで",
        "preset_action_run": "このまま実行",
        "preset_action_edit": "編集してから実行",
        "preset_action_load_to_top": "編集（プリセット読込→トップへ）",
        "preset_action_back": "戻る",
        "preset_apply_mode": "プリセット適用",
        "preset_apply_all": "全部適用（レイアウト＋エフェクト）",
        "preset_apply_layout_only": "レイアウトのみ（エフェクトは現在のまま）",
        "preset_action_effects": "エフェクトを編集して実行",
        "last_action_run_no_wallpaper": "壁紙に設定しないで実行",
        "last_action_repeat": "連続出力（連番保存）",
        "preset_action_run_no_wallpaper": "壁紙に設定しないで実行",
        "preset_action_repeat": "連続出力（連番保存）",
        "repeat_count_prompt": "連続出力回数（0=無限）",
        "apply_wallpaper_prompt": "壁紙に設定する？",
        "apply_wallpaper_yes": "設定する",
        "apply_wallpaper_no": "設定しない",
        "repeat_apply_wallpaper_prompt": "連続出力中も壁紙に設定する？",
        "repeat_wallpaper_on": "設定する",
        "repeat_wallpaper_off": "設定しない",
        "repeat_started": "連続出力を開始します",
        "repeat_stopped": "連続出力を停止しました",
        "preset_save": "プリセットに保存（任意）",
        "preset_save_yes": "保存する",
        "preset_save_no": "保存しない",
        "preset_name": "保存名（必須）",
        "preset_name_empty": "保存名が空のため、保存しません。",
        "preset_overwrite": "同名プリセットがあります。上書きしますか？",
        "preset_overwrite_yes": "上書きする",
        "preset_overwrite_no": "上書きしない",
        "run": "--- 実行 ---",
        "err_log": "ログ: {path}",
        "input_paths": "ドラッグ＆ドロップ入力: {n} 件",

        "back": "戻る",

        # 汎用メッセージ
        "msg_choose": "候補番号（または候補文字列）で選択してください。",
        "msg_enter_number": "数値を入力してください。",
        "msg_range": "{min_v}〜{max_v} の範囲で入力してください。",
        "err_no_core": "本体ファイルが見つかりません。kana_wallpaper_unified_final*.py を同じフォルダに置いてください（_v番号が大きいものを優先します）。",
        "err_core_load": "本体のロードに失敗: {path}",
        "err_no_entry": "本体に main() / run() が見つかりません。",
        "bling_mode": "派手派手Unicodeモード（UI_STYLE + UNICODE_BLING）",
        "bling_on": "on（派手）",
        "bling_off": "off（通常）",
        "bling_hint": "※ Unicodeが崩れる場合はoffに戻すか、FORCE_UTF8_CPを有効にしてください",
        "face_ai_enable": "顔フォーカス：検出モデルを使う（高精度）",
        "face_ai_backend": "検出方式（モデル）",
        "face_ai_backend_yolo": "YOLO（アニメ顔）",
        "face_ai_backend_yunet": "YuNet（実写向け）",
        "face_ai_backend_animeface": "AnimeFace（CPU）",
        "face_ai_model": "モデルのパス（空ならデフォルト）",
        "face_ai_model_cascade": "カスケードXML（lbpcascade_animeface.xml）",
        "face_ai_sense": "検出感度（しきい値）",
        "face_ai_sense_sens": "敏感（拾いやすい）",
        "face_ai_sense_std": "標準",
        "face_ai_sense_strict": "厳しめ（誤検出少）",
        "face_ai_device": "実行デバイス（YOLO）",
        "face_ai_device_auto": "自動",
        "face_ai_device_gpu0": "GPU0（0）",
        "face_ai_device_cpu": "CPU",
    },
    "en": {
        "grid_video_timeline": "Video timestamp order (Grid)",
        "grid_video_timeline_asc": "asc (timeline)",
        "grid_video_timeline_desc": "desc (reverse)",
        "grid_video_timeline_off": "off (no)",
        "title": "=== Kana Wallpaper Launcher ===",
        "core": "core: {core}",
        "presets": "presets: {path}",
        "cache": "cache(dHash): {path}",
        "mode": "Start mode",
        "mode_manual": "Manual setup",
        "mode_preset": "Start from preset",
        "mode_random_preset": "Start with random preset",
        "mode_manage_presets": "Manage presets",
        "mode_effects": "Effects settings",
        "mode_last": "Start with these settings",
        "last_action": "With last settings",
        "last_summary": "{summary}",
        "effects_summary": "Effects: {summary}",
        "last_effects_summary": "Last effects: {summary}",
        "summary_title": "Summary",
        "layout": "Select layout",
        "layout_mosaic_h": "mosaic-uniform-height",
        "layout_mosaic_w": "mosaic-uniform-width",
        "layout_grid": "grid",
        "layout_hex": "hex",
        "layout_quilt": "quilt",
        "layout_random": "random",
        "layout_stained_glass": "stained-glass",
        "select_mode": "Selection mode (SELECT_MODE)",
        "shuffle": "Full shuffle (ARRANGE_FULL_SHUFFLE)",
        "zip": "Read files inside ZIP (ZIP_SCAN_ENABLE)",
        "zip_yes": "read",
        "zip_no": "skip",
        "video": "Extract video frames (VIDEO_SCAN_ENABLE)",
        "video_yes": "extract",
        "video_no": "skip",
        "video_mode": "Video frame extraction mode",
        "video_mode_fixed": "Fixed (max frames per video)",
        "video_mode_auto": "Auto (decide per-video frames from target and number of videos)",
        "video_frames_per_video": "Max frames per video (VIDEO_FRAMES_PER_VIDEO)",
        "video_select_mode": "Frame selection (VIDEO_FRAME_SELECT_MODE)",

        "video_sel_random": "random",
        "video_sel_uniform": "uniform",
        "video_sel_scene": "scene",
        "video_sel_scene_best": "scene_best",
        "video_sel_best_bright": "best_bright",
        "video_sel_best_sharp": "best_sharp",
        "video_sel_best_combo": "best_combo",
        "on": "on",
        "off": "off",
        "rows": "ROWS",
        "cols": "COLS",
        "hex_count": "COUNT (approx.)",
        "quilt_count": "Tiles (QUILT)",
        "stained_glass_count": "COUNT (stained-glass pieces)",
        "stained_glass_lead_width": "Lead width",
        "stained_glass_lead_alpha": "Lead alpha",
        "stained_glass_max_corner_angle": "Max corner angle (deg)",
        "stained_glass_effects_apply_mode": "Effects apply mode (stained-glass)",
        "stained_glass_apply_global": "global",
        "stained_glass_apply_mask": "mask",
        "stained_glass_apply_mask_feather": "mask_feather",
        "stained_glass_effects_include_lead": "Effects: include lead (mask only)",
        "stained_glass_face_focus_enable": "Stained-glass face-focus (piece pick)",
        "stained_glass_face_priority_enable": "Stained-glass face priority (placement)",
        "quilt_split_style": "Split style (QUILT_SPLIT_STYLE)",
        "quilt_split_classic": "classic (center-biased)",
        "quilt_split_mixed": "mixed (center + edges)",
        "quilt_split_extreme": "extreme (edge-biased)",
        "quilt_split_uniform": "uniform (uniform)",
        "hex_orient": "HEX_TIGHT_ORIENT",
        "hex_orient_col": "col-shift",
        "hex_orient_row": "row-shift",
        "arr_simple": "Arrangement (simple)",
        "arr_diag": "Gradient (diagonal)",
        "arr_hilb": "Gradient (Hilbert)",
        "arr_scatter": "Scatter",
        "arr_as_is": "As-is (keep order)",
        "diag_dir": "Diagonal direction",
        "diag_tlbr": "↘ Top-left → Bottom-right",
        "diag_brtl": "↖ Bottom-right → Top-left",
        "diag_trbl": "↙ Top-right → Bottom-left",
        "diag_bltr": "↗ Bottom-left → Top-right",
        "opt_extra": "Optimisation parameters (optional)",
        "opt_default": "Use defaults (fast / no optimisation)",
        "opt_tune": "Tune (run optimisation: steps / reheats / k)",
        "steps": f"Steps ({OPT_STEPS_MIN}–{OPT_STEPS_MAX})",
        "reheats": f"Reheats ({OPT_REHEATS_MIN}–{OPT_REHEATS_MAX})",
        "k": f"k neighbours ({OPT_K_MIN}–{OPT_K_MAX})",
        "mosaic_est": "Est.: {other}={v} / unique~{cnt}",
        "preset_title": "Presets",
        "preset_none": "No presets found. Switching to manual setup.",
        "preset_pick": "Select a preset",
        "preset_random_pick": "Random preset selected.",
        "preset_random_pick_named": "Randomly chosen: {name}",
        "preset_action": "With this preset",
        "preset_action_run": "Run as-is",
        "preset_action_edit": "Edit then run",
        "preset_action_load_to_top": "Load preset and return to main menu",
        "preset_action_back": "Back",
        "preset_apply_mode": "Apply preset",
        "preset_apply_all": "Apply layout + effects",
        "preset_apply_layout_only": "Apply layout only (keep effects)",
        "preset_action_effects": "Edit effects then run",
        "last_action_run_no_wallpaper": "Run without setting wallpaper",
        "last_action_repeat": "Continuous output (numbered)",
        "preset_action_run_no_wallpaper": "Run without setting wallpaper",
        "preset_action_repeat": "Continuous output (numbered)",
        "repeat_count_prompt": "How many outputs (0=infinite)",
        "apply_wallpaper_prompt": "Set this as your wallpaper?",
        "apply_wallpaper_yes": "Yes",
        "apply_wallpaper_no": "No",
        "repeat_apply_wallpaper_prompt": "Apply wallpaper during continuous output?",
        "repeat_wallpaper_on": "Yes",
        "repeat_wallpaper_off": "No",
        "repeat_started": "Starting continuous output",
        "repeat_stopped": "Stopped continuous output",
        "preset_save": "Save as preset (optional)",
        "preset_save_yes": "Save",
        "preset_save_no": "Do not save",
        "preset_name": "Preset name (required)",
        "preset_name_empty": "Empty name; not saving.",
        "preset_overwrite": "A preset with the same name exists. Overwrite?",
        "preset_overwrite_yes": "Overwrite",
        "preset_overwrite_no": "Do not overwrite",
        "run": "--- Run ---",
        "err_log": "Log: {path}",
        "input_paths": "Drag & drop inputs: {n}",

        "back": "Back",

        # 汎用メッセージ
        "msg_choose": "Please select by number (or enter the exact option string).",
        "msg_enter_number": "Please enter a number.",
        "msg_range": "Please enter a value between {min_v} and {max_v}.",
        "err_no_core": "Core file not found. Put kana_wallpaper_unified_final*.py in the same folder (highest _v number wins).",
        "err_core_load": "Failed to load core: {path}",
        "err_no_entry": "Could not find main() / run() entry point in core module.",
        "face_ai_enable": "Use AI for face-focus (fast/accurate)",
        "face_ai_backend": "AI backend/model",
        "face_ai_backend_yolo": "YOLO (anime face)",
        "face_ai_backend_yunet": "YuNet (photo)",
        "face_ai_backend_animeface": "AnimeFace (CPU)",
        "face_ai_model": "Model path (blank = default)",
        "face_ai_model_cascade": "Cascade XML (lbpcascade_animeface.xml)",
        "face_ai_sense": "Detection sensitivity (threshold)",
        "face_ai_sense_sens": "Sensitive (detect more)",
        "face_ai_sense_std": "Standard",
        "face_ai_sense_strict": "Strict (fewer false positives)",
        "face_ai_device": "Device (YOLO)",
        "face_ai_device_auto": "Auto",
        "face_ai_device_gpu0": "GPU0 (0)",
        "face_ai_device_cpu": "CPU",
    },
}


def tr(key: str) -> str:
    lang = DEFAULT_LANG if DEFAULT_LANG in TRANSLATIONS else "en"
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))

def uistr(en: str, ja: str) -> str:
    """UI表示用の簡易ローカライズ。

    - 英語UI: business English
    - 日本語UI: 既存の表示
    """
    try:
        lang = str(DEFAULT_LANG).lower()
    except Exception:
        lang = "en"
    if lang not in TRANSLATIONS:
        lang = "en"
    return en if lang == "en" else ja



# =============================================================================
# Capability diagnostic (optional dependencies / external tools)
# =============================================================================


def _diag_find_spec(mod_name: str) -> bool:
    """Return True if importlib can find the module."""
    try:
        return importlib.util.find_spec(mod_name) is not None
    except Exception:
        return None


def _diag_version(mod_name: str) -> str:
    """Best-effort module version string (no import crash)."""
    try:
        mod = __import__(mod_name)
        return str(getattr(mod, '__version__', '')) or '(unknown)'
    except Exception:
        return ''


def _diag_which(*names: str) -> str:
    """Return first executable found in PATH (or empty)."""
    try:
        import shutil
        for n in names:
            p = shutil.which(n)
            if p:
                return p
    except Exception as e:
        _kana_silent_exc('launcher:L195', e)
        pass
    return ''


def print_capability_diagnostic(core_mod: Any) -> None:
    """起動時に「使える機能」を診断して表示します。

    numpy/cv2/ffprobe/アーカイブ対応などの“あれば強化”系は、環境差で挙動が変わりやすいです。
    ここで可否を一度だけ出しておくと「なぜ効かない？」の迷子が減ります。
    """
    try:
        if not bool(globals().get('LAUNCHER_DIAG_ENABLE', True)):
            return
    except Exception:
        return

    # Optional Python deps
    has_numpy = _diag_find_spec('numpy')
    has_cv2 = _diag_find_spec('cv2')
    has_py7zr = _diag_find_spec('py7zr')
    has_rarfile = _diag_find_spec('rarfile')
    has_pil = _diag_find_spec('PIL') or _diag_find_spec('PIL.Image')

    # External tools
    ffprobe_path = _diag_which(getattr(core_mod, 'VIDEO_FFPROBE_PATH', None) or 'ffprobe')
    ffmpeg_path = _diag_which('ffmpeg')
    # rarfile は外部ツールが必要な場合があるため、一応チェック（7z/unrar/bsdtar など）
    rar_tool = _diag_which('unrar') or _diag_which('7z') or _diag_which('7za') or _diag_which('bsdtar')

    def _st(label: str, status: str) -> str:
        """ステータス文字列（OK/MISSING/LIMITED）を色付けして返す。"""
        s = str(status).upper()
        if s == 'OK':
            return _lc('92', s)
        if s == 'MISSING':
            return _lc('91', s)
        if s == 'LIMITED':
            return _lc('93', s)
        return _lc('90', s)

    def _fmt_bool(ok: bool) -> str:
        return _st('', 'OK' if ok else 'MISSING')

    # 見出し（派手派手ONなら core の虹色バナーになる）
    cap_title = uistr("Capability check", "機能診断")
    try:
        _launcher_banner(cap_title)
    except Exception:
        # 最低限のフォールバック
        print(f"=== {cap_title} ===")

    # 0) all OK なら 1 行に圧縮
    ffprobe_ok = bool(ffprobe_path)
    ffmpeg_ok = bool(ffmpeg_path)
    arch_7z_ok = bool(has_py7zr)
    arch_rar_ok = bool(has_rarfile)
    rar_tool_ok = (bool(rar_tool) if arch_rar_ok else True)

    all_ok = bool(has_pil) and bool(has_numpy) and bool(has_cv2) and ffprobe_ok and ffmpeg_ok and arch_7z_ok and arch_rar_ok and rar_tool_ok
    if all_ok:
        # 二重表示（OK: OK）にならないように、文字列は 1 つにまとめる
        _launcher_note(_lc('92', 'all OK') + " (Pillow/numpy/cv2/ffprobe/ffmpeg/zip/7z/rar)")
        return

    # 1) Python deps
    _launcher_note(f"Pillow (required): {_fmt_bool(has_pil)}")
    _launcher_note(f"numpy (spectral/hilbert): {_fmt_bool(has_numpy)}")
    _launcher_note(f"cv2 (face-focus/video): {_fmt_bool(has_cv2)}")

    # 2) ffprobe/ffmpeg
    if ffprobe_path:
        _launcher_note(f"ffprobe: {_st('', 'OK')}")
    else:
        _launcher_note(f"ffprobe: {_st('', 'MISSING')} (SAR/DAR correction skipped)")
    if ffmpeg_path:
        _launcher_note(f"ffmpeg: {_st('', 'OK')}")

    # 3) archives
    arch_zip = _st('', 'OK')  # zip は標準対応
    arch_7z = _st('', 'OK') if has_py7zr else _st('', 'MISSING')
    arch_rar = _st('', 'OK') if has_rarfile else _st('', 'MISSING')
    _launcher_note(f"archives: zip={arch_zip}, 7z={arch_7z}, rar={arch_rar}")

    # rar tool detail
    if has_rarfile:
        _launcher_note(f"rar tool: {_st('', 'OK') if rar_tool else _st('', 'MISSING')}")
    # 4) hint
    if (not has_cv2) or (not has_numpy) or (has_rarfile and not rar_tool) or (has_py7zr is False):
        _launcher_note("tip: install optional deps (requirements-optional.txt) for full power")


_LAUNCHER_CORE_UI = None  # core モジュール参照（ランチャーの派手派手表示で利用）

def _set_launcher_core_ui(mod: object) -> None:
    """ランチャー表示用に core モジュール参照を保持します。"""
    global _LAUNCHER_CORE_UI
    _LAUNCHER_CORE_UI = mod

def _launcher_supports_ansi() -> bool:
    """ANSI カラーを使えるか（ざっくり判定・WindowsはVTを有効化してみる）。"""
    try:
        import os
        import sys

        # パイプ/ファイルへリダイレクトされている場合は色なし（見た目崩れ防止）
        try:
            if not bool(getattr(sys.stdout, "isatty", lambda: False)()):
                return False
        except Exception:
            return False

        if os.name != "nt":
            return str(out_path)

        # Windows: Virtual Terminal を有効化（失敗しても環境変数で判定）
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            STD_OUTPUT_HANDLE = -11
            h = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
            if h in (0, -1):
                raise RuntimeError("GetStdHandle failed")

            mode = ctypes.c_uint()
            if kernel32.GetConsoleMode(h, ctypes.byref(mode)) == 0:
                raise RuntimeError("GetConsoleMode failed")

            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            if (mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0:
                # 失敗しても致命ではない（戻り値は見ない）
                kernel32.SetConsoleMode(h, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)

            return True
        except Exception:
            env = os.environ
            if env.get("WT_SESSION"):
                return True
            if env.get("TERM_PROGRAM"):
                return True
            if env.get("ANSICON"):
                return True
            if env.get("ConEmuANSI") == "ON":
                return True
            return False
    except Exception:
        return False

def _lc(code: str, s: str) -> str:
    """ランチャー用：色付け（派手派手 ON のときのみ）。"""
    try:
        if not LAUNCHER_ANSI_COLOR:
            return s
        if not bool(LAUNCHER_UNICODE_BLING):
            return s
        if str(LAUNCHER_COLOR_STYLE).strip().lower() == 'none':
            return s
        if not _launcher_supports_ansi():
            return s
        return f"\x1b[{code}m{s}\x1b[0m"
    except Exception:
        return s



# 互換エイリアス（過去の表示コード互換）
C = _lc

# ---------------------------------------------------------------
# 表示幅（全角/半角）を考慮した整形ヘルパー（ランチャー表示用）
# - 日本語を含む文字列は見た目の幅が len() と一致しないため、枠のズレを防ぎます。
# ---------------------------------------------------------------
def _launcher_char_width(ch: str) -> int:
    """表示幅（ざっくり）。日本語UIでは曖昧幅(A)も2幅扱い。"""
    try:
        import unicodedata
        eaw = unicodedata.east_asian_width(ch)
        if eaw in ('F', 'W'):
            return 2
        if eaw == 'A':
            # 日本語UIのときは Ambiguous を2幅扱いにすると枠がズレにくい
            try:
                return 2 if str(DEFAULT_LANG).lower() == 'ja' else 1
            except Exception:
                return 1
        return 1
    except Exception:
        return 1

def _launcher_text_width(s: str) -> int:
    try:
        return sum(_launcher_char_width(ch) for ch in str(s))
    except Exception:
        return len(str(s))

def _launcher_pad_to_width(s: str, width: int) -> str:
    """表示幅 width に収まるよう右側を空白で埋める（切り詰めはしない）。"""
    try:
        cur = _launcher_text_width(s)
        if cur >= width:
            return str(s)
        return str(s) + (' ' * (width - cur))
    except Exception:
        return str(s).ljust(max(0, int(width)))

def _launcher_banner(title: str) -> None:
    """ランチャー用の見出し表示（core の派手派手バナーを優先）。"""
    t = str(title).strip()
    try:
        # 既存の "=== xxx ===" は中身だけ使う
        if t.startswith("===") and t.endswith("==="):
            t = t.strip("=").strip()
    except Exception as e:
        _kana_silent_exc('launcher:L403', e)
        pass
    # core 側の派手派手（虹色）バナーをそのまま使う（可能な場合）
    try:
        if str(LAUNCHER_UI_STYLE).lower() == "unicode" and bool(LAUNCHER_UNICODE_BLING):
            mod = _LAUNCHER_CORE_UI
            if mod is not None and hasattr(mod, "banner") and callable(getattr(mod, "banner")):
                # core 側の表示設定を一時的に合わせる
                old_verbose = bool(getattr(mod, "VERBOSE", True))
                old_style = str(getattr(mod, "UI_STYLE", "ascii"))
                old_bling = bool(getattr(mod, "UNICODE_BLING", False))
                try:
                    setattr(mod, "VERBOSE", True)
                    setattr(mod, "UI_STYLE", "unicode")
                    setattr(mod, "UNICODE_BLING", True)
                    # ANSI を有効化（可能なら）
                    try:
                        if hasattr(mod, "_enable_ansi") and callable(getattr(mod, "_enable_ansi")):
                            getattr(mod, "_enable_ansi")()
                        if hasattr(mod, "UI") and isinstance(getattr(mod, "UI"), dict):
                            mod.UI["style"] = "unicode"
                            mod.UI["ansi"] = bool(getattr(mod, "ANSI_OK", True))
                    except Exception as e:
                        _kana_silent_exc('launcher:L426', e)
                        pass
                    getattr(mod, "banner")(t)
                    return
                finally:
                    try:
                        setattr(mod, "VERBOSE", old_verbose)
                        setattr(mod, "UI_STYLE", old_style)
                        setattr(mod, "UNICODE_BLING", old_bling)
                    except Exception as e:
                        _kana_silent_exc('launcher:L435', e)
                        pass
    except Exception as e:
        _kana_silent_exc('launcher:L437', e)
        pass
    # フォールバック（ランチャー単体）
    try:
        min_w = int(globals().get("LAUNCHER_BANNER_MIN_WIDTH", 0))
    except Exception:
        min_w = 0
    if str(LAUNCHER_UI_STYLE).lower() == "unicode":
        w = max(min_w, _launcher_text_width(t) + 4)
        top = "┌" + "─" * (w - 2) + "┐"
        mid = "│ " + _launcher_pad_to_width(t, w - 4) + " │"
        bot = "└" + "─" * (w - 2) + "┘"
        print(_lc("96", top))
        print(_lc("96", mid))
        print(_lc("96", bot))
    else:
        w = max(min_w, _launcher_text_width(t) + 4)
        top = "+" + "-" * (w - 2) + "+"
        mid = "| " + _launcher_pad_to_width(t, w - 4) + " |"
        bot = "+" + "-" * (w - 2) + "+"
        print(top)
        print(mid)
        print(bot)

def _launcher_note(line: str) -> None:
    """ランチャー用：箇条書き表示（派手派手時は少し色付け）。

    NOTE:
      色付きの断片（例: OK/MISSING など）を行内に埋め込むと、
      途中の "\x1b[0m"（リセット）で外側の色まで解除され、
      "7z=" や "rar=" など一部だけ色が変わって見えることがあります。
      そこで、この行の基本色（灰）を「リセットのたびに再適用」して統一します。
    """
    # _launcher_note() は先頭に '•' を付与する。呼び出し側が '•' を含めても二重にならないよう吸収する。
    try:
        _ln = str(line).lstrip()
        if _ln.startswith("• "):
            line = _ln[2:].lstrip()
        elif _ln.startswith("•"):
            line = _ln[1:].lstrip()

        # 表記ゆれ吸収（UI統一）
        try:
            _t = str(line)
            _t = re.sub(r'\bquilt\b\s*[\(\（]\s*bsp\s*[\)\）]', 'quilt', _t, flags=re.I)
            _t = re.sub(r'\bQuilt\b', 'quilt', _t)
            line = _t
        except Exception as e:
            _kana_silent_exc('launcher:note_norm', e)
            pass
    except Exception as e:
        _kana_silent_exc('launcher:L474', e)
        pass
    s = f"  • {line}"

    try:
        # ANSI + 派手派手Unicode のときだけ、リセット後に灰色を再適用して色ズレを防ぐ
        if (bool(LAUNCHER_UNICODE_BLING)
                and bool(LAUNCHER_ANSI_COLOR)
                and (str(LAUNCHER_COLOR_STYLE).strip().lower() != 'none')
                and _launcher_supports_ansi()):
            grey = "\x1b[90m"
            reset = "\x1b[0m"
            # 内部の reset の後に grey を挿入して、行全体の基調色を維持する
            out = grey + s.replace(reset, reset + grey) + reset
            print(out)
            return
    except Exception as e:
        _kana_silent_exc('launcher:L490', e)
        pass
    # フォールバック（従来どおり）
    if str(LAUNCHER_COLOR_STYLE).strip().lower() == 'none':
        print(s)
    else:
        print(_lc("90", s) if bool(LAUNCHER_UNICODE_BLING) else s)

# =============================================================================
# 入力ヘルパ
# =============================================================================


BACK_TOKEN = "__BACK__"


def _is_back_raw(raw: str) -> bool:
    r = raw.strip().lower()
    return r in ("b", "back", "戻る", "もどる", "0")


def _is_back_raw_no_zero(raw: str) -> bool:
    """戻る入力（数値入力で 0 を値として扱いたい場合用）。

    - 0 は値として入力できるように、戻る判定から除外します。
    - 全角 'ｂ' も戻るとして扱います（日本語IME対策）。
    """
    r = raw.strip().lower()
    return r in ("b", "back", "戻る", "もどる", "ｂ", "ｂａｃｋ")


def _normalize_numeric_text(s: str) -> str:
    """日本語IMEの全角入力などを数値として解釈できる形に寄せます。

    例:
      - '０．５' -> '0.5'
      - '－1'   -> '-1'
      - '1,234' -> '1234'
      - '0,5'   -> '0.5'（小数点が '.' で無い場合）
    """
    if s is None:
        return ""
    t = str(s).strip()
    if not t:
        return t

    # 全角→半角（数字/符号/小数点）
    trans = str.maketrans({
        "０":"0","１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9",
        "．":".","。":".","，":",","、":",",
        "－":"-","−":"-","―":"-","ー":"-",
        "＋":"+",
        "　":" ",
    })
    t = t.translate(trans)

    # 小数点が '.' で無い場合（例: '0,5'）は ',' を '.' として扱う
    if "," in t and "." not in t:
        if t.count(",") == 1:
            t = t.replace(",", ".")
        else:
            t = t.replace(",", "")
    else:
        # それ以外の ',' は桁区切りとして除去
        t = t.replace(",", "")

    return t


def _print_choices(choices: List[Union[str, Tuple[str, str]]]) -> None:
    """選択肢を表示する（表示ラベルがある場合はそれを使う）。"""
    for i, ch in enumerate(choices, 1):
        if isinstance(ch, tuple) and len(ch) == 2:
            _val, disp = ch
        else:
            disp = str(ch)
        print(f"  [{i}] {disp}")


def ask_choice(label: str, choices: List[Union[str, Tuple[str, str]]], default_index: int = 1, allow_back: bool = True) -> str:
    """番号または文字列で選択。

    - choices は、従来どおりの List[str] に加えて、(value, display_label) のタプルも受け付ける。
      英語UIのとき、内部キー（日本語）を維持したまま表示だけ英語にしたい場面で使う。

    - 通常の選択肢は [1..N]
    - 戻るは常に最後に表示し [0]
    - allow_back=False のときは戻るを表示/受付しない
    """
    if not choices:
        raise ValueError("choices が空です")

    # 正規化: [(value, display), ...]
    norm: List[Tuple[str, str]] = []
    for ch in choices:
        if isinstance(ch, tuple) and len(ch) == 2:
            val = str(ch[0])
            disp = str(ch[1])
        else:
            val = str(ch)
            disp = str(ch)
        norm.append((val, disp))

    back_label = tr("back")

    # 既定値は通常選択肢の範囲に丸める
    # allow_back=True かつ default_index=0 のときは「戻る」を既定にできます。
    if allow_back and default_index == 0:
        pass
    else:
        if default_index < 1 or default_index > len(norm):
            default_index = 1

    _print_choices([(v, d) for (v, d) in norm])
    if allow_back:
        print(f"  [0] {back_label}")
    prompt = f"{label} [{default_index}]: "

    sys.stdout.flush()
    while True:
        sys.stdout.flush()
        raw = input(prompt).strip()
        if not raw:
            if allow_back and default_index == 0:
                return BACK_TOKEN
            return norm[default_index - 1][0]

        if allow_back and _is_back_raw(raw):
            return BACK_TOKEN

        # 数字入力
        try:
            n = int(raw)
            if allow_back and n == 0:
                return BACK_TOKEN
            if 1 <= n <= len(norm):
                return norm[n - 1][0]
        except Exception as e:
            _kana_silent_exc('launcher:L908', e)
            pass

        # 文字列入力: value / display のどちらでも一致させる
        raw_norm = raw.strip()
        raw_norm_lower = raw_norm.lower()
        for val, disp in norm:
            if raw_norm == val or raw_norm == disp:
                return val
            # 英語UIでは大文字小文字を無視して一致させる
            if str(DEFAULT_LANG).lower() == 'en':
                if raw_norm_lower == str(val).lower() or raw_norm_lower == str(disp).lower():
                    return val

        print(tr("invalid_choice"))
def ask_int(label: str, default: int, min_v: int, max_v: int, allow_back: bool = True):
    """整数入力。

    NOTE:
      - これまでは「0」を常に “戻る” として扱っていましたが、
        min_v..max_v に 0 が含まれる入力では 0 を値として入れたいことがあります（例: lead width）。
      - その場合は “戻る” を b/back/戻る で受け付け、0 は値として扱います。
    """
    prompt = f"{label} [{default}]: "

    zero_is_value = (int(min_v) <= 0 <= int(max_v))
    if allow_back:
        if zero_is_value:
            print(f"  [b] {tr('back')}")
        else:
            print(f"  [0] {tr('back')}")

    while True:
        raw = input(prompt).strip()

        # 空入力は既定値
        if not raw:
            return default

        # 戻る判定
        if allow_back:
            if zero_is_value:
                if _is_back_raw_no_zero(raw):
                    return BACK_TOKEN
            else:
                if _is_back_raw(raw):
                    return BACK_TOKEN

        # 数値として解釈（全角数字や小数点記号を補正）
        raw_n = _normalize_numeric_text(raw)
        try:
            v = int(raw_n)
        except Exception:
            print(tr("msg_enter_number"))
            continue

        if v < int(min_v) or v > int(max_v):
            print(tr("msg_range").format(min_v=min_v, max_v=max_v))
            continue
        return v


def apply_seed_pref(cfg: Dict[str, Any], seed_mode: str, seed_value: Optional[int], last_seed: Optional[int]) -> None:
    """seed設定（再現性）を cfg へ適用。SHUFFLE/OPT/HEX_LOCAL をなるべく同じ値へ連動させます。"""
    if not isinstance(cfg, dict):
        return

    if seed_mode == "same_last":
        v = last_seed if isinstance(last_seed, int) else None
        if isinstance(v, int):
            cfg["shuffle_seed"] = int(v)
            cfg["opt_seed"] = "same"
            cfg["hex_local_opt_seed"] = None
        else:
            cfg["shuffle_seed"] = "random"
            cfg["opt_seed"] = "random"
            cfg["hex_local_opt_seed"] = None
        return

    if seed_mode == "specify":
        v = seed_value if isinstance(seed_value, int) else None
        if isinstance(v, int):
            cfg["shuffle_seed"] = int(v)
            cfg["opt_seed"] = "same"
            cfg["hex_local_opt_seed"] = None
        else:
            cfg["shuffle_seed"] = "random"
            cfg["opt_seed"] = "random"
            cfg["hex_local_opt_seed"] = None
        return

    # random
    cfg["shuffle_seed"] = "random"
    cfg["opt_seed"] = "random"
    cfg["hex_local_opt_seed"] = None


def seed_menu(seed_mode: str, seed_value: Optional[int], last_seed: Optional[int]) -> Tuple[str, Optional[int]]:
    """シード（再現性）メニュー。戻る時は現状維持。"""
    # 既定は「戻る」に近い挙動にしたいので、default_index=0＆allow_back=True を使う
    choices = [tr("seed_random")]
    if isinstance(last_seed, int):
        choices.append(tr("seed_same").format(seed=last_seed))
    else:
        choices.append(tr("seed_same_na"))
    choices.append(tr("seed_specify"))

    ch = ask_choice(tr("seed_title"), choices, default_index=0, allow_back=True)
    if ch == BACK_TOKEN:
        return seed_mode, seed_value

    if ch == choices[0]:
        return "random", None
    if ch == choices[1]:
        return "same_last", None
    # specify
    dv = int(last_seed) if isinstance(last_seed, int) else (int(seed_value) if isinstance(seed_value, int) else 0)
    v = ask_int(tr("seed_value"), dv, 0, 9223372036854775807, allow_back=True)
    if v == BACK_TOKEN:
        return seed_mode, seed_value
    return "specify", int(v)

def ask_onoff(label: str, default_on: bool, allow_back: bool = True):
    d = 1 if default_on else 2
    c = ask_choice(label, [tr("on"), tr("off")], default_index=d, allow_back=allow_back)
    if c == BACK_TOKEN:
        return BACK_TOKEN
    return str(c).lower() == tr("on")


def ask_text(label: str, default: str = "", allow_back: bool = False):
    if default:
        prompt = f"{label} [{default}]: "
    else:
        prompt = f"{label}: "

    if allow_back:
        print(f"  [0] {tr('back')}")

    sys.stdout.flush()
    raw = input(prompt)
    raw = raw.rstrip("\n").rstrip("\r").strip()

    if allow_back and raw and _is_back_raw(raw):
        return BACK_TOKEN

    if not raw and default:
        return default

    return raw


def _float_or_default(s: str, default: float) -> float:
    """float 変換（日本語IMEの全角入力もなるべく受け付ける）。"""
    t = _normalize_numeric_text(str(s))
    if not t:
        return float(default)

    r = t.strip().lower()
    # 便利入力（0.0扱い）
    if r in ("off", "none", "無効", "なし", "無し"):
        return 0.0

    try:
        return float(t)
    except Exception:
        return float(default)


def ask_float(label: str, default: float, min_v: float, max_v: float, allow_back: bool = True):
    """簡易 float 入力。空入力は default。範囲外は丸める。

    NOTE:
      - allow_back=True のとき、従来は「0」を “戻る” として扱っていました。
      - しかし 0.0 が有効値の入力（例: lead alpha）では困るので、
        min_v..max_v に 0.0 が含まれる場合は “戻る” を b/back/戻る で受け付けます。
    """

    zero_is_value = (float(min_v) <= 0.0 <= float(max_v))

    while True:
        if allow_back and zero_is_value:
            print(f"  [b] {tr('back')}")
            raw = ask_text(label, default=str(default), allow_back=False)
            if _is_back_raw_no_zero(str(raw)):
                return BACK_TOKEN
        else:
            raw = ask_text(label, default=str(default), allow_back=allow_back)
            if raw == BACK_TOKEN:
                return BACK_TOKEN

        v = _float_or_default(str(raw), default)
        if v < float(min_v):
            v = float(min_v)
        if v > float(max_v):
            v = float(max_v)
        return v


# =============================================================================
# プリセット管理
# =============================================================================
def _normalize_cfg_archives(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """互換: zip_scan_enable/zip_scan を archives_enable に寄せる（表示用）。"""
    if not isinstance(cfg, dict):
        return {}
    out = dict(cfg)
    if "archives_enable" not in out:
        if "zip_scan_enable" in out:
            out["archives_enable"] = bool(out.get("zip_scan_enable"))
        elif "zip_scan" in out:
            out["archives_enable"] = bool(out.get("zip_scan"))
    return out


def _state_dir() -> Path:
    """状態/キャッシュ用の基準ディレクトリを返します（環境変数は使わない）。"""
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

def _models_dir() -> Path:
    """AIモデル用の基準ディレクトリを返します（環境変数は使わない）。"""
    try:
        if MODEL_DIR:
            p = Path(MODEL_DIR).expanduser()
            if not p.is_absolute():
                p = (Path(__file__).resolve().parent / p).resolve()
        else:
            # 既定：STATE_DIR 配下（= _kana_state/models）
            p = _state_dir() / MODEL_SUBDIR_NAME
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



def _state_item_path(key: str, *, is_dir: bool = False) -> Path:
    """STATE_DIR 配下の「状態ファイル/キャッシュ」の実体パスを返します。

    優先順位:
      1) STATE_PATH_OVERRIDES[key]（指定があれば）
      2) STATE_DIR（なければ launcher と同じフォルダの _kana_state）
         + STATE_FILE_BASENAMES[key]
    """
    try:
        raw = str(STATE_PATH_OVERRIDES.get(key, "")).strip()
    except Exception:
        raw = ""

    base = _state_dir()
    basename = str(STATE_FILE_BASENAMES.get(key, key))

    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (Path(__file__).resolve().parent / p).resolve()
        # 「フォルダ指定」のヒント（末尾が / or \、または実在ディレクトリ）
        is_dir_hint = raw.endswith(("/", "\\")) or (p.exists() and p.is_dir())

        if is_dir:
            # ディレクトリとして扱う（末尾スラッシュ/実在ディレクトリ/拡張子なし等）
            if (not is_dir_hint) and (p.suffix != ""):
                # もし誤ってファイルっぽいものを入れた場合は、その親を使う
                p = p.parent
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return p

        # ファイルキーに対して「フォルダ指定」が来た場合は、basename を連結
        if is_dir_hint:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return p / basename

        # 拡張子なし＆ファイル名っぽくない場合もフォルダ扱いにする
        if (p.suffix == "") and (not p.name.lower().endswith((".json", ".log", ".txt", ".csv", ".png", ".jpg", ".jpeg", ".webp"))):
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return p / basename

        # ファイルとして採用（親は念のため作る）
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    # 既定（STATE_DIR 配下）
    p = base / basename
    if is_dir:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    else:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    return p


def preset_file_path() -> Path:
    return _state_item_path("preset")


def effect_preset_file_path() -> Path:
    return _state_item_path("effect_preset")


def apply_state_paths_to_core(mod: Any, *, preserve_image_save_dir: bool = False) -> None:
    """core/ダミーへ状態ファイルの置き場をまとめて適用します。"""
    try:
        sd = _state_dir()
        # core の内部ヘルパが参照する基準ディレクトリも合わせる（相対のズレ防止）
        try:
            setattr(mod, "STATE_DIR", str(sd))
        except Exception:
            pass
        try:
            setattr(mod, "MODEL_DIR", str(_models_dir()))
        except Exception:
            pass
        # core 側が理解できるキーだけを設定（存在しない属性は無視）
        try:
            setattr(mod, "LOG_SAVE_DIR", str(sd))
        except Exception:
            pass
        if not bool(preserve_image_save_dir):
            try:
                setattr(mod, "IMAGE_SAVE_DIR", str(sd))
            except Exception:
                pass
        try:
            setattr(mod, "EXTERNAL_LAUNCHER_CONFIG_PATH", str(sd))
        except Exception:
            pass
        try:
            setattr(mod, "DHASH_CACHE_FILE", str(_state_item_path("dhash_cache")))
        except Exception:
            pass
        try:
            setattr(mod, "VIDEO_FRAME_CACHE_DIR", str(_state_item_path("video_cache_dir", is_dir=True)))
        except Exception:
            pass
        # ログ（LOG_ENABLE=True のときだけ使われる）
        try:
            setattr(mod, "LOG_FILE", str(_state_item_path("log_file")))
        except Exception:
            pass
    except Exception:
        return


# LUT ライブラリ（.cube の一覧・最近使ったLUT） ---------------------------------
def lut_library_file_path() -> Path:
    return _state_item_path("lut_library")

def load_lut_library(path: Path) -> Dict[str, Any]:
    """LUT ライブラリ設定を読み込みます。

    形式:
      { "lut_dir": "C:/.../LUTs", "recent": ["C:/.../a.cube", ...] }
    """
    lib: Dict[str, Any] = {"lut_dir": "", "recent": []}
    if not path.exists():
        return lib
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if isinstance(data.get("lut_dir"), str):
                lib["lut_dir"] = data.get("lut_dir", "")
            if isinstance(data.get("recent"), list):
                # 文字列だけを採用
                lib["recent"] = [str(x) for x in data["recent"] if isinstance(x, str)]
    except Exception:
        return lib
    return lib

def save_lut_library(path: Path, lib: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lut_dir": str(lib.get("lut_dir", "") or ""),
            "recent": [str(x) for x in (lib.get("recent") or []) if isinstance(x, str)],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        _kana_silent_exc('launcher:L1150', e)
        pass
def scan_lut_files(lut_dir: Path) -> List[Path]:
    """指定フォルダ配下（サブフォルダ含む）の .cube を列挙します（再帰）。"""
    try:
        if not lut_dir.exists() or not lut_dir.is_dir():
            return []
        files: List[Path] = []
        for p in lut_dir.rglob("*.cube"):
            try:
                if p.is_file() and p.suffix.lower() == ".cube":
                    files.append(p)
            except Exception as e:
                _kana_silent_exc('launcher:L1163', e)
                continue
        def _sort_key(p: Path) -> str:
            try:
                return str(p.relative_to(lut_dir)).lower()
            except Exception:
                return p.name.lower()

        files.sort(key=_sort_key)
        return files
    except Exception:
        return []

def _lut_recent_push(lib: Dict[str, Any], lut_path: str, max_keep: int = 12) -> None:
    p = str(lut_path or "")
    if not p:
        return
    rec = lib.get("recent")
    if not isinstance(rec, list):
        rec = []
    rec2 = [x for x in rec if isinstance(x, str) and x != p]
    rec2.insert(0, p)
    lib["recent"] = rec2[:max_keep]


def last_run_file_path() -> Path:
    return _state_item_path("lastrun")



def load_last_run(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "config" in data and isinstance(data["config"], dict):
            return data["config"]
        if isinstance(data, dict):
            # 旧形式（config 直書き）にも対応
            return data
    except Exception:
        return None
    return None


def save_last_run(path: Path, cfg: Dict[str, Any]) -> None:
    try:
        payload = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": cfg,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        _kana_silent_exc('launcher:L1224', e)
        pass
def load_presets(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict) and "presets" in data:
        data = data["presets"]
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "name" not in item or "config" not in item:
            continue
        out.append(item)
    return out


def save_presets(path: Path, presets: List[Dict[str, Any]]) -> None:
    """プリセットを保存します（原子的に置換 / 軽いリトライ付き）。

    Windows 環境で一時的に書き込みが弾かれることがあるため、
    いったん .tmp に書いてから os.replace で置き換えます。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "presets": presets}
    data = json.dumps(payload, ensure_ascii=False, indent=2)

    tmp = path.with_name(path.name + ".tmp")
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, path)
            return
        except Exception as e:
            last_err = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception as e:
                _kana_silent_exc('launcher:L1269', e)
                pass
            # ほんの少しだけ待って再試行
            time.sleep(0.05 * (attempt + 1))

    if last_err is not None:
        raise last_err


def load_layout_presets_clean(path: Path) -> List[Dict[str, Any]]:
    """並べ方プリセットを読み込む（高速版）。

    - 起動を重くするため、毎回のJSONクリーンアップ（読み込み→加工→保存）は行わない。
    - 「レイアウトプリセットに混入しがちなFX/UIキーの除去」は、適用時に都度（メモリ上で）
      _sanitize_layout_preset_cfg() により行う。
    - これにより、余計な副産物ファイル（マーカー等）を作らず、起動も軽く保つ。
    """
    return load_presets(path)

def _sanitize_layout_preset_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """並べ方プリセットの config を“適用用”にサニタイズして返す（ディスクは変更しない）。

    - 並べ方プリセットは「レイアウトのみ」を扱う方針。
      旧互換で混入しがちなエフェクト設定やUI表示設定は除去する。
    - この関数は cfg を破壊しない（コピーして返す）。
    """
    if not isinstance(cfg, dict):
        return {}

    cfg2 = dict(cfg)

    # 1) 既知のエフェクト設定キー（snake_case）を除去
    try:
        for k in list(_extract_effect_cfg_from_any(cfg2).keys()):
            cfg2.pop(k, None)
    except Exception as e:
        _kana_silent_exc('launcher:sanitize_preset:fx_extract', e)
        pass

    # 2) 旧互換で混入しがちなキーを除去
    extra_keys = {
        'fx', 'effect_cfg', 'effect', 'effects',
        'effect_preset', 'effect_preset_name', 'effect_preset_id',
    }
    for k in extra_keys:
        cfg2.pop(k, None)

    # 3) UI（表示）設定が混入していたら除去（レイアウトプリセットは見た目設定を保存しない方針）
    ui_keys = {
        'ui_style', 'unicode_bling',
        'progress_bar_style', 'progress_width',
    }
    for k in ui_keys:
        cfg2.pop(k, None)

    return cfg2

def get_preset_by_name(presets: list, name: str):
    """指定名のプリセットを探して index を返す（見つからなければ None）。

    ※ プリセット管理（名前変更）で同名衝突チェックに使います。
    """
    target = str(name).strip()
    if not target:
        return None
    for i, p in enumerate(presets):
        try:
            n = str(p.get("name", "")).strip()
        except Exception:
            n = ""
        if n == target:
            return i
    return None


def preset_summary(cfg: Dict[str, Any], include_fx: bool = True) -> str:
    """一覧表示用のサマリを生成（短め・統一フォーマット）。"""
    cfg = _normalize_cfg_archives(cfg)
    layout = str(cfg.get("layout", "?"))
    select_mode = str(cfg.get("select_mode", "random"))
    shuffle = bool(cfg.get("full_shuffle", False))

    parts: List[str] = []

    if layout == "grid":
        r = cfg.get("rows")
        c = cfg.get("cols")
        parts.append(f"grid({r}x{c})" if (r and c) else "grid")
    elif layout == "hex":
        orient = cfg.get("hex_orient")
        parts.append(f"hex({orient})" if orient else "hex")
        parts.append(f"count={cfg.get('count', '?')}")
    else:
        parts.append(layout)
        if layout.startswith("mosaic"):
            rows = cfg.get("rows")
            cols = cfg.get("cols")
            uniq = cfg.get("count")
            if rows: parts.append(f"rows={rows}")
            if cols: parts.append(f"cols~{cols}")
            if uniq: parts.append(f"uniq~{uniq}")

    gvt = str(cfg.get("grid_video_timeline", "off") or "off").strip().lower()
    if layout == "grid" and gvt in ("asc", "desc"):
        parts.append(f"timeline={gvt}")

    preserve = bool(cfg.get("preserve_input_order", False))

    if shuffle:
        parts.append("shuffle=ON")
    elif preserve:
        parts.append("preserve=ON")
    else:
        prof = cfg.get("profile")
        if prof: parts.append(f"place={prof}")
        if prof == "diagonal":
            d = cfg.get("diag_dir")
            if d: parts.append(f"diag={d}")

    opt_mode = str(cfg.get("opt_mode", "") or "").strip().lower()
    steps = cfg.get("steps")
    reheats = cfg.get("reheats")
    has_opt_params = (steps is not None) or (reheats is not None)
    # opt_mode=default のとき steps/reheats が残っていても『最適化ON』扱いにしない
    opt_on = bool(cfg.get("opt_enable")) or (opt_mode in ("tune", "on", "anneal"))

    if opt_on:
        if steps is not None and reheats is not None:
            parts.append(f"opt=anneal({steps}x{reheats})")
        elif steps is not None:
            parts.append(f"opt=steps={steps}")
        elif reheats is not None:
            parts.append(f"opt=reheats={reheats}")
        else:
            parts.append("opt=ON")
        if layout.startswith("mosaic") and cfg.get("k") is not None:
            parts.append(f"k={cfg.get('k')}")

    parts.append(f"select={select_mode}")

    if "archives_enable" in cfg or "zip_scan_enable" in cfg or "zip_scan" in cfg:
        arch_en = bool(cfg.get("archives_enable", cfg.get("zip_scan_enable", cfg.get("zip_scan", False))))
        parts.append(f"archives={tr('on') if arch_en else tr('off')}")
    else:
        parts.append("archives=core")

    # エフェクト（プリセットに含まれている場合のみ、短く反映）
    fx_keys = (
        'effects_enable',
        'halation_enable', 'halation_intensity', 'halation_radius', 'halation_threshold', 'halation_knee',
        'grain_enable', 'grain_amount',
        'clarity_enable', 'clarity_amount', 'clarity_radius',
        'unsharp_enable', 'unsharp_amount', 'unsharp_radius', 'unsharp_threshold',
        'denoise_mode', 'denoise_strength',
        'dehaze_enable', 'dehaze_amount', 'dehaze_radius',
        'shadowhighlight_enable', 'shadow_amount', 'highlight_amount',
        'clarity_enable', 'clarity_amount', 'clarity_radius',
        'unsharp_enable', 'unsharp_amount', 'unsharp_radius', 'unsharp_threshold',
        'dehaze_enable', 'dehaze_amount', 'dehaze_radius',
        'shadowhighlight_enable', 'shadow_amount', 'highlight_amount',
        'vignette_enable', 'vignette_strength', 'vignette_round',
        'sepia_enable', 'sepia_intensity',
        'vibrance_enable', 'vibrance_factor',
        'bw_effect_enable',
        'brightness_mode', 'auto_method', 'auto_target_mean',
        'manual_gain', 'manual_gamma',
    )
    if include_fx and any(k in cfg for k in fx_keys):
        fx_on = bool(cfg.get('effects_enable', True))
        if not fx_on:
            parts.append('fx:off')
        else:
            fx_parts: List[str] = []
            if bool(cfg.get('halation_enable', False)):
                it = float(cfg.get('halation_intensity', 0.30))
                rd = int(cfg.get('halation_radius', 18))
                th = float(cfg.get('halation_threshold', 0.70))
                fx_parts.append(f"H{it:.2f}@{rd}t{th:.2f}")
            if bool(cfg.get('clarity_enable', False)):
                ca = float(cfg.get('clarity_amount', 0.12))
                cr = float(cfg.get('clarity_radius', 2.0))
                crs = (f"{cr:.2f}".rstrip('0').rstrip('.'))
                fx_parts.append(f"C{ca:.2f}@{crs}")
            if bool(cfg.get('unsharp_enable', False)):
                ua = float(cfg.get('unsharp_amount', 0.35))
                ur = float(cfg.get('unsharp_radius', 1.2))
                uth = int(cfg.get('unsharp_threshold', 3))
                urs = (f"{ur:.2f}".rstrip('0').rstrip('.'))
                fx_parts.append(f"U{ua:.2f}@{urs}t{uth}")
            if bool(cfg.get('dehaze_enable', False)):
                da = float(cfg.get('dehaze_amount', 0.10))
                dr = int(cfg.get('dehaze_radius', 24))
                fx_parts.append(f"DH{da:.2f}@{dr}")
            if bool(cfg.get('shadowhighlight_enable', False)):
                sa = float(cfg.get('shadow_amount', 0.22))
                ha = float(cfg.get('highlight_amount', 0.18))
                fx_parts.append(f"SH{sa:.2f}/{ha:.2f}")
            if bool(cfg.get('vibrance_enable', False)):
                vf = float(cfg.get('vibrance_factor', 1.0))
                fx_parts.append(f"V{vf:.2f}")
            if bool(cfg.get('grain_enable', True)):
                ga = float(cfg.get('grain_amount', 0.15))
                fx_parts.append(f"G{ga:.2f}")
            if bool(cfg.get('vignette_enable', False)):
                vs = float(cfg.get('vignette_strength', 0.15))
                fx_parts.append(f"Vi{vs:.2f}")
            bm = str(cfg.get('brightness_mode', 'off')).strip().lower()
            if bm == 'auto':
                am = str(cfg.get('auto_method', 'hybrid')).strip().lower() or 'hybrid'
                fx_parts.append(f"B:auto/{am[:3]}")
            elif bm == 'manual':
                fx_parts.append('B:man')

            fx = ' '.join(fx_parts).strip()
            if not fx:
                fx = 'on'
            # なるべく短く
            parts.append('fx:' + fx)

    return " | ".join(parts)



def preset_summary_verbose_lines(cfg: Dict[str, Any]) -> List[str]:
    """前回表示用：できるだけ詳細なサマリを複数行で生成する。

    - 既存の preset_summary() は「短め」なので、メインメニューの「前回:」表示ではこちらを使う。
    - ここでは“簡略化しすぎない”ことを優先し、主要パラメータをなるべく落とさず表示する。
    """
    try:
        cfg = _normalize_cfg_archives(cfg)
    except Exception:
        pass

    def _onoff(v: Any) -> str:
        return tr('on') if bool(v) else tr('off')

    out_lines: List[str] = []

    # 1行目：出力サイズ / レイアウト / 選別 / 代表パラメータ
    t0: List[str] = []
    w = cfg.get('width', None)
    h = cfg.get('height', None)
    if isinstance(w, int) and isinstance(h, int) and (w > 0) and (h > 0):
        t0.append(f"out={w}x{h}")

    layout = str(cfg.get('layout', '?'))
    if layout == 'grid':
        r = cfg.get('rows')
        c = cfg.get('cols')
        if r and c:
            t0.append(f"layout=grid({r}x{c})")
        else:
            t0.append("layout=grid")
    elif layout == 'hex':
        orient = cfg.get('hex_orient')
        t0.append(f"layout=hex({orient})" if orient else "layout=hex")
    else:
        t0.append(f"layout={layout}")

    count = cfg.get('count', None)
    if isinstance(count, int) and count > 0:
        t0.append(f"count={count}")

    sel = cfg.get('select_mode', None)
    if sel:
        t0.append(f"select={sel}")

    prof = cfg.get('profile', None)
    if prof:
        t0.append(f"place={prof}")

    if bool(cfg.get('full_shuffle', False)):
        t0.append("full_shuffle=ON")
    elif bool(cfg.get('preserve_input_order', False)):
        t0.append("preserve=ON")

    if t0:
        out_lines.append(" | ".join(t0))

    # 2行目：最適化 / アーカイブ / 主要ON/OFF
    t1: List[str] = []

    opt_mode = str(cfg.get("opt_mode", "") or "").strip().lower()
    steps = cfg.get("steps")
    reheats = cfg.get("reheats")
    opt_on = bool(cfg.get("opt_enable")) or (opt_mode in ("tune", "on", "anneal"))
    if opt_on:
        if (steps is not None) and (reheats is not None):
            t1.append(f"opt=anneal({steps}x{reheats})")
        elif steps is not None:
            t1.append(f"opt=steps={steps}")
        elif reheats is not None:
            t1.append(f"opt=reheats={reheats}")
        else:
            t1.append("opt=ON")
        if cfg.get("k") is not None:
            t1.append(f"k={cfg.get('k')}")

    if ("archives_enable" in cfg) or ("zip_scan_enable" in cfg) or ("zip_scan" in cfg):
        arch_en = bool(cfg.get("archives_enable", cfg.get("zip_scan_enable", cfg.get("zip_scan", False))))
        t1.append(f"archives={_onoff(arch_en)}")

    if "video_active" in cfg:
        t1.append(f"video={_onoff(cfg.get('video_active'))}")
    if "face_ai_enable" in cfg:
        t1.append(f"face_ai={_onoff(cfg.get('face_ai_enable'))}")
    if "set_wallpaper" in cfg:
        t1.append(f"wallpaper={_onoff(cfg.get('set_wallpaper'))}")

    if t1:
        out_lines.append(" | ".join(t1))

    # 3行目：動画/顔AIの詳細（有効時のみ）
    t2: List[str] = []
    try:
        if bool(cfg.get("video_active", False)):
            vmode = cfg.get("video_mode")
            vsel = cfg.get("video_select_mode")
            vppv = cfg.get("video_frames_per_video")
            if vmode:
                t2.append(f"video_mode={vmode}")
            if vsel:
                t2.append(f"video_select={vsel}")
            if vppv not in (None, 0, "0"):
                t2.append(f"video_per_video={vppv}")
            gvt = cfg.get("grid_video_timeline")
            if gvt:
                t2.append(f"timeline={gvt}")
        if bool(cfg.get("face_ai_enable", False)):
            b = cfg.get("face_ai_backend")
            d = cfg.get("face_ai_device")
            s = cfg.get("face_ai_sensitivity")
            if b:
                t2.append(f"backend={b}")
            if d:
                t2.append(f"dev={d}")
            if s:
                t2.append(f"sens={s}")
    except Exception:
        pass

    if t2:
        out_lines.append(" | ".join(t2))

    return out_lines

def find_preset_by_name(presets: List[Dict[str, Any]], name: str) -> Optional[int]:
    for i, p in enumerate(presets):
        if str(p.get("name", "")) == name:
            return i
    return None


def print_preset_list(presets: List[Dict[str, Any]]) -> None:
    print(f"{tr('preset_title')}: {len(presets)}")
    for i, p in enumerate(presets, 1):
        name = str(p.get("name", "(no name)"))
        created = str(p.get("created_at", ""))
        cfg = p.get("config") if isinstance(p.get("config"), dict) else {}
        summ = preset_summary(cfg, include_fx=False)
        if created:
            print(f"  [{i}] {name}  ({created})")
        else:
            print(f"  [{i}] {name}")
        print(f"       {summ}")


# =============================================================================
# エフェクトプリセット（ランチャー側）
# =============================================================================

def effect_preset_summary(cfg: Dict[str, Any]) -> str:
    """エフェクト設定のサマリ（表示順を統一）。"""

    def _onoff(x: bool) -> str:
        return 'on' if bool(x) else 'off'

    def _f2(x: Any) -> str:
        try:
            return f"{float(x):.2f}"
        except Exception:
            return str(x)

    parts: List[str] = []
    effects_enable = bool(cfg.get('effects_enable', True))
    parts.append(f"effects={_onoff(effects_enable)}")
    if not effects_enable:
        return " | ".join(parts)

    # 表示順：光 → 色味/グレーディング → ディテール → 明るさ → 仕上げ
    # ---- 光（ハレーション）
    if bool(cfg.get('halation_enable', False)):
        it = float(cfg.get('halation_intensity', 0.30))
        rd = int(cfg.get('halation_radius', 18))
        th = float(cfg.get('halation_threshold', 0.70))
        kn = float(cfg.get('halation_knee', 0.08))
        parts.append(f"halation={it:.2f}@{rd}(thr={th:.2f},k={kn:.2f})")

    # ---- 色味/グレーディング（トーンカーブ / スプリットトーン / LUT / ビブランス / 白黒 / セピア）
    if bool(cfg.get('tonecurve_enable', False)):
        tm = str(cfg.get('tonecurve_mode', 'film')).strip().lower() or 'film'
        ts = float(cfg.get('tonecurve_strength', 0.35))
        parts.append(f"tonecurve={tm}@{ts:.2f}")

    if bool(cfg.get('split_tone_enable', False)):
        shh = float(cfg.get('split_tone_shadow_hue', 220.0))
        shs = float(cfg.get('split_tone_shadow_strength', 0.06))
        hih = float(cfg.get('split_tone_highlight_hue', 35.0))
        his = float(cfg.get('split_tone_highlight_strength', 0.05))
        bal = float(cfg.get('split_tone_balance', 0.0))
        parts.append(f"split=sh{shh:.0f}:{shs:.2f} hi{hih:.0f}:{his:.2f} bal={bal:.2f}")

    if bool(cfg.get('lut_enable', False)):
        lf = str(cfg.get('lut_file', '')).strip()
        if lf:
            try:
                nm = Path(lf).name
            except Exception:
                nm = lf
            ls = float(cfg.get('lut_strength', 0.30))
            parts.append(f"lut={nm}@{ls:.2f}")

    if bool(cfg.get('vibrance_enable', False)):
        vf = float(cfg.get('vibrance_factor', 1.0))
        parts.append(f"vibrance={vf:.2f}")

    if bool(cfg.get('bw_effect_enable', False)):
        parts.append('bw=on')

    if bool(cfg.get('sepia_enable', False)):
        si = float(cfg.get('sepia_intensity', 0.03))
        parts.append(f"sepia={si:.2f}")

    # ---- ディテール（クラリティ / アンシャープ / NR / デヘイズ）
    if bool(cfg.get('clarity_enable', False)):
        ca = float(cfg.get('clarity_amount', 0.12))
        cr = float(cfg.get('clarity_radius', 2.0))
        crs = (f"{cr:.2f}".rstrip('0').rstrip('.'))
        parts.append(f"clarity={ca:.2f}@{crs}")

    if bool(cfg.get('unsharp_enable', False)):
        ua = float(cfg.get('unsharp_amount', 0.35))
        ur = float(cfg.get('unsharp_radius', 1.2))
        uth = int(cfg.get('unsharp_threshold', 3))
        urs = (f"{ur:.2f}".rstrip('0').rstrip('.'))
        parts.append(f"unsharp={ua:.2f}@{urs}t{uth}")

    dm = str(cfg.get('denoise_mode', 'off')).strip().lower()
    if dm == 'bilateral':
        dm = 'edge'
    if dm and dm not in ('off', 'none', '0', 'false'):
        ds = float(cfg.get('denoise_strength', 0.25))
        if ds > 0.0:
            parts.append(f"nr={dm}@{ds:.2f}")

    if bool(cfg.get('dehaze_enable', False)):
        da = float(cfg.get('dehaze_amount', 0.10))
        dr = int(cfg.get('dehaze_radius', 24))
        parts.append(f"dehaze={da:.2f}@{dr}")

    # ---- 仕上げ（グレイン / ビネット）
    if bool(cfg.get('grain_enable', False)):
        amt = float(cfg.get('grain_amount', 0.15))
        parts.append(f"grain={amt:.2f}")

    if bool(cfg.get('vignette_enable', False)):
        st = float(cfg.get('vignette_strength', 0.15))
        vr = float(cfg.get('vignette_round', 0.50))
        parts.append(f"vignette={st:.2f}(r={vr:.2f})")

    # ---- 明るさ（Shadow/Highlight / 明るさ調整）
    if bool(cfg.get('shadowhighlight_enable', False)):
        sa = float(cfg.get('shadow_amount', 0.22))
        ha = float(cfg.get('highlight_amount', 0.18))
        parts.append(f"shadow/high={sa:.2f}/{ha:.2f}")

    bm = str(cfg.get('brightness_mode', 'off')).strip().lower()
    if bm and bm != 'off':
        if bm == 'auto':
            am = str(cfg.get('auto_method', 'hybrid')).strip().lower() or 'hybrid'
            tgt = float(cfg.get('auto_target_mean', 0.50))
            parts.append(f"brightness=auto({am},tgt={tgt:.2f})")
        elif bm == 'manual':
            mg = float(cfg.get('manual_gain', 1.00))
            gm = float(cfg.get('manual_gamma', 1.00))
            parts.append(f"brightness=manual(g={mg:.2f},gma={gm:.2f})")
        else:
            parts.append(f"brightness={bm}")

    return " | ".join(parts)
def _extract_effect_cfg_from_any(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """cfg からエフェクト設定だけを抽出（snake_case）"""
    if not isinstance(cfg, dict):
        return {}
    keys = [
        'effects_enable',
        'halation_enable', 'halation_intensity', 'halation_radius', 'halation_threshold', 'halation_knee',
        'grain_enable', 'grain_amount',
        'clarity_enable', 'clarity_amount', 'clarity_radius',
        'unsharp_enable', 'unsharp_amount', 'unsharp_radius', 'unsharp_threshold',
        'denoise_mode', 'denoise_strength',
        'dehaze_enable', 'dehaze_amount', 'dehaze_radius',
        'shadowhighlight_enable', 'shadow_amount', 'highlight_amount',
        'tonecurve_enable', 'tonecurve_mode', 'tonecurve_strength',
        'lut_enable', 'lut_file', 'lut_strength',
        'split_tone_enable', 'split_tone_shadow_hue', 'split_tone_shadow_strength', 'split_tone_highlight_hue', 'split_tone_highlight_strength', 'split_tone_balance',
        'vignette_enable', 'vignette_strength', 'vignette_round',
        'sepia_enable', 'sepia_intensity',
        'vibrance_enable', 'vibrance_factor',
        'bw_effect_enable',
        'brightness_mode', 'auto_method', 'auto_target_mean',
        'manual_gain', 'manual_gamma',
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if k in cfg:
            out[k] = cfg.get(k)
    return out


def _strip_fx_from_layout_presets(presets: list) -> list:
    """並べ方プリセットからエフェクトキーを除去して返す（後方互換のため）。

    - 旧バージョンで保存された fx 付きプリセットが残っていても、
      ランチャーはレイアウトのみを扱う方針とする。
    - 読み込み時に除去して保存し直すことで、以降の挙動を安定させる。
    """
    if not isinstance(presets, list):
        return []
    out: List[Dict[str, Any]] = []

    # 旧互換の“エフェクト混入”キー（念のため複数パターンを除去）
    extra_keys = {
        'fx', 'effect_cfg', 'effect', 'effects',
        'effect_preset', 'effect_preset_name', 'effect_preset_id',
    }

    for it in presets:
        if not isinstance(it, dict):
            continue

        cfg = it.get('config') if isinstance(it.get('config'), dict) else {}
        cfg2 = dict(cfg)

        # 1) 既知のエフェクト設定キー（snake_case）を除去
        try:
            for k in list(_extract_effect_cfg_from_any(cfg2).keys()):
                cfg2.pop(k, None)
        except Exception as e:
            _kana_silent_exc('launcher:L1663', e)
            pass
        # 2) 旧互換で混入しがちなキーを除去
        for k in extra_keys:
            cfg2.pop(k, None)

        # 2.5) UI（表示）設定が混入していたら除去（レイアウトプリセットは見た目設定を保存しない方針）
        ui_keys = {
            'ui_style', 'unicode_bling',
            'progress_bar_style', 'progress_width',
        }
        for k in ui_keys:
            cfg2.pop(k, None)

        # 3) item 側にも混入していたら除去
        it2 = dict(it)
        for k in extra_keys:
            it2.pop(k, None)
        for k in ui_keys:
            it2.pop(k, None)

        # config を更新（変化がなければ元のまま）
        if cfg2 != cfg:
            it2['config'] = cfg2

        # summary は“レイアウトのみ”に揃える
        try:
            it2['summary'] = preset_summary(cfg2, include_fx=False)
        except Exception as e:
            _kana_silent_exc('launcher:L1682', e)
            pass
        out.append(it2)

    return out
def print_effect_preset_list(presets: List[Dict[str, Any]]) -> None:
    print(f"エフェクトプリセット: {len(presets)}")
    for i, p in enumerate(presets, 1):
        name = str(p.get('name', '(no name)'))
        created = str(p.get('created_at', ''))
        cfg = p.get('config') if isinstance(p.get('config'), dict) else {}
        summ = effect_preset_summary(cfg)
        if created:
            print(f"  [{i}] {name}  ({created})")
        else:
            print(f"  [{i}] {name}")
        print(f"       {summ}")
def _manage_presets_common(
    presets: List[Dict[str, Any]],
    preset_path: Path,
    title: str,
    summary_fn,
    allow_show_created: bool = True,
) -> List[Dict[str, Any]]:
    """プリセットの管理（名前変更/削除/並び替え）。
    - 設定プリセット/エフェクトプリセットの両方で共通利用
    - 変更したら都度保存（ファイル反映）
    """

    while True:
        _launcher_banner(title)
        _launcher_note(f"保存先: {preset_path}")

        if not presets:
            print("（プリセットがありません）")
            # 何もないときは、Enterで戻る（選択肢を出さない）
            _ = input(f"{tr('back')}（Enter）: ")
            return presets

        # 一覧表示
        for i, p in enumerate(presets, 1):
            name = str(p.get("name", "(no name)"))
            created = str(p.get("created_at", ""))
            cfg = p.get("config") if isinstance(p.get("config"), dict) else {}
            summ = str(summary_fn(cfg))

            if allow_show_created and created:
                print(f"  [{i}] {name}  ({created})")
            else:
                print(f"  [{i}] {name}")
            print(f"       {summ}")

        idx = ask_int("管理するプリセット番号", 0, 1, len(presets), allow_back=True)
        if idx == BACK_TOKEN or idx == 0:
            return presets
        j = int(idx) - 1
        if j < 0 or j >= len(presets):
            continue

        cur = presets[j]
        cur_name = str(cur.get("name", "")) or "(no name)"

        act = ask_choice(
            "操作",
            ["名前変更", "削除", "上へ移動", "下へ移動"],
            default_index=0,
            allow_back=True,
        )
        if act == BACK_TOKEN:
            continue

        if act == "名前変更":
            while True:
                new_name = ask_text("新しい名前", default=cur_name, allow_back=True)
                if new_name == BACK_TOKEN:
                    break
                new_name = str(new_name).strip()
                if not new_name:
                    print("（空の名前は不可）")
                    continue
                # 既存名と衝突する場合はやめる（混乱防止）
                exists = get_preset_by_name(presets, new_name)
                if exists is not None and exists != j:
                    print("（同名のプリセットがあります。別名にしてください）")
                    continue
                old_name = cur.get("name", "")
                cur["name"] = new_name
                presets[j] = cur
                try:
                    save_presets(preset_path, presets)
                except Exception as e:
                    # 失敗したらロールバック（名前を元に戻す）
                    cur["name"] = old_name
                    presets[j] = cur
                    print(f"（保存に失敗しました: {e}）")
                    print("（別のランチャーが同時に動いている場合は閉じてから再試行してにゃ）")
                    continue
                print(f"OK: 名前変更 → {new_name}")
                break
            continue

        if act == "削除":
            ok = ask_choice("削除していい？", ["はい", "いいえ"], default_index=2, allow_back=True)
            if ok == BACK_TOKEN or ok == "いいえ":
                continue
            try:
                removed = presets.pop(j)
                try:
                    save_presets(preset_path, presets)
                except Exception as e:
                    # 失敗したらロールバック（削除を取り消し）
                    presets.insert(j, removed)
                    print(f"（保存に失敗しました: {e}）")
                    print("（別のランチャーが同時に動いている場合は閉じてから再試行してにゃ）")
                    continue
                print(f"OK: 削除 → {removed.get('name', '')}")
            except Exception as e:
                _kana_silent_exc('launcher:L1801', e)
                pass
            continue

        if act == "上へ移動":
            if j <= 0:
                print("（これ以上上へ移動できません）")
                continue
            presets[j - 1], presets[j] = presets[j], presets[j - 1]
            try:
                save_presets(preset_path, presets)
            except Exception as e:
                # 失敗したらロールバック（並び替えを取り消し）
                presets[j - 1], presets[j] = presets[j], presets[j - 1]
                print(f"（保存に失敗しました: {e}）")
                print("（別のランチャーが同時に動いている場合は閉じてから再試行してにゃ）")
                continue
            print("OK: 上へ移動")
            continue

        if act == "下へ移動":
            if j >= len(presets) - 1:
                print("（これ以上下へ移動できません）")
                continue
            presets[j + 1], presets[j] = presets[j], presets[j + 1]
            try:
                save_presets(preset_path, presets)
            except Exception as e:
                # 失敗したらロールバック（並び替えを取り消し）
                presets[j + 1], presets[j] = presets[j], presets[j + 1]
                print(f"（保存に失敗しました: {e}）")
                print("（別のランチャーが同時に動いている場合は閉じてから再試行してにゃ）")
                continue
            print("OK: 下へ移動")
            continue


def manage_config_presets(preset_path: Path) -> None:
    """設定プリセット（kana_wallpaper_presets.json）の管理画面。"""
    presets = load_layout_presets_clean(preset_path)

    def _summ(cfg: dict) -> str:
        # 設定プリセットはレイアウト情報のみ表示
        try:
            return preset_summary(cfg, include_fx=False)
        except Exception:
            return preset_summary(cfg, include_fx=False)

    _manage_presets_common(presets, preset_path, 'プリセット管理', _summ, allow_show_created=True)


def manage_effect_presets(preset_path: Path) -> None:
    """エフェクトプリセット（kana_wallpaper_effect_presets.json）の管理画面。"""
    presets = load_presets(preset_path)

    def _summ(cfg: Dict[str, Any]) -> str:
        return effect_preset_summary(cfg)

    _manage_presets_common(presets, preset_path, "エフェクトプリセット管理", _summ, allow_show_created=True)


def _effect_cfg_from_core(mod: Any) -> Dict[str, Any]:
    """core の現在値から、エフェクト設定だけを抽出（snake_case で返す）。"""
    def _g(name: str, default: Any):
        try:
            return getattr(mod, name)
        except Exception:
            return default

    return {
        'effects_enable': bool(_g('EFFECTS_ENABLE', True)),
        'halation_enable': bool(_g('HALATION_ENABLE', False)),
        'halation_intensity': float(_g('HALATION_INTENSITY', 0.30)),
        'halation_radius': int(_g('HALATION_RADIUS', 18)),
        'halation_threshold': float(_g('HALATION_THRESHOLD', 0.70)),
        'halation_knee': float(_g('HALATION_KNEE', 0.08)),
        'grain_enable': bool(_g('GRAIN_ENABLE', True)),
        'grain_amount': float(_g('GRAIN_AMOUNT', 0.15)),
        'clarity_enable': bool(_g('CLARITY_ENABLE', False)),
        'clarity_amount': float(_g('CLARITY_AMOUNT', 0.12)),
        'clarity_radius': float(_g('CLARITY_RADIUS', 2.0)),
        'unsharp_enable': bool(_g('UNSHARP_ENABLE', False)),
        'unsharp_amount': float(_g('UNSHARP_AMOUNT', 0.35)),
        'unsharp_radius': float(_g('UNSHARP_RADIUS', 1.2)),
        'unsharp_threshold': int(_g('UNSHARP_THRESHOLD', 3)),
        'denoise_mode': str(_g('DENOISE_MODE', 'off')),
        'denoise_strength': float(_g('DENOISE_STRENGTH', 0.25)),
        'dehaze_enable': bool(_g('DEHAZE_ENABLE', False)),
        'dehaze_amount': float(_g('DEHAZE_AMOUNT', 0.10)),
        'dehaze_radius': int(_g('DEHAZE_RADIUS', 24)),
        'shadowhighlight_enable': bool(_g('SHADOWHIGHLIGHT_ENABLE', False)),
        'shadow_amount': float(_g('SHADOW_AMOUNT', 0.22)),
        'highlight_amount': float(_g('HIGHLIGHT_AMOUNT', 0.18)),
        'tonecurve_enable': bool(_g('TONECURVE_ENABLE', False)),
        'tonecurve_mode': str(_g('TONECURVE_MODE', 'film')),
        'tonecurve_strength': float(_g('TONECURVE_STRENGTH', 0.35)),
        'split_tone_enable': bool(_g('SPLIT_TONE_ENABLE', False)),
        'split_tone_shadow_hue': float(_g('SPLIT_TONE_SHADOW_HUE', 220.0)),
        'split_tone_shadow_strength': float(_g('SPLIT_TONE_SHADOW_STRENGTH', 0.06)),
        'split_tone_highlight_hue': float(_g('SPLIT_TONE_HIGHLIGHT_HUE', 35.0)),
        'split_tone_highlight_strength': float(_g('SPLIT_TONE_HIGHLIGHT_STRENGTH', 0.05)),
        'split_tone_balance': float(_g('SPLIT_TONE_BALANCE', 0.0)),
        'vignette_enable': bool(_g('VIGNETTE_ENABLE', False)),
        'vignette_strength': float(_g('VIGNETTE_STRENGTH', 0.15)),
        'vignette_round': float(_g('VIGNETTE_ROUND', 0.50)),
        'sepia_enable': bool(_g('SEPIA_ENABLE', True)),
        'sepia_intensity': float(_g('SEPIA_INTENSITY', 0.03)),
        'vibrance_enable': bool(_g('VIBRANCE_ENABLE', False)),
        'vibrance_factor': float(_g('VIBRANCE_FACTOR', 1.0)),
        'bw_effect_enable': bool(_g('BW_EFFECT_ENABLE', False)),
        'brightness_mode': str(_g('BRIGHTNESS_MODE', 'off')),
        'auto_method': str(_g('AUTO_METHOD', 'hybrid')),
        'auto_target_mean': float(_g('AUTO_TARGET_MEAN', 0.50)),
        'manual_gain': float(_g('MANUAL_GAIN', 1.00)),
        'manual_gamma': float(_g('MANUAL_GAMMA', 1.00)),
    }


def _normalize_effect_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg) if isinstance(cfg, dict) else {}
    out['effects_enable'] = bool(out.get('effects_enable', True))
    out['halation_enable'] = bool(out.get('halation_enable', False))
    out['halation_intensity'] = float(out.get('halation_intensity', 0.30))
    out['halation_radius'] = int(out.get('halation_radius', 18))
    out['halation_threshold'] = float(out.get('halation_threshold', 0.70))
    out['halation_knee'] = float(out.get('halation_knee', 0.08))
    out['halation_threshold'] = max(0.0, min(1.0, out['halation_threshold']))
    out['halation_knee'] = max(0.0, min(0.5, out['halation_knee']))
    out['grain_enable'] = bool(out.get('grain_enable', True))
    out['grain_amount'] = float(out.get('grain_amount', 0.15))
    out['clarity_enable'] = bool(out.get('clarity_enable', False))
    out['clarity_amount'] = float(out.get('clarity_amount', 0.12))
    out['clarity_radius'] = float(out.get('clarity_radius', 2.0))
    out['unsharp_enable'] = bool(out.get('unsharp_enable', False))
    out['unsharp_amount'] = float(out.get('unsharp_amount', 0.35))
    out['unsharp_radius'] = float(out.get('unsharp_radius', 1.2))
    out['unsharp_threshold'] = int(out.get('unsharp_threshold', 3))
    
    # ノイズ除去（選択式）
    dm = str(out.get('denoise_mode', 'off')).strip().lower()
    if dm not in ('off', 'light', 'median', 'edge', 'bilateral', 'heavy', 'nlm'):
        dm = 'off'
    if dm == 'bilateral':
        dm = 'edge'
    out['denoise_mode'] = dm
    out['denoise_strength'] = float(out.get('denoise_strength', 0.25))
    out['denoise_strength'] = max(0.0, min(1.0, out['denoise_strength']))
    out['dehaze_enable'] = bool(out.get('dehaze_enable', False))
    out['dehaze_amount'] = float(out.get('dehaze_amount', 0.10))
    out['dehaze_radius'] = int(out.get('dehaze_radius', 24))
    out['dehaze_amount'] = max(0.0, min(1.0, out['dehaze_amount']))
    out['dehaze_radius'] = max(1, min(200, out['dehaze_radius']))
    out['shadowhighlight_enable'] = bool(out.get('shadowhighlight_enable', False))
    out['shadow_amount'] = float(out.get('shadow_amount', 0.22))
    out['highlight_amount'] = float(out.get('highlight_amount', 0.18))
    out['shadow_amount'] = max(0.0, min(1.0, out['shadow_amount']))
    out['highlight_amount'] = max(0.0, min(1.0, out['highlight_amount']))
    out['tonecurve_enable'] = bool(out.get('tonecurve_enable', False))
    tm = str(out.get('tonecurve_mode', 'film')).strip().lower()
    if tm not in ('film', 'liftgamma', 'custom'):
        tm = 'film'
    out['tonecurve_mode'] = tm
    out['tonecurve_strength'] = float(out.get('tonecurve_strength', 0.35))
    out['lut_enable'] = bool(out.get('lut_enable', False))
    out['lut_file'] = str(out.get('lut_file', ''))
    out['lut_strength'] = float(out.get('lut_strength', 0.30))
    out['lut_strength'] = max(0.0, min(1.0, out['lut_strength']))
    out['vignette_enable'] = bool(out.get('vignette_enable', False))
    out['vignette_strength'] = float(out.get('vignette_strength', 0.15))
    out['vignette_round'] = float(out.get('vignette_round', 0.50))
    out['sepia_enable'] = bool(out.get('sepia_enable', True))
    out['sepia_intensity'] = float(out.get('sepia_intensity', 0.03))
    out['vibrance_enable'] = bool(out.get('vibrance_enable', False))
    out['vibrance_factor'] = float(out.get('vibrance_factor', 1.0))
    out['split_tone_enable'] = bool(out.get('split_tone_enable', False))
    out['split_tone_shadow_hue'] = float(out.get('split_tone_shadow_hue', 220.0)) % 360.0
    out['split_tone_shadow_strength'] = float(out.get('split_tone_shadow_strength', 0.06))
    out['split_tone_highlight_hue'] = float(out.get('split_tone_highlight_hue', 35.0)) % 360.0
    out['split_tone_highlight_strength'] = float(out.get('split_tone_highlight_strength', 0.05))
    out['split_tone_balance'] = float(out.get('split_tone_balance', 0.0))
    # ざっくり安全域に丸める
    out['split_tone_shadow_strength'] = max(0.0, min(1.0, out['split_tone_shadow_strength']))
    out['split_tone_highlight_strength'] = max(0.0, min(1.0, out['split_tone_highlight_strength']))
    out['split_tone_balance'] = max(-1.0, min(1.0, out['split_tone_balance']))
    out['bw_effect_enable'] = bool(out.get('bw_effect_enable', False))

    bm = str(out.get('brightness_mode', 'off')).strip().lower()
    if bm not in ('off', 'auto', 'manual'):
        bm = 'off'
    out['brightness_mode'] = bm

    am = str(out.get('auto_method', 'hybrid')).strip().lower() or 'hybrid'
    if am not in ('hybrid', 'gamma', 'gain'):
        am = 'hybrid'
    out['auto_method'] = am
    out['auto_target_mean'] = float(out.get('auto_target_mean', 0.50))
    out['manual_gain'] = float(out.get('manual_gain', 1.00))
    out['manual_gamma'] = float(out.get('manual_gamma', 1.00))
    return out


def effect_menu(effect_cfg: Dict[str, Any], presets: List[Dict[str, Any]], preset_path: Path) -> Dict[str, Any]:
    """エフェクトメニュー（任意で入る）。
    - 「編集:」表記は廃止
    - カテゴリ分けサブメニューで管理しやすくする
    - 「ON項目一覧（カテゴリ横断）」で有効な項目だけを一画面に集約
    """
    effect_cfg = _normalize_effect_cfg(effect_cfg)


    # 表示言語（英語UIはビジネス英語）
    lang_en = (str(DEFAULT_LANG).lower() == "en")

    _EFF_LABEL_EN: Dict[str, str] = {
        'ハレーション': 'Halation',
        'スプリットトーン': 'Split tone',
        'トーンカーブ': 'Tone curve',
        'LUT': 'LUT',
        'ビブランス': 'Vibrance',
        '白黒': 'Black & white',
        'セピア': 'Sepia',
        'クラリティ': 'Clarity',
        'アンシャープ': 'Unsharp mask',
        'ノイズ除去': 'Noise reduction',
        'デヘイズ': 'Dehaze',
        'グレイン': 'Film grain',
        'ビネット': 'Vignette',
        '明るさ調整': 'Brightness adjustment',
        'Shadow/Highlight': 'Shadow/Highlight',
    }

    _EFF_TITLE_EN: Dict[str, str] = {
        'エフェクト': 'Effects',
        'エフェクトメニュー': 'Effects menu',
        'エフェクト: ON項目一覧': 'Effects: Enabled items',
        'エフェクト: 光': 'Effects: Light',
        'エフェクト: 色味/グレーディング': 'Effects: Color / Grading',
        'エフェクト: ディテール': 'Effects: Detail',
        'エフェクト: 仕上げ': 'Effects: Finish',
        'エフェクト: 明るさ': 'Effects: Brightness',
        '操作': 'Action',
        'ON項目一覧': 'Enabled items',
        '最近から選ぶ': 'Select from recent',
        'LUTを選ぶ': 'Select LUT',
        '同名があるよ。上書き？': 'A preset with the same name exists. Overwrite?',
        'ノイズ除去（Noise Reduction）': 'Noise reduction',
    }

    _EFF_MENU_EN: Dict[str, str] = {
        '全体ON/OFF（ワンタッチ）': 'Toggle all (one-touch)',
        'ON項目一覧（カテゴリ横断）': 'Show enabled items (all categories)',
        'プリセットから適用': 'Apply from preset',
        '現在の設定をプリセット保存': 'Save current settings as preset',
        'プリセット管理': 'Manage presets',
        'カテゴリ: 光（H）': 'Category: Light (H)',
        'カテゴリ: 色味/グレーディング（ST/TC/LUT/VB/BW/SP）': 'Category: Color / Grading (ST/TC/LUT/VB/BW/SP)',
        'カテゴリ: ディテール（CL/US/NR/DH）': 'Category: Detail (CL/US/NR/DH)',
        'カテゴリ: 仕上げ（GR/VG）': 'Category: Finish (GR/VG)',
        'カテゴリ: 明るさ（BR/SH）': 'Category: Brightness (BR/SH)',
        'LUTを選択（一覧）': 'Select LUT (list)',
        '最近から選択': 'Select from recent',
        'LUTフォルダを設定': 'Set LUT folder',
        'パスを直接入力': 'Enter path',
        '強さ（strength）': 'Strength',
    }

    def eff_label(name: str) -> str:
        return _EFF_LABEL_EN.get(name, name) if lang_en else name

    def eff_title(title: str) -> str:
        return _EFF_TITLE_EN.get(title, title) if lang_en else title

    def eff_menu_label(value: str) -> str:
        return _EFF_MENU_EN.get(value, value) if lang_en else value


    def _is_on_item(item: str) -> bool:
        """エフェクト項目が『ON』かどうか。ON項目一覧や表示判断に使います。"""
        try:
            if item == 'ハレーション':
                return bool(effect_cfg.get('halation_enable', False))
            if item == 'スプリットトーン':
                return bool(effect_cfg.get('split_tone_enable', False))
            if item == 'トーンカーブ':
                return bool(effect_cfg.get('tonecurve_enable', False))
            if item == 'LUT':
                lf = str(effect_cfg.get('lut_file', '')).strip()
                return bool(effect_cfg.get('lut_enable', False)) and bool(lf)
            if item == 'ビブランス':
                return bool(effect_cfg.get('vibrance_enable', False))
            if item == '白黒':
                return bool(effect_cfg.get('bw_effect_enable', False))
            if item == 'セピア':
                return bool(effect_cfg.get('sepia_enable', False))
            if item == 'クラリティ':
                return bool(effect_cfg.get('clarity_enable', False))
            if item == 'アンシャープ':
                return bool(effect_cfg.get('unsharp_enable', False))
            if item == 'ノイズ除去':
                dm = str(effect_cfg.get('denoise_mode', 'off')).strip().lower()
                return bool(dm) and dm not in ('off','none','0','false') and float(effect_cfg.get('denoise_strength', 0.25)) > 0.0

            if item == 'デヘイズ':
                return bool(effect_cfg.get('dehaze_enable', False))
            if item == 'グレイン':
                return bool(effect_cfg.get('grain_enable', False))
            if item == 'ビネット':
                return bool(effect_cfg.get('vignette_enable', False))
            if item == 'Shadow/Highlight':
                return bool(effect_cfg.get('shadowhighlight_enable', False))
            if item == '明るさ調整':
                bm = str(effect_cfg.get('brightness_mode', 'off')).strip().lower()
                return bool(bm) and bm != 'off'
        except Exception as e:
            _kana_silent_exc('launcher:L2048', e)
            pass
        return True


    # ------------------------------------------------------------
    # 個別編集（カテゴリ内から呼ぶ）
    # ------------------------------------------------------------
    def edit_halation() -> None:
        v = ask_onoff('HALATION_ENABLE', bool(effect_cfg.get('halation_enable', False)), allow_back=True)
        if v == BACK_TOKEN:
            return
        effect_cfg['halation_enable'] = bool(v)
        if effect_cfg['halation_enable']:
            it = ask_float('HALATION_INTENSITY（0.0〜1.0）', float(effect_cfg.get('halation_intensity', 0.30)), 0.0, 1.0, allow_back=True)
            if it == BACK_TOKEN:
                return
            rd = ask_int('HALATION_RADIUS（1〜80）', int(effect_cfg.get('halation_radius', 18)), 1, 80, allow_back=True)
            if rd == BACK_TOKEN:
                return
            th = ask_float(uistr('HALATION_THRESHOLD (0.0-1.0, recommended 0.60-0.80)', 'HALATION_THRESHOLD（0.0〜1.0 / 目安 0.60〜0.80）'), float(effect_cfg.get('halation_threshold', 0.70)), 0.0, 1.0, allow_back=True)
            if th == BACK_TOKEN:
                return
            kn = ask_float(uistr('HALATION_KNEE (0.0-0.5, recommended 0.05-0.15)', 'HALATION_KNEE（0.0〜0.5 / 目安 0.05〜0.15）'), float(effect_cfg.get('halation_knee', 0.08)), 0.0, 0.5, allow_back=True)
            if kn == BACK_TOKEN:
                return
            effect_cfg['halation_intensity'] = float(it)
            effect_cfg['halation_radius'] = int(rd)
            effect_cfg['halation_threshold'] = float(th)
            effect_cfg['halation_knee'] = float(kn)

    def edit_clarity() -> None:
        v = ask_onoff('CLARITY_ENABLE', bool(effect_cfg.get('clarity_enable', False)), allow_back=True)
        if v == BACK_TOKEN:
            return
        effect_cfg['clarity_enable'] = bool(v)
        if effect_cfg['clarity_enable']:
            ca = ask_float(uistr('CLARITY_AMOUNT (0.0-1.0, recommended 0.10-0.16)', 'CLARITY_AMOUNT（0.0〜1.0 / 目安 0.10〜0.16）'),
                           float(effect_cfg.get('clarity_amount', 0.12)), 0.0, 1.0, allow_back=True)
            if ca == BACK_TOKEN:
                return
            cr = ask_float(uistr('CLARITY_RADIUS (0.1-10.0, recommended 1.5-3.0)', 'CLARITY_RADIUS（0.1〜10.0 / 目安 1.5〜3.0）'),
                           float(effect_cfg.get('clarity_radius', 2.0)), 0.1, 10.0, allow_back=True)
            if cr == BACK_TOKEN:
                return
            effect_cfg['clarity_amount'] = float(ca)
            effect_cfg['clarity_radius'] = float(cr)

    def edit_unsharp() -> None:
        v = ask_onoff('UNSHARP_ENABLE', bool(effect_cfg.get('unsharp_enable', False)), allow_back=True)
        if v == BACK_TOKEN:
            return
        effect_cfg['unsharp_enable'] = bool(v)
        if effect_cfg['unsharp_enable']:
            ua = ask_float(uistr('UNSHARP_AMOUNT (0.0-1.0, recommended 0.25-0.55)', 'UNSHARP_AMOUNT（0.0〜1.0 / 目安 0.25〜0.55）'),
                           float(effect_cfg.get('unsharp_amount', 0.35)), 0.0, 1.0, allow_back=True)
            if ua == BACK_TOKEN:
                return
            ur = ask_float(uistr('UNSHARP_RADIUS (0.1-10.0, recommended 0.8-1.8)', 'UNSHARP_RADIUS（0.1〜10.0 / 目安 0.8〜1.8）'),
                           float(effect_cfg.get('unsharp_radius', 1.2)), 0.1, 10.0, allow_back=True)
            if ur == BACK_TOKEN:
                return
            uth = ask_int(uistr('UNSHARP_THRESHOLD (0-20, recommended 2-6)', 'UNSHARP_THRESHOLD（0〜20 / 目安 2〜6）'),
                          int(effect_cfg.get('unsharp_threshold', 3)), 0, 20, allow_back=True)
            if uth == BACK_TOKEN:
                return
            effect_cfg['unsharp_amount'] = float(ua)
            effect_cfg['unsharp_radius'] = float(ur)
            effect_cfg['unsharp_threshold'] = int(uth)


    def edit_denoise() -> None:
        """ノイズ除去（選択式）"""
        modes = [
            ('off',  'off（しない）'),
            ('light','light（軽い：輝度スムージング）'),
            ('median','median（点ノイズ向け）'),
            ('edge', 'edge（エッジ保護：やや重い）'),
            ('heavy','heavy（強い：高品質NR / 重い）'),
        ]
        cur = str(effect_cfg.get('denoise_mode', 'off')).strip().lower()
        if cur == 'bilateral':
            cur = 'edge'
        labels = [m[1] for m in modes]
        # デフォは「戻る」ではなく、現在値を選びやすく（allow_back で 0 戻る可）
        d = 1
        for i, (k, _) in enumerate(modes, 1):
            if k == cur:
                d = i
                break
        ch = ask_choice(eff_title('ノイズ除去（Noise Reduction）'), labels, default_index=d, allow_back=True)
        if ch == BACK_TOKEN:
            return
        sel_mode = modes[labels.index(ch)][0]
        effect_cfg['denoise_mode'] = sel_mode
        if sel_mode == 'off':
            return

        ds = ask_float(uistr('DENOISE_STRENGTH (0.0-1.0, recommended 0.15-0.40)', 'DENOISE_STRENGTH（0.0〜1.0 / 目安 0.15〜0.40）'),
                       float(effect_cfg.get('denoise_strength', 0.25)), 0.0, 1.0, allow_back=True)
        if ds == BACK_TOKEN:
            return
        effect_cfg['denoise_strength'] = float(ds)

    def edit_dehaze() -> None:
        v = ask_onoff('DEHAZE_ENABLE', bool(effect_cfg.get('dehaze_enable', False)), allow_back=True)
        if v == BACK_TOKEN:
            return
        effect_cfg['dehaze_enable'] = bool(v)
        if not v:
            return

        da = ask_float(uistr('DEHAZE_AMOUNT (0.00-1.00, recommended 0.05-0.20)', 'DEHAZE_AMOUNT（0.00〜1.00 / 目安 0.05〜0.20）'),
                       float(effect_cfg.get('dehaze_amount', 0.10)), 0.00, 1.00, allow_back=True)
        if da == BACK_TOKEN:
            return
        dr = ask_int(uistr('DEHAZE_RADIUS (8-80, recommended 16-40)', 'DEHAZE_RADIUS（8〜80 / 目安 16〜40）'),
                     int(effect_cfg.get('dehaze_radius', 24)), 8, 80, allow_back=True)
        if dr == BACK_TOKEN:
            return

        effect_cfg['dehaze_amount'] = float(da)
        effect_cfg['dehaze_radius'] = int(dr)

    def edit_tonecurve() -> None:
        v = ask_onoff('TONECURVE_ENABLE', bool(effect_cfg.get('tonecurve_enable', False)), allow_back=True)
        if v == BACK_TOKEN:
            return
        effect_cfg['tonecurve_enable'] = bool(v)
        if effect_cfg['tonecurve_enable']:
            modes = ['film', 'liftgamma', 'custom']
            labels = ['film（フィルム）', 'liftgamma（ふんわり）', 'custom（将来拡張）']
            cur = str(effect_cfg.get('tonecurve_mode', 'film')).strip().lower()
            d = modes.index(cur) + 1 if cur in modes else 1
            chosen = ask_choice('TONECURVE_MODE', labels, default_index=d, allow_back=True)
            if chosen == BACK_TOKEN:
                return
            try:
                j = labels.index(chosen)
                m2 = modes[j]
            except Exception:
                m2 = 'film'
            st = ask_float(uistr('TONECURVE_STRENGTH (0.0-1.0, recommended 0.20-0.45)', 'TONECURVE_STRENGTH（0.0〜1.0 / 目安 0.20〜0.45）'),
                           float(effect_cfg.get('tonecurve_strength', 0.35)), 0.0, 1.0, allow_back=True)
            if st == BACK_TOKEN:
                return
            effect_cfg['tonecurve_mode'] = str(m2)
            effect_cfg['tonecurve_strength'] = float(st)


    def edit_lut() -> None:
        """3D LUT（.cube）を適用して、色の世界観を切り替えます。

        - LUT フォルダを一度登録すると、以降は一覧から番号で選べます。
        - 最近使ったLUT（MRU）も使えます。
        """
        v = ask_onoff('LUT_ENABLE', bool(effect_cfg.get('lut_enable', False)), allow_back=True)
        if v == BACK_TOKEN:
            return
        effect_cfg['lut_enable'] = bool(v)
        if not v:
            return

        lib_path = lut_library_file_path()
        lib = load_lut_library(lib_path)

        # 既定候補（未設定なら推測）
        default_dir = str(lib.get('lut_dir', '') or '')
        if not default_dir:
            cur_file = str(effect_cfg.get('lut_file', '') or '')
            if cur_file:
                try:
                    default_dir = str(Path(cur_file).expanduser().resolve().parent)
                except Exception:
                    default_dir = ''
        if not default_dir:
            try:
                default_dir = LUT_DEFAULT_DIR_FALLBACK
            except Exception:
                default_dir = ''

        while True:
            cur_file = str(effect_cfg.get('lut_file', '') or '')
            unset_label = "unset" if DEFAULT_LANG == "en" else "（未設定）"
            cur_name = Path(cur_file).name if cur_file else unset_label
            lut_dir = str(lib.get('lut_dir', '') or '')

            print(f"  • LUT: {cur_name}")
            print(f"  • LUT folder: {lut_dir if lut_dir else 'unset'}" if lang_en else f"  • LUTフォルダ: {lut_dir if lut_dir else '（未設定）'}")
            rec = lib.get('recent')
            rec_count = len(rec) if isinstance(rec, list) else 0
            print(f"  • Recent: {rec_count}" if lang_en else f"  • 最近: {rec_count}件")

            choices = [
                'LUTを選択（一覧）',
                '最近から選択',
                'LUTフォルダを設定',
                'パスを直接入力',
                '強さ（strength）',
            ]
            sel = ask_choice(eff_title('操作'), [(c, eff_menu_label(c)) for c in choices], default_index=0, allow_back=True)
            if sel == BACK_TOKEN:
                save_lut_library(lib_path, lib)
                return

            if sel == 'LUTフォルダを設定':
                while True:
                    p = ask_text('LUTフォルダ（.cube を置いた場所）', default=lut_dir or default_dir, allow_back=True)
                    if p == BACK_TOKEN:
                        break
                    p2 = str(p).strip().strip('"')
                    if not p2:
                        print("  • 空です。フォルダパスを入力してください。")
                        continue
                    d = Path(p2).expanduser()
                    if not d.exists() or not d.is_dir():
                        print("  • フォルダが見つかりません。もう一度入力してください。")
                        continue
                    lib['lut_dir'] = str(d)
                    default_dir = str(d)
                    save_lut_library(lib_path, lib)
                    print("  • LUTフォルダを保存しました。")
                    break
                continue

            if sel == 'LUTを選択（一覧）':
                if not lut_dir:
                    print("  • LUTフォルダが未設定です。先に『LUTフォルダを設定』を行ってください。")
                    continue
                base_dir = Path(lut_dir).expanduser()
                files = scan_lut_files(base_dir)
                if not files:
                    print("  • .cube が見つかりません。フォルダ（サブフォルダ含む）を確認してください。")
                    continue
                name_map: Dict[str, Path] = {}
                names: List[str] = []
                for p in files:
                    try:
                        key = str(p.relative_to(base_dir))
                    except Exception:
                        key = p.name
                    # 念のため重複があれば末尾に回数を付けてユニーク化
                    if key in name_map:
                        i = 2
                        new_key = f"{key} ({i})"
                        while new_key in name_map:
                            i += 1
                            new_key = f"{key} ({i})"
                        key = new_key
                    name_map[key] = p
                    names.append(key)

                picked = ask_choice(eff_title('LUTを選ぶ'), names, default_index=0, allow_back=True)
                if picked == BACK_TOKEN:
                    continue
                pick_path = name_map.get(picked)
                if pick_path is None:
                    continue
                effect_cfg['lut_file'] = str(pick_path)
                _lut_recent_push(lib, str(pick_path))
                save_lut_library(lib_path, lib)
                continue

            if sel == '最近から選択':
                rec = lib.get('recent')
                if not isinstance(rec, list):
                    rec = []
                live: List[str] = []
                for s in rec:
                    if isinstance(s, str) and s:
                        try:
                            p = Path(s).expanduser()
                            if p.exists() and p.is_file():
                                live.append(str(p))
                        except Exception as e:
                            _kana_silent_exc('launcher:L2324', e)
                            pass
                if not live:
                    print("  • 最近のLUTがありません。『LUTを選択（一覧）』で選ぶと追加されます。")
                    continue
                disp = [Path(s).name for s in live]
                picked = ask_choice(eff_title('最近から選ぶ'), disp, default_index=0, allow_back=True)
                if picked == BACK_TOKEN:
                    continue
                chosen_path = None
                for s in live:
                    if Path(s).name == picked:
                        chosen_path = s
                        break
                if chosen_path is None:
                    continue
                effect_cfg['lut_file'] = str(chosen_path)
                _lut_recent_push(lib, str(chosen_path))
                if not lut_dir:
                    try:
                        lib['lut_dir'] = str(Path(chosen_path).expanduser().resolve().parent)
                    except Exception as e:
                        _kana_silent_exc('launcher:L2345', e)
                        pass
                save_lut_library(lib_path, lib)
                continue

            if sel == 'パスを直接入力':
                p = ask_text('LUT_FILE（.cube のパス）', default=cur_file, allow_back=True)
                if p == BACK_TOKEN:
                    continue
                p2 = str(p).strip().strip('"')
                effect_cfg['lut_file'] = p2
                try:
                    pp = Path(p2).expanduser()
                    if pp.exists() and pp.is_file():
                        _lut_recent_push(lib, str(pp))
                        if not lut_dir:
                            lib['lut_dir'] = str(pp.resolve().parent)
                except Exception as e:
                    _kana_silent_exc('launcher:L2362', e)
                    pass
                save_lut_library(lib_path, lib)
                continue

            if sel == '強さ（strength）':
                st = ask_float(uistr('LUT_STRENGTH (0.00-1.00, recommended 0.15-0.40)', 'LUT_STRENGTH（0.00〜1.00 / 目安 0.15〜0.40）'),
                               float(effect_cfg.get('lut_strength', 0.30)), 0.00, 1.00, allow_back=True)
                if st == BACK_TOKEN:
                    continue
                effect_cfg['lut_strength'] = float(st)
                continue

    def edit_split_tone() -> None:
        v = ask_onoff('SPLIT_TONE_ENABLE', bool(effect_cfg.get('split_tone_enable', False)), allow_back=True)
        if v == BACK_TOKEN:
            return
        effect_cfg['split_tone_enable'] = bool(v)
        if effect_cfg['split_tone_enable']:
            shh = ask_float('SPLIT_TONE_SHADOW_HUE（0〜360）', float(effect_cfg.get('split_tone_shadow_hue', 220.0)), 0.0, 360.0, allow_back=True)
            if shh == BACK_TOKEN:
                return
            shs = ask_float(uistr('SPLIT_TONE_SHADOW_STRENGTH (0.0-1.0, recommended 0.03-0.10)', 'SPLIT_TONE_SHADOW_STRENGTH（0.0〜1.0 / 目安 0.03〜0.10）'), float(effect_cfg.get('split_tone_shadow_strength', 0.06)), 0.0, 1.0, allow_back=True)
            if shs == BACK_TOKEN:
                return
            hih = ask_float('SPLIT_TONE_HIGHLIGHT_HUE（0〜360）', float(effect_cfg.get('split_tone_highlight_hue', 35.0)), 0.0, 360.0, allow_back=True)
            if hih == BACK_TOKEN:
                return
            his = ask_float(uistr('SPLIT_TONE_HIGHLIGHT_STRENGTH (0.0-1.0, recommended 0.03-0.10)', 'SPLIT_TONE_HIGHLIGHT_STRENGTH（0.0〜1.0 / 目安 0.03〜0.10）'), float(effect_cfg.get('split_tone_highlight_strength', 0.05)), 0.0, 1.0, allow_back=True)
            if his == BACK_TOKEN:
                return
            bal = ask_float(uistr('SPLIT_TONE_BALANCE (-1.0 to 1.0, 0.0 = neutral)', 'SPLIT_TONE_BALANCE（-1.0〜1.0 / 0.0で中間）'), float(effect_cfg.get('split_tone_balance', 0.0)), -1.0, 1.0, allow_back=True)
            if bal == BACK_TOKEN:
                return
            effect_cfg['split_tone_shadow_hue'] = float(shh) % 360.0
            effect_cfg['split_tone_shadow_strength'] = float(shs)
            effect_cfg['split_tone_highlight_hue'] = float(hih) % 360.0
            effect_cfg['split_tone_highlight_strength'] = float(his)
            effect_cfg['split_tone_balance'] = float(bal)

    def edit_grain() -> None:
        v = ask_onoff('GRAIN_ENABLE', bool(effect_cfg.get('grain_enable', True)), allow_back=True)
        if v == BACK_TOKEN:
            return
        effect_cfg['grain_enable'] = bool(v)
        if effect_cfg['grain_enable']:
            amt = ask_float('GRAIN_AMOUNT（0.0〜1.0）', float(effect_cfg.get('grain_amount', 0.15)), 0.0, 1.0, allow_back=True)
            if amt == BACK_TOKEN:
                return
            effect_cfg['grain_amount'] = float(amt)

    def edit_vignette() -> None:
        v = ask_onoff('VIGNETTE_ENABLE', bool(effect_cfg.get('vignette_enable', False)), allow_back=True)
        if v == BACK_TOKEN:
            return
        effect_cfg['vignette_enable'] = bool(v)
        if effect_cfg['vignette_enable']:
            st = ask_float('VIGNETTE_STRENGTH（0.0〜1.0）', float(effect_cfg.get('vignette_strength', 0.15)), 0.0, 1.0, allow_back=True)
            if st == BACK_TOKEN:
                return
            vr = ask_float(uistr('VIGNETTE_ROUND (0.0-1.0, 0.5 = balanced)', 'VIGNETTE_ROUND（0.0〜1.0 / 0.5で均等）'), float(effect_cfg.get('vignette_round', 0.50)), 0.0, 1.0, allow_back=True)
            if vr == BACK_TOKEN:
                return
            effect_cfg['vignette_strength'] = float(st)
            effect_cfg['vignette_round'] = float(vr)

    def edit_sepia() -> None:
        v = ask_onoff('SEPIA_ENABLE', bool(effect_cfg.get('sepia_enable', True)), allow_back=True)
        if v == BACK_TOKEN:
            return
        effect_cfg['sepia_enable'] = bool(v)
        if effect_cfg['sepia_enable']:
            si = ask_float('SEPIA_INTENSITY（0.0〜1.0）', float(effect_cfg.get('sepia_intensity', 0.03)), 0.0, 1.0, allow_back=True)
            if si == BACK_TOKEN:
                return
            effect_cfg['sepia_intensity'] = float(si)

    def edit_vibrance() -> None:
        v = ask_onoff('VIBRANCE_ENABLE', bool(effect_cfg.get('vibrance_enable', False)), allow_back=True)
        if v == BACK_TOKEN:
            return
        effect_cfg['vibrance_enable'] = bool(v)
        if effect_cfg['vibrance_enable']:
            vf = ask_float('VIBRANCE_FACTOR（1.0〜2.0）', float(effect_cfg.get('vibrance_factor', 1.0)), 1.0, 2.0, allow_back=True)
            if vf == BACK_TOKEN:
                return
            effect_cfg['vibrance_factor'] = float(vf)

    def edit_bw() -> None:
        v = ask_onoff('BW_EFFECT_ENABLE', bool(effect_cfg.get('bw_effect_enable', False)), allow_back=True)
        if v == BACK_TOKEN:
            return
        effect_cfg['bw_effect_enable'] = bool(v)
    def edit_shadow_highlight() -> None:
        v = ask_onoff('SHADOWHIGHLIGHT_ENABLE', bool(effect_cfg.get('shadowhighlight_enable', False)), allow_back=True)
        if v == BACK_TOKEN:
            return
        effect_cfg['shadowhighlight_enable'] = bool(v)
        if not v:
            return

        sa = ask_float('SHADOW_AMOUNT（0.00〜0.60）', float(effect_cfg.get('shadow_amount', 0.22)), 0.00, 0.60, allow_back=True)
        if sa == BACK_TOKEN:
            return
        effect_cfg['shadow_amount'] = float(sa)

        ha = ask_float('HIGHLIGHT_AMOUNT（0.00〜0.60）', float(effect_cfg.get('highlight_amount', 0.18)), 0.00, 0.60, allow_back=True)
        if ha == BACK_TOKEN:
            return
        effect_cfg['highlight_amount'] = float(ha)


    def edit_brightness() -> None:
        bm = str(effect_cfg.get('brightness_mode', 'off')).strip().lower()
        opts = ['off', 'auto', 'manual']
        labels = ['off（しない）', 'auto（自動）', 'manual（手動）']
        d = opts.index(bm) + 1 if bm in opts else 1
        chosen = ask_choice('BRIGHTNESS_MODE', labels, default_index=d, allow_back=True)
        if chosen == BACK_TOKEN:
            return
        try:
            j = labels.index(chosen)
            bm2 = opts[j]
        except Exception:
            bm2 = 'off'
        effect_cfg['brightness_mode'] = bm2
        if bm2 == 'auto':
            # AUTO_METHOD は core 側で gamma/gain 以外は hybrid 扱い。
            am = str(effect_cfg.get('auto_method', 'hybrid')).strip().lower() or 'hybrid'
            aopts = ['hybrid', 'gamma', 'gain']
            alabels = ['hybrid（ガンマ→ゲイン）', 'gamma（ガンマのみ）', 'gain（ゲインのみ）']
            ad = aopts.index(am) + 1 if am in aopts else 1
            chosen2 = ask_choice('AUTO_METHOD', alabels, default_index=ad, allow_back=True)
            if chosen2 == BACK_TOKEN:
                return
            try:
                j2 = alabels.index(chosen2)
                am2 = aopts[j2]
            except Exception:
                am2 = 'hybrid'
            effect_cfg['auto_method'] = am2

            tgt = ask_float('AUTO_TARGET_MEAN（0.20〜0.80）', float(effect_cfg.get('auto_target_mean', 0.50)), 0.20, 0.80, allow_back=True)
            if tgt == BACK_TOKEN:
                return
            effect_cfg['auto_target_mean'] = float(tgt)
        elif bm2 == 'manual':
            g = ask_float('MANUAL_GAIN（0.50〜1.80）', float(effect_cfg.get('manual_gain', 1.00)), 0.50, 1.80, allow_back=True)
            if g == BACK_TOKEN:
                return
            gm = ask_float('MANUAL_GAMMA（0.30〜2.50）', float(effect_cfg.get('manual_gamma', 1.00)), 0.30, 2.50, allow_back=True)
            if gm == BACK_TOKEN:
                return
            effect_cfg['manual_gain'] = float(g)
            effect_cfg['manual_gamma'] = float(gm)

    # ------------------------------------------------------------
    # カテゴリサブメニュー（0で戻る）
    # ------------------------------------------------------------
    def submenu(title: str, choices: List[str], handlers: Dict[str, Any]) -> None:
        """カテゴリサブメニュー。内部キーは日本語のまま維持し、表示だけ英語UIで置換する。"""
        while True:
            disp_title = eff_title(title)
            _launcher_banner(disp_title)
            _launcher_note(effect_preset_summary(effect_cfg))
            disp_choices: List[Tuple[str, str]] = [(c, eff_label(c)) for c in choices]
            sel2 = ask_choice(disp_title, disp_choices, default_index=0, allow_back=True)
            if sel2 == BACK_TOKEN:
                return
            fn = handlers.get(sel2)
            if fn:
                fn()
            else:
                print(uistr("Not implemented.", "（未実装）"))

    def on_items_menu() -> None:
        """ON になっている項目だけをカテゴリ横断で一覧表示します（省略しない）。"""
        while True:
            _launcher_banner(eff_title('エフェクト: ON項目一覧'))
            _launcher_note(effect_preset_summary(effect_cfg))
            effects_enable = bool(effect_cfg.get('effects_enable', True))
            if not effects_enable:
                _launcher_note(uistr("Note: effects=off (global OFF; enabled settings below will not be applied).", "注意: effects=off（全体OFFのため、以下は設定がONでも適用されない）"))

            entries: List[Tuple[str, Any]] = []

            def _f2(x: Any) -> str:
                try:
                    return f"{float(x):.2f}"
                except Exception:
                    return str(x)

            def _add(name: str, detail: str, fn: Any) -> None:
                disp = eff_label(name)
                label = f"{disp}: {detail}" if detail else f"{disp}"
                entries.append((label, fn))

            # ---- 光
            if bool(effect_cfg.get('halation_enable', False)):
                it = float(effect_cfg.get('halation_intensity', 0.30))
                rd = int(effect_cfg.get('halation_radius', 18))
                th = float(effect_cfg.get('halation_threshold', 0.70))
                _add('ハレーション', f"{_f2(it)}@{rd}(thr={_f2(th)},k={_f2(float(effect_cfg.get('halation_knee', 0.08)))})", edit_halation)

            # ---- 色味/グレーディング
            if bool(effect_cfg.get('split_tone_enable', False)):
                shh = float(effect_cfg.get('split_tone_shadow_hue', 220.0))
                shs = float(effect_cfg.get('split_tone_shadow_strength', 0.06))
                hih = float(effect_cfg.get('split_tone_highlight_hue', 35.0))
                his = float(effect_cfg.get('split_tone_highlight_strength', 0.05))
                bal = float(effect_cfg.get('split_tone_balance', 0.0))
                _add('スプリットトーン', f"S{_f2(shh)}°×{_f2(shs)} / H{_f2(hih)}°×{_f2(his)} / bal={_f2(bal)}", edit_split_tone)

            if bool(effect_cfg.get('tonecurve_enable', False)):
                mode = str(effect_cfg.get('tonecurve_mode', 'film')).strip().lower() or 'film'
                st = float(effect_cfg.get('tonecurve_strength', 0.35))
                _add('トーンカーブ', f"{mode}@{_f2(st)}", edit_tonecurve)

            if bool(effect_cfg.get('lut_enable', False)):
                lf = str(effect_cfg.get('lut_file', '')).strip()
                if lf:
                    try:
                        nm = Path(lf).name
                    except Exception:
                        nm = lf
                    ls = float(effect_cfg.get('lut_strength', 0.30))
                    _add('LUT', f"{nm}@{_f2(ls)}", edit_lut)

            if bool(effect_cfg.get('vibrance_enable', False)):
                vf = float(effect_cfg.get('vibrance_factor', 1.20))
                _add('ビブランス', f"{_f2(vf)}", edit_vibrance)

            if bool(effect_cfg.get('bw_effect_enable', False)):
                bi = float(effect_cfg.get('bw_effect_intensity', 0.20))
                _add('白黒', f"{_f2(bi)}", edit_bw)

            if bool(effect_cfg.get('sepia_enable', False)):
                si = float(effect_cfg.get('sepia_intensity', 0.03))
                _add('セピア', f"{_f2(si)}", edit_sepia)

            # ---- ディテール
            if bool(effect_cfg.get('clarity_enable', False)):
                ca = float(effect_cfg.get('clarity_amount', 0.12))
                cr = float(effect_cfg.get('clarity_radius', 2.0))
                _add('クラリティ', f"{_f2(ca)}@{_f2(cr)}", edit_clarity)

            if bool(effect_cfg.get('unsharp_enable', False)):
                ua = float(effect_cfg.get('unsharp_amount', 0.35))
                ur = float(effect_cfg.get('unsharp_radius', 1.2))
                ut = int(effect_cfg.get('unsharp_threshold', 3))
                _add('アンシャープ', f"{_f2(ua)}@{_f2(ur)}t{ut}", edit_unsharp)

            # ---- ディテール（ノイズ除去）
            dm = str(effect_cfg.get('denoise_mode', 'off')).strip().lower()
            if dm == 'bilateral':
                dm = 'edge'
            if dm and dm not in ('off', 'none', '0', 'false') and float(effect_cfg.get('denoise_strength', 0.25)) > 0.0:
                ds = float(effect_cfg.get('denoise_strength', 0.25))
                _add('ノイズ除去', f"{dm}@{_f2(ds)}", edit_denoise)

            # ---- ディテール（デヘイズ）
            if bool(effect_cfg.get('dehaze_enable', False)):
                da = float(effect_cfg.get('dehaze_amount', 0.10))
                dr = int(effect_cfg.get('dehaze_radius', 24))
                _add('デヘイズ', f"{_f2(da)}@{dr}", edit_dehaze)

            # ---- 仕上げ
            if bool(effect_cfg.get('grain_enable', False)):
                ga = float(effect_cfg.get('grain_amount', 0.05))
                _add('グレイン', f"{_f2(ga)}", edit_grain)

            if bool(effect_cfg.get('vignette_enable', False)):
                vs = float(effect_cfg.get('vignette_strength', 0.15))
                vr = float(effect_cfg.get('vignette_round', 0.50))
                _add('ビネット', f"{_f2(vs)}(r={_f2(vr)})", edit_vignette)
            # ---- 明るさ
            if bool(effect_cfg.get('shadowhighlight_enable', False)):
                sa = float(effect_cfg.get('shadow_amount', 0.22))
                ha = float(effect_cfg.get('highlight_amount', 0.18))
                _add('Shadow/Highlight', f"S{_f2(sa)} / H{_f2(ha)}", edit_shadow_highlight)

            bm = str(effect_cfg.get('brightness_mode', 'off')).strip().lower()
            if bm and bm != 'off':
                if bm == 'auto':
                    am = str(effect_cfg.get('auto_method', 'hybrid')).strip().lower() or 'hybrid'
                    tgt = float(effect_cfg.get('auto_target_mean', 0.50))
                    _add('明るさ調整', f"auto({am},tgt={_f2(tgt)})", edit_brightness)
                elif bm == 'manual':
                    g = float(effect_cfg.get('manual_gain', 1.00))
                    gm = float(effect_cfg.get('manual_gamma', 1.00))
                    _add('明るさ調整', f"manual gain={_f2(g)} gamma={_f2(gm)}", edit_brightness)
                else:
                    _add('明るさ調整', bm, edit_brightness)
            if not entries:
                _launcher_note(uistr("No enabled items.", "ONの項目がないよ"))
                # ここでメニューを出すと『戻る』が二重に見えることがあるので、Enter 待ちだけにします
                try:
                    input(uistr('Press Enter to go back: ', 'Enterで戻る: '))
                except Exception as e:
                    _kana_silent_exc('launcher:L2656', e)
                    pass
                return
            back_label = tr('back')
            filtered: List[Tuple[str, Any]] = []
            for label, fn in entries:
                if str(label).strip().lower() == str(back_label).strip().lower():
                    # 万一『戻る』と同名の項目が混ざっても、UIが壊れないように回避
                    filtered.append((f"{label}（項目）", fn))
                else:
                    filtered.append((label, fn))
            opts = [x[0] for x in filtered]
            sel3 = ask_choice(eff_title('ON項目一覧'), opts, default_index=0, allow_back=True)
            if sel3 == BACK_TOKEN:
                return
            for label, fn in filtered:
                if sel3 == label:
                    fn()
                    break
            # 編集後は一覧に戻って更新表示
            continue

    while True:
        _launcher_banner(eff_title('エフェクト'))
        _launcher_note(effect_preset_summary(effect_cfg))
        _launcher_note(f"{uistr('Presets: ', 'プリセット: ')}{preset_path}")
        raw_choices = [
            '全体ON/OFF（ワンタッチ）',
            'ON項目一覧（カテゴリ横断）',
            'プリセットから適用',
            '現在の設定をプリセット保存',
            'プリセット管理',
            'カテゴリ: 光（H）',
            'カテゴリ: 色味/グレーディング（ST/TC/LUT/VB/BW/SP）',
            'カテゴリ: ディテール（CL/US/NR/DH）',
            'カテゴリ: 仕上げ（GR/VG）',
            'カテゴリ: 明るさ（BR/SH）',
        ]
        choices: List[Tuple[str, str]] = [(c, eff_menu_label(c)) for c in raw_choices]
        sel = ask_choice(eff_title('エフェクトメニュー'), choices, default_index=0, allow_back=True)
        if sel == BACK_TOKEN:
            return effect_cfg


        if sel == '全体ON/OFF（ワンタッチ）':
            effect_cfg['effects_enable'] = not bool(effect_cfg.get('effects_enable', True))
            continue

        if sel == 'ON項目一覧（カテゴリ横断）':
            on_items_menu()
            continue

        if sel == 'プリセットから適用':
            if not presets:
                print(uistr("No effect presets available.", "（エフェクトプリセットがありません）"))
                continue
            print_effect_preset_list(presets)
            idx = ask_int(uistr('Select', '選ぶ'), 1, 1,  len(presets), allow_back=True)
            if idx == BACK_TOKEN:
                continue
            p = presets[int(idx) - 1]
            pcfg = p.get('config') if isinstance(p.get('config'), dict) else {}
            effect_cfg.update(_normalize_effect_cfg(pcfg))
            continue

        if sel == '現在の設定をプリセット保存':
            while True:
                name = ask_text('保存名（自由）', allow_back=True)
                if name == BACK_TOKEN:
                    break
                if not name:
                    print(uistr('(Empty is not allowed.)', '（空は不可）'))
                    continue
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                item = {
                    'name': name,
                    'created_at': now,
                    'summary': effect_preset_summary(effect_cfg),
                    'config': dict(effect_cfg),
                }
                existing = find_preset_by_name(presets, name)
                if existing is not None:
                    ow = ask_choice(eff_title('同名があるよ。上書き？'), ['はい', 'いいえ'], default_index=2, allow_back=True)
                    if ow == BACK_TOKEN:
                        continue
                    if ow == 'はい':
                        presets[existing] = item
                        save_presets(preset_path, presets)
                else:
                    presets.append(item)
                    save_presets(preset_path, presets)
                break
            continue

        if sel == 'プリセット管理':
            try:
                manage_effect_presets(preset_path)
                presets[:] = load_presets(preset_path)
            except Exception as e:
                print(f"（エフェクトプリセット管理でエラー: {e}）")
            continue

        if sel == 'カテゴリ: 光（H）':
            items = ['ハレーション']
            submenu('エフェクト: 光', items, {
                'ハレーション': edit_halation,
            })
            continue

        if sel == 'カテゴリ: 色味/グレーディング（ST/TC/LUT/VB/BW/SP）':
            items = ['スプリットトーン', 'トーンカーブ', 'LUT', 'ビブランス', '白黒', 'セピア']
            submenu('エフェクト: 色味/グレーディング', items, {
                'スプリットトーン': edit_split_tone,
                'トーンカーブ': edit_tonecurve,
                'LUT': edit_lut,
                'ビブランス': edit_vibrance,
                '白黒': edit_bw,
                'セピア': edit_sepia,
            })
            continue

        if sel == 'カテゴリ: ディテール（CL/US/NR/DH）':
            items = ['クラリティ', 'アンシャープ', 'ノイズ除去', 'デヘイズ']
            submenu('エフェクト: ディテール', items, {
                'クラリティ': edit_clarity,
                'アンシャープ': edit_unsharp,
                'ノイズ除去': edit_denoise,
                'デヘイズ': edit_dehaze,
            })
            continue

        if sel == 'カテゴリ: 仕上げ（GR/VG）':
            items = ['グレイン', 'ビネット']
            submenu('エフェクト: 仕上げ', items, {
                'グレイン': edit_grain,
                'ビネット': edit_vignette,
            })
            continue

        if sel == 'カテゴリ: 明るさ（BR/SH）':
            items = ['明るさ調整', 'Shadow/Highlight']
            submenu('エフェクト: 明るさ', items, {
                '明るさ調整': edit_brightness,
                'Shadow/Highlight': edit_shadow_highlight,
            })
            continue

    return effect_cfg

# =============================================================================
# 本体（core）のロード
# =============================================================================


def load_core_module(core_path: Path):
    # 注:
    # ランチャーは core を importlib で読み込みます。
    # ProcessPool（Windows spawn）で ワーカー（worker）を使う場合、関数の __module__ が import 可能である必要があるため、
    # 固定名（kana_wallpaper_core）ではなく、実ファイル名（stem）を モジュール名に採用します。
    core_path = core_path.resolve()
    core_dir = str(core_path.parent)
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)

    module_name = core_path.stem
    # モジュール名として使える識別子に整形
    module_name = re.sub(r"\W", "_", module_name)
    if re.match(r"^\d", module_name):
        module_name = "_" + module_name

    spec = importlib.util.spec_from_file_location(module_name, str(core_path))
    if not spec or not spec.loader:
        raise RuntimeError(tr("err_core_load").format(path=str(core_path)))
    mod = importlib.util.module_from_spec(spec)
    # __module__ 参照が解決できるように sys.modules に登録
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)   # type: ignore（型チェック用）
    return mod


_RE_VNUM = re.compile(r"_v(\d+)")


def _extract_vnum(name: str) -> int:
    m = _RE_VNUM.search(name)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _collect_core_candidates(dir_path: Path) -> list[Path]:
    """指定フォルダから本体候補を集める（例外は握りつぶして空で返す）。

    NOTE:
      ここは glob が "kana_wallpaper_unified_final*.py" なので、ランチャー自体は
      そもそも候補に入りません。過去に "launcher" という部分文字列で除外したところ、
      本体側のファイル名に "no_launcher_..." のような語が含まれるケースまで除外してしまい、
      本体が見つからない誤検出になりました。

      → 余計な除外条件は持たず、glob の結果をそのまま採用します。
    """
    out: list[Path] = []
    try:
        if not dir_path or (not dir_path.exists()):
            return out
        for p in dir_path.glob("kana_wallpaper_unified_final*.py"):
            try:
                if not p.is_file():
                    continue
                out.append(p)
            except Exception:
                continue
    except Exception:
        return []
    return out


def pick_core_file(here: Path) -> Path:
    """本体ファイルを自動選択します（_v番号が大きいもの優先）。"""
    search_dirs: list[Path] = []

    # 1) ランチャーと同じフォルダ（最優先）
    try:
        if here:
            search_dirs.append(here)
    except Exception:
        pass

    # 2) カレントディレクトリ（cmd から実行した場合の保険）
    try:
        cwd = Path.cwd()
        if cwd not in search_dirs:
            search_dirs.append(cwd)
    except Exception:
        pass

    # 3) 典型的な Desktop（OneDrive リダイレクト等の保険）
    try:
        home = Path.home()
        for d in (
            home / "Desktop",
            home / "OneDrive" / "Desktop",
            home / "OneDrive" / "デスクトップ",
        ):
            if d not in search_dirs:
                search_dirs.append(d)
    except Exception:
        pass

    candidates: list[Path] = []
    for d in search_dirs:
        candidates.extend(_collect_core_candidates(d))

    # 重複除去（resolve できない環境もあるので例外は無視）
    uniq: list[Path] = []
    seen = set()
    for p in candidates:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    candidates = uniq

    if not candidates:
        # どこを探したかをメッセージに含める（ユーザーの切り分け用）
        msg = tr("err_no_core")
        try:
            msg += "\n検索場所: " + " | ".join([str(d) for d in search_dirs if d])
        except Exception:
            pass
        raise FileNotFoundError(msg)

    # _v番号 → 更新時刻 の順で優先
    candidates.sort(key=lambda p: (_extract_vnum(p.name), p.stat().st_mtime), reverse=True)
    return candidates[0]


def set_if(mod: Any, name: str, value: Any) -> None:
    """互換のため、存在に関係なくセットする（本体が参照するものだけ効く）。"""
    setattr(mod, name, value)


def _py_literal(v: Any) -> str:
    """Python へ書き出す値表現（Path等は文字列へ）"""
    try:
        from pathlib import Path as _Path
        if isinstance(v, _Path):
            return repr(str(v))
    except Exception as e:
        _kana_silent_exc('launcher:L2885', e)
        pass
    # numpy 等の型は Python の組込みへ寄せる
    try:
        import numpy as _np
        if isinstance(v, (_np.integer,)):
            return str(int(v))
        if isinstance(v, (_np.floating,)):
            return repr(float(v))
        if isinstance(v, (_np.bool_,)):
            return "True" if bool(v) else "False"
    except Exception as e:
        _kana_silent_exc('launcher:L2896', e)
        pass
    # 通常
    if isinstance(v, bool):
        return "True" if v else "False"
    return repr(v)


def _collect_core_defined_names_from_source(core_py: Path) -> set:
    """core.py を実行せずに、モジュール直下で代入される名前を収集します。

    目的:
      - 外部設定JSONに「coreが知らないキー（= globalsに存在しないキー）」を混ぜない
      - export時に core の import（重い / 依存ライブラリ不足で失敗）を避ける

    注意:
      - ここでの「存在する」は Python ソース上でモジュール直下に代入されている名前の近似です。
        （try/except 内の代入も拾います）
      - 解析に失敗した場合は空集合を返します（その場合はフィルタを行いません）
    """
    try:
        import ast

        src = core_py.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
        defined = set()

        def _add_target(t):
            try:
                if isinstance(t, ast.Name):
                    defined.add(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    for e in t.elts:
                        _add_target(e)
            except Exception as e:
                _kana_silent_exc('launcher:L2930', e)
                pass
        class _Visitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                for t in getattr(node, "targets", []) or []:
                    _add_target(t)
                try:
                    self.generic_visit(node.value)
                except Exception as e:
                    _kana_silent_exc('launcher:L2939', e)
                    pass
            def visit_AnnAssign(self, node):
                _add_target(getattr(node, "target", None))
                if getattr(node, "value", None) is not None:
                    try:
                        self.generic_visit(node.value)
                    except Exception as e:
                        _kana_silent_exc('launcher:L2947', e)
                        pass
            def visit_AugAssign(self, node):
                _add_target(getattr(node, "target", None))
                if getattr(node, "value", None) is not None:
                    try:
                        self.generic_visit(node.value)
                    except Exception as e:
                        _kana_silent_exc('launcher:L2955', e)
                        pass
            # 関数/クラスの中はグローバルではないので覗かない
            def visit_FunctionDef(self, node):
                return

            def visit_AsyncFunctionDef(self, node):
                return

            def visit_ClassDef(self, node):
                return

        _Visitor().visit(tree)
        return defined
    except Exception:
        return set()


def export_settings_to_core_inplace(core_path: str, cfg: dict, dd_paths=None, **_ignored) -> Optional[str]:
    """
    互換名: 以前は「本体へ書き込む」だったが、現在は「同フォルダの外部JSONへ書き出す」。

    - 本体ファイル自体は汚さない（配布・更新が楽）
    - 本体は起動時に JSON があれば自動適用（core側のローダが担当）
    - ここでは cfg（ランチャ内部の設定）を core のグローバル変数名へ変換して保存します
      （apply_config_to_core と同等のマッピングを利用）

    引数互換:
    - dd_paths など、旧実装で渡されていた追加引数は無視します（必要ならメタに残す）。
    """
    try:
        import json, os
        from types import SimpleNamespace

        cpath = Path(core_path).resolve()
        # 出力先は STATE_PATH_OVERRIDES['launcher_export_json'] で個別上書きできます（環境変数は使わない）
        out_path = _state_item_path("launcher_export_json")
        
        # 親フォルダが無ければ作成
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            _kana_silent_exc('launcher:L3009', e)
            pass
        # cfg（ランチャ内部）→ core変数名へマッピング
        dummy = SimpleNamespace()
        try:
            apply_config_to_core(dummy, cfg, list(dd_paths or []))
        except Exception:
            # マッピングに失敗しても、最低限 cfg を落とす（ただし core 側では無視される可能性あり）
            for k, v in (cfg or {}).items():
                try:
                    setattr(dummy, str(k), v)
                except Exception as e:
                    _kana_silent_exc('launcher:L3020', e)
                    pass
        # 状態ファイルの置き場（キャッシュ/ログ等）をまとめて適用
        apply_state_paths_to_core(dummy)

        core_defined = _collect_core_defined_names_from_source(cpath)
        dropped_unknown = 0

        core_kv = {}
        for k, v in vars(dummy).items():
            if not isinstance(k, str):
                continue
            if k.startswith("_"):
                continue
            if core_defined and (k not in core_defined):
                dropped_unknown += 1
                continue
            try:
                json.dumps(v)
                core_kv[k] = v
            except Exception:
                core_kv[k] = str(v)

        payload = dict(core_kv)
        payload["_meta"] = {
            "generated_by": "kana_wallpaper_launcher",
            "filter": "core_defined_names",
            "keys_written": int(len(core_kv)),
            "keys_dropped_unknown": int(dropped_unknown),
        }
        if dd_paths:
            try:
                payload["_meta_dd_paths"] = [str(p) for p in dd_paths]
            except Exception as e:
                _kana_silent_exc('launcher:L3051', e)
                pass
        tmp = str(out_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        os.replace(tmp, str(out_path))

        # note() が無い環境があるので安全に表示
        try:
            _note = globals().get("note", None)
            if callable(_note):
                _note(f"Exported external config: {out_path}")
            else:
                print(f"Exported external config: {out_path}")
        except Exception:
            print(f"Exported external config: {out_path}")

        return True
    except Exception as e:
        try:
            _note = globals().get("note", None)
            if callable(_note):
                _note(f"[ERROR] Failed to export external config: {e}")
            else:
                print(f"[ERROR] Failed to export external config: {e}")
        except Exception:
            print(f"[ERROR] Failed to export external config: {e}")
        return False

def set_many(mod: Any, names: List[str], value: Any) -> None:
    for n in names:
        setattr(mod, n, value)


# =============================================================================
# 設定収集（UI）
# =============================================================================


def _as_int_or_none(v: Any) -> Optional[int]:
    """int っぽい値を int に（失敗時は None）"""
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return int(v)
        s = str(v).strip()
        if s.isdigit():
            return int(s)
    except Exception as e:
        _kana_silent_exc('launcher:L3103', e)
        pass
    return None


def parse_dragdrop_paths(argv: List[str]) -> List[str]:
    """D&D で渡されたパス（引数）を拾う。オプションは無視。"""
    paths: List[str] = []
    for a in argv[1:]:
        if a.startswith("-"):
            continue
        p = Path(a.strip('"'))
        if p.exists():
            paths.append(str(p))
    return paths


def choose_layout(default: str) -> str:
    layouts = [tr("layout_grid"), tr("layout_hex"), tr("layout_mosaic_h"), tr("layout_mosaic_w"), tr("layout_quilt"), tr("layout_stained_glass"), tr("layout_random")]
    d = 1
    if default in layouts:
        d = layouts.index(default) + 1
    chosen = ask_choice(tr("layout"), layouts, default_index=d, allow_back=True)
    return BACK_TOKEN if chosen == BACK_TOKEN else str(chosen)


def choose_select_mode(default: str) -> str:
    # NOTE: 選択肢の「内部コード」は英語のまま維持します。
    #       表示ラベル（括弧内の短い説明）は UI 言語に合わせて自然な日本語にします。
    lang = DEFAULT_LANG if DEFAULT_LANG in TRANSLATIONS else "en"

    options = [
        ("random",   {"en": "random (shuffle)",           "ja": "random (シャッフル)"}),
        ("aesthetic", {"en": "aesthetic (score-based)",    "ja": "aesthetic (スコア重視)"}),
        ("recent",   {"en": "recent (newest first)",      "ja": "recent (新しい順)"}),
        ("oldest",   {"en": "oldest (oldest first)",      "ja": "oldest (古い順)"}),
        ("name_asc", {"en": "name_asc (filename A→Z)", "ja": "name_asc (ファイル名 A→Z)"}),
        ("name_desc",{"en": "name_desc (filename Z→A)", "ja": "name_desc (ファイル名 Z→A)"}),
    ]

    labels = [o[1].get(lang, o[1]["en"]) for o in options]

    d = 1
    for idx, (code, _) in enumerate(options, 1):
        if default == code:
            d = idx
            break

    chosen = ask_choice(tr("select_mode"), labels, default_index=d, allow_back=True)
    if chosen == BACK_TOKEN:
        return BACK_TOKEN

    try:
        j = labels.index(chosen)
        return options[j][0]
    except Exception:
        # fallback: take first token before space/paren
        s = str(chosen).strip().lower()
        for code, _ in options:
            if s.startswith(code):
                return code
        return str(default).strip().lower() or "random"


def choose_video_mode(default: str) -> str:
    # 表示ラベル → 内部コードへ変換
    options = [("auto", tr("video_mode_auto")), ("fixed", tr("video_mode_fixed"))]
    labels = [o[1] for o in options]
    d = 1
    for i, (code, _) in enumerate(options, 1):
        if default == code:
            d = i
            break
    chosen = ask_choice(tr("video_mode"), labels, default_index=d, allow_back=True)
    if chosen == BACK_TOKEN:
        return BACK_TOKEN
    try:
        idx = labels.index(chosen)
        return options[idx][0]
    except Exception:
        return "auto"


def choose_video_select_mode(default: str) -> str:
    options = [
        ("random", tr("video_sel_random")),
        ("uniform", tr("video_sel_uniform")),
        ("scene", tr("video_sel_scene")),
        ("scene_best", tr("video_sel_scene_best")),
        ("best_bright", tr("video_sel_best_bright")),
        ("best_sharp", tr("video_sel_best_sharp")),
        ("best_combo", tr("video_sel_best_combo")),
    ]
    labels = [o[1] for o in options]
    d = 1
    for i, (code, _) in enumerate(options, 1):
        if default == code:
            d = i
            break
    chosen = ask_choice(tr("video_select_mode"), labels, default_index=d, allow_back=True)
    if chosen == BACK_TOKEN:
        return BACK_TOKEN
    try:
        idx = labels.index(chosen)
        return options[idx][0]
    except Exception:
        return "random"


def choose_profile(default: str) -> str:
    choices = [
        tr("arr_diag"),
        tr("arr_hilb"),
        tr("arr_scatter"),
        tr("arr_as_is"),
    ]
    mapping = {
        tr("arr_diag"): "diagonal",
        tr("arr_hilb"): "hilbert",
        tr("arr_scatter"): "scatter",
        tr("arr_as_is"): "as_is",
    }
    rev = {v: k for k, v in mapping.items()}
    d = 1
    if default in rev:
        d = choices.index(rev[default]) + 1
    chosen = ask_choice(tr("arr_simple"), choices, default_index=d, allow_back=True)
    if chosen == BACK_TOKEN:
        return BACK_TOKEN
    return mapping[str(chosen)]


def choose_diag_dir(default: str) -> str:
    choices = [tr("diag_tlbr"), tr("diag_brtl"), tr("diag_trbl"), tr("diag_bltr")]
    mapping = {
        tr("diag_tlbr"): "tl_br",
        tr("diag_brtl"): "br_tl",
        tr("diag_trbl"): "tr_bl",
        tr("diag_bltr"): "bl_tr",
    }
    rev = {v: k for k, v in mapping.items()}
    d = 1
    if default in rev:
        d = choices.index(rev[default]) + 1
    chosen = ask_choice(tr("diag_dir"), choices, default_index=d, allow_back=True)
    if chosen == BACK_TOKEN:
        return BACK_TOKEN
    return mapping[str(chosen)]


def choose_hex_orient(default: str) -> str:
    choices = [tr("hex_orient_col"), tr("hex_orient_row")]
    d = 1
    if default in ("row-shift", "col-shift"):
        d = 2 if default == "row-shift" else 1
    chosen = ask_choice(tr("hex_orient"), choices, default_index=d, allow_back=True)
    if chosen == BACK_TOKEN:
        return BACK_TOKEN
    return "col-shift" if str(chosen) == tr("hex_orient_col") else "row-shift"


def choose_opt_extra(default_tune: bool):
    choices = [tr("opt_default"), tr("opt_tune")]
    d = 2 if default_tune else 1
    chosen = ask_choice(tr("opt_extra"), choices, default_index=d, allow_back=True)
    if chosen == BACK_TOKEN:
        return BACK_TOKEN
    return str(chosen) == tr("opt_tune")


def _get_step_keys(cfg: Dict[str, Any]) -> List[str]:
    """対話ウィザードで訊くキー一覧を、現在の cfg 状態から組み立てる。

    ポイント
    - layout により質問が変わる
    - full_shuffle=ON のとき、順序系（profile/diag_dir/最適化）は原則無効
    - grid の動画タイムライン（asc/desc）は順序固定になるため、順序系/最適化/AIも質問しない
    """

    keys: List[str] = ["layout"]

    layout = str(cfg.get("layout", "grid") or "grid").strip()

    # --- layout 固有 ---
    # NOTE: stained-glass の Lead/顔関連は「本体側デフォルト」で運用し、ランチャーでは質問しません。
    layout_steps_map: Dict[str, List[str]] = {
        "grid": ["rows", "cols"],
        "hex": ["count", "hex_orient"],
        "quilt": ["count", "quilt_split_style"],
        "stained-glass": [
            "count",
            "stained_glass_max_corner_angle",
            "stained_glass_effects_apply_mode",
        ],
        "mosaic-uniform-height": ["rows"],
        "mosaic-uniform-width": ["cols"],
        "random": [],
    }
    keys += layout_steps_map.get(layout, [])

    # --- 抽出・シャッフル ---
    keys += ["select_mode", "full_shuffle"]

    # --- アーカイブ/動画の有無 ---
    if bool(cfg.get("has_zips", False)):
        keys += ["zip_scan_enable"]
    if bool(cfg.get("has_videos", False)):
        keys += ["video_active"]

    # --- 動画フレーム抽出（任意） ---
    if bool(cfg.get("video_active", False)) and bool(cfg.get("has_videos", False)):
        # full_shuffle=ON のときは順序が崩れるので grid タイムラインは質問しない（常に off）
        if layout == "grid":
            if bool(cfg.get("full_shuffle", False)):
                cfg["grid_video_timeline"] = "off"
            else:
                keys += ["grid_video_timeline"]

        keys += ["video_mode"]
        if str(cfg.get("video_mode", "auto") or "auto") == "fixed":
            keys += ["video_frames_per_video"]
        keys += ["video_select_mode"]

    full_shuffle = bool(cfg.get("full_shuffle", False))

    # grid の動画タイムライン（asc/desc）を選んだ場合は、
    # その後の配置（グラデ/散らし）や最適化（焼きなまし等）やAIは順序を壊すので質問しない。
    if (bool(cfg.get("video_active", False)) and bool(cfg.get("has_videos", False)) and layout == "grid"):
        gvt = str(cfg.get("grid_video_timeline", "off") or "off").strip().lower()
        if gvt in ("asc", "desc"):
            return keys

    # --- 並び（profile/diag）と最適化（anneal等） ---
    if (not full_shuffle) and layout in ("grid", "hex", "mosaic-uniform-height", "mosaic-uniform-width", "quilt", "stained-glass"):
        keys += ["profile"]
        prof = str(cfg.get("profile", "diagonal") or "diagonal").strip().lower()

        # as-is: 並び替えも最適化も行わない（他モードと統一）
        if prof == "as_is":
            cfg["opt_mode"] = "default"
            cfg["opt_enable"] = False
        else:
            # diagonal のときは方向（tl_br / tr_bl / bl_tr / br_tl）も選べる
            if prof == "diagonal":
                keys += ["diag_dir"]

            # 並び/最適化（焼きなまし等）
            keys += ["opt_extra"]
            if str(cfg.get("opt_mode", "default") or "default").strip().lower() == "tune":
                keys += ["steps", "reheats"]  # steps の直後に reheats（再加熱）を質問する
                if layout.startswith("mosaic"):
                    keys += ["k"]

    # ---- 顔認識（AI）: モザイク系では使わないので質問しない ----
    if not layout.startswith("mosaic"):
        keys += ["face_ai_enable"]
        if bool(cfg.get("face_ai_enable", False)):
            keys += ["face_ai_backend"]
            _b = str(cfg.get("face_ai_backend", "yolov8_animeface")).lower().strip()
            # モデルパスは「本体側を編集」する運用のため、ランチャーでは質問しない
            keys += ["face_ai_sense"]
            if _b.startswith("yolo") or _b.startswith("yolov"):
                keys += ["face_ai_device"]

    return keys


def _recalc_derived(cfg: Dict[str, Any], core_aspect: float, *, show_mosaic_est: bool = False) -> None:
    layout = str(cfg.get("layout", "grid"))

    if layout == "grid":
        r = int(cfg.get("rows", 0) or 0)
        c = int(cfg.get("cols", 0) or 0)
        if r > 0 and c > 0:
            cfg["count"] = r * c

    elif layout == "mosaic-uniform-height":
        r = int(cfg.get("rows", 0) or 0)
        if r > 0:
            c = max(1, int(round(r * core_aspect)))
            cfg["cols"] = c
            cfg["count"] = r * c
            if show_mosaic_est:
                print(tr("mosaic_est").format(other="COLS", v=cfg["cols"], cnt=cfg["count"]))

    elif layout == "mosaic-uniform-width":
        c = int(cfg.get("cols", 0) or 0)
        if c > 0:
            r = max(1, int(round(c / core_aspect)))
            cfg["rows"] = r
            cfg["count"] = r * c
            if show_mosaic_est:
                print(tr("mosaic_est").format(other="ROWS", v=cfg["rows"], cnt=cfg["count"]))


def build_config(core_aspect: float, defaults: Optional[Dict[str, Any]] = None, *, start_at_last: bool = False, resume_key: Optional[str] = None):
    """手動設定（または編集）で設定辞書を作る。戻る対応。"""
    d = defaults or {}

    cfg: Dict[str, Any] = {}

    # 既存値を引き継ぎ（戻るで再表示するときのデフォルト用）
    for k in (
        "layout",
        "rows",
        "cols",
        "count",
        "quilt_split_style",
        "hex_orient",
        "select_mode",
        "full_shuffle",
        "face_ai_enable",
        "face_ai_backend",
        "face_ai_sense",
        "face_ai_device",
        "video_active",
        "has_videos",
        "has_zips",
        "video_mode",
        "video_frames_per_video",
        "video_select_mode",
        "grid_video_timeline",
        "profile",
        "diag_dir",
        "opt_enable",
        "opt_mode",
        "ui_style",
        "unicode_bling",
        "progress_bar_style",
        "progress_width",
        "steps",
        "reheats",
        "k",
        "random_candidates",
    ):
        if k in d:
            cfg[k] = d[k]

    # 既定値
    cfg.setdefault("layout", "grid")
    cfg.setdefault("select_mode", "random")
    cfg.setdefault("full_shuffle", False)
    cfg.setdefault("face_ai_enable", True)
    cfg.setdefault("face_ai_backend", "yolov8_animeface")
    cfg.setdefault("face_ai_model", str((_models_dir() / "yolov8x6_animeface.pt")))
    cfg.setdefault("face_ai_sense", "std")
    cfg.setdefault("face_ai_device", "auto")
    cfg.setdefault("video_mode", "auto")
    cfg.setdefault("video_frames_per_video", 2)
    cfg.setdefault("video_select_mode", "random")
    cfg.setdefault("grid_video_timeline", str(d.get("grid_video_timeline", "")))
    # 「動画タイムライン順を最後まで守る」= タイムライン整列 + 順序保持（最適化/散らし等を抑止）
    cfg.setdefault("preserve_input_order", bool(d.get("preserve_input_order", False)))
    # この簡易版では preserve_input_order は使わない（過去設定の影響を避ける）
    cfg["preserve_input_order"] = False
    cfg.setdefault("profile", "diagonal")
    cfg.setdefault("diag_dir", "tl_br")

    cfg.setdefault("opt_enable", bool(d.get("opt_enable", False)))
    cfg.setdefault("opt_mode", "tune" if bool(d.get("opt_enable", False)) else "default")
    cfg.setdefault("steps", int(d.get("steps", 20000)))
    cfg.setdefault("reheats", int(d.get("reheats", 4)))
    cfg.setdefault("k", int(d.get("k", 8)))

    # 既定値も安全側へ（UIの範囲と合わせる）
    cfg["steps"] = max(OPT_STEPS_MIN, min(OPT_STEPS_MAX, int(cfg.get("steps", OPT_STEPS_DEFAULT))))
    cfg["reheats"] = max(OPT_REHEATS_MIN, min(OPT_REHEATS_MAX, int(cfg.get("reheats", OPT_REHEATS_DEFAULT))))
    cfg["k"] = max(OPT_K_MIN, min(OPT_K_MAX, int(cfg.get("k", OPT_K_DEFAULT))))

    # 既存値からの派生値を先に整形（mosaic 推定など）
    _recalc_derived(cfg, core_aspect)
    idx = 0
    try:
        steps0 = _get_step_keys(cfg)
    except Exception:
        steps0 = []

    if resume_key and steps0:
        try:
            idx = max(0, steps0.index(resume_key))
        except ValueError:
            idx = max(0, len(steps0) - 1) if start_at_last else 0
    elif start_at_last and steps0:
        idx = max(0, len(steps0) - 1)

    while True:
        steps = _get_step_keys(cfg)
        if idx < 0:
            return BACK_TOKEN
        if idx >= len(steps):
            break

        key = steps[idx]

        if key == "layout":
            chosen = choose_layout(str(cfg.get("layout", "grid")))
            if chosen == BACK_TOKEN:
                return BACK_TOKEN
            cfg["layout"] = str(chosen)

            # random（ランダムレイアウト）の既定候補
            if cfg["layout"] == "random":
                cfg.setdefault("random_candidates", list(d.get("random_candidates", ["grid", "hex", "mosaic-uniform-height", "mosaic-uniform-width"])))

            # レイアウトが変わると後続の質問が変わるので、次のステップへ
            _recalc_derived(cfg, core_aspect)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "rows":
            dv = int(cfg.get("rows", d.get("rows", 5)))
            v = ask_int(tr("rows"), dv, 1, 2000, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["rows"] = int(v)
            _recalc_derived(cfg, core_aspect, show_mosaic_est=(cfg.get("layout") == "mosaic-uniform-height"))
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "cols":
            dv = int(cfg.get("cols", d.get("cols", 13)))
            v = ask_int(tr("cols"), dv, 1, 2000, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["cols"] = int(v)
            _recalc_derived(cfg, core_aspect, show_mosaic_est=(cfg.get("layout") == "mosaic-uniform-width"))
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "count":
            lay = str(cfg.get("layout", "hex"))
            # quilt は「タイル枚数」、stained-glass は「ピース数」、それ以外は従来どおり COUNT（枚数目安）
            if lay == "quilt":
                dv = int(cfg.get("count", d.get("count", 50)))
                label_key = "quilt_count"
            elif lay == "stained-glass":
                dv = int(cfg.get("count", d.get("count", 160)))
                label_key = "stained_glass_count"
            else:
                dv = int(cfg.get("count", d.get("count", 65)))
                label_key = "hex_count"
            v = ask_int(tr(label_key), dv, 1, 20000, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["count"] = int(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "hex_orient":
            v = choose_hex_orient(str(cfg.get("hex_orient", d.get("hex_orient", "col-shift"))))
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["hex_orient"] = str(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue


        # --- stained-glass: Lead（境界線） ---
        # NOTE: 現在はランチャーでは質問しません（本体側デフォルトで運用）。
        #       互換のために処理自体は残しています（プリセット/外部cfgで上書きする場合用）。
        if key == "stained_glass_lead_width":
            dv = int(cfg.get("stained_glass_lead_width", d.get("stained_glass_lead_width", 1)))
            v = ask_int(tr("stained_glass_lead_width"), dv, 0, 50, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["stained_glass_lead_width"] = int(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "stained_glass_lead_alpha":
            dv = float(cfg.get("stained_glass_lead_alpha", d.get("stained_glass_lead_alpha", 0.2)))
            v = ask_float(tr("stained_glass_lead_alpha"), dv, 0.0, 1.0, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["stained_glass_lead_alpha"] = float(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue
        if key == "stained_glass_max_corner_angle":
            dv = float(cfg.get("stained_glass_max_corner_angle", d.get("stained_glass_max_corner_angle", 160)))
            v = ask_float(tr("stained_glass_max_corner_angle"), dv, 30.0, 180.0, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["stained_glass_max_corner_angle"] = float(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "stained_glass_effects_apply_mode":
            cur = str(cfg.get("stained_glass_effects_apply_mode", d.get("stained_glass_effects_apply_mode", "global"))).strip().lower()
            # NOTE: core は "global" / "mask" / "mask_feather" を受け取る
            choices = [tr("stained_glass_apply_global"), tr("stained_glass_apply_mask"), tr("stained_glass_apply_mask_feather")]
            if cur in ("mask_feather", "mask-feather", "feather"):
                dv = 3
            elif cur == "mask":
                dv = 2
            else:
                dv = 1
            v = ask_choice(tr("stained_glass_effects_apply_mode"), choices, dv, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["stained_glass_effects_apply_mode"] = str(v).strip().lower()
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "stained_glass_effects_include_lead":
            mode = str(cfg.get("stained_glass_effects_apply_mode", d.get("stained_glass_effects_apply_mode", "global"))).strip().lower()
            # global のときは lead を含める/含めないの概念が薄い（全体に適用）ので、質問は省略する
            if mode == "global":
                cfg["stained_glass_effects_include_lead"] = bool(cfg.get("stained_glass_effects_include_lead", d.get("stained_glass_effects_include_lead", True)))
                cfg["_wizard_last_key"] = key
                idx += 1
                continue
            dv = bool(cfg.get("stained_glass_effects_include_lead", d.get("stained_glass_effects_include_lead", True)))
            v = ask_onoff(tr("stained_glass_effects_include_lead"), dv, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["stained_glass_effects_include_lead"] = bool(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        # --- stained-glass: 顔フォーカス（ピース選別/配置優先度） ---
        # NOTE: 現在はランチャーでは質問しません（本体側デフォルトで運用）。
        #       互換のために処理自体は残しています（プリセット/外部cfgで上書きする場合用）。
        if key == "stained_glass_face_focus_enable":
            dv = bool(cfg.get("stained_glass_face_focus_enable", d.get("stained_glass_face_focus_enable", True)))
            v = ask_onoff(tr("stained_glass_face_focus_enable"), dv, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["stained_glass_face_focus_enable"] = bool(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "stained_glass_face_priority_enable":
            # 顔フォーカスがOFFなら優先度の意味がないので質問を省略（常にFalse）
            if not bool(cfg.get("stained_glass_face_focus_enable", d.get("stained_glass_face_focus_enable", True))):
                cfg["stained_glass_face_priority_enable"] = False
                cfg["_wizard_last_key"] = key
                idx += 1
                continue
            dv = bool(cfg.get("stained_glass_face_priority_enable", d.get("stained_glass_face_priority_enable", True)))
            v = ask_onoff(tr("stained_glass_face_priority_enable"), dv, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["stained_glass_face_priority_enable"] = bool(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue
        if key == "select_mode":
            v = choose_select_mode(str(cfg.get("select_mode", d.get("select_mode", "random"))))
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["select_mode"] = str(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "full_shuffle":
            v = ask_onoff(tr("shuffle"), default_on=bool(cfg.get("full_shuffle", False)), allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["full_shuffle"] = bool(v)
            # 完全シャッフルONなら grid の動画タイムラインは常に off（質問もスキップ）
            if cfg["full_shuffle"]:
                cfg["grid_video_timeline"] = "off"
            # 完全シャッフルが ON のときは、Grid の動画タイムライン順は常に OFF（質問もしない）
            if cfg.get("layout", "grid") == "grid" and bool(cfg.get("video_active", False)) and bool(cfg.get("has_videos", False)) and bool(cfg.get("full_shuffle", False)):
                cfg["grid_video_timeline"] = "off"
            cfg["_wizard_last_key"] = key
            idx += 1
            continue
        if key == "zip_scan_enable":
            # ZIP内ファイルを読むか（ZIPが存在する場合のみ質問）
            choices = [tr("zip_yes"), tr("zip_no")]
            dflt = 1 if bool(cfg.get("zip_scan_enable", True)) else 2
            v = ask_choice(tr("zip"), choices, default_index=dflt, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["zip_scan_enable"] = (str(v) == tr("zip_yes"))
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "video_active":
            # 動画から抽出するか（動画が存在する場合のみ質問）
            choices = [tr("video_yes"), tr("video_no")]
            dflt = 1 if bool(cfg.get("video_active", True)) else 2
            v = ask_choice(tr("video"), choices, default_index=dflt, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["video_active"] = (str(v) == tr("video_yes"))
            if not bool(cfg.get("video_active", True)):
                # 動画を使わない場合は以降の動画関連質問を省略
                cfg["grid_video_timeline"] = "off"
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "grid_video_timeline":
            # grid のときだけ：動画フレームの時系列順（asc/desc/off）
            choices = [tr("grid_video_timeline_asc"), tr("grid_video_timeline_desc"), tr("grid_video_timeline_off")]
            cur = str(cfg.get("grid_video_timeline", "") or "").strip().lower()
            if cur not in ("asc", "desc", "off"):
                sm = str(cfg.get("select_mode", "random") or "random").strip().lower()
                if sm in ("name_desc", "filename_desc"):
                    cur = "desc"
                else:
                    cur = "asc"
            # 直前の選択をデフォルトにする（編集時に毎回 3 へ戻らないように）
            if cur == "asc":
                dflt = 1
            elif cur == "desc":
                dflt = 2
            else:
                dflt = 3
            v = ask_choice(tr("grid_video_timeline"), choices, default_index=dflt, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            vv = str(v)
            if vv == tr("grid_video_timeline_desc"):
                cfg["grid_video_timeline"] = "desc"
            elif vv == tr("grid_video_timeline_off"):
                cfg["grid_video_timeline"] = "off"
            else:
                cfg["grid_video_timeline"] = "asc"
            # タイムラインを優先する場合は完全シャッフルと矛盾するので自動でOFF
            if str(cfg.get("grid_video_timeline", "off")) != "off":
                cfg["full_shuffle"] = False
            # タイムライン（asc/desc）のときは、配置や最適化で順序が崩れるのを防ぐため
            # 以降の質問（配置/最適化）をスキップし、既定の安全値へ固定します。
            if str(cfg.get("grid_video_timeline", "off")) in ("asc", "desc"):
                cfg["profile"] = "as_is"
                cfg["opt_enable"] = False
                cfg["opt_mode"] = "default"
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "video_mode":
            v = choose_video_mode(str(cfg.get("video_mode", "auto")))
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["video_mode"] = str(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "video_frames_per_video":
            dv = int(cfg.get("video_frames_per_video", 2))
            v = ask_int(tr("video_frames_per_video"), default=dv, min_v=1, max_v=9999, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["video_frames_per_video"] = int(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "video_select_mode":
            v = choose_video_select_mode(str(cfg.get("video_select_mode", "random")))
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["video_select_mode"] = str(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "profile":
            v = choose_profile(str(cfg.get("profile", d.get("profile", "diagonal"))))
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["profile"] = str(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "diag_dir":
            v = choose_diag_dir(str(cfg.get("diag_dir", d.get("diag_dir", "tl_br"))))
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["diag_dir"] = str(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "opt_extra":
            default_tune = (str(cfg.get("opt_mode", "default")) == "tune")
            v = choose_opt_extra(default_tune)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            if bool(v):
                # 調整する（最適化を実行）
                cfg["opt_mode"] = "tune"
                cfg["opt_enable"] = True
            else:
                # デフォルト（高速・最適化なし）
                # プリセット由来の opt_enable を残すと「OFFを選んだのに最適化が走る」ため必ずOFFにする
                cfg["opt_mode"] = "default"
                cfg["opt_enable"] = False
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "steps":
            dv = int(cfg.get("steps", d.get("steps", 20000)))
            v = ask_int(tr("steps"), dv, OPT_STEPS_MIN, OPT_STEPS_MAX, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["steps"] = int(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "reheats":
            dv_r = int(cfg.get("reheats", d.get("reheats", 4)))
            dv_r = max(OPT_REHEATS_MIN, min(OPT_REHEATS_MAX, int(dv_r)))
            # reheats は 1〜10 の範囲で選択メニューにする（他モードと統一）
            choices_r = [str(i) for i in range(OPT_REHEATS_MIN, OPT_REHEATS_MAX + 1)]
            try:
                default_index_r = choices_r.index(str(dv_r)) + 1
            except Exception:
                default_index_r = 4
            sel = ask_choice(tr("reheats"), choices_r, default_index=default_index_r, allow_back=True)
            if sel == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["reheats"] = max(OPT_REHEATS_MIN, min(OPT_REHEATS_MAX, int(sel)))
            cfg["_wizard_last_key"] = key
            idx += 1
            continue


        if key == "k":
            dv = int(cfg.get("k", d.get("k", 8)))
            v = ask_int(tr("k"), dv, OPT_K_MIN, OPT_K_MAX, allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["k"] = int(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue


        # ------------------------------
        # 顔認識（AI）: ウィザード統合（戻る対応）
        # ------------------------------
        if key == "face_ai_enable":
            v = ask_onoff(tr("face_ai_enable"), default_on=bool(cfg.get("face_ai_enable", False)), allow_back=True)
            if v == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["face_ai_enable"] = bool(v)
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "face_ai_backend":
            if not bool(cfg.get("face_ai_enable", False)):
                idx += 1
                continue
            choices = [tr("face_ai_backend_yolo"), tr("face_ai_backend_yunet"), tr("face_ai_backend_animeface")]
            cur = str(cfg.get("face_ai_backend", "yolov8_animeface")).lower().strip()
            dflt = 1
            # NOTE: "yolov8_animeface" は名前に "animeface" を含む（YOLOモデル）ため、単純な部分一致判定はNG
            if ("yunet" in cur):
                dflt = 2
            elif cur.startswith("animeface") or ("lbpcascade" in cur) or ("cascade" in cur):
                dflt = 3
            c = ask_choice(tr("face_ai_backend"), choices, default_index=dflt, allow_back=True)
            if c == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["face_ai_backend"] = "yolov8_animeface" if (c == choices[0]) else ("yunet" if (c == choices[1]) else "animeface_cascade")

            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "face_ai_sense":
            if not bool(cfg.get("face_ai_enable", False)):
                idx += 1
                continue
            s_choices = [tr("face_ai_sense_sens"), tr("face_ai_sense_std"), tr("face_ai_sense_strict")]
            cur_s = str(cfg.get("face_ai_sense", "std")).lower().strip()
            s_dflt = 2
            if cur_s in ("sens", "sensitive"):
                s_dflt = 1
            elif cur_s in ("strict", "hard"):
                s_dflt = 3
            sc = ask_choice(tr("face_ai_sense"), s_choices, default_index=s_dflt, allow_back=True)
            if sc == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue
            cfg["face_ai_sense"] = "sens" if (sc == s_choices[0]) else ("strict" if (sc == s_choices[2]) else "std")
            cfg["_wizard_last_key"] = key
            idx += 1
            continue

        if key == "face_ai_device":
            if not bool(cfg.get("face_ai_enable", False)):
                idx += 1
                continue
            backend = str(cfg.get("face_ai_backend", "yolov8_animeface")).lower().strip()
            if not backend.startswith("yolo"):
                idx += 1
                continue

            choices = [tr("face_ai_device_auto"), tr("face_ai_device_gpu0"), tr("face_ai_device_cpu")]
            cur_d = str(cfg.get("face_ai_device", "auto")).lower().strip()
            dflt = 1
            if cur_d in ("0", "gpu0", "gpu"):
                dflt = 2
            elif cur_d == "cpu":
                dflt = 3
            sel = ask_choice(tr("face_ai_device"), choices, default_index=dflt, allow_back=True)
            if sel == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue

            if sel == choices[1]:
                cfg["face_ai_device"] = "gpu0"
            elif sel == choices[2]:
                cfg["face_ai_device"] = "cpu"
            else:
                cfg["face_ai_device"] = "auto"

            cfg["_wizard_last_key"] = key
            idx += 1
            continue


        if key == "quilt_split_style":
            # Quilt 専用: 分割位置の選び方
            # core 側の選択肢: classic/mixed/extreme/uniform
            choices = [
                tr("quilt_split_classic"),
                tr("quilt_split_mixed"),
                tr("quilt_split_extreme"),
                tr("quilt_split_uniform"),
            ]
            cur = str(cfg.get("quilt_split_style", "mixed")).lower().strip()
            dflt = 2
            if cur == "classic":
                dflt = 1
            elif cur == "extreme":
                dflt = 3
            elif cur == "uniform":
                dflt = 4

            c = ask_choice(tr("quilt_split_style"), choices, default_index=dflt, allow_back=True)
            if c == BACK_TOKEN:
                idx = max(0, idx - 1)
                continue

            if c == choices[0]:
                cfg["quilt_split_style"] = "classic"
            elif c == choices[2]:
                cfg["quilt_split_style"] = "extreme"
            elif c == choices[3]:
                cfg["quilt_split_style"] = "uniform"
            else:
                cfg["quilt_split_style"] = "mixed"

            cfg["_wizard_last_key"] = key
            idx += 1
            continue


        # 想定外のキーが来たら飛ばす
        idx += 1

    # 最終整形
    _recalc_derived(cfg, core_aspect)

    return cfg

def _profile_to_int(profile: str) -> int:
    # profile の番号（1=diagonal／2=hilbert／3=scatter）
    return {"diagonal": 1, "hilbert": 2, "scatter": 3}.get(profile, 1)


def _diag_to_int(diag: str) -> int:
    d = str(diag).strip().lower().replace("-", "_")
    d = {"tlbr": "tl_br", "brtl": "br_tl", "trbl": "tr_bl", "bltr": "bl_tr"}.get(d, d)
    return {"tl_br": 1, "br_tl": 2, "tr_bl": 3, "bl_tr": 4}.get(d, 1)


def detect_video_presence(roots: List[str], video_exts: set, recursive: bool = True, max_checks: int = 200_000) -> bool:
    """指定パス群（ファイル/フォルダ）に動画拡張子が含まれるかを軽く検出（見つけたら即True）"""
    checked = 0
    for r in roots:
        if checked >= max_checks:
            break
        p = Path(str(r))
        try:
            if p.is_file():
                checked += 1
                if p.suffix.lower() in video_exts:
                    return True
            elif p.is_dir():
                it = p.rglob("*") if recursive else p.glob("*")
                for fp in it:
                    if checked >= max_checks:
                        break
                    checked += 1
                    if fp.is_file() and fp.suffix.lower() in video_exts:
                        return True
        except Exception as e:
            _kana_silent_exc('launcher:L4076', e)
            continue
    return False


def detect_zip_presence(roots: List[str], zip_exts: set, recursive: bool = True, max_checks: int = 200_000) -> bool:
    """指定パス群（ファイル/フォルダ）にZIP拡張子が含まれるかを軽く検出（見つけたら即True）"""
    checked = 0
    for r in roots:
        if checked >= max_checks:
            break
        p = Path(str(r))
        try:
            if p.is_file():
                checked += 1
                if p.suffix.lower() in zip_exts:
                    return True
            elif p.is_dir():
                it = p.rglob('*') if recursive else p.glob('*')
                for fp in it:
                    if checked >= max_checks:
                        break
                    checked += 1
                    if fp.is_file() and fp.suffix.lower() in zip_exts:
                        return True
        except Exception as e:
            _kana_silent_exc('launcher:L4101', e)
            continue
    return False

def apply_config_to_core(mod: Any, cfg: Dict[str, Any], dd_paths: List[str]) -> None:
    """互換重視で、複数の候補変数名をセットする。"""
    layout = cfg.get("layout", "mosaic-uniform-height")
    select_mode = cfg.get("select_mode", "random")
    preserve_input_order = False  # この簡易版では使わない
    full_shuffle = bool(cfg.get("full_shuffle", False))
    # ------------------------------
    # 顔認識（AI）: coreへ反映
    # ------------------------------
    # モザイク系は顔認識（face-focus）を使わない（重い/意図しない挙動を避ける）
    if str(layout).startswith("mosaic"):
        # モザイク系: 顔フォーカス自体を無効化
        set_many(mod, ["MOSAIC_FACE_FOCUS_ENABLE"], False)
        set_many(mod, ["FACE_FOCUS_ENABLE"], False)
        set_many(mod, ["FACE_FOCUS_AI_ENABLE"], False)
        set_many(mod, ["FACE_FOCUS_AI_ALWAYS"], False)
    else:
        ai_enable = bool(cfg.get("face_ai_enable", False))
        set_many(mod, ["FACE_FOCUS_AI_ENABLE"], ai_enable)
        # ランチャーで「検出モデルを使う」を選んだら、常時AIを有効にする（本体側のゲートに合わせる）
        set_many(mod, ["FACE_FOCUS_AI_ALWAYS"], ai_enable)
        if ai_enable:
            backend = str(cfg.get("face_ai_backend", "yolov8_animeface")).strip() or "yolov8_animeface"
            set_many(mod, ["FACE_FOCUS_AI_BACKEND"], backend)
            set_many(mod, ["FACE_FOCUS_AI_MAIN"], True)
    
            backend_l = backend.lower()
            if backend_l.startswith("yolo"):

                sense = str(cfg.get("face_ai_sense", "std")).lower().strip()
                if sense == "sens":
                    set_many(mod, ["FACE_FOCUS_YOLO_CONF"], 0.10)
                    set_many(mod, ["FACE_FOCUS_YOLO_IMGSZ"], 1536)
                elif sense == "strict":
                    set_many(mod, ["FACE_FOCUS_YOLO_CONF"], 0.35)
                    set_many(mod, ["FACE_FOCUS_YOLO_IMGSZ"], 1280)
                else:
                    set_many(mod, ["FACE_FOCUS_YOLO_CONF"], 0.25)
                    set_many(mod, ["FACE_FOCUS_YOLO_IMGSZ"], 1536)

                dev = str(cfg.get("face_ai_device", "auto")).lower().strip()
                if dev in ("0", "gpu0", "gpu"):
                    set_many(mod, ["FACE_FOCUS_YOLO_DEVICE"], "0")
                elif dev == "cpu":
                    set_many(mod, ["FACE_FOCUS_YOLO_DEVICE"], "cpu")
                else:
                    set_many(mod, ["FACE_FOCUS_YOLO_DEVICE"], "")
            elif backend_l in ("animeface_cascade", "lbpcascade_animeface"):
                # AnimeFace(CPU): パスは訊かず既定XMLを使用（必要なら本体ファイル編集で変更）
                set_many(mod, ["FACE_FOCUS_AI_BACKEND"], "animeface_cascade")
                set_many(mod, ["FACE_FOCUS_ANIMEFACE_CASCADE"], str(_models_dir() / "lbpcascade_animeface.xml"))
                # 感度（しきい値）をカスケード設定へ割当
                sense = str(cfg.get("face_ai_sense", "std")).lower().strip()
                if sense == "sens":
                    set_many(mod, ["FACE_FOCUS_ANIMEFACE_SCALE_FACTOR"], 1.05)
                    set_many(mod, ["FACE_FOCUS_ANIMEFACE_MIN_NEIGHBORS"], 2)
                    set_many(mod, ["FACE_FOCUS_ANIMEFACE_MIN_SIZE"], 18)
                elif sense == "strict":
                    set_many(mod, ["FACE_FOCUS_ANIMEFACE_SCALE_FACTOR"], 1.15)
                    set_many(mod, ["FACE_FOCUS_ANIMEFACE_MIN_NEIGHBORS"], 5)
                    set_many(mod, ["FACE_FOCUS_ANIMEFACE_MIN_SIZE"], 28)
                else:
                    set_many(mod, ["FACE_FOCUS_ANIMEFACE_SCALE_FACTOR"], 1.10)
                    set_many(mod, ["FACE_FOCUS_ANIMEFACE_MIN_NEIGHBORS"], 3)
                    set_many(mod, ["FACE_FOCUS_ANIMEFACE_MIN_SIZE"], 24)
            else:
                # YuNet（実写向け）: モデルパスは本体側の既定を使う
                pass

    # シード（再現性）：固定したい場合に使用（SHUFFLE/OPT/HEX_LOCAL を連動）
    shuffle_seed = cfg.get('shuffle_seed', 'random')
    opt_seed = cfg.get('opt_seed', 'random')
    hex_local_opt_seed = cfg.get('hex_local_opt_seed', None)
    _ss = _as_int_or_none(shuffle_seed)
    _os = _as_int_or_none(opt_seed)
    _hs = _as_int_or_none(hex_local_opt_seed)
    set_many(mod, ['SHUFFLE_SEED'], int(_ss) if isinstance(_ss, int) else 'random')
    set_many(mod, ['OPT_SEED'], 'same' if isinstance(_ss, int) else 'random')
    set_many(mod, ['HEX_LOCAL_OPT_SEED'], None)


    video_mode = str(cfg.get("video_mode", "auto"))
    video_frames_per_video = int(cfg.get("video_frames_per_video", 2))
    video_select_mode = str(cfg.get("video_select_mode", "random"))

    profile = cfg.get("profile", "diagonal")
    diag_dir = cfg.get("diag_dir", "tl_br")

    # profile=as-is のときは「並べ方に手を加えない」ので、最適化（anneal等）も無効化（他モードと統一）
    if str(profile).strip().lower() == "as_is":
        cfg["opt_mode"] = "default"
        cfg["opt_enable"] = False

    # 最適化モード（opt_mode）: 新UI(default/tune/off) と旧UI(opt_enable)の両対応
    opt_enable = bool(cfg.get("opt_enable", False))
    opt_mode = str(cfg.get("opt_mode", "")).strip().lower()
    if opt_mode:
        if opt_mode == "tune":
            opt_enable = True
        elif opt_mode in ("off", "default"):
            opt_enable = False
        else:
            # 想定外の値は安全側（default）へ
            opt_mode = "default"
            opt_enable = False
    else:
        # 旧UI: opt_enable のみ
        opt_mode = "tune" if opt_enable else "default"
    steps = int(cfg.get("steps", OPT_STEPS_DEFAULT))
    reheats = int(cfg.get("reheats", OPT_REHEATS_DEFAULT))
    k = int(cfg.get("k", OPT_K_DEFAULT))

    # 安全のため、最適化パラメータはここで一度クランプして統一します。
    steps = max(OPT_STEPS_MIN, min(OPT_STEPS_MAX, int(steps)))
    reheats = max(OPT_REHEATS_MIN, min(OPT_REHEATS_MAX, int(reheats)))
    k = max(OPT_K_MIN, min(OPT_K_MAX, int(k)))

    # cfg へも戻しておく（以降の処理／表示の一貫性のため）
    cfg["steps"] = int(steps)
    cfg["reheats"] = int(reheats)
    cfg["k"] = int(k)

    # 本体側の表示言語もランチャーに合わせる
    try:
        set_many(mod, ["UI_LANG"], str(DEFAULT_LANG))
    except Exception as e:
        _kana_silent_exc('launcher:L4202', e)
        pass
    # エフェクト（ランチャー側で上書きできる）
    try:
        eff = _extract_effect_cfg_from_any(cfg) if isinstance(cfg, dict) else {}
        if isinstance(eff, dict) and eff:
            eff = _normalize_effect_cfg(eff)
            set_many(mod, ["EFFECTS_ENABLE"], bool(eff.get('effects_enable', True)))
            set_many(mod, ["HALATION_ENABLE"], bool(eff.get('halation_enable', False)))
            set_many(mod, ["HALATION_INTENSITY"], float(eff.get('halation_intensity', 0.30)))
            set_many(mod, ["HALATION_RADIUS"], int(eff.get('halation_radius', 18)))
            set_many(mod, ["HALATION_THRESHOLD"], float(eff.get('halation_threshold', 0.70)))
            set_many(mod, ["HALATION_KNEE"], float(eff.get('halation_knee', 0.08)))
            set_many(mod, ["BW_EFFECT_ENABLE"], bool(eff.get('bw_effect_enable', False)))
            set_many(mod, ["SEPIA_ENABLE"], bool(eff.get('sepia_enable', True)))
            set_many(mod, ["SEPIA_INTENSITY"], float(eff.get('sepia_intensity', 0.03)))
            set_many(mod, ["GRAIN_ENABLE"], bool(eff.get('grain_enable', True)))
            set_many(mod, ["GRAIN_AMOUNT"], float(eff.get('grain_amount', 0.15)))
            set_many(mod, ["CLARITY_ENABLE"], bool(eff.get('clarity_enable', False)))
            set_many(mod, ["CLARITY_AMOUNT"], float(eff.get('clarity_amount', 0.12)))
            set_many(mod, ["CLARITY_RADIUS"], float(eff.get('clarity_radius', 2.0)))
            set_many(mod, ["UNSHARP_ENABLE"], bool(eff.get('unsharp_enable', False)))
            set_many(mod, ["UNSHARP_AMOUNT"], float(eff.get('unsharp_amount', 0.35)))
            set_many(mod, ["UNSHARP_RADIUS"], float(eff.get('unsharp_radius', 1.2)))
            set_many(mod, ["UNSHARP_THRESHOLD"], int(eff.get('unsharp_threshold', 3)))
            set_many(mod, ["DENOISE_MODE"], str(eff.get('denoise_mode', 'off')))
            set_many(mod, ["DENOISE_STRENGTH"], float(eff.get('denoise_strength', 0.25)))
            set_many(mod, ["DEHAZE_ENABLE"], bool(eff.get('dehaze_enable', False)))
            set_many(mod, ["DEHAZE_AMOUNT"], float(eff.get('dehaze_amount', 0.10)))
            set_many(mod, ["DEHAZE_RADIUS"], int(eff.get('dehaze_radius', 24)))
            set_many(mod, ["SHADOWHIGHLIGHT_ENABLE"], bool(eff.get('shadowhighlight_enable', False)))
            set_many(mod, ["SHADOW_AMOUNT"], float(eff.get('shadow_amount', 0.22)))
            set_many(mod, ["HIGHLIGHT_AMOUNT"], float(eff.get('highlight_amount', 0.18)))
            set_many(mod, ["TONECURVE_ENABLE"], bool(eff.get('tonecurve_enable', False)))
            set_many(mod, ["TONECURVE_MODE"], str(eff.get('tonecurve_mode', 'film')))
            set_many(mod, ["TONECURVE_STRENGTH"], float(eff.get('tonecurve_strength', 0.35)))
            set_many(mod, ["LUT_ENABLE"], bool(eff.get('lut_enable', False)))
            set_many(mod, ["LUT_FILE"], str(eff.get('lut_file', '')))
            set_many(mod, ["LUT_STRENGTH"], float(eff.get('lut_strength', 0.30)))
            set_many(mod, ["VIBRANCE_ENABLE"], bool(eff.get('vibrance_enable', False)))
            set_many(mod, ["VIBRANCE_FACTOR"], float(eff.get('vibrance_factor', 1.0)))
            set_many(mod, ["SPLIT_TONE_ENABLE"], bool(eff.get('split_tone_enable', False)))
            set_many(mod, ["SPLIT_TONE_SHADOW_HUE"], float(eff.get('split_tone_shadow_hue', 220.0)))
            set_many(mod, ["SPLIT_TONE_SHADOW_STRENGTH"], float(eff.get('split_tone_shadow_strength', 0.06)))
            set_many(mod, ["SPLIT_TONE_HIGHLIGHT_HUE"], float(eff.get('split_tone_highlight_hue', 35.0)))
            set_many(mod, ["SPLIT_TONE_HIGHLIGHT_STRENGTH"], float(eff.get('split_tone_highlight_strength', 0.05)))
            set_many(mod, ["SPLIT_TONE_BALANCE"], float(eff.get('split_tone_balance', 0.0)))
            set_many(mod, ["VIGNETTE_ENABLE"], bool(eff.get('vignette_enable', False)))
            set_many(mod, ["VIGNETTE_STRENGTH"], float(eff.get('vignette_strength', 0.15)))
            set_many(mod, ["VIGNETTE_ROUND"], float(eff.get('vignette_round', 0.50)))
            set_many(mod, ["BRIGHTNESS_MODE"], str(eff.get('brightness_mode', 'off')))
            set_many(mod, ["AUTO_METHOD"], str(eff.get('auto_method', 'hybrid')))
            set_many(mod, ["AUTO_TARGET_MEAN"], float(eff.get('auto_target_mean', 0.50)))
            set_many(mod, ["MANUAL_GAIN"], float(eff.get('manual_gain', 1.00)))
            set_many(mod, ["MANUAL_GAMMA"], float(eff.get('manual_gamma', 1.00)))
    except Exception as e:
        _kana_silent_exc('launcher:L4259', e)
        pass
    # ドラッグ＆ドロップ入力を本体へ渡す（互換のため複数名）
    if dd_paths:
        set_many(mod, ["SCAN_ROOTS", "SCAN_DIRS", "INPUT_PATHS", "INPUT_ROOTS", "DEFAULT_FOLDERS"], dd_paths)

    # 共通
    set_many(mod, ["LAYOUT_STYLE"], layout)
    set_many(mod, ["SELECT_MODE"], select_mode)
    set_many(mod, ["ARRANGE_FULL_SHUFFLE"], full_shuffle)
    # 順序保持（動画タイムライン順などを最後まで守る）
    set_many(mod, ["PRESERVE_INPUT_ORDER"], preserve_input_order)
    # grid/mosaic の配置順（core 側が参照しない版でも害はありません）
    set_many(mod, ["PLACEMENT_ORDER"], "row_major")
    # UI（表示）: ランチャー側で明示された場合のみ上書き
    if isinstance(cfg, dict):
        if cfg.get("ui_style") is not None:
            set_many(mod, ["UI_STYLE"], str(cfg.get("ui_style")))
        if cfg.get("unicode_bling") is not None:
            set_many(mod, ["UNICODE_BLING"], bool(cfg.get("unicode_bling")))
        if cfg.get("progress_bar_style") is not None:
            set_many(mod, ["PROGRESS_BAR_STYLE"], str(cfg.get("progress_bar_style")))
        if cfg.get("progress_width") is not None:
            try:
                set_many(mod, ["PROGRESS_WIDTH"], int(cfg.get("progress_width")))
            except Exception as e:
                _kana_silent_exc('launcher:L4285', e)
                pass
        # 副産物（使用画像リスト/メタなど）の保存（None の場合は core の設定を上書きしない）
    if LAUNCHER_SAVE_ARTIFACTS is not None:
        try:
            set_many(mod, ["SAVE_ARTIFACTS"], bool(LAUNCHER_SAVE_ARTIFACTS))
        except Exception:
            pass

# ZIP_SCAN_ENABLE / VIDEO_SCAN_ENABLE はランチャーで切り替え可能
    zip_enable = bool(cfg.get("zip_scan_enable",
                     bool(getattr(mod, "ZIP_SCAN_ENABLE", False))
                     or bool(getattr(mod, "SEVENZ_SCAN_ENABLE", False))
                     or bool(getattr(mod, "RAR_SCAN_ENABLE", False))))
    # アーカイブ全般（zip/7z/rar）を同一トグルで扱う（未対応の変数は無視）
    set_many(mod, ["ZIP_SCAN_ENABLE", "ZIP_SCAN"], zip_enable)
    if hasattr(mod, "SEVENZ_SCAN_ENABLE"):
        set_many(mod, ["SEVENZ_SCAN_ENABLE"], zip_enable)
    if hasattr(mod, "RAR_SCAN_ENABLE"):
        set_many(mod, ["RAR_SCAN_ENABLE"], zip_enable)
    video_active = bool(cfg.get("video_active", bool(getattr(mod, "VIDEO_SCAN_ENABLE", False))))
    set_many(mod, ["VIDEO_SCAN_ENABLE", "VIDEO_ENABLE"], video_active)
    if not video_active:
        set_many(mod, ["GRID_VIDEO_TIMELINE"], "off")
    has_videos = bool(cfg.get("has_videos", False))
    if video_active and has_videos:

        # fixed: 指定枚数 / auto: 0（本体側で必要枚数から自動配分）
        vpp = video_frames_per_video if video_mode == "fixed" else 0
        set_many(mod, ["VIDEO_FRAMES_PER_VIDEO"], vpp)
        set_many(mod, ["VIDEO_FRAME_SELECT_MODE"], video_select_mode)
        # grid のときだけ：動画フレームの時系列順（asc/desc/off）
        if str(layout) == "grid":
            gvt = str(cfg.get("grid_video_timeline", "asc") or "asc").strip().lower()
            if gvt not in ("asc", "desc", "off"):
                gvt = "asc"
            if bool(full_shuffle):
                gvt = "off"
            set_many(mod, ["GRID_VIDEO_TIMELINE"], gvt)
            # If grid video timeline is active, we must preserve input order strictly.
            # Otherwise, grid optimizer/tempo can reorder the sequence and break chronology.
            if gvt in ("asc", "desc"):
                # Force as-is arrangement and disable optimization in strict timeline mode
                profile = "as_is"
                opt_enable = False
                full_shuffle = False
                preserve_input_order = True
                set_many(mod, ["PRESERVE_INPUT_ORDER"], True)
                set_many(mod, ["ARRANGE_FULL_SHUFFLE"], False)
                # Disable grid neighbor optimization explicitly (core default may be max).
                set_many(mod, ["GRID_NEIGHBOR_OBJECTIVE", "GRID_OBJECTIVE"], "off")
                set_many(mod, ["GRID_OPTIMIZER"], "none")
                # Also disable tempo reordering (if supported in core).
                set_many(mod, ["ARRANGE_TEMPO_ENABLE"], False)

    # サイズ
    if layout == "grid":
        r = int(cfg.get("rows", 5))
        c = int(cfg.get("cols", 13))
        set_many(mod, ["ROWS"], r)
        set_many(mod, ["COLS"], c)
        set_many(mod, ["COUNT"], int(cfg.get("count", r * c)))
    elif layout == "hex":
        set_many(mod, ["COUNT"], int(cfg.get("count", 65)))
        if cfg.get("hex_orient"):
            set_many(mod, ["HEX_TIGHT_ORIENT"], str(cfg.get("hex_orient")))
    elif layout in ("mosaic-uniform-height", "mosaic-uniform-width"):
        r = int(cfg.get("rows", 5))
        c = int(cfg.get("cols", 13))
        set_many(mod, ["ROWS"], r)
        set_many(mod, ["COLS"], c)
        # mosaic は COUNT=ユニーク枚数として扱う版が多い
        set_many(mod, ["COUNT"], int(cfg.get("count", r * c)))
        set_many(mod, ["MOSAIC_USE_ROWS_COLS", "MOSAIC_FIXED_ROWS_COLS"], True)
    elif layout == "quilt":
        # quilt は「タイル枚数」を COUNT として扱います
        qn = int(cfg.get("count", 50))
        set_many(mod, ["COUNT"], qn)
        # quilt 独自の上限（存在しない core でも害はありません）
        set_many(mod, ["QUILT_MAX_TILES"], qn)
        # quilt: 分割スタイル（classic/mixed/extreme/uniform）
        set_many(mod, ["QUILT_SPLIT_STYLE"], str(cfg.get("quilt_split_style", "mixed")))
        # quilt はタイルごとに face-focus を効かせる（fill のときのみ）
        set_many(mod, ["QUILT_FACE_FOCUS_ENABLE"], True)
    elif layout == "stained-glass":
        # stained-glass は「ピース数」を COUNT として扱います
        n = int(cfg.get("count", 160))
        if n < 1:
            n = 1
        set_many(mod, ["COUNT"], n)

        set_many(mod, ["STAINED_GLASS_LEAD_WIDTH"], int(cfg.get("stained_glass_lead_width", 1)))
        set_many(mod, ["STAINED_GLASS_LEAD_ALPHA"], float(cfg.get("stained_glass_lead_alpha", 0.2)))
        # 品質（境界）: core 側の既定を launcher で揃える（必要なら presets/external config で上書き可）
        set_many(mod, ["STAINED_GLASS_LEAD_METHOD"], str(cfg.get("stained_glass_lead_method", "edges")))
        try:
            _ss = int(cfg.get("stained_glass_mask_supersample", 4))
        except Exception:
            _ss = 4
        _ss = max(1, min(4, _ss))
        set_many(mod, ["STAINED_GLASS_MASK_SUPERSAMPLE"], int(_ss))
        set_many(mod, ["STAINED_GLASS_MAX_CORNER_ANGLE_DEG"], float(cfg.get("stained_glass_max_corner_angle", 160)))

        mode = str(cfg.get("stained_glass_effects_apply_mode", "global")).strip().lower()
        if mode not in ("global", "mask"):
            mode = "global"
        set_many(mod, ["STAINED_GLASS_EFFECTS_APPLY_MODE"], mode)

        inc = bool(cfg.get("stained_glass_effects_include_lead", True))
        # NOTE: include_lead は "mask" モード時のみ有効（global のときは false で固定）
        set_many(mod, ["STAINED_GLASS_EFFECTS_INCLUDE_LEAD"], bool(inc) if mode == "mask" else False)

        sg_face_focus_enable = bool(cfg.get("stained_glass_face_focus_enable", True))
        sg_face_priority_enable = bool(cfg.get("stained_glass_face_priority_enable", True))
        set_many(mod, ["STAINED_GLASS_FACE_FOCUS_ENABLE"], bool(sg_face_focus_enable))
        set_many(mod, ["STAINED_GLASS_FACE_PRIORITY_ENABLE"], bool(sg_face_priority_enable))
        # stained-glass: facefit（パラメータは core の既定値を尊重）
        # - core 単体起動でも同じ挙動になるよう、ランチャからは「明示キーがある場合のみ」上書きします。
        # - 設定したい場合は、プリセット/外部JSON に stained_glass_facefit_* を追加してください。
        if bool(sg_face_focus_enable):
            if 'stained_glass_facefit_enable' in cfg:
                set_many(mod, ['STAINED_GLASS_FACE_FIT_ENABLE'], bool(cfg.get('stained_glass_facefit_enable')))

            if 'stained_glass_facefit_max_tries' in cfg:
                set_many(mod, ['STAINED_GLASS_FACE_FIT_MAX_TRIES'], int(cfg.get('stained_glass_facefit_max_tries')))

            if 'stained_glass_facefit_safe_margin_px' in cfg:
                set_many(mod, ['STAINED_GLASS_FACE_FIT_SAFE_MARGIN_PX'], int(cfg.get('stained_glass_facefit_safe_margin_px')))

            if 'stained_glass_facefit_eye_y_frac' in cfg:
                set_many(mod, ['STAINED_GLASS_FACE_FIT_EYE_Y_FRAC'], float(cfg.get('stained_glass_facefit_eye_y_frac')))

            if 'stained_glass_facefit_eye_y_alt_delta' in cfg:
                set_many(mod, ['STAINED_GLASS_FACE_FIT_EYE_Y_ALT_DELTA'], float(cfg.get('stained_glass_facefit_eye_y_alt_delta')))

            if 'stained_glass_facefit_spread_x_frac' in cfg:
                set_many(mod, ['STAINED_GLASS_FACE_FIT_EYE_SPREAD_X_FRAC'], float(cfg.get('stained_glass_facefit_spread_x_frac')))

            if 'stained_glass_facefit_spread_y_frac' in cfg:
                set_many(mod, ['STAINED_GLASS_FACE_FIT_EYE_SPREAD_Y_FRAC'], float(cfg.get('stained_glass_facefit_spread_y_frac')))

            if 'stained_glass_facefit_spread_scale' in cfg:
                set_many(mod, ['STAINED_GLASS_FACE_FIT_EYE_SPREAD_SCALE'], float(cfg.get('stained_glass_facefit_spread_scale')))

            if 'stained_glass_facefit_ok_ratio' in cfg:
                set_many(mod, ['STAINED_GLASS_FACE_FIT_POINT_OK_RATIO'], float(cfg.get('stained_glass_facefit_ok_ratio')))

            if 'stained_glass_facefit_strict_eye_band' in cfg:
                set_many(mod, ['STAINED_GLASS_FACE_FIT_STRICT_EYE_BAND'], bool(cfg.get('stained_glass_facefit_strict_eye_band')))

            if 'stained_glass_facefit_thin_mode' in cfg:
                set_many(mod, ['STAINED_GLASS_FACE_FIT_THIN_MODE'], str(cfg.get('stained_glass_facefit_thin_mode')))


    else:
        # random（ランダムレイアウト）
        cand = cfg.get("random_candidates")
        if isinstance(cand, list) and cand:
            set_many(mod, ["RANDOM_LAYOUT_CANDIDATES", "RANDOM_LAYOUTS"], cand)

    # 順序保持モード：順序を壊しやすい後処理（グラデ/散らし/最適化）を無効化
    if preserve_input_order:
        set_many(mod, ["ARRANGE_FULL_SHUFFLE"], False)
        set_many(mod, ["MOSAIC_ENHANCE_ENABLE", "GRID_ENHANCE_ENABLE", "HEX_ENHANCE_ENABLE", "QUILT_ENHANCE_ENABLE"], False)
        set_many(mod, ["MOSAIC_LOCAL_OPT_ENABLE", "GRID_ANNEAL_ENABLE", "HEX_ANNEAL_ENABLE", "HEX_LOCAL_OPT_ENABLE", "QUILT_ANNEAL_ENABLE"], False)
        # hex の global/local 最適化も無効化（順序保持）
        set_many(mod, ["HEX_GLOBAL_ORDER"], "none")
        set_many(mod, ["HEX_GLOBAL_OBJECTIVE"], "min")
        set_many(mod, ["HEX_LOCAL_OPT_ENABLE"], False)
        # quilt 最適化も明示的に無効化（順序保持）
        set_many(mod, ["QUILT_OPTIMIZER"], "none")
        return

    # フルシャッフルONなら配置・最適化はOFF方向へ
    if full_shuffle:
        # 「完全ランダム」は ARRANGE_FULL_SHUFFLE で担保するため、
        # 各レイアウトの後処理／最適化（グラデーション／散らし／焼きなまし等）は明示的に無効化します。
        set_many(mod, ["MOSAIC_ENHANCE_ENABLE", "GRID_ENHANCE_ENABLE", "HEX_ENHANCE_ENABLE", "QUILT_ENHANCE_ENABLE"], False)
        set_many(mod, ["MOSAIC_LOCAL_OPT_ENABLE", "GRID_ANNEAL_ENABLE", "HEX_ANNEAL_ENABLE", "QUILT_ANNEAL_ENABLE"], False)
        # core(v5+) では hex が mosaic の順序を継承しないよう明示（フルシャッフル尊重）
        set_many(mod, ["HEX_GLOBAL_ORDER"], "none")
        set_many(mod, ["HEX_GLOBAL_OBJECTIVE"], "min")
        set_many(mod, ["HEX_LOCAL_OPT_ENABLE"], False)
        set_many(mod, ["QUILT_OPTIMIZER"], "none")
        return

    # 「そのまま（ファイル名順を保つ）」：後処理（グラデ/散らし/最適化）を無効化して順序を守る
    if profile == "as_is":
        set_many(mod, ["MOSAIC_ENHANCE_ENABLE", "GRID_ENHANCE_ENABLE", "HEX_ENHANCE_ENABLE", "QUILT_ENHANCE_ENABLE"], False)
        set_many(mod, ["MOSAIC_LOCAL_OPT_ENABLE", "GRID_ANNEAL_ENABLE", "HEX_ANNEAL_ENABLE", "HEX_LOCAL_OPT_ENABLE", "QUILT_ANNEAL_ENABLE"], False)
        set_many(mod, ["HEX_GLOBAL_ORDER"], "none")
        set_many(mod, ["HEX_GLOBAL_OBJECTIVE"], "min")
        # Prevent core default grid optimizer from reordering even in 'as_is'
        set_many(mod, ["GRID_NEIGHBOR_OBJECTIVE", "GRID_OBJECTIVE"], "off")
        set_many(mod, ["GRID_OPTIMIZER"], "none")
        set_many(mod, ["ARRANGE_TEMPO_ENABLE"], False)
        set_many(mod, ["QUILT_OPTIMIZER"], "none")
        return

    p_int = _profile_to_int(str(profile))
    d_int = _diag_to_int(str(diag_dir))

    # 目的（scatter は maximize／gradient は minimize を基本）
    neighbor_obj = "min"
    if profile == "scatter":
        neighbor_obj = "max"

    # レイアウト別：配置
    if layout.startswith("mosaic"):
        set_many(mod, ["MOSAIC_PROFILE"], p_int)
        set_many(mod, ["MOSAIC_DIAG_DIR", "MOSAIC_DIAGONAL_DIR"], d_int)
        set_many(mod, ["MOSAIC_DIAGONAL_DIRECTION"], str(diag_dir))
        set_many(mod, ["MOSAIC_ENHANCE_ENABLE"], True)
        set_many(mod, ["MOSAIC_ENHANCE_PROFILE"], str(profile))
        set_many(mod, ["MOSAIC_NEIGHBOR_OBJECTIVE", "MOSAIC_OBJECTIVE"], neighbor_obj)
    elif layout == "grid":
        set_many(mod, ["GRID_PROFILE"], p_int)
        set_many(mod, ["GRID_DIAG_DIR", "GRID_DIAGONAL_DIR"], d_int)
        set_many(mod, ["GRID_DIAGONAL_DIRECTION"], str(diag_dir))
        set_many(mod, ["GRID_ENHANCE_ENABLE"], True)
        set_many(mod, ["GRID_ENHANCE_PROFILE"], str(profile))
        set_many(mod, ["GRID_NEIGHBOR_OBJECTIVE", "GRID_OBJECTIVE"], neighbor_obj)
        # grid: 配置プロファイル → optimizer への対応
        #  - diagonal: spectral-diagonal（対角グラデ）
        #  - hilbert : spectral-hilbert（ヒルベルト）
        #  - scatter : checkerboard 風（散らし）
        if profile == "scatter":
            set_many(mod, ["GRID_OPTIMIZER"], "checkerboard")
        elif profile == "diagonal":
            set_many(mod, ["GRID_OPTIMIZER"], "spectral-diagonal")
        else:
            set_many(mod, ["GRID_OPTIMIZER"], "spectral-hilbert")
    elif layout == "quilt":
        # quilt: 位置の並び（diagonal/hilbert/scatter/as-is）は core 側で rect の順序に適用します。
        #       ここではプロファイルと目的（min/max）だけを渡します。
        set_many(mod, ["QUILT_ENHANCE_ENABLE"], True)
        set_many(mod, ["QUILT_ENHANCE_PROFILE"], str(profile))
        set_many(mod, ["QUILT_NEIGHBOR_OBJECTIVE", "QUILT_OBJECTIVE"], neighbor_obj)
        set_many(mod, ["QUILT_DIAGONAL_DIRECTION", "QUILT_DIAG_DIR"], str(diag_dir))
    elif layout == "stained-glass":
        # stained-glass: 他レイアウトと同じ profile で「並び」を決める（core 側は order_* で実装）
        #   diagonal -> grad（対角） / hilbert -> grad（ヒルベルト） / scatter -> scatter（散らし） / as-is -> OFF
        prof = str(profile).strip().lower()

        # 既定: profile が as-is でないなら並びを有効化
        ord_enable = cfg.get("stained_glass_order_enable", None)
        if ord_enable is None:
            ord_enable = (prof != "as_is")

        set_many(mod, ["STAINED_GLASS_ORDER_ENABLE"], bool(ord_enable))
        if bool(ord_enable):
            mode = "grad"
            panel_order = "hilbert"
            img_order = "spectral_hilbert"
            obj = "min"

            if prof == "scatter":
                mode = "scatter"
                panel_order = "checker"
                img_order = "spectral_hilbert"
                obj = "max"
            elif prof == "diagonal":
                mode = "grad"
                panel_order = "diag"
                img_order = "spectral_diagonal"
                obj = "min"
            elif prof == "hilbert":
                mode = "grad"
                panel_order = "hilbert"
                img_order = "spectral_hilbert"
                obj = "min"

            # direction（diagonal のときに使用）
            set_many(mod, ["STAINED_GLASS_DIAG_DIR", "STAINED_GLASS_DIAGONAL_DIRECTION", "STAINED_GLASS_PANEL_DIAG_DIR"], str(diag_dir))

            # opt_mode=tune のときは、他レイアウト同様に anneal を選べる（画像の並びに適用）
            _img_override = cfg.get("stained_glass_image_order", None)
            if opt_mode == "tune" and _img_override is None:
                img_order = "anneal"
                # steps / reheats は共通 UI の値を流用
                try:
                    sg_steps = int(steps) if steps is not None else 20000
                except Exception:
                    sg_steps = 20000
                sg_steps = max(OPT_STEPS_MIN, min(OPT_STEPS_MAX, sg_steps))

                try:
                    sg_reheats = int(reheats) if reheats is not None else 4
                except Exception:
                    sg_reheats = 4
                sg_reheats = max(OPT_REHEATS_MIN, min(OPT_REHEATS_MAX, sg_reheats))

                set_many(mod, ["STAINED_GLASS_ORDER_ANNEAL_STEPS", "STAINED_GLASS_ANNEAL_STEPS"], sg_steps)
                set_many(mod, ["STAINED_GLASS_ORDER_ANNEAL_REHEATS", "STAINED_GLASS_ANNEAL_REHEATS"], sg_reheats)

            # 明示指定（あれば優先）
            mode = str(cfg.get("stained_glass_order_mode", mode) or mode).strip().lower()
            panel_order = str(cfg.get("stained_glass_panel_order", panel_order) or panel_order).strip().lower()
            img_order = str(cfg.get("stained_glass_image_order", img_order) or img_order).strip().lower()
            obj = str(cfg.get("stained_glass_order_objective", obj) or obj).strip().lower()

            set_many(mod, ["STAINED_GLASS_ORDER_MODE"], mode)
            set_many(mod, ["STAINED_GLASS_PANEL_ORDER"], panel_order)
            set_many(mod, ["STAINED_GLASS_IMAGE_ORDER"], img_order)
            set_many(mod, ["STAINED_GLASS_ORDER_OBJECTIVE"], obj)

            # ヒルベルト bits（必要なら外部設定で調整可）
            try:
                hb = int(cfg.get("stained_glass_order_hilbert_bits", 6))
            except Exception:
                hb = 6
            hb = max(3, min(10, hb))
            set_many(mod, ["STAINED_GLASS_ORDER_HILBERT_BITS"], hb)

            # 並びを優先するため、顔の優先度（重み付け）をオフにするのが既定
            disable_fp = cfg.get("stained_glass_order_disable_face_priority", None)
            if disable_fp is None:
                disable_fp = True
            set_many(mod, ["STAINED_GLASS_ORDER_DISABLE_FACE_PRIORITY"], bool(disable_fp))

    elif layout == "hex":
        set_many(mod, ["HEX_GRAD_PROFILE", "HEX_PROFILE"], p_int)
        set_many(mod, ["HEX_DIAGONAL_DIR"], d_int)
        set_many(mod, ["HEX_DIAG_DIR", "HEX_DIAGONAL_DIRECTION"], str(diag_dir))
        set_many(mod, ["HEX_ENHANCE_ENABLE"], True)
        set_many(mod, ["HEX_ENHANCE_PROFILE"], str(profile))
        set_many(mod, ["HEX_NEIGHBOR_OBJECTIVE", "HEX_OBJECTIVE"], neighbor_obj)

        # core(v5+) 用: hex の global/local 最適化を明示（scatter は max）
        #  - HEX_GLOBAL_ORDER: inherit／none／spectral-hilbert／spectral-diagonal／checkerboard／anneal
        #  - HEX_GLOBAL_OBJECTIVE: min／max
        #  - HEX_LOCAL_OPT_OBJECTIVE: min／max
        if profile == "scatter":
            set_many(mod, ["HEX_GLOBAL_ORDER"], "checkerboard")
            set_many(mod, ["HEX_GLOBAL_OBJECTIVE"], "max")
            set_many(mod, ["HEX_LOCAL_OPT_OBJECTIVE"], "max")
        elif profile == "diagonal":
            set_many(mod, ["HEX_GLOBAL_ORDER"], "spectral-diagonal")
            set_many(mod, ["HEX_GLOBAL_OBJECTIVE"], "min")
            set_many(mod, ["HEX_LOCAL_OPT_OBJECTIVE"], "min")
        elif profile == "hilbert":
            set_many(mod, ["HEX_GLOBAL_ORDER"], "spectral-hilbert")
            set_many(mod, ["HEX_GLOBAL_OBJECTIVE"], "min")
            set_many(mod, ["HEX_LOCAL_OPT_OBJECTIVE"], "min")
        else:
            set_many(mod, ["HEX_GLOBAL_ORDER"], "none")
            set_many(mod, ["HEX_GLOBAL_OBJECTIVE"], "min")
            set_many(mod, ["HEX_LOCAL_OPT_OBJECTIVE"], "min")

    # 最適化（任意）
    if opt_enable:
        if layout.startswith("mosaic"):
            set_many(mod, ["MOSAIC_LOCAL_OPT_ENABLE", "MOSAIC_ANNEAL_ENABLE", "MOSAIC_OPT_ENABLE"], True)
            set_many(mod, ["MOSAIC_LOCAL_OPT_STEPS", "MOSAIC_ANNEAL_STEPS", "MOSAIC_STEPS"], steps)
            set_many(mod, ["MOSAIC_LOCAL_OPT_REHEATS", "MOSAIC_ANNEAL_REHEATS", "MOSAIC_REHEATS"], reheats)
            set_many(mod, ["MOSAIC_LOCAL_OPT_K", "MOSAIC_K", "MOSAIC_NEIGHBOR_K"], k)
        elif layout == "grid":
            set_many(mod, ["GRID_ANNEAL_ENABLE", "GRID_OPT_ENABLE"], True)
            set_many(mod, ["GRID_ANNEAL_STEPS", "GRID_STEPS"], steps)
            set_many(mod, ["GRID_ANNEAL_REHEATS", "GRID_REHEATS"], reheats)
        elif layout == "hex":
            set_many(mod, ["HEX_ANNEAL_ENABLE", "HEX_OPT_ENABLE"], True)
            set_many(mod, ["HEX_ANNEAL_STEPS", "HEX_STEPS"], steps)
            set_many(mod, ["HEX_ANNEAL_REHEATS", "HEX_REHEATS"], reheats)
            # core(v5+) 用（ローカル最適化）
            set_many(mod, ["HEX_LOCAL_OPT_ENABLE"], True)
            set_many(mod, ["HEX_LOCAL_OPT_STEPS"], steps)
            set_many(mod, ["HEX_LOCAL_OPT_REHEATS"], reheats)

        elif layout == "quilt":
            # quilt: 近傍の焼きなまし（矩形タイルの隣接グラフで swap 最適化）
            set_many(mod, ["QUILT_ANNEAL_ENABLE", "QUILT_OPT_ENABLE"], True)
            set_many(mod, ["QUILT_ANNEAL_STEPS", "QUILT_STEPS"], steps)
            set_many(mod, ["QUILT_ANNEAL_REHEATS", "QUILT_REHEATS"], reheats)
            set_many(mod, ["QUILT_OPTIMIZER"], "anneal")

    else:
        # opt_enable=False（デフォルト/最適化なし）の場合でも、
        # core 側の既定値が True だと最適化が走ってしまうため、ここで明示的にOFFにします。
        # layout=random の場合でも安全にするため、mosaic/grid/hex をまとめてOFFにします。
        set_many(mod, ["MOSAIC_LOCAL_OPT_ENABLE", "MOSAIC_ANNEAL_ENABLE", "MOSAIC_OPT_ENABLE"], False)
        set_many(mod, ["GRID_ANNEAL_ENABLE", "GRID_OPT_ENABLE"], False)
        set_many(mod, ["HEX_ANNEAL_ENABLE", "HEX_OPT_ENABLE", "HEX_LOCAL_OPT_ENABLE"], False)
        set_many(mod, ["QUILT_ANNEAL_ENABLE", "QUILT_OPT_ENABLE"], False)
        set_many(mod, ["QUILT_OPTIMIZER"], "none")


# =============================================================================
# Run core
# =============================================================================


def run_core(mod: Any) -> None:
    """Call the core entry point."""
    if hasattr(mod, "main") and callable(getattr(mod, "main")):
        mod.main()
        # Optional: cleanup video frames cache at end, if core provides helper.
        try:
            if bool(getattr(mod, "VIDEO_FRAME_CACHE_CLEAR_ON_END", False)) and hasattr(mod, "_video_frames_cache_cleanup_on_exit"):
                fn = getattr(mod, "_video_frames_cache_cleanup_on_exit")
                if callable(fn):
                    fn()
        except Exception as e:
            _kana_silent_exc('launcher:L4555', e)
            pass
        return

    if hasattr(mod, "run") and callable(getattr(mod, "run")):
        mod.run()
        return

    raise RuntimeError(tr("err_no_entry"))


def run_core_repeat(mod: Any, *, count: int, apply_wallpaper: bool) -> None:
    """連続で壁紙画像を生成する（連続出力用途）。

    - count: 1以上 = 指定回数 / 0 = Ctrl+C で止めるまで無限
    - apply_wallpaper: True の場合は毎回壁紙へ反映（最終的には最後の1枚が壁紙になります）
    """
    # 連続出力では「保存してコレクション」したいケースが多いので、念のため強制ON
    orig_basename = getattr(mod, "IMAGE_BASENAME", "kana_wallpaper_current")
    orig_apply = getattr(mod, "APPLY_WALLPAPER", True)
    orig_save = getattr(mod, "SAVE_IMAGE", True)
    orig_save_dir = getattr(mod, "IMAGE_SAVE_DIR", None)
    had_save_dir = hasattr(mod, "IMAGE_SAVE_DIR")

    try:
        setattr(mod, "APPLY_WALLPAPER", bool(apply_wallpaper))
        setattr(mod, "SAVE_IMAGE", True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_label = "∞" if int(count) == 0 else str(int(count))

        # 連続出力では、既存の出力先の配下にサブフォルダを作ってまとめて保存する
        # base_dir の決定:
        # - core 側の IMAGE_SAVE_DIR が有効ならそれを使う
        # - 未設定/空なら _state_dir()（=このランチャーの隣/_kana_state）へ寄せる
        if isinstance(orig_save_dir, (str, Path)) and str(orig_save_dir).strip():
            base_dir = str(orig_save_dir).strip()
        else:
            base_dir = str(_state_dir())

        p_base = Path(base_dir)
        if not p_base.is_absolute():
            # 相対指定は「このランチャーの場所」基準に解釈（CWD依存を避ける）
            p_base = (Path(__file__).resolve().parent / p_base).resolve()
        out_dir = p_base / f"continuous_{ts}"
        if out_dir.exists():
            # 同名衝突を避ける（滅多に起きないが念のため）
            for n in range(1, 1000):
                cand = out_dir.parent / f"{out_dir.name}_{n:02d}"
                if not cand.exists():
                    out_dir = cand
                    break
        out_dir.mkdir(parents=True, exist_ok=True)
        setattr(mod, "IMAGE_SAVE_DIR", str(out_dir))

        print(C("96;1", f"\n[{tr('repeat_started')}]  count={total_label}  wallpaper={'ON' if apply_wallpaper else 'OFF'}  out={out_dir}\n"))

        i = 0
        while True:
            if int(count) > 0 and i >= int(count):
                break
            i += 1

            # 連番で保存（上書きを避ける）
            # 例: kana_wallpaper_current_20260216_210530_0001.png
            setattr(mod, "IMAGE_BASENAME", f"{orig_basename}_{ts}_{i:04d}")

            print(C("96", f"--- Continuous {i}/{total_label} ---"))
            run_core(mod)

    except KeyboardInterrupt:
        print(C("90", f"\n[{tr('repeat_stopped')}]"))
    finally:
        # もとの設定へ戻す
        try:
            setattr(mod, "IMAGE_BASENAME", orig_basename)
            setattr(mod, "APPLY_WALLPAPER", orig_apply)
            setattr(mod, "SAVE_IMAGE", orig_save)
            if had_save_dir:
                setattr(mod, "IMAGE_SAVE_DIR", orig_save_dir)
        except Exception:
            pass


def run_core_repeat_random_preset(
    mod: Any,
    presets: Optional[List[Dict[str, Any]]],
    common_defaults: Dict[str, Any],
    eff_for_run: Dict[str, Any],
    dd_paths: List[str],
    seed_mode: str,
    seed_value: Optional[int],
    last_seed: Optional[int],
    count: int = 0,
    apply_wallpaper: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """連続出力（ランダムプリセット版）

    - 1回ごとにプリセットを選び直してから core へ適用して実行する。
    - どれが選ばれたかは既定で表示しない（RANDOM_PRESET_REVEAL_NAME=Trueで表示）。
    - 出力先は連続出力用サブフォルダ（continuous_YYYYmmdd_HHMMSS）へまとめる。
    """
    orig_basename = getattr(mod, "IMAGE_BASENAME", "kana_wallpaper_current")
    orig_apply = getattr(mod, "APPLY_WALLPAPER", True)
    orig_save = getattr(mod, "SAVE_IMAGE", True)
    orig_save_dir = getattr(mod, "IMAGE_SAVE_DIR", None)
    had_save_dir = hasattr(mod, "IMAGE_SAVE_DIR")

    last_cfg_used: Optional[Dict[str, Any]] = None
    last_seed_out: Optional[int] = last_seed if isinstance(last_seed, int) else None

    try:
        setattr(mod, "APPLY_WALLPAPER", bool(apply_wallpaper))
        setattr(mod, "SAVE_IMAGE", True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_label = "∞" if int(count) == 0 else str(int(count))

        # 連続出力では、既存の出力先の配下にサブフォルダを作ってまとめて保存する
        # base_dir の決定:
        # - core 側の IMAGE_SAVE_DIR が有効ならそれを使う
        # - 未設定/空なら _state_dir()（=このランチャーの隣/_kana_state）へ寄せる
        if isinstance(orig_save_dir, (str, Path)) and str(orig_save_dir).strip():
            base_dir = str(orig_save_dir).strip()
        else:
            base_dir = str(_state_dir())

        p_base = Path(base_dir)
        if not p_base.is_absolute():
            # 相対指定は「このランチャーの場所」基準に解釈（CWD依存を避ける）
            p_base = (Path(__file__).resolve().parent / p_base).resolve()

        out_dir = p_base / f"continuous_{ts}"
        if out_dir.exists():
            # 同名衝突を避ける（滅多に起きないが念のため）
            for n in range(1, 1000):
                cand = out_dir.parent / f"{out_dir.name}_{n:02d}"
                if not cand.exists():
                    out_dir = cand
                    break
        out_dir.mkdir(parents=True, exist_ok=True)
        setattr(mod, "IMAGE_SAVE_DIR", str(out_dir))

        print(C("96;1", f"\n[{tr('repeat_started')}]  count={total_label}  wallpaper={'ON' if apply_wallpaper else 'OFF'}  out={out_dir}\n"))

        i = 0
        while True:
            if int(count) > 0 and i >= int(count):
                break
            i += 1

            # 連番で保存（上書きを避ける）
            setattr(mod, "IMAGE_BASENAME", f"{orig_basename}_{ts}_{i:04d}")
            print(C("96", f"--- Continuous {i}/{total_label} ---"))

            # 1回ごとにプリセットを選び直す
            chosen = None
            if isinstance(presets, list) and presets:
                chosen = _RANDOM_PRESET_RNG.choice(presets)
                cand_cfg = _normalize_cfg_archives(chosen.get("config") if isinstance(chosen.get("config"), dict) else {})
                cand_cfg = _sanitize_layout_preset_cfg(cand_cfg)
                cfg_i: Dict[str, Any] = dict(common_defaults, **dict(cand_cfg))
            else:
                cfg_i = dict(common_defaults)

            # エフェクトはランチャー側の設定を毎回反映
            if isinstance(eff_for_run, dict) and eff_for_run:
                try:
                    cfg_i.update(dict(eff_for_run))
                except Exception as e:
                    _kana_silent_exc('launcher:repeat_random_preset:fx_update', e)
                    pass

            # UI 表示はランチャー設定を優先（プリセット保存値に引きずられない）
            if bool(globals().get('LAUNCHER_FORCE_CORE_UI', True)):
                cfg_i['ui_style'] = LAUNCHER_UI_STYLE
                cfg_i['unicode_bling'] = LAUNCHER_UNICODE_BLING
                cfg_i['progress_bar_style'] = LAUNCHER_PROGRESS_BAR_STYLE
                cfg_i['progress_width'] = LAUNCHER_PROGRESS_WIDTH

            # seed 設定（再現性）
            try:
                apply_seed_pref(cfg_i, seed_mode, seed_value, last_seed_out)
            except Exception as e:
                _kana_silent_exc('launcher:repeat_random_preset:seed', e)
                pass

            # core に反映して実行
            apply_config_to_core(mod, cfg_i, dd_paths)
            apply_state_paths_to_core(mod, preserve_image_save_dir=True)
            try:
                setattr(mod, "IMAGE_SAVE_DIR", str(out_dir))
            except Exception:
                pass

            # 連続中は、選ばれたプリセット名を表示できる（既定: 表示する）
            # ※「ランダム開始」時点では表示しない → 実行中だけ表示、という運用が可能
            try:
                if chosen is not None:
                    name = str(chosen.get("name", "")).strip()
                    reveal = bool(globals().get("RANDOM_PRESET_REVEAL_NAME_DURING_CONTINUOUS", True))
                    if reveal and name:
                        print(tr("preset_random_pick_named").format(name=name))
                    else:
                        print(tr("preset_random_pick"))
            except Exception as e:
                _kana_silent_exc('launcher:repeat_random_preset:print', e)
                pass

            run_core(mod)
            last_cfg_used = dict(cfg_i)

            # 次回の same_last 用に seed_used を更新
            try:
                _su = getattr(mod, '_RUN_SEED_USED', None)
                if isinstance(_su, int):
                    last_seed_out = int(_su)
            except Exception:
                pass

    except KeyboardInterrupt:
        print(C("90", f"\n[{tr('repeat_stopped')}]"))
    finally:
        # もとの設定へ戻す
        try:
            setattr(mod, "IMAGE_BASENAME", orig_basename)
            setattr(mod, "APPLY_WALLPAPER", orig_apply)
            setattr(mod, "SAVE_IMAGE", orig_save)
            if had_save_dir:
                setattr(mod, "IMAGE_SAVE_DIR", orig_save_dir)
        except Exception:
            pass

    return last_cfg_used, last_seed_out




def main() -> None:
    # KANA: 例外の握りつぶしを見える化（プロセス終了時に1回だけ表示）
    try:
        _kana_register_silent_exc_atexit(_launcher_note)
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    dd_paths = parse_dragdrop_paths(sys.argv)

    core_path = pick_core_file(here)
    mod = load_core_module(core_path)
    _set_launcher_core_ui(mod)
    # 状態ファイル（キャッシュ/ログ等）の置き場をまとめて適用
    apply_state_paths_to_core(mod)

    # aspect 計算（mosaic 推定用）
    w = int(getattr(mod, "WIDTH", 3840))
    h = int(getattr(mod, "HEIGHT", 2160))
    aspect = (w / h) if h > 0 else 1.777777
    # 本体設定に従う：動画フレーム抽出が有効かどうか、そして入力に動画が存在するかを検出
    video_active = bool(getattr(mod, "VIDEO_SCAN_ENABLE", False))
    has_videos = False
    try:
        exts = set(getattr(mod, "VIDEO_SCAN_EXTS", set()))
        roots = dd_paths[:] if dd_paths else list(getattr(mod, "DEFAULT_TARGET_DIRS", []) or [])
        recursive = bool(getattr(mod, "SCAN_RECURSIVE", True))
        has_videos = detect_video_presence([str(x) for x in roots], exts, recursive=recursive)
    except Exception:
        has_videos = False

    # 本体設定に従う：アーカイブ内画像スキャンが有効かどうか、そして入力にアーカイブが存在するかを検出
    zip_active = bool(getattr(mod, 'ZIP_SCAN_ENABLE', False))
    sevenz_active = bool(getattr(mod, 'SEVENZ_SCAN_ENABLE', False))
    rar_active = bool(getattr(mod, 'RAR_SCAN_ENABLE', False))
    # 互換のため設定キーは zip_scan_enable を流用（意味は“アーカイブ全般”）
    zip_active = bool(zip_active or sevenz_active or rar_active)
    has_zips = False
    try:
        exts_z = set(getattr(mod, 'ZIP_SCAN_EXTS', {'.zip', '.cbz'}))
        exts_7 = set(getattr(mod, 'SEVENZ_SCAN_EXTS', {'.7z', '.cb7'}))
        exts_r = set(getattr(mod, 'RAR_SCAN_EXTS', {'.rar', '.cbr'}))
        exts_arch = set()
        exts_arch.update({e.lower() for e in exts_z})
        exts_arch.update({e.lower() for e in exts_7})
        exts_arch.update({e.lower() for e in exts_r})
        if not exts_arch:
            exts_arch = {'.zip', '.cbz', '.7z', '.cb7', '.rar', '.cbr'}
        roots = dd_paths[:] if dd_paths else list(getattr(mod, 'DEFAULT_TARGET_DIRS', []) or [])
        recursive = bool(getattr(mod, 'SCAN_RECURSIVE', True))
        has_zips = detect_zip_presence([str(x) for x in roots], exts_arch, recursive=recursive)
    except Exception:
        has_zips = False

    common_defaults = {"video_active": video_active, "has_videos": has_videos, "zip_scan_enable": zip_active, "has_zips": has_zips, "ui_style": LAUNCHER_UI_STYLE, "unicode_bling": LAUNCHER_UNICODE_BLING, "progress_bar_style": LAUNCHER_PROGRESS_BAR_STYLE, "progress_width": LAUNCHER_PROGRESS_WIDTH}

    ppath = preset_file_path()
    presets = load_layout_presets_clean(ppath)
    lpath = last_run_file_path()
    last_cfg = load_last_run(lpath)
    if isinstance(last_cfg, dict):
        last_cfg = _normalize_cfg_archives(last_cfg)
    force_last_action = False  # 前回設定フローで「戻る」を押したとき、前回アクション選択へ戻す
    _launcher_banner(tr("title"))
    _launcher_note(tr("core").format(core=core_path.name))
    _launcher_note(tr("presets").format(path=str(ppath)))
    # dHash永続キャッシュのパス（本体側の実際の解決結果を優先）
    cache_path = ""
    try:
        if hasattr(mod, "_dhash_cache_path") and callable(getattr(mod, "_dhash_cache_path")):
            cache_path = str(getattr(mod, "_dhash_cache_path")())
        else:
            cache_path = str(getattr(mod, "DHASH_CACHE_FILE", ""))
    except Exception:
        cache_path = str(getattr(mod, "DHASH_CACHE_FILE", ""))
    cache_path = (cache_path or "").strip()
    if not cache_path:
        cache_path = "-"
    _launcher_note(tr("cache").format(path=cache_path))
    # Optional capability diagnostic (numpy/cv2/archives/ffprobe etc.)
    try:
        print_capability_diagnostic(mod)
    except Exception as e:
        _kana_silent_exc('launcher:L4639', e)
        pass
    if dd_paths:
        _launcher_note(tr("input_paths").format(n=len(dd_paths)))

    # シード（再現性）設定：random / same_last / specify（ランチャー内で保持）
    seed_mode = 'random'
    seed_value = None  # type: Optional[int]


    # 前回の固定seedがあれば、それを初期値として引き継ぐ
    try:
        if isinstance(last_cfg, dict):
            _sv = last_cfg.get('shuffle_seed', None)
            if isinstance(_sv, int):
                seed_mode = 'specify'
                seed_value = int(_sv)
    except Exception as e:
        _kana_silent_exc('launcher:L4657', e)
        pass
    # エフェクト（任意でメニューに入って編集）
    epath = effect_preset_file_path()
    epresets = load_presets(epath)
    effect_cfg = _normalize_effect_cfg(_effect_cfg_from_core(mod))
    try:
        if isinstance(last_cfg, dict):
            effect_cfg.update(_extract_effect_cfg_from_any(last_cfg))
            effect_cfg = _normalize_effect_cfg(effect_cfg)
    except Exception as e:
        _kana_silent_exc('launcher:L4668', e)
        pass

    # プリセット適用の既定（最後に選んだものを覚える）
    preset_apply_default_all = True

    # 壁紙に設定するか（直前の選択を引き継ぐ）
    last_apply_wallpaper = True
    try:
        if isinstance(last_cfg, dict):
            if 'apply_wallpaper' in last_cfg:
                last_apply_wallpaper = bool(last_cfg.get('apply_wallpaper'))
            elif 'set_wallpaper' in last_cfg:
                last_apply_wallpaper = bool(last_cfg.get('set_wallpaper'))
            elif 'wallpaper' in last_cfg:
                last_apply_wallpaper = bool(last_cfg.get('wallpaper'))
        else:
            last_apply_wallpaper = bool(getattr(mod, 'APPLY_WALLPAPER', True))
    except Exception as e:
        _kana_silent_exc('launcher:last_apply_wallpaper_init', e)
        last_apply_wallpaper = True


    while True:
        # 実行オプション（通常 / 連続出力）
        run_plan = {
            "apply_wallpaper": bool(last_apply_wallpaper),  # core側へ反映
            "continuous": False,
            "repeat_count": 1,         # 0=無限
            # ランダムプリセット開始 + 連続出力のときだけ True になります
            "random_preset_each_run": False,
            "random_preset_pool": None,
        }

        # 起動方法
        # 前回設定フローから「戻る」を押した場合は、起動方法を聞かずに前回アクション選択へ戻す
        if force_last_action and isinstance(last_cfg, dict) and last_cfg:
            mode = tr("mode_last")
            force_last_action = False
        else:
            # 起動方法の最初の質問（モード選択）に戻ってきたときは、
            # 前回設定 / エフェクトのサマリを再表示する（ただし前回設定フロー内では再表示しない）
            try:
                if isinstance(last_cfg, dict) and last_cfg:
                    _cfg0 = dict(common_defaults, **dict(last_cfg))
                    _ls_lines = preset_summary_verbose_lines(_cfg0)
                    if _ls_lines:
                        _launcher_note(tr('last_summary').format(summary=_ls_lines[0]))
                        for _it in _ls_lines[1:]:
                            _launcher_note(_it)
                    else:
                        ls = preset_summary(_cfg0, include_fx=False)
                        _launcher_note(tr('last_summary').format(summary=ls))
                # 現在のエフェクトを表示（メインメニューの概要確認用）
                try:
                    es = effect_preset_summary(effect_cfg)
                    _launcher_note(tr('effects_summary').format(summary=es))
                except Exception as e:
                    _kana_silent_exc('launcher:L4691', e)
                    pass
            except Exception as e:
                _kana_silent_exc('launcher:L4693', e)
                pass
            # seed preview
            try:
                _ls = None
                if isinstance(last_cfg, dict):
                    _ls = last_cfg.get('shuffle_seed', None)
                    if not isinstance(_ls, int):
                        _ls = last_cfg.get('seed_used', None)
                if seed_mode == 'specify' and isinstance(seed_value, int):
                    _launcher_note(f"seed: {seed_value}")
                elif seed_mode == 'same_last':
                    if isinstance(_ls, int):
                        _launcher_note(f"seed: {_ls}")
                    else:
                        _launcher_note("seed: N/A")
                else:
                    _launcher_note("seed: random")
            except Exception as e:
                _kana_silent_exc('launcher:L4712', e)
                pass
            mode_choices = []
            if isinstance(last_cfg, dict) and last_cfg:
                mode_choices.append(tr("mode_last"))
            mode_choices += [tr("mode_manual"), tr("mode_preset"), tr("mode_random_preset"), tr("mode_manage_presets"), tr("mode_effects"), tr("mode_seed"), tr("mode_export_core")]
            mode = ask_choice(
                tr("mode"),
                mode_choices,
                default_index=1,
                allow_back=True,
            )
            if mode == BACK_TOKEN:
                return


        cfg = None
        # この起動で使うエフェクト設定（既定：現在の effect_cfg）
        eff_for_run: Dict[str, Any] = dict(effect_cfg) if isinstance(effect_cfg, dict) else {}

        # プリセット管理（名前変更/削除/並び替え）
        if mode == tr("mode_manage_presets"):
            try:
                manage_config_presets(ppath)
                presets = load_layout_presets_clean(ppath)
            except Exception as e:
                print(f"（プリセット管理でエラー: {e}）")
            continue

        # エフェクト設定
        if mode == tr("mode_effects"):
            effect_cfg = effect_menu(effect_cfg, epresets, epath)
            continue

        # シード（再現性）
        if mode == tr("mode_seed"):
            try:
                last_seed = None
                if isinstance(last_cfg, dict):
                    last_seed = last_cfg.get('shuffle_seed', None)
                    if not isinstance(last_seed, int):
                        last_seed = last_cfg.get('seed_used', None)
                seed_mode, seed_value = seed_menu(seed_mode, seed_value, last_seed if isinstance(last_seed, int) else None)

                # 現在の選択を common_defaults/last_run に保存（次回も維持）
                cfg_tmp: Dict[str, Any] = dict(common_defaults) if isinstance(common_defaults, dict) else {}
                if isinstance(last_cfg, dict) and last_cfg:
                    cfg_tmp.update(dict(last_cfg))

                apply_seed_pref(cfg_tmp, seed_mode, seed_value, last_seed if isinstance(last_seed, int) else None)
                try:
                    save_last_run(lpath, cfg_tmp)
                    last_cfg = dict(cfg_tmp)
                except Exception as e:
                    _kana_silent_exc('launcher:L4766', e)
                    pass
            except Exception as e:
                print(f"（シード設定でエラー: {e}）")
            continue

        # コア用の外部設定JSONをエクスポート（core自体は変更しない）
        if mode == tr("mode_export_core"):
            try:
                # いまの「レイアウト設定」は前回設定（last_cfg）を基準にします（無ければ既定）
                cfg_export: Dict[str, Any] = dict(common_defaults) if isinstance(common_defaults, dict) else {}
                if isinstance(last_cfg, dict) and last_cfg:
                    cfg_export.update(dict(last_cfg))
                # いまのエフェクト設定を合成（snake_case）
                if isinstance(effect_cfg, dict) and effect_cfg:
                    cfg_export.update(dict(effect_cfg))
                # シード設定を反映（再現性）
                # ※ seed 関係はエクスポートでは上書きしません（core 側の seed を保持）

                # 互換: ステンドグラスの角数(min/max)は設定項目を撤去したため、エクスポートから除外します
                cfg_export.pop('stained_glass_min_vertices', None)
                cfg_export.pop('stained_glass_max_vertices', None)



                # 派生値を整える（可能なら）
                try:
                    _recalc_derived(cfg_export, aspect)
                except Exception as e:
                    _kana_silent_exc('launcher:L4789', e)
                    pass
                # エクスポート前に再確認（既定は戻る）
                try:
                    _launcher_note('----')
                    _launcher_note(f'core: {core_path}')
                    try:
                        summary = preset_summary(cfg_export, include_fx=False)
                        # layout別に、サマリに無い追加情報だけ付与（重複を避ける）
                        layout = str(cfg_export.get('layout') or '').strip()
                        extra = []
                        if layout == 'quilt':
                            tiles = cfg_export.get('count') or cfg_export.get('quilt_max_tiles')
                            split_style = cfg_export.get('quilt_split_style')
                            if tiles:
                                extra.append(f"tiles={tiles}")
                            if split_style:
                                extra.append(f"split={split_style}")
                        elif layout == 'stained-glass':
                            pieces = cfg_export.get('count')
                            lead_w = cfg_export.get('stained_glass_lead_width')
                            lead_a = cfg_export.get('stained_glass_lead_alpha')
                            ang = cfg_export.get('stained_glass_max_corner_angle_deg')
                            apm = cfg_export.get('stained_glass_effects_apply_mode')
                            if pieces:
                                extra.append(f"pieces={pieces}")
                            if (lead_w is not None) or (lead_a is not None):
                                if lead_w is not None and lead_a is not None:
                                    extra.append(f"lead={lead_w}@{lead_a}")
                                elif lead_w is not None:
                                    extra.append(f"lead_w={lead_w}")
                                else:
                                    extra.append(f"lead_a={lead_a}")
                            if ang is not None:
                                extra.append(f"max_angle={ang}")
                            if apm:
                                extra.append(f"fx={apm}")
                        elif layout == 'grid':
                            tl = cfg_export.get('grid_video_timeline')
                            if tl:
                                extra.append(f"timeline={tl}")

                        # 結合（すでに含まれる文字は付与しない）
                        for it in extra:
                            if it and (it not in summary):
                                summary += f" | {it}"
                        _launcher_note(tr('last_summary').format(summary=summary))
                    except Exception:
                        _launcher_note(tr('last_summary').format(summary=preset_summary(cfg_export, include_fx=False)))

                    _launcher_note(tr('effects_summary').format(summary=effect_preset_summary(_normalize_effect_cfg(_extract_effect_cfg_from_any(cfg_export)))))

                    # 追加: video / face AI（前回サマリに出にくい）
                    try:
                        def _fmt_onoff(v):
                            return 'on' if bool(v) else 'off'

                        vact = cfg_export.get('video_active')
                        if vact is not None:
                            vmode = cfg_export.get('video_mode')
                            vsel = cfg_export.get('video_select_mode')
                            vppv = cfg_export.get('video_frames_per_video')
                            v = f"video: {_fmt_onoff(vact)}"
                            if vmode:
                                v += f" | mode={vmode}"
                            if vsel:
                                v += f" | select={vsel}"
                            if vppv:
                                v += f" | per_video={vppv}"
                            _launcher_note(v)

                        fai = cfg_export.get('face_ai_enable')
                        if fai is not None:
                            backend = cfg_export.get('face_ai_backend')
                            dev = cfg_export.get('face_ai_device')
                            sens = cfg_export.get('face_ai_sensitivity')
                            s = f"face_ai: {_fmt_onoff(fai)}"
                            if backend:
                                s += f" | backend={backend}"
                            if dev:
                                s += f" | dev={dev}"
                            if sens:
                                s += f" | sens={sens}"
                            _launcher_note(s)
                    except Exception as e:
                        _kana_silent_exc('launcher:L4878', e)
                        pass
                    _launcher_note('----')
                except Exception as e:
                    _kana_silent_exc('launcher:L4881', e)
                    pass
                ok = ask_choice('外部設定JSONをエクスポートしますか？', ['エクスポート'], default_index=0, allow_back=True)
                if ok == BACK_TOKEN:
                    continue

                outp = export_settings_to_core_inplace(core_path, cfg_export, dd_paths=[])
                (_launcher_note(f"OK: exported external config -> {outp}") if outp else _launcher_note("NG: export failed"))
            except Exception as e:
                print(f"（エクスポートでエラー: {e}）")
            continue

        # 上記の設定で開始
        if isinstance(last_cfg, dict) and last_cfg and mode == tr("mode_last"):
            cfg = dict(last_cfg)
            cfg = dict(common_defaults, **cfg)
            # 現在の環境（動画有無など）は都度上書き
            try:
                cfg.update(common_defaults)
                _recalc_derived(cfg, aspect)
            except Exception as e:
                _kana_silent_exc('launcher:L4902', e)
                pass
            # NOTE:
            # 「起動方法」メニューの直前で前回/エフェクトのサマリは表示済み。
            # ここ（前回設定フロー内）では再表示しない。

            act = ask_choice(
                tr("last_action"),
                [
                    tr("preset_action_run"),
                    tr("preset_action_edit"),
                    tr("last_action_repeat"),
                ],
                default_index=1,
                allow_back=True,
            )
            if act == BACK_TOKEN:
                cfg = BACK_TOKEN

            elif act == tr("last_action_repeat"):
                run_plan["continuous"] = True
                run_plan["repeat_count"] = ask_int(tr("repeat_count_prompt"), 10, 0, 999999, allow_back=True)


            elif act == tr("preset_action_edit"):
                cfg2 = build_config(aspect, defaults=cfg, start_at_last=False)
                if cfg2 == BACK_TOKEN:
                    # 前回設定の編集途中で戻る → 起動方法選択をスキップして、前回設定アクション選択へ戻す
                    force_last_action = True
                    cfg = BACK_TOKEN
                else:
                    cfg = cfg2
            # run の場合はそのまま cfg を使う

        # プリセット開始
        if cfg is None and mode in (tr("mode_preset"), tr("mode_random_preset")):
            if not presets:
                print(tr("preset_none"))
                cfg = build_config(aspect, defaults=dict(common_defaults))
                if cfg == BACK_TOKEN:
                    continue
            else:
                if mode == tr("mode_random_preset"):
                    chosen = _RANDOM_PRESET_RNG.choice(presets)
                    name = str(chosen.get("name", "")).strip()
                    if bool(globals().get("RANDOM_PRESET_REVEAL_NAME", False)) and name:
                        print(tr("preset_random_pick_named").format(name=name))
                    else:
                        print(tr("preset_random_pick"))
                    cand_cfg = _normalize_cfg_archives(chosen.get("config") if isinstance(chosen.get("config"), dict) else {})
                else:
                    print_preset_list(presets)
                    idx = ask_int(tr("preset_pick"), 0,  1, len(presets), allow_back=True)
                    if idx == BACK_TOKEN or idx == 0:
                        continue
                    chosen = presets[int(idx) - 1]
                    cand_cfg = _normalize_cfg_archives(chosen.get("config") if isinstance(chosen.get("config"), dict) else {})

                # 並べ方プリセットはレイアウトのみ（エフェクトは別管理のため読み込まない）

                # レイアウト設定だけの cfg（FX/UIキーは適用時に除去）
                cand_layout_cfg = _sanitize_layout_preset_cfg(cand_cfg)

                # この起動で使うエフェクト設定（常に“現在のエフェクト”）
                eff_for_run = dict(effect_cfg) if isinstance(effect_cfg, dict) else {}

                # このまま実行 / 編集（プリセット読込→トップへ） / 連続出力 / 戻る
                act = ask_choice(
                    tr("preset_action"),
                    [
                        tr("preset_action_run"),
                        tr("preset_action_load_to_top"),
                        tr("preset_action_repeat"),
                    ],
                    default_index=1,
                    allow_back=True,
                )
                if act == BACK_TOKEN:
                    continue

                if act == tr("preset_action_load_to_top"):
                    # プリセットを「前回設定」として読み込み、メインメニューへ戻る
                    try:
                        loaded_cfg = dict(cand_layout_cfg) if isinstance(cand_layout_cfg, dict) else {}
                        # 現在のエフェクト状態は維持（後でエフェクトメニューで編集できます）
                        if isinstance(effect_cfg, dict):
                            loaded_cfg.update(dict(effect_cfg))
                        last_cfg = dict(loaded_cfg)
                        save_last_run(lpath, last_cfg)
                    except Exception as e:
                        _kana_silent_exc('launcher:preset_load_to_top', e)
                        pass
                    continue

                elif act == tr("preset_action_repeat"):
                    run_plan["continuous"] = True
                    run_plan["repeat_count"] = ask_int(tr("repeat_count_prompt"), 10, 0, 999999, allow_back=True)

                    # ランダムプリセット開始の場合：連続出力のたびに別プリセットを選び直す
                    if mode == tr("mode_random_preset"):
                        run_plan["random_preset_each_run"] = True
                        run_plan["random_preset_pool"] = list(presets) if isinstance(presets, list) else None


                if act in (tr("preset_action_run"), tr("preset_action_repeat")):
                    # まずはプリセットのレイアウトを採用
                    cfg = dict(cand_layout_cfg)
                    cfg = dict(common_defaults, **cfg)

                    # UI 表示はランチャー設定を優先（プリセット保存値に引きずられない）
                    if bool(globals().get('LAUNCHER_FORCE_CORE_UI', True)):
                        cfg['ui_style'] = LAUNCHER_UI_STYLE
                        cfg['unicode_bling'] = LAUNCHER_UNICODE_BLING
                        cfg['progress_bar_style'] = LAUNCHER_PROGRESS_BAR_STYLE
                        cfg['progress_width'] = LAUNCHER_PROGRESS_WIDTH
                    # 動画が存在する場合のみ：必要な追加設定を質問（プリセット実行時）
                    if bool(common_defaults.get("video_active", False)) and bool(common_defaults.get("has_videos", False)):
                        # 既定は自動配分（VIDEO_FRAMES_PER_VIDEO=0）
                        cfg.setdefault("video_mode", "auto")
                        cfg.setdefault("video_frames_per_video", 0)
                        # 選別方式はプリセットに無ければ質問
                        if "video_select_mode" not in cfg:
                            vsm = choose_video_select_mode(str(getattr(mod, "VIDEO_FRAME_SELECT_MODE", "random")))
                            if vsm != BACK_TOKEN:
                                cfg["video_select_mode"] = str(vsm)
                        # キャッシュ削除（終了後）は本体デフォルトを尊重（プリセットに無ければ True）
                else:
                    # レイアウト設定を編集してから実行
                    cfg = build_config(aspect, defaults=dict(common_defaults, **dict(cand_layout_cfg)))
                    if cfg == BACK_TOKEN:
                        continue

        if cfg is None:
            # 手動
            cfg = build_config(aspect, defaults=dict(common_defaults))
            if cfg == BACK_TOKEN:
                continue

        # build_config から BACK が返った場合は、開始メニューへ戻る
        if cfg == BACK_TOKEN:
            continue

        # エフェクト設定を最終cfgへ反映（起動ごとの eff_for_run を使用）
        if isinstance(cfg, dict) and isinstance(eff_for_run, dict):
            try:
                cfg.update(dict(eff_for_run))
            except Exception as e:
                _kana_silent_exc('launcher:L5023', e)
                pass
        # 保存（任意）
        while True:
            save_choice = ask_choice(
                tr("preset_save"),
                [tr("preset_save_yes"), tr("preset_save_no")],
                default_index=2,
                allow_back=True,
            )

            if save_choice == BACK_TOKEN:
                # 「戻る」は直前の質問へ戻る（前回設定フローでも同様）
                resume = None
                if isinstance(cfg, dict):
                    resume = cfg.get("_wizard_last_key", None)

                _defaults = dict(common_defaults)
                if isinstance(cfg, dict):
                    _defaults.update(dict(cfg))

                cfg2 = build_config(
                    aspect,
                    defaults=_defaults,
                    start_at_last=True,
                    resume_key=resume,
                )
                if cfg2 == BACK_TOKEN:
                    cfg = BACK_TOKEN
                    break
                cfg = cfg2
                continue

            if save_choice == tr("preset_save_yes"):
                # 名前入力（戻る対応）
                while True:
                    name = ask_text(tr("preset_name"), allow_back=True)
                    if name == BACK_TOKEN:
                        break  # 保存メニューへ
                    if not name:
                        print(tr("preset_name_empty"))
                        break

                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # 並べ方プリセットにはエフェクト情報を保存しない
                    cfg_for_save = dict(cfg) if isinstance(cfg, dict) else {}
                    try:
                        for k in list(_extract_effect_cfg_from_any(cfg_for_save).keys()):
                            cfg_for_save.pop(k, None)
                    except Exception as e:
                        _kana_silent_exc('launcher:L5073', e)
                        pass
                    item = {
                        'name': name,
                        'created_at': now,
                        'summary': preset_summary(cfg_for_save, include_fx=False),
                        'config': cfg_for_save,
                    }

                    existing = find_preset_by_name(presets, name)
                    if existing is not None:
                        ow = ask_choice(
                            tr("preset_overwrite"),
                            [tr("preset_overwrite_yes"), tr("preset_overwrite_no")],
                            default_index=2,
                            allow_back=True,
                        )
                        if ow == BACK_TOKEN:
                            continue  # もう一度名前入力へ
                        if ow == tr("preset_overwrite_yes"):
                            presets[existing] = item
                            try:
                                if 'zip_scan_enable' in cfg and 'archives_enable' not in cfg:
                                    cfg['archives_enable'] = bool(cfg.get('zip_scan_enable'))
                            except Exception as e:
                                _kana_silent_exc('launcher:L5097', e)
                                pass
                            save_presets(ppath, presets)
                        # no overwrite: 何もしない
                    else:
                        presets.append(item)
                        save_presets(ppath, presets)

                    break

            # 保存しない or 保存完了
            break

        if cfg == BACK_TOKEN:
            continue

        # 本体へ設定適用→実行
        if isinstance(cfg, dict):
            try:
                if 'zip_scan_enable' in cfg and 'archives_enable' not in cfg:
                    cfg['archives_enable'] = bool(cfg.get('zip_scan_enable'))
            except Exception as e:
                _kana_silent_exc('launcher:archives_compat', e)
                pass
            save_last_run(lpath, cfg)
        # シード設定を反映（再現性）
        last_seed_for_seedpref = None
        try:
            _ls = None
            if isinstance(last_cfg, dict):
                _ls = last_cfg.get('seed_used', None)
            last_seed_for_seedpref = _ls if isinstance(_ls, int) else None
        except Exception as e:
            _kana_silent_exc('launcher:L5127', e)
            last_seed_for_seedpref = None

        # ランダムプリセット + 連続出力のときは、1回ごとに cfg を作り直すため
        # ここでは seed/apply_config を確定させず、ループ内で適用する。
        if not (bool(run_plan.get("continuous", False)) and bool(run_plan.get("random_preset_each_run", False))):
            try:
                apply_seed_pref(cfg, seed_mode, seed_value, last_seed_for_seedpref)
            except Exception as e:
                _kana_silent_exc('launcher:L5127', e)
                pass
            apply_config_to_core(mod, cfg, dd_paths)

        # 状態ファイル（キャッシュ/ログ等）の置き場をまとめて適用（毎回共通）
        apply_state_paths_to_core(mod)

        print("\n" + tr("run") + "\n")
        try:
            # 壁紙へ反映するか（毎回確認）
            try:
                wp = ask_choice(
                    tr("apply_wallpaper_prompt"),
                    [tr("apply_wallpaper_yes"), tr("apply_wallpaper_no")],
                    default_index=1 if bool(run_plan.get("apply_wallpaper", True)) else 2,
                    allow_back=True,
                )
                if wp == BACK_TOKEN:
                    continue
                run_plan["apply_wallpaper"] = (wp == tr("apply_wallpaper_yes"))
                last_apply_wallpaper = bool(run_plan.get("apply_wallpaper", True))
            except Exception as e:
                _kana_silent_exc('launcher:apply_wallpaper_prompt', e)
                pass
            try:
                setattr(mod, "APPLY_WALLPAPER", bool(run_plan.get("apply_wallpaper", True)))
            except Exception:
                pass

            if bool(run_plan.get("continuous", False)):
                if bool(run_plan.get("random_preset_each_run", False)):
                    pool = run_plan.get("random_preset_pool", None)
                    last_cfg_used, _ = run_core_repeat_random_preset(
                        mod,
                        presets=pool if isinstance(pool, list) else None,
                        common_defaults=dict(common_defaults) if isinstance(common_defaults, dict) else {},
                        eff_for_run=dict(eff_for_run) if isinstance(eff_for_run, dict) else {},
                        dd_paths=dd_paths,
                        seed_mode=str(seed_mode),
                        seed_value=seed_value if isinstance(seed_value, int) else None,
                        last_seed=last_seed_for_seedpref if isinstance(last_seed_for_seedpref, int) else None,
                        count=int(run_plan.get("repeat_count", 1)),
                        apply_wallpaper=bool(run_plan.get("apply_wallpaper", True)),
                    )
                    if isinstance(last_cfg_used, dict) and last_cfg_used:
                        cfg = dict(last_cfg_used)
                else:
                    run_core_repeat(
                        mod,
                        count=int(run_plan.get("repeat_count", 1)),
                        apply_wallpaper=bool(run_plan.get("apply_wallpaper", True)),
                    )
            else:
                run_core(mod)

            # 実行で使われた seed を保存（前回と同じ生成に使える）
            try:
                _su = getattr(mod, '_RUN_SEED_USED', None)
                if isinstance(_su, int):
                    cfg['seed_used'] = int(_su)
                    save_last_run(lpath, cfg)
            except Exception as e:
                _kana_silent_exc('launcher:L5141', e)
                pass
        except Exception:
            tb = traceback.format_exc()
            log_path = _state_item_path("launcher_error_log")
            try:
                log_path.write_text(tb, encoding="utf-8")
            except Exception as e:
                _kana_silent_exc('launcher:L5149', e)
                pass
            print(tb)
            print(tr("err_log").format(path=str(log_path)))
            raise

        return


if __name__ == "__main__":
    main()
