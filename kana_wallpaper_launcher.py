"""\
Kana Wallpaper Launcher
==================================================

目的
----
このランチャーは、Kana Wallpaper の各レイアウト（mosaic／grid／hex／random）で
質問の流れや用語をできるだけ統一し、不要な質問を減らしつつ、
必要なところは選べる（グラデ方向・散らし・最適化パラメータ等）ようにするためのものです。

本ファイルの特徴
----------------
* **プリセット機能あり**
  - 手動設定 → 任意で保存
  - プリセットから開始／ランダムプリセットから開始
  - プリセット一覧は「名前＋概要（サマリ）＋作成日時」を表示

  - mosaic-uniform-height：ROWS（行数）だけ指定
  - mosaic-uniform-width：COLS（列数）だけ指定
  - キャンバス縦横比（core の WIDTH/HEIGHT）からもう一方を推定し、ユニーク枚数も推定

  - 共通設定（SELECT_MODE／フルシャッフル／ZIP）→ サイズ → 配置（簡易）→ 方向（必要時）→ 最適化（任意）
  - フルシャッフルONなら、配置（簡易）や最適化の質問はスキップ

  - DEFAULT_LANG を "ja"／"en" に変更すると UI 表示が切り替わります。

対応コア
--------
同じフォルダにある `kana_wallpaper_unified_final.py` を読み込みます。
"""


from __future__ import annotations

import json
import os
import random
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import importlib.util

sys.dont_write_bytecode = True  # __pycache__ を作らない


# =============================================================================
# 設定：ここはユーザーが編集してOK
# =============================================================================

# 表示言語（"ja" or "en"）
DEFAULT_LANG = "ja"

# プリセット保存先（None の場合はOSの一時フォルダに自動）
# 例）PRESET_FILE = r".\\kana_wallpaper_presets.json"  # このフォルダに保存（内容にはパスが入ることがあります）
PRESET_FILE: Optional[str] = None

# 環境変数での上書き（あれば優先）
ENV_PRESET_FILE = "KANA_WALLPAPER_PRESET_FILE"

# プリセットの保存ファイル名（一時フォルダ利用時）
DEFAULT_PRESET_BASENAME = "kana_wallpaper_presets.json"


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
        "layout": "レイアウトを選択",
        "layout_mosaic_h": "mosaic-uniform-height",
        "layout_mosaic_w": "mosaic-uniform-width",
        "layout_grid": "grid",
        "layout_hex": "hex",
        "layout_random": "random",
        "select_mode": "抽出モード（SELECT_MODE）",
        "shuffle": "完全シャッフル（ARRANGE_FULL_SHUFFLE）",
        "zip": "ZIP内画像も候補に追加（ZIP_SCAN_ENABLE）",
        "on": "on",
        "off": "off",
        "rows": "ROWS（行数）",
        "cols": "COLS（列数）",
        "hex_count": "COUNT（枚数目安）",
        "hex_orient": "HEX_TIGHT_ORIENT（行ずらし/列ずらし）",
        "hex_orient_col": "col-shift",
        "hex_orient_row": "row-shift",
        "arr_simple": "配置（簡易）",
        "arr_diag": "グラデーション（対角）",
        "arr_hilb": "グラデーション（ヒルベルト）",
        "arr_scatter": "散らす（ばらけ）",
        "diag_dir": "対角方向",
        "diag_tlbr": "左上→右下 (↘)",
        "diag_brtl": "右下→左上 (↖)",
        "diag_trbl": "右上→左下 (↙)",
        "diag_bltr": "左下→右上 (↗)",
        "opt_extra": "最適化パラメータ（任意）",
        "opt_default": "デフォルト（高速・最適化なし）",
        "opt_tune": "調整する（最適化を実行：steps / reheats / k）",
        "steps": "最適化 steps（1000〜200000）",
        "reheats": "再加熱回数（0〜6）",
        "k": "近傍 k（3〜24）",
        "mosaic_est": " -> 推定 {other}={v} / ユニーク枚数≈{cnt}",
        "preset_title": "プリセット",
        "preset_none": "プリセットがありません。手動設定に切り替えます。",
        "preset_pick": "プリセットを選択",
        "preset_random_pick": "ランダムで選ばれました：{name}",
        "preset_action": "このプリセットで",
        "preset_action_run": "このまま実行",
        "preset_action_edit": "編集してから実行",
        "preset_action_back": "戻る",
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
        "err_no_core": "本体ファイルが見つかりません。kana_wallpaper_unified_final.py を同じフォルダに置いてください。",
        "err_core_load": "本体のロードに失敗: {path}",
    },
    "en": {
        "title": "=== Kana Wallpaper Launcher ===",
        "core": "core: {core}",
        "presets": "presets: {path}",
        "cache": "cache(dHash): {path}",
        "mode": "Start mode",
        "mode_manual": "Manual setup",
        "mode_preset": "Start from preset",
        "mode_random_preset": "Start with random preset",
        "layout": "Select layout",
        "layout_mosaic_h": "mosaic-uniform-height",
        "layout_mosaic_w": "mosaic-uniform-width",
        "layout_grid": "grid",
        "layout_hex": "hex",
        "layout_random": "random",
        "select_mode": "Selection mode (SELECT_MODE)",
        "shuffle": "Full shuffle (ARRANGE_FULL_SHUFFLE)",
        "zip": "Scan images inside ZIP files (ZIP_SCAN_ENABLE)",
        "on": "on",
        "off": "off",
        "rows": "ROWS",
        "cols": "COLS",
        "hex_count": "COUNT (approx.)",
        "hex_orient": "HEX_TIGHT_ORIENT",
        "hex_orient_col": "col-shift",
        "hex_orient_row": "row-shift",
        "arr_simple": "Arrangement (simple)",
        "arr_diag": "Gradient (diagonal)",
        "arr_hilb": "Gradient (Hilbert)",
        "arr_scatter": "Scatter",
        "diag_dir": "Diagonal direction",
        "diag_tlbr": "↘ Top-left → Bottom-right",
        "diag_brtl": "↖ Bottom-right → Top-left",
        "diag_trbl": "↙ Top-right → Bottom-left",
        "diag_bltr": "↗ Bottom-left → Top-right",
        "opt_extra": "Optimisation parameters (optional)",
        "opt_default": "Use defaults (fast / no optimisation)",
        "opt_tune": "Tune (run optimisation: steps / reheats / k)",
        "steps": "Steps (1000–200000)",
        "reheats": "Reheats (0–6)",
        "k": "k neighbours (3–24)",
        "mosaic_est": " -> estimated {other}={v} / unique≈{cnt}",
        "preset_title": "Presets",
        "preset_none": "No presets found. Switching to manual setup.",
        "preset_pick": "Select a preset",
        "preset_random_pick": "Randomly chosen: {name}",
        "preset_action": "With this preset",
        "preset_action_run": "Run as-is",
        "preset_action_edit": "Edit then run",
        "preset_action_back": "Back",
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
        "err_no_core": "本体ファイルが見つかりません。kana_wallpaper_unified_final.py を同じフォルダに置いてください。",
        "err_core_load": "Failed to load core: {path}",
    },
}


def tr(key: str) -> str:
    lang = DEFAULT_LANG if DEFAULT_LANG in TRANSLATIONS else "en"
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))


# =============================================================================
# 入力ヘルパ
# =============================================================================


BACK_TOKEN = "__BACK__"


def _is_back_raw(raw: str) -> bool:
    r = raw.strip().lower()
    return r in ("b", "back", "戻る", "もどる", "0")


def _print_choices(choices: List[str]) -> None:
    for i, ch in enumerate(choices, 1):
        print(f"  [{i}] {ch}")


def ask_choice(label: str, choices: List[str], default_index: int = 1, allow_back: bool = True) -> str:
    """番号または文字列で選択。

    - 通常の選択肢は [1..N]
    - 戻るは常に最後に表示し [0]
    - allow_back=False のときは戻るを表示/受付しない
    """
    if not choices:
        raise ValueError("choices が空です")

    opts = list(choices)
    back_label = tr("back")

    # 既定値は通常選択肢の範囲に丸める
    if default_index < 1 or default_index > len(opts):
        default_index = 1

    _print_choices(opts)
    if allow_back:
        print(f"  [0] {back_label}")
    prompt = f"{label} [{default_index}]: "

    while True:
        raw = input(prompt).strip()
        if not raw:
            return opts[default_index - 1]

        if allow_back and _is_back_raw(raw):
            return BACK_TOKEN

        # 数字入力
        try:
            n = int(raw)
            if allow_back and n == 0:
                return BACK_TOKEN
            if 1 <= n <= len(opts):
                return opts[n - 1]
        except Exception:
            pass

        # 文字列入力（完全一致）
        raw_l = raw.lower()
        if allow_back and raw_l == back_label.lower():
            return BACK_TOKEN

        for ch in opts:
            if raw_l == str(ch).lower():
                return ch

        print(tr("msg_choose"))

def ask_int(label: str, default: int, min_v: int, max_v: int, allow_back: bool = True):
    prompt = f"{label} [{default}]: "
    if allow_back:
        print(f"  [0] {tr('back')}")
    while True:
        raw = input(prompt).strip()
        if not raw:
            return default
        if allow_back and _is_back_raw(raw):
            return BACK_TOKEN
        try:
            v = int(raw)
        except Exception:
            print(tr("msg_enter_number"))
            continue
        if v < min_v or v > max_v:
            print(tr("msg_range").format(min_v=min_v, max_v=max_v))
            continue
        return v


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

    raw = input(prompt)
    raw = raw.rstrip("\n").rstrip("\r").strip()

    if allow_back and raw and _is_back_raw(raw):
        return BACK_TOKEN

    if not raw and default:
        return default

    return raw


# =============================================================================
# プリセット管理
# =============================================================================


def _default_temp_dir() -> Path:
    # Windows: TEMP/TMP、その他: /tmp など
    for k in ("TEMP", "TMP"):
        v = os.environ.get(k)
        if v:
            p = Path(v)
            if p.exists():
                return p
    return Path(os.getenv("TMPDIR", "/tmp"))


def preset_file_path() -> Path:
    env = os.environ.get(ENV_PRESET_FILE)
    if env:
        return Path(env).expanduser()
    if PRESET_FILE:
        return Path(PRESET_FILE).expanduser()
    return (_default_temp_dir() / DEFAULT_PRESET_BASENAME)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "presets": presets}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def preset_summary(cfg: Dict[str, Any]) -> str:
    """一覧表示用のサマリを生成（短め）。"""
    layout = cfg.get("layout", "?")
    select_mode = cfg.get("select_mode", "random")
    shuffle = cfg.get("full_shuffle", False)
    zip_en = cfg.get("zip_scan", True)

    parts: List[str] = []
    parts.append(str(layout))
    if layout.startswith("mosaic"):
        rows = cfg.get("rows")
        cols = cfg.get("cols")
        uniq = cfg.get("count")
        if rows:
            parts.append(f"rows={rows}")
        if cols:
            parts.append(f"cols~{cols}")
        if uniq:
            parts.append(f"uniq~{uniq}")
    elif layout == "grid":
        r = cfg.get("rows")
        c = cfg.get("cols")
        if r and c:
            parts.append(f"{r}x{c}")
    elif layout == "hex":
        parts.append(f"count={cfg.get('count', '?')}")
        if cfg.get("hex_orient"):
            parts.append(str(cfg.get("hex_orient")))

    if shuffle:
        parts.append("shuffle=ON")
    else:
        prof = cfg.get("profile")
        if prof:
            parts.append(f"profile={prof}")
        if prof == "diagonal":
            d = cfg.get("diag_dir")
            if d:
                parts.append(f"diag={d}")
        if cfg.get("opt_enable"):
            parts.append(f"steps={cfg.get('steps')}")
            parts.append(f"reheats={cfg.get('reheats')}")
            if layout.startswith("mosaic"):
                parts.append(f"k={cfg.get('k')}")

    parts.append(f"mode={select_mode}")
    parts.append(f"zip={'ON' if zip_en else 'OFF'}")
    return " | ".join(parts)


def print_preset_list(presets: List[Dict[str, Any]]) -> None:
    print(f"{tr('preset_title')}: {len(presets)}")
    for i, p in enumerate(presets, 1):
        name = str(p.get("name", "(no name)"))
        created = str(p.get("created_at", ""))
        cfg = p.get("config") if isinstance(p.get("config"), dict) else {}
        summ = preset_summary(cfg)
        if created:
            print(f"  [{i}] {name}  ({created})")
        else:
            print(f"  [{i}] {name}")
        print(f"       {summ}")




def find_preset_by_name(presets: List[Dict[str, Any]], name: str) -> Optional[int]:
    for i, p in enumerate(presets):
        if str(p.get("name", "")) == name:
            return i
    return None


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


def pick_core_file(here: Path) -> Path:
    """同じフォルダの kana_wallpaper_unified_final.py を選択します。"""
    core = here / "kana_wallpaper_unified_final.py"
    if not core.exists():
        raise FileNotFoundError(tr("err_no_core"))
    return core


def set_if(mod: Any, name: str, value: Any) -> None:
    """互換のため、存在に関係なくセットする（本体が参照するものだけ効く）。"""
    setattr(mod, name, value)


def set_many(mod: Any, names: List[str], value: Any) -> None:
    for n in names:
        setattr(mod, n, value)


# =============================================================================
# 設定収集（UI）
# =============================================================================


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
    layouts = [tr("layout_grid"), tr("layout_hex"), tr("layout_mosaic_h"), tr("layout_mosaic_w"), tr("layout_random")]
    d = 1
    if default in layouts:
        d = layouts.index(default) + 1
    chosen = ask_choice(tr("layout"), layouts, default_index=d, allow_back=True)
    return BACK_TOKEN if chosen == BACK_TOKEN else str(chosen)


def choose_select_mode(default: str) -> str:
    modes = ["random", "aesthetic", "recent", "oldest", "name_asc", "name_desc"]
    d = 1
    if default in modes:
        d = modes.index(default) + 1
    chosen = ask_choice(tr("select_mode"), modes, default_index=d, allow_back=True)
    if chosen == BACK_TOKEN:
        return BACK_TOKEN
    return str(chosen).lower()


def choose_profile(default: str) -> str:
    choices = [
        tr("arr_diag"),
        tr("arr_hilb"),
        tr("arr_scatter"),
    ]
    mapping = {
        tr("arr_diag"): "diagonal",
        tr("arr_hilb"): "hilbert",
        tr("arr_scatter"): "scatter",
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
    keys: List[str] = ["layout"]

    layout = str(cfg.get("layout", "grid"))

    if layout == "grid":
        keys += ["rows", "cols"]
    elif layout == "hex":
        keys += ["count", "hex_orient"]
    elif layout == "mosaic-uniform-height":
        keys += ["rows"]
    elif layout == "mosaic-uniform-width":
        keys += ["cols"]

    keys += ["select_mode", "full_shuffle", "zip_scan"]

    full_shuffle = bool(cfg.get("full_shuffle", False))
    if (not full_shuffle) and layout in ("grid", "hex", "mosaic-uniform-height", "mosaic-uniform-width"):
        keys += ["profile"]
        if str(cfg.get("profile", "diagonal")) == "diagonal":
            keys += ["diag_dir"]

        keys += ["opt_extra"]
        if str(cfg.get("opt_mode", "default")) == "tune":
            keys += ["steps"]  # reheats（再加熱）は steps の直後に質問する
            if layout.startswith("mosaic"):
                keys += ["k"]

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


def build_config(core_aspect: float, defaults: Optional[Dict[str, Any]] = None, *, start_at_last: bool = False):
    """手動設定（または編集）で設定辞書を作る。戻る対応。"""
    d = defaults or {}

    cfg: Dict[str, Any] = {}

    # 既存値を引き継ぎ（戻るで再表示するときのデフォルト用）
    for k in (
        "layout",
        "rows",
        "cols",
        "count",
        "hex_orient",
        "select_mode",
        "full_shuffle",
        "zip_scan",
        "profile",
        "diag_dir",
        "opt_enable",
        "opt_mode",
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
    cfg.setdefault("zip_scan", True)
    cfg.setdefault("profile", "diagonal")
    cfg.setdefault("diag_dir", "tl_br")

    cfg.setdefault("opt_enable", bool(d.get("opt_enable", False)))
    cfg.setdefault("opt_mode", "tune" if bool(d.get("opt_enable", False)) else "default")
    cfg.setdefault("steps", int(d.get("steps", 40000)))
    cfg.setdefault("reheats", int(d.get("reheats", 2)))
    cfg.setdefault("k", int(d.get("k", 8)))

    # 既存値からの派生値を先に整形（mosaic 推定など）
    _recalc_derived(cfg, core_aspect)

    idx = 0
    if start_at_last:
        try:
            steps0 = _get_step_keys(cfg)
            if steps0:
                idx = max(0, len(steps0) - 1)
        except Exception:
            idx = 0

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
            idx += 1
            continue

        if key == "rows":
            dv = int(cfg.get("rows", d.get("rows", 5)))
            v = ask_int(tr("rows"), dv, 1, 2000, allow_back=True)
            if v == BACK_TOKEN:
                idx -= 1
                continue
            cfg["rows"] = int(v)
            _recalc_derived(cfg, core_aspect, show_mosaic_est=(cfg.get("layout") == "mosaic-uniform-height"))
            idx += 1
            continue

        if key == "cols":
            dv = int(cfg.get("cols", d.get("cols", 13)))
            v = ask_int(tr("cols"), dv, 1, 2000, allow_back=True)
            if v == BACK_TOKEN:
                idx -= 1
                continue
            cfg["cols"] = int(v)
            _recalc_derived(cfg, core_aspect, show_mosaic_est=(cfg.get("layout") == "mosaic-uniform-width"))
            idx += 1
            continue

        if key == "count":
            dv = int(cfg.get("count", d.get("count", 65)))
            v = ask_int(tr("hex_count"), dv, 1, 20000, allow_back=True)
            if v == BACK_TOKEN:
                idx -= 1
                continue
            cfg["count"] = int(v)
            idx += 1
            continue

        if key == "hex_orient":
            v = choose_hex_orient(str(cfg.get("hex_orient", d.get("hex_orient", "col-shift"))))
            if v == BACK_TOKEN:
                idx -= 1
                continue
            cfg["hex_orient"] = str(v)
            idx += 1
            continue

        if key == "select_mode":
            v = choose_select_mode(str(cfg.get("select_mode", d.get("select_mode", "random"))))
            if v == BACK_TOKEN:
                idx -= 1
                continue
            cfg["select_mode"] = str(v)
            idx += 1
            continue

        if key == "full_shuffle":
            v = ask_onoff(tr("shuffle"), default_on=bool(cfg.get("full_shuffle", False)), allow_back=True)
            if v == BACK_TOKEN:
                idx -= 1
                continue
            cfg["full_shuffle"] = bool(v)
            idx += 1
            continue

        if key == "zip_scan":
            v = ask_onoff(tr("zip"), default_on=bool(cfg.get("zip_scan", True)), allow_back=True)
            if v == BACK_TOKEN:
                idx -= 1
                continue
            cfg["zip_scan"] = bool(v)
            idx += 1
            continue

        if key == "profile":
            v = choose_profile(str(cfg.get("profile", d.get("profile", "diagonal"))))
            if v == BACK_TOKEN:
                idx -= 1
                continue
            cfg["profile"] = str(v)
            idx += 1
            continue

        if key == "diag_dir":
            v = choose_diag_dir(str(cfg.get("diag_dir", d.get("diag_dir", "tl_br"))))
            if v == BACK_TOKEN:
                idx -= 1
                continue
            cfg["diag_dir"] = str(v)
            idx += 1
            continue

        if key == "opt_extra":
            default_tune = (str(cfg.get("opt_mode", "default")) == "tune")
            v = choose_opt_extra(default_tune)
            if v == BACK_TOKEN:
                idx -= 1
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
            idx += 1
            continue

        if key == "steps":
            dv = int(cfg.get("steps", d.get("steps", 40000)))
            v = ask_int(tr("steps"), dv, 1000, 200000, allow_back=True)
            if v == BACK_TOKEN:
                idx -= 1
                continue
            cfg["steps"] = int(v)
            # steps を訊いた直後に reheats も続けて訊く（質問の流れを一定にするため）
            while True:
                dv_r = int(cfg.get("reheats", d.get("reheats", 2)))
                # reheats は 0 も有効値のため、Back(0) と衝突しないよう選択メニューにする
                choices_r = [str(i) for i in range(0, 7)]
                default_index_r = min(max(dv_r, 0), 6) + 1
                sel_r = ask_choice(tr("reheats"), choices_r, default_index=default_index_r, allow_back=True)
                if sel_r == BACK_TOKEN:
                    # steps 入力へ戻る
                    break
                cfg["reheats"] = int(sel_r)
                # 次のステップへ進む
                sel_r = None
                break
            if sel_r == BACK_TOKEN:
                continue
            idx += 1
            continue

        # reheats は steps の直後にインラインで質問して流れを統一する
        # （この単独ステップは互換のため残すが、通常は使わない）
        if key == "reheats":
            idx += 1
            continue
            cfg["reheats"] = int(sel)
            idx += 1
            continue

        if key == "k":
            dv = int(cfg.get("k", d.get("k", 8)))
            v = ask_int(tr("k"), dv, 3, 24, allow_back=True)
            if v == BACK_TOKEN:
                idx -= 1
                continue
            cfg["k"] = int(v)
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


def apply_config_to_core(mod: Any, cfg: Dict[str, Any], dd_paths: List[str]) -> None:
    """互換重視で、複数の候補変数名をセットする。"""
    layout = cfg.get("layout", "mosaic-uniform-height")
    select_mode = cfg.get("select_mode", "random")
    full_shuffle = bool(cfg.get("full_shuffle", False))
    zip_scan = bool(cfg.get("zip_scan", True))
    profile = cfg.get("profile", "diagonal")
    diag_dir = cfg.get("diag_dir", "tl_br")
    opt_enable = bool(cfg.get("opt_enable", False))
    steps = int(cfg.get("steps", 40000))
    reheats = int(cfg.get("reheats", 2))
    k = int(cfg.get("k", 8))

    # ドラッグ＆ドロップ入力を本体へ渡す（互換のため複数名）
    if dd_paths:
        set_many(mod, ["SCAN_ROOTS", "SCAN_DIRS", "INPUT_PATHS", "INPUT_ROOTS", "DEFAULT_FOLDERS"], dd_paths)

    # 共通
    set_many(mod, ["LAYOUT_STYLE"], layout)
    set_many(mod, ["SELECT_MODE"], select_mode)
    set_many(mod, ["ARRANGE_FULL_SHUFFLE"], full_shuffle)
    set_many(mod, ["ZIP_SCAN_ENABLE"], zip_scan)

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
    else:
        # random（ランダムレイアウト）
        cand = cfg.get("random_candidates")
        if isinstance(cand, list) and cand:
            set_many(mod, ["RANDOM_LAYOUT_CANDIDATES", "RANDOM_LAYOUTS"], cand)

    # フルシャッフルONなら配置・最適化はOFF方向へ
    if full_shuffle:
        # 「完全ランダム」は ARRANGE_FULL_SHUFFLE で担保するため、
        # 各レイアウトの後処理／最適化（グラデーション／散らし／焼きなまし等）は明示的に無効化します。
        set_many(mod, ["MOSAIC_ENHANCE_ENABLE", "GRID_ENHANCE_ENABLE", "HEX_ENHANCE_ENABLE"], False)
        set_many(mod, ["MOSAIC_LOCAL_OPT_ENABLE", "GRID_ANNEAL_ENABLE", "HEX_ANNEAL_ENABLE"], False)
        # core(v5+) では hex が mosaic の順序を継承しないよう明示（フルシャッフル尊重）
        set_many(mod, ["HEX_GLOBAL_ORDER"], "none")
        set_many(mod, ["HEX_GLOBAL_OBJECTIVE"], "min")
        set_many(mod, ["HEX_LOCAL_OPT_ENABLE"], False)
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
    else:
        set_many(mod, [
            "MOSAIC_LOCAL_OPT_ENABLE", "MOSAIC_ANNEAL_ENABLE", "MOSAIC_OPT_ENABLE",
            "GRID_ANNEAL_ENABLE", "GRID_OPT_ENABLE",
            "HEX_ANNEAL_ENABLE", "HEX_OPT_ENABLE",
            "HEX_LOCAL_OPT_ENABLE",
        ], False)


# =============================================================================
# 実行
# =============================================================================


def run_core(mod: Any) -> None:
    if hasattr(mod, "main") and callable(getattr(mod, "main")):
        mod.main()
        return
    if hasattr(mod, "run") and callable(getattr(mod, "run")):
        mod.run()
        return
    raise RuntimeError("本体に main() / run() が見つかりません。")


def main() -> None:
    here = Path(__file__).resolve().parent
    dd_paths = parse_dragdrop_paths(sys.argv)

    core_path = pick_core_file(here)
    mod = load_core_module(core_path)

    # aspect 計算（mosaic 推定用）
    w = int(getattr(mod, "WIDTH", 3840))
    h = int(getattr(mod, "HEIGHT", 2160))
    aspect = (w / h) if h > 0 else 1.777777

    ppath = preset_file_path()
    presets = load_presets(ppath)

    print(tr("title"))
    print(tr("core").format(core=core_path.name))
    print(tr("presets").format(path=str(ppath)))

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
    print(tr("cache").format(path=cache_path))
    if dd_paths:
        print(tr("input_paths").format(n=len(dd_paths)))

    while True:
        # 起動方法
        mode = ask_choice(
            tr("mode"),
            [tr("mode_manual"), tr("mode_preset"), tr("mode_random_preset")],
            default_index=1,
            allow_back=False,
        )
        if mode == BACK_TOKEN:
            return

        cfg = None

        # プリセット開始
        if mode in (tr("mode_preset"), tr("mode_random_preset")):
            if not presets:
                print(tr("preset_none"))
                cfg = build_config(aspect, defaults=None)
                if cfg == BACK_TOKEN:
                    continue
            else:
                if mode == tr("mode_random_preset"):
                    chosen = random.choice(presets)
                    name = str(chosen.get("name", ""))
                    print(tr("preset_random_pick").format(name=name))
                    cand_cfg = chosen.get("config") if isinstance(chosen.get("config"), dict) else {}
                else:
                    print_preset_list(presets)
                    idx = ask_int(tr("preset_pick"), 1, 1, len(presets), allow_back=True)
                    if idx == BACK_TOKEN:
                        continue
                    chosen = presets[int(idx) - 1]
                    cand_cfg = chosen.get("config") if isinstance(chosen.get("config"), dict) else {}

                # このまま実行 / 編集して実行 / 戻る
                act = ask_choice(
                    tr("preset_action"),
                    [tr("preset_action_run"), tr("preset_action_edit")],
                    default_index=1,
                    allow_back=True,
                )

                if act == BACK_TOKEN:
                    continue
                if act == tr("preset_action_run"):
                    cfg = dict(cand_cfg)
                else:
                    cfg = build_config(aspect, defaults=dict(cand_cfg))
                    if cfg == BACK_TOKEN:
                        continue

        else:
            # 手動
            cfg = build_config(aspect, defaults=None)
            if cfg == BACK_TOKEN:
                continue

        # 保存（任意）
        while True:
            save_choice = ask_choice(
                tr("preset_save"),
                [tr("preset_save_yes"), tr("preset_save_no")],
                default_index=2,
                allow_back=True,
            )

            if save_choice == BACK_TOKEN:
                # 直前の質問（設定の末尾）へ戻る
                cfg2 = build_config(aspect, defaults=dict(cfg), start_at_last=True)
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
                    item = {
                        "name": name,
                        "created_at": now,
                        "summary": preset_summary(cfg),
                        "config": cfg,
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
        apply_config_to_core(mod, cfg, dd_paths)
        print("\n" + tr("run") + "\n")
        try:
            run_core(mod)
        except Exception:
            tb = traceback.format_exc()
            log_path = here / "kana_wallpaper_launcher_error.log"
            try:
                log_path.write_text(tb, encoding="utf-8")
            except Exception:
                pass
            print(tb)
            print(tr("err_log").format(path=str(log_path)))
            raise

        return


if __name__ == "__main__":
    main()
