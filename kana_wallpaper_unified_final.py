# -*- coding: utf-8 -*-
r"""
Kana Wallpaper - Unified FINAL
===========================================================
大量画像をタイル状に敷き詰めて壁紙を生成し、Windows の壁紙に設定します。

実行方法
--------
- ダブルクリック：既定フォルダ群（サブフォルダ含む）を走査
- ドラッグ&ドロップ / CLI：複数フォルダを引数で渡せます（サブフォルダ含む）
  例) py -3 kana_wallpaper_unified_final.py "D:\Pictures" "E:\Photos"
- 保存・ログ関係オプション（任意）
  --img-dir <dir>       出力画像の保存先
  --img-name <name>     出力画像のベース名
  --log-dir <dir>       ログの保存先
  --image / --no-image  作成画像の保存 ON/OFF
  --logs  / --no-logs   使用リスト等の保存 ON/OFF

注意
----
- 本スクリプトはコンソールの Unicode/ANSI に配慮しています（docstring、幅計算、色付けなど）。
- 外部依存は Pillow（PIL）推奨。numpy はあると "spectral-hilbert" が高精度になりますが、無くても自動で簡易版に落ちます。
"""

# ============================================================
# 目次 / Sections（整理整頓パス v30）
#   - 依存関係（import / optional imports）
#   - 小物ユーティリティ（安全な互換・フォールバック）
#   - キャッシュ（dHash / タイル / 顔検出など）
#   - レイアウト（grid / mosaic / hex）
#   - 色順・最適化（spectral / hilbert / annealing）
#   - レンダリング（貼り込み・マスク・効果・保存）
#   - エントリーポイント（main）
# ============================================================
# 設定インデックス（ざっくり早見表）
#
# ランチャーがセットするキーと同名のものが多いです（env / 直書き両対応）。
# 迷子になったら「この表 → 章見出し → 変数名検索」の順で辿るのが早いです。
#
# ■ 入力・収集
#   - IMG_DIRS / INPUT_DIRS        : 画像探索フォルダ（サブフォルダ含む）
#   - ZIP_SCAN_ENABLE              : ZIP内画像を候補に含める（on/off）
#
# ■ 抽出（SELECT_MODE）と重複近似排除
#   - SELECT_MODE                  : random / aesthetic / recent / oldest / name_asc / name_desc
#   - SELECT_RANDOM_DEDUP          : Trueで近似重複排除（dHash + Hamming）
#   - DEDUPE_HAMMING               : 近似判定の許容距離（小さいほど厳しい）
#   - DHASH_CACHE_FILE             : 永続dHashキャッシュ（kana_wallpaper.dhash_cache.json）
#
# ■ レイアウト
#   - LAYOUT_STYLE                 : grid / hex / mosaic-uniform-height / mosaic-uniform-width / random
#   - ROWS / COLS                  : grid系（タイル枚数の目安）
#   - HEX_TIGHT_ORIENT             : hexの詰め方（row-shift/col-shift）
#
# ■ 配置（色順）/ ポスト処理（mosaic系）
#   - ORDER_MODE / GLOBAL_ORDER    : avgLAB / spectral-hilbert など（色順の作り方）
#   - MOSAIC_POST                  : diagonal / hilbert / scatter（mosaicの“見せ方”）
#   - MOSAIC_DIAG_DIR              : diagonalの向き（tl_br など）
#
# ■ 最適化（焼きなまし / 近傍k）
#   - *_LOCAL_OPT_ENABLE           : 近傍最適化を有効にするか（mosaic/hex/grid など）
#   - *_LOCAL_OPT_STEPS            : 試行回数（大きいほど重いが改善しやすい）
#   - *_LOCAL_OPT_REHEATS          : 再加熱（局所解脱出。増やすと重い）
#   - *_LOCAL_OPT_K                : 近傍候補数（大きいほど探索が広い）
#   - *_ANNEAL_T0 / *_ANNEAL_TEND  : 温度（探索の荒さ→締め）
#
# ■ デバッグ（必要なときだけ）
#   - EXC_PASS_DEBUG               : “握りつぶし例外”を警告表示（デフォFalse）
#   - _FDBG / _FDBG2               : Face-focus / 目検証の統計（ログで出る）
# ============================================================

# ============================================================


from __future__ import annotations
import sys, os, math, time, random, tempfile, csv, json, secrets, textwrap, threading, atexit
import hashlib
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Union, Any, Dict
ImageRef = Union[Path, str]


try:
    import numpy as np  # オプション（無くてもOK）
except Exception:
    np = None

from PIL import Image, ImageOps, ImageFilter, ImageStat, ImageEnhance, ImageDraw

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


# ---------------------------------------------------------------
# 顔フォーカス用デバッグカウンタ（常に定義）
# ---------------------------------------------------------------
# どのレイアウト経路でも NameError を避けるため、モジュールロード時に初期化します。
# カウンタの意味（ログの読み方）
#   _FDBG（顔フォーカスの統計）
#     frontal/profile/upper : Haar 検出で候補になった回数（生ヒット数）
#     saliency/center       : サリエンシー/中心フォールバックを使った回数
#     reject_pos            : 位置条件で除外した回数（例：画面下すぎ等）
#     reject_ratio          : 縦横比など品質条件で除外した回数
#     errors                : OpenCV 処理で例外になった回数
#   _FDBG2（目検証・低品質扱いの統計）
#     eyes_ok/eyes_ng       : 目検証（strict_eyes）の成否回数
#     low_reject            : low-quality 扱いで除外した回数（allow_low 等）
_FDBG: Dict[str, Any] = {"cv2": None, "frontal":0, "profile":0, "upper":0, "saliency":0, "center":0,
                         "reject_pos":0, "reject_ratio":0, "errors":0}
_FDBG2: Dict[str, Any] = {"eyes_ok":0, "eyes_ng":0, "low_reject":0}

# ---------------------------------------------------------------
# キャッシュ抑止（__pycache__を作らない）
# ---------------------------------------------------------------
sys.dont_write_bytecode = True

# ---------------------------------------------------------------
# 例外握りつぶし箇所のデバッグ（既定は無音）
# ---------------------------------------------------------------
# 既存コードには「except Exception: pass」が多数あります。
# ふだんは無音のまま動作を変えず、必要なときだけ原因を追えるようにします。
try:
    EXC_PASS_DEBUG
except NameError:
    EXC_PASS_DEBUG = False  # True のとき、握りつぶした例外を 1 回だけ警告表示

_EXC_PASS_WARNED = set()

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
        print(f"[WARN] swallowed exception at {key}: {e}")
    except Exception as _e:
        # ここで例外を出すと本末転倒なので、無音で戻る
        return


# ---------------------------------------------------------------
# 既定の対象フォルダ（ダブルクリック起動時に走査）
#   複数指定OK。サブフォルダも含めて走査します。
# ---------------------------------------------------------------
# 既定の入力フォルダ（ダブルクリック時に走査）
# - 環境に合わせて変更してください（ドラッグ＆ドロップ指定も可能）
DEFAULT_TARGET_DIRS = [
    r".\images",
]

# ---------------------------------------------------------------
# 保存とログ（どこに何を残すか）
#   画像/リスト/統計などを Temp ではなく任意の場所に保存可能。
#   * 壁紙セットだけして画像を残さない、も選べます。
# ---------------------------------------------------------------
_P = Path  # 型注釈用の別名（互換のため残しています）

TEMP_DIR = _P(tempfile.gettempdir())

IMAGE_SAVE_DIR: _P = TEMP_DIR      # 出力画像の保存先（変更可）
IMAGE_BASENAME: str = "kana_wallpaper_current"
LOG_SAVE_DIR:   _P = TEMP_DIR      # 使用画像リスト等の保存先

SAVE_IMAGE:    bool = True         # 生成画像自体を保存するか
SAVE_ARTIFACTS:bool = False         # 使用リスト・統計CSVなどの副産物を保存するか
DELETE_OLD_WHEN_DISABLED: bool = False  # 上2つを False にした時、古いファイルを消すか

# ---------------------------------------------------------------
# 出力キャンバス（完成画像の土台）
#   壁紙として最終的に生成される “キャンバス” の大きさや見た目。
# ---------------------------------------------------------------

# =============================================================================
# セクション: グローバル設定（キャンバス/基本パラメータ）
# =============================================================================
WIDTH, HEIGHT = 3840, 2160         # キャンバスサイズ（px）
MARGIN        = 0                  # 外周の余白（px）
GUTTER        = 1                  # 画像と画像の間隔（px）
FORMAT        = "png"              # "png" か "jpg"
BG_COLOR      = "#000000"          # 背景色（FIT時の余白やモザイクの隙間に見える色）

# レイアウトスタイルを指定します。
# - "mosaic-uniform-height" : 行の高さを一定にして横方向に詰めるモザイク
# - "mosaic-uniform-width"  : 列の幅を一定にして縦方向に詰めるモザイク
# - "grid"                  : 固定グリッド（ROWS×COLS）に均等配置
# - "hex"                   : 正六角・フラットトップ ハニカム充填
# - "random"                : 上記候補からランダムに選択
LAYOUT_STYLE = "random"

# random の候補に含めるレイアウト（必要に応じて編集）
RANDOM_LAYOUT_CANDIDATES = [
    "mosaic-uniform-height",
    "mosaic-uniform-width",
    "grid",
    "hex",
]

# Grid レイアウト用の行数と列数（目安）。ROWS×COLS 枚を上限として使用します。
ROWS, COLS = 5, 13
COUNT      = ROWS * COLS

# ---------------------------------------------------------------
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

def _cover_rect_center(im: Image.Image, cw: int, ch: int) -> Image.Image:
    """
    画像を (cw, ch) を「覆う」ように等倍スケールし、中央を切り出します。

    - アスペクト比は維持（引き伸ばしを避ける）
    - 入力が不正（幅/高さが 0 など）の場合は、指定サイズの空画像を返します
    """
    try:
        iw, ih = im.size
    except Exception:
        # 画像サイズが取得できない場合は黒いキャンバスを返す
        return Image.new("RGB", (max(1, cw), max(1, ch)), color=(0, 0, 0))
    # 不正な入力は空画像で返す
    if iw <= 0 or ih <= 0 or cw <= 0 or ch <= 0:
        return Image.new(im.mode if hasattr(im, 'mode') else "RGB", (max(1, cw), max(1, ch)))

    sc = max(float(cw) / iw, float(ch) / ih)

    new_w = max(1, int(math.ceil(iw * sc)))
    new_h = max(1, int(math.ceil(ih * sc)))
    try:
        scaled = im.resize((new_w, new_h), Resampling.LANCZOS)
    except Exception:
        # LANCZOS が失敗したら BILINEAR で代替
        scaled = im.resize((new_w, new_h), Resampling.BILINEAR)

    left = max(0, (new_w - cw) // 2)
    top = max(0, (new_h - ch) // 2)
    right = left + cw
    bottom = top + ch
    return scaled.crop((left, top, right, bottom))

# === エフェクト ===
# ハレーション（Bloom）エフェクトを追加するかどうか。
# 元画像とガウシアンぼかし版を合成し、明るい部分をふんわり光らせる処理を行います。
# HALATION_INTENSITY と HALATION_RADIUS の値によって、強さと広がりを調整します。
try: HALATION_ENABLE
except NameError: HALATION_ENABLE = False
try: HALATION_INTENSITY
except NameError: HALATION_INTENSITY = 0.30
try: HALATION_RADIUS
except NameError: HALATION_RADIUS = 18

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
# ランダムノイズを重ね、レトロな質感を与えます。
# 強度は GRAIN_AMOUNT で 0.0〜1.0 の範囲で調整します（値が大きいほどノイズが強くなります）。
try: GRAIN_ENABLE
except NameError: GRAIN_ENABLE = False
try: GRAIN_AMOUNT
except NameError: GRAIN_AMOUNT = 0.15  # 粒状ノイズの強さ（0.0〜1.0目安）

# 彩度ブースト（ビブランス）エフェクトを追加するかどうか。True にすると
# 彩度を高める処理を行い、鮮やかな印象を与えます。VIBRANCE_FACTOR で調整可能。
try:
    VIBRANCE_ENABLE
except NameError:
    VIBRANCE_ENABLE = False
try:
    VIBRANCE_FACTOR
except NameError:
    VIBRANCE_FACTOR = 1.0  # 1.0 が無調整、1.30 なら約30%彩度アップ

try: VIGNETTE_ENABLE
except NameError: VIGNETTE_ENABLE = False
try: VIGNETTE_STRENGTH
except NameError: VIGNETTE_STRENGTH = 0.15  # ビネットの強さ（0.0〜1.0目安）
try: VIGNETTE_ROUND
except NameError: VIGNETTE_ROUND = 0.90

# ---------------------------------------------------------------
# 明るさ（背景を無視して自動調整）
#   “黒い余白やボーダー”の影響を受けないよう、貼り付け領域のマスクで統計。
#   - "auto"   : 目標平均輝度に合わせて gain/gamma を自動調整
#   - "manual" : MANUAL_* をそのまま適用
#   - "off"    : 無調整
# ---------------------------------------------------------------
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

        - None / "random" / 変換不能: secrets.randbits(bits) で新しい seed を生成します
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
# ---------------------------------------------------------------
# Grid 近傍色差の最適化（隣接セルの色を似せる/離す）
#   目的(OBJECTIVE):
#     - "max": 色差を最大化 → バラけて“パズル感”
#     - "min": 色差を最小化 → グラデーション/連続感
#   アルゴリズム(GRID_OPTIMIZER):
#     - "hill"             : ヒルクライム（軽い・速い）
#     - "anneal"           : 焼きなまし（局所解脱出に強い／重い）
#     - "checkerboard"     : 明暗で2分割→市松→近傍貪欲
#     - "spectral-hilbert" : 色ベクトル→2D射影→Hilbert 曲線沿い
#     - "spectral-diagonal": 色ベクトルの射影を対角線方向に並べるスペクトル対角モード
# ---------------------------------------------------------------
GRID_NEIGHBOR_OBJECTIVE = "max"    # "max" or "min"
GRID_OPTIMIZER = "anneal"
GRID_DIAGONAL_DIRECTION = "random"   # "random","tlbr","trbl","bltr","brtl"
GRID_DIAGONAL_ZIGZAG   = True

# 焼きなましの調整（重いほど効く）
GRID_ANNEAL_STEPS   = 40000        # 総ステップ数
GRID_ANNEAL_T0      = 1.0          # 初期温度（大きいほど悪化移動も受理しやすい＝探索が広い）
GRID_ANNEAL_TEND    = 1e-3         # 終了温度（小さいほど収束が強い＝仕上げが締まる）
GRID_ANNEAL_REHEATS = 2            # 再加熱回数（0〜3程度）
GRID_ANNEAL_ENABLE = False       # True で“焼きなまし(anneal)”を追加適用（Tune用）

# ---------------------------------------------------------------
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

# ---- Hex フェイスフォーカス設定 ----
FACE_FOCUS_ZOOM_MAX = 1.0           # ← 寄り過ぎなら1.8〜2.0から少し下げる
FACE_FOCUS_MIN_EYE_DIST_FRAC = 0.18 # ← 必要ズームを控えめなら0.18から少し下げる
FACE_FOCUS_ZOOM_MIN = 0.50          # ← さらに“引き”余地を作りたい場合のみ（任意）

# ---------------------------------------------------------------
# Mosaic の最適化（行/列の “詰まり具合” と “並びの色差”）
#   - バランス最適化: 行幅/列高の偏りを下げる（溢れ・バラつき抑制）
#   - 並び色差      : 各行/列の順番を最適化（swap / 2opt）
#   - グローバル順序: 詰める前に全体の順序を整える（色の並びを先に決める）
# ---------------------------------------------------------------
MOSAIC_BALANCE_ENABLE = True       # 行/列のバランス最適化を有効にする

# 行/列内の色差目的（"max"=バラけ、"min"=滑らか）
MOSAIC_NEIGHBOR_OBJECTIVE = "min"
MOSAIC_NEIGHBOR_ITERS_PER_LINE = 200

# 行/列の並び最適化アルゴリズム
#   "swap" / "2opt" / "swap+2opt"（粗→微の二段構え推奨）
MOSAIC_SEQ_ALGO = "swap+2opt"

# バランス最適化の挙動
BALANCE_EARLY_STOP       = True    # 改善停滞で早期終了
BALANCE_RESTART_ON_STALL = True    # 一度だけランダム再スタート

# 詰める前の “全体の並び” を決める（mosaic 系の前処理）
#   "none" / "spectral-hilbert" / "anneal"（フェイスフォーカス時の並び替えモード）
MOSAIC_GLOBAL_ORDER      = "spectral-hilbert"
MOSAIC_GLOBAL_OBJECTIVE  = "min"   # "max" or "min"
MOSAIC_GLOBAL_ITERS      = 40000   # anneal 時の反復量の目安

# モザイクの拡張割り当て（post-pack）
#  - 有効時は先にレイアウトの幾何を決め、その後タイルへ画像を割り当てる
#      タイル位置順（diagonal / Hilbert / checkerboard風）で割り当て、必要なら
#      ローカルk近傍アニールで微調整する。
MOSAIC_ENHANCE_ENABLE     = True   # True: post-pack割り当て（拡張）＋（任意）ローカル最適化を有効化
MOSAIC_ENHANCE_PROFILE    = "diagonal"  # "diagonal" / "hilbert" / "scatter"（旧: "checker"） ※ "off" は旧互換
MOSAIC_DIAGONAL_DIRECTION = "tl_br"  # 対角グラデの向き: "tl_br" / "br_tl" / "tr_bl" / "bl_tr"

MOSAIC_LOCAL_OPT_ENABLE   = True    # 初期配置後のローカル最適化（k近傍アニール）を回す
MOSAIC_LOCAL_OPT_STEPS    = 40000   # 総ステップ数（大きいほど強いが重い）
MOSAIC_LOCAL_OPT_REHEATS  = 2       # 再加熱回数（局所解脱出用。0〜3程度が目安）
MOSAIC_LOCAL_OPT_K        = 8       # k近傍グラフの k（>=3 推奨。大きいほど“広い近傍”を見て重い）
MOSAIC_POS_HILBERT_ORDER  = 10      # 位置ヒルベルトの次数（10 → 1024×1024 解像度相当）


# ------------------------------------------------------------------------
# モザイクレイアウトのギャップレス拡張フラグ
# True にすると、uniform-height / uniform-width のモザイクで行や列を追加画像で拡張し、
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

# ------------------------------------------------------------------------
# 自動補間 / 隙間検出フラグ
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

# ---------------------------------------------------------------
# リサンプル品質（縮小の画質）
#   "default": 直接 LANCZOS（十分高品質。軽め）
#   "hq"     : 段階的 BOX → 最終 LANCZOS（大幅縮小での滲み/モアレ低減）
# ---------------------------------------------------------------
RESAMPLE_MODE = "hq"

# ---------------------------------------------------------------
# 最適化ループのスケール（ヒルクライム/その他で参照）
#   処理時間と相談。値を上げるほど“粘る”。
# ---------------------------------------------------------------
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
# - 選択肢: "first-rank" / "avg-rank" / "avgLAB" / "none"
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

# ===============================================================
# KANA PRE-MAIN PATCH: Hex tiling v8（グリッドエイリアス "hex"、GUTTER 反映、フェイスフォーカス対応）
# - LAYOUT_STYLE="hex" で本モードに切替（内部は grid に再マップ）
# - gap は既定で GUTTER を反映（HEX_TIGHT_GAP=None）／数値指定で上書き可
# - 顔フォーカス切り抜き（OpenCV Haar があれば利用）
# - 端まで敷き詰め（エッジ・クリップ）＆六角マスクで明るさ調整揃え
# ===============================================================
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

# --- HEX 全体/局所最適化（色グラデ） ---
# hex でも grid/mosaic のように「色の並び最適化」を効かせるための設定。
#   HEX_GLOBAL_ORDER（全体の並び）:
#     - "inherit" : MOSAIC_GLOBAL_ORDER を流用
#     - 選択肢: "none" / "spectral-hilbert" / "anneal"
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

# hex 専用: 6近傍（六方向）隣接コストでの局所最適化（焼きなまし）
try: HEX_LOCAL_OPT_ENABLE
except NameError: HEX_LOCAL_OPT_ENABLE = True
try: HEX_LOCAL_OPT_OBJECTIVE
except NameError: HEX_LOCAL_OPT_OBJECTIVE = "inherit"   # "inherit": HEX_GLOBAL_OBJECTIVE を継承 / "min" / "max"
try: HEX_LOCAL_OPT_STEPS
except NameError: HEX_LOCAL_OPT_STEPS = 40000         # 総ステップ数（大きいほど強いが重い）
try: HEX_LOCAL_OPT_REHEATS
except NameError: HEX_LOCAL_OPT_REHEATS = 2           # 再加熱回数（局所脱出用。0〜3程度が目安）
try: HEX_LOCAL_OPT_T0
except NameError: HEX_LOCAL_OPT_T0 = 1.0              # 初期温度（大きいほど悪化移動も受理しやすい）
try: HEX_LOCAL_OPT_TEND
except NameError: HEX_LOCAL_OPT_TEND = 1e-3           # 終了温度（小さいほど収束が強い／貪欲寄り）
try: HEX_LOCAL_OPT_MAX_DEG
except NameError: HEX_LOCAL_OPT_MAX_DEG = 6
try: HEX_LOCAL_OPT_SEED
except NameError: HEX_LOCAL_OPT_SEED = None  # None のときは OPT_SEED を使う
try: FACE_FOCUS_ENABLE
except NameError: FACE_FOCUS_ENABLE = True         # 顔フォーカスを使うか: 顔検出によるクロップを有効にするかどうか。
try: FACE_FOCUS_RATIO
except NameError: FACE_FOCUS_RATIO = 0.42          # 顔の高さがタイルの何割になるよう縮尺調整。0.42なら約4割に顔をフィットさせます。

# 顔中心を上にずらす比率（負値で上寄せ）。-0.10 は上に 10% 移動。
try: FACE_FOCUS_BIAS_Y
except NameError:
    FACE_FOCUS_BIAS_Y = -1.5

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

# Mosaic レイアウトでも顔フォーカスを有効にするかどうか。
# True にすると mosaic 系レイアウトでタイルを貼る際に顔検出に基づいてクロップします。
try:
    MOSAIC_FACE_FOCUS_ENABLE
except NameError:
    MOSAIC_FACE_FOCUS_ENABLE = False

# ===============================================================
# KANA PRE-MAIN PATCH: Persistent HEX v8k+（エイリアス + hex レンダラ + 厳密フェイス + 実行時ゲート）
# ===============================================================
# --- デフォルト値と各種調整パラメータ ---
try: KANA_FORCE_HEX
except NameError:
    # （既定）hex を強制しません。"on" にすると LAYOUT_STYLE に関係なく hex レンダラを使います。
    # grid を優先したい場合や互換性重視なら "off" のままにしてください。
    KANA_FORCE_HEX = "off"
try: FACE_FOCUS_ZOOM_MIN
except NameError: FACE_FOCUS_ZOOM_MIN = 0.5  # 顔フォーカスズームの下限倍率（小さいほど引き気味になります）
try: FACE_FOCUS_ZOOM_MAX
except NameError: FACE_FOCUS_ZOOM_MAX = 1.0  # 顔フォーカスズームの上限倍率（大きいほど寄り気味になります）
try: FACE_FOCUS_ALLOW_LOW
except NameError: FACE_FOCUS_ALLOW_LOW = False  # 顔検出精度が低くてもフォーカス処理を行うか
try: FACE_FOCUS_DEBUG
except NameError: FACE_FOCUS_DEBUG = True  # 顔検出・目検出時のデバッグ情報を表示するか
try: FACE_FOCUS_USE_PROFILE
except NameError: FACE_FOCUS_USE_PROFILE = True  # 横顔用の顔検出器を使用するか
try: FACE_FOCUS_USE_UPPER
except NameError: FACE_FOCUS_USE_UPPER = True  # 上半身検出器を使用するか
try: FACE_FOCUS_USE_SALIENCY
except NameError: FACE_FOCUS_USE_SALIENCY = False  # 顔検出失敗時に視覚的顕著度（saliency）を利用するか
try: FACE_FOCUS_STRICT_EYES
except NameError: FACE_FOCUS_STRICT_EYES = False  # 目検出が成功した場合のみ厳密にズームを適用するか
try: FACE_FOCUS_EYE_MIN
except NameError: FACE_FOCUS_EYE_MIN = 1  # 有効とみなす目検出の最小数
try: FACE_FOCUS_TOP_FRAC
except NameError: FACE_FOCUS_TOP_FRAC = 0.70  # 顔や目検出領域の上部をどれだけ無視するか（割合）
# 顔の縦横比(幅/高さ)の許容レンジ
try: FACE_FOCUS_FACE_RATIO_MIN
except NameError: FACE_FOCUS_FACE_RATIO_MIN = 0.5
try: FACE_FOCUS_FACE_RATIO_MAX
except NameError: FACE_FOCUS_FACE_RATIO_MAX = 2.0

# ---------------------------------------------------------------
# 入力スキャン＆抽出（どの画像を何枚使うか）
#   サブフォルダ含めて集め、好みで“美選抜＋重複除去”をかけられます。
# ---------------------------------------------------------------
RECURSIVE   = True
IMAGE_EXTS  = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff",".jfif"}

# Zip 圧縮ファイル内の画像も候補に含める（.zip / .cbz）
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

# Zip 内エントリを表すキー（内部用）
ZIP_KEY_PREFIX = "zip://"
ZIP_KEY_SEP = "::"


# 抽出方法:
#   - "random"    : 無作為に選ぶ
#   - "aesthetic" : 明瞭度/コントラスト/エッジ等の簡易スコアで並べて上位を選ぶ
# ---------------------------------------------------------------------------
# 抽出モードの既定値
#   - "random"  : 画像リストをシャッフルして先頭から COUNT 枚を選びます。
#   - "aesthetic" : 明瞭度/コントラスト/エッジ等の簡易スコアで並べ替え、上位を選びます。
#   - "recent"    : 更新日時が新しい順に選びます（もっとも最近更新されたファイルが先頭）。
#   - "oldest"    : 更新日時が古い順に選びます。
#   - "name_asc"  : ファイル名の昇順（英数字順）に選びます。
#   - "name_desc" : ファイル名の降順に選びます。
SELECT_MODE = "random"

# ---------------------------------------------------------------------------
# 近似重複の除去設定
SHOW_RANDOM_DEDUP_PROGRESS = False  # 近似重複除去の走査バー表示（分母が大きく分かりづらいので既定OFF）
#   pick_recent() / pick_sorted_generic() は、True の場合 dHash による近似重複除去を行います。
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

# 重複近似排除（dHash の Hamming 距離しきい値）。値を大きくすると「近い画像」まで重複扱いになり、除去が強くなります。
DEDUPE_HAMMING = 15  # 近似重複排除：dHashの許容Hamming距離（大きいほど緩い）

# しきい値の一貫性確保
try:
    DEDUP_DHASH_THRESHOLD  # type: ignore[name-defined]
except NameError:
    DEDUP_DHASH_THRESHOLD = DEDUPE_HAMMING

# ---------------------------------------------------------------------------
# dHash 計算の高速化（永続キャッシュ・先読み）
#   近似重複除去を有効にしたとき、同じ画像に対して dHash を何度も計算すると遅くなるため、
#   ここでは dHash の結果をファイルに保存して再利用できるようにします。
#
#   - 永続キャッシュ: 実行ディレクトリに .dhash_cache.json を自動生成（既定）
#   - 先読み（prefetch）: 近似重複除去の対象になりそうな画像の dHash を前もって計算
#
# 速度にしか影響しない設計にしてあるため、基本は ON 推奨です。
# ---------------------------------------------------------------------------
try: DHASH_CACHE_ENABLE
except NameError: DHASH_CACHE_ENABLE = True      # dHash 永続キャッシュを使う
try: LAB_CACHE_ENABLE
except NameError: LAB_CACHE_ENABLE = True       # 平均LabベクトルをdHashキャッシュに保存して再利用（checkerboard/spectral等の高速化）
try: FACE_CACHE_ENABLE
except NameError: FACE_CACHE_ENABLE = True      # 顔/上半身検出結果をdHashキャッシュに保存して再利用（face focus高速化）
try: DHASH_CACHE_FILE
except NameError: DHASH_CACHE_FILE = os.path.join(tempfile.gettempdir(), "kana_wallpaper.dhash_cache.json")  # dHash キャッシュ保存先（既定=Temp）
try: DHASH_CACHE_MAX
except NameError: DHASH_CACHE_MAX = 200000      # キャッシュの上限（超えたら古いものから間引き）

try: DHASH_PREFETCH_ENABLE
except NameError: DHASH_PREFETCH_ENABLE = True   # dHash 先読みを使う
try: DHASH_PREFETCH_WORKERS
except NameError: DHASH_PREFETCH_WORKERS = max(1, min(8, (os.cpu_count() or 4)))  # スレッド数
try: DHASH_PREFETCH_AHEAD
except NameError: DHASH_PREFETCH_AHEAD = 0       # 先読み対象の枚数（0=自動。大きすぎると無駄が増えます）

# ---------------------------------------------------------------
# 表示（進捗・コンソールUI）
#   派手さと見やすさをコントロール。Unicode 罫線が崩れる環境では "ascii" を推奨。
# ---------------------------------------------------------------
VERBOSE        = True              # 処理の進捗や統計を表示
PROGRESS_EVERY = 0                 # 何枚ごとにバーを更新するか
PROGRESS_WIDTH = 40                # 進捗バーの横幅（文字数）
# 進捗バーの更新間隔。0.0 に設定すると時間ベースの制御を無効にし、
# PROGRESS_EVERY のステップ数のみで更新します。
# これにより処理完了後の一時停止感がなくなります。
PROGRESS_UPDATE_SECS = 0.033         # 進捗バーの更新間隔（秒）
PROGRESS_UPDATE_MODE = "every"  # 進捗更新の方式: "every" 更新間隔ごとに進捗を出力
UI_STYLE       = "ascii"           # "unicode" / "ascii"
UI_LANG = "en"                     # "en" / "ja"
TREAT_AMBIGUOUS_WIDE = True        # East Asian Ambiguous を全角幅扱いにする（日本語向け）
FORCE_UTF8_CP  = False             # Windows コンソールを UTF-8 に切替（chcp 65001）
PROGRESS_BAR_STYLE = "segment"     #"segment"|"paint"

# 進捗バーとバナーに使うネオン調グラデ（RGB, 0-255）
UNICODE_NEON_PALETTE = [
    (255,  85,  85),  # ネオン赤
    (255, 255,  85),  # ネオン黄
    ( 85, 255, 170),  # ネオンミント
    ( 85, 170, 255),  # ネオン空色
    (170,  85, 255),  # ネオン紫
]

# ---- セクション別の Unicode バナー用パレット -------------------
# セクションごとの配色。値は (R,G,B) の配列。自由にカスタム可。
BANNER_PALETTES = {
    "scan":              [( 85,255,170),(170,255, 85),(255,255, 85)],   # スキャン系
    "render-grid":       [(255,170, 85),(255, 85, 85),(255,170, 85)],   # Grid 描画
    "render-mosaic-h":   [( 85,170,255),( 85,255,170),(170,255, 85)],   # Mosaic 高さ均一
    "render-mosaic-w":   [(170, 85,255),( 85,170,255),( 85,255,170)],   # Mosaic 幅均一
    "opt-hill":          [(255,255, 85),(255,170, 85),(255, 85, 85)],   # 近傍最適化（hill）
    "opt-anneal":        [(255, 85, 85),(255,170, 85),(255,255, 85)],   # 焼きなまし
    "opt-checker":       [(255,255,255),(170,170,170),( 85, 85, 85)],   # チェッカーボード
    "opt-spectral":      [( 85,170,255),(170, 85,255),(255, 85,170)],   # スペクトル→Hilbert
    "preprocess":        [( 85,255,170),(170, 85,255),( 85,170,255)],   # 前処理（順序付け）
    "brightness":        [(255,255, 85),(255,170, 85),(255, 85, 85)],   # 明るさ調整
    "done":              [( 85,255,170),( 85,170,255),(170, 85,255)],   # 完了
    "default":           [(255, 85, 85),(255,255, 85),( 85,255,170),( 85,170,255),(170, 85,255)],  # 既定
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

# ---------------------------------------------------------------
# 乱数シード（再現性）
#   "random" で毎回変える／整数で固定再現可。
# ---------------------------------------------------------------
SHUFFLE_SEED = "random"            # 画像シャッフル用
OPT_SEED     = "random"            # 最適化（ヒルクライム等）用。0 か "random"

# ===============================================================
# コンソールユーティリティ
# ===============================================================
ANSI_OK = False
UI = {"style":"unicode","emoji":False,"ansi":False}

def _get_codepage():
    if os.name != "nt": return 65001
    try:
        import ctypes
        return ctypes.windll.kernel32.GetConsoleOutputCP()
    except Exception:
        return 0

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

    NOTE: `str.ljust()` は全角を 1 文字扱いするため、日本語を含むと枠線（| / │）がずれます。
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


# =============================================================================
# セクション: コンソール表示・ログ出力ユーティリティ
# =============================================================================
def banner(title: str):
    if not VERBOSE:
        return
    _ensure_newline_if_bar_active()
    # i18n 済みタイトル
    try:
        t = _tr(title)
    except Exception:
        t = title

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
def _kana_ensure_cv2_cascades():
    try:
        import cv2, os
        base = getattr(cv2, "data", None)
        root = getattr(base, "haarcascades", "") or ""
        def _try(name):
            p = os.path.join(root, name)
            if os.path.isfile(p):
                cc = cv2.CascadeClassifier(p)
                if not cc.empty():
                    return cc
            return None
        g = globals()
        if g.get("CV2_CASCADE_FRONTAL") is None:
            g["CV2_CASCADE_FRONTAL"] = _try("haarcascade_frontalface_alt2.xml") or _try("haarcascade_frontalface_default.xml")
        if g.get("CV2_CASCADE_PROFILE") is None:
            g["CV2_CASCADE_PROFILE"] = _try("haarcascade_profileface.xml")
        if g.get("CV2_CASCADE_UPPER") is None:
            g["CV2_CASCADE_UPPER"] = _try("haarcascade_upperbody.xml")
        if g.get("CV2_CASCADE_EYE") is None:
            g["CV2_CASCADE_EYE"] = _try("haarcascade_eye_tree_eyeglasses.xml") or _try("haarcascade_eye.xml")
        if g.get("CV2_CASCADE_PROFILE") is not None:
            g["FACE_FOCUS_USE_PROFILE"] = True
        if g.get("CV2_CASCADE_UPPER") is not None:
            g["FACE_FOCUS_USE_UPPER"] = True
        return True
    except Exception:
        return False


def note(msg: str):
    _ensure_newline_if_bar_active()
    # 行ごとに翻訳して出力
    for ln in textwrap.dedent(str(msg)).splitlines():
        print("  • " + C("97", _tr(ln)))


# 進捗バーは '\r' で同じ行を書き換えるため、最後に改行しないと次の枠表示が崩れます。
# 現在バー更新中かを記録し、枠/ログ出力前に改行して整列させます。
_BAR_ACTIVE = False

def _ensure_newline_if_bar_active():
    global _BAR_ACTIVE
    if _BAR_ACTIVE:
        print()
        _BAR_ACTIVE = False

def bar(done: int, total: int, prefix: str="", final: bool=False):
    # 進捗更新方式の分岐（秒間隔 or ステップ間隔）
    mode = str(globals().get('PROGRESS_UPDATE_MODE', 'secs')).lower()
    if mode == 'every':
        ev = int(globals().get('PROGRESS_EVERY', 1) or 1)
        global _BAR_LAST_STEP
        try:
            _BAR_LAST_STEP
        except NameError:
            _BAR_LAST_STEP = {}
        key = prefix or 'default'
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
        key = prefix or 'default'
        last = float(_BAR_LAST_TS.get(key, 0.0))
        interval = float(globals().get('PROGRESS_UPDATE_SECS', 0.25))  # 進捗表示の更新間隔（秒）
        if not final and (now - last) < max(0.01, interval):
            return
        _BAR_LAST_TS[key] = now

    if not VERBOSE:
        return
    total = max(1, int(total)); done = max(0, min(int(done), total))
    ratio = done / total
    width = max(10, int(PROGRESS_WIDTH))
    filled = int(width * ratio + 1e-9)
    empty  = width - filled
    pct    = f"{int(ratio*100):>3d}%"

    pal = globals().get("CURRENT_PALETTE") or BANNER_PALETTES["default"]
    bar_core = neon_bar(filled, empty, palette=pal)

    left  = f"{prefix:<6} "
    right = f" {done}/{total} ({pct})"

    if globals().get("UI_STYLE","ascii") == "unicode" and globals().get("UNICODE_BLING", False):
        s = C("97", left) + bar_core + C("97", right)
    else:
        s = f"{left}[{BAR_FILL_CHAR*filled}{BAR_EMPTY_CHAR*empty}]{right}"

    end = "\n" if final else "\r"
    print(s, end=end, flush=True)
    # 最終行が '' のまま残ると表示が崩れるため、状態を記録
    global _BAR_ACTIVE
    _BAR_ACTIVE = (not final)

# --- モードタグ用ヘルパー：一部設定が欠けていても落ちない安全設計 ---
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


def _note_opt_improve_sumdelta(init_sum: float, best_sum: float, objective: str, accepted: int, steps: int, label: str = "ΣΔcolor") -> None:
    """統一表示: 近傍色差（ΣΔcolor）の改善量をまとめて表示する。

    Mosaic の表示形式に合わせて、
      init -> best (Δ=..., ...%) / accepted a/b
    を出します。

    - objective は "min" / "max"（表示の % は「改善量」を正にするため objective を考慮）
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
    # ===== バナー / フェーズ =====
    "スキャン完了": "Scan complete",
    "描画中: Grid": "Rendering (Grid)",
    "描画中: Mosaic / Uniform Height": "Rendering (Mosaic - Uniform Height)",
    "描画中: Mosaic / Uniform Width":  "Rendering (Mosaic - Uniform Width)",
    "最適化: Grid 近傍色差（hill）":        "Optimize: Grid neighbor color (hill)",
    "最適化: Grid 焼きなまし":              "Optimize: Grid annealing",
    "最適化: Grid 市松（checkerboard）":     "Optimize: Grid checkerboard",
    "最適化: Grid スペクトル→ヒルベルト":   "Optimize: Grid spectral→Hilbert",
    "前処理: Mosaic グローバル並び（スペクトル→ヒルベルト）": "Preprocess: Mosaic global order (spectral→Hilbert)",
    "前処理: Mosaic グローバル並び（anneal/hill）":            "Preprocess: Mosaic global order (anneal/hill)",

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

    # ===== 補足 / ラベル =====
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

def _tr(s: str) -> str:
    """UI_LANG == 'en' のとき、日本語メッセージを英語に置換して返します。
    部分一致置換のあと、記号・句読点などを英語 UI 向けに正規化します。
    """
    try:
        s = str(s)
    except Exception:
        return s
    if globals().get("UI_LANG", "ja") != "en":
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
def _mode_label(mode: str) -> str:
    m = str(mode or "").lower()
    # モード名を正規化して表示ラベルを決める
    if m in ("recent", "newest", "mtime", "modified"):
        return _lang("更新順", "Recent")
    if m in ("oldest", "older", "mtime_asc"):
        return _lang("古い順", "Oldest")
    if m in ("name_asc", "name", "filename", "filename_asc"):
        return _lang("名前昇順", "Name-asc")
    if m in ("name_desc", "filename_desc"):
        return _lang("名前降順", "Name-desc")
    if m == "aesthetic":
        return _lang("美選抜", "Aesthetic")
    # 既定はランダム
    return _lang("ランダム", "Random")

# ====== ネオン/グラデ関連（Unicode専用。ASCII時は自動フォールバック） ======
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
    elif "rendering (grid)" in s or ("描画中" in s and "grid" in s):
        key = "render-grid"
    elif "uniform height" in s or "mosaic / uniform height" in s or "高さ" in s:
        key = "render-mosaic-h"
    elif "uniform width" in s or "mosaic / uniform width" in s or "幅" in s:
        key = "render-mosaic-w"
    elif "anneal" in s or "焼きなまし" in s:
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
    # ASCII / Bling 無効時は地味な進捗バー
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

# ===============================================================
# 画像ユーティリティ
# ===============================================================
def parse_color(color: str) -> Tuple[int,int,int]:
    if color.startswith("#"): color=color[1:]
    if len(color)==6 and all(c in "0123456789abcdefABCDEF" for c in color):
        return (int(color[0:2],16), int(color[2:4],16), int(color[4:6],16))
    return Image.new("RGB",(1,1),color).getpixel((0,0))

# ---------------------------------------------------------------
# Zip 画像サポート（zip://<abs_zip_path>::<member>）
#   - 既存コードをなるべく壊さず、Path 互換として「文字列キー」を扱います。
#   - PIL へは BytesIO 経由で渡します（ZipExtFile は seek 非対応のことがあるため）。
# ---------------------------------------------------------------
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
                except Exception:
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
        return int(Path(s).stat().st_size)
    except Exception:
        try:
            return int(os.path.getsize(str(p)))
        except Exception:
            return 0

def _imgref_display(p: ImageRef) -> str:
    s = str(p)
    if _is_zip_key(s):
        return s
    try:
        return str(Path(s).resolve())
    except Exception:
        return s

# ---------------------------------------------------------------------------
# 画像読み込みの安全ガード（巨大画像 / ZIP爆弾 対策）
#   - MAX_IMAGE_PIXELS_LIMIT: 画像の最大ピクセル数（これを超える画像はスキップ）
#   - ZIP_MEMBER_MAX_BYTES  : ZIP内メンバーの展開後サイズ上限（bytes）
#   - ZIP_MEMBER_MAX_RATIO  : ZIPの展開比上限（file_size / compress_size）
# ---------------------------------------------------------------------------
try: MAX_IMAGE_PIXELS_LIMIT
except NameError: MAX_IMAGE_PIXELS_LIMIT = 200_000_000  # 200MP（4K/8Kは余裕、異常に巨大な画像を避ける）
try: ZIP_MEMBER_MAX_BYTES
except NameError: ZIP_MEMBER_MAX_BYTES = 256 * 1024 * 1024  # 256MB（展開後）
try: ZIP_MEMBER_MAX_RATIO
except NameError: ZIP_MEMBER_MAX_RATIO = 300  # 展開比が異常に高いもの（ZIP爆弾対策）

def open_image_safe(p: ImageRef) -> Image.Image:
    """Path または zip:// キーから PIL.Image を安全に開く。

    安全のために以下を行います：
    - 元の Image.open() 由来オブジェクト（ファイルハンドル）を必ず close
    - exif_transpose/convert で別オブジェクトになっても、load()+copy() で完全に分離
    - 巨大画像（MAX_IMAGE_PIXELS_LIMIT）や ZIP 爆弾疑い（ZIP_MEMBER_MAX_BYTES / ZIP_MEMBER_MAX_RATIO）はスキップ
    """
    s = str(p)

    def _postprocess_and_detach(im0: Image.Image) -> Image.Image:
        im = ImageOps.exif_transpose(im0)
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
        ImageDraw.Draw(mask).rectangle([rx,ry,rx+rez.size[0]-1,ry+rez.size[1]-1], fill=255)
    else:
        rez=resize_into_cell(im,w,h,"fill")
        canvas.paste(rez,(x,y))
        ImageDraw.Draw(mask).rectangle([x,y,x+w-1,y+h-1], fill=255)

def hq_resize(img: Image.Image, size: tuple[int,int]) -> Image.Image:
    """大幅縮小に強い高品質リサイズ：段階的に BOX → 最終 LANCZOS。"""
    tw, th = max(1, size[0]), max(1, size[1])
    iw, ih = img.size
    if RESAMPLE_MODE != "hq":
        return img.resize((tw, th), Resampling.LANCZOS)

    # 2倍刻みで近づける
    cur = img
    while iw // 2 >= tw * 1.1 and ih // 2 >= th * 1.1:
        iw //= 2; ih //= 2
        cur = cur.resize((max(1, iw), max(1, ih)), Image.BOX)
    # 軽い最終調整
    return cur.resize((tw, th), Resampling.LANCZOS)

def collect_images(paths: Sequence[ImageRef], recursive: bool=True) -> List[ImageRef]:
    out: List[ImageRef] = []
    seen = 0
    zip_files = 0
    zip_entries = 0

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
                    elif bool(globals().get("ZIP_SCAN_ENABLE", False)) and (suf in ZIP_SCAN_EXTS):
                        if zip_files < int(ZIP_SCAN_MAX_ZIPS):
                            zip_files += 1
                            _iter_zip_members(fp)
                seen += 1
                if VERBOSE:
                    bar(seen, seen + 1, prefix="scan   ")
        elif p.is_file():
            suf = p.suffix.lower()
            if suf in IMAGE_EXTS:
                out.append(p)
            elif bool(globals().get("ZIP_SCAN_ENABLE", False)) and (suf in ZIP_SCAN_EXTS):
                if zip_files < int(ZIP_SCAN_MAX_ZIPS):
                    zip_files += 1
                    _iter_zip_members(p)

    if VERBOSE:
        bar(len(out), len(out), prefix="scan   ", final=True)
    banner("スキャン完了")
    note(f"候補: {len(out)}")
    if bool(globals().get("ZIP_SCAN_ENABLE", False)):
        note(f"zip: {zip_entries}  (zip files opened: {zip_files})")
    return out

def reorder_global_spectral_hilbert(paths: list[Path], objective: str = "min") -> list[Path]:
    """全画像を1列に並べる前処理。色→2D→ヒルベルト順。
       objective="max" の場合は蛇行反転で“バラけ”を少し増やす。"""
    n = len(paths)
    if n <= 1: return paths

    # 進捗：特徴量→射影→ランク
    banner("前処理: Mosaic グローバル並び（スペクトル→ヒルベルト）")
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


def reorder_global_spectral_linear(paths: list[Path], objective: str = "min") -> list[Path]:
    """
    avgLAB を PCA（1次元）で並べ替えて、より素直なグラデ順にします。

        - objective="min": PC1 の昇順（なめらかな遷移）
        - objective="max": 両端から交互（散らし気味）

        注記:
          - reorder_global_spectral_hilbert は“局所連続”を狙うため、
            画像群の分布によっては順序の途中で大きく色がジャンプし、
            ヒルバート配置のときに「真ん中で途切れて見える」ことがあります。
          - ここでは“グラデーションの一貫性”を優先して 1D に落として並べます。
    """

    if not paths:
        return []

    print_box("Preprocess: Mosaic global order (spectral->PCA-linear)")

    # --- 特徴量 ---
    vecs: list[tuple[float, float, float]] = []
    for p in tqdm(paths, desc="feat", total=len(paths)):
        vecs.append(_avg_lab_vector(p))

    # numpy が無い環境でも動くように、最低限のフォールバックを用意
    try:
        import numpy as np  # type: ignore

        X = np.asarray(vecs, dtype=np.float64)
        X0 = X - X.mean(axis=0, keepdims=True)
        # PCA（1次元）
        _, _, VT = np.linalg.svd(X0, full_matrices=False)
        pc1 = VT[0]
        scores = X0 @ pc1
        idx = np.argsort(scores)
        idx = idx.tolist()
    except Exception:
        # フォールバック: L* を主、a*,b* を副でソート
        idx = list(range(len(paths)))
        idx.sort(key=lambda i: (vecs[i][0], vecs[i][1], vecs[i][2]))

    # objective="max" は両端から交互（散らし寄り）
    if str(objective).lower().startswith("max"):
        lo, hi = 0, len(idx) - 1
        zig: list[int] = []
        while lo <= hi:
            zig.append(idx[lo])
            lo += 1
            if lo <= hi:
                zig.append(idx[hi])
                hi -= 1
        idx = zig

    return [paths[i] for i in idx]


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

def reorder_global_anneal(paths: list[Path], objective: str = "max", iters: int = 20000, seed: int | str = 0) -> list[Path]:
    """全画像を1列の並びで最適化（ΣΔ色）。_optimize_sequence を再利用。"""
    n = len(paths)
    if n <= 2:
        return paths

    rng = random.Random(seed if seed != "random" else secrets.randbits(32))
    vecs = [_avg_lab_vector(p) for p in paths]
    order = list(range(n))
    prog = {"i": 0, "n": max(1, int(iters))}

    banner("前処理: Mosaic グローバル並び（anneal/hill）")
    order, ini, fin, imp, acc = _optimize_sequence(order, vecs, int(iters), str(objective).lower(), rng, prog)
    if VERBOSE:
        bar(prog["n"], prog["n"], prefix="opt-col", final=True)

    # 統一表示（mosaic の anneal と同じ形式）
    _note_opt_improve_sumdelta(ini, fin, str(objective).lower(), acc, int(iters))

    return [paths[i] for i in order]
# ===============================================================
# 簡易“美選抜”と重複除去
# ===============================================================
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


# ===============================================================
# dHash（近似ハッシュ）: キャッシュ & 先読み
# ===============================================================

# =============================================================================
# セクション: 永続キャッシュ（dHash / Aesthetic / Lab / Face など）
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
    try:
        return os.path.normcase(os.path.abspath(s))
    except Exception:
        return s

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
    if bool(globals().get("DHASH_CACHE_ENABLE", True)) and h is not None:
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
    parts = [
        "v1",
        f"strict={1 if bool(g.get('FACE_FOCUS_STRICT_EYES', True)) else 0}",
        f"allow_low={1 if bool(g.get('FACE_FOCUS_ALLOW_LOW', True)) else 0}",
        f"ratio_min={float(g.get('FACE_FOCUS_FACE_RATIO_MIN', 0.65)):.4f}",
        f"ratio_max={float(g.get('FACE_FOCUS_FACE_RATIO_MAX', 1.60)):.4f}",
        f"eye_min={int(g.get('FACE_FOCUS_EYE_MIN', 1))}",
        f"min_eye_dist={float(g.get('FACE_FOCUS_MIN_EYE_DIST_FRAC', 0.06)):.4f}",
        f"use_profile={1 if bool(g.get('FACE_FOCUS_USE_PROFILE', True)) else 0}",
        f"use_upper={1 if bool(g.get('FACE_FOCUS_USE_UPPER', False)) else 0}",
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

        out = {"face": None, "upper": None}
        fv = ent.get("face", None)
        uv = ent.get("upper", None)

        if isinstance(fv, (list, tuple)) and len(fv) == 5 and isinstance(fv[0], str):
            try:
                out["face"] = (str(fv[0]), int(fv[1]), int(fv[2]), int(fv[3]), int(fv[4]))
            except Exception:
                out["face"] = None

        if isinstance(uv, (list, tuple)) and len(uv) == 4:
            try:
                out["upper"] = (int(uv[0]), int(uv[1]), int(uv[2]), int(uv[3]))
            except Exception:
                out["upper"] = None

        return out


def _face_cache_put(p, face, upper) -> None:
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

    u_json = None
    if isinstance(upper, tuple) and len(upper) == 4:
        try:
            u_json = [int(upper[0]), int(upper[1]), int(upper[2]), int(upper[3])]
        except Exception:
            u_json = None

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
        ent["t"] = float(time.time())
        _DHASH_CACHE[key] = ent
        _DHASH_CACHE_DIRTY = True


# ---------------------------------------------------------------
# タイル描画メモキャッシュ（メモリ上・レイアウト共通）
#   - 同じ画像を複数回使う場合の描画を高速化する
#       （topped-up / wrap / extend / 重複など）
#   - タイル描画専用で JPEG の "draft" デコードも有効化する。
# ---------------------------------------------------------------
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
            except Exception:
                break

        # バイト数で追い出し
        while _TILE_MEMCACHE_BYTES > max_bytes and len(_TILE_MEMCACHE) > 0:
            try:
                k, v = _TILE_MEMCACHE.popitem(last=False)
                _TILE_MEMCACHE_BYTES -= _tile_memcache_est_bytes(v)
            except Exception:
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
    key = _tile_cache_key(p, cw, ch, mode, use_face_focus)
    hit = _tile_memcache_get(key)
    if hit is not None:
        return hit

    # 描画
    with open_image_safe(p) as im:
        # JPEG：タイル描画専用で小さめにデコード（dhash/aestheticには影響しない）
        try:
            if str(getattr(im, "format", "")).upper() in ("JPEG", "JPG"):
                im.draft("RGB", (max(1, int(cw)), max(1, int(ch))))
        except Exception as e:
            _warn_exc_once(e)
            pass
        rgb = im.convert("RGB")
        if use_face_focus:
            tile = _cover_rect_face_focus(rgb, int(cw), int(ch))
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

    banner("特徴量抽出")

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
                from concurrent.futures import ThreadPoolExecutor, as_completed
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
            bar(i, len(paths), prefix="select ", final=(i == len(paths)))

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
            bar(i, len(feats), prefix="select ", final=(i == len(feats)))

    return [f["path"] for f in uniq[:count]]


# ===============================================================
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
            return str(p).lower()
        except Exception:
            try:
                return str(p)
            except Exception:
                return ""
    # 更新日時を取得する。取得失敗時は epoch=0 で一番古い扱い
    def _key_mtime(p):
        return _imgref_mtime(p)
    # recent / newest / mtime / modified → 降順
    if mode in ("recent", "newest", "mtime", "modified"):
        return sorted(paths, key=_key_mtime, reverse=True)
    # oldest / older / mtime_asc → 昇順
    if mode in ("oldest", "older", "mtime_asc"):
        return sorted(paths, key=_key_mtime)
    # name_desc / filename_desc → 降順
    if mode in ("name_desc", "filename_desc"):
        return sorted(paths, key=_key_name, reverse=True)
    # name_asc / name / filename / filename_asc → 昇順
    if mode in ("name_asc", "name", "filename", "filename_asc"):
        return sorted(paths, key=_key_name)
    # デフォルト: 並べ替え無し
    return paths

# ===============================================================
# 更新順・ソート順抽出関数
#  pick_recent(paths, count, dedupe): 最近のファイルから優先的に選ぶサンプラー
#       更新日時が新しい順に並べ、必要枚数だけ選択。
#       dedupe=True なら近似重複(dHash)を除去しながら選択します。
#  pick_sorted_generic(paths, count, dedupe): 任意のキーでソートして先頭から選ぶ汎用サンプラー
#       sort_by_select_mode() に従って全体を並べ替え、必要枚数だけ選択。
#       dedupe=True なら近似重複(dHash)を除去しながら選択します。

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
    skipped = 0
    scanned = 0
    banner("類似除去 (dHash)")
    # 先読み: ループ中の待ち時間を減らすため、dHash を前もって計算します。
    dhash_prefetch_paths(sorted_paths)
    n = len(sorted_paths)
    for i, p in enumerate(sorted_paths, 1):
        h = dhash64_for_path_cached(p)
        if h is not None and any(hamming(h, hv) <= int(globals().get("DEDUP_DHASH_THRESHOLD", 4)) for hv in hashes):
            pass
        else:
            uniq.append(p)
            if h is not None:
                hashes.append(h)
        if VERBOSE and bool(globals().get("SHOW_RANDOM_DEDUP_PROGRESS", False)):
            bar(i, n, prefix="dedup ", final=False)
        if len(uniq) >= count:
            break
    # 走査バーを出している場合は、ここで改行して締める
    if VERBOSE and bool(globals().get("SHOW_RANDOM_DEDUP_PROGRESS", False)):
        bar(scanned if scanned else n, n, prefix="dedup ", final=True)
    else:
        # 分母が大きいと進捗が 0% に見えやすいので、要点だけ表示
        if VERBOSE:
            if globals().get("UI_LANG", "ja") == "en":
                note(f"Near-duplicate filtering: picked {min(len(uniq), count)}/{count}, scanned {scanned}/{n}, skipped {skipped}")
            else:
                note(f"近似重複除去: 選抜 {min(len(uniq), count)}/{count}、走査 {scanned}/{n}、除外 {skipped}")
    return uniq[:count]

def pick_sorted_generic(paths: list, count: int, dedupe: bool = True) -> list:
    """SELECT_MODE に応じて並べ替えて抽出する。

    sort_by_select_mode() に従って paths を並べ替え、先頭 count 枚を返す。
    dedupe=True なら dHash で近似重複を除去しながら選択。
    """
    sorted_paths = sort_by_select_mode(list(paths))
    if not dedupe:
        return sorted_paths[:count]
    uniq: list = []
    hashes: list = []
    banner("類似除去 (dHash)")
    # 先読み: ループ中の待ち時間を減らすため、dHash を前もって計算します。
    dhash_prefetch_paths(sorted_paths)
    n = len(sorted_paths)
    for i, p in enumerate(sorted_paths, 1):
        h = dhash64_for_path_cached(p)
        if h is not None and any(hamming(h, hv) <= int(globals().get("DEDUP_DHASH_THRESHOLD", 4)) for hv in hashes):
            pass
        else:
            uniq.append(p)
            if h is not None:
                hashes.append(h)
        if VERBOSE: bar(i, n, prefix="select ", final=(i == n))
        if len(uniq) >= count:
            break
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

    # --- 足りない場合は「遠い順」で補充 ---
    topped_up = 0
    if len(uniq) < count:
        need = count - len(uniq)

        sel_keys = {_pkey(x) for x in uniq}
        pool = [p for p in shuffled_paths if _pkey(p) not in sel_keys]

        sel_hashes = [h for h in hashes if h is not None]

        # 候補の「今の集合に対する最小距離」を持たせる（greedy farthest-first）
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

        while need > 0 and cand:
            # 最大 mind を選ぶ（同点は先に出たものを優先）
            best_idx = 0
            best_score = cand[0][2]
            for j in range(1, len(cand)):
                sc = cand[j][2]
                if sc > best_score:
                    best_score = sc
                    best_idx = j

            p, h, _ = cand.pop(best_idx)
            uniq.append(p)
            topped_up += 1
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
                bar(min(len(uniq), count), max(1, count), prefix="dedup ", final=False)


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

    # 表示まとめ
    if show_bar:
        bar(min(len(uniq), count), max(1, count), prefix="dedup ", final=True)
    else:
        # 分母が大きいと進捗が分かりづらいので、要点だけ表示
        if VERBOSE:
            note(_lang(
                f"近似重複除去: 選抜 {min(len(uniq), count)}/{count}、走査 {scanned}/{n}、除外 {skipped}、補充 {topped_up}" + ("、並び分散" if spread_applied else ""),
                f"Near-duplicate filtering: picked {min(len(uniq), count)}/{count}, scanned {scanned}/{n}, skipped {skipped}, topped up {topped_up}" + (", spread order" if spread_applied else ""),
            ))

    return picked


# ===============================================================
# 色ベクトル（LAB 近似）と距離
# ===============================================================
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

# ===============================================================
# Grid 近傍色差：従来ヒルクライム
# ===============================================================
def _grid_edges(rows:int, cols:int):    # レイアウト情報（1回だけ）を表示
    global _PRINTED_LAYOUT_ONCE
    if not _PRINTED_LAYOUT_ONCE:
        try:
            uw = globals().get('MOSAIC_UW_ASSIGN', '(n/a)')
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
# セクション: 最適化（grid/hex/mosaic の近傍目的関数・焼きなまし）
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
    if str(globals().get('UI_LANG','')).lower() == 'en':
        banner("Optimize: Grid neighbor color (hill)")
        obj_label = "maximize (diversify)" if objective == "max" else "minimize (similarize)"
        note(f"Objective: {obj_label} / iterations {iters}")
    else:
        banner("最適化: Grid 近傍色差（hill）")
        note(f"目的: {'最大化(バラけ)' if objective=='max' else '最小化(似せる)'} / 反復 {iters}")
    for t in range(iters):
        a = rnd.randrange(n); b = rnd.randrange(n)
        new_sum  = local_delta_sum(a, b, curr_sum)
        new_cost = cost(new_sum)
        if new_cost <= best_cost:
            curr_sum  = new_sum; best_cost = new_cost; accepted += 1
        else:
            order[a], order[b] = order[b], order[a]
        if VERBOSE: bar(t+1, iters, prefix="opt-col", final=(t+1==iters))

    imp = ((curr_sum-init_sum)/init_sum*100.0) if objective=="max" and init_sum>0 else \
          ((init_sum-curr_sum)/init_sum*100.0) if objective=="min" and init_sum>0 else 0.0
    if str(globals().get('UI_LANG','')).lower() == 'en':
        note(f"ΣΔcolor: {init_sum:.1f} → {curr_sum:.1f} ({imp:+.1f}%) / accepted {accepted}/{iters}")
    else:
        note(f"ΣΔ色: {init_sum:.1f} → {curr_sum:.1f} ({imp:+.1f}%) / 採用 {accepted}/{iters}")

    new_paths = [paths[i] for i in order]
    summary = {"grid_neighbor":{"objective":objective,"initial":init_sum,"final":curr_sum,"improved_pct":imp,"accepted":accepted,"iters":iters}}
    return new_paths, summary

# ===============================================================
# Grid 近傍色差：焼きなまし（Simulated Annealing）
# ===============================================================
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
    Grid: 焼きなましで Σ(隣接セルの色差) を最適化。
    steps  : 総ステップ（大きいほど強い）
    T0/Tend: 温度（指数冷却）
    reheats: 再加熱回数（局所脱出用）
    objective: "max"=バラけ / "min"=似る
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

    # Grid 焼きなまし最適化のバナー/目的を表示します（UI_LANG='en' の場合は英語）。
    if str(globals().get("UI_LANG", "")).lower() == "en":
        banner("Optimize: Grid annealing")
        obj_label = "maximize (diversify)" if objective == "max" else "minimize (similarize)"
        note(f"Objective: {obj_label} / steps={int(steps)} reheats={int(reheats)}")
    else:
        banner("最適化: Grid 焼きなまし")
        note(f"目的: {'最大化(バラけ)' if objective=='max' else '最小化(似せる)'} / steps={int(steps)} reheats={int(reheats)}")

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
                bar(done, steps, prefix="opt-col", final=False)

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
        bar(steps, steps, prefix="opt-col", final=True)

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
# ===============================================================
# Grid：Checkerboard 市松＋貪欲
# ===============================================================
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
    vecs = [_avg_lab_vector(p) for p in paths]
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

    # 市松（checkerboard）最適化のバナーを表示します（UI_LANG='en' の場合は英語）。
    if str(globals().get('UI_LANG','')).lower() == 'en':
        banner("Optimize: Grid checkerboard")
    else:
        banner("最適化: Grid 市松（checkerboard）")
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
            if VERBOSE: bar(done, total, prefix="opt-col", final=(done==total))

    new_paths = [paths[i] for i in order if i is not None]
    summary = {"grid_checkerboard":{"objective":objective,"filled":len(new_paths),"rows":rows,"cols":cols}}
    return new_paths, summary

# ===============================================================
# Grid：Spectral→Hilbert（PCAが無ければLAB簡易2D）
# ===============================================================
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


def reorder_global_spectral_diagonal(paths, objective="min"):
    """全画像を1列に並べる前処理（hex用にも使う）: PCA→2D→対角スイープ。
       objective="max" の場合は端から交互に取って“バラけ”を増やす。"""
    n = len(paths)
    if n <= 1:
        return list(paths)

    try:
        banner("前処理: Hex/Global（スペクトル→対角）")
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

    keys = []
    for u, v, i in uv:
        uu = (u - umin) / du
        vv = (v - vmin) / dv
        # 対角: uu+vv（同値のときは uu）
        keys.append((uu + vv, uu, i))

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
        banner("前処理: Hex 市松（checkerboard seed）")
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
    """Hex向け: 6近傍の隣接コスト（ΣΔcolor）を局所最適化する（焼きなまし）。
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
        if str(globals().get("UI_LANG", "")).lower() == "en":
            banner("Optimize: Hex 6-neighbor annealing")
            obj_label = "maximize (diversify)" if obj == "max" else "minimize (similarize)"
            note(f"Objective: {obj_label} / steps={int(steps)} reheats={int(reheats)}")
        else:
            banner("最適化: Hex 6近傍 焼きなまし")
            note(f"目的: {'最大化(バラけ)' if obj=='max' else '最小化(似せる)'} / steps={int(steps)} reheats={int(reheats)}")
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
                bar(min(done, steps), steps, prefix="opt-col", final=False)

        if ph < phases - 1:
            order = best_order[:]
            for _ in range(max(1, n // 20)):
                i = rng.randrange(n)
                j = rng.randrange(n)
                order[i], order[j] = order[j], order[i]
            curr_sum = sumdiff_for(order)
            curr_cost = to_cost(curr_sum)

    if VERBOSE:
        bar(steps, steps, prefix="opt-col", final=True)

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
    """Mosaic向け: k近傍の隣接コスト（ΣΔcolor）を局所最適化する（焼きなまし）。
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
        banner(_lang("最適化: Mosaic k近傍 焼きなまし", "Optimize: Mosaic k-neighbor annealing"))
        obj_label = _lang("散らす（多様化）", "maximize (diversify)") if obj == "max" else _lang("近づける（類似化）", "minimize (similarize)")
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
                bar(done, steps, prefix="opt-col ", final=False)
            except Exception as e:
                _warn_exc_once(e)
                pass
        # フェーズ間で再加熱（真の総和を再計算してドリフトを避ける）
        if ph < phases - 1:
            curr_sum = sumdiff_for(order)
            curr_cost = to_cost(curr_sum)

    # 真値を再計算（丸めの錯覚 / 累積ドリフトを回避）
    try:
        bar(steps, steps, prefix="opt-col ", final=True)
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
    # フルシャッフルは Mosaic の post-pack 割り当て（グラデ/散らし）を上書きし、出力をランダムに保つ。
    try:
        if bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
            try:
                prof0 = str(globals().get("MOSAIC_ENHANCE_PROFILE", "")).strip().lower()
            except Exception:
                prof0 = ""
            if prof0 and prof0 not in ("off", "none", "random"):
                if not bool(globals().get("_MOSAIC_FULLSHUFFLE_POST_NOTE_ONCE", False)):
                    try:
                        note(_lang("完全シャッフル有効: MosaicのPOST割当を無効化します", "Full shuffle enabled: disabling Mosaic POST assignment"))
                    except Exception as e:
                        _warn_exc_once(e)
                        pass
                    globals()["_MOSAIC_FULLSHUFFLE_POST_NOTE_ONCE"] = True
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
        banner(_lang("前処理: Mosaic 市松（散らし）", "Preprocess: Mosaic checkerboard-ish (scatter)"))
        assigned = _mosaic_checkerboard_seed_paths(base_paths, centers[:n], k=k, img_order=img_order)
    else:
        objective = "min"
        if prof in ("diagonal", "diag", "grad-diagonal", "gradient-diagonal"):
            banner(_lang("前処理: Mosaic 対角グラデ", "Preprocess: Mosaic diagonal gradient"))
            pos_order = _mosaic_pos_order_diagonal(centers[:n], diag_dir=diag_dir)
        else:
            banner(_lang("前処理: Mosaic ヒルベルトグラデ", "Preprocess: Mosaic Hilbert gradient"))
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
        # 方向は HEX_DIAG_DIR で選択：tlbr / brtl / trbl / bltr
        try:
            diag = str(globals().get("HEX_DIAG_DIR", "tlbr")).lower()
        except Exception:
            diag = "tlbr"
        try:
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
            if centers_d:
                if len(images) > 0 and len(images) < len(centers_d):
                    rep = (len(centers_d) + len(images) - 1) // len(images)
                    images = (list(images) * rep)[:len(centers_d)]

                use_n = min(len(images), len(centers_d))
                sorted_imgs = reorder_global_spectral_diagonal(list(images[:use_n]), objective=hobj)

                xs = [c[0] for c in centers_d[:use_n]]
                ys = [c[1] for c in centers_d[:use_n]]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                dx = (max_x - min_x) if max_x > min_x else 1.0
                dy = (max_y - min_y) if max_y > min_y else 1.0
                xn = [(x - min_x) / dx for x in xs]
                yn = [(y - min_y) / dy for y in ys]

                def score_xy(x, y, mode):
                    if mode == "tlbr":
                        return x + y
                    if mode == "trbl":
                        return (1.0 - x) + y
                    if mode == "bltr":
                        return x + (1.0 - y)
                    if mode == "brtl":
                        return (1.0 - x) + (1.0 - y)
                    return x + y

                scores = [score_xy(x, y, diag) for x, y in zip(xn, yn)]
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
        do_local = bool(globals().get("HEX_LOCAL_OPT_ENABLE", False))
    except Exception:
        do_local = False

    if not do_local:
        return images

    centers = _kana_hex_collect_visible_centers(orient, S, step_x, step_y, margin, width, height, extend, r_used, c_used)
    if not centers:
        return images

    edges, neigh = _hex_neighbor_graph(centers, step_x, step_y, max_deg=int(globals().get("HEX_LOCAL_OPT_MAX_DEG", 6)))

    try:
        steps = int(globals().get("HEX_LOCAL_OPT_STEPS", 40000))
    except Exception:
        steps = 20000
    try:
        reheats = int(globals().get("HEX_LOCAL_OPT_REHEATS", 2))
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
    objective="min": 滑らか / "max": 逆順・蛇行でバラけ。
    """
    n = min(len(paths), rows*cols)
    paths = list(paths[:n])

    banner("最適化: Grid スペクトル→ヒルベルト")

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

# ===============================================================
# Mosaic バランス最適化（強化版）
# ===============================================================


# ===============================================================
# Grid：Spectral→Diagonal Sweep（PCA→2D→対角スイープ）
# ===============================================================
def optimize_grid_spectral_diagonal(paths, rows:int, cols:int, objective:str="min",
                                    diagonal:str="tlbr", zigzag:bool=True):
    try:
        banner("最適化: Grid スペクトル→対角スイープ")
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
    if sp: banner("最適化: Mosaic バランス（行）")
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
    note(f"行バランス: {initial:.1f} → {best:.1f} ({imp:+.1f}%) / 採用 {accepted}/{t+1}")
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
    if sp: banner("最適化: Mosaic バランス（列）")
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

# ===============================================================
# Mosaic 色差ヒルクライム（swap / 2opt）
# ===============================================================
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
                if VERBOSE: bar(prog["i"], prog["n"], prefix="opt-col", final=(prog["i"]==prog["n"]))
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
            if VERBOSE: bar(prog["i"], prog["n"], prefix="opt-col", final=(prog["i"]==prog["n"]))
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
                if VERBOSE: bar(prog["i"], prog["n"], prefix="opt-col", final=(prog["i"]==prog["n"]))
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
            if VERBOSE: bar(prog["i"], prog["n"], prefix="opt-col", final=(prog["i"]==prog["n"]))
    return order, curr, accepted

def optimize_rows_color_neighbors(rows, objective="max", iters_per_line=200, seed=0):
    rng = random.Random(seed if seed!="random" else secrets.randbits(32))
    tot_iters = sum(max(0, iters_per_line if len(r)>1 else 0) for r,_ in rows)
    # 2opt も回すなら加算
    if MOSAIC_SEQ_ALGO in ("2opt","swap+2opt"):
        tot_iters += sum(max(0, iters_per_line if len(r)>1 else 0) for r,_ in rows)
    prog={"i":0,"n":max(1,tot_iters)}
    # Mosaic の行内並び最適化のバナー/目的を表示します（UI_LANG='en' の場合は英語）。
    if str(globals().get('UI_LANG','')).lower() == 'en':
        banner("Optimize: Mosaic color diff (row order)")
        obj_label = "maximize (diversify)" if objective == "max" else "minimize (similarize)"
        note(f"Objective: {obj_label} / per row {iters_per_line}")
    else:
        banner("最適化: Mosaic 色差（行の並び）")
        note(f"目的: {'最大化(バラけ)' if objective=='max' else '最小化(似せる)'} / 行あたり {iters_per_line}")
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
    if VERBOSE: bar(prog["n"], prog["n"], prefix="opt-col", final=True)
    imp=((total_final-total_init)/total_init*100.0) if objective=="max" and total_init>0 else \
        ((total_init-total_final)/total_init*100.0) if objective=="min" and total_init>0 else 0.0
    # 行最適化の合計結果（ΣΔ色と採用数）を UI_LANG に応じて表示します
    if str(globals().get('UI_LANG','')).lower() == 'en':
        note(f"ΣΔcolor(row): {total_init:.1f} → {total_final:.1f} ({imp:+.1f}%) / accepted {total_acc}")
    else:
        note(f"ΣΔ色(行): {total_init:.1f} → {total_final:.1f} ({imp:+.1f}%) / 採用 {total_acc}")
    return rows, {"rows_adj_initial":total_init,"rows_adj_final":total_final,"rows_adj_imp_pct":imp,"objective":objective}

def optimize_cols_color_neighbors(cols, objective="max", iters_per_line=200, seed=0):
    rng = random.Random(seed if seed!="random" else secrets.randbits(32))
    tot_iters = sum(max(0, iters_per_line if len(c)>1 else 0) for c in cols)
    if MOSAIC_SEQ_ALGO in ("2opt","swap+2opt"):
        tot_iters += sum(max(0, iters_per_line if len(c)>1 else 0) for c in cols)
    prog={"i":0,"n":max(1,tot_iters)}
    # Mosaic の列内並び最適化のバナー/目的を表示します（UI_LANG='en' の場合は英語）。
    if str(globals().get('UI_LANG','')).lower() == 'en':
        banner("Optimize: Mosaic color diff (column order)")
        obj_label = "maximize (diversify)" if objective == "max" else "minimize (similarize)"
        note(f"Objective: {obj_label} / per column {iters_per_line}")
    else:
        banner("最適化: Mosaic 色差（列の並び）")
        note(f"目的: {'最大化(バラけ)' if objective=='max' else '最小化(似せる)'} / 列あたり {iters_per_line}")
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
    if VERBOSE: bar(prog["n"], prog["n"], prefix="opt-col", final=True)
    imp=((total_final-total_init)/total_init*100.0) if objective=="max" and total_init>0 else \
        ((total_init-total_final)/total_init*100.0) if objective=="min" and total_init>0 else 0.0
    # 列最適化の合計結果（ΣΔ色と採用数）を UI_LANG に応じて表示します
    if str(globals().get('UI_LANG','')).lower() == 'en':
        note(f"ΣΔcolor(column): {total_init:.1f} → {total_final:.1f} ({imp:+.1f}%) / accepted {total_acc}")
    else:
        note(f"ΣΔ色(列): {total_init:.1f} → {total_final:.1f} ({imp:+.1f}%) / 採用 {total_acc}")
    return cols, {"cols_adj_initial":total_init,"cols_adj_final":total_final,"cols_adj_imp_pct":imp,"objective":objective}

# ===============================================================
# レイアウト（grid / mosaic）
# ===============================================================
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
# セクション: レイアウト生成（grid / mosaic / hex）
# =============================================================================
def layout_grid(images: List[Path], width:int, height:int, margin:int, gutter:int,
                rows:Optional[int], cols:Optional[int], mode:str, bg_rgb:Tuple[int,int,int]):
    """ROWS×COLS の等間隔グリッドに画像を並べるレイアウト関数。

    images: 使用する画像ファイルパスのリスト
    width, height: 出力キャンバスのサイズ
    margin, gutter: 外枠の余白・セル間のすき間
    rows, cols: グリッドの行数・列数
    mode: リサイズモード (cover / contain など)
    bg_rgb: 背景色 (R, G, B)
    """
    # レイアウト情報（1回だけ）を表示 (grid)
    global _PRINTED_LAYOUT_ONCE
    if not _PRINTED_LAYOUT_ONCE:
        try:
            uw = globals().get('MOSAIC_UW_ASSIGN', '(n/a)')
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
            _FDBG = {"cv2": None, "frontal":0, "profile":0, "upper":0, "saliency":0, "center":0, "reject_pos":0, "reject_ratio":0, "errors":0}
            _FDBG2 = {"eyes_ok":0, "eyes_ng":0, "low_reject":0}
        except Exception as e:
            _warn_exc_once(e)
            pass
    # 完全シャッフル（ARRANGE_FULL_SHUFFLE）が有効なら、レイアウト前に images を全体シャッフルします。
    # OPT_SEED が固定値なら再現性あり（"random" なら毎回変化）。
    # この関数では、完全シャッフル時は tempo 並び替え（_tempo_apply）は行いません。
    if images and bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
        try:
            _seed = globals().get("OPT_SEED", None)
            # 乱数器の内部状態に依存しない“ハッシュシャッフル”で、最終集合を一度だけ並べ替える
            hash_shuffle_inplace(images, _seed, salt="grid_fullshuffle")
        except Exception as e:
            _warn_exc_once(e)
            pass
        # シャッフル状態を表示
        try:
            if str(globals().get('UI_LANG','')).lower() == 'en':
                note("Full shuffle enabled")
            else:
                note("完全シャッフル有効")
        except Exception as e:
            _warn_exc_once(e)
            pass
    layout_info={}
    # 完全シャッフルが無効な場合のみ、近傍色差の最適化（選択した方式）を適用します。
    if (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and GRID_NEIGHBOR_OBJECTIVE in ("min","max"):
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
        # 追加で焼きなまし(anneal)を回して、近傍の色差目的をさらに詰めます。
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
    if (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and images:
        try:
            images = _tempo_apply(images)
        except Exception as e:
            _warn_exc_once(e)
            pass
    canvas=Image.new("RGB",(width,height),bg_rgb); mask=Image.new("L",(width,height),0)
    banner("描画中: Grid")
    total=min(len(images), rows*cols); done=0
    y=margin; idx=0
    for r in range(rows):
        x=margin
        for c in range(cols):
            if idx>=len(images): break
            w, h = col_w[c], row_h[r]
            try:
                p = images[idx]
                # 描画段のタイルキャッシュ（同一画像の再利用で効く：topped-up / wrap / extend）
                if mode == "fill" and globals().get("GRID_FACE_FOCUS_ENABLE", False) and globals().get("FACE_FOCUS_ENABLE", True):
                    try:
                        tile = _tile_render_cached(p, w, h, "fill", use_face_focus=True)
                        canvas.paste(tile, (x, y))
                        ImageDraw.Draw(mask).rectangle([x, y, x + w - 1, y + h - 1], fill=255)
                    except Exception:
                        # フォールバック：旧パス
                        with open_image_safe(p) as im:
                            paste_cell(canvas, mask, im, x, y, w, h, mode)
                else:
                    if mode == "fit":
                        tile = _tile_render_cached(p, w, h, "fit", use_face_focus=False)
                        rx = x + (w - tile.size[0]) // 2
                        ry = y + (h - tile.size[1]) // 2
                        canvas.paste(tile, (rx, ry))
                        ImageDraw.Draw(mask).rectangle([rx, ry, rx + tile.size[0] - 1, ry + tile.size[1] - 1], fill=255)
                    else:
                        tile = _tile_render_cached(p, w, h, "fill", use_face_focus=False)
                        canvas.paste(tile, (x, y))
                        ImageDraw.Draw(mask).rectangle([x, y, x + w - 1, y + h - 1], fill=255)
            except Exception as e:
                print(f"[WARN] {images[idx]}: {e}")
            x += w + gutter; idx += 1
            done = min(done + 1, total)
            if VERBOSE:
                bar(done, max(1, total), prefix="draw   ", final=(done == total))
        y += row_h[r] + gutter
    # 画像が 0 枚のときでも進捗バーを確実に閉じる（未定義変数参照を避ける）
    if total == 0:
        bar(done, 1, prefix="draw   ", final=True)
    # face-focus のデバッグ（grid+fill のとき、FACE_FOCUS_DEBUG=True なら検出カウンタを表示）
    try:
        if (mode == "fill" and globals().get("GRID_FACE_FOCUS_ENABLE", False)
                and globals().get("FACE_FOCUS_ENABLE", True)
                and globals().get("FACE_FOCUS_DEBUG", False)):
            note("Face-focus:")
            note("  detectors: frontal={fr} profile={pr} upper={ub} | saliency={sa} center={ce}".format(
                 fr=_FDBG.get("frontal",0), pr=_FDBG.get("profile",0), ub=_FDBG.get("upper",0),
                 sa=_FDBG.get("saliency",0), ce=_FDBG.get("center",0)))
            note("  rejects: pos={rp} ratio={rr} errors={er} | eyes: ok={eo} ng={en} low_reject={lr}".format(
                 rp=_FDBG.get("reject_pos",0), rr=_FDBG.get("reject_ratio",0), er=_FDBG.get("errors",0),
                 eo=_FDBG2.get("eyes_ok",0), en=_FDBG2.get("eyes_ng",0), lr=_FDBG2.get("low_reject",0)))
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
    # 完全シャッフル（ARRANGE_FULL_SHUFFLE）が有効なら、Mosaic の処理前に paths を全体シャッフルします。
    # OPT_SEED が固定値なら再現性あり（"random" なら毎回変化）。
    if bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
        try:
            _seed = globals().get("OPT_SEED", None)
            paths = hash_shuffle(list(paths), _seed, salt="mosaic_uh_fullshuffle")
        except Exception as e:
            _warn_exc_once(e)
            pass
        # シャッフル状態を表示
        try:
            if str(globals().get('UI_LANG','')).lower() == 'en':
                note("Full shuffle enabled")
            else:
                note("完全シャッフル有効")
        except Exception as e:
            _warn_exc_once(e)
            pass
    # まずグローバル順序の並べ替えを適用（この後のパックに効かせるため）。
    # 全シャッフルが有効な場合はスキップします。
    if (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and (not bool(globals().get("MOSAIC_ENHANCE_ENABLE", False))):
        if MOSAIC_GLOBAL_ORDER == "spectral-hilbert":
            paths = reorder_global_spectral_hilbert(list(paths), objective=MOSAIC_GLOBAL_OBJECTIVE)
        elif MOSAIC_GLOBAL_ORDER == "anneal":
            paths = reorder_global_anneal(
                list(paths), objective=MOSAIC_GLOBAL_OBJECTIVE,
                iters=MOSAIC_GLOBAL_ITERS, seed=OPT_SEED
            )

    # （完全シャッフルが無効な場合）グローバル並び替えの後、行詰め込み前に tempo 並び替えを適用します。
    if (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and paths:
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
        # JUSTIFY_MIN_ROW_H / JUSTIFY_MAX_ROW_H の範囲に制限
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
    if (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and (not bool(globals().get("MOSAIC_ENHANCE_ENABLE", False))):
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
            if uh_order_l and uh_order_l not in ('none', 'off', 'false', '0'):
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
    # --------------------------------------------------------------------
    # テンポ並べ替え（post / blend）
    # 事前（pre）でテンポを適用していても、その後の近傍最適化や平均LABソートで
    # 「速い/遅いの交互」が崩れることがあります。
    # stage が post / blend のときは、現在の行順を一度フラット化してテンポ並べ替えを再適用し、
    # その順番で行を再構築して最終結果のテンポ感を守ります。
    try:
        if (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False)) and
            globals().get("ARRANGE_TEMPO_ENABLE", False)):
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
    if globals().get("MOSAIC_GAPLESS_EXTEND", False):
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
            # 供給リストを順番に取り出すイテレータです。末尾に到達したら先頭に戻って再利用します。
            supply_idx = 0
            supply_ar_cache: dict = {}
            def _next_supply():
                nonlocal supply_idx
                if not supply_paths:
                    return None
                psp = supply_paths[supply_idx]
                supply_idx = (supply_idx + 1) % len(supply_paths)
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
            #     MOSAIC_ENHANCE_PROFILE / ローカル最適化パラメータが効いていないように見えていた。
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
            for ridx, (rrow, rhh) in enumerate(ext_rows):
                y_cur = y_positions[ridx]
                # 画面外（上）ならスキップ
                if y_cur + rhh <= margin:
                    done_cnt += len(rrow)
                    continue
                # 画面外（下）に到達したら終了
                if y_cur >= margin + H:
                    break
                x_cur = x_off_global
                for (p_t, ar_t, wj_t) in rrow:
                    # 画面外（左）ならスキップ
                    if x_cur + wj_t <= margin:
                        x_cur += wj_t + gutter
                        done_cnt += 1
                        continue
                    # 画面外（右）ならスキップ
                    if x_cur >= margin + W:
                        done_cnt += 1
                        continue
                    # 水平方向のクリッピング量を計算します
                    l_clip = max(margin - x_cur, 0)
                    r_clip = max((x_cur + wj_t) - (margin + W), 0)
                    v_w = wj_t - l_clip - r_clip
                    # 垂直方向のクリッピング量を計算します
                    t_clip = max(margin - y_cur, 0)
                    b_clip = max((y_cur + rhh) - (margin + H), 0)
                    v_h = rhh - t_clip - b_clip
                    if v_w > 0 and v_h > 0:
                        try:
                            with open_image_safe(p_t) as im_tt:
                                # mosaic：タイル矩形そのものをアス比に合わせて作り、アス比維持（クロップなし・黒帯なし）
                                rez = hq_resize(im_tt, (max(1, wj_t), max(1, rhh)))
                                # 画面外にはみ出した分を切り抜きます
                                if l_clip != 0 or r_clip != 0 or t_clip != 0 or b_clip != 0:
                                    rez = rez.crop((int(l_clip), int(t_clip), int(l_clip + v_w), int(t_clip + v_h)))
                                nx = x_cur + l_clip
                                ny = y_cur + t_clip
                                canvas_ext.paste(rez, (int(nx), int(ny)))
                                ImageDraw.Draw(mask_ext).rectangle([
                                    int(nx), int(ny), int(nx + v_w - 1), int(ny + v_h - 1)
                                ], fill=255)
                        except Exception as ex_draw:
                            print(f"[WARN] {p_t}: {ex_draw}")
                    x_cur += wj_t + gutter
                    done_cnt += 1
                    if VERBOSE:
                        bar(done_cnt, max(1, total_draw), prefix="draw   ", final=False)
            # 描画がクリップされると done_cnt と total_draw が一致しないことがあります。
            # 最後に bar(..., final=True) を呼んで必ず 100% 表示で締めます。
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
    banner("描画中: Mosaic / Uniform Height")
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
                        if globals().get("MOSAIC_FACE_FOCUS_ENABLE", False) and globals().get("FACE_FOCUS_ENABLE", True):
                            try:
                                rez = _cover_rect_face_focus(im, max(1, nw), max(1, h))
                            except Exception:
                                rez = _cover_rect_center(im, max(1, nw), max(1, h))
                        else:
                            rez = _cover_rect_center(im, max(1, nw), max(1, h))
                        canvas.paste(rez, (x, y))
                        ImageDraw.Draw(mask).rectangle([x, y, x + nw - 1, y + h - 1], fill=255)
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
                        # 長方形顔フォーカスを適用できる場合は使用
                        if globals().get("MOSAIC_FACE_FOCUS_ENABLE", False) and globals().get("FACE_FOCUS_ENABLE", True):
                            try:
                                # 顔フォーカスの矩形クロップを試します
                                rez = _cover_rect_face_focus(im, max(1, wj), max(1, h))
                            except Exception:
                                # 顔フォーカスに失敗したら中央クロップにフォールバックします
                                rez = _cover_rect_center(im, max(1, wj), max(1, h))
                        else:
                            # 顔フォーカス無効時は中央クロップを使います
                            rez = _cover_rect_center(im, max(1, wj), max(1, h))
                        canvas.paste(rez, (x, y))
                        ImageDraw.Draw(mask).rectangle([x, y, x + wj - 1, y + h - 1], fill=255)
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
            ls = globals().get('LAYOUT_STYLE', 'grid')
            uw = globals().get('MOSAIC_UW_ASSIGN', '(n/a)')
            _mprof = str(globals().get("MOSAIC_ENHANCE_PROFILE", "off")).strip()
            _mpost = ""
            if _mosaic_enhance_active() and _mprof and _mprof.lower() not in ("off", "none", "random"):
                _mpost = f" | POST: {_mprof}"
            note(f"LAYOUT: mosaic-uniform-width | MOSAIC_UW_ASSIGN: {uw}{_mpost}")
        except Exception as e:
            _warn_exc_once(e)
            pass
        _PRINTED_LAYOUT_ONCE = True
    # フルシャッフルが有効な場合は、全体ソート等の前に paths を完全にシャッフルします。
    # OPT_SEED が指定されていて "random" でないときは決定的（再現性のある）順序になります。
    if bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
        try:
            _seed = globals().get("OPT_SEED", None)
            paths = hash_shuffle(list(paths), _seed, salt="mosaic_uw_fullshuffle")
        except Exception as e:
            _warn_exc_once(e)
            pass
        # フルシャッフル有効の注記を表示
        try:
            if str(globals().get('UI_LANG','')).lower() == 'en':
                note("Full shuffle enabled")
            else:
                note("完全シャッフル有効")
        except Exception as e:
            _warn_exc_once(e)
            pass
    # グローバル順序を先に適用（全体の色の流れを先に決める）
    if (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and (not bool(globals().get("MOSAIC_ENHANCE_ENABLE", False))):
        if MOSAIC_GLOBAL_ORDER == "spectral-hilbert":
            paths = reorder_global_spectral_hilbert(list(paths), objective=MOSAIC_GLOBAL_OBJECTIVE)
        elif MOSAIC_GLOBAL_ORDER == "anneal":
            paths = reorder_global_anneal(list(paths), objective=MOSAIC_GLOBAL_OBJECTIVE,
                                          iters=MOSAIC_GLOBAL_ITERS, seed=OPT_SEED)

    # テンポ並べ替え（有効時）: グローバル順序のあと、列詰め（pack）前に paths を入れ替えます。
    # フルシャッフル時はランダム性を優先するためスキップします。
    if (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and paths:
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
            if (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False))) and (not bool(globals().get("MOSAIC_ENHANCE_ENABLE", False))):
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
    # --------------------------------------------------------------------
    # テンポ並べ替え（post / blend）
    # 列バランス調整や最終整列でテンポ（速い/遅いの交互）が崩れることがあるため、
    # stage が post / blend のときは列内パスをフラット化してテンポ並べ替えを再適用し、
    # 固定列幅 w に合わせて高さを再計算しつつ列を再構築します。
    try:
        if (not bool(globals().get("ARRANGE_FULL_SHUFFLE", False)) and
            globals().get("ARRANGE_TEMPO_ENABLE", False)):
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
    if globals().get("MOSAIC_GAPLESS_EXTEND", False):
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
            # 供給リストを順番に返すイテレータです（尽きたら先頭へ循環）。
            # アスペクト比はキャッシュして、毎回の IO を避けます。
            supply_idx = 0
            supply_ar_cache: dict = {}
            def _next_supply() -> Optional[Tuple[Path, float]]:
                """供給リストから次の (path, aspect_ratio) を返します。末尾まで行ったら先頭に戻ります。"""
                nonlocal supply_idx
                if not supply_paths:
                    return None
                psp = supply_paths[supply_idx]
                supply_idx = (supply_idx + 1) % len(supply_paths)
                ar_sp = supply_ar_cache.get(psp)
                if ar_sp is None:
                    try:
                        with open_image_safe(psp) as im_sp:
                            iw_sp, ih_sp = im_sp.size
                            ar_sp = (iw_sp / float(ih_sp)) if ih_sp > 0 else 1.0
                    except Exception:
                        ar_sp = 1.0
                    supply_ar_cache[psp] = ar_sp
                return (psp, ar_sp)
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

            # グリッド座標に従って、クリッピング無しで全タイルを描画します
            for jdx, ext_col in enumerate(ext_cols):
                x_cur = jdx * (w + gutter)
                y_cur_col = 0
                for (p_t, wj_t, h_t) in ext_col:
                    try:
                        with open_image_safe(p_t) as im_tt:
                            # mosaic：タイル矩形そのものをアス比に合わせて作り、アス比維持（クロップなし・黒帯なし）
                            rez = hq_resize(im_tt, (max(1, wj_t), max(1, h_t)))
                            mosaic_img.paste(rez, (int(x_cur), int(y_cur_col)))
                            ImageDraw.Draw(mosaic_mask).rectangle([int(x_cur), int(y_cur_col), int(x_cur + wj_t - 1), int(y_cur_col + h_t - 1)], fill=255)
                    except Exception as ex_draw:
                        print(f'[WARN] {p_t}: {ex_draw}')
                    y_cur_col += h_t + gutter
                    done_cnt += 1
                    if VERBOSE:
                        bar(done_cnt, max(1, total_draw), prefix='draw   ', final=False)
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
    banner("描画中: Mosaic / Uniform Width")
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
                            if globals().get("MOSAIC_FACE_FOCUS_ENABLE", False) and globals().get("FACE_FOCUS_ENABLE", True):
                                try:
                                    rez = _cover_rect_face_focus(im, max(1, wj), max(1, nh))
                                except Exception:
                                    # 顔フォーカスに失敗したら中央クロップにフォールバックします
                                    rez = _cover_rect_center(im, max(1, wj), max(1, nh))
                            else:
                                # 顔フォーカス無効時は中央クロップします
                                rez = _cover_rect_center(im, max(1, wj), max(1, nh))
                            canvas.paste(rez, (x, y))
                            ImageDraw.Draw(mask).rectangle([x, y, x + wj - 1, y + nh - 1], fill=255)
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
                                # 顔フォーカスが有効な場合は長方形クロップを適用
                                if globals().get("MOSAIC_FACE_FOCUS_ENABLE", False) and globals().get("FACE_FOCUS_ENABLE", True):
                                    try:
                                        rez = _cover_rect_face_focus(im, max(1, wj), max(1, h))
                                    except Exception:
                                        # 顔フォーカスに失敗したら中央クロップにフォールバックします
                                        rez = _cover_rect_center(im, max(1, wj), max(1, h))
                                else:
                                    # 顔フォーカス無効時は中央クロップします
                                    rez = _cover_rect_center(im, max(1, wj), max(1, h))
                                canvas.paste(rez, (x, y))
                                ImageDraw.Draw(mask).rectangle([x, y, x + wj - 1, y + h - 1], fill=255)
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
                                if globals().get("MOSAIC_FACE_FOCUS_ENABLE", False) and globals().get("FACE_FOCUS_ENABLE", True):
                                    try:
                                        rez = _cover_rect_face_focus(im, max(1, wj), max(1, h))
                                    except Exception:
                                        rez = _cover_rect_center(im, max(1, wj), max(1, h))
                                else:
                                    rez = _cover_rect_center(im, max(1, wj), max(1, h))
                                canvas.paste(rez, (x, y))
                                ImageDraw.Draw(mask).rectangle([x, y, x + wj - 1, y + h - 1], fill=255)
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

# ===============================================================
# 明るさ補正（背景マスク込み / None安全）
# ===============================================================
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

# ------------------------------------------------------------------------
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
def _apply_halation_bloom(img, intensity=0.25, radius=18):  # intensity: ハレーション強度 / radius: ぼかし半径
    try:
        from PIL import ImageFilter, ImageChops
    except Exception:
        return img
    if intensity <= 0 or radius <= 0:
        return img
    blur = img.filter(ImageFilter.GaussianBlur(radius=radius))
    inv_mul = ImageChops.multiply(ImageChops.invert(img), ImageChops.invert(blur))
    screen = ImageChops.invert(inv_mul)
    return ImageChops.blend(img, screen, max(0.0, min(1.0, float(intensity))))

def _apply_vignette(img, strength=0.15, roundness=0.9):  # strength: ビネット強度 / roundness: 角の丸さ（大きいほど丸）
    if strength <= 0:
        return img
    from PIL import Image, ImageFilter, ImageChops
    import numpy as np
    w, h = img.size
    cx, cy = w * 0.5, h * 0.5
    rx = max(1.0, cx) * (0.9 + 0.1*roundness)
    ry = max(1.0, cy) * (0.9 + 0.1*(1.0-roundness))
    mw, mh = max(64, w//6), max(64, h//6)
    yy, xx = np.mgrid[0:mh, 0:mw]
    sx = (xx * w / mw - cx) / rx
    sy = (yy * h / mh - cy) / ry
    r = (sx*sx + sy*sy) ** 0.5
    mask_small = np.clip(1.0 - r, 0.0, 1.0) ** 1.8
    mask_small = (mask_small * (1.0 - strength))
    mask = (mask_small * 255).astype("uint8")
    mimg = Image.fromarray(mask).convert("L").resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(radius=ry*0.15))  # ぼかし半径の係数（値を上げると境界が柔らかくなる）
    base = img.convert("RGB") if img.mode != "RGB" else img
    m3 = Image.merge("RGB", (mimg, mimg, mimg))
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

# フィルムグレインエフェクト。白黒ノイズを合成してフィルムの粒子感を再現します。
def _apply_grain(img, amount=0.05):
    """フィルムグレイン（粒子）エフェクトを加えます。

    amount はノイズ強度（0.0〜1.0 推奨）。ノイズ生成に失敗した場合は元画像を返します。
    """
    try:
        from PIL import Image
        w, h = img.size
        # Pillow の effect_noise でノイズ画像（L）を生成します。第二引数は 0〜100 程度の強さ。
        noise = Image.effect_noise((w, h), max(0.0, min(1.0, float(amount))) * 100.0)
        # 入力画像のモードに合わせてチャンネル数を揃える
        if img.mode == "RGB":
            noise_rgb = Image.merge("RGB", (noise, noise, noise))
        elif img.mode == "RGBA":
            noise_rgb = Image.merge("RGBA", (noise, noise, noise, Image.new("L", (w,h), 255)))
        else:
            noise_rgb = noise.convert(img.mode)
        # ノイズをブレンド（amount をそのままブレンド係数として扱う）
        alpha = max(0.0, min(1.0, float(amount)))
        return Image.blend(img, noise_rgb, alpha)
    except Exception:
        return img

# 彩度ブースト（ビブランス）エフェクト。色彩の鮮やかさを強めることで
# Bloom/Halation 後にさらに華やかな印象を持たせます。
def _apply_vibrance(img, factor=1.2):
    """彩度（ビブランス）を上げます。

    factor: 1.0 で変化なし。1.1〜1.5 くらいが控えめで使いやすい目安です。
    失敗した場合は元画像を返します。
    """
    try:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Color(img)
        # factor は 0 以上に丸める（負値は意味が薄いので防ぐ）
        f = max(0.0, float(factor))
        return enhancer.enhance(f)
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
# また、除外理由などの統計を `_FDBG` / `_FDBG2` に加算します。
def _get_focus_candidates(im: Image.Image, src_path=None) -> dict:
    """フォーカスクロップ用の候補（顔/上半身/サリエンシー）を検出して返します。

    戻り値（dict）:
      - face     : ("frontal"|"profile", x, y, w, h) または None
      - upper    : (x, y, w, h) または None（FACE_FOCUS_USE_UPPER=True のときのみ）
      - saliency : (cx, cy) または None（FACE_FOCUS_USE_SALIENCY=True のときのみ）

    検出中に `_FDBG` / `_FDBG2`（除外理由・目検証の成否など）の統計を更新します。
    ただし、実際にどの候補を採用したか（frontal/profile の採用カウントなど）は
    呼び出し側で更新します。

    OpenCV が使えない/失敗した場合は例外を投げず、各要素は None のまま返します。
    """
    # 戻り値を初期化
    result = {"face": None, "upper": None, "saliency": None}
    # 永続キャッシュ（顔/上半身）を先に参照して、重いカスケード検出をできるだけ省略
    try:
        if src_path is not None and bool(globals().get("FACE_CACHE_ENABLE", True)) and bool(globals().get("DHASH_CACHE_ENABLE", True)):
            c = _face_cache_get(src_path)
            if isinstance(c, dict):
                if c.get("face") is not None:
                    result["face"] = c.get("face")
                if c.get("upper") is not None:
                    result["upper"] = c.get("upper")
    except Exception as e:
        _warn_exc_once(e)
        pass
    need_face = (result.get("face") is None)
    need_upper = bool(globals().get("FACE_FOCUS_USE_UPPER", False)) and (result.get("upper") is None)
    try:
        import numpy as _np  # type: ignore
        import cv2  # type: ignore
        # 解析を揃えるため RGB 化して ndarray へ
        rgb = _np.array(im.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # 検出が安定しやすいようコントラストを整える（ヒストグラム平坦化）
        gray = cv2.equalizeHist(gray)
        ih_cv, iw_cv = gray.shape[:2]

        # 検出された顔候補から「使えるもの」だけに絞り込む
        def pick_faces(casc) -> list:
            if not need_face or casc is None:
                return []
            faces = casc.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=(max(24, int(0.08 * min(gray.shape))),) * 2,  # 顔検出の最小サイズ（短辺の8%程度、最小24px）
            )
            valid: list = []
            for (x, y, w, h) in faces if faces is not None else []:
                # アスペクト比が極端なものは除外
                ratio = w / float(h + 1e-6)
                # 顔候補の縦横比（w/h）。誤検出（極端に縦長/横長）を弾くためのチェック。
                # 許容範囲は FACE_FOCUS_FACE_RATIO_MIN / MAX で調整できます。
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
                # 厳密モードでは目検証（失敗したら除外）
                # 目検証は Haar の eye カスケードで簡易チェックします（顔矩形の上側のみ）。
                # FACE_FOCUS_STRICT_EYES=True のときだけ適用し、誤検出を減らす目的です。
                ok, _eyes = _kana_face_eye_verify(gray, (x, y, w, h)) if globals().get(
                    "FACE_FOCUS_STRICT_EYES", True
                ) else (True, 0)
                if not ok:
                    _FDBG2["eyes_ng"] += 1
                    continue
                _FDBG2["eyes_ok"] += 1
                valid.append((x, y, w, h))
            return valid

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
        # 採用：正面優先＋面積が大きいもの
        if faces_all:
            faces_all.sort(key=lambda t: (0 if t[0] == 'frontal' else 1, -(t[3] * t[4])))
            result["face"] = faces_all[0]
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
        # サリエンシー（ラプラシアンの熱マップ）で「目立つ点」を推定
        if globals().get("FACE_FOCUS_USE_SALIENCY", True):
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
                pass        # キャッシュ書き戻し（cv2 が動作した場合のみ）
        try:
            if src_path is not None and (need_face or need_upper):
                _face_cache_put(src_path, result.get("face"), result.get("upper"))
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

# ===============================================================
# 壁紙設定・保存・ログ
# ===============================================================
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
        style: 表示スタイル（"Fill" / "Fit" / "Stretch" / "Center"）
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
            r = i//max(1,cols)+1 if cols else 1
            c = i%max(1,cols)+1 if cols else i
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

# ===============================================================
# メイン
# ===============================================================
_ALLOWED = {"grid","mosaic-uniform-height","mosaic-uniform-width","hex"}

def choose_random_layout(rng: random.Random, candidates: Sequence[str]) -> str:
    valid=[c for c in candidates if str(c) in _ALLOWED]
    if not valid: valid=["mosaic-uniform-height","mosaic-uniform-width","grid","hex"]
    return rng.choice(valid)


# =============================================================================
# セクション: エントリーポイント（main）
# =============================================================================
def main():
    """コマンドライン引数や既定フォルダを解釈し、壁紙生成処理を一通り実行するエントリーポイント。"""
    init_console(); banner("Kana Wallpaper - Unified FINAL")

    roots: List[str] = []
    global SAVE_ARTIFACTS, SAVE_IMAGE, IMAGE_SAVE_DIR, IMAGE_BASENAME, LOG_SAVE_DIR, RECURSIVE
    next_is=None
    for a in sys.argv[1:]:
        if a in ("-h","--help"):
            print("Usage: py -3 kana_wallpaper_unified_final.py [TARGET_DIR ...] [--options]\n"
                  "  --img-dir <dir>   : 出力画像の保存先\n"
                  "  --img-name <name> : 出力画像のベース名\n"
                  "  --log-dir <dir>   : ログの保存先\n"
                  "  --image/--no-image: 生成画像の保存 ON/OFF\n"
                  "  --logs/--no-logs  : 使用リスト等の保存 ON/OFF\n")
            return
        if next_is=="img_dir": IMAGE_SAVE_DIR = Path(a); next_is=None; continue
        if next_is=="log_dir": LOG_SAVE_DIR  = Path(a); next_is=None; continue
        if next_is=="img_name": IMAGE_BASENAME = a;    next_is=None; continue
        if a in ("--no-logs","--no-records"): SAVE_ARTIFACTS=False
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
        targets=[Path(s).resolve() for s in DEFAULT_TARGET_DIRS]
        note("ダブルクリック（既定の複数フォルダ）:")
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
    # スキャン
    all_imgs=collect_images(exists, recursive=RECURSIVE)
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
    seed_used = (SHUFFLE_SEED if isinstance(SHUFFLE_SEED,int) else secrets.randbits(64))
    rng=random.Random(seed_used); note(f"Seed: {seed_used}")
    # 抽出モードを表示（ログの追跡用）
    try:
        note(f"{_lang('抽出モード: ', 'Selection mode: ')}{_mode_label(SELECT_MODE)}")
    except Exception as e:
        _warn_exc_once(e)
        pass
    # 抽出
    mode = str(SELECT_MODE).lower()
    if mode == "aesthetic":
        picked_paths = score_and_pick(all_imgs, COUNT, seed=seed_used)
        banner("美選抜抽出完了"); note(f"選抜: {len(picked_paths)}")
    elif mode in ("recent", "newest", "mtime", "modified"):
        # 更新日時が新しい順に抽出
        picked_paths = pick_recent(all_imgs, COUNT, dedupe=globals().get("SELECT_RECENT_DEDUP", True))
        banner("更新順抽出完了"); note(f"選抜: {len(picked_paths)}")
    elif mode in (
        "oldest", "older", "mtime_asc",
        "name", "filename", "name_asc", "filename_asc",
        "name_desc", "filename_desc"
    ):
        # ソート抽出（古い順・名前順）
        picked_paths = pick_sorted_generic(all_imgs, COUNT, dedupe=globals().get("SELECT_SORT_DEDUP", True))
        banner("並び替え抽出完了"); note(f"選抜: {len(picked_paths)}")
    else:
        # ランダム抽出
        _shuf = hash_shuffle(all_imgs, seed_used, salt="select_random")
        if bool(globals().get("SELECT_RANDOM_DEDUP", False)):
            banner("ランダム抽出（近似重複除去）")
            picked_paths = pick_random_dedup(_shuf, COUNT)
        else:
            picked_paths = _shuf[:COUNT]
        banner("ランダム抽出完了"); note(f"選抜: {len(picked_paths)}")

    # --- Tempo（pre）: 入力順の事前整列 ---
    try:
        if ARRANGE_TEMPO_ENABLE and str(ARRANGE_TEMPO_STAGE).lower() == "pre":
            picked_paths = _arrange_by_tempo(picked_paths, ARRANGE_TEMPO_MODE)
    except Exception as e:
        _warn_exc_once(e)
        pass
    # レイアウト選択
    bg=(0,0,0) if BG_COLOR in ("#000","#000000") else parse_color(BG_COLOR)
    layout_info={"style":LAYOUT_STYLE}
    style=LAYOUT_STYLE.lower(); chosen=None
    if style=="random":
        chosen=choose_random_layout(rng, RANDOM_LAYOUT_CANDIDATES)
    start=time.perf_counter()
    if style=="grid" or (style=="random" and chosen=="grid"):
        # --- Tempo 配置（賑やか/静かの交互） ---
        try:
            if ARRANGE_TEMPO_ENABLE:
                picked_paths = _arrange_by_tempo(picked_paths, ARRANGE_TEMPO_MODE)
        except Exception as e:
            _warn_exc_once(e)
            pass
        canvas, mask, info, r, c = layout_grid(picked_paths, WIDTH, HEIGHT, MARGIN, GUTTER, ROWS, COLS, MODE, bg)
        rows_used, cols_used = r, c
        layout_info.update(info)
    elif style=="hex" or (style=="random" and chosen=="hex"):
        # 六角レイアウトを選択した場合は、ランダムモードでもラッパが動作するように一時的に
        # KANA_FORCE_HEX を "on" に設定します。戻すときは元の値を復元します。
        _old_force_hex = globals().get("KANA_FORCE_HEX", "off")
        try:
            globals()["KANA_FORCE_HEX"] = "on"
            canvas, mask, info, r, c = layout_grid(picked_paths, WIDTH, HEIGHT, MARGIN, GUTTER, ROWS, COLS, MODE, bg)
        finally:
            globals()["KANA_FORCE_HEX"] = _old_force_hex
        rows_used, cols_used = r, c
        layout_info.update(info)
    elif style=="mosaic-uniform-height" or (style=="random" and chosen=="mosaic-uniform-height"):
        canvas, mask, info = layout_mosaic_uniform_height(picked_paths, WIDTH, HEIGHT, MARGIN, GUTTER, bg)
        rows_used, cols_used = 0, 0
        layout_info.update(info)
    else:
        canvas, mask, info = layout_mosaic_uniform_width(picked_paths, WIDTH, HEIGHT, MARGIN, GUTTER, bg)
        rows_used, cols_used = 0, 0
        layout_info.update(info)

    # 明るさ調整（背景マスク込み）
    pre = mean_luma_masked(canvas, mask)    # --- 各種エフェクト適用前の明るさ ---
    # --- Bloom/Halation エフェクト（有効な場合のみ） ---
    try:
        # ハレーション処理は有効なときのみ行います。失敗してもそのまま進行します。
        if bool(globals().get("HALATION_ENABLE", False)):
            canvas = _apply_halation_bloom(canvas, HALATION_INTENSITY, HALATION_RADIUS)
    except Exception as e:
        _warn_exc_once(e)
        pass
    # --- 独立したアートエフェクト群 ---
    # ここから下のエフェクトはハレーションの有無に関係なく適用できます。各フラグを組み合わせて自由に調整してください。
    try:
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
        # フィルムグレインエフェクト
        if bool(globals().get("GRAIN_ENABLE", False)):
            amount = globals().get("GRAIN_AMOUNT", 0.05)
            canvas = _apply_grain(canvas, amount)
    except Exception as e:
        _warn_exc_once(e)
        pass
    canvas, binfo = adjust_brightness_with_mask(canvas, mask)

    # --- Vignette（明るさ調整の後） ---
    try:
        _ = canvas
        if VIGNETTE_ENABLE:
            canvas = _apply_vignette(canvas, VIGNETTE_STRENGTH, VIGNETTE_ROUND)
    except NameError:
        pass
    if pre is not None: binfo["original_mean"]=pre

    banner("明るさ 調整")
    mb = binfo.get("original_mean"); ma = binfo.get("final_mean"); tgt = binfo.get("target")
    method = binfo.get("method") or BRIGHTNESS_MODE
    if mb is None:
        note("調整: なし（対象領域が検出できませんでした）")
    else:
        note(f"平均: {fmt_num(mb)} → {fmt_num(ma)}（目標 {fmt_num(tgt,2)}）")
        if binfo.get("gain")  is not None:  note(f"ゲイン: ×{fmt_num(binfo['gain'])}")
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
    note(f"描画完了: {time.perf_counter()-start:.2f}s → {out_path}")

    # 使用リスト
    write_used_lists(picked_paths, rows_used, cols_used, seed_used, targets, layout_info, binfo, log_dir=Path(LOG_SAVE_DIR))

    # 壁紙更新（コンソール表示はローカライズ対応）
    try:
        set_wallpaper(out_path, style="Fill")
        tag = _mode_tag_for_console()  # Fill/Fit/Uniform Height/Uniform Width を安全に取得
        print(C("92;1", "\n" + _lang(
            f"壁紙を更新しました（{tag}）。お楽しみください！",
            f"Wallpaper updated ({tag}). Enjoy!"
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

# --- LAYOUT エイリアス（gridsafe 有効時は無効化） ---
try:
    _KANA_HEX_ALIAS = False
    _KANA_ORIG_LAYOUT_NAME = str(globals().get('LAYOUT_STYLE',''))
except Exception:
    _KANA_HEX_ALIAS = False
except Exception as e:
    _warn_exc_once(e)
    pass
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
def _solve_S_v8(width:int, height:int, margin:int, N:int, max_cols:int, gap:int, eps:int, orient:str):
    # v7 と同じ「S（タイル基準サイズ）の決定」ロジック（extend で端まで敷き詰める前提）
    best = None
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
        key = (S, -(rows*cols - N))
        if (best is None) or (key > best[0]):
            best = (key, S, rows, cols)
    if best is None:
        rows = N; cols = 1
        denom_h = (1.0 + _SQRT3_2*(rows-1))
        if orient != "row-shift":
            denom_h += 0.5*_SQRT3_2
        avail_h = height - 2*margin - max(0, rows-1)*(gap - eps)
        S = max(8, int(avail_h/max(1e-6, denom_h)))
        return S, rows, cols
    _, S, rows, cols = best
    return S, rows, cols

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

def _kana_layout_grid_hex_tight_v8_wrapper(fn):
    def _wrapped(images, width, height, margin, gutter, rows, cols, mode, bg_rgb):
        # tempo 並べ替え（images が空ならスキップ）
        if images:
            try:
                images = _tempo_apply(images)
            except Exception as e:
                _warn_exc_once(e)
                pass
        # hex エイリアス指定時 or layout が grid のときのみ hex 描画を許可（従来どおり）
        if not globals().get("HEX_TIGHT_ENABLE", True):
            return fn(images, width, height, margin, gutter, rows, cols, mode, bg_rgb)
        if not (globals().get("_KANA_HEX_ALIAS", False) or str(globals().get("LAYOUT_STYLE","grid")).lower()=="grid"):
            return fn(images, width, height, margin, gutter, rows, cols, mode, bg_rgb)
        # エイリアスが有効、または KANA_FORCE_HEX="on" の場合のみ hex モードを使用
        use_hex = globals().get("_KANA_HEX_ALIAS", False) or str(globals().get("KANA_FORCE_HEX","")).lower()=="on"
        if not use_hex:
            # grid要求なら通常gridへ
            return fn(images, width, height, margin, gutter, rows, cols, mode, bg_rgb)

        total = len(images)
        _cur = 0  # FIX(hex): init cursor
        # HEX_TIGHT_GAP 未指定なら gutter 値を gap として使う
        _gap_cfg = globals().get("HEX_TIGHT_GAP", None)
        if _gap_cfg is None:
            gap = int(max(0, float(gutter)))
        else:
            gap = int(max(0, float(_gap_cfg)))
        eps = int(max(0, min(2, float(globals().get("HEX_TIGHT_SEAM_EPS", 0)))))
        orient = str(globals().get("HEX_TIGHT_ORIENT", "col-shift")).lower()
        extend = int(max(0, float(globals().get("HEX_TIGHT_EXTEND", 2))))

        S, r_used, c_used = _solve_S_v8(width, height, margin, total, int(globals().get("HEX_TIGHT_MAX_COLS", 128)), gap, eps, orient)
        # --- 修正(hex)：一部の分岐で代入がスキップされても locals が必ず存在するようにする ---
        try:
            _vis_needed
        except Exception:
            try:
                _vis_needed = int(max(1, (r_used + 2*extend) * (c_used + 2*extend)))
            except Exception:
                _vis_needed = 1
        try:
            _cur
        except Exception:
            _cur = 0

        from PIL import Image
        canvas = Image.new("RGB", (width, height), HEX_TIGHT_BG if HEX_TIGHT_BG is not None else bg_rgb)
        mask_canvas = Image.new("L", (width, height), 0)

        try:
            note(f"レイアウト: hex({orient}) | 行×列: {r_used}×{c_used} | S={S} | gap={gap} | eps={eps} | extend={extend}")
        except Exception: pass
        try:
            _TR_MAP
            banner("Rendering (Hex / Honeycomb)" if str(globals().get("UI_LANG","")).lower()=="en" else "描画中: Hex / Honeycomb")
        except Exception as e:
            _warn_exc_once(e)
            pass
        hexmask = _hexmask_square(S)
        # --- KANA: draw 高速化（タイル生成のメモリキャッシュ + mask 用白タイル再利用） ---
        from collections import OrderedDict
        _tile_cache_max = int(max(0, int(globals().get("HEX_TILE_MEMCACHE_MAX", 512))))
        _tile_cache = OrderedDict()
        _whiteL = Image.new("L", (S, S), 255)

        def _tile_cache_get(k: str):
            if _tile_cache_max <= 0:
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
            if _tile_cache_max <= 0:
                return
            try:
                _tile_cache[k] = v
                _tile_cache.move_to_end(k)
                while len(_tile_cache) > _tile_cache_max:
                    _tile_cache.popitem(last=False)
            except Exception as e:
                _warn_exc_once(e)
                pass
        step_x = 0.75*S + (gap - eps)
        step_y = _SQRT3_2*S + (gap - eps)

        # --- KANA(hex 重複抑止 hard v8): 画像が尽きるまで重複を出さない ---
        def _kana_count_visible(orient_):
            _cnt = 0
            if orient_ == "row-shift":
                _half = S/2.0
                _min_r, _max_r = -extend, r_used + extend
                _min_c, _max_c = -extend, c_used + extend
                for _r in range(_min_r, _max_r):
                    _shift = _half if (_r % 2 != 0) else 0.0
                    _y = margin + int(round(_r*step_y))
                    for _c in range(_min_c, _max_c):
                        _x = margin + int(round(_shift + _c*step_x))
                        if _x + S <= 0 or _y + S <= 0 or _x >= width or _y >= height:
                            continue
                        _cnt += 1
            else:
                _half_v = step_y/2.0
                _min_c, _max_c = -extend, c_used + extend
                _min_r, _max_r = -extend, r_used + extend
                for _c in range(_min_c, _max_c):
                    _shift_y = _half_v if (_c % 2 != 0) else 0.0
                    _x = margin + int(round(_c*step_x))
                    for _r in range(_min_r, _max_r):
                        _y = margin + int(round(_shift_y + _r*step_y))
                        if _x + S <= 0 or _y + S <= 0 or _x >= width or _y >= height:
                            continue
                        _cnt += 1
            return _cnt

        _vis_needed = _kana_count_visible(orient)

        # 可視セル数（画面内に見えるタイル数）に合わせて、
        #   1) パス正規化で重複を除去しつつ images を一意化
        #   2) 足りなければ走査済み全プール（KANA_SCAN_ALL）から補充
        # します。
        import os as _kana_os
        def _kana_norm(p):
            try:
                return _kana_os.path.normcase(_kana_os.path.normpath(p))
            except Exception:
                return p
        _seen = set()
        _unique = []
        for _p in images:
            _k = _kana_norm(_p)
            if _k not in _seen:
                _unique.append(_p); _seen.add(_k)
        try:
            _gpool = globals().get("KANA_SCAN_ALL", []) or []
            # SELECT_MODE に応じて並べ替え（recent/oldest/name_asc/name_desc）
            _gpool = sort_by_select_mode(_gpool)
            for _p in _gpool:
                if len(_unique) >= _vis_needed:
                    break
                _k = _kana_norm(_p)
                if _k not in _seen:
                    _unique.append(_p); _seen.add(_k)
        except Exception as e:
            _warn_exc_once(e)
            pass
        images = _unique
        total = len(images)
        try:
            if str(globals().get("UI_LANG","")).lower() == "en":
                note(f"Dedupe(NORM): unique={total} / visible~{_vis_needed} | wrap={'YES' if total < _vis_needed else 'NO'}")
            else:
                note(f"重複抑止(NORM): 一意={total} / 目安~{_vis_needed} | 巡回={'あり' if total < _vis_needed else 'なし'}")
        except Exception as e:
            _warn_exc_once(e)
            pass
        _cur = 0

        # 完全シャッフル（ARRANGE_FULL_SHUFFLE）が有効なら、
        # 重複除去＆補充が終わった「最終的な集合」に対して一度だけシャッフルします。
        # OPT_SEED が数値かつ "random" でない場合は決定的（再現性あり）になります。
        if bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
            try:
                _seed = globals().get("OPT_SEED", None)
                hash_shuffle_inplace(images, _seed, salt="hex_fullshuffle")
            except Exception as e:
                _warn_exc_once(e)
                pass
            # 完全シャッフル有効の旨を表示
            try:
                if str(globals().get('UI_LANG','')).lower() == 'en':
                    note("Full shuffle enabled")
                else:
                    note("完全シャッフル有効")
            except Exception as e:
                _warn_exc_once(e)
                pass
        idx = 0; done = 0
        if orient == "row-shift":
            half_shift = S/2.0
            min_r = -extend; max_r = r_used + extend
            min_c = -extend; max_c = c_used + extend

            # 内容重複対策：ファイルSHA1先頭64bit + 知覚dHash(64bit)
            import hashlib as _kana_hash
            from PIL import Image as _KIMG

            def _kana_file_hash8(path):
                try:
                    h = _kana_hash.sha1()
                    with open(path, "rb") as f:
                        for chunk in iter(lambda: f.read(1<<20), b""):
                            h.update(chunk)
                    return h.digest()[:8]  # 64bit（先頭8バイト）
                except Exception:
                    return None

            def _kana_dhash64(path):
                try:
                    im = _KIMG.open(path).convert("L").resize((9, 8), _KIMG.LANCZOS)
                    px = list(im.getdata())
                    bits = 0
                    for y in range(8):
                        o = y*9
                        for x in range(8):
                            bits = (bits << 1) | (1 if px[o+x] < px[o+x+1] else 0)
                    return bits
                except Exception:
                    return None

            def _kana_ham64(a, b):
                x = (a ^ b) & ((1<<64)-1)
                return int(x).bit_count()

            def _kana_is_dup(path, seen_exact, seen_dh):
                ex = _kana_file_hash8(path)
                if ex is not None and ex in seen_exact:
                    return True, ex, None
                dh = _kana_dhash64(path)
                if dh is not None:
                    for d in seen_dh:
                        if _kana_ham64(dh, d) <= int(globals().get("DEDUP_DHASH_THRESHOLD", 4)):
                            return True, ex, dh
                return False, ex, dh

            # 1) 現在の images から、内容重複（hash/dHash）を除外して一意リストを作成
            _seen_exact = set()
            _seen_dh = []
            _content_unique = []
            _dup_skipped = 0
            for _p in images:
                _isdup, _ex, _dh = _kana_is_dup(_p, _seen_exact, _seen_dh)
                if _isdup:
                    _dup_skipped += 1
                    continue
                _content_unique.append(_p)
                if _ex is not None: _seen_exact.add(_ex)
                if _dh is not None: _seen_dh.append(_dh)

            # 2) KANA_SCAN_ALL から、内容重複を避けながら不足分を補充
            try:
                _pool_all = (globals().get("KANA_SCAN_ALL", []) or [])
                # SELECT_MODE に応じて並べ替え（recent/oldest/name_asc/name_desc）
                _pool_all = sort_by_select_mode(_pool_all)
            except Exception:
                _pool_all = []
            if len(_content_unique) < _vis_needed and _pool_all:
                for _p in _pool_all:
                    if len(_content_unique) >= _vis_needed:
                        break
                    _isdup, _ex, _dh = _kana_is_dup(_p, _seen_exact, _seen_dh)
                    if _isdup:
                        _dup_skipped += 1
                        continue
                    _content_unique.append(_p)
                    if _ex is not None: _seen_exact.add(_ex)
                    if _dh is not None: _seen_dh.append(_dh)

            images = _content_unique
            total = len(images)
            try:
                if str(globals().get("UI_LANG","")).lower() == "en":
                    note(f"Dedupe(NORM+pHash): unique={total} / visible~{_vis_needed} | wrap={'YES' if total < _vis_needed else 'NO'} | skipped={_dup_skipped}")
                else:
                    note(f"重複抑止(NORM+pHash): 一意={total} / 目安~{_vis_needed} | 巡回={'あり' if total < _vis_needed else 'なし'} | 除外={_dup_skipped}")
            except Exception as e:
                _warn_exc_once(e)
                pass
            _cur = 0

            _cur = 0
            for r in range(min_r, max_r):
                shift = half_shift if (r % 2 != 0) else 0.0
                y = margin + int(round(r*step_y))
                for c in range(min_c, max_c):
                    x = margin + int(round(shift + c*step_x))
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
                    except Exception as e:
                        print(f"[WARN] {images[_cur]}: {e}")
                    if _did:
                        _cur += 1
                    done += 1
                    if VERBOSE: bar(done, (max_r-min_r)*(max_c-min_c), prefix="draw   ", final=False)
        else:
            half_v = step_y/2.0
            min_c = -extend; max_c = c_used + extend
            min_r = -extend; max_r = r_used + extend
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

            for c in range(min_c, max_c):
                shift_y = half_v if (c % 2 != 0) else 0.0
                x = margin + int(round(c*step_x))
                for r in range(min_r, max_r):
                    y = margin + int(round(shift_y + r*step_y))
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
                    except Exception as e:
                        print(f"[WARN] {images[_cur]}: {e}")
                    if _did:
                        _cur += 1
                    done += 1
                    if VERBOSE: bar(done, (max_r-min_r)*(max_c-min_c), prefix="draw   ", final=False)

        bar(max(done, (max_r-min_r)*(max_c-min_c)), (max_r-min_r)*(max_c-min_c), prefix="draw   ", final=True)

        layout_info = {"style":"hex","rows":r_used,"cols":c_used,"S":S,
                       "step_x":step_x,"step_y":step_y,"orient":orient,"extend":extend,"gap":gap}
        return canvas, mask_canvas, layout_info, r_used, c_used
    return _wrapped

# 【KANA修正】ラッパーが二重に適用されないように無効化しました。
_SQRT3_2 = (3 ** 0.5) * 0.5
_HEX_NAMES = ("hex","honeycomb","hex-tight")

# --- インポート時のエイリアス設定（gridsafe 有効時は無効化） ---
_KANA_HEX_ALIAS = False

# --- 幾何計算系ヘルパー ---
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

def _kana_face_eye_verify(gray: Any, rect: Tuple[int, int, int, int]) -> Tuple[bool, int]:
    """顔矩形内で目が検出できるかの簡易チェックです。

    戻り値:
      - eyes_ok    : 目が所定数以上検出できたか
      - eyes_count : 検出された目の数（簡易）
    """
    import cv2
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
    return (eyes >= int(globals().get("FACE_FOCUS_EYE_MIN",1))), eyes

def _cover_square_face_focus(im, S: int, src_path=None):
    """
    S×S の正方形タイルを作るために、画像を「拡大（fill）→クロップ」します。
    顔（正面/横顔）・上半身・サリエンシー（目立つ領域）の候補を `_get_focus_candidates()` で取得し、
    候補があればその中心を基準にズーム倍率とクロップ位置を決めます。
    候補が無い／検出が無効／例外が起きた場合は、中央クロップ（必要ならバイアス込み）にフォールバックします。
    ※デバッグ用に `_FDBG` / `_FDBG2` のカウンタが更新されます。
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

    try:
        # 顔フォーカスが無効なら検出処理は行わずフォールバックへ
        if not bool(globals().get("FACE_FOCUS_ENABLE", True)):
            raise RuntimeError("face focus disabled")
        # 統一ヘルパ `_get_focus_candidates()` で候補（顔/上半身/サリエンシー）を取得します
        cand = _get_focus_candidates(im, src_path)
        face = cand.get("face")
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
            _FDBG["frontal" if name == "frontal" else "profile"] += 1
            return crop_center(im2, cx, cy)
        # 上半身候補があればそれを基準にクロップ（元実装に近い係数で軽くズーム）
        if upper:
            x, y, w, h = upper  # type: ignore
            cx = x + w / 2.0
            cy = y + 0.28 * h  # 上半身の中心より少し上を狙う（経験則）
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
def _cover_rect_face_focus(im: Image.Image, cw: int, ch: int) -> Image.Image:
    """
    cw×ch の矩形タイルを作るために、画像を「拡大（fill）→クロップ」します。
    顔（正面/横顔）・上半身・サリエンシーの候補を `_get_focus_candidates()` で取得し、
    候補があればその中心を基準にズーム倍率とクロップ位置を決めます。
    候補が無い／検出が無効／例外が起きた場合は、中央クロップ（必要ならバイアス込み）にフォールバックします。
    ※デバッグ用に `_FDBG` / `_FDBG2` のカウンタが更新されます。
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

    try:
        # 顔フォーカスが無効な場合は検出せずフォールバックへ
        if not bool(globals().get("FACE_FOCUS_ENABLE", True)):
            raise RuntimeError("face focus disabled")
        if not (bool(globals().get("GRID_FACE_FOCUS_ENABLE", False)) or bool(globals().get("MOSAIC_FACE_FOCUS_ENABLE", False))):
            raise RuntimeError("face focus disabled for this layout")
        # 統一ヘルパ `_get_focus_candidates()` で候補（顔/上半身/サリエンシー）を取得します
        cand = _get_focus_candidates(im)
        face = cand.get("face")
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
            _FDBG["frontal" if name == "frontal" else "profile"] += 1
            return crop_center(im2, cx, cy)
        # 上半身候補: 上半身が収まるよう倍率を調整し、縦位置は少し上寄せのヒューリスティックを使います
        if upper:
            x, y, w, h = upper  # type: ignore
            cx = x + w / 2.0
            cy = y + 0.28 * h
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
                    note("完全シャッフル有効")
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

        from PIL import Image
        canvas = Image.new("RGB", (width, height), HEX_TIGHT_BG if HEX_TIGHT_BG is not None else bg_rgb)
        mask_canvas = Image.new("L", (width, height), 0)
        try: note_lang(f"レイアウト: hex({orient}) | 行x列: {r_used}x{c_used} | S={S} | gap={gap} | eps={eps} | extend={extend}", f"Layout: hex({orient}) | rows×cols: {r_used}x{c_used} | S={S} | gap={gap} | eps={eps} | extend={extend}")
        except Exception: pass
        try:
            banner("Rendering (Hex / Honeycomb)" if str(globals().get("UI_LANG","")).lower()=="en" else "描画中: Hex / Honeycomb")
        except Exception: pass

        hexmask = _hexmask_square(S)
        # --- KANA: draw 高速化（タイル生成のメモリキャッシュ + mask 用白タイル再利用） ---
        from collections import OrderedDict
        _tile_cache_max = int(max(0, int(globals().get("HEX_TILE_MEMCACHE_MAX", 512))))
        _tile_cache = OrderedDict()  # key(str path) -> PIL.Image (SxS RGB)
        _whiteL = Image.new("L", (S, S), 255)

        def _tile_cache_get(k: str):
            if _tile_cache_max <= 0:
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
            if _tile_cache_max <= 0:
                return
            try:
                _tile_cache[k] = v
                _tile_cache.move_to_end(k)
                while len(_tile_cache) > _tile_cache_max:
                    _tile_cache.popitem(last=False)
            except Exception as e:
                _warn_exc_once(e)
                pass
        step_x = 0.75*S + (gap - eps)
        step_y = _SQRT3_2*S + (gap - eps)

        # フェイスフォーカスのデバッグカウンタをリセット
        global _FDBG, _FDBG2
        _FDBG = {"cv2": None, "frontal":0, "profile":0, "upper":0, "saliency":0, "center":0, "reject_pos":0, "reject_ratio":0, "errors":0}
        _FDBG2 = {"eyes_ok":0, "eyes_ng":0, "low_reject":0}

        idx = 0; done = 0
        # --------------------------------------------------------------
        # 内部ヘルパ: 重複除去→不足分補充→（必要なら）シャッフル→tempo 配置
        # row-shift / col-shift どちらでも同じ前処理を行うため、共通化しています。
        def _hex_prepare_images_local(img_list: List[Path], vis_needed: int) -> List[Path]:
            """hex レイアウト用の画像リスト前処理。

            - 同一パスを（順序を保ったまま）重複除去
            - タイル数が足りない場合は、グローバルスキャン結果から未使用画像で補充
            - 設定に応じてシャッフル（完全シャッフル/補充後シャッフル）
            - 可能なら tempo 配置（忙しさ/静けさの並び）を適用

            返り値の順序が、そのままタイル消費順になります。
            """
            images_local = list(img_list)
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
                    if mode_now == "random":
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
                    if globals().get("HEX_TOPUP_INTERLEAVE", False):
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
                    if bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
                        _seed = globals().get("OPT_SEED", None)
                        hash_shuffle_inplace(images_prepared, _seed, salt="hex_fullshuffle")
                        # 通知は何度も出さない（重複表示を避けるフラグ）
                        try:
                            if not globals().get("_FULL_SHUFFLE_NOTE_DONE_HEX", False):
                                if str(globals().get("UI_LANG", "")).lower() == 'en':
                                    note("Full shuffle enabled")
                                else:
                                    note("完全シャッフル有効")
                                globals()["_FULL_SHUFFLE_NOTE_DONE_HEX"] = True
                        except Exception as e:
                            _warn_exc_once(e)
                            pass
                except Exception as e:
                    _warn_exc_once(e)
                    pass
                # 5) 補充が発生した場合、必要ならもう一度シャッフルして位置の偏りを減らします
                if globals().get("HEX_TOPUP_SHUFFLE", True) and topup_count > 0:
                    try:
                        _seed = globals().get("SHUFFLE_SEED", "random")
                        hash_shuffle_inplace(images_prepared, _seed, salt="hex_topup_shuffle")
                    except Exception as e:
                        _warn_exc_once(e)
                        pass
                # 6) 完全シャッフル中でない場合のみ、tempo 配置（忙/静の並び）を適用します（失敗しても無視）。
                try:
                    if images_prepared and not bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
                        images_prepared = _tempo_apply(images_prepared)
                except Exception as e:
                    _warn_exc_once(e)
                    pass
                # 7) 重複除去/補充の結果を、この呼び出しの中で一度だけログ表示します
                try:
                    total_loc = len(images_prepared)
                    wrap = 'YES' if total_loc < vis_needed else 'NO'
                    if str(globals().get("UI_LANG", "")).lower() == 'en':
                        note(f"Dedupe(NORM): unique={total_loc} / visible~{vis_needed} | wrap={wrap}")
                    else:
                        note(f"重複抑止(NORM): 一意={total_loc} / 目安~{vis_needed} | 巡回={'あり' if wrap=='YES' else 'なし'}")
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
                    if VERBOSE: bar(done, (max_r-min_r)*(max_c-min_c), prefix="draw   ", final=False)
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
                if _mode_now == "random":
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
                if globals().get("HEX_TOPUP_INTERLEAVE", False):
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
                if bool(globals().get("ARRANGE_FULL_SHUFFLE", False)):
                    _seed = globals().get("OPT_SEED", None)
                    hash_shuffle_inplace(images, _seed, salt="hex_fullshuffle")
                    # 通知は一度だけ（重複表示を避けるフラグ）
                    try:
                        if not globals().get("_FULL_SHUFFLE_NOTE_DONE_HEX", False):
                            if str(globals().get('UI_LANG','')).lower() == 'en':
                                note("Full shuffle enabled")
                            else:
                                note("完全シャッフル有効")
                            globals()["_FULL_SHUFFLE_NOTE_DONE_HEX"] = True
                    except Exception as e:
                        _warn_exc_once(e)
                        pass
            except Exception as e:
                _warn_exc_once(e)
                pass
            # オプション: 補充後に画像をシャッフルして順序の偏りを減らす
            if globals().get("HEX_TOPUP_SHUFFLE", True) and _topup_count > 0:
                try:
                    _seed = globals().get("SHUFFLE_SEED", "random")
                    hash_shuffle_inplace(images, _seed, salt="hex_topup_shuffle")
                except Exception as e:
                    _warn_exc_once(e)
                    pass
            # tempo 配置（忙/静の並び）は最終リストに適用します（失敗しても無視）。
            if images:
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
                if str(globals().get("UI_LANG","")).lower() == "en":
                    note(f"Dedupe(NORM): unique={total} / visible~{_vis_needed} | wrap={'YES' if total < _vis_needed else 'NO'}")
                else:
                    note(f"重複抑止(NORM): 一意={total} / 目安~{_vis_needed} | 巡回={'あり' if total < _vis_needed else 'なし'}")
            except Exception as e:
                _warn_exc_once(e)
                pass
            _cur = 0
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
                    if VERBOSE: bar(done, (max_r-min_r)*(max_c-min_c), prefix="draw   ", final=False)

        bar(max(done, (max_r-min_r)*(max_c-min_c)), (max_r-min_r)*(max_c-min_c), prefix="draw   ", final=True)
        if globals().get("FACE_FOCUS_DEBUG", True):
            try:
                cv2_state = _FDBG.get("cv2")
                note("Face-focus:")
                note("  detectors: frontal={fr} profile={pr} upper={ub} | saliency={sa} center={ce}".format(
                     fr=_FDBG["frontal"], pr=_FDBG["profile"], ub=_FDBG["upper"],
                     sa=_FDBG["saliency"], ce=_FDBG["center"]))
                note("  rejects: pos={rp} ratio={rr} errors={er} | eyes: ok={eo} ng={en} low_reject={lr}".format(
                     rp=_FDBG["reject_pos"], rr=_FDBG["reject_ratio"], er=_FDBG["errors"],
                     eo=_FDBG2["eyes_ok"], en=_FDBG2["eyes_ng"], lr=_FDBG2["low_reject"]))
            except Exception: pass

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
def note_lang(jp: str, en: str):
    try:
        lang = str(globals().get("UI_LANG", "")).lower()
    except Exception:
        lang = ""
    msg = en if lang == "en" else jp
    try:
        note(msg)
    except Exception:
        print(msg)

if __name__=="__main__":
    main()
