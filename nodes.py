"""
nodes.py  ─  ComfyUI カスタムノード（生成物QA / 破綻検出）

提供ノード:
  - QualityGate        : 1バッチを評価し、集約スコア・合否・内訳レポートを出力。
                         リトライループの判定に使う（passed が False の間だけ再生成）。
  - QualityFilterBatch : バッチから合格画像だけを抜き出して出力。
                         「生成 → QA → 合格分だけ納品」を1ノードで見せる目玉。

設計方針:
  - 検出ロジックは quality_checks/ 側（ComfyUI非依存の純粋関数）に置き、
    ここは ComfyUI の IMAGE テンソル <-> np.uint8(BGR) の変換と配線だけを担う。
  - IMAGE は ComfyUI 慣習どおり torch.Tensor [B,H,W,C] float32 0..1 を想定。
"""

from __future__ import annotations

from typing import List

import numpy as np

try:
    import torch
except Exception:  # ComfyUI 外での import 用フォールバック
    torch = None  # type: ignore

import cv2

from .quality_checks import QAResult, available_checks, run_cascade

# このノードで既定採用するカスケード（順に評価）。
# hands は誤検出のため既定から除外（quality_checks/hands.py 冒頭の注記参照）。
# body_proportion は参照比率が必要なので専用ノード ProportionMatchRank を使う。
DEFAULT_CHECKS = ["face_presence", "sharpness"]


def _tensor_to_bgr_list(images) -> List[np.ndarray]:
    """ComfyUI IMAGE [B,H,W,C] float(0..1) -> list of BGR uint8。"""
    arr = images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images)
    if arr.ndim == 3:
        arr = arr[None, ...]
    out = []
    for im in arr:
        rgb = np.clip(im * 255.0, 0, 255).astype(np.uint8)
        out.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return out


def _bgr_list_to_tensor(bgr_list: List[np.ndarray]):
    """list of BGR uint8 -> ComfyUI IMAGE [B,H,W,C] float(0..1)。"""
    rgbs = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for b in bgr_list]
    stacked = np.stack(rgbs, axis=0)
    if torch is not None:
        return torch.from_numpy(stacked)
    return stacked


def _format_report(per_image: List[dict]) -> str:
    lines = ["# QualityGate report", ""]
    for row in per_image:
        flag = "PASS" if row["passed"] else "FAIL"
        checks = ", ".join(
            (f"{r.name}=skip" if r.skipped else f"{r.name}={r.score:.2f}")
            for r in row["results"]
        )
        lines.append(f"[{flag}] #{row['index']:02d} agg={row['score']:.2f} | {checks}")
        for r in row["results"]:
            lines.append(f"        - {r.name}: {r.detail}")
    passed = sum(1 for r in per_image if r["passed"])
    lines += ["", f"passed {passed}/{len(per_image)}"]
    return "\n".join(lines)


def _evaluate(images, checks: List[str], threshold: float, expected_faces: int):
    """共通評価ルーチン。per_image の dict リストを返す。"""
    params = {"face_presence": {"expected": int(expected_faces)}}
    bgr_list = _tensor_to_bgr_list(images)
    per_image = []
    for i, bgr in enumerate(bgr_list):
        score, results = run_cascade(bgr, checks, params=params)
        # skipped（評価不能）は合否判定から除外する
        hard_pass = all(r.passed for r in results if not r.skipped)
        soft_pass = score >= threshold                       # 集約スコアが閾値以上
        per_image.append({
            "index": i,
            "bgr": bgr,
            "score": score,
            "results": results,
            "passed": bool(hard_pass and soft_pass),
        })
    return per_image


class QualityGate:
    """バッチを評価し、集約スコア・合否・レポートを出す。リトライ判定用。"""

    CATEGORY = "QualityGate"
    FUNCTION = "evaluate"
    RETURN_TYPES = ("IMAGE", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("images", "score", "all_passed", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "expected_faces": ("INT", {"default": 1, "min": 0, "max": 20}),
            },
        }

    def evaluate(self, images, threshold, expected_faces):
        per_image = _evaluate(images, DEFAULT_CHECKS, threshold, expected_faces)
        mean_score = float(np.mean([r["score"] for r in per_image])) if per_image else 0.0
        all_passed = all(r["passed"] for r in per_image) if per_image else False
        report = _format_report(per_image)
        return (images, mean_score, all_passed, report)


class ProportionMatchRank:
    """参照画像の頭サイズ比に近い順でバッチを並べ替える。アイデア②の目玉ノード。

    「顔写真→全身生成」で毎回ブレる頭身バランスから、参照(お手本)に一番近いものを
    自動で上位に出す。指標は body_proportion.measure（耳幅÷肩幅・TTA平均, 髪型に頑健）。
    """

    CATEGORY = "QualityGate"
    FUNCTION = "rank"
    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("ranked_images", "best_image", "target_ratio", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE",),   # お手本（正解比率）1枚
                "images": ("IMAGE",),      # 並べ替え対象バッチ
                "tolerance": ("FLOAT", {"default": 0.02, "min": 0.005, "max": 0.1, "step": 0.005}),
            },
        }

    def rank(self, reference, images, tolerance):
        from .quality_checks import body_proportion as bp
        bp._init()

        ref_bgr = _tensor_to_bgr_list(reference)[0]
        ref_m = bp.measure(ref_bgr) if bp._pose is not None else None
        if ref_m is None:
            # 参照から比率が取れない → 並べ替えず素通し（downstream保護）
            report = "ProportionMatchRank: 参照画像から人物比率を取得できませんでした（未変更で通過）"
            return (images, _bgr_list_to_tensor(_tensor_to_bgr_list(images)[:1]), 0.0, report)

        target = ref_m["ratio"]
        bgr_list = _tensor_to_bgr_list(images)
        rows = []
        for i, bgr in enumerate(bgr_list):
            m = bp.measure(bgr)
            if m is None:
                rows.append({"i": i, "bgr": bgr, "ratio": None, "score": -1.0})
            else:
                rows.append({"i": i, "bgr": bgr, "ratio": m["ratio"],
                             "score": bp.ratio_score(m["ratio"], target, tolerance)})

        ranked = sorted(rows, key=lambda r: r["score"], reverse=True)
        ranked_t = _bgr_list_to_tensor([r["bgr"] for r in ranked])
        best_t = _bgr_list_to_tensor([ranked[0]["bgr"]])

        lines = [f"# ProportionMatchRank (target ear/shoulder = {target:.3f}, tol={tolerance:.3f})", ""]
        for rank, r in enumerate(ranked, 1):
            rat = "n/a" if r["ratio"] is None else f"{r['ratio']:.3f}"
            lines.append(f"{rank:2d}. #{r['i']:02d}  score={r['score']:.2f}  ear/shoulder={rat}")
        report = "\n".join(lines)
        return (ranked_t, best_t, float(target), report)


class CompositeRank:
    """顔類似 × 頭身比 × 鮮鋭度 の重み付き合成でバッチを自動ランキング。②の本命ノード。

    「参照顔で・お手本の頭身バランスで・ボケてない」を総合スコアで上位に出す。
    単一指標では拾えない最適な妥協点を選ぶ（顔類似と比率は往々に相反するため）。

      face_reference       : 保持したい本人の顔（正解頭）
      proportion_reference : 目標の頭身比のお手本（正解比率）
      w_identity/w_proportion/w_sharpness : 各軸の重み（0で無効化）
    """

    CATEGORY = "QualityGate"
    FUNCTION = "rank"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("ranked_images", "best_image", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_reference": ("IMAGE",),
                "proportion_reference": ("IMAGE",),
                "images": ("IMAGE",),
                "w_identity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "w_proportion": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "w_sharpness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.1}),
                "tolerance": ("FLOAT", {"default": 0.02, "min": 0.005, "max": 0.1, "step": 0.005}),
            },
        }

    def rank(self, face_reference, proportion_reference, images,
             w_identity, w_proportion, w_sharpness, tolerance):
        from .quality_checks import body_proportion as bp
        from .quality_checks import identity as idt
        from .quality_checks.sharpness import check_sharpness

        bp._init()
        idt._init()

        # 参照から target を用意（取れない軸は重み0扱いにフォールバック）
        ref_prop = bp.measure(_tensor_to_bgr_list(proportion_reference)[0]) if bp._pose else None
        ref_emb = idt.embed(_tensor_to_bgr_list(face_reference)[0]) if idt.available() else None
        target_ratio = ref_prop["ratio"] if ref_prop else None
        wi = w_identity if ref_emb is not None else 0.0
        wp = w_proportion if target_ratio is not None else 0.0
        ws = w_sharpness

        bgr_list = _tensor_to_bgr_list(images)
        rows = []
        for i, bgr in enumerate(bgr_list):
            # identity
            s_id = None
            if wi > 0:
                e = idt.embed(bgr)
                s_id = idt.identity_score(float(np.dot(ref_emb, e))) if e is not None else 0.0
            # proportion
            s_pr = None
            if wp > 0:
                m = bp.measure(bgr)
                s_pr = bp.ratio_score(m["ratio"], target_ratio, tolerance) if m else 0.0
            # sharpness
            s_sh = check_sharpness(bgr).score if ws > 0 else None

            parts = [(wi, s_id), (wp, s_pr), (ws, s_sh)]
            wsum = sum(w for w, s in parts if s is not None and w > 0)
            total = (sum(w * s for w, s in parts if s is not None and w > 0) / wsum
                     if wsum > 0 else 0.0)
            rows.append({"i": i, "bgr": bgr, "total": total,
                         "id": s_id, "pr": s_pr, "sh": s_sh})

        ranked = sorted(rows, key=lambda r: r["total"], reverse=True)
        ranked_t = _bgr_list_to_tensor([r["bgr"] for r in ranked])
        best_t = _bgr_list_to_tensor([ranked[0]["bgr"]])

        def fmt(v):
            return "  -  " if v is None else f"{v:.2f}"
        lines = [f"# CompositeRank  weights: id={wi} prop={wp} sharp={ws}  (target_ratio="
                 + (f"{target_ratio:.3f}" if target_ratio else "n/a") + ")", ""]
        lines.append("rank  #img  total | identity proportion sharpness")
        for rank, r in enumerate(ranked, 1):
            lines.append(f"{rank:2d}.  #{r['i']:02d}  {r['total']:.2f}  | "
                         f"{fmt(r['id'])}     {fmt(r['pr'])}       {fmt(r['sh'])}")
        report = "\n".join(lines)
        return (ranked_t, best_t, report)


def _imread_unicode(path):
    """日本語等の非ASCIIパスでも読める imread。"""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _imwrite_unicode(path, bgr):
    """日本語等の非ASCIIパスでも書ける imwrite（PNG）。"""
    ok, buf = cv2.imencode(".png", bgr)
    if ok:
        buf.tofile(path)
    return ok


def _output_directory():
    """ComfyUI の output ディレクトリ。ComfyUI外(テスト等)では None。"""
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        return None


def _list_output_subfolders():
    """output 配下のサブフォルダ名一覧（ドロップダウンの選択肢）。1階層＋直下の入れ子も含む。"""
    import os
    base = _output_directory()
    if not base or not os.path.isdir(base):
        return ["(output フォルダ未検出)"]
    names = []
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            names.append(name)
            # 直下の入れ子も "親/子" 形式で1階層だけ拾う
            for sub in sorted(os.listdir(p)):
                if os.path.isdir(os.path.join(p, sub)):
                    names.append(f"{name}/{sub}")
    return names or ["(サブフォルダなし)"]


def _resolve_folder(folder):
    """ドロップダウンの選択名(output相対) or 絶対パス を実パスに解決。"""
    import os
    if os.path.isabs(folder) and os.path.isdir(folder):
        return folder
    base = _output_directory()
    if base:
        cand = os.path.join(base, folder)
        if os.path.isdir(cand):
            return cand
    return folder


class SaveToFolder:
    """指定フォルダに画像を保存する（無ければ作成）。二段構成の第1段の受け皿。

    SaveImage と違い output/ 配下に縛られず、任意の絶対パスへ保存できる。
    同じ prefix の既存ファイルを見て連番を継続するので、Queue を複数回まわすと
    候補がフォルダに積み上がる（第2段の CompositeRankFolder が同じフォルダを読む）。
    """

    CATEGORY = "QualityGate"
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("folder",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                # 相対名は ComfyUI の output 配下に解決される（絶対パスも可）。
                "folder": ("STRING", {"default": "qg_candidates"}),
                "filename_prefix": ("STRING", {"default": "cand"}),
            },
        }

    def save(self, images, folder, filename_prefix):
        import glob
        import os
        import re

        # 相対名は output 配下に解決（絶対パスはそのまま）。個人環境に依存しない既定に。
        if not os.path.isabs(folder):
            base = _output_directory()
            folder = os.path.join(base, folder) if base else folder
        os.makedirs(folder, exist_ok=True)
        # 既存の同prefix連番を調べ、続きから採番（Queue複数回で積み上げる）
        pat = re.compile(re.escape(filename_prefix) + r"_(\d+)\.png$")
        start = 0
        for p in glob.glob(os.path.join(folder, f"{filename_prefix}_*.png")):
            m = pat.search(os.path.basename(p))
            if m:
                start = max(start, int(m.group(1)) + 1)

        bgr_list = _tensor_to_bgr_list(images)
        saved = 0
        for k, bgr in enumerate(bgr_list):
            path = os.path.join(folder, f"{filename_prefix}_{start + k:05d}.png")
            if _imwrite_unicode(path, bgr):
                saved += 1
        print(f"[SaveToFolder] {saved} 枚を保存 -> {folder} ({filename_prefix}_{start:05d}..)")
        return (folder,)


class CompositeRankFolder:
    """フォルダを1枚ずつストリーミング採点し、上位だけ返す。大量候補向け（メモリ非依存）。

    二段構成の第2段。第1段（生成）が候補をフォルダに書き出し、ここで一括選別する。
    画像を全部メモリに載せず、1枚ずつ [読込→採点→破棄] するので、
    候補が何百枚でもRAMは top_k 枚ぶんしか使わない。採点式は CompositeRank と同じ。
    """

    CATEGORY = "QualityGate"
    FUNCTION = "rank_folder"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("top_images", "best_image", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_reference": ("IMAGE",),
                "proportion_reference": ("IMAGE",),
                # output配下のサブフォルダをドロップダウンで選択（パス直打ち不要）
                "folder": (_list_output_subfolders(),),
                "w_identity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "w_proportion": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "w_sharpness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.1}),
                "tolerance": ("FLOAT", {"default": 0.02, "min": 0.005, "max": 0.1, "step": 0.005}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 256}),
            },
        }

    def rank_folder(self, face_reference, proportion_reference, folder,
                    w_identity, w_proportion, w_sharpness, tolerance, top_k):
        import glob
        import os
        from .quality_checks import body_proportion as bp
        from .quality_checks import identity as idt
        from .quality_checks.sharpness import check_sharpness

        bp._init()
        idt._init()
        ref_prop = bp.measure(_tensor_to_bgr_list(proportion_reference)[0]) if bp._pose else None
        ref_emb = idt.embed(_tensor_to_bgr_list(face_reference)[0]) if idt.available() else None
        target = ref_prop["ratio"] if ref_prop else None
        wi = w_identity if ref_emb is not None else 0.0
        wp = w_proportion if target is not None else 0.0
        ws = w_sharpness

        folder = _resolve_folder(folder)          # ドロップダウン選択名→実パス
        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
        paths = sorted(p for e in exts for p in glob.glob(os.path.join(folder, e)))

        rows = []  # 画素は持たず (path, total, s_id, s_pr, s_sh) だけ蓄える
        for p in paths:
            bgr = _imread_unicode(p)
            if bgr is None:
                continue
            s_id = s_pr = s_sh = None
            if wi > 0:
                e = idt.embed(bgr)
                s_id = idt.identity_score(float(np.dot(ref_emb, e))) if e is not None else 0.0
            if wp > 0:
                m = bp.measure(bgr)
                s_pr = bp.ratio_score(m["ratio"], target, tolerance) if m else 0.0
            if ws > 0:
                s_sh = check_sharpness(bgr).score
            parts = [(wi, s_id), (wp, s_pr), (ws, s_sh)]
            wsum = sum(w for w, s in parts if s is not None and w > 0)
            total = (sum(w * s for w, s in parts if s is not None and w > 0) / wsum
                     if wsum > 0 else 0.0)
            rows.append({"path": p, "total": total, "id": s_id, "pr": s_pr, "sh": s_sh})
            del bgr

        if not rows:
            raise RuntimeError(f"CompositeRankFolder: 画像が見つかりません: {folder}")

        rows.sort(key=lambda r: r["total"], reverse=True)
        top = rows[:top_k]
        # 上位のみメモリに読む。バッチ化のため best のサイズに揃える。
        best_bgr = _imread_unicode(top[0]["path"])
        H, W = best_bgr.shape[:2]
        top_bgrs = [best_bgr] + [cv2.resize(_imread_unicode(r["path"]), (W, H)) for r in top[1:]]
        top_t = _bgr_list_to_tensor(top_bgrs)
        best_t = _bgr_list_to_tensor([best_bgr])

        def fmt(v):
            return "  -  " if v is None else f"{v:.2f}"
        lines = [f"# CompositeRankFolder  {len(rows)}枚採点 → top{len(top)}  "
                 f"weights id={wi} prop={wp} sharp={ws}", ""]
        lines.append("rank  total | identity proportion sharpness  file")
        for i, r in enumerate(top, 1):
            lines.append(f"{i:2d}.  {r['total']:.2f}  | {fmt(r['id'])}     {fmt(r['pr'])}"
                         f"       {fmt(r['sh'])}   {os.path.basename(r['path'])}")
        return (top_t, best_t, "\n".join(lines))


class QualityFilterBatch:
    """バッチから合格画像だけを通す。'合格分だけ納品' パイプラインの目玉ノード。"""

    CATEGORY = "QualityGate"
    FUNCTION = "filter"
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("passed_images", "rejected_images", "passed_count", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "expected_faces": ("INT", {"default": 1, "min": 0, "max": 20}),
            },
        }

    def filter(self, images, threshold, expected_faces):
        per_image = _evaluate(images, DEFAULT_CHECKS, threshold, expected_faces)
        passed = [r["bgr"] for r in per_image if r["passed"]]
        rejected = [r["bgr"] for r in per_image if not r["passed"]]

        # 空バッチにならないよう、片方が空なら元画像でフォールバック（downstream 保護）。
        fallback = _tensor_to_bgr_list(images)[:1]
        passed_t = _bgr_list_to_tensor(passed or fallback)
        rejected_t = _bgr_list_to_tensor(rejected or fallback)
        report = _format_report(per_image)
        return (passed_t, rejected_t, len(passed), report)
