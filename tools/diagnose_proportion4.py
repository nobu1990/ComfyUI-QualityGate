"""
diagnose_proportion4.py ─ 新指標の検証: 耳幅/肩幅 を TTA(K回平均)でノイズ除去

R = 耳〜耳(Pose 7,8) / 肩幅(11,12) を、微小クロップK回の平均で算出（単発ノイズを1/√K に）。
オーバーレイに TTA平均R を表示し、ランキングが目視と合うか確認する用。

  <wan22 python> tools/diagnose_proportion4.py <入力フォルダ> [出力フォルダ] [K]
"""

import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quality_checks.body_proportion as bp  # noqa: E402

_rng = np.random.default_rng(12345)


def imread_jp(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def imwrite_jp(p, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(str(p))


def _ear_sh(img):
    import mediapipe as mp
    h, w = img.shape[:2]
    pr = bp._pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
    if not pr.pose_landmarks:
        return None, None
    lm = pr.pose_landmarks[0]
    def P(i): return np.array([lm[i].x * w, lm[i].y * h])
    shw = float(np.linalg.norm(P(11) - P(12)))
    if shw <= 1:
        return None, None
    ear = float(np.linalg.norm(P(7) - P(8)))
    return ear / shw, (P(7), P(8), P(11), P(12), P(23), P(24))


def measure_tta(img, K=8):
    """K回の微小クロップ平均で R を算出。描画用に元画像のキーポイントも返す。"""
    h, w = img.shape[:2]
    vals = []
    for _ in range(K):
        t, b, l, r = [int(_rng.uniform(0, 0.03) * s) for s in (h, h, w, w)]
        v, _ = _ear_sh(img[t:h - b, l:w - r])
        if v:
            vals.append(v)
    if not vals:
        return None, None
    _, pts = _ear_sh(img)          # 描画は元画像で
    return float(np.mean(vals)), pts


def overlay(img_bgr, R, pts):
    vis = img_bgr.copy()
    e7, e8, s11, s12, h23, h24 = pts
    def pt(a): return (int(a[0]), int(a[1]))
    cv2.line(vis, pt(e7), pt(e8), (255, 255, 0), 2)      # 耳幅(シアン)
    cv2.line(vis, pt(s11), pt(s12), (0, 0, 255), 2)      # 肩幅(赤)
    for a in (e7, e8):
        cv2.circle(vis, pt(a), 5, (0, 255, 255), -1)
    for a in (s11, s12):
        cv2.circle(vis, pt(a), 5, (0, 0, 255), -1)
    txt = f"R(ear/shoulder, TTA)={R:.3f}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(vis, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis


def main():
    if len(sys.argv) < 2:
        print("usage: diagnose_proportion4.py <入力フォルダ> [出力フォルダ] [K]"); return
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent / (src.name + "_diag4")
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out.mkdir(parents=True, exist_ok=True)
    bp._init()
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in exts)
    rows = []
    for p in imgs:
        img = imread_jp(p)
        if img is None:
            continue
        R, pts = measure_tta(img, K)
        if R is None:
            continue
        imwrite_jp(out / p.name, overlay(img, R, pts))
        rows.append((p.name, R))

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    v = np.array([r[1] for r in rows])
    print(f"測定 {len(rows)}/{len(imgs)}枚 (K={K}) -> {out}")
    print(f"R=耳幅/肩幅(TTA): mean{v.mean():.3f} range{v.min():.3f}..{v.max():.3f} CV{v.std()/v.mean():.2f}")
    s = lambda n: n.replace('cand_', '').replace('.png', '')
    sr = sorted(rows, key=lambda r: r[1])
    print("\n--- R 小10（頭が小さい判定・小顔）---")
    for n, r in sr[:10]:
        print(f"  {r:.3f}  {s(n)}")
    print("--- R 大10（頭が大きい判定・頭でっかち）---")
    for n, r in sr[-10:]:
        print(f"  {r:.3f}  {s(n)}")
    print("\nオーバーレイ: 黄点=耳/シアン線=耳幅  赤点=肩/赤線=肩幅  上部=TTA平均R")


if __name__ == "__main__":
    main()
