"""
diagnose_proportion3.py ─ Pose只の頑健指標を検証（顔検出・セグメント不要）

顎ライン方式は生成画像で顔検出が2-4割失敗するため断念。代わりに:
  頭の幅 = 耳〜耳(Pose 7,8)         ← 髪が下に垂れても不変。髪で耳が隠れても推定される
  body基準 = 肩幅(11,12) / 胴体長(肩中心〜腰中心)
  R_shoulder = 耳幅 / 肩幅
  R_torso    = 耳幅 / 胴体長

オーバーレイ: 耳(黄)・肩(赤)・耳線(シアン)・肩線(青)。数値を上部に表示。
  <wan22 python> tools/diagnose_proportion3.py <入力フォルダ> [出力フォルダ]
"""

import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quality_checks.body_proportion as bp  # noqa: E402


def imread_jp(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def imwrite_jp(p, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(str(p))


def measure(img_bgr):
    import mediapipe as mp
    h, w = img_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pr = bp._pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not pr.pose_landmarks:
        return None
    lm = pr.pose_landmarks[0]

    def P(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    ear = float(np.linalg.norm(P(7) - P(8)))          # 耳〜耳 = 頭幅
    shw = float(np.linalg.norm(P(11) - P(12)))         # 肩幅
    sh_mid = (P(11) + P(12)) / 2
    hip_mid = (P(23) + P(24)) / 2
    torso = float(np.linalg.norm(sh_mid - hip_mid))
    if ear <= 1 or shw <= 1 or torso <= 1:
        return None
    return {
        "R_shoulder": ear / shw, "R_torso": ear / torso,
        "ear": ear, "shw": shw, "torso": torso,
        "e7": P(7), "e8": P(8), "s11": P(11), "s12": P(12),
        "sh_mid": sh_mid, "hip_mid": hip_mid,
    }


def overlay(img_bgr, m):
    vis = img_bgr.copy()
    def pt(a): return (int(a[0]), int(a[1]))
    cv2.line(vis, pt(m["e7"]), pt(m["e8"]), (255, 255, 0), 2)     # 耳線(シアン)
    cv2.line(vis, pt(m["s11"]), pt(m["s12"]), (0, 0, 255), 2)     # 肩線(赤)
    cv2.line(vis, pt(m["sh_mid"]), pt(m["hip_mid"]), (255, 0, 0), 2)  # 胴体(青)
    for a, c in [(m["e7"], (0, 255, 255)), (m["e8"], (0, 255, 255)),
                 (m["s11"], (0, 0, 255)), (m["s12"], (0, 0, 255))]:
        cv2.circle(vis, pt(a), 5, c, -1)
    txt = f"R_sh={m['R_shoulder']:.2f}  R_torso={m['R_torso']:.2f}  ear={m['ear']:.0f} shw={m['shw']:.0f}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(vis, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis


def main():
    if len(sys.argv) < 2:
        print("usage: diagnose_proportion3.py <入力フォルダ> [出力フォルダ]"); return
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent / (src.name + "_diag3")
    out.mkdir(parents=True, exist_ok=True)
    bp._init()
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in exts)
    rows = []
    for p in imgs:
        img = imread_jp(p)
        if img is None:
            continue
        m = measure(img)
        if m is None:
            continue
        imwrite_jp(out / p.name, overlay(img, m))
        rows.append((p.name, m["R_shoulder"], m["R_torso"]))

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print(f"測定成功 {len(rows)}/{len(imgs)}枚 -> {out}")
    for idx, label in [(1, "R_sh = 耳幅/肩幅"), (2, "R_torso = 耳幅/胴体長")]:
        v = np.array([r[idx] for r in rows])
        print(f"\n[{label}] mean{v.mean():.3f} range{v.min():.3f}..{v.max():.3f} CV{v.std()/v.mean():.2f}")
        sr = sorted(rows, key=lambda r: r[idx])
        s = lambda n: n.replace('cand_', '').replace('.png', '')
        print("   小5:", [s(r[0]) for r in sr[:5]])
        print("   大5:", [s(r[0]) for r in sr[-5:]])
    print("\nオーバーレイ: 黄点=耳 シアン線=耳幅 赤=肩 青=胴体")


if __name__ == "__main__":
    main()
