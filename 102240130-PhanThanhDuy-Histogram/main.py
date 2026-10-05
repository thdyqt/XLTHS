"""
KIEM THU - PHAN DOAN TIENG NOI / KHOANG LANG TREN TIN HIEU KIEM THU
Dung CHUNG nguong T1, T2 (va T1_high, hang) da tim o huanluyen.py (doc tu thresholds.json).

Cau truc thu muc:
    huanluyen.py   (huan luyen: tim nguong, luu thresholds.json)
    main.py        (file nay: kiem thu)
    TinHieuHuanLuyen/*.wav + *.lab
    TinHieuKiemThu/*.wav  + *.lab

Chay:  python main.py   (1 lan duy nhat, ra 1 figure cho moi file kiem thu)
Neu chua co thresholds.json, chuong trinh tu huan luyen roi luu lai.
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# Cac ham xu ly tin hieu va tham so co dinh dung chung voi file huan luyen
from huanluyen import (FRAME_MS, SHIFT_MS, TEST_DIR,
                       read_lab, extract_smoothed_features, segment_signal, boundary_errors,
                       train, save_thresholds, load_thresholds)


# =====================================================================
# BLOCK 8: KIEM THU VA VE FIGURE (1 FIGURE / 1 FILE)
# =====================================================================
def plot_result(path, feat, mask, pred, true, T1, T2, T1_high, mae, rmse):
    """Ve figure kieu bai bao Giannakopoulos cho 1 file kiem thu, 3 do thi:
    (1) STE goc (xanh la) + da loc (xanh lo) + nguong T1 (den), truc x = chi so khung
    (2) Spectral Centroid goc + da loc + nguong T2, truc x = chi so khung
    (3) dang song + STE xep chong + bien chuan (do) + bien thuat toan (xanh duong), truc x = giay.
    path: ten file; feat: dac trung; pred/true: bien thuat toan/chuan;
    T1, T2, T1_high: nguong; mae, rmse: sai so (ms)."""
    sig, fs = feat["signal"], feat["fs"]
    L = int(round(FRAME_MS * fs / 1000))
    S = int(round(SHIFT_MS * fs / 1000))
    t_sig = np.arange(len(sig)) / fs
    t_fr = (np.arange(len(feat["ste"])) * S + L / 2) / fs
    idx = np.arange(len(feat["ste"]))

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    # (1) STE
    ax[0].plot(idx, feat["ste"], color='lime', linewidth=1, label='STE (original)')
    ax[0].plot(idx, feat["ste_s"], color='cyan', linewidth=1.5, label='STE (filtered)')
    ax[0].axhline(T1, color='black', linewidth=2.5, label=f'T1 = {T1:.3f}')
    if T1_high < 1.0:
        ax[0].axhline(T1_high, color='magenta', linestyle='--', linewidth=1.2,
                      label=f'T1_high = {T1_high:.2f}')
    ax[0].set_xlim(0, len(idx))
    ax[0].set_title("Short time energy", fontsize=10, loc="left")
    ax[0].set_xlabel("Frame index (10 ms / frame)")
    ax[0].set_ylabel("Normalized STE")
    ax[0].legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=7,
                 frameon=False, columnspacing=1.0, handlelength=1.5)

    # (2) Spectral centroid
    ax[1].plot(idx, feat["sc"], color='lime', linewidth=1, label='SC (original)')
    ax[1].plot(idx, feat["sc_s"], color='cyan', linewidth=1.5, label='SC (filtered)')
    ax[1].axhline(T2, color='black', linewidth=2.5, label=f'T2 = {T2:.3f}')
    ax[1].set_xlim(0, len(idx))
    ax[1].set_title("Spectral centroid", fontsize=10, loc="left")
    ax[1].set_xlabel("Frame index (10 ms / frame)")
    ax[1].set_ylabel("Normalized SC")
    ax[1].legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=7,
                 frameon=False, columnspacing=1.0, handlelength=1.5)

    # (3) Dang song + STE xep chong + bien
    ax[2].plot(t_sig, sig, color='silver', label='Waveform')
    ax[2].plot(t_fr, feat["ste_s"] * np.max(np.abs(sig)), color='green',
               linewidth=1.5, label='STE overlay')
    for k, (s0, e0) in enumerate(true):
        ax[2].axvline(s0, color='red', linewidth=1.5, label='Groundtruth' if k == 0 else None)
        ax[2].axvline(e0, color='red', linewidth=1.5)
    for k, (s0, e0) in enumerate(pred):
        ax[2].axvline(s0, color='blue', linestyle='--', linewidth=1.5,
                      label='Algorithm' if k == 0 else None)
        ax[2].axvline(e0, color='blue', linestyle='--', linewidth=1.5)
    ax[2].set_xlim(0, t_sig[-1])
    ax[2].set_title("Signal and boundaries", fontsize=10, loc="left")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Amplitude")
    ax[2].legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=7,
                 frameon=False, columnspacing=1.0, handlelength=1.5)

    fig.suptitle(f"{os.path.basename(path)} | MAE = {mae:.2f} ms | RMSE = {rmse:.2f} ms",
                 fontsize=11, fontweight='bold')


def test(model, test_dir=TEST_DIR):
    """Chay thuat toan voi nguong da huan luyen tren moi file kiem thu,
    ve 1 figure/file, in bang sai so MAE, RMSE (ms).
    model: dict tu train(); test_dir: thu muc tin hieu kiem thu."""
    files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
    results = []
    for f in files:
        feat = extract_smoothed_features(f)
        mask, pred = segment_signal(feat, model["T1"], model["T2"], model["T1_high"], model["hang"],
                                     model.get("shrink", 0))
        true = read_lab(f.replace(".wav", ".lab"))
        mae, rmse = boundary_errors(pred, true)
        print(f"{os.path.basename(f)}\n  chuan : {[[round(float(a), 3), round(float(b), 3)] for a, b in true]}"
              f"\n  thuat toan: {[[round(float(a), 3), round(float(b), 3)] for a, b in pred]}")
        results.append((os.path.basename(f), len(true) * 2, len(pred) * 2, mae, rmse))
        plot_result(f, feat, mask, pred, true, model["T1"], model["T2"],
                    model["T1_high"], mae, rmse)

    print("\n[KIEM THU]")
    print(f"{'File':30s} {'#bien chuan':>12s} {'#bien TT':>9s} {'MAE(ms)':>9s} {'RMSE(ms)':>9s}")
    for name, nt, npred, mae, rmse in results:
        print(f"{name:30s} {nt:12d} {npred:9d} {mae:9.2f} {rmse:9.2f}")
    ok = [r for r in results if np.isfinite(r[3])]
    if ok:
        print(f"{'TRUNG BINH':30s} {'':12s} {'':9s} "
              f"{np.mean([r[3] for r in ok]):9.2f} {np.mean([r[4] for r in ok]):9.2f}")


# =====================================================================
# BLOCK 9: MAIN (KIEM THU)
# =====================================================================
def get_params():
    """Lay tham so da huan luyen (T1, T2, T1_high, hang). Doc tu thresholds.json; neu chua
    co thi huan luyen ngay roi luu lai. Tra ve: dict tham so."""
    params = load_thresholds()
    if params is None:
        print("Chua co thresholds.json -> huan luyen truoc...")
        model = train()
        save_thresholds(model)
        params = load_thresholds()
    return params


def main():
    """Nap nguong chung -> kiem thu tren TinHieuKiemThu -> hien thi cac figure."""
    params = get_params()
    print(f"[NGUONG DUNG CHUNG] T1 (STE) = {params['T1']:.5f} | T2 (SC) = {params['T2']:.5f} | "
          f"T1_high = {params['T1_high']} | hang = {params['hang']} | "
          f"shrink = {params.get('shrink', 0)}\n")
    test(params)
    plt.show()


if __name__ == "__main__":
    main()