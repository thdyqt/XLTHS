"""
HUAN LUYEN - XAC DINH NGUONG PHAN BIET DUNG CHUNG CHO 04 TIN HIEU HUAN LUYEN
Thuat toan histogram (Giannakopoulos 2014): dac trung STE + Spectral Centroid.

Viec file nay lam:
  1) Doc 4 tin hieu trong TinHieuHuanLuyen, gop dac trung -> histogram -> M1, M2 -> T1, T2.
  2) Duyet luoi W_ste, W_sc, T1_high, hang de chon bo tham so co MAE nho nhat so voi *.lab.
  3) LUU ket qua vao thresholds.json  (main.py se doc file nay de kiem thu)
     va luu hinh histogram_nguong.png (dung cho slide).

Chay:  python huanluyen.py   (chay 1 lan truoc; sau do chay main.py)
"""
import glob
import json
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

# ----------------------- THAM SO CO DINH THEO DE BAI -----------------------
FRAME_MS = 25          # do dai khung
SHIFT_MS = 10          # do dich khung
MIN_SILENCE_MS = 200   # khoang lang ngan hon se bi coi la "ao" va bi lap day

# ----------------------- THAM SO THUAT TOAN --------------------------------
HIST_BINS = 100        # so bin cua histogram
HIST_SMOOTH_LEN = 5    # do dai bo loc lam min histogram
TIME_SMOOTH_LEN = 5    # do dai bo loc lam min dac trung theo thoi gian
PEAK_RATIO = 0.05      # chi nhan cuc dai cao hon 5% cuc dai lon nhat
W_GRID = [0.5, 1, 2, 3, 5, 8, 10, 15, 20]   # luoi tim W toi uu
T_HIGH_GRID = [0.05, 0.1, 0.2, 0.4, 1.0]   # luoi nguong STE cao (1.0 = tat quy tac nay)
H_GRID = [0, 1, 2, 3, 5]                     # luoi so khung keo dai (hangover) 2 phia moi doan
SHRINK_GRID = [-3, -2, -1, 0, 1, 2, 3, 4]              # luoi so khung dich bien vao trong (am = mo rong ra)
SPEECH_LABELS = {'v', 'uv'}                  # nhan trong .lab duoc coi la tieng noi

# ĐÃ BẬT CỜ HIỂN THỊ ĐỒ THỊ LÊN MÀN HÌNH
SHOW_HIST_FIG = True   

THRESHOLD_FILE = "thresholds.json"   # file luu T1, T2... de main.py doc lai
TRAIN_DIR = "TinHieuHuanLuyen"
TEST_DIR = "TinHieuKiemThu"


# =====================================================================
# BLOCK 1: DOC FILE AM THANH VA FILE NHAN (.lab)
# =====================================================================
def read_wav(path):
    with warnings.catch_warnings():          
        warnings.simplefilter('ignore')
        fs, x = wavfile.read(path)
    if x.ndim > 1:                       
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x / float(np.iinfo(x.dtype).max + 1)
    return x.astype(np.float64), fs


def read_lab(lab_path, speech_labels=SPEECH_LABELS):
    segs = []
    if not os.path.exists(lab_path):
        return segs
    with open(lab_path, 'r') as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 3 and p[2].lower() in speech_labels:
                s, e = float(p[0]), float(p[1])
                if segs and abs(segs[-1][1] - s) < 1e-9:
                    segs[-1][1] = e      
                else:
                    segs.append([s, e])
    return segs


# =====================================================================
# BLOCK 2: CAT KHUNG VA LAM MIN
# =====================================================================
def create_frames(signal, fs, frame_ms=FRAME_MS, shift_ms=SHIFT_MS):
    L = int(round(frame_ms * fs / 1000))
    S = int(round(shift_ms * fs / 1000))
    n = (len(signal) - L) // S + 1
    idx = np.arange(L)[None, :] + S * np.arange(n)[:, None]
    return signal[idx]


def moving_average(x, length):
    x = np.asarray(x, dtype=np.float64)
    h = length // 2
    padded = np.concatenate([np.full(h, x[0]), x, np.full(h, x[-1])])
    csum = np.concatenate([[0.0], np.cumsum(padded)])
    return (csum[length:] - csum[:-length]) / length


# =====================================================================
# BLOCK 3: TRICH XUAT DAC TRUNG (STE VA SPECTRAL CENTROID)
# =====================================================================
def compute_ste(frames):
    ste = np.mean(frames ** 2, axis=1)
    m = np.max(ste)
    return ste / m if m > 0 else ste


def dft_magnitude_half(frames):
    N = frames.shape[1]
    n = np.arange(N)[:, None]
    k = np.arange(N // 2)[None, :]
    ang = 2 * np.pi * n * k / N
    re = frames @ np.cos(ang)
    im = frames @ np.sin(ang)
    return np.sqrt(re ** 2 + im ** 2)


def compute_spectral_centroid(frames):
    X = dft_magnitude_half(frames)
    k = np.arange(1, X.shape[1] + 1)[None, :]
    s = np.sum(X, axis=1)
    sc = np.where(s > 0, np.sum(k * X, axis=1) / np.where(s > 0, s, 1), 0.0)
    m = np.max(sc)
    return sc / m if m > 0 else sc


def extract_smoothed_features(path):
    signal, fs = read_wav(path)
    frames = create_frames(signal, fs)
    ste = compute_ste(frames)
    sc = compute_spectral_centroid(frames)
    return {"signal": signal, "fs": fs, "ste": ste, "sc": sc,
            "ste_s": moving_average(ste, TIME_SMOOTH_LEN),
            "sc_s": moving_average(sc, TIME_SMOOTH_LEN)}


# =====================================================================
# BLOCK 4: HISTOGRAM -> CUC DAI M1, M2 -> NGUONG T
# =====================================================================
def build_histogram(features, bins=HIST_BINS, lo=None, hi=None):
    features = np.asarray(features)
    lo = features.min() if lo is None else lo
    hi = features.max() if hi is None else hi
    idx = np.clip(((features - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
    hist = np.zeros(bins)
    for k in idx:
        hist[k] += 1
    edges = np.linspace(lo, hi, bins + 1)
    return hist, (edges[:-1] + edges[1:]) / 2


def find_first_two_peaks(smooth_hist, centers, peak_ratio=PEAK_RATIO):
    min_h = peak_ratio * smooth_hist.max()
    peaks = []
    for i in range(len(smooth_hist)):
        left = smooth_hist[i - 1] if i > 0 else -np.inf
        right = smooth_hist[i + 1] if i < len(smooth_hist) - 1 else -np.inf
        if smooth_hist[i] > left and smooth_hist[i] >= right and smooth_hist[i] >= min_h:
            peaks.append(centers[i])
    M1 = peaks[0] if len(peaks) >= 1 else centers[0]
    M2 = peaks[1] if len(peaks) >= 2 else M1
    return M1, M2


def threshold_from_peaks(M1, M2, W):
    return (W * M1 + M2) / (W + 1)


def analyze_histogram(features):
    hist, centers = build_histogram(features)
    smooth = moving_average(hist, HIST_SMOOTH_LEN)
    M1, M2 = find_first_two_peaks(smooth, centers)
    return M1, M2, hist, smooth, centers


# =====================================================================
# BLOCK 5: PHAN LOAI KHUNG, HAU XU LY, DOI RA BIEN THOI GIAN
# =====================================================================
def classify_frames(ste_s, sc_s, T1, T2, T1_high=1.0):
    return ((ste_s > T1) & (sc_s > T2)) | (ste_s > T1_high)


def extend_speech(is_speech, h):
    if h <= 0:
        return is_speech.copy()
    x = is_speech.astype(int)
    padded = np.concatenate([np.zeros(h, dtype=int), x, np.zeros(h, dtype=int)])
    csum = np.concatenate([[0], np.cumsum(padded)])
    window_sum = csum[2 * h + 1:] - csum[:-(2 * h + 1)]
    return window_sum > 0


def shrink_speech(is_speech, n):
    if n == 0:
        return is_speech.copy()
    if n < 0:
        return extend_speech(is_speech, -n)
    return ~extend_speech(~is_speech, n)


def fill_short_silences(is_speech, min_silence_frames):
    out = is_speech.copy()
    n, i = len(out), 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if i > 0 and j < n and (j - i) < min_silence_frames:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def mask_to_segments(is_speech, fs):
    L = int(round(FRAME_MS * fs / 1000))
    S = int(round(SHIFT_MS * fs / 1000))
    centers = (np.arange(len(is_speech)) * S + L / 2) / fs
    half = S / fs / 2
    segs, start = [], None
    for i, v in enumerate(is_speech):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append([centers[start] - half, centers[i - 1] + half])
            start = None
    if start is not None:
        segs.append([centers[start] - half, centers[-1] + half])
    return segs


def segment_signal(feat, T1, T2, T1_high=1.0, hang=0, shrink=0):
    mask = classify_frames(feat["ste_s"], feat["sc_s"], T1, T2, T1_high)
    mask = extend_speech(mask, hang)
    mask = fill_short_silences(mask, MIN_SILENCE_MS // SHIFT_MS)
    mask = shrink_speech(mask, hang + shrink)
    return mask, mask_to_segments(mask, feat["fs"])


# =====================================================================
# BLOCK 6: DANH GIA SAI SO (MAE, RMSE - DON VI ms)
# =====================================================================
def boundary_errors(pred_segs, true_segs):
    pred = np.array([t for sg in pred_segs for t in sg])
    true = np.array([t for sg in true_segs for t in sg])
    if len(true) == 0:
        return np.nan, np.nan
    if len(pred) == 0:
        return np.inf, np.inf
    e1 = [np.min(np.abs(pred - t)) for t in true]
    e2 = [np.min(np.abs(true - p)) for p in pred]
    err = np.array(e1 + e2) * 1000.0
    return float(np.mean(err)), float(np.sqrt(np.mean(err ** 2)))


# =====================================================================
# BLOCK 7: HUAN LUYEN - TIM NGUONG CHUNG VA W TOI UU
# =====================================================================
def train(train_dir=TRAIN_DIR):
    files = sorted(glob.glob(os.path.join(train_dir, "*.wav")))
    data = []
    for f in files:
        feat = extract_smoothed_features(f)
        feat["true"] = read_lab(f.replace(".wav", ".lab"))
        data.append(feat)

    all_ste = np.concatenate([d["ste_s"] for d in data])
    all_sc = np.concatenate([d["sc_s"] for d in data])
    M1s, M2s, h_s, hs_s, c_s = analyze_histogram(all_ste)
    M1c, M2c, h_c, hs_c, c_c = analyze_histogram(all_sc)

    best = (np.inf, None, None, None, None, None)
    for Ws in W_GRID:
        for Wc in W_GRID:
            T1 = threshold_from_peaks(M1s, M2s, Ws)
            T2 = threshold_from_peaks(M1c, M2c, Wc)
            for Th in T_HIGH_GRID:
                for hang in H_GRID:
                    for shrink in SHRINK_GRID:
                        maes = []
                        for d in data:
                            _, segs = segment_signal(d, T1, T2, Th, hang, shrink)
                            mae, _ = boundary_errors(segs, d["true"])
                            if np.isfinite(mae):
                                maes.append(mae)
                        score = np.mean(maes) if maes else np.inf
                        if score < best[0]:
                            best = (score, Ws, Wc, Th, hang, shrink)

    _, Ws, Wc, Th, hang, shrink = best
    T1 = threshold_from_peaks(M1s, M2s, Ws)
    T2 = threshold_from_peaks(M1c, M2c, Wc)
    print(f"[HUAN LUYEN] {len(files)} file | MAE trung binh = {best[0]:.2f} ms")
    print(f"  STE: M1={M1s:.4f} M2={M2s:.4f} W={Ws} -> T1={T1:.4f}")
    print(f"  SC : M1={M1c:.4f} M2={M2c:.4f} W={Wc} -> T2={T2:.4f}")
    print(f"  STE cao: T1_high={Th} | keo dai gop: {hang} khung | dich bien vao trong: {shrink} khung")
    per_file = [(os.path.basename(f), d["ste_s"], d["sc_s"]) for f, d in zip(files, data)]
    return {"per_file": per_file, "train_mae": float(best[0]), "T1": T1, "T2": T2, "T1_high": Th, "hang": hang, "shrink": shrink, "W_ste": Ws, "W_sc": Wc,
            "ste": (M1s, M2s, h_s, hs_s, c_s), "sc": (M1c, M2c, h_c, hs_c, c_c)}


def save_histogram_figure(model, out_path="histogram_nguong.png"):
    colors = ['tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown', 'tab:pink']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, key, idx, T, W, name in [
            (axes[0], "ste", 1, model["T1"], model["W_ste"], "STE"),
            (axes[1], "sc", 2, model["T2"], model["W_sc"], "Spectral Centroid")]:
        M1, M2, hist, smooth, centers = model[key]
        all_x = np.concatenate([p[idx] for p in model["per_file"]])
        lo, hi = all_x.min(), all_x.max()
        width = centers[1] - centers[0]
        
        # Sửa nhãn legend để giống y hệt hình mẫu
        for j, p in enumerate(model["per_file"]):
            h, c = build_histogram(p[idx], lo=lo, hi=hi)
            ax.plot(c, h / h.sum(), color=colors[j % len(colors)], linewidth=1,
                    alpha=0.6, label=p[0])
            
        ax.plot(centers, smooth / hist.sum(), color='blue', linewidth=2.5,
                label='Histogram gop da lam min')
        ax.axvline(M1, color='green', linestyle=':', linewidth=2.5, label=f'M1 = {M1:.3f}')
        ax.axvline(M2, color='orange', linestyle=':', linewidth=2.5, label=f'M2 = {M2:.3f}')
        ax.axvline(T, color='black', linewidth=3, label=f'Nguong T = {T:.3f} (W = {W})')
        
        ax.bar(centers, hist / hist.sum(), width=width, color='lightgray',
               alpha=0.8, label=f'Gop {len(model["per_file"])} file huan luyen')
        
        ax.set_yscale('log')
        ax.set_title(f"Nguong chung cua {name}", fontsize=16, fontweight='bold')
        ax.set_xlabel(f"Gia tri {name} (chuan hoa)", fontsize=14)
        ax.set_ylabel("Ty le khung (thang log)", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=10, loc='upper right')
        
    fig.suptitle("Xac dinh nguong tu histogram cua cac tin hieu huan luyen",
                 fontsize=18, fontweight='bold')
    fig.savefig(out_path, dpi=200)
    if SHOW_HIST_FIG:
        plt.show(block=False)
    else:
        plt.close(fig)


# =====================================================================
# BLOCK 8: LUU / NAP THAM SO DA HUAN LUYEN
# =====================================================================
def save_thresholds(model, path=THRESHOLD_FILE):
    out = {"T1": float(model["T1"]), "T2": float(model["T2"]),
           "T1_high": float(model["T1_high"]), "hang": int(model["hang"]),
           "shrink": int(model["shrink"]),
           "W_ste": float(model["W_ste"]), "W_sc": float(model["W_sc"]),
           "M1_ste": float(model["ste"][0]), "M2_ste": float(model["ste"][1]),
           "M1_sc": float(model["sc"][0]), "M2_sc": float(model["sc"][1]),
           "train_mae_ms": model["train_mae"]}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Da luu nguong vao {path}")


def load_thresholds(path=THRESHOLD_FILE):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# =====================================================================
# BLOCK 9: MAIN (HUAN LUYEN)
# =====================================================================
def main():
    model = train()
    save_thresholds(model)
    save_histogram_figure(model)
    print("\n>>> NGUONG DUNG CHUNG: T1 (STE) = %.5f | T2 (SC) = %.5f | T1_high = %.2f | hang = %d | shrink = %d"
          % (model["T1"], model["T2"], model["T1_high"], model["hang"], model["shrink"]))
    print("Hinh histogram: histogram_nguong.png. Bay gio chay: python main.py")
    if SHOW_HIST_FIG:
        plt.show()


if __name__ == "__main__":
    main()