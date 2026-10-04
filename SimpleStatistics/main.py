import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk  # Dùng để lấy độ phân giải màn hình

# Thêm thư mục gốc vào đường dẫn hệ thống để import SharedFunctions.py
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import SharedFunctions

# ================= CÁC THAM SỐ CỐ ĐỊNH =================
FRAME_LEN_MS = 25       # Độ dài khung: 25ms
FRAME_SHIFT_MS = 10     # Độ dịch khung: 10ms
MIN_SILENCE_MS = 200    # Khoảng lặng tối thiểu: 200ms
# =======================================================

def read_lab_file(lab_path):
    """
    Chức năng: Đọc file .lab và trích xuất các mốc thời gian của từng nhãn.
    Đầu vào: lab_path (string) - Đường dẫn tới file .lab
    Đầu ra: intervals (list) - Danh sách các tuple (start_time, end_time, label)
    """
    intervals = []
    with open(lab_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and parts[0] not in ['F0mean', 'F0std']:
                start, end, label = float(parts[0]), float(parts[1]), parts[2]
                intervals.append((start, end, label))
    return intervals

def get_frame_label(time_sec, intervals):
    """
    Chức năng: Gán nhãn cho 1 khung dựa vào mốc thời gian so với dữ liệu .lab
    Đầu vào: time_sec (float), intervals (list)
    Đầu ra: int - 0 (Khoảng lặng), 1 (Tiếng nói)
    """
    for start, end, label in intervals:
        if start <= time_sec <= end:
            return 0 if label == 'sil' else 1
    return 0

def extract_features(signal, fs):
    """
    Chức năng: Chia khung, tính đặc trưng STE và ZCR (đã chuẩn hóa).
    Đầu vào: signal (numpy array), fs (int)
    Đầu ra: ste, zcr, times (numpy arrays)
    """
    frame_len = int(fs * FRAME_LEN_MS / 1000)
    frame_shift = int(fs * FRAME_SHIFT_MS / 1000)
    num_frames = int(np.ceil(len(signal) / frame_shift))
    
    ste = np.zeros(num_frames)
    zcr = np.zeros(num_frames)
    times = np.zeros(num_frames)
    window = np.hamming(frame_len)
    
    # Duyệt từng khung, áp dụng cửa sổ và tính toán đặc trưng
    for i in range(num_frames):
        start_idx = i * frame_shift
        end_idx = min(start_idx + frame_len, len(signal))
        frame = signal[start_idx:end_idx]
        
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)), 'constant')
            
        frame = frame * window
        ste[i] = np.sum(frame ** 2)
        zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_len)
        times[i] = start_idx / fs
        
    if np.max(ste) > 0: ste = ste / np.max(ste)
    if np.max(zcr) > 0: zcr = zcr / np.max(zcr)
    
    return ste, zcr, times

def calculate_f0_autocorr(signal, fs):
    """
    Chức năng: Trích xuất F0 bằng phương pháp Tự tương quan.
    Đầu vào: signal (numpy array), fs (int)
    Đầu ra: f0, t_f0 (numpy arrays)
    """
    frame_len = int(fs * FRAME_LEN_MS / 1000)
    frame_shift = int(fs * FRAME_SHIFT_MS / 1000)
    num_frames = int(np.ceil(len(signal) / frame_shift))
    
    f0 = np.zeros(num_frames)
    t_f0 = np.zeros(num_frames)
    min_lag = int(fs / 400)
    max_lag = int(fs / 70)
    
    # Tính tự tương quan để tìm chu kỳ tuần hoàn (đỉnh của hàm tự tương quan)
    for i in range(num_frames):
        start = i * frame_shift
        end = min(start + frame_len, len(signal))
        frame = signal[start:end]
        
        if len(frame) == frame_len:
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:] 
            if len(corr) > max_lag:
                peak_idx = np.argmax(corr[min_lag:max_lag]) + min_lag
                f0[i] = fs / peak_idx
        t_f0[i] = start / fs
        
    f0 = np.where(f0 > 400, 0, f0)
    return f0, t_f0

def find_intersection(mu1, std1, mu2, std2):
    """
    Chức năng: Tìm ngưỡng phân loại bằng cách giải phương trình giao điểm 2 phân bố chuẩn.
    Đầu vào: Trung bình và độ lệch chuẩn của 2 nhóm (float)
    Đầu ra: Ngưỡng tối ưu (float)
    """
    a = 1.0/(2*std1**2) - 1.0/(2*std2**2)
    b = mu2/(std2**2) - mu1/(std1**2)
    c = mu1**2 /(2*std1**2) - mu2**2 /(2*std2**2) - np.log(std2/std1)
    
    roots = np.roots([a, b, c])
    for root in roots:
        if min(mu1, mu2) <= root <= max(mu1, mu2):
            return root
    return roots[0]

def smooth_boundaries(flags, min_sil_frames):
    """
    Chức năng: Xóa các khoảng lặng và tiếng nói ảo quá ngắn (hậu xử lý).
    Đầu vào: cờ nhãn ban đầu (numpy array), số khung tối thiểu (int)
    Đầu ra: cờ nhãn đã mượt (numpy array)
    """
    smoothed = flags.copy()
    
    # 1. Xóa các điểm 1 đơn độc
    for i in range(1, len(smoothed) - 1):
        if smoothed[i-1] == 0 and smoothed[i] == 1 and smoothed[i+1] == 0:
            smoothed[i] = 0
            
    # 2. Hợp nhất các khoảng 0 (Silence) ngắn hơn 200ms thành Speech (1)
    zero_count = 0
    zero_start = -1
    for i in range(len(smoothed)):
        if smoothed[i] == 0:
            if zero_count == 0: zero_start = i
            zero_count += 1
        else:
            if 0 < zero_count < min_sil_frames:
                smoothed[zero_start:i] = 1
            zero_count = 0
            
    if 0 < zero_count < min_sil_frames:
        smoothed[zero_start:len(smoothed)] = 1
        
    return smoothed

def get_true_speech_boundaries(intervals):
    """
    Chức năng: Gộp v và uv thành 1 đoạn Speech liên tục để lấy biên thực tế (đỏ).
    Đầu vào: intervals (list)
    Đầu ra: Danh sách các mốc thời gian bắt đầu/kết thúc Speech (list)
    """
    bounds = []
    is_speech = False
    for start, end, label in intervals:
        if label != 'sil':
            if not is_speech:
                bounds.append(start)  # Bắt đầu Speech
                is_speech = True
        else:
            if is_speech:
                bounds.append(start)  # Kết thúc Speech
                is_speech = False
    if is_speech:
        bounds.append(intervals[-1][1])
    return bounds

def calculate_errors(true_bounds, pred_bounds):
    """
    Chức năng: Tính sai số MAE và RMSE (bằng ms) bằng thuật toán ghép cặp (Nearest Neighbor).
    Đầu vào: true_bounds (biên đỏ), pred_bounds (biên xanh)
    Đầu ra: mae, rmse (float) đơn vị mili-giây
    """
    if not true_bounds or not pred_bounds:
        return 0.0, 0.0
        
    errors = []
    for tb in true_bounds:
        # Tìm biên dự đoán (xanh) nằm gần nhất với biên thực tế (đỏ) hiện tại
        closest_pb = min(pred_bounds, key=lambda pb: abs(pb - tb))
        error_ms = abs(tb - closest_pb) * 1000 # Đổi giây sang ms
        errors.append(error_ms)
        
    errors_arr = np.array(errors)
    mae = np.mean(errors_arr)                      # Mean Absolute Error
    rmse = np.sqrt(np.mean(errors_arr ** 2))       # Root Mean Squared Error
    
    return mae, rmse

def plot_results(signal, fs, t_sig, times, ste, zcr, flags, intervals, wav_name, fig_index, screen_w, screen_h):
    """
    Chức năng: Trực quan hóa dữ liệu, in sai số và ép vị trí vào 4 góc màn hình.
    """
    f0, t_f0 = calculate_f0_autocorr(signal, fs)
    
    # 1. Trích xuất danh sách biên để tính sai số
    true_bounds = get_true_speech_boundaries(intervals)
    
    diff = np.diff(np.insert(flags, 0, 0))
    pred_bounds = [times[i] for i, d in enumerate(diff) if d == 1 or d == -1]
    if flags[-1] == 1: pred_bounds.append(times[-1])
        
    mae, rmse = calculate_errors(true_bounds, pred_bounds)

    # 2. Khởi tạo và thiết lập vị trí cửa sổ (TkAgg)
    fig = plt.figure(figsize=(9, 5))
    plt.suptitle(f'Kết quả {wav_name} | MAE: {mae:.1f}ms - RMSE: {rmse:.1f}ms')
    
    mgr = plt.get_current_fig_manager()
    try:
        w, h = screen_w // 2, int(screen_h // 2.15)
        x, y = (fig_index % 2) * w, (fig_index // 2) * h
        mgr.window.geometry(f"{w}x{h}+{x}+{y}")
    except Exception:
        pass 
    
    # ============ TẦNG 1: INTERMEDIATE SUBPLOT ============
    plt.subplot(2, 1, 1)
    plt.plot(t_sig, signal, label='Normalized Signal', color='lightgray')
    plt.plot(times, ste, label='STE', color='blue', linewidth=1.5)
    plt.plot(times, zcr, label='ZCR', color='green', linewidth=1.5)
    plt.title('Đặc trưng trung gian')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # ============ TẦNG 2: RESULT SUBPLOT ============
    plt.subplot(2, 1, 2)
    plt.plot(t_sig, signal, label='Signal', color='lightgray')
    
    f0_plot = np.where(f0 > 0, f0, np.nan) 
    f0_norm = (f0_plot - np.nanmin(f0_plot)) / (np.nanmax(f0_plot) - np.nanmin(f0_plot)) * np.max(np.abs(signal))
    plt.plot(t_f0, f0_norm, 'r.', label='F0 Contour', markersize=3)
    
    # Vẽ biên thực tế (ĐỎ)
    for tb in true_bounds:
        plt.axvline(x=tb, color='red', linestyle='-', linewidth=2, alpha=0.7)
            
    # Vẽ biên dự đoán (XANH)
    for pb in pred_bounds:
        plt.axvline(x=pb, color='blue', linestyle='-', linewidth=2)
            
    plt.title('So sánh biên thời gian và F0')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude / F0 (Scaled)')
    
    import matplotlib.lines as mlines
    red_line = mlines.Line2D([], [], color='red', linestyle='-', label='Chuẩn (Đỏ)')
    blue_line = mlines.Line2D([], [], color='blue', linestyle='-', label='Dự đoán (Xanh)')
    f0_dot = mlines.Line2D([], [], color='red', marker='.', linestyle='None', label='F0')
    plt.legend(handles=[red_line, blue_line, f0_dot], loc='upper right')
    
    plt.tight_layout()

def main():
    # ==== 1. HUẤN LUYỆN (TRAINING) ====
    train_dir = 'TinHieuHuanLuyen' 
    speech_ste, silence_ste = [], []
    
    print("--- 1. HUẤN LUYỆN (TÌM PHÂN BỐ CHUẨN) ---")
    for file in os.listdir(train_dir):
        if file.endswith('.wav'):
            wav_path = os.path.join(train_dir, file)
            lab_path = os.path.join(train_dir, file.replace('.wav', '.lab'))
            
            signal, fs, t_sig = SharedFunctions.readAudio(wav_path)
            intervals = read_lab_file(lab_path)
            ste, _, times = extract_features(signal, fs)
            
            for i, t in enumerate(times):
                label = get_frame_label(t, intervals)
                if label == 1: speech_ste.append(ste[i])
                else: silence_ste.append(ste[i])
                    
    mu_sp, std_sp = np.mean(speech_ste), np.std(speech_ste)
    mu_sil, std_sil = np.mean(silence_ste), np.std(silence_ste)
    threshold = find_intersection(mu_sil, std_sil, mu_sp, std_sp)
    print(f"=> Ngưỡng STE tối ưu (Threshold): {threshold:.4f}\n")
    
    # ==== 2. KIỂM THỬ (TESTING) ====
    test_dir = 'TinHieuKiemThu'
    min_sil_frames = int(MIN_SILENCE_MS / FRAME_SHIFT_MS)
    
    # Lấy độ phân giải để neo 4 góc
    try:
        root = tk.Tk()
        root.withdraw()
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
    except Exception:
        screen_w, screen_h = 1920, 1080
    
    print("--- 2. XUẤT KẾT QUẢ KIỂM THỬ VÀ SAI SỐ (CHỜ HIỂN THỊ ĐỒ THỊ) ---")
    fig_index = 0
    for file in os.listdir(test_dir):
        if file.endswith('.wav'):
            wav_path = os.path.join(test_dir, file)
            lab_path = os.path.join(test_dir, file.replace('.wav', '.lab'))
            
            signal, fs, t_sig = SharedFunctions.readAudio(wav_path)
            intervals = read_lab_file(lab_path)
            ste, zcr, times = extract_features(signal, fs)
            
            flags = np.where(ste > threshold, 1, 0)
            smoothed_flags = smooth_boundaries(flags, min_sil_frames)
            
            plot_results(signal, fs, t_sig, times, ste, zcr, smoothed_flags, intervals, file, fig_index, screen_w, screen_h)
            fig_index += 1

    plt.show()

if __name__ == "__main__":
    main()