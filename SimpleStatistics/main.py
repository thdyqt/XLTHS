import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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
    Đầu ra: intervals (list) - Danh sách các tuple chứa (start_time, end_time, label)
    """
    intervals = []
    with open(lab_path, 'r') as f:
        # Đọc từng dòng, bỏ qua 2 dòng chứa chuỗi F0mean và F0std ở cuối
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and parts[0] not in ['F0mean', 'F0std']:
                start, end, label = float(parts[0]), float(parts[1]), parts[2]
                intervals.append((start, end, label))
    return intervals

def get_frame_label(time_sec, intervals):
    """
    Chức năng: Đối chiếu 1 mốc thời gian với dữ liệu gốc để xem thuộc Speech hay Silence.
    Đầu vào: time_sec (float) - Mốc thời gian của khung hiện tại
             intervals (list) - Dữ liệu nhãn lấy từ hàm read_lab_file
    Đầu ra: int - 0 nếu là khoảng lặng ('sil'), 1 nếu là tiếng nói ('v' hoặc 'uv')
    """
    # Quét qua toàn bộ các đoạn nhãn, nếu thời gian nằm trong đoạn nào thì lấy nhãn đó
    for start, end, label in intervals:
        if start <= time_sec <= end:
            return 0 if label == 'sil' else 1
    return 0

def extract_features(signal, fs):
    """
    Chức năng: Chia khung tín hiệu, áp dụng cửa sổ Hamming, tính giá trị STE và ZCR.
    Đầu vào: signal (numpy array) - Mảng tín hiệu biên độ
             fs (int) - Tần số lấy mẫu (Sample rate)
    Đầu ra: ste, zcr, times (cả 3 là numpy array) - Các đặc trưng đã chuẩn hóa và trục thời gian
    """
    # Đổi tham số thời gian từ ms sang số lượng mẫu (samples)
    frame_len = int(fs * FRAME_LEN_MS / 1000)
    frame_shift = int(fs * FRAME_SHIFT_MS / 1000)
    
    num_frames = int(np.ceil(len(signal) / frame_shift))
    ste = np.zeros(num_frames)
    zcr = np.zeros(num_frames)
    times = np.zeros(num_frames)
    window = np.hamming(frame_len)
    
    # Duyệt qua từng khung tín hiệu (mỗi block xử lý 5-10 dòng logic cắt khung)
    for i in range(num_frames):
        start_idx = i * frame_shift
        end_idx = min(start_idx + frame_len, len(signal))
        frame = signal[start_idx:end_idx]
        
        # Thêm padding số 0 nếu khung cuối cùng bị thiếu mẫu
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)), 'constant')
            
        frame = frame * window # Áp dụng cửa sổ Hamming chống rò rỉ phổ
        
        # Tính toán đặc trưng cho khung hiện tại
        ste[i] = np.sum(frame ** 2)
        zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_len)
        times[i] = start_idx / fs
        
    # Chuẩn hóa giá trị STE và ZCR về dải [0, 1]
    if np.max(ste) > 0: ste = ste / np.max(ste)
    if np.max(zcr) > 0: zcr = zcr / np.max(zcr)
    
    return ste, zcr, times

def calculate_f0_autocorr(signal, fs):
    """
    Chức năng: Tính đường F0 contour bằng phương pháp tự tương quan (Autocorrelation) dùng thuần Numpy.
    Đầu vào: signal (numpy array) - Tín hiệu, fs (int) - Tần số lấy mẫu
    Đầu ra: f0 (numpy array) - Mảng giá trị tần số cơ bản, t_f0 - Trục thời gian
    """
    frame_len = int(fs * FRAME_LEN_MS / 1000)
    frame_shift = int(fs * FRAME_SHIFT_MS / 1000)
    num_frames = int(np.ceil(len(signal) / frame_shift))
    
    f0 = np.zeros(num_frames)
    t_f0 = np.zeros(num_frames)
    
    # Giới hạn dải tần số F0 tìm kiếm từ 70Hz đến 400Hz (giọng người chuẩn)
    min_lag = int(fs / 400)
    max_lag = int(fs / 70)
    
    # Tính tự tương quan cho từng khung
    for i in range(num_frames):
        start = i * frame_shift
        end = min(start + frame_len, len(signal))
        frame = signal[start:end]
        
        # Chỉ tính F0 cho các khung đủ độ dài
        if len(frame) == frame_len:
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:] # Lấy nửa sau của mảng đối xứng
            
            # Tìm đỉnh cao nhất trong dải lag giới hạn
            if len(corr) > max_lag:
                peak_idx = np.argmax(corr[min_lag:max_lag]) + min_lag
                f0[i] = fs / peak_idx
                
        t_f0[i] = start / fs
        
    # Lọc nhiễu: Đặt các giá trị F0 bất thường hoặc nền tảng về 0
    f0 = np.where(f0 > 400, 0, f0)
    return f0, t_f0

def find_intersection(mu1, std1, mu2, std2):
    """
    Chức năng: Tìm ngưỡng (Threshold) tối ưu bằng cách giải phương trình giao điểm 2 phân bố chuẩn.
    Đầu vào: mu1, std1 (float) - Trung bình và lệch chuẩn nhóm Silence
             mu2, std2 (float) - Trung bình và lệch chuẩn nhóm Speech
    Đầu ra: root (float) - Giá trị ngưỡng giao cắt tối ưu
    """
    # Tính các hệ số a, b, c của phương trình bậc 2
    a = 1.0/(2*std1**2) - 1.0/(2*std2**2)
    b = mu2/(std2**2) - mu1/(std1**2)
    c = mu1**2 /(2*std1**2) - mu2**2 /(2*std2**2) - np.log(std2/std1)
    
    # Giải phương trình tìm nghiệm
    roots = np.roots([a, b, c])
    
    # Trả về nghiệm có giá trị nằm giữa 2 mốc trung bình
    for root in roots:
        if min(mu1, mu2) <= root <= max(mu1, mu2):
            return root
    return roots[0]

def smooth_boundaries(flags, min_sil_frames):
    """
    Chức năng: Hậu xử lý xóa các số 1 đơn lẻ và hợp nhất các khoảng lặng ảo (dưới 200ms).
    Đầu vào: flags (numpy array) - Mảng nhị phân 0/1 ban đầu, min_sil_frames (int) - Số khung tối thiểu
    Đầu ra: smoothed (numpy array) - Mảng nhị phân sau khi đã làm mượt
    """
    smoothed = flags.copy()
    
    # 1. Quét tìm và xóa các khung 1 đơn độc bị kẹp giữa các số 0
    for i in range(1, len(smoothed) - 1):
        if smoothed[i-1] == 0 and smoothed[i] == 1 and smoothed[i+1] == 0:
            smoothed[i] = 0
            
    # 2. Xử lý các chuỗi 0 (Khoảng lặng) nếu có thời lượng ngắn hơn ngưỡng min_sil_frames
    zero_count = 0
    zero_start = -1
    for i in range(len(smoothed)):
        if smoothed[i] == 0:
            if zero_count == 0: zero_start = i
            zero_count += 1
        else:
            if 0 < zero_count < min_sil_frames:
                smoothed[zero_start:i] = 1  # Đảo thành Speech vì quá ngắn
            zero_count = 0
            
    # Xử lý dọn dẹp chuỗi 0 nếu rơi vào đoạn cuối cùng của file
    if 0 < zero_count < min_sil_frames:
        smoothed[zero_start:len(smoothed)] = 1
        
    return smoothed

def plot_results(signal, fs, t_sig, times, ste, zcr, flags, intervals, wav_name):
    """
    Chức năng: Dựng dữ liệu lên biểu đồ (Không gọi plt.show() tại đây để tránh chặn chương trình).
    Đầu vào: Các dữ liệu tín hiệu, đặc trưng, cờ nhãn, trục thời gian và tên file.
    Đầu ra: None (Hàm thay đổi trạng thái giao diện của matplotlib)
    """
    # Tính toán F0 tự code thuần Numpy
    f0, t_f0 = calculate_f0_autocorr(signal, fs)

    # Khởi tạo 1 cửa sổ mới cho từng tín hiệu kiểm thử
    plt.figure(figsize=(10, 6))
    plt.suptitle(f'Kết quả thực nghiệm thuật toán Simple Statistics: {wav_name}')
    
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
    
    # Chuẩn hóa F0 để hiển thị đè lên đồ thị tín hiệu, bỏ qua các điểm bằng 0
    f0_plot = np.where(f0 > 0, f0, np.nan) 
    f0_norm = (f0_plot - np.nanmin(f0_plot)) / (np.nanmax(f0_plot) - np.nanmin(f0_plot)) * np.max(np.abs(signal))
    plt.plot(t_f0, f0_norm, 'r.', label='F0 Contour', markersize=3)
    
    # Vẽ biên thực tế (.lab) và biên dự đoán (flags)
    for start, end, label in intervals:
        if label != 'sil':
            plt.axvline(x=start, color='black', linestyle='--', alpha=0.7)
            plt.axvline(x=end, color='black', linestyle='--', alpha=0.7)
            
    diff = np.diff(np.insert(flags, 0, 0))
    for i, d in enumerate(diff):
        if d == 1 or d == -1: # Bắt đầu hoặc kết thúc vùng Speech
            plt.axvline(x=times[i], color='blue', linestyle='-', linewidth=2)
            
    plt.title('So sánh biên thời gian và F0')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude / F0 (Scaled)')
    
    # Định dạng Legend thủ công
    import matplotlib.lines as mlines
    black_line = mlines.Line2D([], [], color='black', linestyle='--', label='Chuẩn (LAB)')
    blue_line = mlines.Line2D([], [], color='blue', linestyle='-', label='Dự đoán')
    red_line = mlines.Line2D([], [], color='red', marker='.', linestyle='None', label='F0')
    plt.legend(handles=[black_line, blue_line, red_line], loc='upper right')
    
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
            ste, zcr, times = extract_features(signal, fs)
            
            # Gán giá trị STE vào mảng phân bố tương ứng
            for i, t in enumerate(times):
                label = get_frame_label(t, intervals)
                if label == 1:
                    speech_ste.append(ste[i])
                else:
                    silence_ste.append(ste[i])
                    
    mu_sp, std_sp = np.mean(speech_ste), np.std(speech_ste)
    mu_sil, std_sil = np.mean(silence_ste), np.std(silence_ste)
    threshold = find_intersection(mu_sil, std_sil, mu_sp, std_sp)
    print(f"=> Ngưỡng STE tối ưu (Threshold): {threshold:.4f}\n")
    
    # ==== 2. KIỂM THỬ (TESTING) ====
    test_dir = 'TinHieuKiemThu'
    min_sil_frames = int(MIN_SILENCE_MS / FRAME_SHIFT_MS)
    
    print("--- 2. XUẤT KẾT QUẢ KIỂM THỬ (CHỜ HIỂN THỊ ĐỒ THỊ) ---")
    for file in os.listdir(test_dir):
        if file.endswith('.wav'):
            wav_path = os.path.join(test_dir, file)
            lab_path = os.path.join(test_dir, file.replace('.wav', '.lab'))
            
            signal, fs, t_sig = SharedFunctions.readAudio(wav_path)
            intervals = read_lab_file(lab_path)
            ste, zcr, times = extract_features(signal, fs)
            
            # Phân loại và làm mượt
            flags = np.where(ste > threshold, 1, 0)
            smoothed_flags = smooth_boundaries(flags, min_sil_frames)
            
            # Vẽ lên buffer của plot (Chưa hiển thị ngay)
            plot_results(signal, fs, t_sig, times, ste, zcr, smoothed_flags, intervals, file)

    # Hiển thị tất cả 4 figure CÙNG MỘT LÚC ở bước cuối
    plt.show()

if __name__ == "__main__":
    main()