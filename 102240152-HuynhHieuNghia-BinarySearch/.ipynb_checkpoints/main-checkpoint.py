import sys
import os
import wave
import math
import numpy as np
import matplotlib.pyplot as pyplot

# --- CONSTANTS ---
FRAME_SIZE = 25
FRAME_STEP = 10
MIN_SILENCE = 200

# --- DIRECTORY SETUP ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- STEP 1: FEATURE EXTRACTION ---
def readAudio(path):
    with wave.open(path, 'rb') as WavFile:
        SampleRate = WavFile.getframerate()            
        SampleAmount = WavFile.getnframes()            
        RawData = WavFile.readframes(SampleAmount)    

        Signal = np.frombuffer(RawData, dtype=np.int16)
        MaxAmp = np.max(np.abs(Signal))
        if MaxAmp > 0:
            SignalNormalized = Signal / MaxAmp
        else:
            SignalNormalized = Signal                  

        Duration = SampleAmount / SampleRate
        TimeAxis = np.linspace(0, Duration, num=SampleAmount)
        return SignalNormalized, SampleRate, TimeAxis

def frameWindow(SignalNormalized, SampleRate):
    FrameLength = int(SampleRate * (FRAME_SIZE / 1000))     
    FrameStep = int(SampleRate * (FRAME_STEP / 1000))       
    SignalLength = len(SignalNormalized)

    if (SignalLength < FrameLength):
        FrameNumber = 1
    else:
        FrameNumber = 1 + int(np.floor((SignalLength - FrameLength) / FrameStep))

    Window = np.hamming(FrameLength)
    Frames = []
    
    for i in range(FrameNumber):
        StartID = i * FrameStep
        EndID = StartID + FrameLength
        CurrentFrame = SignalNormalized[StartID:EndID]

        if (len(CurrentFrame) < FrameLength):
            PaddingLength = FrameLength - len(CurrentFrame)
            CurrentFrame = np.pad(CurrentFrame, (0, PaddingLength), 'constant')

        Frames.append(CurrentFrame * Window)

    return np.array(Frames)

def calculate_STE(Frames):
    return np.sum(Frames ** 2, axis=1)

def calculate_MA(Frames):
    return np.sum(np.abs(Frames), axis=1)

def calculate_ZCR(Frames):
    Signs = np.sign(Frames)
    SignChanges = np.abs(np.diff(Signs, axis=1)) > 0
    return np.sum(SignChanges, axis=1)

# --- HELPER FUNCTIONS ---
def read_lab_boundaries(lab_path):
    raw_segments = []
    with open(lab_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    label = parts[2].lower()
                    if label in ['v', 'uv']:
                        raw_segments.append([start_time, end_time])
                except ValueError:
                    continue

    if not raw_segments: return []

    merged_boundaries = []
    current_start = raw_segments[0][0]
    current_end = raw_segments[0][1]

    for i in range(1, len(raw_segments)):
        next_start = raw_segments[i][0]
        next_end = raw_segments[i][1]
        if abs(next_start - current_end) < 0.001:
            current_end = next_end
        else:
            merged_boundaries.append([current_start, current_end])
            current_start = next_start
            current_end = next_end

    merged_boundaries.append([current_start, current_end])
    return merged_boundaries

def get_groundtruth_mask(true_boundaries, num_frames, frame_step_s):
    """Converts boundary timestamps into a boolean array of frames."""
    mask = np.zeros(num_frames, dtype=bool)
    for s, e in true_boundaries:
        start_idx = int(s / frame_step_s)
        end_idx = int(e / frame_step_s)
        mask[start_idx:end_idx] = True
    return mask

def get_boundaries_from_bool(speech_frames, frame_step_s):
    boundaries = []
    in_speech = False
    start_time = 0
    for i, is_speech in enumerate(speech_frames):
        if is_speech and not in_speech:
            start_time = i * frame_step_s
            in_speech = True
        elif not is_speech and in_speech:
            end_time = i * frame_step_s
            boundaries.append([round(start_time, 3), round(end_time, 3)])
            in_speech = False
            
    if in_speech:
        end_time = len(speech_frames) * frame_step_s
        boundaries.append([round(start_time, 3), round(end_time, 3)])
    return boundaries

# --- STEP 5: POST-PROCESSING (Gộp khoảng lặng ảo) ---
def smoothingLogic(SpeechFrames):
    MinSilenceFrames = MIN_SILENCE // FRAME_STEP    
    SmoothedSpeech = np.copy(SpeechFrames)
    InSilence = False
    SilenceStart = 0

    for i in range (len(SmoothedSpeech)):
        if not SmoothedSpeech[i] and not InSilence:
            InSilence = True
            SilenceStart = i
        elif SmoothedSpeech[i] and InSilence:
            InSilence = False
            SilenceLength = i - SilenceStart
            if SilenceLength < MinSilenceFrames:
                SmoothedSpeech[SilenceStart:i] = True

    return SmoothedSpeech

# --- STEP 3 & 4: RABINER ALGORITHM (STE & ZCR) ---
def apply_vad_pipeline(STE, ZCR, T_upper, T_lower, T_zcr, frame_step_s):
    # Step 3: Phân đoạn sơ bộ bằng STE (State Machine)
    rough_segments = []
    in_speech = False
    start_idx = 0
    
    for i, e in enumerate(STE):
        if not in_speech:
            # Trigger speech start
            if e > T_upper:
                start_idx = i
                # Look backwards to find exactly where energy dropped below T_lower
                while start_idx > 0 and STE[start_idx] > T_lower:
                    start_idx -= 1
                in_speech = True
        else:
            # Trigger speech end
            if e < T_lower:
                end_idx = i
                rough_segments.append([start_idx, end_idx])
                in_speech = False
                
    if in_speech:
        rough_segments.append([start_idx, len(STE)-1])

    # Step 4: Tinh chỉnh biên bằng ZCR (Mở rộng 25 frames / 250ms)
    refined_segments = []
    for start_idx, end_idx in rough_segments:
        # Check backwards
        lookback_start = max(0, start_idx - 25)
        zcr_back = ZCR[lookback_start:start_idx]
        if np.sum(zcr_back > T_zcr) >= 3: # If at least 3 frames cross ZCR threshold
            valid_indices = np.where(zcr_back > T_zcr)[0]
            if len(valid_indices) > 0:
                start_idx = lookback_start + valid_indices[0]
                
        # Check forwards
        lookforward_end = min(len(ZCR), end_idx + 25)
        zcr_fwd = ZCR[end_idx:lookforward_end]
        if np.sum(zcr_fwd > T_zcr) >= 3:
            valid_indices = np.where(zcr_fwd > T_zcr)[0]
            if len(valid_indices) > 0:
                end_idx = end_idx + valid_indices[-1]
                
        refined_segments.append([start_idx, end_idx])

    # Convert refined segments into a boolean array
    speech_frames = np.zeros(len(STE), dtype=bool)
    for s_idx, e_idx in refined_segments:
        speech_frames[s_idx:e_idx] = True

    # Step 5: Hậu xử lý
    smoothed_speech = smoothingLogic(speech_frames)
    
    return get_boundaries_from_bool(smoothed_speech, frame_step_s)

# --- STEP 6: EVALUATION & VISUALIZATION ---
def evaluate_and_print_terminal(results_dict, T_upper, T_lower, T_zcr):
    print(f"\n[THRESHOLDS] T_upper: {T_upper:.5f} | T_lower: {T_lower:.5f} | T_zcr: {T_zcr:.5f}\n") 
    print(f"{'[Result of program]':<15}")
    print(f"{'File':<40} {'#Ground truth borders':>22} {'#Algorithm borders':>19} {'MAE(ms)':>10} {'RMSE(ms)':>10}")
    
    total_mae, total_rmse = 0, 0
    count = 0
    
    for filename, data in results_dict.items():
        true_b = data['true']
        pred_b = data['pred']
        
        num_true = len(true_b) * 2  
        num_pred = len(pred_b) * 2

        print(f"{filename}")
        print(f"  ground truth : {true_b}")
        print(f"  algorithm: {pred_b}")
        
        if len(true_b) > 0 and len(pred_b) > 0:
            start_err_ms = abs(pred_b[0][0] - true_b[0][0]) * 1000
            end_err_ms = abs(pred_b[0][1] - true_b[0][1]) * 1000
            mae = (start_err_ms + end_err_ms) / 2
            rmse = math.sqrt((start_err_ms**2 + end_err_ms**2) / 2)
            
            total_mae += mae
            total_rmse += rmse
            count += 1
            
            print("-" * 32)
            print(f"{filename:<30} {num_true:>22} {num_pred:>19} {mae:>19.2f} {rmse:>10.2f}")
    
    if count > 0:
        print(f"{'AVERAGE':<73} {total_mae/count:>19.2f} {total_rmse/count:>10.2f}")

def plotFinalFigure(TimeAxes, NormalizedSignals, FrameTimeAxes, STEs, MAs, ZCRs, TrueBoundariesList, PredBoundariesList, Titles, T_upper, T_lower):
    fig, axes = pyplot.subplots(4, 2, figsize=(18, 12))
    fig.suptitle("Testing Set Evaluation - Rabiner VAD Algorithm", fontsize=18, fontweight='bold')
    grid_positions = [(0, 0), (0, 1), (2, 0), (2, 1)]

    for i in range(len(Titles)):
        row, col = grid_positions[i]
        ax_main = axes[row, col]       
        ax_bound = axes[row+1, col]    

        STE_norm = STEs[i] / np.max(STEs[i]) if np.max(STEs[i]) > 0 else STEs[i]
        MA_norm = MAs[i] / np.max(MAs[i]) if np.max(MAs[i]) > 0 else MAs[i]
        ZCR_norm = ZCRs[i] / np.max(ZCRs[i]) if np.max(ZCRs[i]) > 0 else ZCRs[i]

        # Draw Features
        ax_main.plot(TimeAxes[i], NormalizedSignals[i], color='lightgray', linewidth=0.5, label='Signal')
        ax_main.plot(FrameTimeAxes[i], STE_norm, color='blue', linewidth=1.5, label='STE')
        ax_main.plot(FrameTimeAxes[i], MA_norm, color='orange', linewidth=1.5, label='MA')
        ax_main.plot(FrameTimeAxes[i], ZCR_norm, color='green', linewidth=1.5, label='ZCR')
        
        # Plot T_upper and T_lower (normalized visually to match the STE scale)
        norm_T_upper = T_upper / np.max(STEs[i]) if np.max(STEs[i]) > 0 else T_upper
        norm_T_lower = T_lower / np.max(STEs[i]) if np.max(STEs[i]) > 0 else T_lower
        ax_main.axhline(norm_T_upper, color='purple', linestyle='--', linewidth=1.5, label='T_upper')
        ax_main.axhline(norm_T_lower, color='cyan', linestyle='--', linewidth=1.5, label='T_lower')
        
        ax_main.set_title(f"Audio: {Titles[i]}")
        ax_main.set_ylabel("Amplitude")
        ax_main.set_ylim(-1.1, 1.1)
        ax_main.grid(True, which='both', linestyle='--', linewidth=0.7)
        if i == 0: 
            ax_main.legend(loc="upper right", fontsize=9)

        # Draw Boundaries
        ax_bound.plot(TimeAxes[i], NormalizedSignals[i], color='lightgray', linewidth=0.5)
        for start, end in TrueBoundariesList[i]:
            ax_bound.axvline(start, color='red', linewidth=1.5, label='Ground truth (Red)')
            ax_bound.axvline(end, color='red', linewidth=1.5)
        for start, end in PredBoundariesList[i]:
            ax_bound.axvline(start, color='blue', linewidth=1.5, label='Prediction (Blue)')
            ax_bound.axvline(end, color='blue', linewidth=1.5)

        ax_bound.set_ylabel("Amplitude")
        ax_bound.set_xlabel("Time (s)")
        ax_bound.set_ylim(-1.1, 1.1)
        ax_bound.grid(True, which='both', linestyle='--', linewidth=0.7)
        handles, labels = ax_bound.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        if i == 0:  
            ax_bound.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', fontsize=9)
        ax_main.sharex(ax_bound)

    pyplot.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0, w_pad=4.0)
    pyplot.show()

# --- MAIN EXECUTION ---
def run_pipeline():
    # Directories for Phase 1 (Training)
    train_audio = [
        "TinHieuHuanLuyen/phone_F1.wav",
        "TinHieuHuanLuyen/phone_M1.wav",
        "TinHieuHuanLuyen/studio_F1.wav",
        "TinHieuHuanLuyen/studio_M1.wav"
    ]
    train_lab = [
        "TinHieuHuanLuyen/phone_F1.lab",
        "TinHieuHuanLuyen/phone_M1.lab",
        "TinHieuHuanLuyen/studio_F1.lab",
        "TinHieuHuanLuyen/studio_M1.lab"
    ]
    
    # Directories for Phase 2 (Testing)
    test_audio = [
        "TinHieuKiemThu/phone_F2.wav",
        "TinHieuKiemThu/phone_M2.wav",
        "TinHieuKiemThu/studio_F2.wav",
        "TinHieuKiemThu/studio_M2.wav"
    ]
    test_lab = [
        "TinHieuKiemThu/phone_F2.lab",
        "TinHieuKiemThu/phone_M2.lab",
        "TinHieuKiemThu/studio_F2.lab",
        "TinHieuKiemThu/studio_M2.lab"
    ]

    print("Phase 1: Analyzing Training Set to Calculate Thresholds...")
    all_sil_ste, all_sil_zcr, all_speech_ste = [], [], []
    frame_step_seconds = FRAME_STEP / 1000.0

    for wav_path, lab_path in zip(train_audio, train_lab):
        full_wav = os.path.join(parent_dir, wav_path)
        full_lab = os.path.join(parent_dir, lab_path)
        
        SignalNormalized, SampleRate, _ = readAudio(full_wav)
        Frames = frameWindow(SignalNormalized, SampleRate)
        STE = calculate_STE(Frames)
        ZCR = calculate_ZCR(Frames)
        true_boundaries = read_lab_boundaries(full_lab)
        
        # Isolate silence vs speech frames using ground truth
        mask = get_groundtruth_mask(true_boundaries, len(STE), frame_step_seconds)
        all_sil_ste.extend(STE[~mask])
        all_sil_zcr.extend(ZCR[~mask])
        all_speech_ste.extend(STE[mask])

    # Step 2: Calculate Thresholds Statistically
    mu_sil_ste = np.mean(all_sil_ste)
    std_sil_ste = np.std(all_sil_ste)
    mu_sil_zcr = np.mean(all_sil_zcr)
    std_sil_zcr = np.std(all_sil_zcr)
    
    T_lower = mu_sil_ste + 2 * std_sil_ste
    T_upper = mu_sil_ste + 5 * std_sil_ste
    T_zcr = mu_sil_zcr + 2 * std_sil_zcr
    
    print("Phase 2: Applying Algorithm to Testing Set...")
    TimeAxes, NormalizedSignals, FrameTimeAxes = [], [], []
    STEs, MAs, ZCRs, TrueBoundariesList, PredBoundariesList, Titles = [], [], [], [], [], []
    evaluation_results = {}

    for wav_path, lab_path in zip(test_audio, test_lab):
        full_wav = os.path.join(parent_dir, wav_path)
        full_lab = os.path.join(parent_dir, lab_path)
        filename = full_wav.split(os.sep)[-1]
        
        SignalNormalized, SampleRate, TimeAxis = readAudio(full_wav)
        Frames = frameWindow(SignalNormalized, SampleRate)
        STE = calculate_STE(Frames)
        MA = calculate_MA(Frames)
        ZCR = calculate_ZCR(Frames)
        FrameTimeAxis = np.arange(len(Frames)) * frame_step_seconds
        
        true_boundaries = read_lab_boundaries(full_lab)
        
        # Apply the Rabiner Pipeline
        pred_boundaries = apply_vad_pipeline(STE, ZCR, T_upper, T_lower, T_zcr, frame_step_seconds)
        
        # Store for Output
        evaluation_results[filename] = {'true': true_boundaries, 'pred': pred_boundaries}
        Titles.append(filename)
        TimeAxes.append(TimeAxis)
        NormalizedSignals.append(SignalNormalized)
        FrameTimeAxes.append(FrameTimeAxis)
        STEs.append(STE)
        MAs.append(MA)
        ZCRs.append(ZCR)
        TrueBoundariesList.append(true_boundaries)
        PredBoundariesList.append(pred_boundaries)

    print("="*60)
    evaluate_and_print_terminal(evaluation_results, T_upper, T_lower, T_zcr)

    plotFinalFigure(
        TimeAxes, NormalizedSignals, FrameTimeAxes, 
        STEs, MAs, ZCRs, TrueBoundariesList, PredBoundariesList, Titles, T_upper, T_lower
    )

if __name__ == "__main__":
    run_pipeline()