import sys
import os
import wave
import math
import numpy
import numpy as np
import matplotlib.pyplot as pyplot

# --- CONSTANTS ---
FRAME_SIZE = 25
FRAME_STEP = 10
MIN_SILENCE = 200

# --- DIRECTORY SETUP ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- FUNCTIONS ---
def readAudio(path):
    """Get the data of an .wav file for further processes (normalized Signal, sample rate, time axis)"""
    with wave.open(path, 'rb') as WavFile:
        SampleRate = WavFile.getframerate()            
        SampleAmount = WavFile.getnframes()            
        RawData = WavFile.readframes(SampleAmount)    

        Signal = numpy.frombuffer(RawData, dtype=numpy.int16)

        MaxAmp = numpy.max(numpy.abs(Signal))
        if MaxAmp > 0:
            SignalNormalized = Signal / MaxAmp
        else:
            SignalNormalized = Signal                  

        Duration = SampleAmount / SampleRate
        TimeAxis = numpy.linspace(0, Duration, num=SampleAmount)

        return SignalNormalized, SampleRate, TimeAxis

def plotWave(path):
    """Make graph of an .wav file using the data from function \"readAudio\" """
    SignalNormalized, SampleRate, TimeAxis = readAudio(path)

    pyplot.figure(figsize=(10,3))
    pyplot.plot(TimeAxis, SignalNormalized, color='darkgray', linewidth=0.5)

    Title = path.split('/')[-1]
    pyplot.title(f'Signal of: {Title}')
    pyplot.xlabel("Time (s)")
    pyplot.ylabel("Amplitude")

    pyplot.tight_layout()
    pyplot.show()

def frameWindow(SignalNormalized, SampleRate):
    """Frame the signal to the size required and to apply windowing to smoothen the edges of the frames""" 
    FrameLength = int(SampleRate * (FRAME_SIZE / 1000))     
    FrameStep = int(SampleRate * (FRAME_STEP / 1000))       

    SignalLength = len(SignalNormalized)

    if (SignalLength < FrameLength):
        FrameNumber = 1
    else:
        FrameNumber = 1 + int(numpy.floor((SignalLength - FrameLength) / FrameStep))

    Window = numpy.hamming(FrameLength)

    Frames = []
    for i in range(FrameNumber):
        StartID = i * FrameStep
        EndID = StartID + FrameLength

        CurrentFrame = SignalNormalized[StartID:EndID]

        if (len(CurrentFrame) < FrameLength):
            PaddingLength = FrameLength - len(CurrentFrame)
            CurrentFrame = numpy.pad(CurrentFrame, (0, PaddingLength), 'constant')

        WindowedFrame = CurrentFrame * Window
        
        Frames.append(WindowedFrame)

    return numpy.array(Frames)

def calculate_STE(Frames):
    """Calculate the STE value of frames"""
    return numpy.sum(Frames ** 2, axis=1)

def calculate_MA(Frames):
    """Calculate the MA value of frames"""
    return numpy.sum(numpy.abs(Frames), axis=1)

def calculate_ZCR(Frames):
    """Calculate the ZCR value of frames"""
    Signs = numpy.sign(Frames)
    SignChanges = numpy.abs(numpy.diff(Signs, axis=1)) > 0
    return numpy.sum(SignChanges, axis=1)

def smoothingLogic(SpeechFrames):
    """Apply smoothing logic onto speech regions"""
    MinSilenceFrames = MIN_SILENCE // FRAME_STEP    

    SmoothedSpeech = numpy.copy(SpeechFrames)
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

def calculateAccuracy(DetectedSpeech, GroundTruth):
    """Compare the program against ground truth in .lab files"""
    if len(DetectedSpeech) != len(GroundTruth):
        raise ValueError("Detected frames and ground truth frames must be identical in length.")

    CorrectFrames = numpy.sum(DetectedSpeech == GroundTruth)
    Accuracy = (CorrectFrames / len(GroundTruth)) * 100
    return Accuracy

def plotFinalFigure(TimeAxes, NormalizedSignals, FrameTimeAxes, STEs, MAs, ZCRs, TrueBoundariesList, PredBoundariesList, Titles):
    """Plot 4 audio files in a 4-corner grid: Overlapped features (top) and vertical boundaries (bottom)."""
    fig, axes = pyplot.subplots(4, 2, figsize=(18, 12))
    fig.suptitle("Speech Discrimination - Overlapped Features & Boundaries", fontsize=18, fontweight='bold')

    grid_positions = [(0, 0), (0, 1), (2, 0), (2, 1)]

    for i in range(len(Titles)):
        row, col = grid_positions[i]
        ax_main = axes[row, col]       
        ax_bound = axes[row+1, col]    

        STE_norm = STEs[i] / numpy.max(STEs[i]) if numpy.max(STEs[i]) > 0 else STEs[i]
        MA_norm = MAs[i] / numpy.max(MAs[i]) if numpy.max(MAs[i]) > 0 else MAs[i]
        ZCR_norm = ZCRs[i] / numpy.max(ZCRs[i]) if numpy.max(ZCRs[i]) > 0 else ZCRs[i]

        ax_main.plot(TimeAxes[i], NormalizedSignals[i], color='lightgray', linewidth=0.5, label='Signal')
        ax_main.plot(FrameTimeAxes[i], STE_norm, color='blue', linewidth=1.5, label='STE')
        ax_main.plot(FrameTimeAxes[i], MA_norm, color='orange', linewidth=1.5, label='MA')
        ax_main.plot(FrameTimeAxes[i], ZCR_norm, color='green', linewidth=1.5, label='ZCR')
        
        ax_main.set_title(f"Audio: {Titles[i]}")
        ax_main.set_ylabel("Amplitude")
        ax_main.set_ylim(-1.1, 1.1)
        ax_main.grid(True, which='both', linestyle='--', linewidth=0.7)
        
        if i == 0: 
            ax_main.legend(loc="upper right", fontsize=9)

        ax_bound.plot(TimeAxes[i], NormalizedSignals[i], color='lightgray', linewidth=0.5)
        
        for start, end in TrueBoundariesList[i]:
            ax_bound.axvline(start, color='red', linewidth=1.5, label='Chuẩn (Đỏ)')
            ax_bound.axvline(end, color='red', linewidth=1.5)
            
        for start, end in PredBoundariesList[i]:
            ax_bound.axvline(start, color='blue', linewidth=1.5, label='Dự đoán (Xanh)')
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

def read_lab_boundaries(lab_path):
    """Reads a .lab file, extracts 'v' and 'uv' segments, ignores 'sil', and merges contiguous speech blocks."""
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

    if not raw_segments:
        return []

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

def get_boundaries_from_bool(speech_frames, frame_step_s):
    """Converts the True/False array into [start_time, end_time] pairs."""
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

def find_threshold_binary_search(feature_array, true_boundaries, frame_step_ms=10, min_silence_ms=200):
    """Finds the optimal threshold T by matching predicted speech duration to ground truth."""
    total_true_duration_s = sum([end - start for start, end in true_boundaries])
    target_frames = int(total_true_duration_s / (frame_step_ms / 1000.0))
    
    low = numpy.min(feature_array)
    high = numpy.max(feature_array)
    best_T = low
    
    for _ in range(50): 
        mid = (low + high) / 2.0
        raw_speech = feature_array > mid
        smoothed_speech = smoothingLogic(raw_speech) 
        
        current_frames = numpy.sum(smoothed_speech)
        
        if current_frames > target_frames:
            low = mid
        elif current_frames < target_frames:
            high = mid
        else:
            best_T = mid
            break
            
        best_T = mid
        
    return best_T

def evaluate_and_print_terminal(results_dict):
    """Prints terminal output matching the required format."""
    print(f"{'[KIEM THU]':<15}")
    print(f"{'File':<20} {'#bien chuan':>11} {'#bien TT':>9} {'MAE(ms)':>9} {'RMSE(ms)':>9}")
    
    total_mae, total_rmse = 0, 0
    count = 0
    
    for filename, data in results_dict.items():
        true_b = data['true']
        pred_b = data['pred']
        
        num_true = len(true_b) * 2  
        num_pred = len(pred_b) * 2
        
        print(f"{filename}")
        print(f"  chuan : {true_b}")
        print(f"  thuat toan: {pred_b}")
        
        if len(true_b) > 0 and len(pred_b) > 0:
            start_err_ms = abs(pred_b[0][0] - true_b[0][0]) * 1000
            end_err_ms = abs(pred_b[0][1] - true_b[0][1]) * 1000
            
            mae = (start_err_ms + end_err_ms) / 2
            rmse = math.sqrt((start_err_ms**2 + end_err_ms**2) / 2)
            
            total_mae += mae
            total_rmse += rmse
            count += 1
            
            print(f"{filename:<20} {num_true:>11} {num_pred:>9} {mae:>9.2f} {rmse:>9.2f}")
    
    if count > 0:
        print(f"{'TRUNG BINH':<42} {total_mae/count:>9.2f} {total_rmse/count:>9.2f}")

def plot_boundary_comparison(TimeAxis, SignalNormalized, true_boundaries, pred_boundaries, Title="So sánh biên thời gian"):
    """Plot an individual boundary comparison graph."""
    fig, ax = pyplot.subplots(1, 1, figsize=(12, 4))
    
    ax.plot(TimeAxis, SignalNormalized, color='lightgray', linewidth=0.5)
    
    for start, end in true_boundaries:
        ax.axvline(start, color='red', linewidth=1.5, label='Chuẩn (Đỏ)')
        ax.axvline(end, color='red', linewidth=1.5)
        
    for start, end in pred_boundaries:
        ax.axvline(start, color='blue', linewidth=1.5, label='Dự đoán (Xanh)')
        ax.axvline(end, color='blue', linewidth=1.5)

    ax.set_title(Title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.1, 1.1)
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    
    pyplot.tight_layout()
    pyplot.show()

# --- PIPELINE EXECUTION ---
def run_pipeline():
    audio_files = [
        "TinHieuHuanLuyen/phone_F1.wav",
        "TinHieuHuanLuyen/phone_M1.wav",
        "TinHieuHuanLuyen/studio_F1.wav",
        "TinHieuHuanLuyen/studio_M1.wav"
    ]
    lab_files = [
        "TinHieuHuanLuyen/phone_F1.lab",
        "TinHieuHuanLuyen/phone_M1.lab",
        "TinHieuHuanLuyen/studio_F1.lab",
        "TinHieuHuanLuyen/studio_M1.lab"
    ]

    TimeAxes, NormalizedSignals, FrameTimeAxes = [], [], []
    STEs, MAs, ZCRs, TrueBoundariesList, PredBoundariesList, Titles = [], [], [], [], [], []
    
    evaluation_results = {}

    for wav_path, lab_path in zip(audio_files, lab_files):
        full_wav_path = os.path.join(parent_dir, wav_path)
        full_lab_path = os.path.join(parent_dir, lab_path)
        filename = full_wav_path.split(os.sep)[-1]
        print(f"Processing: {filename}...")

        SignalNormalized, SampleRate, TimeAxis = readAudio(full_wav_path)
        Frames = frameWindow(SignalNormalized, SampleRate)
        STE = calculate_STE(Frames)
        MA = calculate_MA(Frames)
        ZCR = calculate_ZCR(Frames)
        frame_step_seconds = FRAME_STEP / 1000.0
        FrameTimeAxis = np.arange(len(Frames)) * frame_step_seconds

        true_boundaries = read_lab_boundaries(full_lab_path)
        best_T = find_threshold_binary_search(
            feature_array=STE, 
            true_boundaries=true_boundaries, 
            frame_step_ms=FRAME_STEP, 
            min_silence_ms=MIN_SILENCE
        )
        
        raw_speech = STE > best_T
        detected_speech = smoothingLogic(raw_speech)
        pred_boundaries = get_boundaries_from_bool(detected_speech, frame_step_seconds)
        
        evaluation_results[filename] = {'true': true_boundaries, 'pred': pred_boundaries}

        TimeAxes.append(TimeAxis)
        NormalizedSignals.append(SignalNormalized)
        FrameTimeAxes.append(FrameTimeAxis)
        STEs.append(STE)
        MAs.append(MA)
        ZCRs.append(ZCR)
        TrueBoundariesList.append(true_boundaries)
        PredBoundariesList.append(pred_boundaries)
        Titles.append(f"{filename} (T={best_T:.4f})")

    print("\n" + "="*60)
    evaluate_and_print_terminal(evaluation_results)

    plotFinalFigure(
        TimeAxes, NormalizedSignals, FrameTimeAxes, 
        STEs, MAs, ZCRs, TrueBoundariesList, PredBoundariesList, Titles
    )

if __name__ == "__main__":
    run_pipeline()