import numpy 
import wave
import matplotlib.pyplot as pyplot
import math

FRAME_SIZE = 25
FRAME_STEP = 10
MIN_SILENCE = 200

def readAudio(path):
    """Get the data of an .wav file for further processes (normalized Signal, sample rate, time axis)"""
    with wave.open(path, 'rb') as WavFile:
        SampleRate = WavFile.getframerate()            #how many samples are there in an amount of time aka frequency
        SampleAmount = WavFile.getnframes()            #how many samples are there 
        RawData = WavFile.readframes(SampleAmount)    #raw strings of all samples in 16 bit format

        #read the raw string in 16 bit, as conventional .wav files are saved in 16 bit format
        #this is to ensure that numpy read the raw data correctly
        Signal = numpy.frombuffer(RawData, dtype=numpy.int16)

        #normalize the amp to range of -1 to 1
        MaxAmp = numpy.max(numpy.abs(Signal))
        if MaxAmp > 0:
            SignalNormalized = Signal / MaxAmp
        else:
            SignalNormalized = Signal                  #in case Signal file is empty so that it doesnt crash

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
    FrameLength = int(SampleRate * (FRAME_SIZE / 1000))     #how many data points there are in a frame
    FrameStep = int(SampleRate * (FRAME_STEP / 1000))       #how many data points there are in a step 

    SignalLength = len(SignalNormalized)

    #fallback (that shouldnt even happen)
    #if your audio signal lasts less than 25ms you deserve to get culled
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
    MinSilenceFrames = MIN_SILENCE // FRAME_STEP    #the number of frames it takes to be 200ms

    SmoothedSpeech = numpy.copy(SpeechFrames)
    InSilence = False
    SilenceStart = 0

    for i in range (len(SmoothedSpeech)):
        #record silence region
        if not SmoothedSpeech[i] and not InSilence:
            InSilence = True
            SilenceStart = i
        elif SmoothedSpeech[i] and InSilence:
            InSilence = False
            SilenceLength = i - SilenceStart
            #switch all to speech frame
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
        ax_main = axes[row, col]       # Top cell: Signal + STE/MA/ZCR
        ax_bound = axes[row+1, col]    # Bottom cell: Waveform + Vertical Boundaries

        # Normalize features
        STE_norm = STEs[i] / numpy.max(STEs[i]) if numpy.max(STEs[i]) > 0 else STEs[i]
        MA_norm = MAs[i] / numpy.max(MAs[i]) if numpy.max(MAs[i]) > 0 else MAs[i]
        ZCR_norm = ZCRs[i] / numpy.max(ZCRs[i]) if numpy.max(ZCRs[i]) > 0 else ZCRs[i]

        # --- TOP GRAPH: Overlapped Features ---
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

        # --- BOTTOM GRAPH: Boundary Comparison ---
        ax_bound.plot(TimeAxes[i], NormalizedSignals[i], color='lightgray', linewidth=0.5)
        
        # Plot Ground Truth lines (Red)
        for start, end in TrueBoundariesList[i]:
            ax_bound.axvline(start, color='red', linewidth=1.5, label='Chuẩn (Đỏ)')
            ax_bound.axvline(end, color='red', linewidth=1.5)
            
        # Plot Predicted lines (Blue)
        for start, end in PredBoundariesList[i]:
            ax_bound.axvline(start, color='blue', linewidth=1.5, label='Dự đoán (Xanh)')
            ax_bound.axvline(end, color='blue', linewidth=1.5)

        ax_bound.set_ylabel("Amplitude")
        ax_bound.set_xlabel("Time (s)")
        ax_bound.set_ylim(-1.1, 1.1)
        ax_bound.grid(True, which='both', linestyle='--', linewidth=0.7)
        
        # Fix duplicate legend entries for the boundaries
        handles, labels = ax_bound.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        if i == 0:  
            ax_bound.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', fontsize=9)
        
        # Link X-axes
        ax_main.sharex(ax_bound)

    pyplot.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0, w_pad=4.0)
    pyplot.show()

def read_lab_boundaries(lab_path):
    """
    Reads a .lab file, extracts 'v' and 'uv' segments, ignores 'sil', 
    and merges contiguous speech blocks into singular [start, end] pairs.
    """
    raw_segments = []
    
    with open(lab_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            # Ensure the line has at least start, end, and label
            if len(parts) >= 3:
                try:
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    label = parts[2].lower()
                    
                    # Only collect voiced (v) and unvoiced (uv) speech
                    if label in ['v', 'uv']:
                        raw_segments.append([start_time, end_time])
                        
                except ValueError:
                    # Safely skip non-numerical lines like 'F0mean 145' and 'F0std 33.7'
                    continue

    # If no speech was found, return empty
    if not raw_segments:
        return []

    # Merge contiguous segments (e.g., [1.02, 1.88] + [1.88, 1.95] -> [1.02, 1.95])
    merged_boundaries = []
    current_start = raw_segments[0][0]
    current_end = raw_segments[0][1]

    for i in range(1, len(raw_segments)):
        next_start = raw_segments[i][0]
        next_end = raw_segments[i][1]

        # If the next segment starts exactly where the current one ends (allow 0.001s float tolerance)
        if abs(next_start - current_end) < 0.001:
            current_end = next_end
        else:
            # Gap detected, save the merged block and start a new one
            merged_boundaries.append([current_start, current_end])
            current_start = next_start
            current_end = next_end

    # Append the final block
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
            
    # Catch if it ends while still speaking
    if in_speech:
        end_time = len(speech_frames) * frame_step_s
        boundaries.append([round(start_time, 3), round(end_time, 3)])
        
    return boundaries

def find_threshold_binary_search(feature_array, true_boundaries, frame_step_ms=10, min_silence_ms=200):
    """Finds the optimal threshold T by matching predicted speech duration to ground truth."""
    # Calculate target number of speech frames based on the .lab file
    total_true_duration_s = sum([end - start for start, end in true_boundaries])
    target_frames = int(total_true_duration_s / (frame_step_ms / 1000.0))
    
    low = numpy.min(feature_array)
    high = numpy.max(feature_array)
    best_T = low
    
    for _ in range(50): # 50 iterations is more than enough for high precision
        mid = (low + high) / 2.0
        
        # Predict using current threshold
        raw_speech = feature_array > mid
        
        # Apply smoothing logic to get the true final count
        # (Assuming you renamed your smoothing function to apply_gap_bridging)
        smoothed_speech = smoothingLogic(raw_speech) 
        
        current_frames = numpy.sum(smoothed_speech)
        
        if current_frames > target_frames:
            # Predicted too much speech -> Threshold is too low
            low = mid
        elif current_frames < target_frames:
            # Predicted too little speech -> Threshold is too high
            high = mid
        else:
            best_T = mid
            break
            
        best_T = mid
        
    return best_T

def evaluate_and_print_terminal(results_dict):
    """
    Prints terminal output matching the required format.
    results_dict format: { 'filename.wav': {'true': [[s, e]], 'pred': [[s, e]]} }
    """
    print(f"{'[KIEM THU]':<15}")
    print(f"{'File':<20} {'#bien chuan':>11} {'#bien TT':>9} {'MAE(ms)':>9} {'RMSE(ms)':>9}")
    
    total_mae, total_rmse = 0, 0
    count = 0
    
    for filename, data in results_dict.items():
        true_b = data['true']
        pred_b = data['pred']
        
        num_true = len(true_b) * 2  # 2 points (start, end) per boundary pair
        num_pred = len(pred_b) * 2
        
        # Print file breakdown[cite: 8]
        print(f"{filename}")
        print(f"  chuan : {true_b}")
        print(f"  thuat toan: {pred_b}")
        
        # Calculate Errors (assuming exactly 1 boundary pair per file for the math)
        if len(true_b) > 0 and len(pred_b) > 0:
            start_err_ms = abs(pred_b[0][0] - true_b[0][0]) * 1000
            end_err_ms = abs(pred_b[0][1] - true_b[0][1]) * 1000
            
            mae = (start_err_ms + end_err_ms) / 2
            rmse = math.sqrt((start_err_ms**2 + end_err_ms**2) / 2)
            
            total_mae += mae
            total_rmse += rmse
            count += 1
            
            # Print table row[cite: 8]
            print(f"{filename:<20} {num_true:>11} {num_pred:>9} {mae:>9.2f} {rmse:>9.2f}")
    
    if count > 0:
        print(f"{'TRUNG BINH':<42} {total_mae/count:>9.2f} {total_rmse/count:>9.2f}")

def plot_boundary_comparison(TimeAxis, SignalNormalized, true_boundaries, pred_boundaries, Title="So sánh biên thời gian"):
    fig, ax = pyplot.subplots(1, 1, figsize=(12, 4))
    
    # Plot the raw waveform in light gray[cite: 7]
    ax.plot(TimeAxis, SignalNormalized, color='lightgray', linewidth=0.5)
    
    # Plot Ground Truth lines (Red)[cite: 7]
    for start, end in true_boundaries:
        ax.axvline(start, color='red', linewidth=1.5, label='Chuẩn (Đỏ)')
        ax.axvline(end, color='red', linewidth=1.5)
        
    # Plot Predicted lines (Blue)[cite: 7]
    for start, end in pred_boundaries:
        ax.axvline(start, color='blue', linewidth=1.5, label='Dự đoán (Xanh)')
        ax.axvline(end, color='blue', linewidth=1.5)

    ax.set_title(Title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.1, 1.1)
    
    # Fix duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    
    pyplot.tight_layout()
    pyplot.show()