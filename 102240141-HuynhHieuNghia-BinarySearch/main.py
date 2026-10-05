import sys
import os
import numpy as np

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import SharedFunctions as sf

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

    # Data accumulators
    TimeAxes, NormalizedSignals, FrameTimeAxes = [], [], []
    STEs, MAs, ZCRs, TrueBoundariesList, PredBoundariesList, Titles = [], [], [], [], [], []
    
    evaluation_results = {}

    for wav_path, lab_path in zip(audio_files, lab_files):
        full_wav_path = os.path.join(parent_dir, wav_path)
        full_lab_path = os.path.join(parent_dir, lab_path)
        filename = full_wav_path.split(os.sep)[-1]
        print(f"Processing: {filename}...")

        # Feature Extraction
        SignalNormalized, SampleRate, TimeAxis = sf.readAudio(full_wav_path)
        Frames = sf.frameWindow(SignalNormalized, SampleRate)
        STE = sf.calculate_STE(Frames)
        MA = sf.calculate_MA(Frames)
        ZCR = sf.calculate_ZCR(Frames)
        frame_step_seconds = sf.FRAME_STEP / 1000.0
        FrameTimeAxis = np.arange(len(Frames)) * frame_step_seconds

        # Truth & Thresholding
        true_boundaries = sf.read_lab_boundaries(full_lab_path)
        best_T = sf.find_threshold_binary_search(
            feature_array=STE, 
            true_boundaries=true_boundaries, 
            frame_step_ms=sf.FRAME_STEP, 
            min_silence_ms=sf.MIN_SILENCE
        )
        
        # Prediction & Boundaries
        raw_speech = STE > best_T
        detected_speech = sf.smoothingLogic(raw_speech)
        pred_boundaries = sf.get_boundaries_from_bool(detected_speech, frame_step_seconds)
        
        evaluation_results[filename] = {'true': true_boundaries, 'pred': pred_boundaries}

        # Append to visualization lists
        TimeAxes.append(TimeAxis)
        NormalizedSignals.append(SignalNormalized)
        FrameTimeAxes.append(FrameTimeAxis)
        STEs.append(STE)
        MAs.append(MA)
        ZCRs.append(ZCR)
        TrueBoundariesList.append(true_boundaries)
        PredBoundariesList.append(pred_boundaries)
        Titles.append(f"{filename} (T={best_T:.4f})")

    # Final Execution
    print("\n" + "="*60)
    sf.evaluate_and_print_terminal(evaluation_results)

    sf.plotFinalFigure(
        TimeAxes, NormalizedSignals, FrameTimeAxes, 
        STEs, MAs, ZCRs, TrueBoundariesList, PredBoundariesList, Titles
    )

if __name__ == "__main__":
    run_pipeline()