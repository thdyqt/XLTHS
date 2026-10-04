import sys
import os
import numpy

# Add parent directory to path to import SharedFunctions_2
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import SharedFunctions

def run_pipeline_test():
    audio_files = [
        "TinHieuHuanLuyen/phone_F1.wav",
        "TinHieuHuanLuyen/phone_M1.wav",
        "TinHieuKiemThu/phone_F2.wav",
        "TinHieuKiemThu/phone_M2.wav"
    ]

    # Empty lists to hold the data for all 4 files
    TimeAxes, NormalizedSignals, FrameTimeAxes = [], [], []
    STEs, MAs, ZCRs, DetectedSpeeches, Titles = [], [], [], [], []

    for path in audio_files:
        print(f"Processing: {path}...")
        
        SignalNormalized, SampleRate, TimeAxis = SharedFunctions.readAudio(path)
        Frames = SharedFunctions.frameWindow(SignalNormalized, SampleRate)
        
        STE = SharedFunctions.calculate_STE(Frames)
        MA = SharedFunctions.calculate_MA(Frames)
        ZCR = SharedFunctions.calculate_ZCR(Frames)

        frame_step_seconds = SharedFunctions.FRAME_STEP / 1000.0
        FrameTimeAxis = numpy.arange(len(Frames)) * frame_step_seconds
        DummySpeech = numpy.zeros(len(Frames), dtype=bool)

        # Save all results to the lists
        TimeAxes.append(TimeAxis)
        NormalizedSignals.append(SignalNormalized)
        FrameTimeAxes.append(FrameTimeAxis)
        STEs.append(STE)
        MAs.append(MA)
        ZCRs.append(ZCR)
        DetectedSpeeches.append(DummySpeech)
        Titles.append(path.split('/')[-1])

    # Send the bundled data to the shared plotting function
    SharedFunctions.plotFinalFigure(
        TimeAxes, NormalizedSignals, FrameTimeAxes, 
        STEs, MAs, ZCRs, DetectedSpeeches, Titles
    )

if __name__ == "__main__":
    run_pipeline_test()