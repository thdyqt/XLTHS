import numpy 
import wave
import matplotlib.pyplot as pyplot

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

def plotFinalFigure(TimeAxes, NormalizedSignals, FrameTimeAxes, STEs, MAs, ZCRs, DetectedSpeeches, Titles):
    """Plot 4 audio files in a 4-corner grid with overlapped features and separation grids."""
    # 4 rows, 2 columns. Rows 0&1 = Top files. Rows 2&3 = Bottom files.
    fig, axes = pyplot.subplots(4, 2, figsize=(18, 12))
    fig.suptitle("Speech Discrimination - Overlapped Features", fontsize=18, fontweight='bold')

    # Grid mapping for the 4 corners: (row, col)
    grid_positions = [(0, 0), (0, 1), (2, 0), (2, 1)]

    for i in range(len(Titles)):
        row, col = grid_positions[i]
        ax_main = axes[row, col]       # Top cell: Overlapped Waveform + Features
        ax_bool = axes[row+1, col]     # Bottom cell: Boolean boundaries

        # Normalize features so they fit cleanly over the [-1, 1] audio waveform
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
        
        # Add internal background grid lines
        ax_main.grid(True, which='both', linestyle='--', linewidth=0.7)
        
        if i == 0: 
            ax_main.legend(loc="upper right", fontsize=9)

        # --- BOTTOM GRAPH: Detected Speech ---
        ax_bool.plot(FrameTimeAxes[i], DetectedSpeeches[i], color='red', drawstyle='steps-pre')
        ax_bool.set_ylabel("1=Speech")
        ax_bool.set_xlabel("Time (s)")
        ax_bool.set_ylim(-0.2, 1.2)
        
        # Add internal background grid lines
        ax_bool.grid(True, which='both', linestyle='--', linewidth=0.7)
        
        # Link X-axes so zooming the waveform zooms the threshold graph
        ax_main.sharex(ax_bool)

    # Add explicit spacing between the 4 quadrants (h_pad for vertical, w_pad for horizontal)
    pyplot.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0, w_pad=4.0)
    pyplot.show()
        