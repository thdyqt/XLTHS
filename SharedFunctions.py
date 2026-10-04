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

def plotFinalFigure(TimeAxis, NormalizedSignal, FrameTimeAxis, STE, MA, ZCR, DetectedSpeech, Title="Result"):
    """Plot the resulting graphs"""
    fig, axes = pyplot.subplots(5, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(Title, fontsize=16)

    # Subplot 1: Original Waveform with Speech Boundaries highlighted
    axes[0].plot(TimeAxis, NormalizedSignal, color='darkgray', linewidth=0.5)
    axes[0].set_title("Normalized Audio Waveform")
    axes[0].set_ylabel("Amplitude")
    
    for i in range(len(DetectedSpeech)):
        if DetectedSpeech[i]:
            start_time = FrameTimeAxis[i]
            axes[0].axvspan(start_time, start_time + 0.01, color='green', alpha=0.3, lw=0)

    # Subplot 2: Short-Time Energy (STE)
    axes[1].plot(FrameTimeAxis, STE, color='blue', linewidth=1)
    axes[1].set_title("Short-Time Energy (STE)")
    axes[1].set_ylabel("Energy")

    # Subplot 3: Magnitude Average (MA)
    axes[2].plot(FrameTimeAxis, MA, color='orange', linewidth=1)
    axes[2].set_title("Magnitude Average (MA)")
    axes[2].set_ylabel("Magnitude")
    
    # Subplot 4: Zero-Crossing Rate (ZCR)
    axes[3].plot(FrameTimeAxis, ZCR, color='purple', linewidth=1)
    axes[3].set_title("Zero-Crossing Rate (ZCR)")
    axes[3].set_ylabel("Crossings")

    # Subplot 5: Boolean Speech Classification (1 for Speech, 0 for Silence)
    axes[4].plot(FrameTimeAxis, DetectedSpeech, color='red', drawstyle='steps-pre')
    axes[4].set_title("Detected Speech Boundaries (Boolean)")
    axes[4].set_ylabel("1 = Speech")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_ylim(-0.2, 1.2)

    pyplot.tight_layout()
    pyplot.show()
        