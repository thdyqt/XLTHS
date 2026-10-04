import numpy 
import wave
import matplotlib.pyplot as pyplot

def readAudio(path):
    """This function is used to get the data of an .wav file for further processes (normalized Signal, sample rate, time axis)"""
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
    """This function is used to make graph of an .wav file using the data from function \"readAudio\" """
    SignalNormalized, SampleRate, TimeAxis = readAudio(path)

    pyplot.figure(figsize=(10,3))
    pyplot.plot(TimeAxis, SignalNormalized, color='darkgray', linewidth=0.5)

    Title = path.split('/')[-1]
    pyplot.title(f'Signal of: {Title}')
    pyplot.xlabel("Time (s)")
    pyplot.ylabel("Amplitude")

    pyplot.tight_layout()
    pyplot.show()




