import numpy 
import wave
import matplotlib.pyplot as pyplot

def read_audio(path):
    """This function is used to get the data of an audio file for further processes (normalized signal, sample rate, time axis)"""
    with wave.open(path, 'rb') as WavFile:
        sample_rate = WavFile.getframerate()            #how many samples are there in an amount of time aka frequency
        sample_amount = WavFile.getnframes()            #how many samples are there 
        raw_data = WavFile.readframes(sample_amount)    #raw strings of all samples in 16 bit format

        #read the raw string in 16 bit, as conventional .wav files are saved in 16 bit format
        #this is to ensure that numpy read the raw data correctly
        signal = numpy.frombuffer(raw_data, dtype=numpy.int16)

        #normalize the amp to range of -1 to 1
        max_amp = numpy.max(numpy.max(signal))
        if max_amp > 0:
            signal_normalized = signal / max_amp
        else:
            signal_normalized = signal                  #in case signal file is empty so that it doesnt crash

        duration = sample_amount / sample_rate
        time_axis = numpy.linspace(0, duration, num=sample_amount)

        return signal_normalized, sample_rate, time_axis
        

