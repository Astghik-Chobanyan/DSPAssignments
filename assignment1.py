"""This file contains tasks and solutions of assignment 1"""
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from data import __file__ as DATA_DIR

from simple_wav import (read_audio, compute_power, plot_waveform,
                        write_audio, change_sampling_rate)

data_dir = Path(DATA_DIR).parent


def task1(data: np.array):
    """
        Compute and print the power and mean square amplitude of the given audio data.

        Parameters:
        data (np.array): The audio data array.

        Returns:
        float: The computed power of the audio data.
    """
    power = compute_power(data=data)
    print(f"Power = {power}")
    print(f"Mean square amplitude = {np.sqrt(power)}")
    return power


def task2(data: np.array, start: int, end: int, sampling_rate: int):
    """
    Plot the waveform of a segment of the audio data.

    Parameters:
    data (np.array): The audio data array.
    start (int): The starting index of the segment.
    end (int): The ending index of the segment.
    sampling_rate (int): The sampling rate of the audio data.
    """
    segment = data[start:end]
    plot_waveform(segment, sampling_rate)


def task3(data: np.array, sampling_rate: int):
    """
    Plot two specific segments of the audio data.

    Parameters:
    data (np.array): The audio data array.
    sampling_rate (int): The sampling rate of the audio data.
    """
    task2(data, sampling_rate, sampling_rate+2000, sampling_rate)
    task2(data, 2*sampling_rate, 2*sampling_rate + 540, sampling_rate)


def task4(data: np.array, samling_rate: int):
    """
    Write a new audio file with double the volume of the original data.

    Parameters:
    data (np.array): The original audio data array.
    samling_rate (int): The sampling rate of the audio data.
    """
    new_data = 2 * data
    write_audio(data_dir.joinpath('new_v5.wav'), samling_rate, new_data)


def task5(audio_path: Path, sampling_rate: int):
    """
    Decrease the sampling rate of an audio file by half and save it as a new file.

    Parameters:
    audio_path (str): The path to the original audio file.
    sampling_rate (int): The original sampling rate of the audio file.
    """
    new_sr = sampling_rate // 2
    change_sampling_rate(audio_path, data_dir.joinpath('v_half_sr.wav'), new_sr)


def task6(frequency, duration, amplitude=1.0, sampling_rate=44100):
    """
    Generate a sinusoidal signal of a given frequency and duration, and save it as a WAV
    file.

    Parameters:
    frequency (float): The frequency of the sinusoidal signal.
    duration (float): The duration of the signal in seconds.
    amplitude (float, optional): The amplitude of the signal. Default is 1.0.
    sampling_rate (int, optional): The sampling rate of the signal. Default is 44100 Hz.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), False)
    tone = amplitude * np.sin(2 * np.pi * frequency * t)
    tone = np.int16(tone * 32767)
    wavfile.write(data_dir.joinpath('sine_wave.wav'), sampling_rate, tone)


if __name__ == '__main__':
    audio_path = data_dir.joinpath("v5.wav")
    sr, data = read_audio(path=audio_path)

    print("Task 1: Computing power of v5.wav file")
    task1(data=data)

    print("Task 2 and 3: Plotting different parts of data")
    task3(data=data, sampling_rate=sr)

    print("Task 4: writing new wav with 2 times greater volume")
    task4(data=data, samling_rate=sr)

    print("Task 5: decreasing the audio sampling rate 2 times")
    task5(audio_path=audio_path, sampling_rate=sr)

    print("Task 6: Write a function which generates a sinusoidal signal with the "
          "given frequency and saves it as .wav file")
    task6(440, 3)

    print("All tasks completed!")
