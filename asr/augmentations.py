import torch
from torch.distributions.beta import Beta
import numpy as np


def insert_silence(y, sr, num_pauses, length):
    """
    Insert silent segments into an audio signal.
    
    Args:
        y (np.ndarray): The audio signal.
        sr (int): The sampling rate of the audio.
        num_pauses (int): The number of silent segments to insert.
        length (float): The length of each silent segment in seconds.
    
    Returns:
        np.ndarray: The audio signal with silent segments inserted.
    """
    # Calculate the number of samples for the silence
    #y = np.asarray(y, dtype=np.float64)
    silence_samples = int(length * sr)
    silence = np.zeros(silence_samples, dtype=y.dtype)
    
    '''# Ensure num_pauses does not exceed the possible number of insertion points
    max_insertions = len(y) + num_pauses * silence_samples
    num_pauses = min(num_pauses, max_insertions // (len(y) + silence_samples))'''
    
    # Generate random insertion points
    insertion_points = np.sort(np.random.choice(len(y), size=num_pauses, replace=False))
    
    # Insert silence at the specified points
    y_with_silence = y.copy()
    offset = 0
    for point in insertion_points:
        point += offset  # Adjust for previously inserted silences
        y_with_silence = np.concatenate((y_with_silence[:point], silence, y_with_silence[point:]))
        offset += silence_samples
    
    return y_with_silence
