import numpy as np
import pyaudio
import mediapipe as mp

# Import necessary classes
from mediapipe.tasks import python
from mediapipe.tasks.python import audio

# Define necessary classes
AudioClassifier = mp.tasks.audio.AudioClassifier
AudioClassifierOptions = mp.tasks.audio.AudioClassifierOptions
AudioRunningMode = mp.tasks.audio.RunningMode
BaseOptions = mp.tasks.BaseOptions

# Define a function to print classification results
def print_result(result: audio.AudioClassifierResult, timestamp_ms: int):
    print("AudioClassifierResult result:", result)

# Set up options for the audio classifier
options = AudioClassifierOptions(
    base_options=BaseOptions(model_asset_path='C:/Projects_2024/audio_classification/audio_classification_rnd/classifier.tflite'),
    running_mode=AudioRunningMode.AUDIO_STREAM,
    max_results=5,
    result_callback=print_result
)

# Callback function to handle microphone audio data
def callback(in_data, frame_count, time_info, status):
    # Convert the audio data to numpy array
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    
    # Perform audio classification
    classifier.classify_async(audio_data, int(time_info['input_buffer_adc_time'] * 1000))
    
    return (in_data, pyaudio.paContinue)

# Set up audio stream
audio_format = pyaudio.paFloat32
channels = 1
sample_rate = 44100
chunk_size = 1024

p = pyaudio.PyAudio()
stream = p.open(format=audio_format,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
                stream_callback=callback)

# Create the audio classifier task
with AudioClassifier.create_from_options(options) as classifier:
    # Start the audio stream
    stream.start_stream()

    try:
        # Keep the script running indefinitely
        while True:
            pass
    except KeyboardInterrupt:
        # Stop the audio stream
        stream.stop_stream()
        stream.close()
        p.terminate()
