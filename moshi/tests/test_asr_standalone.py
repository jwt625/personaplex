#!/usr/bin/env python3
"""
Standalone test script for faster-whisper ASR integration.

This script tests:
1. Model loading on GPU
2. Transcription of a test audio file
3. Real-time transcription simulation with audio chunks

Run from the personaplex directory:
    source .venv/bin/activate
    python moshi/tests/test_asr_standalone.py
"""

import time
import numpy as np
import torch

def test_model_loading():
    """Test that the model loads correctly on GPU."""
    print("=" * 60)
    print("Test 1: Model Loading")
    print("=" * 60)
    
    from faster_whisper import WhisperModel
    
    # Check available GPUs
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load model on GPU 1 (GPU 0 is for PersonaPlex)
    # Using large-v3-turbo for best accuracy with good speed
    device = "cuda"
    device_index = 1 if torch.cuda.device_count() > 1 else 0
    
    print(f"\nLoading Whisper large-v3-turbo on GPU {device_index}...")
    start = time.time()
    
    model = WhisperModel(
        "large-v3-turbo",
        device=device,
        device_index=device_index,
        compute_type="float16"  # Use FP16 for speed on H100
    )
    
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f} seconds")
    
    return model


def test_transcription_from_audio_array(model):
    """Test transcription from a numpy audio array (simulating real-time audio)."""
    print("\n" + "=" * 60)
    print("Test 2: Transcription from Audio Array")
    print("=" * 60)
    
    # Generate a simple test audio (1 second of silence + sine wave)
    # In real use, this would be PCM audio from the microphone
    sample_rate = 16000  # Whisper expects 16kHz
    duration = 3.0  # seconds
    
    # Create a simple sine wave (440 Hz = A4 note)
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    print(f"Audio shape: {audio.shape}")
    print(f"Audio duration: {duration} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    
    # Transcribe
    print("\nTranscribing...")
    start = time.time()
    
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        language="en",
        vad_filter=True,  # Filter out non-speech
    )
    
    transcribe_time = time.time() - start
    
    print(f"Transcription completed in {transcribe_time:.2f} seconds")
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    text_parts = []
    for segment in segments:
        print(f"  [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        text_parts.append(segment.text)
    
    if not text_parts:
        print("  (No speech detected - expected for sine wave test)")
    
    return transcribe_time


def test_streaming_simulation(model):
    """Simulate streaming transcription with audio chunks."""
    print("\n" + "=" * 60)
    print("Test 3: Streaming Simulation")
    print("=" * 60)
    
    # Simulate 2-second audio chunks (typical for real-time ASR)
    sample_rate = 16000
    chunk_duration = 2.0  # seconds per chunk
    num_chunks = 3
    
    print(f"Simulating {num_chunks} chunks of {chunk_duration}s each")
    
    total_time = 0
    for i in range(num_chunks):
        # Generate chunk (silence for testing)
        chunk = np.zeros(int(sample_rate * chunk_duration), dtype=np.float32)
        
        start = time.time()
        segments, _ = model.transcribe(
            chunk,
            beam_size=5,
            language="en",
            vad_filter=True,
        )
        # Consume the generator
        list(segments)
        chunk_time = time.time() - start
        total_time += chunk_time
        
        rtf = chunk_time / chunk_duration  # Real-time factor
        print(f"  Chunk {i+1}: {chunk_time:.3f}s (RTF: {rtf:.3f})")
    
    avg_rtf = total_time / (num_chunks * chunk_duration)
    print(f"\nAverage RTF: {avg_rtf:.3f}")
    print(f"RTF < 1.0 means faster than real-time: {'YES' if avg_rtf < 1.0 else 'NO'}")
    
    return avg_rtf


def main():
    print("Faster-Whisper ASR Standalone Test")
    print("=" * 60)
    
    # Test 1: Load model
    model = test_model_loading()
    
    # Test 2: Transcribe audio array
    test_transcription_from_audio_array(model)
    
    # Test 3: Streaming simulation
    avg_rtf = test_streaming_simulation(model)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Model: Whisper large-v3-turbo")
    print(f"Average RTF: {avg_rtf:.3f}")
    print(f"Suitable for real-time: {'YES' if avg_rtf < 0.5 else 'MARGINAL' if avg_rtf < 1.0 else 'NO'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

