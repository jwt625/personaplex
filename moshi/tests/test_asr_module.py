#!/usr/bin/env python3
"""
Test script for the ASR module.

Tests the ASRTranscriber class with simulated audio input.

Run from the personaplex directory:
    source .venv/bin/activate
    python moshi/tests/test_asr_module.py
"""

import asyncio
import sys
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, "moshi")

from moshi.asr import ASRConfig, ASRTranscriber


def test_asr_module():
    """Test the ASR module with simulated audio."""
    print("=" * 60)
    print("ASR Module Test")
    print("=" * 60)
    
    # Track received transcripts
    transcripts = []
    
    def on_transcript(text: str):
        print(f"  Received transcript: '{text}'")
        transcripts.append(text)
    
    # Create config
    config = ASRConfig(
        model_name="large-v3-turbo",
        device="cuda",
        device_index=1,
        compute_type="float16",
        language="en",
        buffer_duration_sec=2.0,
        sample_rate=24000,  # PersonaPlex sample rate
    )
    
    print(f"\nConfig:")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}:{config.device_index}")
    print(f"  Buffer duration: {config.buffer_duration_sec}s")
    print(f"  Sample rate: {config.sample_rate} Hz")
    
    # Create transcriber
    transcriber = ASRTranscriber(config, on_transcript=on_transcript)
    
    # Load model
    print("\nLoading model...")
    transcriber.load_model()
    print("Model loaded.")
    
    # Start worker
    loop = asyncio.new_event_loop()
    transcriber.start(loop)
    print("Worker started.")
    
    # Simulate audio input
    print("\n" + "-" * 40)
    print("Test 1: Silence (should produce no transcript)")
    print("-" * 40)
    
    # Generate 3 seconds of silence in chunks
    chunk_duration = 0.08  # 80ms chunks (similar to PersonaPlex frame size)
    chunk_samples = int(config.sample_rate * chunk_duration)
    num_chunks = int(3.0 / chunk_duration)
    
    for i in range(num_chunks):
        chunk = np.zeros(chunk_samples, dtype=np.float32)
        transcriber.add_audio(chunk)
        time.sleep(0.01)  # Small delay to simulate real-time
    
    # Wait for processing
    time.sleep(1.0)
    
    print(f"  Transcripts received: {len(transcripts)}")
    
    # Test 2: Simulated speech-like audio (noise)
    print("\n" + "-" * 40)
    print("Test 2: Random noise (may or may not produce transcript)")
    print("-" * 40)
    
    transcripts.clear()
    transcriber.reset()
    
    for i in range(num_chunks):
        # Generate random noise (not real speech, but tests the pipeline)
        chunk = np.random.randn(chunk_samples).astype(np.float32) * 0.1
        transcriber.add_audio(chunk)
        time.sleep(0.01)
    
    # Wait for processing
    time.sleep(1.0)
    
    print(f"  Transcripts received: {len(transcripts)}")
    
    # Test 3: Verify resampling
    print("\n" + "-" * 40)
    print("Test 3: Verify resampling (24kHz -> 16kHz)")
    print("-" * 40)
    
    test_audio = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz
    resampled = transcriber._resample(test_audio)
    expected_length = 16000  # 1 second at 16kHz
    
    print(f"  Input length: {len(test_audio)} samples (24kHz)")
    print(f"  Output length: {len(resampled)} samples (16kHz)")
    print(f"  Expected length: {expected_length} samples")
    print(f"  Resampling correct: {len(resampled) == expected_length}")
    
    # Cleanup
    print("\n" + "-" * 40)
    print("Cleanup")
    print("-" * 40)
    
    transcriber.stop()
    loop.close()
    print("Worker stopped.")
    
    print("\n" + "=" * 60)
    print("ASR Module Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_asr_module()

