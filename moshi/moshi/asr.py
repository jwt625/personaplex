# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""
ASR (Automatic Speech Recognition) module for user speech transcription.

This module provides real-time transcription of user audio using faster-whisper.
It runs asynchronously to avoid blocking the main audio processing loop.
"""

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ASRConfig:
    """Configuration for ASR model."""
    model_name: str = "large-v3-turbo"
    device: str = "cuda"
    device_index: int = 1  # Use GPU 1 by default (GPU 0 for PersonaPlex)
    compute_type: str = "float16"
    language: str = "en"
    beam_size: int = 5
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500  # Minimum silence to split speech
    # Buffer settings - longer buffer = better accuracy but more latency
    buffer_duration_sec: float = 4.0  # Accumulate this much audio before transcribing
    sample_rate: int = 24000  # PersonaPlex uses 24kHz, will resample to 16kHz for Whisper
    # Prompt to help with context
    initial_prompt: str = ""  # Optional prompt to guide transcription


class ASRTranscriber:
    """
    Asynchronous ASR transcriber using faster-whisper.
    
    Buffers user audio and transcribes in chunks to balance latency and accuracy.
    """
    
    def __init__(self, config: ASRConfig, on_transcript: Optional[Callable[[str], None]] = None):
        """
        Initialize the ASR transcriber.
        
        Args:
            config: ASR configuration
            on_transcript: Callback function called with transcribed text
        """
        self.config = config
        self.on_transcript = on_transcript
        self.model = None
        self._audio_buffer = []
        self._buffer_samples = 0
        self._target_samples = int(config.buffer_duration_sec * config.sample_rate)
        self._whisper_sample_rate = 16000
        self._running = False
        self._transcribe_queue: queue.Queue = queue.Queue()
        self._transcribe_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
    def load_model(self):
        """Load the Whisper model."""
        from faster_whisper import WhisperModel

        logger.info(f"Loading ASR model {self.config.model_name} on GPU {self.config.device_index}...")
        start = time.time()

        # Save current CUDA device to restore after loading
        # (faster-whisper may change the default CUDA device)
        current_device = None
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()

        self.model = WhisperModel(
            self.config.model_name,
            device=self.config.device,
            device_index=self.config.device_index,
            compute_type=self.config.compute_type
        )

        # Restore the original CUDA device
        if current_device is not None:
            torch.cuda.set_device(current_device)

        load_time = time.time() - start
        logger.info(f"ASR model loaded in {load_time:.2f} seconds")
        
    def start(self, loop: asyncio.AbstractEventLoop):
        """Start the transcription worker thread."""
        self._loop = loop
        self._running = True
        self._transcribe_thread = threading.Thread(target=self._transcribe_worker, daemon=True)
        self._transcribe_thread.start()
        logger.info("ASR transcription worker started")
        
    def stop(self):
        """Stop the transcription worker."""
        self._running = False
        if self._transcribe_thread:
            self._transcribe_queue.put(None)  # Signal to stop
            self._transcribe_thread.join(timeout=2.0)
        logger.info("ASR transcription worker stopped")
        
    def add_audio(self, pcm_data: np.ndarray):
        """
        Add audio data to the buffer.
        
        Args:
            pcm_data: PCM audio data at self.config.sample_rate Hz
        """
        self._audio_buffer.append(pcm_data)
        self._buffer_samples += len(pcm_data)
        
        # Check if we have enough audio to transcribe
        if self._buffer_samples >= self._target_samples:
            self._flush_buffer()
            
    def _flush_buffer(self):
        """Flush the audio buffer and queue for transcription."""
        if not self._audio_buffer:
            return
            
        # Concatenate all buffered audio
        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer = []
        self._buffer_samples = 0
        
        # Queue for transcription
        self._transcribe_queue.put(audio)
        
    def reset(self):
        """Reset the audio buffer."""
        self._audio_buffer = []
        self._buffer_samples = 0
        # Clear the queue
        while not self._transcribe_queue.empty():
            try:
                self._transcribe_queue.get_nowait()
            except queue.Empty:
                break
                
    def _resample(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from source rate to Whisper's 16kHz."""
        if self.config.sample_rate == self._whisper_sample_rate:
            return audio
            
        # Simple linear interpolation resampling
        ratio = self._whisper_sample_rate / self.config.sample_rate
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        
    def _transcribe_worker(self):
        """Worker thread that processes audio from the queue."""
        while self._running:
            try:
                audio = self._transcribe_queue.get(timeout=0.1)
                if audio is None:
                    break
                    
                # Resample to 16kHz for Whisper
                audio_16k = self._resample(audio)
                
                # Transcribe with VAD parameters
                vad_params = {
                    "min_silence_duration_ms": self.config.vad_min_silence_duration_ms,
                } if self.config.vad_filter else None

                segments, info = self.model.transcribe(
                    audio_16k,
                    beam_size=self.config.beam_size,
                    language=self.config.language,
                    vad_filter=self.config.vad_filter,
                    vad_parameters=vad_params,
                    initial_prompt=self.config.initial_prompt if self.config.initial_prompt else None,
                )
                
                # Collect text from segments
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text.strip())
                    
                if text_parts:
                    full_text = " ".join(text_parts)
                    if self.on_transcript and self._loop:
                        # Schedule callback in the event loop
                        self._loop.call_soon_threadsafe(
                            lambda t=full_text: self.on_transcript(t)
                        )
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"ASR transcription error: {e}")

