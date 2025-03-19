"""
Tests for the AudioQualitySpecification and related domain models.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyhearingai.core.domain.audio_quality import AudioCodec, AudioFormat, AudioQualitySpecification


class TestAudioQualitySpecification:
    """Tests for the AudioQualitySpecification class."""

    def test_default_values(self):
        """Test default values for AudioQualitySpecification."""
        spec = AudioQualitySpecification()

        assert spec.sample_rate == 16000
        assert spec.channels == 1
        assert spec.bit_depth == 16
        assert spec.format == AudioFormat.WAV
        assert spec.codec == AudioCodec.PCM_S16LE
        assert spec.max_size_bytes == 25 * 1024 * 1024  # 25MB
        assert spec.quality is None

    def test_factory_methods(self):
        """Test factory methods for creating quality specifications."""
        # Test whisper API spec
        whisper_spec = AudioQualitySpecification.for_whisper_api()
        assert whisper_spec.sample_rate == 16000
        assert whisper_spec.channels == 1
        assert whisper_spec.bit_depth == 16
        assert whisper_spec.format == AudioFormat.WAV
        assert whisper_spec.codec == AudioCodec.PCM_S16LE
        assert whisper_spec.max_size_bytes == 25 * 1024 * 1024  # 25MB

        # Test local processing spec
        local_spec = AudioQualitySpecification.for_local_processing()
        assert local_spec.sample_rate == 16000
        assert local_spec.channels == 1
        assert local_spec.max_size_bytes == 0  # No limit

        # Test high quality spec
        high_quality_spec = AudioQualitySpecification.high_quality()
        assert high_quality_spec.sample_rate == 44100
        assert high_quality_spec.channels == 2
        assert high_quality_spec.bit_depth == 24
        assert high_quality_spec.format == AudioFormat.FLAC
        assert high_quality_spec.codec == AudioCodec.FLAC

    def test_with_methods(self):
        """Test the with_* methods for creating modified specifications."""
        base_spec = AudioQualitySpecification()

        # Test with_size_limit
        sized_spec = base_spec.with_size_limit(10 * 1024 * 1024)  # 10MB
        assert sized_spec.sample_rate == base_spec.sample_rate
        assert sized_spec.channels == base_spec.channels
        assert sized_spec.bit_depth == base_spec.bit_depth
        assert sized_spec.format == base_spec.format
        assert sized_spec.codec == base_spec.codec
        assert sized_spec.max_size_bytes == 10 * 1024 * 1024

        # Test with_sample_rate
        rate_spec = base_spec.with_sample_rate(8000)
        assert rate_spec.sample_rate == 8000
        assert rate_spec.channels == base_spec.channels
        assert rate_spec.bit_depth == base_spec.bit_depth
        assert rate_spec.format == base_spec.format
        assert rate_spec.codec == base_spec.codec
        assert rate_spec.max_size_bytes == base_spec.max_size_bytes

    def test_estimated_bytes_per_second(self):
        """Test the estimated_bytes_per_second calculation."""
        # Test WAV PCM 16-bit mono at 16kHz
        spec = AudioQualitySpecification(
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            format=AudioFormat.WAV,
            codec=AudioCodec.PCM_S16LE,
        )
        # 16000 samples/s * 1 channel * 2 bytes/sample = 32000 bytes/s
        assert spec.estimated_bytes_per_second() == 32000

        # Test WAV PCM 16-bit stereo at 44.1kHz
        spec = AudioQualitySpecification(
            sample_rate=44100,
            channels=2,
            bit_depth=16,
            format=AudioFormat.WAV,
            codec=AudioCodec.PCM_S16LE,
        )
        # 44100 samples/s * 2 channels * 2 bytes/sample = 176400 bytes/s
        assert spec.estimated_bytes_per_second() == 176400

        # Test WAV PCM 24-bit mono at 16kHz
        spec = AudioQualitySpecification(
            sample_rate=16000,
            channels=1,
            bit_depth=24,
            format=AudioFormat.WAV,
            codec=AudioCodec.PCM_S24LE,
        )
        # 16000 samples/s * 1 channel * 3 bytes/sample = 48000 bytes/s
        assert spec.estimated_bytes_per_second() == 48000

        # Test non-PCM formats (approximate calculations)
        spec = AudioQualitySpecification(
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            format=AudioFormat.MP3,
            codec=AudioCodec.MP3,
        )
        # Should be using approximation for MP3
        assert spec.estimated_bytes_per_second() > 0
