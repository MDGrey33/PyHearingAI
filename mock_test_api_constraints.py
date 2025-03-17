#!/usr/bin/env python
"""
Test script to demonstrate API constraints, audio validation, and size-aware conversion.

This script showcases the capabilities of:
1. The API constraints domain model for defining provider-specific limits
2. The AudioValidationService for validating audio files against quality and API constraints
3. The SizeAwareFFmpegConverter for intelligently reducing file sizes to meet constraints

Usage:
    python mock_test_api_constraints.py [audio_file_path]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.pyhearingai.core.domain.api_constraints import ApiProvider, ApiSizeLimitPolicy
from src.pyhearingai.core.domain.audio_quality import AudioQualitySpecification
from src.pyhearingai.core.domain.audio_validation import AudioValidationService
from src.pyhearingai.core.domain.events import (
    AudioConversionEvent,
    AudioSizeExceededEvent,
    AudioValidationEvent,
    EventPublisher,
)
from src.pyhearingai.infrastructure.adapters.audio_format_service import FFmpegAudioFormatService
from src.pyhearingai.infrastructure.adapters.size_aware_audio_converter import (
    SizeAwareFFmpegConverter,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mock_test_api_constraints")

# Create a temp directory for output files
TEMP_DIR = Path("test_outputs")
TEMP_DIR.mkdir(exist_ok=True)


def log_event(event):
    """Log domain events."""
    if isinstance(event, AudioValidationEvent):
        if event.is_valid:
            logger.info("✅ Audio file validated: %s", event.file_path)
        else:
            logger.error("❌ Audio validation failed: %s", event.error_message)

    elif isinstance(event, AudioConversionEvent):
        if event.is_successful:
            compression = event.metadata.get("compression_ratio", 0)
            adjustments = event.metadata.get("adjustments_made", [])

            logger.info(
                "✅ Conversion successful: %s -> %s", event.source_path.name, event.target_path.name
            )
            logger.info("   Original size: %.2f MB", event.original_size / 1_000_000)
            logger.info("   Converted size: %.2f MB", event.converted_size / 1_000_000)

            if compression:
                logger.info("   Compression ratio: %.2fx", compression)

            if adjustments:
                logger.info("   Adjustments: %s", adjustments)
        else:
            logger.error("❌ Conversion failed: %s", event.error_message)

    elif isinstance(event, AudioSizeExceededEvent):
        logger.warning("⚠️ Size limit exceeded: %s", event.source_path)
        logger.warning("   Target size: %.2f MB", event.target_size_bytes / 1_000_000)
        logger.warning("   Best achieved: %.2f MB", event.best_achieved_size / 1_000_000)

        if event.adjustments_tried:
            logger.warning("   Adjustments tried: %s", event.adjustments_tried)


def test_api_constraints(audio_path):
    """Test API constraints with different providers."""
    logger.info("🔍 Testing API constraints with %s", audio_path)

    # Subscribe to domain events
    EventPublisher.subscribe(AudioValidationEvent, log_event)
    EventPublisher.subscribe(AudioConversionEvent, log_event)
    EventPublisher.subscribe(AudioSizeExceededEvent, log_event)

    # Create services
    format_service = FFmpegAudioFormatService()
    converter = SizeAwareFFmpegConverter()

    # Get audio metadata
    try:
        metadata = format_service.get_audio_metadata(audio_path)
        logger.info("📊 Audio metadata:")
        for key, value in metadata.items():
            logger.info("   %s: %s", key, value)
    except Exception as e:
        logger.error("Error getting metadata: %s", str(e))
        return

    # Test validation against different providers
    providers = [ApiProvider.OPENAI_WHISPER, ApiProvider.ASSEMBLY_AI, ApiProvider.GOOGLE_SPEECH]

    for provider in providers:
        # Get provider limits
        limits = ApiSizeLimitPolicy.get_limit_for_provider(provider)
        logger.info("\n📋 Testing against %s limits:", provider.name)
        logger.info("   Max file size: %.2f MB", limits.max_file_size_bytes / 1_000_000)
        logger.info(
            "   Max duration: %.1f minutes if applicable (0 means no limit)",
            limits.max_duration_seconds / 60,
        )

        # Validate file
        is_valid, error_message = AudioValidationService.validate_audio_file(audio_path, provider)

        if is_valid:
            logger.info("✅ File is valid for %s", provider.name)
        else:
            logger.info("❌ File is invalid for %s: %s", provider.name, error_message)

            # Try to suggest quality reduction
            current_spec = AudioQualitySpecification.high_quality()
            suggestion = AudioValidationService.suggest_quality_reduction(
                audio_path, current_spec, provider
            )
            if suggestion:
                logger.info("💡 Suggested quality reduction: %s", suggestion)

            # Try size-constrained conversion
            logger.info("🔄 Attempting size-constrained conversion for %s...", provider.name)
            try:
                output_path = (
                    TEMP_DIR / f"{audio_path.stem}_{provider.name.lower()}{audio_path.suffix}"
                )
                converted_path, conversion_metadata = converter.convert_with_size_constraint(
                    audio_path, limits.max_file_size_bytes, output_dir=TEMP_DIR
                )
                logger.info("✅ Successfully converted to meet %s size constraints", provider.name)
                logger.info("   Output: %s", converted_path)
            except ValueError as e:
                logger.error("❌ Could not meet size constraints: %s", str(e))

    # Test with custom quality specifications
    logger.info("\n🔄 Testing with different quality specifications:")

    quality_specs = [
        ("High Quality", AudioQualitySpecification.high_quality()),
        ("Whisper API", AudioQualitySpecification.for_whisper_api()),
        ("Compressed", AudioQualitySpecification.for_compressed()),
    ]

    for name, spec in quality_specs:
        logger.info("\n📊 Converting with %s specification:", name)
        logger.info("   Sample rate: %s Hz", spec.sample_rate)
        logger.info("   Channels: %s", spec.channels)
        logger.info("   Format: %s", spec.format.value)

        try:
            output_path = (
                TEMP_DIR / f"{audio_path.stem}_{name.lower().replace(' ', '_')}.{spec.format.value}"
            )
            converted_path = converter.convert_with_quality_spec(
                audio_path, spec, output_path=output_path
            )

            # Get metadata for the converted file
            converted_metadata = format_service.get_audio_metadata(converted_path)
            logger.info("✅ Converted file:")
            logger.info("   Path: %s", converted_path)
            logger.info("   Size: %.2f MB", converted_metadata.get("size_bytes", 0) / 1_000_000)
            logger.info("   Duration: %.2f seconds", converted_metadata.get("duration", 0))
            logger.info("   Sample rate: %s Hz", converted_metadata.get("sample_rate", 0))
            logger.info("   Channels: %s", converted_metadata.get("channels", 0))
        except Exception as e:
            logger.error("❌ Conversion failed: %s", str(e))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test API constraints, audio validation, and size-aware conversion"
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default="test data/short_conversation.m4a",
        help="Path to audio file (default: test data/short_conversation.m4a)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        logger.error("Audio file not found: %s", audio_path)
        return 1

    test_api_constraints(audio_path)

    logger.info("\n✨ Done! Converted files are in %s", TEMP_DIR.absolute())
    return 0


if __name__ == "__main__":
    sys.exit(main())
