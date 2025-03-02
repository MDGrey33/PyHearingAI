"""
Registry for adapter implementations.

This module provides functions to register and retrieve implementations of the core ports.
It acts as a service locator and simple dependency injection mechanism.
"""

from functools import lru_cache
from typing import Callable, Dict, Optional, Type, TypeVar

from pyhearingai.core.ports import (
    AudioConverter,
    Diarizer,
    OutputFormatter,
    SpeakerAssigner,
    Transcriber,
)

# Type variable for generic registry functions
T = TypeVar("T")

# Registries for different adapter types
_transcribers: Dict[str, Type[Transcriber]] = {}
_diarizers: Dict[str, Type[Diarizer]] = {}
_output_formatters: Dict[str, Type[OutputFormatter]] = {}
_converter_class: Optional[Type[AudioConverter]] = None
_speaker_assigner_class: Optional[Type[SpeakerAssigner]] = None


# Registration functions


def register_transcriber(name: str) -> Callable[[Type[Transcriber]], Type[Transcriber]]:
    """
    Decorator to register a transcriber implementation.

    Args:
        name: The name to register the transcriber under

    Returns:
        Decorator function
    """

    def decorator(cls: Type[Transcriber]) -> Type[Transcriber]:
        _transcribers[name] = cls
        return cls

    return decorator


def register_diarizer(name: str) -> Callable[[Type[Diarizer]], Type[Diarizer]]:
    """
    Decorator to register a diarizer implementation.

    Args:
        name: The name to register the diarizer under

    Returns:
        Decorator function
    """

    def decorator(cls: Type[Diarizer]) -> Type[Diarizer]:
        _diarizers[name] = cls
        return cls

    return decorator


def register_output_formatter(
    format_name: str,
) -> Callable[[Type[OutputFormatter]], Type[OutputFormatter]]:
    """
    Decorator to register an output formatter implementation.

    Args:
        format_name: The format name to register the formatter under (e.g., 'txt', 'json')

    Returns:
        Decorator function
    """

    def decorator(cls: Type[OutputFormatter]) -> Type[OutputFormatter]:
        _output_formatters[format_name] = cls
        return cls

    return decorator


def register_converter(cls: Type[AudioConverter]) -> Type[AudioConverter]:
    """
    Decorator to register an audio converter implementation.

    Args:
        cls: The converter class to register

    Returns:
        The registered class
    """
    global _converter_class
    _converter_class = cls
    return cls


def register_speaker_assigner(cls: Type[SpeakerAssigner]) -> Type[SpeakerAssigner]:
    """
    Decorator to register a speaker assigner implementation.

    Args:
        cls: The speaker assigner class to register

    Returns:
        The registered class
    """
    global _speaker_assigner_class
    _speaker_assigner_class = cls
    return cls


# Accessor functions


@lru_cache(maxsize=None)
def get_transcriber(name: str) -> Transcriber:
    """
    Get an instance of a transcriber by name.

    Args:
        name: The name of the transcriber to get

    Returns:
        Instance of the requested transcriber

    Raises:
        ValueError: If the transcriber is not registered
    """
    if name not in _transcribers:
        raise ValueError(
            f"Transcriber '{name}' not found. Available transcribers: {list(_transcribers.keys())}"
        )
    return _transcribers[name]()


@lru_cache(maxsize=None)
def get_diarizer(name: str) -> Diarizer:
    """
    Get an instance of a diarizer by name.

    Args:
        name: The name of the diarizer to get

    Returns:
        Instance of the requested diarizer

    Raises:
        ValueError: If the diarizer is not registered
    """
    if name not in _diarizers:
        raise ValueError(
            f"Diarizer '{name}' not found. Available diarizers: {list(_diarizers.keys())}"
        )
    return _diarizers[name]()


@lru_cache(maxsize=None)
def get_output_formatter(format_name: str) -> OutputFormatter:
    """
    Get an instance of an output formatter by format name.

    Args:
        format_name: The format name to get a formatter for

    Returns:
        Instance of the requested output formatter

    Raises:
        ValueError: If the output formatter is not registered
    """
    if format_name not in _output_formatters:
        raise ValueError(
            f"Output formatter '{format_name}' not found. Available formatters: {list(_output_formatters.keys())}"
        )
    return _output_formatters[format_name]()


@lru_cache(maxsize=None)
def get_converter() -> AudioConverter:
    """
    Get an instance of the audio converter.

    Returns:
        Instance of the audio converter

    Raises:
        ValueError: If no converter is registered
    """
    if _converter_class is None:
        raise ValueError("No audio converter registered")
    return _converter_class()


@lru_cache(maxsize=None)
def get_speaker_assigner() -> SpeakerAssigner:
    """
    Get an instance of the speaker assigner.

    Returns:
        Instance of the speaker assigner

    Raises:
        ValueError: If no speaker assigner is registered
    """
    if _speaker_assigner_class is None:
        raise ValueError("No speaker assigner registered")
    return _speaker_assigner_class()


# Registry inspection functions


def list_transcribers() -> list[str]:
    """List all registered transcribers."""
    return list(_transcribers.keys())


def list_diarizers() -> list[str]:
    """List all registered diarizers."""
    return list(_diarizers.keys())


def list_output_formatters() -> list[str]:
    """List all registered output formatters."""
    return list(_output_formatters.keys())
