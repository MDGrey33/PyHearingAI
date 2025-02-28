"""
OpenAI Whisper API transcriber implementation.

This module provides a concrete implementation of the Transcriber port
using the OpenAI Whisper API.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import openai

from pyhearingai.core.models import Segment
from pyhearingai.core.ports import Transcriber
from pyhearingai.infrastructure.registry import register_transcriber


logger = logging.getLogger(__name__)


@register_transcriber("whisper_openai")
class WhisperOpenAITranscriber(Transcriber):
    """Transcriber implementation using the OpenAI Whisper API."""
    
    def __init__(self):
        """Initialize the OpenAI Whisper transcriber."""
        self._api_key = os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            logger.warning(
                "OPENAI_API_KEY environment variable not set. "
                "You will need to provide the API key when calling transcribe()."
            )
    
    @property
    def name(self) -> str:
        """Get the name of the transcriber."""
        return "whisper_openai"
    
    @property
    def supports_segmentation(self) -> bool:
        """Whether this transcriber provides timing and segmentation."""
        return True
    
    def transcribe(self, audio_path: Path, **kwargs) -> List[Segment]:
        """
        Transcribe speech in an audio file using the OpenAI Whisper API.
        
        Args:
            audio_path: Path to the audio file to transcribe
            **kwargs: Additional transcription options
                - api_key: OpenAI API key (overrides environment variable)
                - model: Whisper model to use (default: "whisper-1")
                - language: Language code (default: None, auto-detect)
                - prompt: Initial prompt for the model (default: None)
                - temperature: Sampling temperature (default: 0)
                - response_format: Format of the response (default: "verbose_json")
        
        Returns:
            List of segments with transcribed text and timing information
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Extract options
        api_key = kwargs.get("api_key", self._api_key)
        model = kwargs.get("model", "whisper-1")
        language = kwargs.get("language")
        prompt = kwargs.get("prompt")
        temperature = kwargs.get("temperature", 0)
        response_format = kwargs.get("response_format", "verbose_json")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set the OPENAI_API_KEY environment variable or provide it as api_key parameter."
            )
        
        # Set the API key
        openai.api_key = api_key
        
        try:
            # Build request parameters
            params = {
                "model": model,
                "temperature": temperature,
                "response_format": response_format
            }
            
            if language:
                params["language"] = language
                
            if prompt:
                params["prompt"] = prompt
            
            # Make the API request
            logger.debug(f"Transcribing {audio_path} with OpenAI Whisper API")
            
            with open(audio_path, "rb") as audio_file:
                response = openai.audio.transcriptions.create(
                    file=audio_file,
                    **params
                )
            
            # Parse the response
            logger.debug("Parsing OpenAI Whisper API response")
            
            # In verbose_json mode, we get segments with timing information
            segments = []
            
            # Check response format and parse accordingly
            if response_format == "verbose_json":
                # Extract segments from the response
                for segment_data in response.segments:
                    segment = Segment(
                        text=segment_data.text,
                        start=segment_data.start,
                        end=segment_data.end
                    )
                    segments.append(segment)
            else:
                # For simple text response, create a single segment
                segments.append(Segment(
                    text=response.text,
                    start=0.0,
                    end=0.0  # We don't know the duration
                ))
            
            return segments
            
        except Exception as e:
            logger.error(f"Error transcribing with OpenAI Whisper API: {str(e)}")
            raise 