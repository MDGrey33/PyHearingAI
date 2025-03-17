"""
Initialization module for PyHearingAI.

This module explicitly imports all components to ensure they are registered
before they are used. This is important for the plugin architecture to work.
"""

# Direct imports of all components that need to be registered
from pyhearingai.infrastructure.diarizers.pyannote import PyannoteDiarizer
from pyhearingai.infrastructure.formatters.json import JSONFormatter
from pyhearingai.infrastructure.formatters.markdown import MarkdownFormatter
from pyhearingai.infrastructure.formatters.srt import SRTFormatter
from pyhearingai.infrastructure.formatters.text import TextFormatter
from pyhearingai.infrastructure.formatters.vtt import VTTFormatter
from pyhearingai.infrastructure.transcribers.whisper_openai import WhisperOpenAITranscriber


# Function to initialize all components
def initialize_all_components():
    """
    Initialize all components by accessing them, which ensures they are registered.
    This function should be called early in the application startup.
    """
    # Access the components to ensure they are loaded
    components = [
        PyannoteDiarizer,
        WhisperOpenAITranscriber,
        TextFormatter,
        JSONFormatter,
        SRTFormatter,
        VTTFormatter,
        MarkdownFormatter,
    ]
    return True


# Call the initialization function
initialize_all_components()
