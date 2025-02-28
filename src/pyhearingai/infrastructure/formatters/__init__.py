"""
Formatters package initialization.

This module imports all formatter implementations to ensure they are registered
with the registry when the package is imported.
"""

# Import all formatters so they are registered
from pyhearingai.infrastructure.formatters.text import TextFormatter
from pyhearingai.infrastructure.formatters.json import JSONFormatter
from pyhearingai.infrastructure.formatters.srt import SRTFormatter
from pyhearingai.infrastructure.formatters.vtt import VTTFormatter
from pyhearingai.infrastructure.formatters.markdown import MarkdownFormatter
