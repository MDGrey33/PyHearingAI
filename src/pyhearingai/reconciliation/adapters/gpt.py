"""
GPT-4 adapter for reconciling diarization and transcription results.

This module provides an adapter for using OpenAI's GPT-4 to reconcile
diarization and transcription results into a coherent final transcript.
"""

import logging
import json
import os
import re
from typing import Dict, List, Optional, Any, Union

import openai

from pyhearingai.core.models import Segment, DiarizationSegment
from pyhearingai.core.idempotent import ProcessingJob
from pyhearingai.reconciliation.adapters.base import BaseReconciliationAdapter

logger = logging.getLogger(__name__)


class GPT4ReconciliationAdapter(BaseReconciliationAdapter):
    """
    Adapter for using GPT-4 to reconcile diarization and transcription data.

    This adapter handles the creation of prompts, communication with the OpenAI API,
    and parsing of responses to produce the final reconciled transcript.
    """

    def __init__(self, model: str = "gpt-4"):
        """
        Initialize the GPT-4 reconciliation adapter.

        Args:
            model: The GPT model to use (default: "gpt-4")
        """
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._log = logger
        if not self.client.api_key:
            logger.warning(
                "OPENAI_API_KEY environment variable not set. "
                "Please set the OPENAI_API_KEY environment variable."
            )

    def reconcile(
        self,
        job: ProcessingJob,
        diarization_segments: Dict[str, List[DiarizationSegment]],
        transcription_segments: Dict[str, List[Segment]],
        segment_transcriptions: Dict[str, str],
        options: Optional[Dict[str, Any]] = None,
    ) -> List[Segment]:
        """
        Reconcile diarization and transcription data using GPT-4.

        Args:
            job: The processing job
            diarization_segments: Dictionary mapping chunk IDs to diarization segments
            transcription_segments: Dictionary mapping chunk IDs to transcription segments
            segment_transcriptions: Dictionary mapping segment IDs to transcribed text
            options: Optional settings for the reconciliation process

        Returns:
            List of reconciled segments with speaker information
        """
        # Verify the API key is available
        if not self.client.api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set the OPENAI_API_KEY environment variable."
            )

        # Generate the prompt
        prompt = self._create_prompt(
            job, diarization_segments, transcription_segments, segment_transcriptions, options
        )

        # Send to GPT-4
        try:
            logger.debug(f"Sending reconciliation request to GPT-4 for job {job.id}")

            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(options)},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more deterministic output
                max_tokens=4000,
            )

            # Extract and parse the JSON response
            content = response.choices[0].message.content
            result = self._parse_response(content)

            return result

        except Exception as e:
            logger.error(f"Error during GPT-4 reconciliation: {str(e)}")
            raise

    def _get_system_prompt(self, options: Optional[Dict[str, Any]] = None) -> str:
        """Generate the system prompt for GPT-4"""
        return """You are an expert audio transcription assistant. Your task is to reconcile speaker diarization 
and transcription data from multiple audio chunks into a coherent final transcript.

Follow these guidelines:
1. Maintain speaker continuity across chunk boundaries
2. Resolve discrepancies between whole-chunk and segment transcriptions
3. Handle multilingual content appropriately
4. Properly attribute overlapping speech
5. Output the final transcript as a JSON structure

The user will provide you with information about diarization segments, transcription segments, 
and individual segment transcriptions for each audio chunk. Use this information to create the best 
possible final transcript."""

    def _create_prompt(
        self,
        job: ProcessingJob,
        diarization_segments: Dict[str, List[DiarizationSegment]],
        transcription_segments: Dict[str, List[Segment]],
        segment_transcriptions: Dict[str, str],
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create the prompt for GPT-4.

        Args:
            job: The processing job
            diarization_segments: Dictionary mapping chunk IDs to diarization segments
            transcription_segments: Dictionary mapping chunk IDs to transcription segments
            segment_transcriptions: Dictionary mapping segment IDs to transcribed text
            options: Optional settings for the reconciliation process

        Returns:
            The prompt string for GPT-4
        """
        # Build the prompt
        prompt_parts = [
            f"# Audio Transcription Reconciliation Task\n\n",
            f"## Overview\n\n",
        ]

        # Add source file info if available
        if hasattr(job, "original_audio_path"):
            if isinstance(job.original_audio_path, str):
                prompt_parts.append(f"Source file: {os.path.basename(job.original_audio_path)}\n")
            else:
                prompt_parts.append(f"Source file: {job.original_audio_path.name}\n")

        # Add duration and chunk info
        prompt_parts.append(
            f"Total duration: {self._format_duration(job.duration if hasattr(job, 'duration') else 0)}\n"
        )
        prompt_parts.append(f"Number of chunks: {len(diarization_segments)}\n\n")
        prompt_parts.append(f"## Audio Chunks\n\n")

        # Sort chunk IDs by their order in the audio
        sorted_chunk_ids = list(diarization_segments.keys())
        try:
            # Try to sort chunks if they have a numerical index in their ID
            sorted_chunk_ids.sort(key=lambda cid: int(cid.split("_")[-1]) if "_" in cid else cid)
        except:
            # Fall back to alphabetical sorting if that fails
            sorted_chunk_ids.sort()

        # Add information about each chunk
        for chunk_id in sorted_chunk_ids:
            prompt_parts.append(f"### Chunk: {chunk_id}\n\n")

            # Add diarization segments for this chunk
            di_segments = diarization_segments.get(chunk_id, [])
            tr_segments = transcription_segments.get(chunk_id, [])

            prompt_parts.append(f"#### Diarization Segments ({len(di_segments)} segments)\n\n")
            for i, segment in enumerate(di_segments):
                prompt_parts.append(
                    f"- Speaker {segment.speaker_id}: {self._format_timestamp(segment.start)} → "
                    f"{self._format_timestamp(segment.end)} "
                    f"(Duration: {self._format_duration(segment.end - segment.start)})\n"
                )

                # If we have a transcription for this segment, include it
                segment_id = f"{chunk_id}_segment_{i}_{segment.speaker_id}"
                if segment_id in segment_transcriptions:
                    text = segment_transcriptions[segment_id]
                    prompt_parts.append(f'  - **Segment Transcription**: "{text}"\n')

            prompt_parts.append(f"\n#### Chunk Transcription ({len(tr_segments)} segments)\n\n")
            for segment in tr_segments:
                prompt_parts.append(
                    f"- {self._format_timestamp(segment.start)} → "
                    f'{self._format_timestamp(segment.end)}: "{segment.text}"\n'
                )

            prompt_parts.append("\n")

        # Add instructions for the response format
        prompt_parts.append(
            f"""
## Task

Please reconcile the diarization and transcription data to create a coherent final transcript.
Your response should be a valid JSON object with this structure:

```json
{{
  "segments": [
    {{
      "speaker_id": "SPEAKER1",
      "start": 0.0,
      "end": 5.2,
      "text": "Transcribed text for this segment."
    }},
    ...more segments...
  ]
}}
```

Guidelines:
1. Merge segments from the same speaker that are close together (less than 0.5 seconds apart)
2. Use the segment-level transcriptions when available, as they're more accurate
3. Use the chunk-level transcriptions to fill in any gaps
4. Maintain proper time ordering of all segments
5. Ensure speaker continuity across chunk boundaries
6. Handle any multilingual content appropriately
"""
        )

        return "".join(prompt_parts)

    def _parse_response(self, response_content: str) -> List[Segment]:
        """
        Parse the JSON response from GPT-4.

        Args:
            response_content: The JSON string response from GPT-4

        Returns:
            List of reconciled segments
        """
        try:
            # Initialize the JSON content
            json_content = response_content
            
            # Check for different code block formats (```json, ```JSON, ``` with no language spec)
            code_block_patterns = [
                r'```(?:json|JSON)?[\s\n]+(.*?)[\s\n]*```',  # Standard markdown code block
                r'{[\s\n]*"segments"[\s\n]*:.*?}',  # Direct JSON object with segments
                r'\[[\s\n]*{.*?}[\s\n]*(?:,[\s\n]*{.*?})*[\s\n]*\]'  # Direct JSON array
            ]
            
            # Try each pattern
            for pattern in code_block_patterns:
                matches = re.findall(pattern, response_content, re.DOTALL)
                if matches:
                    # Use the first match as our JSON content
                    json_content = matches[0].strip()
                    logger.debug(f"Extracted JSON with pattern {pattern}: {json_content[:100]}...")
                    break
            
            # If the content doesn't look like JSON, try to find any JSON-like structure
            if not (json_content.startswith('{') or json_content.startswith('[')):
                # Look for { or [ as starting points
                json_start = max(response_content.find('{'), response_content.find('['))
                if json_start >= 0:
                    # Find matching closing bracket
                    if response_content[json_start] == '{':
                        # Look for matching }
                        depth = 0
                        for i, char in enumerate(response_content[json_start:]):
                            if char == '{':
                                depth += 1
                            elif char == '}':
                                depth -= 1
                                if depth == 0:
                                    json_content = response_content[json_start:json_start+i+1]
                                    break
                    else:
                        # Look for matching ]
                        depth = 0
                        for i, char in enumerate(response_content[json_start:]):
                            if char == '[':
                                depth += 1
                            elif char == ']':
                                depth -= 1
                                if depth == 0:
                                    json_content = response_content[json_start:json_start+i+1]
                                    break
            
            # Try to parse the JSON
            data = json.loads(json_content)

            # Handle both direct array response and object with segments key
            if isinstance(data, list):
                segment_list = data
            elif isinstance(data, dict):
                segment_list = data.get("segments", [])
            else:
                raise ValueError(f"Unexpected data type: {type(data)}")

            # Extract the segments with more robust handling
            segments = []
            for segment_data in segment_list:
                # Handle differently formatted time values
                start = segment_data.get("start", 0)
                end = segment_data.get("end", 0)
                
                # Convert string time formats (HH:MM:SS.mmm) to seconds
                if isinstance(start, str):
                    if ':' in start:
                        parts = start.replace(',', '.').split(':')
                        if len(parts) == 2:  # MM:SS format
                            start = float(parts[0]) * 60 + float(parts[1])
                        elif len(parts) == 3:  # HH:MM:SS format
                            start = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                    else:
                        # Try direct conversion, with fallback to 0
                        try:
                            start = float(start)
                        except ValueError:
                            start = 0
                
                if isinstance(end, str):
                    if ':' in end:
                        parts = end.replace(',', '.').split(':')
                        if len(parts) == 2:  # MM:SS format
                            end = float(parts[0]) * 60 + float(parts[1])
                        elif len(parts) == 3:  # HH:MM:SS format
                            end = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                    else:
                        # Try direct conversion, with fallback to 0
                        try:
                            end = float(end)
                        except ValueError:
                            end = 0
                
                # Create the segment, with fallbacks for missing fields
                try:
                    segment = Segment(
                        text=segment_data.get("text", ""),
                        start=float(start),
                        end=float(end),
                        speaker_id=segment_data.get("speaker_id", None),
                    )
                    segments.append(segment)
                except Exception as segment_error:
                    logger.warning(f"Error creating segment: {str(segment_error)}, skipping")
                    continue

            # Sort segments by start time to ensure proper order
            segments.sort(key=lambda s: s.start)
            
            # Log summary
            logger.info(f"Successfully parsed {len(segments)} segments")
            
            return segments

        except Exception as e:
            import traceback
            logger.error(f"Error parsing GPT-4 response: {str(e)}")
            logger.error(f"Response content: {response_content}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Include the failed response in the error to aid extraction in ReconciliationService
            error_msg = f"Failed to parse GPT-4 response: {str(e)}\n\n{response_content}"
            raise ValueError(error_msg)

    def _format_timestamp(self, seconds: float) -> str:
        """Format a timestamp in seconds as HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def _format_duration(self, seconds: float) -> str:
        """Format a duration in seconds as HH:MM:SS.mmm"""
        return self._format_timestamp(seconds)
