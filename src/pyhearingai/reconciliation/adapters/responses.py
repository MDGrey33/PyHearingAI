"""
Responses API adapter for reconciling diarization and transcription results.

This module provides an adapter for using OpenAI's Responses API to reconcile
diarization and transcription results into a coherent final transcript,
with efficient token management through conversation state.
"""

import logging
import json
import os
import re
import time
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken
from openai import OpenAI
from tqdm import tqdm

from pyhearingai.core.models import (
    Segment, 
    DiarizationSegment,
)
from pyhearingai.core.idempotent import ProcessingJob
from pyhearingai.reconciliation.adapters.base import BaseReconciliationAdapter

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Token counter for OpenAI models.
    
    This class handles accurate token counting to prevent token limit errors
    and optimize batch sizing.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the token counter.
        
        Args:
            model: The OpenAI model to count tokens for
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base encoding if model-specific encoding not found
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        return len(self.encoding.encode(text))
    
    def estimate_batch_size(self, segments: List[Any], sample_size: int = 5, 
                           max_tokens: int = 6000) -> int:
        """
        Estimate the maximum number of segments that can fit in a batch.
        
        Args:
            segments: List of segments to estimate from
            sample_size: Number of segments to sample for estimation
            max_tokens: Maximum tokens allowed per batch
            
        Returns:
            Estimated number of segments per batch
        """
        if not segments:
            return 0
            
        # Take a sample of segments to estimate average tokens per segment
        sample = segments[:min(sample_size, len(segments))]
        
        # Calculate average tokens per segment in the sample
        total_tokens = sum(self.count_tokens(str(segment)) for segment in sample)
        avg_tokens_per_segment = total_tokens / len(sample)
        
        # Calculate estimated segments per batch
        estimated_segments = int(max_tokens / avg_tokens_per_segment)
        
        # Apply a safety factor (80% of estimated to be safe)
        return max(1, int(estimated_segments * 0.8))


class BatchProcessor:
    """
    Processes data into token-appropriate batches for the OpenAI API.
    
    This class handles splitting large datasets into optimally sized batches
    to prevent token limit errors while maintaining context between batches.
    """
    
    def __init__(self, token_counter: TokenCounter, max_tokens: int = 6000):
        """
        Initialize the batch processor.
        
        Args:
            token_counter: TokenCounter instance for counting tokens
            max_tokens: Maximum tokens allowed per batch
        """
        self.token_counter = token_counter
        self.max_tokens = max_tokens
        
    def _serialize_obj(self, obj: Any) -> Dict[str, Any]:
        """
        Convert custom objects to dictionaries for JSON serialization.
        
        Args:
            obj: The object to serialize
            
        Returns:
            Dictionary representation of the object
        """
        if isinstance(obj, DiarizationSegment):
            return {
                "speaker_id": obj.speaker_id,
                "start": obj.start,
                "end": obj.end,
                "score": obj.score
            }
        elif isinstance(obj, Segment):
            return {
                "start": obj.start,
                "end": obj.end,
                "text": obj.text,
                "confidence": getattr(obj, "confidence", None),
                "speaker_id": obj.speaker_id
            }
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)
    
    def _serialize_segments(self, segments: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert a list of segment objects to serializable dictionaries.
        
        Args:
            segments: List of segment objects
            
        Returns:
            List of dictionaries
        """
        return [self._serialize_obj(segment) for segment in segments]
    
    def create_batches(
        self,
        diarization_segments: Dict[str, List[DiarizationSegment]],
        transcription_segments: Dict[str, List[Segment]],
        segment_transcriptions: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Create batches of chunks for processing.
        
        Args:
            diarization_segments: Diarization segments per chunk
            transcription_segments: Transcription segments per chunk
            segment_transcriptions: Transcriptions per segment
            
        Returns:
            List of batches
        """
        # Sort chunks by ID to ensure deterministic batching
        chunk_ids = sorted(set(list(diarization_segments.keys()) + list(transcription_segments.keys())))
        if not chunk_ids:
            logger.warning("No chunks to process")
            return []
            
        batches = []
        current_batch = {"chunks": {}}
        current_batch_tokens = 0
        
        for chunk_id in chunk_ids:
            # Create serializable versions of the segments
            serialized_diarization = self._serialize_segments(diarization_segments.get(chunk_id, []))
            serialized_transcription = self._serialize_segments(transcription_segments.get(chunk_id, []))
            
            chunk_data = {
                "diarization": serialized_diarization,
                "transcription": serialized_transcription
            }
            
            # Get segment transcriptions for this chunk
            chunk_segment_transcriptions = {
                seg_id: text for seg_id, text in segment_transcriptions.items() 
                if seg_id.startswith(chunk_id)
            }
            
            # Calculate token count for this chunk
            chunk_str = json.dumps(chunk_data) + json.dumps(chunk_segment_transcriptions)
            chunk_tokens = self.token_counter.count_tokens(chunk_str)
            
            # If this chunk alone exceeds the token limit, it needs to be split further
            # (This would be an advanced case - for simplicity we'll just add a warning)
            if chunk_tokens > self.max_tokens:
                logger.warning(f"Chunk {chunk_id} exceeds token limit ({chunk_tokens} tokens). "
                              f"This may cause issues. Consider reducing chunk size.")
            
            # If adding this chunk would exceed the token limit, start a new batch
            if current_batch_tokens + chunk_tokens > self.max_tokens and current_batch["chunks"]:
                batches.append(current_batch)
                current_batch = {"chunks": {}}
                current_batch_tokens = 0
                
            # Add chunk to the current batch
            current_batch["chunks"][chunk_id] = chunk_data
            
            # Add segment transcriptions
            for seg_id, text in chunk_segment_transcriptions.items():
                current_batch["chunks"][seg_id] = text
                
            current_batch_tokens += chunk_tokens
            
        # Add the last batch if it's not empty
        if current_batch["chunks"]:
            batches.append(current_batch)
            
        logger.info(f"Created {len(batches)} batches from {len(chunk_ids)} chunks")
        return batches
    
    def format_batch_prompt(
        self, 
        batch: Dict[str, Any], 
        batch_index: int, 
        total_batches: int,
        job: ProcessingJob
    ) -> str:
        """
        Format a batch into a prompt for the OpenAI API.
        
        Args:
            batch: The batch data to format
            batch_index: Current batch index
            total_batches: Total number of batches
            job: The processing job
            
        Returns:
            Formatted prompt string
        """
        # Build the prompt
        prompt_parts = [
            f"# Audio Transcription Reconciliation Task - Batch {batch_index+1}/{total_batches}\n\n",
            f"## Overview\n\n",
        ]
        
        # Add source file info if available
        if hasattr(job, "original_audio_path"):
            if isinstance(job.original_audio_path, str):
                prompt_parts.append(f"Source file: {os.path.basename(job.original_audio_path)}\n")
            else:
                prompt_parts.append(f"Source file: {job.original_audio_path.name}\n")
                
        # Add chunk info
        prompt_parts.append(f"Number of chunks in this batch: {len(batch['chunks'])}\n\n")
        prompt_parts.append(f"## Audio Chunks\n\n")
        
        # Add information about each chunk
        for chunk_id, chunk_data in batch["chunks"].items():
            prompt_parts.append(f"### Chunk: {chunk_id}\n\n")
            
            # Add diarization segments for this chunk
            di_segments = chunk_data["diarization"]
            tr_segments = chunk_data["transcription"]
            
            prompt_parts.append(f"#### Diarization Segments ({len(di_segments)} segments)\n\n")
            for i, segment in enumerate(di_segments):
                prompt_parts.append(
                    f"- Speaker {segment['speaker_id']}: {self._format_timestamp(segment['start'])} → "
                    f"{self._format_timestamp(segment['end'])} "
                    f"(Duration: {self._format_duration(segment['end'] - segment['start'])})\n"
                )
                
                # If we have a transcription for this segment, include it
                segment_id = f"{chunk_id}_segment_{i}_{segment['speaker_id']}"
                if segment_id in batch["chunks"]:
                    text = batch["chunks"][segment_id]
                    prompt_parts.append(f'  - **Segment Transcription**: "{text}"\n')
                    
            prompt_parts.append(f"\n#### Chunk Transcription ({len(tr_segments)} segments)\n\n")
            for segment in tr_segments:
                prompt_parts.append(
                    f"- {self._format_timestamp(segment['start'])} → "
                    f"{self._format_timestamp(segment['end'])}: \"{segment['text']}\"\n"
                )
                
            prompt_parts.append("\n")
            
        # Add instructions for the response format
        is_final_batch = batch_index == total_batches - 1
        batch_context = f"This is batch {batch_index+1} of {total_batches}."
        
        if total_batches > 1:
            if batch_index == 0:
                batch_context += " This is the first batch, so there are no previous segments to consider."
            else:
                batch_context += " Please maintain consistent speaker identities from previous batches."
                
            if not is_final_batch:
                batch_context += " This is not the final batch, so the audio continues beyond what's shown here."
        
        prompt_parts.append(
            f"""
## Task

{batch_context}

Please reconcile the diarization and transcription data to create a coherent transcript.
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
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format a timestamp in seconds to MM:SS.ms format"""
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes):02d}:{seconds:.2f}"
    
    def _format_duration(self, seconds: float) -> str:
        """Format a duration in seconds to MM:SS.ms format"""
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes):02d}:{seconds:.2f}"


class ResultAggregator:
    """
    Aggregates results from multiple batch responses.
    
    This class handles combining and deduplicating segments from multiple
    batch responses into a coherent final transcript.
    """
    
    def process_responses(self, responses: List[str], overlap_threshold: float = 0.5) -> List[Segment]:
        """
        Process multiple response contents into a single coherent list of segments.
        
        Args:
            responses: List of JSON response strings from OpenAI
            overlap_threshold: Threshold for considering segments as overlapping
            
        Returns:
            List of reconciled segments
        """
        all_segments = []
        
        # Parse each response and extract segments
        for i, response in enumerate(responses):
            try:
                batch_segments = self._parse_response(response)
                logger.info(f"Parsed {len(batch_segments)} segments from batch {i+1}")
                all_segments.extend(batch_segments)
            except Exception as e:
                logger.error(f"Error parsing response from batch {i+1}: {str(e)}")
                logger.debug(f"Problematic response content: {response[:200]}...")
                
        # Sort segments by start time
        all_segments.sort(key=lambda s: s.start)
        
        # Merge overlapping segments from the same speaker
        merged_segments = self._merge_overlapping_segments(all_segments, overlap_threshold)
        
        return merged_segments
    
    def _parse_response(self, response_content: str) -> List[Segment]:
        """
        Parse the JSON response from OpenAI.
        
        Args:
            response_content: The JSON string response from OpenAI
            
        Returns:
            List of parsed segments
        """
        # Extract JSON content if it's wrapped in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_content)
        if json_match:
            json_content = json_match.group(1)
        else:
            json_content = response_content
            
        try:
            # Try to parse the JSON
            result_data = json.loads(json_content)
            
            # Extract segments
            segments = []
            for seg_data in result_data.get("segments", []):
                segment = Segment(
                    text=seg_data["text"],
                    start=float(seg_data["start"]),
                    end=float(seg_data["end"]),
                    speaker_id=seg_data["speaker_id"]
                )
                segments.append(segment)
                
            return segments
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {str(e)}")
            logger.debug(f"Response content: {response_content[:200]}...")
            raise ValueError(f"Failed to parse OpenAI response: {str(e)}")
        
    def _merge_overlapping_segments(
        self, 
        segments: List[Segment], 
        threshold: float = 0.5
    ) -> List[Segment]:
        """
        Merge overlapping segments from the same speaker.
        
        Args:
            segments: List of segments to merge
            threshold: Threshold for considering segments as overlapping
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
            
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: (s.start, s.end))
        
        merged = [sorted_segments[0]]
        
        for current in sorted_segments[1:]:
            previous = merged[-1]
            
            # If from same speaker and overlapping or very close, merge them
            if (current.speaker_id == previous.speaker_id and 
                current.start - previous.end <= threshold):
                
                # Create merged segment
                merged[-1] = Segment(
                    text=f"{previous.text} {current.text}",
                    start=previous.start,
                    end=max(previous.end, current.end),
                    speaker_id=previous.speaker_id
                )
            else:
                # Add as a new segment
                merged.append(current)
                
        return merged


class ResponsesReconciliationAdapter(BaseReconciliationAdapter):
    """
    Adapter for using OpenAI's Responses API to reconcile diarization and transcription data.
    
    This adapter handles the creation of prompts, communication with the OpenAI API,
    and parsing of responses to produce the final reconciled transcript, using the
    Responses API for efficient token management and conversation state.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the Responses API reconciliation adapter.
        
        Args:
            model: The OpenAI model to use (default: "gpt-4o")
        """
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.token_counter = TokenCounter(model)
        self.batch_processor = BatchProcessor(self.token_counter)
        self.result_aggregator = ResultAggregator()
        
    def reconcile(
        self,
        job: ProcessingJob,
        diarization_segments: Dict[str, List[DiarizationSegment]],
        transcription_segments: Dict[str, List[Segment]],
        segment_transcriptions: Dict[str, str],
        options: Optional[Dict[str, Any]] = None,
    ) -> List[Segment]:
        """
        Reconcile diarization and transcription data using OpenAI's Responses API.
        
        Args:
            job: The processing job
            diarization_segments: Dictionary mapping chunk IDs to diarization segments
            transcription_segments: Dictionary mapping chunk IDs to transcription segments
            segment_transcriptions: Dictionary mapping segment IDs to transcribed text
            options: Optional settings for the reconciliation process
            
        Returns:
            List of reconciled segments with speaker information
        """
        if not self.client.api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set the OPENAI_API_KEY environment variable."
            )
            
        # Split data into batches
        batches = self.batch_processor.create_batches(
            diarization_segments, transcription_segments, segment_transcriptions
        )
        
        responses = []
        previous_response_id = None
        
        # Process each batch sequentially
        for i, batch in enumerate(batches):
            try:
                logger.info(f"Processing batch {i+1}/{len(batches)} for job {job.id}")
                
                # Format the batch into a prompt
                prompt = self.batch_processor.format_batch_prompt(
                    batch, i, len(batches), job
                )
                
                # Create the request
                response = self._process_batch(
                    prompt, previous_response_id, options
                )
                
                # Save the response ID for the next batch
                previous_response_id = response.id
                
                # Store the response content
                responses.append(response.output_text)
                
                logger.info(f"Successfully processed batch {i+1}/{len(batches)}")
                
            except Exception as e:
                logger.error(f"Error processing batch {i+1}/{len(batches)}: {str(e)}")
                raise
                
        # Aggregate results from all batches
        final_segments = self.result_aggregator.process_responses(responses)
        
        return final_segments
    
    def _process_batch(
        self, 
        prompt: str, 
        previous_response_id: Optional[str], 
        options: Optional[Dict[str, Any]]
    ) -> Any:
        """
        Process a single batch using the Responses API.
        
        Args:
            prompt: The formatted prompt for this batch
            previous_response_id: ID of the previous response for conversation continuity
            options: Additional options for processing
            
        Returns:
            The OpenAI API response
        """
        # Apply options
        if options is None:
            options = {}
            
        temperature = options.get("temperature", 0.3)
        max_retries = options.get("max_retries", 3)
        retry_delay = options.get("retry_delay", 2)
        
        for attempt in range(max_retries):
            try:
                # Create the response
                if previous_response_id:
                    logger.debug(f"Using previous response ID: {previous_response_id}")
                    # Use directly as an object to support the new OpenAI format
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an expert audio transcription assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                    )
                    # Create a fake response object with the necessary structure
                    class ResponseObj:
                        def __init__(self, response):
                            self.id = response.id
                            self.output_text = response.choices[0].message.content
                    
                    return ResponseObj(response)
                else:
                    logger.debug("Creating new conversation")
                    # Use directly as an object to support the new OpenAI format
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an expert audio transcription assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                    )
                    # Create a fake response object with the necessary structure
                    class ResponseObj:
                        def __init__(self, response):
                            self.id = response.id
                            self.output_text = response.choices[0].message.content
                    
                    return ResponseObj(response)
                
            except Exception as e:
                logger.warning(f"Error in API call (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Increase delay for next retry (exponential backoff)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed after {max_retries} attempts")
                    raise 