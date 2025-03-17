class AudioChunk:
    """A chunk of audio from a processing job."""

    # ... existing code ...

    def is_complete(self) -> bool:
        """
        Check if the chunk has completed processing.

        Returns:
            bool: True if the chunk is fully processed, False otherwise
        """
        from pyhearingai.core.models.enums import ChunkStatus

        # A chunk is complete if its status is COMPLETED or SEGMENTS_TRANSCRIBED
        return self.status in [ChunkStatus.COMPLETED, ChunkStatus.SEGMENTS_TRANSCRIBED]
