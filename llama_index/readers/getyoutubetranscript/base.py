"""LlamaIndex reader for YouTube transcripts via the GetYouTubeTranscript API."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from pydantic import Field, PrivateAttr

from getyoutubetranscript import Client, to_timed_text

API_KEY_ENV = "GETYOUTUBETRANSCRIPT_API_KEY"


class GetYouTubeTranscriptReader(BasePydanticReader):
    """Load YouTube video transcripts as Documents, one per video.

    Same ``load_data(ytlinks=[...])`` call as ``YoutubeTranscriptReader``, but the transcripts are
    fetched by the GetYouTubeTranscript API, so it works from cloud servers without proxies.

    .. code-block:: python

        from llama_index.readers.getyoutubetranscript import GetYouTubeTranscriptReader

        docs = GetYouTubeTranscriptReader().load_data(ytlinks=["https://youtu.be/jNQXAC9IVRw"])

    Args:
        api_key: API key. Defaults to the ``GETYOUTUBETRANSCRIPT_API_KEY`` environment variable.
        language: Caption language code, e.g. ``"en"``.
        timestamps: ``text`` becomes ``[m:ss]`` lines and ``metadata["segments"]`` holds the timings.
    """

    is_remote: bool = True
    api_key: str | None = Field(default=None, exclude=True)
    language: str | None = None
    timestamps: bool = False
    _client: Client | None = PrivateAttr(default=None)

    @classmethod
    def class_name(cls) -> str:
        return "GetYouTubeTranscriptReader"

    @property
    def client(self) -> Client:
        if self._client is None:
            key = self.api_key or os.environ.get(API_KEY_ENV)
            if not key:
                raise ValueError(
                    f"A GetYouTubeTranscript API key is required. Pass api_key=... or set {API_KEY_ENV}. "
                    "Get one at https://getyoutubetranscript.com/dashboard"
                )
            self._client = Client(api_key=key)
        return self._client

    def lazy_load_data(self, ytlinks: list[str], **load_kwargs: Any) -> Iterable[Document]:
        language = load_kwargs.get("language", self.language)
        timestamps = load_kwargs.get("timestamps", self.timestamps)
        for link in ytlinks:
            result = self.client.get_transcript(link, language=language, timestamps=timestamps)
            timed = timestamps and bool(result.get("segments"))
            metadata: dict[str, Any] = {
                "url": f"https://www.youtube.com/watch?v={result['video_id']}",
                "video_id": result["video_id"],
                "title": result.get("title", ""),
                "author_name": result.get("author_name", ""),
                "language_code": result.get("language_code", ""),
                "word_count": result.get("word_count", 0),
            }
            if timed:
                metadata["segments"] = result["segments"]
            yield Document(
                text=to_timed_text(result) if timed else result.get("transcript", ""),
                id_=result["video_id"],
                metadata=metadata,
                # Segments are long; keep them out of what embeddings and LLM prompts see.
                excluded_embed_metadata_keys=["segments"],
                excluded_llm_metadata_keys=["segments"],
            )

    def load_data(self, ytlinks: list[str], **load_kwargs: Any) -> list[Document]:
        """Load one Document per video. ``language`` and ``timestamps`` can be overridden per call."""
        return list(self.lazy_load_data(ytlinks, **load_kwargs))
