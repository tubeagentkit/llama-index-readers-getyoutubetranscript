from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import MetadataMode

from llama_index.readers.getyoutubetranscript import GetYouTubeTranscriptReader

RESULT = {
    "video_id": "jNQXAC9IVRw",
    "language_code": "en",
    "title": "Me at the zoo",
    "author_name": "jawed",
    "transcript": "All right, so here we are",
    "word_count": 6,
    "segments": [{"start": 61.0, "duration": 2.0, "text": "All right, so here we are"}],
}


def _reader(**kwargs):
    reader = GetYouTubeTranscriptReader(api_key="test-key", **kwargs)
    client = MagicMock()
    client.get_transcript.return_value = RESULT
    reader._client = client
    return reader, client


def test_is_a_reader():
    assert issubclass(GetYouTubeTranscriptReader, BaseReader)
    assert GetYouTubeTranscriptReader.class_name() == "GetYouTubeTranscriptReader"


def test_one_document_per_link():
    reader, client = _reader(language="en")
    docs = reader.load_data(ytlinks=["a", "b"])
    assert len(docs) == 2
    assert client.get_transcript.call_args.kwargs == {"language": "en", "timestamps": False}
    assert docs[0].text == "All right, so here we are"
    assert docs[0].id_ == "jNQXAC9IVRw"
    assert docs[0].metadata["title"] == "Me at the zoo"
    assert "segments" not in docs[0].metadata


def test_timestamps_and_segments_hidden_from_embeddings():
    reader, _ = _reader()
    doc = reader.load_data(ytlinks=["a"], timestamps=True)[0]
    assert doc.text == "[1:01] All right, so here we are"
    assert doc.metadata["segments"] == RESULT["segments"]
    assert "segments" not in doc.get_content(metadata_mode=MetadataMode.EMBED)


def test_api_key_not_serialized(monkeypatch):
    assert "api_key" not in GetYouTubeTranscriptReader(api_key="secret").model_dump()
    monkeypatch.delenv("GETYOUTUBETRANSCRIPT_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GETYOUTUBETRANSCRIPT_API_KEY"):
        _ = GetYouTubeTranscriptReader().client
