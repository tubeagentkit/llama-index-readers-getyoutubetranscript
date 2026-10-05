# llama-index-readers-getyoutubetranscript

A [LlamaIndex](https://www.llamaindex.ai) reader that loads YouTube video transcripts as `Document`s, powered by the [GetYouTubeTranscript](https://getyoutubetranscript.com) API.

It has the same `load_data(ytlinks=[...])` call as `YoutubeTranscriptReader`, but transcripts are fetched by the API on its own servers. That means it keeps working on cloud servers, where YouTube blocks direct requests (`RequestBlocked` / `IpBlocked`), with no proxies to manage.

## Install

```bash
pip install llama-index-readers-getyoutubetranscript
```

Get an API key at [getyoutubetranscript.com/developers](https://getyoutubetranscript.com/developers) (free tier included):

```bash
export GETYOUTUBETRANSCRIPT_API_KEY=sk_live_...
```

## Usage

```python
from llama_index.readers.getyoutubetranscript import GetYouTubeTranscriptReader

reader = GetYouTubeTranscriptReader()
documents = reader.load_data(ytlinks=["https://youtu.be/jNQXAC9IVRw", "5e37ZT3SQbk"])

print(documents[0].metadata)
# {'url': 'https://www.youtube.com/watch?v=jNQXAC9IVRw', 'video_id': 'jNQXAC9IVRw', 'title': 'Me at the zoo',
#  'author_name': 'jawed', 'language_code': 'en', 'word_count': 39}
```

Switching from `YoutubeTranscriptReader`: change the import and class name; `load_data(ytlinks=...)` stays the same. Links can be any YouTube URL (watch, youtu.be, Shorts, live) or a video ID.

| Option | Default | Description |
| --- | --- | --- |
| `api_key` | `GETYOUTUBETRANSCRIPT_API_KEY` env var | API key (never serialized) |
| `language` | API default | Caption language code, e.g. `"en"` |
| `timestamps` | `False` | `text` becomes `[m:ss]` lines; `metadata["segments"]` holds `{start, duration, text}` per line, hidden from embeddings and LLM prompts |

`language` and `timestamps` can also be passed per call: `reader.load_data(ytlinks=[...], timestamps=True)`.

### Build an index

```python
from llama_index.core import VectorStoreIndex
from llama_index.readers.getyoutubetranscript import GetYouTubeTranscriptReader

documents = GetYouTubeTranscriptReader().load_data(ytlinks=["https://youtu.be/5e37ZT3SQbk"])
index = VectorStoreIndex.from_documents(documents)
print(index.as_query_engine().query("What does the speaker say about education?"))
```

## Pricing

Each transcript uses one credit from your GetYouTubeTranscript account. Failed requests are not charged.

## Links

- [GetYouTubeTranscript API docs](https://getyoutubetranscript.com/docs)
- [Python SDK](https://pypi.org/project/getyoutubetranscript/) (this package is built on it)
- [License: MIT](LICENSE)
