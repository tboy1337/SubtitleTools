# Translation Services

SubtitleTools supports two Google translation modes:

## `google` (default, web interface)

- Uses the unofficial Google Translate web API
- Requires a JavaScript runtime for `pyexecjs` (install [Node.js](https://nodejs.org/))
- Subject to rate limits and may break without notice
- Suitable for light personal use

```bash
subtitletools translate input.srt output.srt --service google
```

## `google_cloud` (production recommended)

- Uses the official [Google Cloud Translation API](https://cloud.google.com/translate)
- Requires `--api-key` with a valid API key
- More stable for repeated or automated use

```bash
subtitletools translate input.srt output.srt \
  --service google_cloud \
  --api-key YOUR_API_KEY
```

API keys are sent via the `X-Goog-Api-Key` header, not in the URL.

## Rate limiting

The web translator retries with exponential backoff on HTTP 429 responses. For large batches, prefer `google_cloud`.
