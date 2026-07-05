# Translation Services

SubtitleTools supports two Google translation modes.

## `google` (default)

Uses the Google Translate web interface via `pyexecjs`.

- Requires a JavaScript runtime (typically [Node.js](https://nodejs.org/)) when no API key is set
- Default for all translate/workflow commands

```bash
subtitletools translate input.srt output.srt --service google
```

## `google_cloud`

Uses the [Google Cloud Translation API](https://cloud.google.com/translate).

- Requires `--api-key` or `SUBTITLETOOLS_GOOGLE_API_KEY`
- Same HTTP client; API key is sent via the `X-Goog-Api-Key` header

```bash
subtitletools translate input.srt output.srt \
  --service google_cloud \
  --api-key YOUR_API_KEY
```

Or:

```bash
set SUBTITLETOOLS_GOOGLE_API_KEY=YOUR_API_KEY
subtitletools translate input.srt output.srt --service google_cloud
```

## Rate limiting

The web translator retries with exponential backoff on HTTP 429 responses. Large batches may take longer on the `google` service.
