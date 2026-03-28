# video-transcription

High-accuracy transcription workflow for long Russian audio/video files using Whisper `large-v3`.

## What is in this repository

- `transcribe_whisper_large_v3.ipynb` - Jupyter notebook with:
  - short ASR theory,
  - dependency setup,
  - robust transcription pipeline,
  - medium post-processing pipeline.

## Key features

- Transcription with `faster-whisper` + `Whisper large-v3`.
- Anti-loop strategy for long files:
  - 20-minute chunk processing,
  - `condition_on_previous_text=False`,
  - temperature fallback,
  - repetition controls (`repetition_penalty`, `no_repeat_ngram_size`),
  - suspicious repetition checks per chunk.
- Output in two formats:
  - plain text (`*.txt`),
  - text with timestamps (`*_timestamps.txt`).
- Optional medium cleanup:
  - `*_clean.txt`,
  - `*_timestamps_clean.txt`.

## Quick start

1. Create and activate a Python virtual environment.
2. Open `transcribe_whisper_large_v3.ipynb`.
3. Run cells top-to-bottom.
4. Put source files into `audio/` (locally, not tracked by git).
5. Collect results from `transcripts/` (also local, ignored by git).

## Dependencies

Main libraries used in the notebook:

- `faster-whisper`
- `ctranslate2`
- `av`
- `tqdm`
- CUDA runtime wheels (for Linux GPU environments):
  - `nvidia-cublas-cu12`
  - `nvidia-cudnn-cu12`
  - `nvidia-cuda-runtime-cu12`

## Notes

- This repo intentionally ignores large/runtime artifacts (models, audio, transcripts, virtual environments).
- If you run on CPU, transcription is possible but much slower than GPU.
