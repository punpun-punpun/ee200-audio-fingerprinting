# EE200: Audio Fingerprinting — Q3B App

A Shazam-style song identifier built on the spectrogram-peak-hashing pipeline from Q3A.

## What's shipped with this app

- `app.py` — the Streamlit app (Library / Identify / Batch tabs)
- `app_data.pkl` — the **precomputed** fingerprint database, generated once from the 50-song library:
  - `hash_db`: hash key → list of (song name, anchor time) — used for matching
  - `song_peaks`: song name → list of (frequency, time) — used for the "where in the song" visualization and the Library tab thumbnails
  - `song_hash_counts`: song name → number of hashes — shown in the Library tab

The original audio files are **not** shipped with the app — only the derived fingerprint data, which is what the app actually needs at runtime. The only audio decoding that happens live is on whatever clip a user uploads to test.

## Confidence threshold (for "none" predictions)

A match is only reported as confident if:
1. The winning offset has at least `CONFIDENCE_MIN_VOTES = 5` votes, **and**
2. The winner beats the runner-up song by at least `CONFIDENCE_MARGIN = 2.0`×

Both constants are defined near the top of `app.py` and can be tuned. Anything not meeting this bar is reported as `none` in batch mode.

## Regenerating app_data.pkl

If the song library changes, rebuild `app_data.pkl` from a freshly built `hash_db` (see the Q3A notebook for the indexing pipeline), then re-run the reconstruction step that derives `song_peaks` and `song_hash_counts` from it.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deployed on Streamlit Community Cloud. `packages.txt` installs `ffmpeg` at the system level, which `librosa`/`audioread` need to decode MP3/M4A files (WAV/FLAC/OGG work without it, but the song library and most test clips are MP3).
