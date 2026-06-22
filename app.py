import streamlit as st
import numpy as np
import pickle
import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter
from collections import defaultdict
import librosa

st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide")

# ============================================================
# Config (must match the settings used to build app_data.pkl)
# ============================================================
SR_TARGET = 22050
NPERSEG = 2048
NOVERLAP = 1024
NEIGHBORHOOD = 20
THRESHOLD_DB = -40
FAN_VALUE = 5
MAX_TIME_DELTA = 5.0

CONFIDENCE_MIN_VOTES = 5      # minimum votes to even consider a match real
CONFIDENCE_MARGIN = 2.0       # winner must beat runner-up by this factor

DATA_PATH = "app_data.pkl"


# ============================================================
# Load precomputed fingerprint data (cached so it loads once)
# ============================================================
@st.cache_resource
def load_data():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    return data["hash_db"], data["song_peaks"], data["song_hash_counts"]


hash_db, song_peaks, song_hash_counts = load_data()


# ============================================================
# Core fingerprinting pipeline (same logic used to build the DB)
# ============================================================
def compute_spectrogram(y, sr):
    f, t, Sxx = signal.spectrogram(y, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    return f, t, Sxx


def get_constellation(Sxx, f, t, neighborhood_size=NEIGHBORHOOD, threshold_db=THRESHOLD_DB):
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    local_max = maximum_filter(Sxx_db, size=neighborhood_size)
    peaks_mask = (Sxx_db == local_max) & (Sxx_db > threshold_db)
    freq_idxs, time_idxs = np.where(peaks_mask)
    return f[freq_idxs], t[time_idxs]


def make_hashes(peak_freqs, peak_times, fan_value=FAN_VALUE, max_time_delta=MAX_TIME_DELTA):
    idx_sorted = np.argsort(peak_times)
    freqs_sorted = peak_freqs[idx_sorted]
    times_sorted = peak_times[idx_sorted]

    hashes = []
    n = len(times_sorted)
    for i in range(n):
        f1, t1 = freqs_sorted[i], times_sorted[i]
        count = 0
        for j in range(i + 1, n):
            f2, t2 = freqs_sorted[j], times_sorted[j]
            dt = t2 - t1
            if dt <= 0:
                continue
            if dt > max_time_delta:
                break
            hash_key = (round(f1), round(f2), round(dt, 2))
            hashes.append((hash_key, t1))
            count += 1
            if count >= fan_value:
                break
    return hashes


def match_query(query_hashes, database):
    offset_counts = defaultdict(lambda: defaultdict(int))
    for hash_key, t_query in query_hashes:
        if hash_key in database:
            for song_name, t_db in database[hash_key]:
                offset = round(t_db - t_query, 1)
                offset_counts[song_name][offset] += 1

    best_song, best_offset, best_count = None, None, 0
    for song_name, offsets in offset_counts.items():
        for offset, count in offsets.items():
            if count > best_count:
                best_song, best_offset, best_count = song_name, offset, count
    return best_song, best_offset, best_count, offset_counts


def get_runner_up(offset_counts, best_song):
    best_per_song = {s: max(o.values()) for s, o in offset_counts.items()}
    others = {s: c for s, c in best_per_song.items() if s != best_song}
    return max(others.values()) if others else 0


def is_confident(best_count, runner_up):
    if best_count < CONFIDENCE_MIN_VOTES:
        return False
    if runner_up > 0 and best_count < CONFIDENCE_MARGIN * runner_up:
        return False
    return True


def load_audio_from_upload(uploaded_file, max_duration=60):
    """Save the uploaded file to a temp path first -- librosa/audioread
    need a real file path for formats like mp3/m4a, not just an in-memory buffer.
    Caps loaded duration since queries are meant to be short clips, not full songs --
    this also protects against memory spikes if a full song gets uploaded by mistake."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=SR_TARGET, mono=True, duration=max_duration)
    finally:
        os.remove(tmp_path)
    return y, sr


def identify_clip(y, sr):
    f, t, Sxx = compute_spectrogram(y, sr)
    peak_freqs, peak_times = get_constellation(Sxx, f, t)
    hashes = make_hashes(peak_freqs, peak_times)
    best_song, best_offset, best_count, offset_counts = match_query(hashes, hash_db)
    runner_up = get_runner_up(offset_counts, best_song) if best_song else 0
    confident = is_confident(best_count, runner_up) if best_song else False
    return {
        "f": f, "t": t, "Sxx": Sxx,
        "peak_freqs": peak_freqs, "peak_times": peak_times,
        "hashes": hashes,
        "best_song": best_song, "best_offset": best_offset, "best_count": best_count,
        "offset_counts": offset_counts, "runner_up": runner_up, "confident": confident,
    }


# ============================================================
# Plot helpers
# ============================================================
def plot_spectrogram(f, t, Sxx, title=""):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud", cmap="magma")
    ax.set_ylim(0, 4000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_constellation(f, t, Sxx, peak_freqs, peak_times, title=""):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud", cmap="magma")
    ax.scatter(peak_times, peak_freqs, s=8, facecolors="none", edgecolors="cyan")
    ax.set_ylim(0, 4000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_song_alignment(song_name, query_offset, query_duration):
    peaks = song_peaks.get(song_name, [])
    if not peaks:
        return None
    freqs = [p[0] for p in peaks]
    times = [p[1] for p in peaks]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.scatter(times, freqs, s=4, color="#2a9d8f", alpha=0.6)
    if query_offset is not None:
        ax.axvspan(query_offset, query_offset + query_duration, color="orange", alpha=0.3,
                   label="query location")
        ax.legend(loc="upper right")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Where in the song? — {song_name}")
    fig.tight_layout()
    return fig


def plot_offset_histogram(offset_counts, best_song):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if best_song in offset_counts:
        offsets = sorted(offset_counts[best_song].keys())
        counts = [offset_counts[best_song][o] for o in offsets]
        ax.bar(offsets, counts, width=0.4, color="#e76f51")
    ax.set_xlabel("Time offset (database frame − query frame)")
    ax.set_ylabel("# hashes")
    ax.set_title("The alignment spike")
    fig.tight_layout()
    return fig


# ============================================================
# UI
# ============================================================
st.title("🎵 EE200: Audio Fingerprinting")
st.caption("Signals, Systems & Networks · Project Demo")
st.write("Index a library of songs as spectrogram fingerprints, then identify any short clip against it.")

tab1, tab2, tab3 = st.tabs(["📚 Library", "🔍 Identify", "📦 Batch"])

# ---------------- Library tab ----------------
with tab1:
    st.subheader("Indexed song library")
    st.caption("Song indexing is precomputed. Drop a clip in the Identify tab to test the library.")
    st.write(f"**{len(song_hash_counts)} songs indexed.**")

    cols = st.columns(4)
    for i, (song, count) in enumerate(sorted(song_hash_counts.items())):
        with cols[i % 4]:
            st.markdown(f"**{song}**")
            st.caption(f"{count:,} hashes")
            peaks = song_peaks.get(song, [])
            if peaks:
                freqs = [p[0] for p in peaks]
                times = [p[1] for p in peaks]
                fig, ax = plt.subplots(figsize=(2.2, 1.4))
                ax.scatter(times, freqs, s=1, color="#2a9d8f")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

# ---------------- Identify tab ----------------
with tab2:
    st.subheader("Identify a clip")
    uploaded = st.file_uploader(
        "Upload a query clip", type=["wav", "mp3", "flac", "ogg", "m4a"], key="single"
    )

    if uploaded is not None:
        with st.spinner("Identifying..."):
            y, sr = load_audio_from_upload(uploaded)
            result = identify_clip(y, sr)

        if result["best_song"] and result["confident"]:
            st.success(f"✅ Match found: **{result['best_song']}**")
        elif result["best_song"]:
            st.warning(f"Weak / unconfident match: {result['best_song']}")
        else:
            st.error("No match found")

        st.write(
            f"Winning offset votes: **{result['best_count']}**  |  "
            f"Runner-up votes: **{result['runner_up']}**"
        )

        st.markdown("#### Step 1 — Spectrogram & constellation")
        c1, c2 = st.columns(2)
        with c1:
            fig_a = plot_spectrogram(result["f"], result["t"], result["Sxx"], "Query spectrogram")
            st.pyplot(fig_a)
            plt.close(fig_a)
        with c2:
            fig_b = plot_constellation(
                result["f"], result["t"], result["Sxx"],
                result["peak_freqs"], result["peak_times"], "Constellation"
            )
            st.pyplot(fig_b)
            plt.close(fig_b)

        if result["best_song"]:
            st.markdown("#### Step 2 — Where in the song?")
            query_duration = result["t"][-1] if len(result["t"]) else 0
            fig2 = plot_song_alignment(result["best_song"], result["best_offset"], query_duration)
            if fig2:
                st.pyplot(fig2)
                plt.close(fig2)

            st.markdown("#### Step 3 — The alignment spike")
            fig3 = plot_offset_histogram(result["offset_counts"], result["best_song"])
            st.pyplot(fig3)
            plt.close(fig3)

# ---------------- Batch tab ----------------
with tab3:
    st.subheader("Identify many clips at once")
    st.caption(
        "Upload a set of query clips. Each is identified against the indexed library, "
        "and results are written to results.csv with columns: filename, prediction. "
        "prediction is 'none' when no candidate clears the confidence threshold."
    )
    uploaded_batch = st.file_uploader(
        "Upload clips",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True,
        key="batch",
    )

    if uploaded_batch and st.button("Run batch"):
        rows = []
        progress = st.progress(0)
        status = st.empty()
        for i, f in enumerate(uploaded_batch):
            status.text(f"Identifying... {i + 1}/{len(uploaded_batch)}")
            y, sr = load_audio_from_upload(f)
            result = identify_clip(y, sr)
            prediction = (
                result["best_song"] if (result["best_song"] and result["confident"]) else "none"
            )
            rows.append({"filename": f.name, "prediction": prediction})
            progress.progress((i + 1) / len(uploaded_batch))
        status.text("Done.")

        df = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.dataframe(df)
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("Download results.csv", csv_bytes, file_name="results.csv", mime="text/csv")
