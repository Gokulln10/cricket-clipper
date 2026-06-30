# 🏏 Cricket Highlight Clipper

Takes a full cricket video and automatically exports highlight clips. It finds
exciting moments by combining three signals:

1. **Crowd-noise peaks** — boundaries and wickets trigger a sustained volume
   spike in the crowd/commentary. This is the primary detector.
2. **Scene-cut snapping** — clip starts are aligned to the broadcast's natural
   camera cuts (via [PySceneDetect](https://www.scenedetect.com/)) so they begin
   cleanly.
3. **Motion / event scoring** — sampled-frame motion intensity is used to rank
   clips, so genuinely action-packed moments float to the top. This is a
   pluggable stand-in for a trained ML event model (see *Extending* below).

## Requirements

- **ffmpeg** and **ffprobe** on your PATH
  - macOS: `brew install ffmpeg`
- Python 3.9+

## Setup

```bash
cd cricket-clipper
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Web app

```bash
streamlit run app.py
```

Then in the browser:
1. Paste a local video path (or upload a file).
2. Adjust **Sensitivity** and padding in the sidebar.
3. Click **Process video** — clips are exported and previewed with download buttons.

## Command line

```bash
python cli.py /path/to/match.mp4 -o clips/ --sensitivity 1.4 --max-clips 12
```

Useful flags:
- `--sensitivity` — lower catches more (quieter) moments. Default `1.6`.
- `--pre` / `--post` — seconds of lead-in / tail around each moment.
- `--max-clips N` — keep only the top N by score.
- `--no-scene-snap`, `--no-motion` — disable those stages.
- `--fast` — stream-copy cuts (much faster, but cuts land on keyframes).

## How it works

```
video ──┬─► extract audio ─► RMS energy ─► robust z-score peaks ─┐
        │                                                        ├─► candidate clips
        ├─► scene cuts (PySceneDetect) ─► snap clip starts ──────┘
        │
        └─► motion scoring (frame diff) ─► rank & keep top N ─► ffmpeg export
```

Tuning tips:
- Too few clips? Lower `--sensitivity` (e.g. `1.2`).
- Clips cut off the shot? Increase `--pre`.
- Lots of near-duplicates? Increase `merge_gap` (Advanced in the app).

## Extending: plug in a real ML model

`clipper/event_score.py` exposes `score_segment(video_path, start, end) -> float`
(0..1). Replace its body with a call to a trained classifier (e.g. a CNN/temporal
model over sampled frames, or an audio-event classifier) to detect specific
events like **wickets** vs **boundaries**. The pipeline already blends this score
with the crowd-noise score for ranking.

## Project layout

```
cricket-clipper/
├── app.py                  # Streamlit web app
├── cli.py                  # Command-line interface
├── requirements.txt
└── clipper/
    ├── config.py           # Settings dataclass
    ├── ffmpeg_utils.py     # ffmpeg/ffprobe helpers
    ├── audio_peaks.py      # crowd-noise detection
    ├── scene_detect.py     # scene-cut detection + snapping
    ├── event_score.py      # motion / event scoring (ML hook)
    ├── export.py           # ffmpeg clip export
    └── pipeline.py         # orchestration: signals -> ranked clips
```
