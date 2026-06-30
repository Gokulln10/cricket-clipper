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

## Tune detection on a real match

Before doing full runs, use `tune.py` to see *what each detector fires on* and
to find the right thresholds — it analyses the audio only (no clip export), so
it's fast to iterate.

List every candidate with which detectors fired (level/onset/swell) and a CSV
you can scrub through:

```bash
python tune.py match.mp4 --csv detections.csv
```

Sweep sensitivity to watch the candidate count change:

```bash
python tune.py match.mp4 --sweep 1.0 1.2 1.5 1.8 2.2
```

Measure real recall/precision against known event times. Put one timestamp per
line (`MM:SS`, `H:MM:SS` or raw seconds) in a file, then:

```bash
python tune.py match.mp4 --truth events.txt --sweep 1.0 1.2 1.4 1.6 1.8
```

It reports, per sensitivity, how many real events were caught (recall) and how
many candidates matched no event (precision), so you can pick the lowest
sensitivity that still keeps precision acceptable. Other levers: `--onset-sensitivity`,
`--swell-sensitivity`, `--local-baseline`, `--no-onset`, `--no-swell`, `--tol`.

## Extending: plug in a real ML model

`clipper/event_score.py` exposes `score_segment(video_path, start, end) -> float`
(0..1). Replace its body with a call to a trained classifier (e.g. a CNN/temporal
model over sampled frames, or an audio-event classifier) to detect specific
events like **wickets** vs **boundaries**. The pipeline already blends this score
with the crowd-noise score for ranking.

## Train an audio-event classifier (most reliable boundary/six detection)

Heuristics can miss subtle events (e.g. straight-driven boundaries). For the
highest accuracy, train a small CNN on the audio of labelled clips. It plugs in
as an extra stage that labels each clip (`four`/`six`/`wicket`/…) and boosts
confident events in the ranking.

Install the optional ML deps first:

```bash
pip install -r requirements-ml.txt
```

**1. Build a labelled set.** Either organise clips into per-class folders:

```
data/
  four/    *.mp4|*.wav
  six/     ...
  wicket/  ...
  negative/  ...        # non-events (recommended for precision)
```

…or bootstrap a CSV from the clipper's own detections and label the `label`
column:

```bash
python -m training.make_labels match.mp4 --out data/match1 --clip-seconds 3
# edit data/match1/match1_labels.csv -> fill in the 'label' column
```

**2. Train:**

```bash
python -m training.train --data data/ --epochs 40 --out models/event_clf.pt
# or:  --data data/match1/match1_labels.csv
```

The checkpoint stores the labels and feature geometry, so it is self-describing.

**3. Use it** in detection (boosts + labels confident events):

```bash
python cli.py match.mp4 --classifier models/event_clf.pt --classifier-min-prob 0.6
```

In the web app, paste the model path into **"Audio-event model"** under
*Advanced*. Each clip's detected event + probability appears in its tags.

**4. Evaluate** on a held-out labelled set (confusion matrix + per-class
precision/recall, plus a focused boundary/six recall):

```bash
python -m training.eval --model models/event_clf.pt --data data/val/
# or:  --data data/val.csv   ·   add --min-prob 0.6 to score at a threshold
```

`recall` on the positive events tells you what fraction of real
boundaries/wickets the model catches; `precision` tells you how many of its
flagged events are correct.

## Project layout

```
cricket-clipper/
├── app.py                  # Streamlit web app
├── cli.py                  # Command-line interface
├── tune.py                 # detection tuning / diagnostics (no clip export)
├── requirements.txt        # core deps
├── requirements-ml.txt     # optional: YOLO + classifier (PyTorch, librosa)
├── clipper/
│   ├── config.py           # Settings dataclass
│   ├── ffmpeg_utils.py     # ffmpeg/ffprobe helpers + HW-encoder detection
│   ├── audio_peaks.py      # crowd-noise detection (level/onset/swell)
│   ├── scene_detect.py     # scene-cut detection + replay clusters
│   ├── event_score.py      # motion scoring
│   ├── player_detect.py    # optional YOLO player/ball detection
│   ├── classifier.py       # optional trained audio-event classifier hook
│   ├── export.py           # ffmpeg clip export
│   └── pipeline.py         # orchestration: signals -> ranked clips
└── training/
    ├── features.py         # waveform -> log-mel spectrogram
    ├── dataset.py          # folder/CSV loaders
    ├── model.py            # small CNN
    ├── train.py            # training loop -> checkpoint
    ├── infer.py            # inference (EventClassifier)
    ├── eval.py             # held-out evaluation (confusion matrix, P/R/F1)
    └── make_labels.py      # bootstrap a labelling CSV from detections
```