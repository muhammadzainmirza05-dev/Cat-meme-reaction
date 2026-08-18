# 🐱 Cat Meme Reactor

Webcam feed on the left, a matching cat meme on the right — switches live
based on your facial expression **or** hand gesture.

## Setup

Use Python 3.10 or 3.11. The application uses MediaPipe 0.10.11, which is
incompatible with Protobuf 4+.

If `venv` already exists, remove or rename it before recreating it so that pip
does not keep the incompatible packages.

```bash
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r Requirements.txt
```

## Add your cat images
Drop images into `assets/cats/` with these exact base filenames
(`.jpg` or `.png`):

| Label        | Filename         | Trigger                          |
|--------------|------------------|-----------------------------------|
| happy        | happy.jpg        | wide mouth / smile               |
| surprised    | surprised.jpg     | open mouth + raised eyebrows     |
| angry        | angry.jpg         | lowered/furrowed eyebrows        |
| neutral      | neutral.jpg       | default / no strong signal       |
| shush        | shush.jpg         | index finger near mouth 🤫       |
| peace        | peace.jpg         | index + middle finger up ✌️      |
| thumbs_up    | thumbs_up.jpg     | thumb up, rest closed 👍         |

If a file is missing, it falls back to `neutral.jpg` so make sure that
one exists at minimum.

## Run
```bash
streamlit run app.py
```

Allow camera access in your browser, then select the camera and press **Start**.
The supplied WebRTC settings target local use. If you deploy the app to a remote
server, configure a reachable STUN/TURN service in `app.py`.

## Tuning
- Expression thresholds are in `expression_detector.py` (mouth_open,
  mouth_width, brow_raise) — adjust if your lighting/camera gives
  different baseline ratios.
- Gesture rules are in `gesture_detector.py` — pure finger-up/down logic,
  easy to add new gestures (e.g. rock-on, OK sign).
- `STABILITY_FRAMES` in `main.py` controls how many consecutive frames
  a label must hold before the cat image switches (reduces flicker).

## Ideas to extend for your FYP
- Swap rule-based expression detection for a trained CNN on FER2013
- Add sound effects per cat reaction
- Save/export a "cat reaction reel" video of a session
- Turn it into a Snapchat-style filter using face landmark warping
