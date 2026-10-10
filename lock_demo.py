#!/usr/bin/env python3
"""
ASL Lock Demo — single-file webcam sign-language lock.
Requirements: torch, torchvision, opencv-python, pillow, numpy
Usage live: python lock_demo.py [--camera 0]
Usage test: python lock_demo.py --image my_image.jpg
"""
import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
THRESHOLD = 0.90

LABEL_ORDER = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]

# Safe-word registry (exact, case-insensitive)
SAFEWORDS = {
    "BRAVO":  ("AG-001", "Falcon"),
    "CIPHER": ("AG-002", "Nightjar"),
    "DELTA":  ("AG-003", "Wraith"),
    "EMBER":  ("AG-004", "Kestrel"),
    "QUIET":  ("AG-005", "Lynx"),
}

# ------------------------------------------------------------------
# Model setup (must match training architecture)
# ------------------------------------------------------------------
def load_model(device):
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 29)
    weights_path = "best_model.pt"
    if not os.path.isfile(weights_path):
        print(f"ERROR: {weights_path} not found in {os.getcwd()}")
        exit(1)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# ------------------------------------------------------------------
# Preprocessing (must match training)
# ------------------------------------------------------------------
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def preprocess_frame(bgr_img):
    # BGR -> RGB
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    # Resize to 224x224 (PIL-style resize, nearest is fine with cv2)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    # HWC -> CHW float [0,1]
    arr = rgb.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    # Normalize
    arr = (arr - np.array(MEAN)[:, None, None]) / np.array(STD)[:, None, None]
    tensor = torch.from_numpy(arr).unsqueeze(0).float()
    return tensor

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
LOG_FILE = "entry_log.csv"

def init_log():
    if not os.path.isfile(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "outcome", "agent_id", "codename",
                "word_read", "reason"
            ])

def log_event(outcome, agent_id="-", codename="-", word_read="", reason=""):
    init_log()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [now, outcome, agent_id, codename, word_read, reason]
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    # Print to terminal as well
    print(f"[{now}] {outcome} | agent={agent_id} | code={codename} | word={word_read} | reason={reason}")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ASL Lock Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--image", type=str, default=None, help="Run on a single image file instead of camera")
    args = parser.parse_args()

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(device)

    # Image-only mode
    if args.image:
        if not os.path.isfile(args.image):
            print(f"ERROR: image file not found: {args.image}")
            exit(1)
        img = cv2.imread(args.image)
        if img is None:
            print(f"ERROR: could not read image: {args.image}")
            exit(1)
        crop = crop_to_square(img)
        tensor = preprocess_frame(crop).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1).squeeze()
        top3 = torch.topk(probs.cpu(), 3)
        for idx in top3.indices.tolist():
            label = LABEL_ORDER[idx]
            conf = probs.cpu()[idx].item()
            print(f"  {label}  {conf:.4f}")
        return

    # Camera mode ---------------------------------------------------
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(
            "ERROR: Could not open camera.\n"
            "  - Check that the camera index is correct.\n"
            "  - On macOS: System Settings > Privacy & Security > Camera\n"
            "    and ensure this terminal/app is allowed."
        )
        exit(1)

    # Warm-up: discard ~30 frames
    for _ in range(30):
        cap.read()
        time.sleep(0.05)

    # State variables
    word = []
    flip_input = False
    banner_text = ""
    banner_color = (255, 255, 255)
    banner_timer = 0.0
    cooldown_timer = 0.0
    log_entries = []  # last 5 log lines for display

    # Read log for display
    def refresh_logs():
        nonlocal log_entries
        try:
            with open(LOG_FILE, newline="") as f:
                lines = f.read().splitlines()
            log_entries = lines[-5:] if len(lines) > 1 else [lines[-1]] if lines else ["(no entries)"]
        except Exception:
            log_entries = ["(log not found)"]

    refresh_logs()

    # Helper to crop frame to centre square
    def crop_to_square(frame):
        h, w = frame.shape[:2]
        size = min(h, w)
        x = (w - size) // 2
        y = (h - size) // 2
        return frame[y:y+size, x:x+size]

    # Helper to show banner with timeout (~3 sec)
    def set_banner(text, color):
        nonlocal banner_text, banner_color, banner_timer
        banner_text = text
        banner_color = color
        banner_timer = time.time()

    # Helper to handle a capture result
    def handle_capture(label, conf):
        nonlocal word, cooldown_timer
        cooldown_timer = time.time()

        # Nothing
        if label == "nothing":
            # Ignore; maybe show a small note
            return

        # Below threshold -> RETRY
        if conf < THRESHOLD:
            word.clear()
            reason = "confidence below threshold"
            set_banner("RETRY: not sure, sign again", (0, 180, 255))  # amber-ish
            log_event("RETRY", "-", "-", "", reason)
            refresh_logs()
            return

        # DEL -> delete last
        if label == "del":
            if word:
                word.pop()
            return

        # SPACE -> ignore
        if label == "space":
            return

        # Normal letter
        word.append(label)
        return

    # Helper to submit word
    def submit_word():
        nonlocal word, banner_text, banner_color, banner_timer
        joined = "".join(word)
        # Case-insensitive match
        key = joined.upper()
        if key in SAFEWORDS:
            agent_id, codename = SAFEWORDS[key]
            set_banner(f"ACCESS GRANTED: {agent_id} {codename}", (0, 255, 0))
            log_event("GRANTED", agent_id, codename, joined, "exact match")
        else:
            agent_id, codename = "-", "-"
            set_banner("ACCESS DENIED", (0, 0, 255))
            reason = "no match" if joined else "empty word"
            log_event("DENIED", agent_id, codename, joined, reason)
        word.clear()
        refresh_logs()

    # Main loop ----------------------------------------------------
    print("Starting ASL Lock demo. Controls:")
    print("  SPACE = capture sign | ENTER = submit word")
    print("  BACKSPACE / r = clear attempt | f = flip model input")
    print("  q / ESC = quit")
    print("Keep your hand inside the green box.")

    while True:
        ret, frame = None, None
        for attempt in range(30):
            ret, frame = cap.read()
            if ret:
                break
            time.sleep(0.02)
        if not ret:
            print("ERROR: Failed to read frame from camera.")
            break

        # Mirror display only
        display = cv2.flip(frame, 1)
        h, w = display.shape[:2]

        # Square crop from original (unmirrored for model by default)
        model_feed = frame if not flip_input else cv2.flip(frame, 1)
        crop = crop_to_square(model_feed)

        # Draw guide box on mirrored display
        guide_size = min(h, w) // 2
        gx = (w - guide_size) // 2
        gy = (h - guide_size) // 2
        cv2.rectangle(display, (gx, gy), (gx + guide_size, gy + guide_size), (0, 255, 0), 2)

        # Model prediction
        tensor = preprocess_frame(crop).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1).squeeze()
        conf, pred_idx = torch.max(probs, dim=0)
        conf = conf.cpu()
        pred_idx = pred_idx.cpu()
        label = LABEL_ORDER[int(pred_idx)]
        confidence = float(conf)

        # Handle input with cooldown (after each capture)
        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        # Flip toggle (always available)
        if key == ord("f"):
            flip_input = not flip_input

        # Capture only after cooldown
        if now - cooldown_timer > 0.5:
            if key == ord(" "):
                handle_capture(label, confidence)

        # Submit, clear, quit (always available)
        if key == 13:  # ENTER
            submit_word()
        if key == 8 or key == ord("r") or key == ord("R"):
            word.clear()
            banner_text = ""
        if key == ord("q") or key == 27:
            break

        # Update banner timeout (~3 sec)
        if banner_text and (now - banner_timer > 3.0):
            banner_text = ""

        # Refresh logs occasionally
        if int(now * 2) % 5 == 0:
            refresh_logs()

        # Overlays -------------------------------------------------
        # Prediction
        pred_text = f"{label} {confidence:.2f}"
        cv2.putText(display, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Flip state
        flip_text = f"Flip model input: {'ON' if flip_input else 'OFF'}"
        cv2.putText(display, flip_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Word typed so far (large)
        word_str = "".join(word)
        cv2.putText(display, word_str, (gx, gy + guide_size + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        # Guide text
        guide_text = "Keep hand inside box  |  SPACE=capture  ENTER=submit  r=clear  f=flip  q=quit"
        cv2.putText(display, guide_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Banner (status)
        if banner_text:
            # Draw a filled rectangle banner
            text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            tw, th = text_size[0]
            bx = (w - tw) // 2
            by = 100
            cv2.rectangle(display, (bx - 10, by - 30), (bx + tw + 10, by + th + 10), banner_color, -1)
            cv2.putText(display, banner_text, (bx, by + th // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Last 5 log entries (bottom-left)
        log_start_y = h - 130
        for i, line in enumerate(log_entries[-5:]):
            # Truncate very long lines for readability
            short = line[:80]
            color = (180, 180, 180)
            if "GRANTED" in line:
                color = (100, 255, 100)
            elif "DENIED" in line:
                color = (100, 100, 255)
            elif "RETRY" in line:
                color = (100, 100, 255)  # similar to denied but distinct if needed
            cv2.putText(display, short, (10, log_start_y + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow("ASL Lock Demo", display)

    cap.release()
    cv2.destroyAllWindows()
    print("Exiting.")

if __name__ == "__main__":
    main()
