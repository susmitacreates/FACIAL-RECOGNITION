# ═══════════════════════════════════════════════════════════════════════════════
#  client.py  ·  J.A.R.V.I.S. Face Recognition Client  ·  LAPTOP SIDE
#  Camera → CloudFlare Tunnel → Server → Processed Frame + Data → Jarvis UI
# ═══════════════════════════════════════════════════════════════════════════════

import cv2
import numpy as np
import requests
import base64
import time
import threading
import json
from datetime import datetime

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SERVER_URL  = "https://your-tunnel-name.trycloudflare.com"   # ← CHANGE THIS
CAM_INDEX   = 0
FRAME_W     = 640
FRAME_H     = 480
SEND_EVERY  = 3          # send every N frames (saves bandwidth)
TIMEOUT     = 10         # HTTP timeout (seconds)
UI_W        = 1280
UI_H        = 720

# ─── PALETTE (BGR) ─────────────────────────────────────────────────────────────
C_CYAN        = (255, 210,  0)
C_CYAN_DIM    = (120, 105,  0)
C_CYAN_GLOW   = (255, 255,  0)
C_GREEN       = (  0, 230, 80)
C_GREEN_DIM   = (  0, 120, 40)
C_RED         = (  0,  60,255)
C_RED_DIM     = (  0,  30,130)
C_YELLOW      = (  0,220,255)
C_ORANGE      = (  0,160,255)
C_MAGENTA     = (200,  0,255)
C_WHITE       = (255,255,255)
C_GRAY        = ( 80, 80, 80)
C_GRAY_DARK   = ( 30, 30, 30)
C_BG          = (  8,  8, 14)
C_BG_PANEL    = ( 12, 12, 20)
C_BORDER      = ( 40, 35,  5)
C_BG_SCAN     = ( 20, 25, 35)

# ─── STATE ─────────────────────────────────────────────────────────────────────
class State:
    connected      = False
    last_frame     = None
    faces          = []
    fps            = 0.0
    frame_count    = 0
    last_send      = 0
    scan_pos       = 0.0
    pulse_phase    = 0.0
    hex_phase      = 0.0
    boot_time      = time.time()
    history        = []          # list of (time, name, conf)
    current_target = ("STANDBY", 0.0)
    target_flash   = 0.0
    error_msg      = ""
    error_time     = 0
    accel          = "?"
    total_photos   = 0
    total_people   = 0

S = State()

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def bgr(*v):
    return (int(v[0]), int(v[1]), int(v[2]))

def lerp_color(c1, c2, t):
    return (int(c1[0]+(c2[0]-c1[0])*t),
            int(c1[1]+(c2[1]-c1[1])*t),
            int(c1[2]+(c2[2]-c1[2])*t))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def draw_glow_rect(frame, x1, y1, x2, y2, color, glow=6, alpha=0.15):
    for i in range(glow, 0, -1):
        a = alpha * (i / glow)
        ov = frame[y1-i:y2+i+1, x1-i:x2+i+1].copy()
        cv2.rectangle(ov, (0,0), (ov.shape[1]-1, ov.shape[0]-1), color, cv2.FILLED)
        cv2.addWeighted(ov, a, frame[y1-i:y2+i+1, x1-i:x2+i+1], 1-a, 0,
                        frame[y1-i:y2+i+1, x1-i:x2+i+1])

def draw_hex_grid(frame, w, h, phase, spacing=40, size=4, alpha=0.08):
    pts = []
    for row in range(0, h+spacing, spacing):
        offset = spacing//2 if (row // spacing) % 2 else 0
        for col in range(0, w+spacing, spacing):
            x = col + offset
            y = row
            if 0 <= x < w and 0 <= y < h:
                pts.append((x, y))
    for (x, y) in pts:
        d = np.hypot(x - w//2, y - h//2)
        pulse = 0.5 + 0.5 * np.sin(phase - d * 0.02)
        a = alpha * pulse
        sz = max(1, int(size * pulse))
        cv2.circle(frame, (x, y), sz, C_CYAN, -1)
        cv2.addWeighted(frame[y-sz:y+sz+1, x-sz:x+sz+1].copy(),
                        a,
                        frame[y-sz:y+sz+1, x-sz:x+sz+1], 1-a, 0,
                        frame[y-sz:y+sz+1, x-sz:x+sz+1])

def draw_arc(frame, cx, cy, r, start, end, color, thick=2):
    cv2.ellipse(frame, (cx, cy), (r, r), 0, start, end, color, thick)

def draw_pulse_ring(frame, cx, cy, base_r, phase, color, max_r=80):
    for i in range(3):
        p = (phase + i * 0.33) % 1.0
        r = int(base_r + p * max_r)
        a = 1.0 - p
        c = lerp_color(color, C_BG, p * 0.6)
        cv2.circle(frame, (cx, cy), r, c, 1)
        cv2.addWeighted(frame[cy-r:cy+r+1, cx-r:cx+r+1].copy(),
                        a*0.3,
                        frame[cy-r:cy+r+1, cx-r:cx+r+1], 1-a*0.3, 0,
                        frame[cy-r:cy+r+1, cx-r:cx+r+1])

def draw_radial_text(frame, cx, cy, radius, text, color, font_scale=0.45):
    chars = list(text)
    total = len(chars)
    for i, ch in enumerate(chars):
        angle = (i / total) * 2 * np.pi - np.pi/2
        x = cx + int(radius * np.cos(angle))
        y = cy + int(radius * np.sin(angle))
        cv2.putText(frame, ch, (x-4, y+4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 1, cv2.LINE_AA)

def draw_scan_sweep(frame, x1, y1, x2, y2, phase, color):
    h = y2 - y1
    if h <= 0: return
    sweep_y = y1 + int(phase * h)
    cv2.line(frame, (x1, sweep_y), (x2, sweep_y), color, 1)
    for i in range(2, 8):
        a = 0.15 / i
        cv2.line(frame, (x1, sweep_y+i), (x2, sweep_y+i),
                 lerp_color(color, C_BG, i/8), 1)

def draw_corner_brackets(frame, x1, y1, x2, y2, color, thick=2, ln=18):
    cv2.line(frame, (x1,y1), (x1+ln,y1), color, thick)
    cv2.line(frame, (x1,y1), (x1,y1+ln), color, thick)
    cv2.line(frame, (x2,y1), (x2-ln,y1), color, thick)
    cv2.line(frame, (x2,y1), (x2,y1+ln), color, thick)
    cv2.line(frame, (x1,y2), (x1+ln,y2), color, thick)
    cv2.line(frame, (x1,y2), (x1,y2-ln), color, thick)
    cv2.line(frame, (x2,y2), (x2-ln,y2), color, thick)
    cv2.line(frame, (x2,y2), (x2,y2-ln), color, thick)

def draw_panel_frame(frame, x1, y1, x2, y2, color, title=None):
    cv2.rectangle(frame, (x1,y1), (x2,y2), C_BG_PANEL, cv2.FILLED)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
    draw_corner_brackets(frame, x1, y1, x2, y2, color, 2, 12)
    if title:
        cv2.putText(frame, title, (x1+10, y1+18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)
        cv2.line(frame, (x1+10, y1+24), (x2-10, y1+24), color, 1)

def draw_progress_bar(frame, x, y, w, h, pct, color, bg=(20,20,30)):
    cv2.rectangle(frame, (x, y), (x+w, y+h), bg, cv2.FILLED)
    cv2.rectangle(frame, (x, y), (x+w, y+h), C_GRAY_DARK, 1)
    fill = int(w * clamp(pct, 0, 1))
    cv2.rectangle(frame, (x+1, y+1), (x+fill, y+h-1), color, cv2.FILLED)
    cv2.putText(frame, f"{pct*100:.0f}%", (x+w+6, y+h-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

def draw_bars(frame, x, y, w, count, active, color, gap=3, h=3):
    for i in range(count):
        bx = x + i * (w + gap)
        if i < active:
            cv2.rectangle(frame, (bx, y), (bx+w, y+h), color, cv2.FILLED)
        else:
            cv2.rectangle(frame, (bx, y), (bx+w, y+h), C_GRAY_DARK, 1)

# ─── BOOT SEQUENCE ─────────────────────────────────────────────────────────────

def boot_screen():
    W, H = 800, 500
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.putText(frame, "J.A.R.V.I.S.", (W//2-120, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, C_CYAN, 3, cv2.LINE_AA)
    cv2.putText(frame, "JUST A RATHER VERY INTELLIGENT SYSTEM", (W//2-230, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN_DIM, 1, cv2.LINE_AA)

    lines = [
        "▸ Initializing camera interface...",
        "▸ Establishing secure tunnel...",
        "▸ Loading recognition modules...",
        "▸ Calibrating neural network...",
        "▸ System ready.",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (W//2-180, 250+i*32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_CYAN_DIM, 1, cv2.LINE_AA)
        for f in range(15):
            frame2 = frame.copy()
            cv2.rectangle(frame2, (W//2-200, 230+i*32), (W//2+200, 258+i*32),
                          C_CYAN_DIM, 1)
            cv2.imshow("J.A.R.V.I.S. — Booting", frame2)
            cv2.waitKey(80)

    print("  [J.A.R.V.I.S.] Boot complete. Starting main interface.\n")
    cv2.destroyAllWindows()

# ─── CONNECTION CHECK ──────────────────────────────────────────────────────────

def check_health():
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        d = r.json()
        S.connected      = True
        S.accel          = d.get("accelerator", "?")
        S.total_people   = d.get("total_people", 0)
        S.total_photos   = d.get("total_photos", 0)
        S.error_msg      = ""
        return True
    except Exception as e:
        S.connected  = False
        S.error_msg  = str(e)[:60]
        S.error_time = time.time()
        return False

def send_frame(img_bgr):
    try:
        _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        payload = {'frame': base64.b64encode(buf).decode('utf-8')}
        r = requests.post(f"{SERVER_URL}/process", json=payload, timeout=TIMEOUT)
        d = r.json()
        raw = base64.b64decode(d['frame'])
        S.last_frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        S.faces      = d.get('faces', [])
        S.connected  = True
        S.error_msg  = ""
        return True
    except Exception as e:
        S.connected  = False
        S.error_msg  = str(e)[:60]
        S.error_time = time.time()
        return False

# ─── SEND WORKER (background thread) ───────────────────────────────────────────

_send_img    = None
_send_lock   = threading.Lock()

def send_worker():
    while True:
        with _send_lock:
            if _send_img is not None and S.frame_count - S.last_send >= SEND_EVERY:
                S.last_send = S.frame_count
                send_frame(_send_img.copy())
        time.sleep(0.03)

# ─── MAIN UI RENDER ────────────────────────────────────────────────────────────

def render_ui(cam_frame):
    global _send_img
    with _send_lock:
        _send_img = cam_frame

    W, H = UI_W, UI_H
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Background
    canvas[:] = C_BG

    # Subtle hex grid (animated)
    S.hex_phase += 0.025
    draw_hex_grid(canvas, W, H, S.hex_phase, spacing=48, size=2, alpha=0.04)

    # ── LAYOUT ──────────────────────────────────────────────────────────────
    top_h     = 56
    bot_h     = 36
    left_w    = 260
    right_w   = 300
    pad       = 10
    vid_x1    = left_w + pad
    vid_y1    = top_h + pad
    vid_x2    = W - right_w - pad
    vid_y2    = H - bot_h - pad
    vid_w     = vid_x2 - vid_x1
    vid_h     = vid_y2 - vid_y1

    # ── TOP BAR ─────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0,0), (W, top_h), C_BG_PANEL)
    cv2.line(canvas, (0, top_h), (W, top_h), C_CYAN_DIM, 1)

    cv2.putText(canvas, "J.A.R.V.I.S.", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_CYAN, 2, cv2.LINE_AA)
    cv2.putText(canvas, "FACE RECOGNITION MODULE", (160, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_CYAN_DIM, 1, cv2.LINE_AA)

    # Connection indicator
    conn_color = C_GREEN if S.connected else C_RED
    cv2.circle(canvas, (W-120, 28), 6, conn_color, -1)
    if S.connected:
        S.pulse_phase += 0.05
        pr = int(8 + 4 * np.sin(S.pulse_phase))
        cv2.circle(canvas, (W-120, 28), pr, C_GREEN_DIM, 1)
    cv2.putText(canvas, "LINK" if S.connected else "OFFLINE",
                (W-108, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.38, conn_color, 1, cv2.LINE_AA)

    # Clock
    now_str = datetime.now().strftime("%H:%M:%S")
    cv2.putText(canvas, now_str, (W-230, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA)

    # FPS
    cv2.putText(canvas, f"{S.fps:.0f} FPS", (W-290, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN_DIM, 1, cv2.LINE_AA)

    # ── LEFT PANEL — SYSTEM STATUS ─────────────────────────────────────────
    lx1, ly1, lx2, ly2 = pad, top_h+pad, left_w+pad, H-bot_h-pad
    draw_panel_frame(canvas, lx1, ly1, lx2, ly2, C_CYAN_DIM, "SYSTEM STATUS")

    ty = ly1 + 42
    line_h = 28
    # Accelerator
    cv2.putText(canvas, "ACCEL:", (lx1+12, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GRAY, 1)
    acc_c = C_GREEN if S.accel not in ("?", "CPU") else C_ORANGE
    cv2.putText(canvas, S.accel, (lx1+72, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, acc_c, 1)
    ty += line_h

    # Known faces
    cv2.putText(canvas, "FACES:", (lx1+12, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GRAY, 1)
    cv2.putText(canvas, str(S.total_photos), (lx1+72, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_WHITE, 1)
    ty += line_h
    cv2.putText(canvas, "PEOPLE:", (lx1+12, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GRAY, 1)
    cv2.putText(canvas, str(S.total_people), (lx1+72, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_WHITE, 1)
    ty += line_h + 8

    # Signal bars (fake, based on connection)
    cv2.putText(canvas, "SIGNAL:", (lx1+12, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GRAY, 1)
    bars = 5 if S.connected else 1
    draw_bars(canvas, lx1+72, ty-10, 10, 5, bars,
              C_GREEN if S.connected else C_RED, gap=3, h=12)
    ty += line_h + 8

    # Scan sweep animation in left panel
    sweep_phase = (time.time() * 0.4) % 1.0
    draw_scan_sweep(canvas, lx1+10, ty, lx2-10, ly2-10, sweep_phase, C_CYAN_DIM)

    # Boot time
    elapsed = int(time.time() - S.boot_time)
    mm, ss  = divmod(elapsed, 60)
    cv2.putText(canvas, f"UPTIME {mm:02d}:{ss:02d}", (lx1+12, ly2-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_CYAN_DIM, 1)

    # ── RIGHT PANEL — TARGET INFO ──────────────────────────────────────────
    rx1, ry1, rx2, ry2 = W-right_w-pad, top_h+pad, W-pad, H-bot_h-pad
    draw_panel_frame(canvas, rx1, ry1, rx2, ry2, C_CYAN_DIM, "TARGET ANALYSIS")

    # Current target (big)
    name, conf = S.current_target
    is_known   = name != "STANDBY" and name != "Unknown"
    t_color    = C_GREEN if is_known else (C_RED if name == "Unknown" else C_CYAN_DIM)

    # Pulsing ring around target name
    S.pulse_phase += 0.03
    ring_cx, ring_cy = (rx1+rx2)//2, ry1+110
    draw_pulse_ring(canvas, ring_cx, ring_cy, 50, S.pulse_phase, t_color, max_r=40)

    # Target name
    disp_name = name.upper()[:14]
    cv2.putText(canvas, disp_name, (ring_cx - len(disp_name)*9, ring_cy-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, t_color, 2, cv2.LINE_AA)

    # Status label
    stxt = "IDENTIFIED" if is_known else ("SCANNING" if name == "STANDBY" else "UNKNOWN")
    cv2.putText(canvas, f"[ {stxt} ]", (ring_cx - 35, ring_cy+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, t_color, 1, cv2.LINE_AA)

    # Confidence bar
    bar_y = ry1 + 170
    cv2.putText(canvas, "CONFIDENCE", (rx1+14, bar_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_GRAY, 1)
    draw_progress_bar(canvas, rx1+14, bar_y+8, rx2-rx1-110, 14,
                      conf/100.0, t_color)

    # Face count
    fc = len(S.faces)
    cv2.putText(canvas, "FACES IN FRAME", (rx1+14, bar_y+50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_GRAY, 1)
    cv2.putText(canvas, str(fc), (rx1+14, bar_y+72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_WHITE, 2, cv2.LINE_AA)

    # History
    cv2.putText(canvas, "DETECTION LOG", (rx1+14, bar_y+100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_CYAN_DIM, 1)
    cv2.line(canvas, (rx1+14, bar_y+106), (rx2-14, bar_y+106), C_GRAY_DARK, 1)

    hist_y = bar_y + 120
    recent = S.history[-6:]
    for i, (ts, nm, cf) in enumerate(recent):
        hc = C_GREEN if cf >= 50 else C_RED_DIM
        tstr = ts.strftime("%H:%M:%S")
        cv2.putText(canvas, f"{tstr}  {nm[:10]}  {cf:.0f}%",
                    (rx1+14, hist_y + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, hc, 1)

    # ── CENTER VIDEO PANEL ─────────────────────────────────────────────────
    # Panel border glow
    draw_glow_rect(canvas, vid_x1, vid_y1, vid_x2, vid_y2, C_CYAN_DIM, glow=8, alpha=0.1)
    cv2.rectangle(canvas, (vid_x1, vid_y1), (vid_x2, vid_y2), C_BG_PANEL, cv2.FILLED)
    draw_corner_brackets(canvas, vid_x1, vid_y1, vid_x2, vid_y2, C_CYAN, 2, 24)

    # Title
    cv2.putText(canvas, "LIVE FEED", (vid_x1+12, vid_y1+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_CYAN_DIM, 1, cv2.LINE_AA)
    cv2.line(canvas, (vid_x1+12, vid_y1+26), (vid_x2-12, vid_y1+26), C_CYAN_DIM, 1)

    # Video area
    vx1, vy1 = vid_x1+4, vid_y1+30
    vx2, vy2 = vid_x2-4, vid_y2-4
    vw, vh   = vx2-vx1, vy2-vy1

    if S.last_frame is not None:
        disp = cv2.resize(S.last_frame, (vw, vh), interpolation=cv2.INTER_LINEAR)
    else:
        disp = np.full((vh, vw, 3), C_BG_SCAN, dtype=np.uint8)
        cv2.putText(disp, "AWAITING SERVER RESPONSE...", (vw//2-140, vh//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_CYAN_DIM, 1, cv2.LINE_AA)
        # Animated dots
        dots = "." * (int(time.time()*3) % 4)
        cv2.putText(disp, dots, (vw//2+80, vh//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_CYAN, 1, cv2.LINE_AA)

    canvas[vy1:vy2, vx1:vx2] = disp

    # Scan line over video
    scan_p = (time.time() * 0.6) % 1.0
    scan_y = vy1 + int(scan_p * vh)
    cv2.line(canvas, (vx1, scan_y), (vx2, scan_y), C_CYAN_DIM, 1)
    for i in range(1, 5):
        cv2.line(canvas, (vx1, scan_y+i), (vx2, scan_y+i),
                 lerp_color(C_CYAN_DIM, C_BG, i/5), 1)

    # Crosshair in center
    ccx, ccy = (vx1+vx2)//2, (vy1+vy2)//2
    cv2.line(canvas, (ccx-12, ccy), (ccx+12, ccy), C_CYAN_DIM, 1)
    cv2.line(canvas, (ccx, ccy-12), (ccx, ccy+12), C_CYAN_DIM, 1)
    cv2.circle(canvas, (ccx, ccy), 4, C_CYAN_DIM, 1)

    # ── BOTTOM BAR ─────────────────────────────────────────────────────────
    by1 = H - bot_h
    cv2.rectangle(canvas, (0, by1), (W, H), C_BG_PANEL)
    cv2.line(canvas, (0, by1), (W, by1), C_CYAN_DIM, 1)

    # Status text
    if S.error_msg and time.time() - S.error_time < 4:
        stxt = f"ERR: {S.error_msg}"
        sc   = C_RED
    elif S.connected:
        stxt = f"◉ CONNECTED → {S.accel}"
        sc   = C_GREEN
    else:
        stxt = "○ DISCONNECTED — RETRYING..."
        sc   = C_ORANGE

    cv2.putText(canvas, stxt, (16, by1+24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, sc, 1)

    # Frame counter
    cv2.putText(canvas, f"FRAME {S.frame_count}", (W-200, by1+24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_GRAY, 1)

    # Animated bottom border
    bp = (time.time() * 1.2) % 1.0
    for x in range(0, W, 4):
        pulse = 0.5 + 0.5 * np.sin(x * 0.02 - time.time() * 3)
        cv2.line(canvas, (x, H-1), (x+3, H-1),
                 lerp_color(C_CYAN_DIM, C_BG, 1-pulse), 1)

    return canvas


# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main():
    boot_screen()

    # Camera
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("  ❌ Camera not found!")
        return

    print(f"  [J.A.R.V.I.S.] Camera opened.")
    print(f"  [J.A.R.V.I.S.] Server: {SERVER_URL}")
    print(f"  [J.A.R.V.I.S.] Send interval: every {SEND_EVERY} frames\n")

    # Background sender
    t = threading.Thread(target=send_worker, daemon=True)
    t.start()

    # Initial health check
    check_health()

    # UI window
    cv2.namedWindow("J.A.R.V.I.S. — Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("J.A.R.V.I.S. — Face Recognition", UI_W, UI_H)

    last_tick = time.time()
    fps_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        S.frame_count += 1
        fps_count += 1

        # FPS calc
        now = time.time()
        if now - last_tick >= 1.0:
            S.fps       = fps_count / (now - last_tick)
            fps_count   = 0
            last_tick   = now

        # Update current target
        if S.faces:
            best = max(S.faces, key=lambda f: f.get('confidence', 0))
            nm   = best['name']
            cf   = best['confidence']
            if (nm, round(cf)) != S.current_target:
                if S.current_target[0] != "STANDBY" or S.faces:
                    S.history.append((datetime.now(), nm, cf))
                    if len(S.history) > 100:
                        S.history.pop(0)
                S.current_target = (nm, cf)
                S.target_flash   = time.time()
        else:
            if S.current_target[0] != "STANDBY":
                S.current_target = ("STANDBY", 0.0)

        # Health check every 5s
        if int(now) % 5 == 0 and int(now) != getattr(main, '_last_hc', -1):
            main._last_hc = int(now)
            check_health()

        canvas = render_ui(frame)
        cv2.imshow("J.A.R.V.I.S. — Face Recognition", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('r'):
            try:
                requests.get(f"{SERVER_URL}/reload", timeout=5)
                print("  [J.A.R.V.I.S.] Face database reloaded.")
            except:
                pass

    cap.release()
    cv2.destroyAllWindows()
    print("  [J.A.R.V.I.S.] Shutdown complete.")


if __name__ == '__main__':
    main()
