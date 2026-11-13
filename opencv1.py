# ─────────────────────────────────────────────────────────────────────────────
#  server.py  ·  Face Recognition GPU Server  ·  CLEAN FINAL EDITION
#  Strategy: NO monkey-patching. Load on CPU. Replace sessions after. Done.
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import cv2
import base64
import os
import time
import warnings

app = Flask(__name__)

# ── COLORS (BGR) ──────────────────────────────────────────────────────────────
CYAN        = (255, 255,   0)
GREEN_NEON  = (  0, 255, 128)
RED_NEON    = (  0,  50, 255)
MAGENTA     = (255,   0, 200)
WHITE       = (255, 255, 255)
DARK        = ( 10,  10,  10)
YELLOW_NEON = (  0, 220, 255)
GRAY        = (120, 120, 120)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 · SAVE THE REAL ORT SESSION CLASS BEFORE ANYTHING TOUCHES IT       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

_RealSession = ort.InferenceSession      # saved immediately, nothing patched


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 · DETECT WORKING GPU PROVIDER                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

INSIGHTFACE_ROOT = r'C:\Users\amd\.insightface'
MODEL_DIR        = os.path.join(INSIGHTFACE_ROOT, 'models', 'buffalo_l')
TRT_CACHE_DIR    = os.path.join(INSIGHTFACE_ROOT, 'trt_cache')
TEST_MODEL       = os.path.join(MODEL_DIR, 'det_10g.onnx')

print("=" * 65)
print("  GPU PROVIDER DETECTION")
print("=" * 65)
print(f"  onnxruntime version : {ort.__version__}")

available_provs = ort.get_available_providers()
print(f"  ORT compiled with   : {available_provs}")
print()

ort.set_default_logger_severity(0)      # verbose — shows real errors


def probe(providers_cfg, label):
    if not os.path.exists(TEST_MODEL):
        print(f"  ⏭   {label:44s} → model not on disk yet")
        return False
    try:
        sess  = _RealSession(TEST_MODEL, providers=providers_cfg)
        first = (sess.get_providers() or ['none'])[0]
        ok    = 'CPU' not in first
        icon  = '✅' if ok else '❌  (fell back to CPU!)'
        print(f"  {icon}  {label:44s} → {first}")
        return ok
    except Exception as exc:
        print(f"  ❌  {label:44s} → {type(exc).__name__}: {exc}")
        return False


os.makedirs(TRT_CACHE_DIR, exist_ok=True)

TRT_CFG = [
    ('TensorrtExecutionProvider', {
        'device_id':               0,
        'trt_max_workspace_size':  2 * 1024 ** 3,
        'trt_fp16_enable':         True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path':   TRT_CACHE_DIR,
    }),
    ('CUDAExecutionProvider', {'device_id': 0}),
    'CPUExecutionProvider',
]

CUDA_CFG = [
    ('CUDAExecutionProvider', {
        'device_id':                 0,
        'gpu_mem_limit':             8 * 1024 ** 3,
        'cudnn_conv_algo_search':    'DEFAULT',
        'do_copy_in_default_stream': True,
        'arena_extend_strategy':     'kSameAsRequested',
    }),
    'CPUExecutionProvider',
]

DML_CFG = ['DmlExecutionProvider', 'CPUExecutionProvider']
CPU_CFG = ['CPUExecutionProvider']

ACTIVE_CFG  = CPU_CFG
ACTIVE_NAME = 'CPU'

if 'TensorrtExecutionProvider' in available_provs:
    if probe(TRT_CFG, 'TensorRT  (TRT + CUDA + FP16)'):
        ACTIVE_CFG  = TRT_CFG
        ACTIVE_NAME = 'TensorRT'

if ACTIVE_NAME == 'CPU' and 'CUDAExecutionProvider' in available_provs:
    if probe(CUDA_CFG, 'CUDA'):
        ACTIVE_CFG  = CUDA_CFG
        ACTIVE_NAME = 'CUDA'

if ACTIVE_NAME == 'CPU':
    if 'DmlExecutionProvider' in available_provs:
        if probe(DML_CFG, 'DirectML  (Windows GPU — any brand)'):
            ACTIVE_CFG  = DML_CFG
            ACTIVE_NAME = 'DirectML'
    else:
        print("  ℹ   DirectML not found → pip install onnxruntime-directml")

ort.set_default_logger_severity(3)      # back to quiet

print()
if ACTIVE_NAME == 'CPU':
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  ⚠️  ALL GPU PROVIDERS FAILED — CPU MODE                  ║")
    print("  ║  pip install onnxruntime-directml   ← quickest fix       ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
else:
    print(f"  🚀  ACTIVE ACCELERATOR : {ACTIVE_NAME}")
print("=" * 65 + "\n")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 · LOAD INSIGHTFACE ON CPU (no patching — totally safe)             ║
# ║  We deliberately let InsightFace do whatever it wants here.                ║
# ║  We will replace all sessions in Step 4 after load completes.              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

print("  Loading InsightFace models on CPU (will move to GPU next)...")

from insightface.app import FaceAnalysis

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    face_app = FaceAnalysis(name='buffalo_l', root=INSIGHTFACE_ROOT)
    face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

print("  InsightFace loaded.\n")


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 · POST-LOAD SESSION REPLACEMENT — THE RELIABLE GPU METHOD          ║
# ║                                                                             ║
# ║  Now that InsightFace has finished loading, we directly replace every      ║
# ║  model's .session with a brand-new _RealSession using ACTIVE_CFG.          ║
# ║  _RealSession is the original ort.InferenceSession saved in Step 1 —       ║
# ║  nothing can intercept or redirect it.                                     ║
# ║                                                                             ║
# ║  model.model_file is set by InsightFace on every model object and          ║
# ║  contains the absolute path to the .onnx file.                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

print("  Moving all models to GPU...")
print("  " + "─" * 52)
all_on_gpu = True

for task_name, model in face_app.models.items():
    try:
        # ── Find the .onnx file path ───────────────────────────
        # InsightFace stores it in model_file on every model object
        model_file = None
        for attr in ('model_file', 'onnx_file', 'model_path', '_model_file'):
            val = getattr(model, attr, None)
            if isinstance(val, str) and val.endswith('.onnx'):
                model_file = val
                break

        # Last resort: read it from the existing session
        if model_file is None and hasattr(model, 'session'):
            sess_path = getattr(model.session, '_model_path', None)
            if sess_path:
                model_file = sess_path

        if not model_file or not os.path.exists(model_file):
            all_on_gpu = False
            print(f"    ⚠️   {task_name:26s} → .onnx path not found, skipping")
            continue

        # ── Create brand-new GPU session using the REAL ORT class ─
        new_session   = _RealSession(model_file, providers=ACTIVE_CFG)
        model.session = new_session

        # ── Verify ────────────────────────────────────────────────
        first  = (new_session.get_providers() or ['unknown'])[0]
        is_gpu = 'CPU' not in first
        icon   = '🟢' if is_gpu else '🔴'
        if not is_gpu:
            all_on_gpu = False
        print(f"    {icon}  {task_name:26s} → {first}")

    except Exception as exc:
        all_on_gpu = False
        print(f"    ❌  {task_name:26s} → {exc}")

print("  " + "─" * 52)
if all_on_gpu:
    print(f"  🚀  ALL MODELS ON {ACTIVE_NAME}! Full speed!")
else:
    print(f"  ⚠️   Some models on CPU  (target: {ACTIVE_NAME})")
print("=" * 65 + "\n")


# ── FACE DATABASE ──────────────────────────────────────────────────────────────
known_embeddings: list = []
known_names:      list = []


def load_my_faces():
    global known_embeddings, known_names
    known_embeddings = []
    known_names      = []

    # ╔══════════════════════════════════╗
    folder = r"C:\Users\amd\Documents\knownfaces"
    # ╚══════════════════════════════════╝

    if not os.path.exists(folder):
        print(f"  ❌ Folder not found: {folder}")
        return

    print(f"  Loading faces from:\n  {folder}\n")

    for person_name in sorted(os.listdir(folder)):
        person_dir = os.path.join(folder, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"  👤  {person_name}")
        count = 0

        for fname in os.listdir(person_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                print(f"       SKIP (unreadable): {fname}")
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                faces = face_app.get(img)

            if not faces:
                print(f"       ⚠️  No face found: {fname}")
                continue

            best = max(faces, key=lambda f:
                       (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            known_embeddings.append(best.normed_embedding)
            known_names.append(person_name)
            count += 1
            print(f"       ✅  {fname}")

        print(f"       📊  {count} photo(s) loaded\n")

    print("=" * 65)
    print(f"  Total people : {len(set(known_names))}")
    print(f"  Total photos : {len(known_names)}")
    print(f"  Names        : {sorted(set(known_names))}")
    print("=" * 65 + "\n")


load_my_faces()


# ── DRAWING HELPERS ────────────────────────────────────────────────────────────
def draw_corner_brackets(frame, x1, y1, x2, y2,
                          color, thickness=2, length=25):
    cv2.line(frame, (x1, y1), (x1+length, y1),  color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1+length),  color, thickness)
    cv2.line(frame, (x2, y1), (x2-length, y1),  color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1+length),  color, thickness)
    cv2.line(frame, (x1, y2), (x1+length, y2),  color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2-length),  color, thickness)
    cv2.line(frame, (x2, y2), (x2-length, y2),  color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2-length),  color, thickness)


def draw_dashed_rect(frame, x1, y1, x2, y2,
                     color, thickness=1, dash=10):
    pts = [
        ((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1)),
    ]
    for (sx, sy), (ex, ey) in pts:
        dx, dy  = ex - sx, ey - sy
        seg_len = int(np.hypot(dx, dy))
        if seg_len == 0:
            continue
        for i in range(0, seg_len, dash * 2):
            t1 = i / seg_len
            t2 = min((i + dash) / seg_len, 1.0)
            cv2.line(frame,
                     (int(sx + t1*dx), int(sy + t1*dy)),
                     (int(sx + t2*dx), int(sy + t2*dy)),
                     color, thickness)


def draw_label_hud(frame, x1, y1, x2, name, confidence, is_known):
    color  = GREEN_NEON if is_known else RED_NEON
    accent = CYAN       if is_known else MAGENTA
    stxt   = "IDENTIFIED" if is_known else "UNKNOWN"

    overlay = frame.copy()
    lh      = 52
    cv2.rectangle(overlay, (x1, y1-lh-4), (x2, y1-2), DARK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.line(frame, (x1, y1-lh-4), (x2, y1-lh-4), color, 1)
    cv2.putText(frame, name.upper(),
                (x1+8, y1-lh+16), cv2.FONT_HERSHEY_DUPLEX,
                0.6, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"[ {stxt} ]",
                (x1+8, y1-lh+34), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, color, 1, cv2.LINE_AA)
    bx1, bx2, by = x1+8, x2-8, y1-8
    cv2.rectangle(frame, (bx1, by-6), (bx2, by), (40, 40, 40), cv2.FILLED)
    cv2.rectangle(frame, (bx1, by-6),
                  (bx1 + int((bx2-bx1) * (confidence/100)), by),
                  color, cv2.FILLED)
    cv2.putText(frame, f"{confidence:.1f}%",
                (bx2-45, by-9), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, accent, 1, cv2.LINE_AA)


def draw_scan_line(frame, scan_y):
    h, w   = frame.shape[:2]
    scan_y = scan_y % h
    ov     = frame.copy()
    cv2.line(ov, (0, scan_y), (w, scan_y), CYAN, 2)
    cv2.addWeighted(ov, 0.3, frame, 0.7, 0, frame)
    for i in range(1, 5):
        glow = frame.copy()
        cv2.line(glow, (0, scan_y-i), (w, scan_y-i), CYAN, 1)
        cv2.line(glow, (0, scan_y+i), (w, scan_y+i), CYAN, 1)
        cv2.addWeighted(glow, 0.06/i, frame, 1-(0.06/i), 0, frame)


def draw_corner_hud(frame):
    h, w   = frame.shape[:2]
    mg, sz = 12, 20
    for cx, cy, dx, dy in [
        (mg,   mg,    1,  1), (w-mg, mg,   -1,  1),
        (mg,   h-mg,  1, -1), (w-mg, h-mg, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx+dx*sz, cy),        CYAN, 1)
        cv2.line(frame, (cx, cy), (cx,        cy+dy*sz), CYAN, 1)
    cv2.rectangle(frame, (mg+2, mg+2), (w-mg-2, h-mg-2), (30, 30, 30), 1)


# ── FACE MATCHING ──────────────────────────────────────────────────────────────
def find_who_is_this(embedding, threshold=0.5):
    if not known_embeddings:
        return "Unknown", 0.0
    best_name  = "Unknown"
    best_score = 0.0
    for person in set(known_names):
        embs  = [known_embeddings[i]
                 for i, n in enumerate(known_names) if n == person]
        score = float(np.mean(np.dot(np.array(embs), embedding)))
        if score > best_score:
            best_score = score
            best_name  = person
    if best_score >= threshold:
        return best_name, round(best_score * 100, 1)
    return "Unknown", round(best_score * 100, 1)


scan_offset = [0]


# ── FLASK ROUTES ───────────────────────────────────────────────────────────────
@app.route('/process', methods=['POST'])
def process():
    data        = request.get_json()
    frame_bytes = base64.b64decode(data['frame'])
    frame       = cv2.imdecode(
                      np.frombuffer(frame_bytes, np.uint8),
                      cv2.IMREAD_COLOR,
                  )
    scan_offset[0] += 6
    draw_scan_line(frame, scan_offset[0])

    faces   = face_app.get(frame)
    results = []

    for face in faces:
        x1, y1, x2, y2  = map(int, face.bbox)
        name, confidence = find_who_is_this(face.normed_embedding)
        is_known         = name != "Unknown"
        color            = GREEN_NEON if is_known else RED_NEON

        draw_dashed_rect(frame, x1+4, y1+4, x2-4, y2-4, color, 1, 8)
        draw_corner_brackets(frame, x1, y1, x2, y2, color, 2, 28)

        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(frame, (cx, cy), 3, color, -1)
        cv2.line(frame, (cx-10, cy), (cx+10, cy), color, 1)
        cv2.line(frame, (cx, cy-10), (cx, cy+10), color, 1)

        if hasattr(face, 'kps') and face.kps is not None:
            for kp in face.kps.astype(int):
                cv2.circle(frame, tuple(kp), 2, YELLOW_NEON, -1)
                cv2.circle(frame, tuple(kp), 5, YELLOW_NEON,  1)

        draw_label_hud(frame, x1, max(y1, 60), x2, name, confidence, is_known)
        results.append({'name': name, 'confidence': confidence})

    draw_corner_hud(frame)

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return jsonify({
        'frame': base64.b64encode(buf).decode('utf-8'),
        'count': len(faces),
        'faces': results,
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'running',
        'accelerator':  ACTIVE_NAME,
        'known_faces':  sorted(set(known_names)),
        'total_people': len(set(known_names)),
        'total_photos': len(known_names),
        'gpu':          ACTIVE_NAME != 'CPU',
    })


@app.route('/reload', methods=['GET'])
def reload_faces():
    load_my_faces()
    return jsonify({
        'status':       'reloaded',
        'names':        sorted(set(known_names)),
        'total_people': len(set(known_names)),
        'total_photos': len(known_names),
    })


if __name__ == '__main__':
    print(f"  Server       → http://0.0.0.0:5000")
    print(f"  Accelerator  → {ACTIVE_NAME}")
    print(f"  Waiting for laptop...\n")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
