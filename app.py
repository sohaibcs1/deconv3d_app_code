#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, gc, json, time, uuid, shutil, zipfile, tempfile, threading, traceback, sqlite3, re
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import imageio.v2 as imageio

# NEW: multi-format IO
import tifffile as tiff
import h5py
from aicsimageio import AICSImage
from werkzeug.exceptions import HTTPException

from flask import (
    Flask, render_template, request, redirect, url_for, abort, jsonify,
    send_from_directory, send_file, session, flash
)

from deconv3d_core import load_model, robust_norm, pad_to_multiple, unpad, infer_volume

# -------------------------
# Paths
# -------------------------
APP_ROOT = Path(__file__).parent.resolve()
STATIC_DIR = APP_ROOT / "static"
JOBS_DIR = STATIC_DIR / "jobs"
URL_PREFIX = "/deconv3d"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# store all user jobs under jobs/users/<email_slug>/<job_id>/
USERS_ROOT = JOBS_DIR / "users"
USERS_ROOT.mkdir(parents=True, exist_ok=True)

DB_PATH = APP_ROOT / "users.db"

# -------------------------
# Config (override via env)
# -------------------------
# ✅ NEW: two weights
WEIGHTS_LINEAR = os.getenv("DECONV3D_WEIGHTS_LINEAR", str(APP_ROOT / "checkpoint" / "best_3d_deconv.pt"))
WEIGHTS_BLOB   = os.getenv("DECONV3D_WEIGHTS_BLOB",   str(APP_ROOT / "checkpoint" / "best_3d_deconv.pt"))

BASE_CH = int(os.getenv("DECONV3D_BASE_CH", "24"))
WIN     = int(os.getenv("DECONV3D_WIN", "4"))
ROI     = int(os.getenv("DECONV3D_ROI", "64"))
OVERLAP = float(os.getenv("DECONV3D_OVERLAP", "0.75"))
PAD_WIN = os.getenv("DECONV3D_PAD_WIN", "1") == "1"
NO_AMP  = os.getenv("DECONV3D_NO_AMP", "1") == "1"
PNG_STRIDE = int(os.getenv("DECONV3D_PNG_STRIDE", "1"))
MAX_CONCURRENT = int(os.getenv("DECONV3D_MAX_CONCURRENT", "2"))

# retention: 10 hours
RETENTION_SEC = int(os.getenv("DECONV3D_RETENTION_SEC", str(10 * 60 * 60)))

# public samples visible to everyone forever (folders live directly under static/jobs/<job_id>/)
PUBLIC_SAMPLE_JOB_IDS = {"cc54276d0710", "f46b7f1f2076"}

# -------------------------
# Device
# -------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
USE_AMP = (DEVICE == "cuda") and (not NO_AMP)

torch.set_num_threads(int(os.getenv("DECONV3D_TORCH_THREADS", "1")))
torch.set_num_interop_threads(int(os.getenv("DECONV3D_TORCH_INTEROP_THREADS", "1")))

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB
app.secret_key = os.getenv("FLASK_SECRET_KEY", "muhammad_joseph_sahar_siyavash_scot_lan__bahram_parvin")

# throttle inference threads
JOB_SEM = threading.Semaphore(MAX_CONCURRENT)

# ✅ NEW: model cache (per weights path)
_MODELS = {}  # weights_path -> model
_MODEL_LOCK = threading.Lock()

# cleanup throttle
_LAST_CLEANUP_TS = 0.0
_CLEANUP_EVERY_SEC = 60  # run cleanup at most once per minute

# -------------------------
# Allowed extensions (NEW)
# -------------------------
ALLOWED_EXTS = (".nii", ".nii.gz", ".tif", ".tiff", ".h5", ".hdf5", ".czi")

# -------------------------
# DB helpers (sqlite3)
# -------------------------
def db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Email-only users table:
      users(email PRIMARY KEY, last_seen_ts REAL NOT NULL)

    This function ALSO migrates older schemas that had password_hash NOT NULL
    or missing last_seen_ts, by rebuilding the table safely.
    """
    conn = db()

    # ---- Ensure logs table exists (archive of deleted users) ----
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          email TEXT NOT NULL,
          last_seen_ts REAL,
          deleted_ts REAL NOT NULL,
          reason TEXT
        )
    """)
    conn.commit()

    # ✅ minimal migration: add total_jobs column if missing
    log_cols = [r["name"] for r in conn.execute("PRAGMA table_info(logs)").fetchall()]
    if "total_jobs" not in log_cols:
        conn.execute("ALTER TABLE logs ADD COLUMN total_jobs INTEGER")
        conn.commit()

    # If users table doesn't exist, create new correct one
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
    ).fetchone()

    if not exists:
        conn.execute("""
            CREATE TABLE users (
              email TEXT PRIMARY KEY,
              last_seen_ts REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        return

    # If exists, inspect columns
    cols_info = conn.execute("PRAGMA table_info(users)").fetchall()
    cols = [r["name"] for r in cols_info]

    # If schema already good, done
    if ("email" in cols) and ("last_seen_ts" in cols) and (len(cols) <= 3):
        conn.execute("UPDATE users SET last_seen_ts=? WHERE last_seen_ts IS NULL", (time.time(),))
        conn.commit()
        conn.close()
        return

    # Otherwise rebuild to new schema
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users_new (
          email TEXT PRIMARY KEY,
          last_seen_ts REAL NOT NULL
        )
    """)

    src_last = None
    if "last_seen_ts" in cols:
        src_last = "last_seen_ts"
    elif "last_seen" in cols:
        src_last = "last_seen"
    elif "ts" in cols:
        src_last = "ts"
    elif "created_ts" in cols:
        src_last = "created_ts"

    if src_last:
        conn.execute(f"""
            INSERT OR IGNORE INTO users_new(email, last_seen_ts)
            SELECT LOWER(TRIM(email)) AS email, COALESCE({src_last}, ?) AS last_seen_ts
            FROM users
            WHERE email IS NOT NULL AND TRIM(email) != ''
        """, (time.time(),))
    else:
        conn.execute("""
            INSERT OR IGNORE INTO users_new(email, last_seen_ts)
            SELECT LOWER(TRIM(email)) AS email, ?
            FROM users
            WHERE email IS NOT NULL AND TRIM(email) != ''
        """, (time.time(),))

    conn.execute("DROP TABLE users")
    conn.execute("ALTER TABLE users_new RENAME TO users")
    conn.commit()
    conn.close()


def list_all_users():
    conn = db()
    rows = conn.execute("SELECT email, last_seen_ts FROM users").fetchall()
    conn.close()
    return rows


def safe_slug_from_email(email: str) -> str:
    e = (email or "").strip().lower()
    e = re.sub(r"[^a-z0-9@._-]+", "_", e)
    return e[:120] if e else "unknown"


def count_user_jobs(email: str) -> int:
    slug = safe_slug_from_email(email)
    udir = USERS_ROOT / slug
    if not udir.exists():
        return 0
    return sum(1 for p in udir.iterdir() if p.is_dir())


def log_user_deletion(email: str, last_seen_ts: float, total_jobs: int, reason: str = "retention_expired"):
    email = (email or "").strip().lower()
    if not email:
        return
    conn = db()
    conn.execute("""
        INSERT INTO logs(email, last_seen_ts, deleted_ts, reason, total_jobs)
        VALUES(?, ?, ?, ?, ?)
    """, (email, float(last_seen_ts or 0), time.time(), reason, int(total_jobs)))
    conn.commit()
    conn.close()


def touch_user(email: str):
    email = (email or "").strip().lower()
    if not email:
        return
    now = time.time()
    conn = db()
    conn.execute("""
        INSERT INTO users(email, last_seen_ts) VALUES(?, ?)
        ON CONFLICT(email) DO UPDATE SET last_seen_ts=excluded.last_seen_ts
    """, (email, now))
    conn.commit()
    conn.close()


def delete_user(email: str):
    conn = db()
    conn.execute("DELETE FROM users WHERE email=?", ((email or "").strip().lower(),))
    conn.commit()
    conn.close()


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    # Return JSON for API endpoints
    if request.path.startswith("/predict") or request.path.startswith("/jobs") or request.path.startswith("/status"):
        return jsonify({"error": e.description}), e.code
    return e

# -------------------------
# Session helpers
# -------------------------
def current_email():
    return session.get("email")


def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_email():
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapper


def user_root_dir(email: str) -> Path:
    slug = safe_slug_from_email(email)
    p = USERS_ROOT / slug
    p.mkdir(parents=True, exist_ok=True)
    return p


def job_dir_for_email(email: str, job_id: str) -> Path:
    return user_root_dir(email) / job_id


# -------------------------
# Model / JSON helpers
# -------------------------
# ✅ NEW: pass weights_path
def get_model(weights_path: str):
    with _MODEL_LOCK:
        m = _MODELS.get(weights_path)
        if m is None:
            if not Path(weights_path).exists():
                raise FileNotFoundError(f"Weights not found: {weights_path}")
            _MODELS[weights_path] = load_model(weights_path, base_ch=BASE_CH, win=WIN, device=DEVICE)
            m = _MODELS[weights_path]
    return m


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2))


def safe_read_json(path: Path, default=None):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return default
    return default


def set_status(job_dir: Path, state: str, **kwargs):
    existing = safe_read_json(job_dir / "status.json", default={}) or {}
    data = {"state": state, "ts": time.time()}
    if "started_ts" in existing:
        data["started_ts"] = existing["started_ts"]
    data.update(kwargs)
    write_json(job_dir / "status.json", data)


# -------------------------
# Volume IO (NEW: multi-format)
# -------------------------
def _as_xyz(vol):
    """
    Standardize to (X,Y,Z) because your viewer/slicer expects Z on axis=2.
    Heuristic: if first dim is the smallest, assume it's Z and transpose ZYX->XYZ.
    """
    vol = np.asarray(vol)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume (X,Y,Z). Got shape {vol.shape}.")
    if vol.shape[0] <= vol.shape[1] and vol.shape[0] <= vol.shape[2]:
        vol = np.transpose(vol, (2, 1, 0))  # ZYX -> XYZ
    return vol


def load_any_volume(path: Path):
    """
    Returns:
      vol_xyz float32 in [raw], meta dict for saving, affine/header for NIfTI (else None)
    """
    p = str(path).lower()

    if p.endswith((".nii", ".nii.gz")):
        nii = nib.load(str(path))
        vol = nii.get_fdata(dtype=np.float32)
        vol = _as_xyz(vol).astype(np.float32)
        return vol, {"fmt": "nifti", "orig_ext": ".nii.gz" if p.endswith(".nii.gz") else ".nii"}, nii.affine, nii.header

    if p.endswith((".tif", ".tiff")):
        vol = tiff.imread(str(path))  # often (Z,Y,X)
        vol = _as_xyz(vol).astype(np.float32)
        return vol, {"fmt": "tiff", "orig_ext": ".tif" if p.endswith(".tif") else ".tiff"}, None, None

    if p.endswith((".h5", ".hdf5")):
        with h5py.File(str(path), "r") as f:
            dset_name = None
            def walk(name, obj):
                nonlocal dset_name
                if dset_name is None and isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                    dset_name = name
            f.visititems(walk)
            if dset_name is None:
                raise ValueError("No 3D dataset found in HDF5.")
            vol = f[dset_name][...]
        vol = _as_xyz(vol).astype(np.float32)
        return vol, {"fmt": "hdf5", "dset_in": dset_name, "orig_ext": ".h5" if p.endswith(".h5") else ".hdf5"}, None, None

    if p.endswith(".czi"):
        img = AICSImage(str(path))
        vol_zyx = img.get_image_data("ZYX")
        vol = _as_xyz(vol_zyx).astype(np.float32)
        return vol, {"fmt": "czi", "orig_ext": ".czi"}, None, None

    raise ValueError(f"Unsupported file: {path.name}")


def pad_basename(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    if lower.endswith(".tiff"):
        return name[:-5]
    if lower.endswith(".tif"):
        return name[:-4]
    if lower.endswith(".hdf5"):
        return name[:-5]
    if lower.endswith(".h5"):
        return name[:-3]
    if lower.endswith(".czi"):
        return name[:-4]
    return Path(name).stem


def save_same_format(pred_xyz: np.ndarray, job_dir: Path, in_name: str, meta: dict, affine=None, header=None):
    """
    Save prediction in the same format when feasible:
      - nifti -> nifti (same ext)
      - tiff  -> tiff stack (ZYX) float32
      - hdf5  -> hdf5 with dataset 'pred' (ZYX) float32
      - czi   -> NOT writable reliably; save as OME-TIFF
    Returns output filename.
    """
    fmt = meta.get("fmt")
    base = pad_basename(in_name)

    if fmt == "nifti":
        new_header = header.copy()
        new_header.set_data_dtype(np.float32)
        ext = meta.get("orig_ext", ".nii.gz")
        out_name = f"output_{base}{ext}"
        out_path = job_dir / out_name
        nib.save(nib.Nifti1Image(pred_xyz.astype(np.float32), affine=affine, header=new_header), str(out_path))
        return out_name

    if fmt == "tiff":
        out_name = f"output_{base}{meta.get('orig_ext', '.tif')}"
        out_path = job_dir / out_name
        pred_zyx = np.transpose(pred_xyz, (2, 1, 0))
        tiff.imwrite(str(out_path), pred_zyx.astype(np.float32))
        return out_name

    if fmt == "hdf5":
        out_ext = meta.get("orig_ext", ".h5")
        out_name = f"output_{base}{out_ext}"
        out_path = job_dir / out_name
        pred_zyx = np.transpose(pred_xyz, (2, 1, 0))
        dset_in = meta.get("dset_in", "")
        with h5py.File(str(out_path), "w") as f:
            f.create_dataset("pred", data=pred_zyx.astype(np.float32), compression="gzip")
            if dset_in:
                f.attrs["source_dataset"] = dset_in
        return out_name

    if fmt == "czi":
        out_name = f"output_{base}.ome.tif"
        out_path = job_dir / out_name
        pred_zyx = np.transpose(pred_xyz, (2, 1, 0))
        tiff.imwrite(str(out_path), pred_zyx.astype(np.float32), ome=True)
        return out_name

    raise ValueError(f"Unknown fmt: {fmt}")


# -------------------------
# PNG slice helpers (unchanged)
# -------------------------
def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return (img01 * 255.0).clip(0, 255).astype(np.uint8)


def save_slices_png(vol_xyz: np.ndarray, out_dir: Path, prefix: str, stride: int = 1):
    out_dir.mkdir(parents=True, exist_ok=True)
    if vol_xyz.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol_xyz.shape}")
    Z = vol_xyz.shape[2]
    stride = max(1, int(stride))
    z_indices = []
    k = 0
    for z in range(0, Z, stride):
        img2d = vol_xyz[:, :, z].T
        imageio.imwrite(str(out_dir / f"{prefix}_{k:04d}.png"), _to_uint8(img2d))
        z_indices.append(int(z))
        k += 1
    return z_indices


# -------------------------
# Inference (UPDATED: model selection)
# -------------------------
def run_infer(in_path: Path, weights_path: str):
    t0 = time.time()
    vol_xyz, fmt_meta, affine, header = load_any_volume(in_path)
    vol_xyz = robust_norm(vol_xyz)
    t_load = time.time() - t0

    vol_zyx = np.transpose(vol_xyz, (2, 1, 0)).astype(np.float32)

    if PAD_WIN:
        t1 = time.time()
        vol_pad, pads, orig_shape = pad_to_multiple(
            vol_zyx, mult=(WIN, WIN, WIN), min_size=(ROI, ROI, ROI), mode="reflect"
        )
        t_pad = time.time() - t1
    else:
        vol_pad, pads, orig_shape = vol_zyx, (0, 0, 0), vol_zyx.shape
        t_pad = 0.0

    t2 = time.time()
    pred_pad = infer_volume(
        get_model(weights_path),  # ✅ NEW
        vol_pad,
        roi=(ROI, ROI, ROI),
        overlap=OVERLAP,
        device=DEVICE,
        use_amp=USE_AMP
    )
    t_inf = time.time() - t2

    pred_zyx = unpad(pred_pad, pads, orig_shape) if PAD_WIN else pred_pad
    pred_zyx = np.clip(pred_zyx.astype(np.float32), 0.0, 1.0)

    pred_xyz = np.transpose(pred_zyx, (2, 1, 0))

    timing = {
        "load_sec": float(t_load),
        "pad_sec": float(t_pad),
        "infer_sec": float(t_inf),
        "total_sec": float(t_load + t_pad + t_inf),
    }
    return vol_xyz.astype(np.float32), pred_xyz, fmt_meta, affine, header, timing


def worker_process(email: str, job_id: str, job_dir: Path, in_path: Path, weights_path: str, mode: str):
    with JOB_SEM:
        in01 = pred01 = None
        try:
            set_status(job_dir, "running", message=f"Running inference ({mode})...", started_ts=time.time())
            in01, pred01, fmt_meta, affine, header, timing = run_infer(in_path, weights_path)

            set_status(job_dir, "running", message="Saving outputs...")

            pred_filename = save_same_format(
                pred01,
                job_dir=job_dir,
                in_name=in_path.name,
                meta=fmt_meta,
                affine=affine,
                header=header
            )
            pred_path = job_dir / pred_filename

            in_z = save_slices_png(in01, job_dir, "in", stride=PNG_STRIDE)
            _ = save_slices_png(pred01, job_dir, "pred", stride=PNG_STRIDE)

            meta = {
                "job_id": job_id,
                "email": email,
                "shape": list(map(int, in01.shape)),          # XYZ
                "input_format": fmt_meta.get("fmt"),
                "saved": {"input": in_path.name, "pred": pred_path.name},
                "z_indices": in_z,
                "timing": timing,
                "device": DEVICE,
                "model_mode": mode,
                "weights": Path(weights_path).name,
            }
            write_json(job_dir / "meta.json", meta)

            set_status(job_dir, "done", message="Done", n_slices=len(in_z))

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            write_json(job_dir / "error.json", {"error": err, "traceback": tb})
            set_status(job_dir, "error", message=err)
        finally:
            try:
                del in01, pred01
            except Exception:
                pass
            gc.collect()
            try:
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                elif DEVICE == "mps":
                    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
            except Exception:
                pass


# -------------------------
# Cleanup expired users (never touch samples)
# -------------------------
def cleanup_expired_users():
    now = time.time()
    for r in list_all_users():
        email = (r["email"] or "").strip().lower()
        last_seen = float(r["last_seen_ts"] or 0)
        if not email:
            continue
        if (now - last_seen) > RETENTION_SEC:
            slug = safe_slug_from_email(email)
            udir = USERS_ROOT / slug

            total_jobs = count_user_jobs(email)

            if udir.exists():
                shutil.rmtree(udir, ignore_errors=True)

            try:
                log_user_deletion(email, last_seen, total_jobs, reason="retention_expired")
            except Exception as e:
                print("[LOGS] failed to write logs:", e)

            delete_user(email)


def maybe_cleanup():
    global _LAST_CLEANUP_TS
    now = time.time()
    if (now - _LAST_CLEANUP_TS) < _CLEANUP_EVERY_SEC:
        return
    _LAST_CLEANUP_TS = now
    try:
        cleanup_expired_users()
    except Exception as e:
        print("[CLEANUP] error:", e)



@app.get("/samples/<path:fname>")
def samples(fname):
    # serves exact bytes; and forces download
    return send_from_directory(
        directory=str(STATIC_DIR / "samples"),
        path=fname,
        as_attachment=True,
        mimetype="application/gzip" if fname.endswith(".gz") else "application/octet-stream"
    )

@app.before_request
def _before():
    maybe_cleanup()


# -------------------------
# Public sample helpers
# -------------------------
def list_public_sample_jobs():
    out = []
    for job_id in sorted(PUBLIC_SAMPLE_JOB_IDS):
        jdir = JOBS_DIR / job_id
        if not jdir.exists():
            continue

        st = safe_read_json(jdir / "status.json", default={"state": "done", "ts": 0, "message": "Sample"})
        meta = safe_read_json(jdir / "meta.json", default=None)

        n_slices = None
        pred_name = None
        input_name = None
        if meta:
            z = meta.get("z_indices", [])
            n_slices = len(z) if isinstance(z, list) else None
            saved = meta.get("saved", {})
            pred_name = saved.get("pred")
            input_name = saved.get("input")

        out.append({
            "job_id": job_id,
            "state": st.get("state", "done"),
            "message": "Public sample",
            "ts": st.get("ts", 0),
            "n_slices": n_slices,
            "input_name": input_name,
            "pred_name": pred_name,
            "view_url": f"/view_public/{job_id}",
            "download_pred": f"/download_public/{job_id}/{pred_name}" if pred_name else None,
            "is_sample": True,
        })
    return out


# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return render_template("home.html", user_email=current_email())


@app.get("/login")
def login():
    return render_template("login.html", user_email=current_email(), action="/deconv3d/login", next=request.args.get("next") or "/demo")


@app.post("/login")
def login_post():
    email = (request.form.get("email") or "").strip().lower()
    if not email or "@" not in email:
        flash("Please enter a valid email.", "error")
        return redirect(url_for("login"))

    session["email"] = email
    touch_user(email)

    nxt = request.form.get("next") or "/demo"
    return redirect(nxt)


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.get("/demo")
@login_required
def demo():
    touch_user(current_email())
    return render_template("index.html", user_email=current_email())


# -------------------------
# API
# -------------------------
@app.post("/predict")
@login_required
def predict():
    """
    UPDATED: multi-file upload.
    Accepts:
      - files via form field "nii" (your JS)
      - optional form field: mode = "linear" or "blob"
    """
    touch_user(current_email())

    # ✅ NEW: read mode from form
    mode = (request.form.get("mode") or "linear").strip().lower()
    if mode not in ("linear", "blob"):
        mode = "linear"
    weights_path = WEIGHTS_BLOB if mode == "blob" else WEIGHTS_LINEAR

    files = []
    if "nii" in request.files:
        files = request.files.getlist("nii")
    elif "files" in request.files:
        files = request.files.getlist("files")

    if not files:
        abort(400, "Missing file field: nii (or files)")

    email = current_email()
    job_ids = []

    for f in files:
        if not f or not f.filename:
            continue

        orig = Path(f.filename).name
        fn = orig.lower()
        if not fn.endswith(ALLOWED_EXTS):
            abort(400, f"Unsupported file: {orig}. Allowed: {', '.join(ALLOWED_EXTS)}")

        job_id = uuid.uuid4().hex[:12]
        jdir = job_dir_for_email(email, job_id)
        jdir.mkdir(parents=True, exist_ok=True)

        in_path = jdir / orig
        f.save(str(in_path))

        # ✅ QUICK VALIDATION: must be 3D
        try:
            vol_xyz, _, _, _ = load_any_volume(in_path)
            if vol_xyz.ndim != 3:
                raise ValueError(f"Expected 3D volume, got shape {vol_xyz.shape}")
        except Exception as e:
            shutil.rmtree(jdir, ignore_errors=True)
            abort(400, f"{type(e).__name__}: {e}")

        set_status(jdir, "queued", message=f"Queued... ({mode})")

        # ✅ NEW: pass weights + mode into worker
        t = threading.Thread(target=worker_process, args=(email, job_id, jdir, in_path, weights_path, mode), daemon=True)
        t.start()

        job_ids.append(job_id)

    if not job_ids:
        abort(400, "No valid files uploaded")

    if len(job_ids) == 1:
        return jsonify({"job_id": job_ids[0]})
    return jsonify({"job_ids": job_ids})


@app.get("/jobs")
@login_required
def jobs():
    touch_user(current_email())

    email = current_email()
    udir = user_root_dir(email)

    jobs_out = []
    if udir.exists():
        for d in udir.iterdir():
            if not d.is_dir():
                continue
            job_id = d.name
            st = safe_read_json(d / "status.json", default={"state": "unknown", "ts": 0, "message": ""})
            meta = safe_read_json(d / "meta.json", default=None)

            n_slices = None
            pred_name = None
            input_name = None
            if meta:
                z = meta.get("z_indices", [])
                n_slices = len(z) if isinstance(z, list) else None
                saved = meta.get("saved", {})
                pred_name = saved.get("pred")
                input_name = saved.get("input")

            jobs_out.append({
                "job_id": job_id,
                "state": st.get("state", "unknown"),
                "message": st.get("message", ""),
                "ts": st.get("ts", 0),
                "n_slices": n_slices,
                "input_name": input_name,
                "pred_name": pred_name,
                "view_url": f"/view_center/{job_id}",
                "download_pred": f"/download/{job_id}/{pred_name}" if pred_name else None,
                "is_sample": False,
            })

    jobs_out.sort(key=lambda x: x.get("ts", 0), reverse=True)
    running = [j for j in jobs_out if j["state"] in ("queued", "running", "unknown")]
    done    = [j for j in jobs_out if j["state"] == "done"]

    done = list_public_sample_jobs() + done

    return jsonify({"running": running, "done": done})


@app.get("/status/<job_id>")
@login_required
def status(job_id):
    touch_user(current_email())

    email = current_email()
    jdir = job_dir_for_email(email, job_id)
    if not jdir.exists():
        return jsonify({"state": "missing"}), 404

    st = safe_read_json(jdir / "status.json", default={"state": "unknown"}) or {"state":"unknown"}
    state = st.get("state", "unknown")

    started_ts = st.get("started_ts")
    if started_ts and state in ("queued", "running"):
        st["elapsed_sec"] = float(time.time() - float(started_ts))

    return jsonify(st), 200


@app.get("/view_center/<job_id>")
@login_required
def view_center(job_id):
    touch_user(current_email())

    email = current_email()
    jdir = job_dir_for_email(email, job_id)
    meta = safe_read_json(jdir / "meta.json", default=None)
    if not meta:
        abort(404, "Meta not found (job not finished yet?)")
    n = int(len(meta.get("z_indices", [])))
    center = max(0, n // 2)
    return redirect(url_for("view_job", job_id=job_id, idx=center))


@app.get("/view/<job_id>/<int:idx>")
@login_required
def view_job(job_id, idx):
    touch_user(current_email())

    email = current_email()
    slug = safe_slug_from_email(email)
    jdir = job_dir_for_email(email, job_id)

    meta = safe_read_json(jdir / "meta.json", default=None)
    if not meta:
        abort(404, "Meta not found (job not finished yet?)")

    n = int(len(meta.get("z_indices", [])))
    if n <= 0:
        abort(404, "No slices saved")
    idx = max(0, min(idx, n - 1))

    in_url  = URL_PREFIX + url_for("static", filename=f"jobs/users/{slug}/{job_id}/in_{idx:04d}.png")
    pred_url = URL_PREFIX + url_for("static", filename=f"jobs/users/{slug}/{job_id}/pred_{idx:04d}.png")
    img_base = URL_PREFIX + url_for("static", filename=f"jobs/users/{slug}/{job_id}")

    pred_name = meta.get("saved", {}).get("pred")
    download_pred_url = f"{URL_PREFIX}/download/{job_id}/{pred_name}" if pred_name else None

    return render_template(
        "view.html",
        job_id=job_id,
        idx=idx,
        n_slices=n,
        in_url=in_url,
        pred_url=pred_url,
        img_base=img_base,
        download_pred_url=download_pred_url,
        user_email=email,
        badge_text="Your job"
    )


@app.get("/download/<job_id>/<filename>")
@login_required
def download_file(job_id, filename):
    touch_user(current_email())

    email = current_email()
    jdir = job_dir_for_email(email, job_id)
    if not jdir.exists():
        abort(404, "Job not found")
    file_path = jdir / filename
    if not file_path.exists():
        abort(404, "File not found")
    return send_from_directory(directory=str(jdir), path=filename, as_attachment=True)


@app.get("/view_public/<job_id>")
@login_required
def view_public_center(job_id):
    touch_user(current_email())

    if job_id not in PUBLIC_SAMPLE_JOB_IDS:
        abort(404)
    jdir = JOBS_DIR / job_id
    meta = safe_read_json(jdir / "meta.json", default=None)
    if not meta:
        abort(404)
    n = int(len(meta.get("z_indices", [])))
    center = max(0, n // 2)
    return redirect(url_for("view_public", job_id=job_id, idx=center))


@app.get("/view_public/<job_id>/<int:idx>")
@login_required
def view_public(job_id, idx):
    touch_user(current_email())

    if job_id not in PUBLIC_SAMPLE_JOB_IDS:
        abort(404)
    jdir = JOBS_DIR / job_id
    meta = safe_read_json(jdir / "meta.json", default=None)
    if not meta:
        abort(404)

    n = int(len(meta.get("z_indices", [])))
    idx = max(0, min(idx, n - 1))

    in_url  = URL_PREFIX + url_for("static", filename=f"jobs/{job_id}/in_{idx:04d}.png")
    pred_url = URL_PREFIX + url_for("static", filename=f"jobs/{job_id}/pred_{idx:04d}.png")
    img_base = URL_PREFIX + url_for("static", filename=f"jobs/{job_id}")

    pred_name = meta.get("saved", {}).get("pred")
    download_pred_url = f"{URL_PREFIX}/download_public/{job_id}/{pred_name}" if pred_name else None

    return render_template(
        "view.html",
        job_id=job_id,
        idx=idx,
        n_slices=n,
        in_url=in_url,
        pred_url=pred_url,
        img_base=img_base,
        download_pred_url=download_pred_url,
        user_email=current_email(),
        badge_text="Public sample"
    )


@app.get("/download_public/<job_id>/<filename>")
@login_required
def download_public(job_id, filename):
    touch_user(current_email())

    if job_id not in PUBLIC_SAMPLE_JOB_IDS:
        abort(404)
    jdir = JOBS_DIR / job_id
    file_path = jdir / filename
    if not file_path.exists():
        abort(404)
    return send_from_directory(directory=str(jdir), path=filename, as_attachment=True)


@app.get("/download_selected.zip")
@login_required
def download_selected_zip():
    touch_user(current_email())

    email = current_email()
    slug = safe_slug_from_email(email)
    job_ids = request.args.getlist("job_id")
    job_ids = [j.strip() for j in job_ids if j and j.strip()]
    if not job_ids:
        abort(400, "No job_id provided")

    tmpdir = tempfile.TemporaryDirectory()
    tmp_path = os.path.join(tmpdir.name, "selected_predictions.zip")

    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for job_id in job_ids:
            if job_id in PUBLIC_SAMPLE_JOB_IDS:
                jdir = JOBS_DIR / job_id
                meta = safe_read_json(jdir / "meta.json", default=None)
                if not meta:
                    continue
                pred_name = meta.get("saved", {}).get("pred")
                if not pred_name:
                    continue
                pred_path = jdir / pred_name
                if pred_path.exists():
                    zf.write(str(pred_path), arcname=f"samples/{job_id}/{pred_name}")
                continue

            jdir = USERS_ROOT / slug / job_id
            meta = safe_read_json(jdir / "meta.json", default=None)
            if not meta:
                continue
            pred_name = meta.get("saved", {}).get("pred")
            if not pred_name:
                continue
            pred_path = jdir / pred_name
            if pred_path.exists():
                zf.write(str(pred_path), arcname=f"{job_id}/{pred_name}")

    resp = send_file(
        tmp_path,
        as_attachment=True,
        download_name="selected_predictions.zip",
        mimetype="application/zip",
    )
    resp.call_on_close(tmpdir.cleanup)
    return resp


@app.route("/jobs/<job_id>", methods=["DELETE"])
@login_required
def delete_job(job_id):
    touch_user(current_email())

    if job_id in PUBLIC_SAMPLE_JOB_IDS:
        return jsonify({"ok": False, "error": "Cannot delete sample job"}), 403

    email = current_email()
    jdir = job_dir_for_email(email, job_id)
    if jdir.exists():
        shutil.rmtree(jdir, ignore_errors=True)
    return jsonify({"ok": True, "job_id": job_id})


if __name__ == "__main__":
    init_db()
    cleanup_expired_users()
    app.run(host="0.0.0.0", port=8011, debug=True, use_reloader=False)
