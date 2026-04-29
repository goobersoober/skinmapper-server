"""
SkinMapper Server — GPU-accelerated photogrammetry pipeline
COLMAP (dense reconstruction) + open3d (meshing) + texture baking from photos
"""

import os, uuid, zipfile, subprocess, threading, json, shutil, logging, traceback, time, struct, multiprocessing
from collections import deque

# COLMAP requires a display on headless servers — force offscreen Qt
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ''

from flask import Flask, request, jsonify, send_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

JOBS_DIR = '/workspace/jobs'  # persistent volume — survives gunicorn worker restarts
MAX_CONCURRENT = 2
MAX_QUEUE = 10
MAX_PHOTOS = 60
JOB_TTL = 3600
MAX_DIM = 1600  # COLMAP feature matching is sensitive to resolution — keep at proven value

# Set FAST_MODE=1 on RunPod for quick test runs (lower res, faster turnaround ~4-5 min)
# Set FAST_MODE=0 (or unset) for full quality production runs (~15-20 min)
FAST_MODE = os.environ.get('FAST_MODE', '0') == '1'
if FAST_MODE:
    logging.info('FAST_MODE enabled — using reduced resolution for quick testing')

os.makedirs(JOBS_DIR, exist_ok=True)

# ── Check GPU availability at startup ────────────────────────────────
def check_gpu():
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True)
        gpu_name = gpu.stdout.strip() if gpu.returncode == 0 else 'No GPU detected'
    except Exception:
        gpu_name = 'No GPU detected'
        gpu = type('r', (), {'returncode': 1})()
    logging.info(f'GPU: {gpu_name}')
    return gpu.returncode == 0, gpu_name

HAS_GPU, GPU_NAME = check_gpu()

# ── Job state ────────────────────────────────────────────────────────
jobs = {}
lock = threading.Lock()
queue = deque()
active_count = 0

def set_job(job_id, status, progress, message, result_url=None, error=None):
    data = dict(status=status, progress=progress, message=message,
                result_url=result_url, error=error, updated=time.time())
    with lock:
        jobs[job_id] = data
    # Write to disk so new gunicorn workers can read status after a reload.
    # The pipeline runs in a subprocess; gunicorn workers read this file.
    try:
        job_dir = os.path.join(JOBS_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        tmp = os.path.join(job_dir, 'status.json.tmp')
        dst = os.path.join(job_dir, 'status.json')
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, dst)  # atomic
    except Exception as _e:
        logging.warning(f'set_job disk write failed for {job_id}: {_e}')

def get_job(job_id):
    # Always prefer disk — the pipeline is a subprocess and writes status
    # to disk; the in-memory dict in the gunicorn worker is always stale.
    try:
        path = os.path.join(JOBS_DIR, job_id, 'status.json')
        if os.path.exists(path):
            with open(path) as f:
                j = json.load(f)
            with lock:
                jobs[job_id] = j
            return j
    except Exception:
        pass
    with lock:
        return jobs.get(job_id)

# ── Job queue ────────────────────────────────────────────────────────
def enqueue_job(job_id, img_dir, job_dir, scan_type='halfWrap', body_part='leg'):
    global active_count
    with lock:
        if active_count < MAX_CONCURRENT:
            active_count += 1
            threading.Thread(target=_run_and_release,
                             args=(job_id, img_dir, job_dir, scan_type, body_part),
                             daemon=True).start()
        else:
            queue.append((job_id, img_dir, job_dir, scan_type, body_part))
            set_job(job_id, 'queued', 0.0, f'In queue (position {len(queue)})')

def _pipeline_in_new_session(job_id, img_dir, job_dir, scan_type, body_part):
    """Child-process entry point — runs in its own session, immune to gunicorn SIGHUP."""
    os.setsid()
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        stream=open('/workspace/server.log', 'a'))
    run_pipeline(job_id, img_dir, job_dir, scan_type=scan_type, body_part=body_part)


def _run_and_release(job_id, img_dir, job_dir, scan_type='halfWrap', body_part='leg'):
    global active_count
    try:
        p = multiprocessing.Process(
            target=_pipeline_in_new_session,
            args=(job_id, img_dir, job_dir, scan_type, body_part),
            daemon=False)
        p.start()
        logging.info(f'[{job_id[:8]}] Pipeline subprocess PID={p.pid} (session-isolated)')
        while p.is_alive():
            p.join(timeout=30)
        logging.info(f'[{job_id[:8]}] Pipeline subprocess finished (exit={p.exitcode})')
    finally:
        with lock:
            active_count -= 1
        _start_next()

def _start_next():
    global active_count
    with lock:
        if queue and active_count < MAX_CONCURRENT:
            active_count += 1
            job_id, img_dir, job_dir, scan_type, body_part = queue.popleft()
            threading.Thread(target=_run_and_release,
                             args=(job_id, img_dir, job_dir, scan_type, body_part),
                             daemon=True).start()

def cleanup_old_jobs():
    now = time.time()
    with lock:
        expired = [jid for jid, j in jobs.items() if now - j.get('updated', 0) > JOB_TTL]
        for jid in expired:
            del jobs[jid]
            shutil.rmtree(os.path.join(JOBS_DIR, jid), ignore_errors=True)


# ── Foreground masking (rembg) ───────────────────────────────────────
# We segment skin/body from background BEFORE feeding photos to COLMAP.
# Without this, walls, floor, clothing, and other off-body geometry get
# reconstructed as part of the mesh and produce a lumpy non-skin shape
# plus a shattered texture (different cameras pick up different chunks
# of background and bake them onto the unwrap).

# rembg session is heavy to construct; cache one per worker process.
_REMBG_SESSION = None

def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        from rembg import new_session
        _REMBG_SESSION = new_session('u2net')
    return _REMBG_SESSION


def generate_foreground_masks(image_dir, masks_dir, tag):
    """
    For each photo write a mask file at masks_dir/<image_name>.png.

    COLMAP's --ImageReader.mask_path expects exactly this layout: a folder
    of PNG masks named <original_image_filename>.png. White = features
    allowed, black = features rejected. So `IMG_001.jpg` needs mask
    `IMG_001.jpg.png`.

    The mask is dilated by ~2% so silhouette features (which COLMAP needs
    for matching) survive — without dilation the mask hugs the body too
    tightly and matching gets weaker on the body's outline.
    """
    from rembg import remove
    import cv2
    import numpy as np
    os.makedirs(masks_dir, exist_ok=True)
    session = _get_rembg_session()
    files = [f for f in os.listdir(image_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()
    n_ok = 0
    for fname in files:
        try:
            with open(os.path.join(image_dir, fname), 'rb') as f:
                data = f.read()
            mask_bytes = remove(data, session=session, only_mask=True,
                                post_process_mask=True)
            arr = np.frombuffer(mask_bytes, dtype=np.uint8)
            mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            # Slight dilation (~2% of image dim) so silhouette features survive.
            k = max(3, int(min(mask.shape) * 0.02) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)
            # Binarise — COLMAP only checks > 0.
            _, mask = cv2.threshold(mask, 64, 255, cv2.THRESH_BINARY)
            cv2.imwrite(os.path.join(masks_dir, fname + '.png'), mask)
            n_ok += 1
        except Exception as e:
            logging.warning(f'[{tag}] mask {fname}: {e}')
    logging.info(f'[{tag}] Foreground masks: {n_ok}/{len(files)} generated')
    return n_ok


def filter_pcd_by_masks(pts, colors, masks_dir, sparse_dir, image_dir, tag,
                       min_visible=2, min_foreground_ratio=0.55):
    """
    Reject points whose reprojection lands on background in most cameras.

    For each 3D point we reproject into every COLMAP-registered camera.
    A point survives only if among the cameras that actually see it
    (in front of camera + within image bounds), the majority show it as
    foreground in their rembg mask.

    This catches background that slipped past the per-image feature mask
    (e.g. wood-floor patches that share texture with skin).
    """
    import cv2
    import numpy as np
    cameras = _read_colmap_cameras_txt(os.path.join(sparse_dir, 'cameras.txt'))
    images  = _read_colmap_images_txt(os.path.join(sparse_dir, 'images.txt'))
    if not cameras or not images:
        logging.info(f'[{tag}] mask filter: no COLMAP cameras, skipping')
        return pts, colors

    cam_list = []
    mask_cache = {}
    for img_info in images:
        cam = cameras.get(img_info['camera_id'])
        if cam is None:
            continue
        mask_path = os.path.join(masks_dir, img_info['name'] + '.png')
        if not os.path.exists(mask_path):
            continue
        mask = mask_cache.get(mask_path)
        if mask is None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask_cache[mask_path] = mask
        cam_list.append({**img_info, 'cam': cam, 'mask': mask})
    if not cam_list:
        logging.info(f'[{tag}] mask filter: no camera/mask pairs, skipping')
        return pts, colors

    P = pts.astype(np.float64)
    visible_count   = np.zeros(len(P), dtype=np.int32)
    foreground_count = np.zeros(len(P), dtype=np.int32)

    for c in cam_list:
        R, t  = c['R'], c['t']
        cam   = c['cam']
        mask  = c['mask']
        mh, mw = mask.shape
        # COLMAP image dimensions may differ from the mask if we resized
        # the mask after generation — scale projected pixels to mask space.
        sx = mw / float(cam['w'])
        sy = mh / float(cam['h'])

        pts_cam = (R @ P.T).T + t
        in_front = pts_cam[:, 2] > 0.01
        if not in_front.any():
            continue
        z = np.where(in_front, pts_cam[:, 2], 1.0)
        u = cam['fx'] * pts_cam[:, 0] / z + cam['cx']
        v = cam['fy'] * pts_cam[:, 1] / z + cam['cy']
        mu = (u * sx).astype(np.int32)
        mv = (v * sy).astype(np.int32)
        in_bounds = in_front & (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
        if not in_bounds.any():
            continue
        visible_count[in_bounds] += 1
        idx = np.where(in_bounds)[0]
        fg = mask[mv[idx], mu[idx]] > 64
        foreground_count[idx[fg]] += 1

    seen   = visible_count >= min_visible
    ratio  = np.divide(foreground_count, np.maximum(visible_count, 1),
                       dtype=np.float32)
    keep   = (~seen) | (ratio >= min_foreground_ratio)
    n_drop = int((~keep).sum())
    logging.info(f'[{tag}] Mask filter: dropped {n_drop}/{len(P)} pts '
                 f'({n_drop/len(P)*100:.1f}%) as background')
    if colors is not None:
        return P[keep].astype(np.float32), colors[keep]
    return P[keep].astype(np.float32), None


# ── Pipeline ─────────────────────────────────────────────────────────

def run_pipeline(job_id, image_dir, job_dir, scan_type='halfWrap', body_part='leg'):
    tag = job_id[:8]
    use_gpu = '1' if HAS_GPU else '0'

    # Resolution parameters — scale down in fast mode for quick test iterations
    _max_dim  = 800  if FAST_MODE else MAX_DIM   # COLMAP image size
    _tex_size = 1024 if FAST_MODE else 4096       # texture atlas resolution
    _p_depth  = 7    if FAST_MODE else 9          # Poisson depth (7≈fast, 9≈quality)
    _max_pts  = 20_000 if FAST_MODE else 60_000   # max pts fed to Poisson
    _tgt_tris = 15_000 if FAST_MODE else 50_000   # target triangle count after decimation
    # Determine UV/trimming strategy from scan_type sent by iOS app
    IS_PARTIAL_SCAN = scan_type != 'fullWrap'

    if FAST_MODE:
        logging.info(f'[{tag}] FAST_MODE: dim={_max_dim} tex={_tex_size} '
                     f'depth={_p_depth} maxpts={_max_pts} tris={_tgt_tris}')
    logging.info(f'[{tag}] scan_type={scan_type} body_part={body_part} → '
                 f'trim={75 if IS_PARTIAL_SCAN else 85}th pct '
                 f'(UV wrap/clip auto-detected from angular coverage)')

    try:
        db     = os.path.join(job_dir, 'db.db')
        sparse = os.path.join(job_dir, 'sparse')
        dense  = os.path.join(job_dir, 'dense')
        os.makedirs(sparse, exist_ok=True)
        os.makedirs(dense, exist_ok=True)

        def run(cmd, progress, msg):
            set_job(job_id, 'processing', progress, msg)
            logging.info(f'[{tag}] {msg}')
            t0 = time.time()
            # Run subprocess non-blocking so we can keep job timestamp fresh
            # Use separate thread to drain stderr to avoid pipe deadlock
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            stderr_lines = []
            def drain_stderr():
                for line in proc.stderr:
                    stderr_lines.append(line)
            drain_thread = threading.Thread(target=drain_stderr, daemon=True)
            drain_thread.start()
            while True:
                try:
                    proc.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    elapsed_so_far = time.time() - t0
                    set_job(job_id, 'processing', progress, f'{msg} ({int(elapsed_so_far)}s)')
                    if elapsed_so_far > 900:
                        proc.kill()
                        raise RuntimeError(f'{msg} timed out after 15 minutes')
            drain_thread.join(timeout=5)
            elapsed = time.time() - t0
            if proc.returncode != 0:
                err = ''.join(stderr_lines).strip() or '(no output)'
                if proc.returncode < 0:
                    sig = -proc.returncode
                    signames = {9: 'SIGKILL (out of memory)', 11: 'SIGSEGV (crash)', 6: 'SIGABRT (abort)'}
                    signame = signames.get(sig, f'signal {sig}')
                    err = f'Process killed by {signame}. {err}'
                logging.error(f'[{tag}] {msg} FAILED ({elapsed:.1f}s): {err[:500]}')
                raise RuntimeError(f'{msg} failed: {err[:800]}')
            logging.info(f'[{tag}] {msg} done ({elapsed:.1f}s)')

        # Verify images
        imgs = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        logging.info(f'[{tag}] Pipeline starting: {len(imgs)} images, GPU={HAS_GPU}')
        if not imgs:
            raise RuntimeError('No valid images found')

        # Downscale if needed
        set_job(job_id, 'processing', 0.03, 'Preparing images…')
        from PIL import Image
        for fname in imgs:
            fpath = os.path.join(image_dir, fname)
            try:
                with Image.open(fpath) as im:
                    w, h = im.size
                    if max(w, h) > _max_dim:
                        scale = _max_dim / max(w, h)
                        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                        im.save(fpath, quality=90)
            except Exception as e:
                logging.warning(f'[{tag}] Resize {fname}: {e}')

        # 0.5 — Foreground masks (rembg). Skin/body vs background.
        # Without this, background features (wall, floor, clothing) match
        # across photos and end up reconstructed as part of the mesh.
        masks_dir = os.path.join(job_dir, 'masks')
        set_job(job_id, 'processing', 0.05, 'Segmenting body from background…')
        try:
            n_masks = generate_foreground_masks(image_dir, masks_dir, tag)
        except Exception as _e:
            logging.warning(f'[{tag}] rembg masking failed: {_e}; continuing without masks')
            n_masks = 0

        # 1 — Feature extraction (COLMAP 4.x renamed options to FeatureExtraction.*)
        feat_cmd = ['colmap', 'feature_extractor',
                    '--database_path', db,
                    '--image_path', image_dir,
                    '--ImageReader.single_camera', '1',
                    '--FeatureExtraction.use_gpu', '0',
                    '--FeatureExtraction.max_image_size', str(_max_dim),
                    '--SiftExtraction.max_num_features', '8192']
        if n_masks > 0:
            feat_cmd += ['--ImageReader.mask_path', masks_dir]
        run(feat_cmd, 0.08, 'Extracting features…')

        # 2 — Matching (COLMAP 4.x uses FeatureMatching.* prefix)
        run(['colmap', 'exhaustive_matcher',
             '--database_path', db,
             '--FeatureMatching.use_gpu', '0'],
            0.18, 'Matching features…')

        # 3 — Sparse reconstruction
        run(['colmap', 'mapper',
             '--database_path', db,
             '--image_path', image_dir,
             '--output_path', sparse,
             '--Mapper.num_threads', '4',
             '--Mapper.max_num_models', '1'],
            0.30, 'Reconstructing structure…')

        sparse0 = os.path.join(sparse, '0')
        if not os.path.exists(sparse0):
            raise RuntimeError('Could not reconstruct — COLMAP could not register any cameras. Try again.')

        # Count how many cameras COLMAP actually registered by reading images.bin/txt.
        # If fewer than 8 succeeded we'll produce garbage — reject early with a
        # clear error rather than delivering a yellow inpaint blob.
        _img_bin = os.path.join(sparse0, 'images.bin')
        _img_txt = os.path.join(sparse0, 'images.txt')
        _n_registered = 0
        try:
            if os.path.exists(_img_bin):
                # binary format: first 8 bytes = uint64 count
                import struct as _struct
                with open(_img_bin, 'rb') as _f:
                    _n_registered = _struct.unpack('<Q', _f.read(8))[0]
            elif os.path.exists(_img_txt):
                with open(_img_txt) as _f:
                    _n_registered = sum(
                        1 for l in _f if l.strip() and not l.startswith('#')
                            and len(l.split()) >= 9
                    )
        except Exception as _e:
            logging.warning(f'[{tag}] Could not count registered cameras: {_e}')
        logging.info(f'[{tag}] COLMAP registered {_n_registered} cameras')
        if 0 < _n_registered < 8:
            raise RuntimeError(
                f'Only {_n_registered} of your photos could be matched '
                f'(need at least 8). Move more slowly and keep the leg in '
                f'frame throughout — do not change lighting mid-scan.'
            )

        # 4 — Undistort images
        run(['colmap', 'image_undistorter',
             '--image_path', image_dir,
             '--input_path', sparse0,
             '--output_path', dense,
             '--output_type', 'COLMAP',
             '--max_image_size', str(_max_dim)],
            0.38, 'Undistorting images…')

        # 4b — Convert undistorted sparse model to text format (COLMAP 3.9 defaults to binary)
        run(['colmap', 'model_converter',
             '--input_path', os.path.join(dense, 'sparse'),
             '--output_path', os.path.join(dense, 'sparse'),
             '--output_type', 'TXT'],
            0.40, 'Converting camera model to text…')

        # 5 — Dense stereo (GPU accelerated)
        run(['colmap', 'patch_match_stereo',
             '--workspace_path', dense,
             '--workspace_format', 'COLMAP',
             '--PatchMatchStereo.geom_consistency', 'true',
             '--PatchMatchStereo.max_image_size', str(_max_dim)],
            0.52, 'Computing depth maps (GPU)…')

        # 6 — Fuse into dense point cloud
        ply_path = os.path.join(job_dir, 'dense.ply')
        run(['colmap', 'stereo_fusion',
             '--workspace_path', dense,
             '--workspace_format', 'COLMAP',
             '--input_type', 'geometric',
             '--output_path', ply_path,
             '--StereoFusion.max_image_size', '1200',
             '--StereoFusion.min_num_pixels', '5'],
            0.62, 'Fusing point cloud…')

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2 — Point cloud → textured mesh (xatlas + vertex colors)
        # ═══════════════════════════════════════════════════════════════

        set_job(job_id, 'processing', 0.66, 'Loading point cloud…')
        import open3d as o3d
        import numpy as np
        import xatlas
        import cv2
        from scipy.spatial import cKDTree

        pcd = o3d.io.read_point_cloud(ply_path)
        pts = len(pcd.points)
        has_colors = pcd.has_colors()
        logging.info(f'[{tag}] Dense cloud: {pts} points, has_colors={has_colors}')
        if pts < 5000:
            raise RuntimeError(
                f'Dense reconstruction only produced {pts} points — '
                f'the photos could not be matched well enough. '
                f'Ensure even lighting and 60%+ overlap between frames.'
            )

        # ── Step 7: Clean point cloud ────────────────────────────────
        set_job(job_id, 'processing', 0.68, 'Cleaning point cloud…')

        # Statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        # Radius outlier removal (removes isolated clusters)
        pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        logging.info(f'[{tag}] After outlier removal: {len(pcd.points)} points')

        # Downsample for speed (preserves colors)
        if len(pcd.points) > 200000:
            pcd = pcd.voxel_down_sample(0.003)
            logging.info(f'[{tag}] After voxel downsample: {len(pcd.points)} points')

        # DBSCAN: remove isolated noise, keep all substantial clusters
        # Use larger eps to avoid over-fragmenting the limb surface
        nn_dists_arr = np.asarray(pcd.compute_nearest_neighbor_distance())
        eps = np.mean(nn_dists_arr) * 15  # mean*15 keeps limb together; median*5 was too small
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=10, print_progress=False))
        if labels.max() >= 0:
            counts = np.bincount(labels[labels >= 0])
            largest_count = counts.max()
            # Keep all clusters that are at least 5% the size of the largest
            # (limb may fragment slightly at edges — keep all meaningful pieces)
            keep_ids = np.where(counts >= max(50, largest_count * 0.05))[0]
            keep_mask = np.isin(labels, keep_ids)
            pcd = pcd.select_by_index(np.where(keep_mask)[0])
            logging.info(f'[{tag}] DBSCAN: kept {keep_mask.sum()} pts across {len(keep_ids)} clusters (of {len(counts)} total, eps={eps:.4f})')

        # ── RANSAC floor/plane removal ─────────────────────────────────
        # Removes the dominant flat surface (floor, table) from the point
        # cloud BEFORE Poisson. This is the correct place to separate the
        # body part from background — not at the mesh stage.
        # DBSCAN can't separate arm+table if they're touching (same cluster).
        # RANSAC finds and removes the largest flat plane regardless.
        try:
            _plane, _inliers = pcd.segment_plane(
                distance_threshold=0.01,   # 1 cm tolerance for plane membership
                ransac_n=3,
                num_iterations=1000
            )
            _pct = len(_inliers) / len(pcd.points)
            if 0.05 < _pct < 0.65:        # found a real plane (5-65% of points)
                pcd = pcd.select_by_index(_inliers, invert=True)
                logging.info(f'[{tag}] RANSAC: removed {len(_inliers)} floor/plane pts ({_pct*100:.1f}%)')
            else:
                logging.info(f'[{tag}] RANSAC: no dominant plane ({_pct*100:.1f}% inliers), skipping')
        except Exception as _e:
            logging.info(f'[{tag}] RANSAC plane removal skipped: {_e}')

        # ── Mask reprojection filter ──────────────────────────────────
        # For every surviving point, project into every COLMAP camera and
        # check the rembg mask. Drop points that land on background in
        # the majority of cameras that see them. This is the second line
        # of defence against background contamination — catches points
        # the per-image feature mask let through.
        if os.path.isdir(masks_dir):
            try:
                _pts_in  = np.asarray(pcd.points)
                _cols_in = np.asarray(pcd.colors) if pcd.has_colors() else None
                _pts_out, _cols_out = filter_pcd_by_masks(
                    _pts_in, _cols_in, masks_dir,
                    os.path.join(dense, 'sparse'), image_dir, tag)
                if len(_pts_out) > 1000:  # only swap if filter left enough points
                    pcd.points = o3d.utility.Vector3dVector(_pts_out)
                    if _cols_out is not None:
                        pcd.colors = o3d.utility.Vector3dVector(_cols_out)
                else:
                    logging.warning(f'[{tag}] Mask filter would leave only '
                                    f'{len(_pts_out)} pts — skipping')
            except Exception as _e:
                logging.warning(f'[{tag}] Mask reprojection filter failed: {_e}')

        # Save point cloud for later (point-cloud guided mesh trimming)
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        # ── Step 8: Surface reconstruction (Poisson + cleanup) ──────
        # Poisson creates a smooth continuous watertight surface.
        # The "back" of the limb (unscanned) gets hallucinated, but has very
        # low density — we remove it via aggressive density trimming.
        # This gives a clean continuous topology, equivalent to Blender's
        # Quadraflow remesh in the manual workflow.
        set_job(job_id, 'processing', 0.71, 'Estimating normals…')

        # Downsample to at most 60K points before Poisson.
        # Open3D Poisson is single-threaded; 120K+ points can take >10 min
        # and hit the gunicorn timeout. 60K is plenty of density for a limb
        # scan at Poisson depth=9 (voxel grid keeps point distribution even).
        MAX_POISSON_PTS = _max_pts
        n_pts = len(pcd.points)
        if n_pts > MAX_POISSON_PTS:
            pts_np = np.asarray(pcd.points)
            vox = ((pts_np.max(axis=0) - pts_np.min(axis=0)).max() /
                   (MAX_POISSON_PTS ** (1/3)))
            pcd = pcd.voxel_down_sample(voxel_size=float(vox))
            logging.info(f'[{tag}] Downsampled {n_pts} → {len(pcd.points)} pts '
                         f'(voxel={vox:.5f}m) for Poisson')

        # Estimate and orient normals (critical for Poisson quality)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        # Orient normals using the centroid as a reference (away from inside)
        # This works better than tangent plane for partial-wrap scans
        centroid = np.asarray(pcd.points).mean(axis=0)
        pts_arr2 = np.asarray(pcd.points)
        normals_arr = np.asarray(pcd.normals)
        # For a body part scan, normals should point AWAY from the scanned axis
        # Compute axis (longest dimension via SVD)
        centered = pts_arr2 - centroid
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, -1]  # principal axis (longest dim - the limb direction)
        # Project to plane perpendicular to axis to get radial outward direction
        radial = centered - (centered @ axis)[:, None] * axis
        radial_len = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-10
        radial_dir = radial / radial_len
        # Flip normals that point inward (toward centroid)
        flip_mask = np.einsum('ij,ij->i', normals_arr, radial_dir) < 0
        normals_arr[flip_mask] = -normals_arr[flip_mask]
        pcd.normals = o3d.utility.Vector3dVector(normals_arr)
        logging.info(f'[{tag}] Oriented {flip_mask.sum()}/{len(normals_arr)} normals outward')

        set_job(job_id, 'processing', 0.73, 'Creating surface (Poisson)…')

        # Poisson reconstruction — depth controls detail vs speed tradeoff
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=_p_depth, scale=1.1, linear_fit=False)
        densities = np.asarray(densities)
        logging.info(f'[{tag}] Poisson raw: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris')

        # Make all face normals consistently outward-facing.
        # Poisson can produce faces with inconsistent winding order — this causes
        # the "shattered mirror" look in Procreate because some faces render inside-out.
        mesh.orient_triangles()
        mesh.compute_vertex_normals()

        # ── Point-cloud guided trimming + AGGRESSIVE topology repair ───
        # CRITICAL: previous IQR cut at 3.0×IQR was drilling swiss-cheese
        # holes through the mesh, which then propagated to texture as the
        # white-outlined shattered pattern. New approach:
        #   1. Per-vertex distance to scan cloud (smoother than per-face)
        #   2. Soft percentile cut (95th) instead of hard IQR
        #   3. Morphological closing of triangle mask via vertex dilation
        #      to reseal small gaps before they become holes
        #   4. Aggressive multi-pass hole fill including large pass
        #   5. Drop tiny floating fragments BEFORE largest-component
        set_job(job_id, 'processing', 0.73, 'Removing background…')

        v_arr = np.asarray(mesh.vertices)
        f_arr = np.asarray(mesh.triangles)
        # Per-VERTEX distance is smoother than per-face → fewer holes
        v_dists, _ = cKDTree(pcd_points).query(v_arr)

        # Partial scans (front-only) have much more Poisson hallucination relative
        # to real data — use 75th percentile to aggressively cut the unscanned sides.
        # Full wraps have real data all around — 85th is safe.
        # Vertex-erosion below prevents over-cutting into the real scanned surface.
        _trim_pct = 75 if IS_PARTIAL_SCAN else 85
        cutoff = float(np.percentile(v_dists, _trim_pct))
        logging.info(f'[{tag}] Trim percentile: {_trim_pct}th ({"partial" if IS_PARTIAL_SCAN else "full"} scan)')
        # Floor it at 1cm so we never trim aggressively on very-clean scans
        cutoff = max(cutoff, 0.01)
        bad_v = v_dists > cutoff

        # Vertex-mask EROSION before trim: only kill a vertex if BOTH it AND
        # all its neighbours are far from scan data. This stops the trim from
        # nibbling holes into the limb surface where one stray Poisson vertex
        # extruded outward.
        from collections import defaultdict
        v_neighbors = defaultdict(set)
        for tri in f_arr:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            v_neighbors[a].update((b, c))
            v_neighbors[b].update((a, c))
            v_neighbors[c].update((a, b))
        bad_v_eroded = bad_v.copy()
        for vi in np.where(bad_v)[0]:
            if not all(bad_v[nv] for nv in v_neighbors[vi]):
                bad_v_eroded[vi] = False

        # Now mark faces bad only if ALL THREE vertices are bad (was ANY before)
        bad_face = bad_v_eroded[f_arr[:, 0]] & bad_v_eroded[f_arr[:, 1]] & bad_v_eroded[f_arr[:, 2]]
        mesh.remove_triangles_by_mask(bad_face)
        mesh.remove_unreferenced_vertices()
        logging.info(f'[{tag}] Trim: cutoff={cutoff*1000:.1f}mm, '
                     f'removed {int(bad_face.sum())}/{len(bad_face)} faces '
                     f'({bad_face.mean()*100:.1f}%), {len(mesh.triangles)} remain')

        # Smooth BEFORE fill so the fill triangulation lays into coherent normals.
        # More iterations for partial scans — the trimmed boundary is jagged
        # and needs extra smoothing to avoid saw-tooth edges.
        _smooth_iters = 30 if IS_PARTIAL_SCAN else 20
        mesh = mesh.filter_smooth_taubin(number_of_iterations=_smooth_iters)

        # Standard cleanup helper
        def _clean_mesh(m):
            m.remove_degenerate_triangles()
            m.remove_duplicated_triangles()
            m.remove_duplicated_vertices()
            m.remove_non_manifold_edges()
            try:
                m = m.merge_close_vertices(0.001)  # weld verts within 1 mm
            except Exception:
                pass
            return m

        # Drop tiny floating fragments BEFORE hole fill — fill_holes can't
        # close gaps that are bridged by stray fragments.
        tri_clusters, tri_counts, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        tri_counts   = np.asarray(tri_counts)
        if len(tri_counts) > 1:
            biggest = int(tri_counts.argmax())
            tiny = tri_counts < max(50, tri_counts[biggest] * 0.01)
            kill = np.isin(tri_clusters, np.where(tiny)[0])
            if kill.any():
                mesh.remove_triangles_by_mask(kill)
                mesh.remove_unreferenced_vertices()
                logging.info(f'[{tag}] Dropped {int(kill.sum())} tris in {int(tiny.sum())} tiny fragments')

        mesh = _clean_mesh(mesh)

        # fill_holes is absent from the open3d CUDA build. Use available ops:
        # aggressive manifold repair + vertex welding closes most small holes.
        for _weld in (0.005, 0.002, 0.001):
            try:
                mesh = mesh.merge_close_vertices(_weld)
            except Exception:
                pass
        mesh.remove_non_manifold_edges()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        mesh = _clean_mesh(mesh)

        # Keep only largest connected component (post-fill, post-cleanup)
        tri_clusters, tri_counts, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        tri_counts   = np.asarray(tri_counts)
        if len(tri_counts) > 1:
            keep = tri_clusters == tri_counts.argmax()
            mesh.remove_triangles_by_mask(~keep)
            mesh.remove_unreferenced_vertices()
            logging.info(f'[{tag}] Kept largest component: {len(mesh.triangles)} tris '
                         f'(of {len(tri_counts)} total)')

        # Second manifold repair pass after component selection
        mesh.remove_non_manifold_edges()
        try:
            mesh = mesh.merge_close_vertices(0.001)
        except Exception:
            pass
        mesh = _clean_mesh(mesh)

        target_tris = min(_tgt_tris, len(mesh.triangles))
        if len(mesh.triangles) > target_tris:
            mesh = mesh.simplify_quadric_decimation(target_tris)
        mesh = mesh.filter_smooth_taubin(number_of_iterations=15)
        mesh.compute_vertex_normals()

        # Topology gate: log Euler characteristic + boundary count.
        # χ = V - E + F. For a closed surface χ=2 (sphere). Lower means
        # the mesh has handles or holes. We just LOG it as a quality signal.
        try:
            _v = len(mesh.vertices)
            _f = len(mesh.triangles)
            # Count unique edges
            _tri_e = np.asarray(mesh.triangles)
            _edges = np.vstack([_tri_e[:, [0, 1]], _tri_e[:, [1, 2]], _tri_e[:, [0, 2]]])
            _edges.sort(axis=1)
            _e_count = len(np.unique(_edges, axis=0))
            _chi = _v - _e_count + _f
            logging.info(f'[{tag}] Topology: V={_v} E={_e_count} F={_f} χ={_chi} '
                         f'(closed sphere χ=2; one cylinder opening χ=1; two openings χ=0)')
        except Exception:
            pass

        verts = np.asarray(mesh.vertices).astype(np.float32)
        faces = np.asarray(mesh.triangles).astype(np.uint32)
        logging.info(f'[{tag}] Final mesh: {len(verts)} verts, {len(faces)} tris')
        if len(faces) < 100:
            raise RuntimeError('Mesh too small — try more overlapping photos.')

        # ── Step 9: Universal cylindrical UV unwrap ─────────────────────
        # One algorithm works for every scan type — fullWrap, halfWrap, cap,
        # patch, backPanel — by detecting actual angular coverage from the
        # scan data and choosing wrap-vs-clip dynamically:
        #
        #   1. PCA on mesh → principal "long" axis (limb axis).
        #   2. For each vertex compute (axial, angle, radius) cylindrical coords
        #      where angle = atan2 around the limb axis.
        #   3. Look at the point cloud's angular coverage. The largest empty
        #      angular gap is the "back" of the scan (unphotographed side).
        #          covered_arc = 2π − largest_gap.
        #      • covered_arc > 270° → wrap mode (true cylinder, seam in the
        #        empty gap, duplicate seam-crossing vertices).
        #      • covered_arc ≤ 270° → clip mode (open strip, no seam, just
        #        normalise the angular extent to [0,1]).
        #   4. v = position along limb axis, normalised to [0,1].
        #   5. u uses arc-length scaling (mean_radius × angle) so the unwrap
        #      is isometric — 1 cm of skin in u ≈ 1 cm of skin in v.
        #
        # This is the proven photogrammetry skin-peel parameterisation —
        # zero islands, minimal distortion, single seam (or no seam).
        set_job(job_id, 'processing', 0.76, 'UV unwrapping…')

        _cov_verts = verts - verts.mean(axis=0)
        try:
            _eigvals, _eigvecs = np.linalg.eigh(_cov_verts.T @ _cov_verts)
            _elongation = float(_eigvals[-1] / (_eigvals[-2] + 1e-10))
        except np.linalg.LinAlgError:
            logging.warning(f'[{tag}] PCA eigendecomposition failed — using identity axes')
            _elongation = 1.0
            _eigvecs = np.eye(3, dtype=np.float32)

        limb_axis = _eigvecs[:, -1]
        perp1     = _eigvecs[:, -2]
        perp2     = np.cross(limb_axis, perp1)
        perp2    /= np.linalg.norm(perp2) + 1e-10

        proj_along = _cov_verts @ limb_axis
        radial     = _cov_verts - proj_along[:, None] * limb_axis
        radial_x   = radial @ perp1
        radial_y   = radial @ perp2
        vert_angle = np.arctan2(radial_y, radial_x)
        vert_radius = np.hypot(radial_x, radial_y)

        # Angular coverage from the SCAN POINT CLOUD (not the mesh — the
        # mesh contains hallucinated Poisson surface that lies behind the
        # camera coverage). Sort the angles and find the largest cyclic gap.
        _pcd_centered = pcd_points - verts.mean(axis=0)
        _pcd_proj     = _pcd_centered @ limb_axis
        _pcd_radial   = _pcd_centered - _pcd_proj[:, None] * limb_axis
        _pcd_angles   = np.arctan2(_pcd_radial @ perp2, _pcd_radial @ perp1)
        _sorted = np.sort(_pcd_angles)
        _gaps   = np.diff(np.concatenate([_sorted, [_sorted[0] + 2 * np.pi]]))
        _gap_i  = int(np.argmax(_gaps))
        _max_gap     = float(_gaps[_gap_i])
        covered_arc  = 2 * np.pi - _max_gap
        gap_centre   = float(_sorted[_gap_i] + _max_gap / 2)
        if gap_centre > np.pi:
            gap_centre -= 2 * np.pi
        logging.info(f'[{tag}] Angular coverage: {np.degrees(covered_arc):.1f}°, '
                     f'largest gap centred at {np.degrees(gap_centre):.1f}°, '
                     f'PCA elongation {_elongation:.2f}')

        # iOS scan_type is treated as a hint, not a hard rule. If iOS said
        # halfWrap but coverage is 350°, we use wrap mode anyway.
        WRAP_THRESHOLD = np.radians(270)
        wrap_mode = covered_arc >= WRAP_THRESHOLD

        # Place the seam (or the cut for clip mode) at the centre of the
        # empty gap — that's the artist-invisible side of the body.
        # Re-centre angles so 0 = seam.
        rel_angle = ((vert_angle - gap_centre + np.pi) % (2 * np.pi)) - np.pi

        # ─── v coordinate (axial position) ──────────────────────────────
        v_min, v_max = float(proj_along.min()), float(proj_along.max())
        v_uv = (proj_along - v_min) / (v_max - v_min + 1e-10)

        # ─── u coordinate ───────────────────────────────────────────────
        verts_list = list(verts)
        u_list     = []
        v_list     = list(v_uv)
        faces_arr  = faces.astype(np.int64).copy()
        shifted_v  = {}

        if wrap_mode:
            # u ∈ [0,1) maps to angles seam→seam going around the body.
            # Shift so seam is at u=0 and u=1 (cyclic).
            u_uv = (rel_angle + np.pi) / (2 * np.pi)   # [0,1)
            u_list = list(u_uv)
            # Duplicate seam-crossing vertices: triangles whose vertex u-values
            # span > 0.5 (i.e. straddle the seam) get their low-u vertices
            # remapped to u' = u + 1.0 so the triangle rasterises continuously
            # past the right edge of the texture. The bake function wraps
            # texel writes via modulo so it's still a single tileable chart.
            u_per_face = u_uv[faces]
            crosses = (u_per_face.max(axis=1) - u_per_face.min(axis=1)) > 0.5
            for fi in np.where(crosses)[0]:
                tri = faces[fi]
                for i in range(3):
                    v_idx = int(tri[i])
                    if u_uv[v_idx] < 0.5:
                        nidx = shifted_v.get(v_idx)
                        if nidx is None:
                            nidx = len(verts_list)
                            verts_list.append(verts[v_idx])
                            u_list.append(float(u_uv[v_idx]) + 1.0)
                            v_list.append(float(v_uv[v_idx]))
                            shifted_v[v_idx] = nidx
                        faces_arr[fi, i] = nidx
        else:
            # Clip mode: rel_angle is naturally in [-arc/2, +arc/2] for
            # vertices near the front of the body; vertices on the trimmed
            # back will be clamped to that range. Normalise the actual
            # vertex angular extent to [0,1].
            a_lo, a_hi = float(rel_angle.min()), float(rel_angle.max())
            u_uv = (rel_angle - a_lo) / (a_hi - a_lo + 1e-10)
            u_list = list(u_uv)

        new_verts = np.asarray(verts_list, dtype=np.float32)
        new_faces = faces_arr.astype(np.uint32)
        uvs = np.stack(
            [np.asarray(u_list, dtype=np.float32),
             np.asarray(v_list, dtype=np.float32)],
            axis=1
        )

        # Physical dimensions of the unwrapped chart (for 1:1 print scaling).
        # Mesh is still in COLMAP units; final scale-normalise happens later.
        mean_radius = float(np.median(vert_radius)) if len(vert_radius) else 0.0
        if wrap_mode:
            chart_u_units = 2 * np.pi * mean_radius
        else:
            chart_u_units = (float(rel_angle.max()) - float(rel_angle.min())) * mean_radius
        chart_v_units = float(v_max - v_min)
        # Stash for scale.json
        _uv_meta = dict(
            wrap_mode=bool(wrap_mode),
            covered_arc_deg=float(np.degrees(covered_arc)),
            mean_radius_units=mean_radius,
            chart_u_units=chart_u_units,
            chart_v_units=chart_v_units,
            seam_angle_deg=float(np.degrees(gap_centre)) if wrap_mode else None,
        )

        logging.info(f'[{tag}] UV {"WRAP" if wrap_mode else "CLIP"}: '
                     f'u=[{uvs[:,0].min():.3f},{uvs[:,0].max():.3f}] '
                     f'v=[{uvs[:,1].min():.3f},{uvs[:,1].max():.3f}] '
                     f'duplicated={len(shifted_v)} verts, '
                     f'aspect u/v={chart_u_units/(chart_v_units+1e-9):.2f}')

        # ── Step 10: Bake texture from photos using COLMAP camera poses ──
        # This projects the original photos onto the UV map — same as Blender's
        # "Selected to Active" bake. Gives photographic-quality textures.
        set_job(job_id, 'processing', 0.82, 'Baking texture from photos…')
        TEX_SIZE = _tex_size
        dense_sparse = os.path.join(dense, 'sparse')
        undistorted_images = os.path.join(dense, 'images')

        texture = bake_texture_from_photos(
            new_verts, new_faces, uvs,
            undistorted_images, dense_sparse,
            TEX_SIZE, tag,
            wrap_mode=wrap_mode,
        )

        # Fallback: if photo bake failed, use vertex colors from point cloud
        if texture is None:
            logging.warning(f'[{tag}] Photo bake failed, falling back to vertex colors')
            set_job(job_id, 'processing', 0.85, 'Baking texture (fallback)…')
            if pcd_colors is not None and len(pcd_colors) > 0:
                tree = cKDTree(pcd_points)
                _, idx = tree.query(new_verts, k=3)
                new_colors = (pcd_colors[idx].mean(axis=1) * 255).clip(0, 255).astype(np.uint8)
            else:
                new_colors = np.full((len(new_verts), 3), 180, dtype=np.uint8)
            texture = bake_vertex_colors_to_texture(new_verts, new_faces, uvs, new_colors, TEX_SIZE, tag)

        # ── Step 12: Normalize scale + export ────────────────────────
        set_job(job_id, 'processing', 0.92, 'Packaging result…')

        # Per body part, pick a sensible default longest-axis size when
        # there's no calibration object. iOS/Procreate can rescale later
        # using scale.json once a known reference is available.
        BODY_PART_LONGEST_M = {
            'leg':       0.55,
            'arm':       0.45,
            'forearm':   0.30,
            'hand':      0.20,
            'shoulder':  0.30,
            'back':      0.55,
            'chest':     0.45,
            'thigh':     0.50,
        }
        target_long = BODY_PART_LONGEST_M.get(body_part, 0.40)
        bbox = new_verts.max(axis=0) - new_verts.min(axis=0)
        max_dim = float(bbox.max())
        if max_dim > 0:
            scale = target_long / max_dim
            new_verts = new_verts * scale
            logging.info(f'[{tag}] Normalised: longest axis → {target_long} m '
                         f'(scale={scale:.4f})')
        else:
            scale = 1.0

        result_dir = os.path.join(job_dir, 'result')
        os.makedirs(result_dir, exist_ok=True)

        tex_path = os.path.join(result_dir, 'mesh.png')
        texture.save(tex_path)

        obj_path = os.path.join(result_dir, 'mesh.obj')
        mtl_path = os.path.join(result_dir, 'mesh.mtl')
        write_obj(new_verts, new_faces, uvs, obj_path, mtl_path)

        # scale.json — physical metadata for the iOS app and Procreate prints.
        # All units in metres unless noted. Lets the artist export a 1:1
        # printable stencil from the texture without guessing the size.
        scale_path = os.path.join(result_dir, 'scale.json')
        bbox_after = (new_verts.max(axis=0) - new_verts.min(axis=0)).tolist()
        scale_meta = dict(
            scan_type=scan_type,
            body_part=body_part,
            target_longest_m=target_long,
            applied_scale=float(scale),
            mesh_bbox_m=bbox_after,
            uv=dict(
                wrap_mode=_uv_meta['wrap_mode'],
                covered_arc_deg=_uv_meta['covered_arc_deg'],
                seam_angle_deg=_uv_meta['seam_angle_deg'],
                # Physical chart dimensions, scaled to metres (texture aspect
                # the artist should print at: chart_u_m × chart_v_m).
                chart_u_m=_uv_meta['chart_u_units'] * scale,
                chart_v_m=_uv_meta['chart_v_units'] * scale,
            ),
            texture_size=TEX_SIZE,
            verts=int(len(new_verts)),
            tris=int(len(new_faces)),
        )
        with open(scale_path, 'w') as _sf:
            json.dump(scale_meta, _sf, indent=2)

        zip_path = os.path.join(job_dir, f'{job_id}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            zf.write(obj_path, 'mesh.obj')
            zf.write(mtl_path, 'mesh.mtl')
            zf.write(tex_path, 'mesh.png')
            zf.write(scale_path, 'scale.json')

        result_size = os.path.getsize(zip_path)

        # ── Pipeline quality validation ─────────────────────────────────
        tri_cl, tri_ct, _ = mesh.cluster_connected_triangles()
        n_components = len(np.asarray(tri_ct))
        uv_label = ('cylindrical wrap (1 island, with seam)' if wrap_mode
                    else 'cylindrical clip (1 island, no seam)')
        logging.info(f'[{tag}] ── Pipeline quality summary ─────────────')
        logging.info(f'[{tag}]   Mesh components : {n_components} (want 1)')
        logging.info(f'[{tag}]   UV method       : {uv_label}')
        logging.info(f'[{tag}]   Verts / tris    : {len(new_verts)} / {len(new_faces)}')
        logging.info(f'[{tag}]   Texture size    : {TEX_SIZE}x{TEX_SIZE}')
        logging.info(f'[{tag}]   Output zip      : {result_size} bytes')
        if n_components > 1:
            logging.warning(f'[{tag}]   !! Mesh has {n_components} components — '
                            f'check RANSAC/IQR trim settings')
        logging.info(f'[{tag}] ─────────────────────────────────────────')

        logging.info(f'[{tag}] Done! {len(mesh.vertices)} verts, {len(mesh.triangles)} tris, zip={result_size} bytes')
        set_job(job_id, 'done', 1.0, 'Reconstruction complete!',
                result_url=f'/result/{job_id}')

    except Exception as e:
        logging.error(f'[{tag}] Pipeline error: {traceback.format_exc()}')
        set_job(job_id, 'error', 0, str(e), error=str(e))
    finally:
        for d in [image_dir, sparse, dense, os.path.join(job_dir, 'result')]:
            shutil.rmtree(d, ignore_errors=True)
        for f in [db, os.path.join(job_dir, 'dense.ply')]:
            try: os.remove(f)
            except: pass


# ── Photo-based texture baking using COLMAP camera poses ─────────────

def _read_colmap_cameras_txt(path):
    """Parse cameras.txt → dict of camera_id → intrinsics."""
    cameras = {}
    if not os.path.exists(path):
        return cameras
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            if len(p) < 5:
                continue
            cam_id = int(p[0])
            model = p[1]
            w, h = int(p[2]), int(p[3])
            params = [float(x) for x in p[4:]]
            # Map all COLMAP camera models to fx,fy,cx,cy
            if model == 'SIMPLE_PINHOLE':
                fx = fy = params[0]; cx = params[1]; cy = params[2]
            elif model == 'PINHOLE':
                fx = params[0]; fy = params[1]; cx = params[2]; cy = params[3]
            elif model in ('SIMPLE_RADIAL', 'RADIAL'):
                fx = fy = params[0]; cx = params[1]; cy = params[2]
            elif model in ('OPENCV', 'FULL_OPENCV', 'OPENCV_FISHEYE'):
                fx = params[0]; fy = params[1]; cx = params[2]; cy = params[3]
            else:
                fx = fy = params[0]
                cx = params[1] if len(params) > 1 else w / 2
                cy = params[2] if len(params) > 2 else h / 2
            cameras[cam_id] = dict(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h)
    return cameras


def _read_colmap_images_txt(path):
    """Parse images.txt → list of dicts with R, t, camera_id, name."""
    import numpy as np  # needed: this is a module-level function
    images = []
    if not os.path.exists(path):
        return images
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    i = 0
    while i < len(lines):
        p = lines[i].split()
        if len(p) < 9:
            i += 1
            continue
        qw, qx, qy, qz = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        tx, ty, tz = float(p[5]), float(p[6]), float(p[7])
        camera_id = int(p[8])
        name = p[9] if len(p) > 9 else ''
        # Quaternion → rotation matrix
        R = np.array([
            [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
            [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
            [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
        ], dtype=np.float64)
        images.append(dict(R=R, t=np.array([tx, ty, tz]), camera_id=camera_id, name=name))
        i += 2  # skip the 2D point correspondences line
    return images


def bake_texture_from_photos(verts, faces, uvs, image_dir, sparse_dir, tex_size, tag,
                             wrap_mode=False):
    """
    Project original photos onto the UV texture using COLMAP camera poses.

    Per-texel best-camera selection with three quality features:

      1. Occlusion testing — for every (texel, camera), cast a ray from
         the texel along its surface normal toward the camera centre.
         If the mesh occludes the camera before the ray reaches it, the
         texel is invisible from that camera (e.g. front-of-leg texel
         being scored by a back-of-leg photo).
      2. Top-K weighted blending — instead of argmax(dot) per texel
         (which switches camera abruptly at neighbouring texels and
         produces visible "shards"), we keep the top-3 most face-on
         cameras per texel and weighted-average them. Same approach as
         RealityCapture's blended texturing.
      3. Per-camera colour normalisation — each camera's skin pixels
         are matched (mean+std) to a reference camera so blends are
         seamless across exposure differences.

    Three passes:
      Pass 1 — geometry: rasterise UV → 3D position + face normal map.
      Pass 2 — visibility: for each camera, project all texels, occlusion
               test, score; maintain top-K cameras per texel.
      Pass 3 — sampling: load each image once, colour-normalise, sample
               at the K stored pixel coords for every texel using it,
               weighted accumulate into the result.

    `wrap_mode=True` means the UV is a true cylindrical wrap with seam
    vertices duplicated at u > 1; we modulo-wrap pixel writes so the
    seam is tileable. `wrap_mode=False` (clip mode) does not wrap.
    """
    import cv2
    import numpy as np
    import open3d as o3d
    from PIL import Image

    TOP_K = 3

    try:
        cameras = _read_colmap_cameras_txt(os.path.join(sparse_dir, 'cameras.txt'))
        images  = _read_colmap_images_txt(os.path.join(sparse_dir, 'images.txt'))
        if not cameras or not images:
            logging.warning(f'[{tag}] No COLMAP cameras/images found — skipping photo bake')
            return None

        cam_list = []
        for img_info in images:
            img_path = os.path.join(image_dir, img_info['name'])
            if not os.path.exists(img_path):
                continue
            cam = cameras.get(img_info['camera_id'])
            if cam is None:
                continue
            cam_list.append({**img_info, 'path': img_path, 'cam': cam})
        logging.info(f'[{tag}] Photo bake: {len(cam_list)} valid images out of {len(images)}')
        if not cam_list:
            return None

        verts_np = np.array(verts, dtype=np.float64)
        faces_np = np.array(faces, dtype=np.int32)
        uvs_np   = np.array(uvs,   dtype=np.float32)

        # ── Pass 1 — UV → 3D position + face normal ────────────────────
        pos_map    = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
        normal_map = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
        has_data   = np.zeros((tex_size, tex_size),    dtype=bool)

        v0_all = verts_np[faces_np[:, 0]]
        v1_all = verts_np[faces_np[:, 1]]
        v2_all = verts_np[faces_np[:, 2]]

        for fi in range(len(faces_np)):
            v0, v1, v2 = v0_all[fi], v1_all[fi], v2_all[fi]
            uv0 = uvs_np[faces_np[fi, 0]]
            uv1 = uvs_np[faces_np[fi, 1]]
            uv2 = uvs_np[faces_np[fi, 2]]

            # Wrap mode safety net: post-duplication, no triangle should
            # span > 0.5 in u. If one does, skip it to avoid wrap smear.
            if wrap_mode:
                u_span = max(uv0[0], uv1[0], uv2[0]) - min(uv0[0], uv1[0], uv2[0])
                if u_span > 0.5:
                    continue

            n = np.cross(v1 - v0, v2 - v0)
            n_len = np.linalg.norm(n)
            if n_len < 1e-10:
                continue
            n = n / n_len

            px = np.array([
                [uv0[0] * tex_size, (1 - uv0[1]) * tex_size],
                [uv1[0] * tex_size, (1 - uv1[1]) * tex_size],
                [uv2[0] * tex_size, (1 - uv2[1]) * tex_size],
            ], dtype=np.float32)

            min_u = max(0, int(px[:, 0].min()) - 1)
            max_u = min(tex_size - 1, int(px[:, 0].max()) + 1)
            min_v = max(0, int(px[:, 1].min()) - 1)
            max_v = min(tex_size - 1, int(px[:, 1].max()) + 1)
            if max_u <= min_u or max_v <= min_v:
                continue

            us = np.arange(min_u, max_u + 1, dtype=np.float32) + 0.5
            vs = np.arange(min_v, max_v + 1, dtype=np.float32) + 0.5
            uu, vv = np.meshgrid(us, vs)

            p0, p1, p2 = px[0], px[1], px[2]
            d_x = uu - p0[0]; d_y = vv - p0[1]
            d1_x = p1[0]-p0[0]; d1_y = p1[1]-p0[1]
            d2_x = p2[0]-p0[0]; d2_y = p2[1]-p0[1]
            denom = d1_x*d2_y - d1_y*d2_x
            if abs(denom) < 1e-6:
                continue

            w1 = (d_x*d2_y - d_y*d2_x) / denom
            w2 = (d1_x*d_y - d1_y*d_x) / denom
            w0 = 1.0 - w1 - w2
            inside = (w0 >= -0.01) & (w1 >= -0.01) & (w2 >= -0.01)
            if not inside.any():
                continue

            w0c = np.clip(w0, 0, 1); w1c = np.clip(w1, 0, 1); w2c = np.clip(w2, 0, 1)
            ws = w0c + w1c + w2c + 1e-10
            w0c /= ws; w1c /= ws; w2c /= ws

            pos3d = (w0c[..., None]*v0 + w1c[..., None]*v1 + w2c[..., None]*v2).astype(np.float32)
            tv = np.clip((vv - 0.5).astype(int), 0, tex_size-1)
            if wrap_mode:
                tu = np.mod((uu - 0.5).astype(int), tex_size)
            else:
                tu = np.clip((uu - 0.5).astype(int), 0, tex_size-1)

            pos_map   [tv[inside], tu[inside]] = pos3d[inside]
            normal_map[tv[inside], tu[inside]] = n.astype(np.float32)
            has_data  [tv[inside], tu[inside]] = True

        valid_tv, valid_tu = np.where(has_data)
        if len(valid_tv) == 0:
            logging.warning(f'[{tag}] UV position map empty')
            return None

        valid_pos = pos_map[valid_tv, valid_tu].astype(np.float64)
        valid_n   = normal_map[valid_tv, valid_tu].astype(np.float64)
        # Normalise the rasterised face-normals (they're unit by construction
        # but rasterisation may have written a degenerate face's all-zeros).
        n_lens   = np.linalg.norm(valid_n, axis=1)
        good_n   = n_lens > 1e-6
        valid_n[good_n] /= n_lens[good_n, None]
        M = len(valid_tv)
        logging.info(f'[{tag}] UV rasterised: {M} texels')

        # ── Build raycaster from the final mesh for occlusion testing ──
        # An open3d t.geometry RaycastingScene is C++/embree backed and
        # accepts batched ray casts for fast per-camera occlusion checks.
        try:
            mesh_t = o3d.t.geometry.TriangleMesh()
            mesh_t.vertex.positions = o3d.core.Tensor(verts_np.astype(np.float32))
            mesh_t.triangle.indices = o3d.core.Tensor(faces_np.astype(np.uint32))
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(mesh_t)
            occlusion_ok = True
        except Exception as _e:
            logging.warning(f'[{tag}] RaycastingScene init failed: {_e} — '
                            f'continuing without occlusion testing')
            scene = None
            occlusion_ok = False

        # ── Pass 2 — per camera: project, occlusion-test, top-K update ──
        cam_texel_count = np.zeros(len(cam_list), dtype=np.int64)
        topk_score = np.full((M, TOP_K), -np.inf, dtype=np.float32)
        topk_cam   = np.full((M, TOP_K), -1, dtype=np.int32)
        topk_ix    = np.zeros((M, TOP_K), dtype=np.float32)
        topk_iy    = np.zeros((M, TOP_K), dtype=np.float32)

        # Use a small offset along the surface normal so rays start just
        # above the surface and don't self-intersect.
        ray_origin = (valid_pos + valid_n * 5e-4).astype(np.float32)

        for ci, c in enumerate(cam_list):
            R, t    = c['R'], c['t']
            cam     = c['cam']
            cam_pos = -R.T @ t

            pts_cam = (R @ valid_pos.T).T + t
            in_front = pts_cam[:, 2] > 0.01

            fx, fy, cx_c, cy_c = cam['fx'], cam['fy'], cam['cx'], cam['cy']
            iw, ih = cam['w'], cam['h']
            iz    = np.where(in_front, 1.0 / (pts_cam[:, 2] + 1e-10), 0.0)
            img_x = fx * pts_cam[:, 0] * iz + cx_c
            img_y = fy * pts_cam[:, 1] * iz + cy_c

            in_bounds = (img_x >= 0) & (img_x < iw - 1) & \
                        (img_y >= 0) & (img_y < ih - 1)

            view = cam_pos - valid_pos
            view_len = np.linalg.norm(view, axis=1)
            view_dir = view / (view_len[:, None] + 1e-10)
            dot = np.einsum('ij,ij->i', valid_n, view_dir)

            visible = in_front & in_bounds & (dot > 0.15)

            # Occlusion test: cast rays from texel toward camera; first-hit
            # distance must be at least the texel→camera distance (i.e. the
            # ray exits the mesh before reaching the camera). 2 mm slack.
            if occlusion_ok and visible.any():
                idx_v = np.where(visible)[0]
                rays = np.empty((len(idx_v), 6), dtype=np.float32)
                rays[:, :3] = ray_origin[idx_v]
                rays[:, 3:] = view_dir[idx_v].astype(np.float32)
                try:
                    hits = scene.cast_rays(o3d.core.Tensor(rays))
                    t_hit = hits['t_hit'].numpy()
                    occluded = t_hit < (view_len[idx_v] - 0.002)
                    visible[idx_v[occluded]] = False
                except Exception as _e:
                    logging.warning(f'[{tag}] cast_rays failed for {c["name"]}: {_e}')

            cam_texel_count[ci] = int(visible.sum())
            if not visible.any():
                continue

            # Top-K update — for each visible texel, replace the worst slot
            # if this camera scores higher.
            score = dot.astype(np.float32)
            min_slot = np.argmin(topk_score, axis=1)                # (M,)
            min_val  = topk_score[np.arange(M), min_slot]           # (M,)
            update   = visible & (score > min_val)
            if update.any():
                ui = np.where(update)[0]
                slots = min_slot[ui]
                topk_score[ui, slots] = score[ui]
                topk_cam  [ui, slots] = ci
                topk_ix   [ui, slots] = img_x[ui].astype(np.float32)
                topk_iy   [ui, slots] = img_y[ui].astype(np.float32)

        any_cam = (topk_cam >= 0).any(axis=1)
        claimed = int(any_cam.sum())
        logging.info(f'[{tag}] Per-texel top-{TOP_K}: {claimed}/{M} texels '
                     f'({claimed/M*100:.1f}%) covered after occlusion test')
        if claimed == 0:
            return None

        # ── Pass 3 — colour-normalise, sample, weighted blend ──────────
        ref_ci  = int(cam_texel_count.argmax())
        ref_img = cv2.cvtColor(cv2.imread(cam_list[ref_ci]['path']),
                               cv2.COLOR_BGR2RGB).astype(np.float32)
        ref_flat = ref_img.reshape(-1, 3)
        skin_mask = (ref_flat.max(axis=1) > 20) & (ref_flat.max(axis=1) < 250)
        if skin_mask.sum() > 100:
            ref_mean = ref_flat[skin_mask].mean(axis=0)
            ref_std  = ref_flat[skin_mask].std(axis=0) + 1e-6
        else:
            ref_mean = ref_flat.mean(axis=0)
            ref_std  = ref_flat.std(axis=0) + 1e-6
        logging.info(f'[{tag}] Colour ref: {cam_list[ref_ci]["name"]} '
                     f'mean={ref_mean.round(1)}')

        # Soft top-K weighting: weight ∝ score - min_top_k_score, so the
        # most face-on camera dominates but the others still contribute.
        valid_slot = topk_cam >= 0
        scores = np.where(valid_slot, topk_score, -np.inf).astype(np.float32)
        # Shift scores so the lowest valid one is 0 → makes weights non-negative.
        row_min = np.where(valid_slot, scores, np.inf).min(axis=1, keepdims=True)
        row_min = np.where(np.isfinite(row_min), row_min, 0)
        shifted = np.where(valid_slot, scores - row_min + 1e-3, 0)
        wsum    = shifted.sum(axis=1, keepdims=True)
        weights = np.where(wsum > 0, shifted / wsum, 0).astype(np.float32)

        result_rgb = np.zeros((M, 3), dtype=np.float32)

        for ci, c in enumerate(cam_list):
            slots = (topk_cam == ci)
            if not slots.any():
                continue

            img_bgr = cv2.imread(c['path'])
            if img_bgr is None:
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

            flat = img.reshape(-1, 3)
            sk   = (flat.max(axis=1) > 20) & (flat.max(axis=1) < 250)
            if sk.sum() > 100:
                c_mean = flat[sk].mean(axis=0)
                c_std  = flat[sk].std(axis=0) + 1e-6
            else:
                c_mean = flat.mean(axis=0)
                c_std  = flat.std(axis=0) + 1e-6
            img_norm = ((img - c_mean) / c_std * ref_std + ref_mean).clip(0, 255)
            ih_i, iw_i = img_norm.shape[:2]

            # For each slot k, gather the texels that selected this camera
            # at slot k, sample (bilinear) at the stored pixel coord, and
            # add weighted contribution.
            for k in range(TOP_K):
                mask_ck = slots[:, k]
                if not mask_ck.any():
                    continue
                sx = topk_ix[mask_ck, k].astype(np.float64)
                sy = topk_iy[mask_ck, k].astype(np.float64)
                x0 = np.clip(sx.astype(int), 0, iw_i - 2)
                y0 = np.clip(sy.astype(int), 0, ih_i - 2)
                xf = sx - x0; yf = sy - y0
                sampled = (img_norm[y0,   x0  ] * ((1-xf)*(1-yf))[:, None] +
                           img_norm[y0,   x0+1] * (   xf *(1-yf))[:, None] +
                           img_norm[y0+1, x0  ] * ((1-xf)*   yf )[:, None] +
                           img_norm[y0+1, x0+1] * (   xf *   yf )[:, None])
                w_k = weights[mask_ck, k:k+1]
                result_rgb[mask_ck] += sampled.astype(np.float32) * w_k / 255.0

            logging.info(f'[{tag}] {c["name"]}: '
                         f'{int(slots.any(axis=1).sum())} texels '
                         f'(mean→ref shift {(c_mean-ref_mean).round(0).tolist()})')

        result_rgb = np.clip(result_rgb, 0, 1)
        result = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
        result[valid_tv[any_cam], valid_tu[any_cam]] = \
            (result_rgb[any_cam] * 255).astype(np.uint8)

        # Bilateral filter on the covered region — kills micro colour
        # boundaries between top-K transitions while keeping tattoo edges.
        try:
            sm = cv2.bilateralFilter(result, d=7, sigmaColor=18, sigmaSpace=7)
            covered_mask = np.zeros((tex_size, tex_size), dtype=bool)
            covered_mask[valid_tv[any_cam], valid_tu[any_cam]] = True
            result[covered_mask] = sm[covered_mask]
        except Exception:
            pass

        # Inpaint uncovered region with TELEA (propagates real skin colour
        # from the edge of the photographic patch via fast marching).
        covered = np.zeros((tex_size, tex_size), dtype=np.uint8)
        covered[valid_tv[any_cam], valid_tu[any_cam]] = 255
        uncovered = 255 - covered
        if uncovered.any():
            result = cv2.inpaint(result, uncovered, 25, cv2.INPAINT_TELEA)
            try:
                soft = cv2.GaussianBlur(result, (0, 0), 1.5)
                inpaint_only = uncovered > 0
                result[inpaint_only] = soft[inpaint_only]
            except Exception:
                pass

        pct = claimed / M * 100
        logging.info(f'[{tag}] ── Bake quality ────────────────────────')
        logging.info(f'[{tag}]   UV mode       : {"wrap (cylinder + seam)" if wrap_mode else "clip (open strip)"}')
        logging.info(f'[{tag}]   Texels covered: {pct:.1f}%')
        logging.info(f'[{tag}]   Occlusion     : {"ON" if occlusion_ok else "OFF"}')
        logging.info(f'[{tag}]   Top-K blend   : {TOP_K} cameras/texel weighted')
        if pct < 50:
            logging.warning(f'[{tag}]   !! Low coverage — check mesh/camera poses')
        logging.info(f'[{tag}] ────────────────────────────────────────')

        return Image.fromarray(result)

    except Exception:
        logging.error(f'[{tag}] Photo bake error: {traceback.format_exc()}')
        return None


# ── Texture baking (vertex colors → UV texture via rasterization) ────

def bake_vertex_colors_to_texture(verts, faces, uvs, colors, tex_size, tag):
    """
    Rasterize per-vertex colors into a UV texture map.
    Uses OpenCV for scanline triangle rasterization (fast) + inpainting (fills gaps).
    """
    import numpy as np
    import cv2
    from PIL import Image

    try:
        tex = np.full((tex_size, tex_size, 3), 0, dtype=np.uint8)
        mask = np.zeros((tex_size, tex_size), dtype=np.uint8)

        # Convert UVs to pixel coords (xatlas UVs are already 0-1 normalized)
        uv_px = uvs.copy()
        uv_px[:, 0] = (uvs[:, 0] * (tex_size - 1)).clip(0, tex_size - 1)
        uv_px[:, 1] = ((1 - uvs[:, 1]) * (tex_size - 1)).clip(0, tex_size - 1)
        uv_px = uv_px.astype(np.int32)

        logging.info(f'[{tag}] Rasterizing {len(faces)} triangles via OpenCV fillPoly...')

        # Rasterize each triangle with interpolated vertex colors
        for tri in faces:
            v0, v1, v2 = tri
            p0 = uv_px[v0]; p1 = uv_px[v1]; p2 = uv_px[v2]
            c0 = colors[v0].astype(np.float32)
            c1 = colors[v1].astype(np.float32)
            c2 = colors[v2].astype(np.float32)

            # Bounding box
            min_x = max(0, min(p0[0], p1[0], p2[0]))
            max_x = min(tex_size-1, max(p0[0], p1[0], p2[0]))
            min_y = max(0, min(p0[1], p1[1], p2[1]))
            max_y = min(tex_size-1, max(p0[1], p1[1], p2[1]))

            # Skip degenerate
            if max_x - min_x < 1 and max_y - min_y < 1:
                continue

            # Use OpenCV to create triangle mask for this region
            pts = np.array([[p0[0], p0[1]], [p1[0], p1[1]], [p2[0], p2[1]]], dtype=np.int32)

            # For small triangles, just fill with average color
            area = abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
            if area < 4:
                avg_color = ((c0 + c1 + c2) / 3).clip(0, 255).astype(np.uint8)
                cv2.fillConvexPoly(tex, pts, avg_color.tolist())
                cv2.fillConvexPoly(mask, pts, 255)
                continue

            # For larger triangles, use barycentric interpolation
            # Create sub-image for this triangle's bbox
            h = max_y - min_y + 1
            w = max_x - min_x + 1
            if h > 500 or w > 500:
                continue  # skip anomalous

            # Shift points to local coords
            lp0 = [p0[0]-min_x, p0[1]-min_y]
            lp1 = [p1[0]-min_x, p1[1]-min_y]
            lp2 = [p2[0]-min_x, p2[1]-min_y]

            # Create meshgrid for barycentric
            ys, xs = np.mgrid[0:h, 0:w]
            denom = (lp1[1]-lp2[1])*(lp0[0]-lp2[0]) + (lp2[0]-lp1[0])*(lp0[1]-lp2[1])
            if abs(denom) < 1e-6:
                avg_color = ((c0 + c1 + c2) / 3).clip(0, 255).astype(np.uint8)
                cv2.fillConvexPoly(tex, pts, avg_color.tolist())
                cv2.fillConvexPoly(mask, pts, 255)
                continue

            w0 = ((lp1[1]-lp2[1])*(xs-lp2[0]) + (lp2[0]-lp1[0])*(ys-lp2[1])) / denom
            w1 = ((lp2[1]-lp0[1])*(xs-lp2[0]) + (lp0[0]-lp2[0])*(ys-lp2[1])) / denom
            w2 = 1.0 - w0 - w1

            inside = (w0 >= -0.01) & (w1 >= -0.01) & (w2 >= -0.01)

            if inside.any():
                for ch in range(3):
                    color_ch = (w0 * c0[ch] + w1 * c1[ch] + w2 * c2[ch]).clip(0, 255)
                    tex[min_y:max_y+1, min_x:max_x+1, ch][inside] = color_ch[inside].astype(np.uint8)
                mask[min_y:max_y+1, min_x:max_x+1][inside] = 255

        painted_pct = (mask > 0).sum() / mask.size * 100
        logging.info(f'[{tag}] Rasterized: {painted_pct:.1f}% painted')

        # Inpaint unpainted regions using OpenCV TELEA algorithm
        if painted_pct < 99:
            inpaint_mask = (255 - mask).astype(np.uint8)
            tex = cv2.inpaint(tex, inpaint_mask, 8, cv2.INPAINT_TELEA)
            logging.info(f'[{tag}] Inpainted gaps')

        # Convert BGR→RGB if needed (OpenCV uses BGR)
        # Actually our array is already RGB, so just save
        return Image.fromarray(tex)

    except Exception as e:
        logging.error(f'[{tag}] Texture bake failed: {traceback.format_exc()}')
        return Image.fromarray(np.full((tex_size, tex_size, 3), 180, dtype=np.uint8))


# ── OBJ writer ───────────────────────────────────────────────────────

def write_obj(verts, faces, uvs, obj_path, mtl_path):
    """Write OBJ + MTL with texture reference."""
    with open(mtl_path, 'w') as f:
        f.write("newmtl skin\n")
        f.write("Ka 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n")
        f.write("d 1.0\nillum 1\n")
        f.write("map_Kd mesh.png\n")

    with open(obj_path, 'w') as f:
        f.write("# SkinMapper mesh\nmtllib mesh.mtl\nusemtl skin\n\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for uv in uvs:
            # Clamp u to [0,1] — Procreate rejects u > 1.0 ("Overlapping UV").
            # Seam-duplicated vertices have u ∈ (1.0, 1.5]; wrapping them back
            # gives a small stretch at the single back-of-limb seam line, which
            # is acceptable for tattoo placement. The texture bake already
            # correctly skips/handles those triangles internally.
            f.write(f"vt {uv[0] % 1.0:.6f} {uv[1]:.6f}\n")
        f.write("\n")
        for tri in faces:
            f.write(f"f {tri[0]+1}/{tri[0]+1} {tri[1]+1}/{tri[1]+1} {tri[2]+1}/{tri[2]+1}\n")


# ── Routes ───────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify(error='Upload too large. Use fewer or smaller photos.'), 413

@app.errorhandler(500)
def server_error(e):
    original = getattr(e, 'original_exception', None)
    return jsonify(error=f'Server error: {str(original or e)}'), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f'Unhandled: {traceback.format_exc()}')
    return jsonify(error=f'{type(e).__name__}: {str(e)}'), 500


@app.route('/health')
def health():
    cleanup_old_jobs()
    colmap_ok = shutil.which('colmap') is not None or os.path.isfile('/usr/local/bin/colmap')
    try:
        import open3d; o3d_ok = True
    except ImportError:
        o3d_ok = False
    with lock:
        q, a = len(queue), active_count
    return jsonify(status='ok', colmap=colmap_ok, open3d=o3d_ok, gpu=HAS_GPU,
                   gpu_name=GPU_NAME, version='4.0.0', active_jobs=a, queued_jobs=q)


@app.route('/submit', methods=['POST'])
def submit():
    try:
        cleanup_old_jobs()
        with lock:
            if active_count + len(queue) >= MAX_QUEUE:
                return jsonify(error='Server busy — try again in a few minutes.'), 503

        if 'photos' not in request.files:
            return jsonify(error='No photos in upload'), 400

        f = request.files['photos']
        job_id = str(uuid.uuid4())
        job_dir = os.path.join(JOBS_DIR, job_id)
        img_dir = os.path.join(job_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)

        zip_path = os.path.join(job_dir, 'photos.zip')
        f.save(zip_path)
        logging.info(f'[{job_id[:8]}] Upload: {os.path.getsize(zip_path)} bytes')

        with zipfile.ZipFile(zip_path) as zf:
            for m in zf.namelist():
                name = os.path.basename(m)
                if name.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                    with zf.open(m) as src, open(os.path.join(img_dir, name), 'wb') as dst:
                        dst.write(src.read())
        os.remove(zip_path)

        count = len(os.listdir(img_dir))
        if count < 10:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify(error=f'Only {count} photos — need at least 10.'), 400
        if count > MAX_PHOTOS:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify(error=f'{count} photos — max is {MAX_PHOTOS}.'), 400

        # Read scan metadata sent by iOS app
        scan_type = request.form.get('scan_type', 'halfWrap')   # fullWrap|halfWrap|cap|patch|backPanel
        body_part = request.form.get('body_part', 'leg')
        logging.info(f'[{job_id[:8]}] scan_type={scan_type} body_part={body_part}')

        # Persist metadata alongside the job so retry can reuse it
        meta = dict(scan_type=scan_type, body_part=body_part)
        with open(os.path.join(job_dir, 'meta.json'), 'w') as _mf:
            json.dump(meta, _mf)

        set_job(job_id, 'queued', 0.0, f'Queued — {count} photos')
        enqueue_job(job_id, img_dir, job_dir, scan_type=scan_type, body_part=body_part)
        return jsonify(job_id=job_id, image_count=count), 202

    except Exception as e:
        logging.error(f'Submit failed: {traceback.format_exc()}')
        return jsonify(error=f'{type(e).__name__}: {str(e)}'), 500


@app.route('/status/<job_id>')
def status(job_id):
    job = get_job(job_id)
    if not job:
        if os.path.exists(os.path.join(JOBS_DIR, job_id, f'{job_id}.zip')):
            return jsonify(status='done', progress=1.0, message='Complete', result_url=f'/result/{job_id}')
        return jsonify(error='Job not found'), 404
    return jsonify(job)


@app.route('/result/<job_id>')
def result(job_id):
    zip_path = os.path.join(JOBS_DIR, job_id, f'{job_id}.zip')
    if not os.path.exists(zip_path):
        return jsonify(error='Result not found'), 404
    return send_file(zip_path, mimetype='application/zip',
                     as_attachment=True, download_name='scan.zip')


@app.route('/retry/<job_id>', methods=['POST'])
def retry_job(job_id):
    """
    Re-run the pipeline using the photos.zip saved from a previous job.
    The original photos.zip is kept after cleanup so the user can retry
    without re-uploading photos from the device.
    Returns the same shape as /submit: {job_id, image_count}.
    """
    try:
        old_zip = os.path.join(JOBS_DIR, job_id, 'photos.zip')
        if not os.path.exists(old_zip):
            return jsonify(error='Original photos not found — job may have expired'), 404

        new_job_id = str(uuid.uuid4())
        new_job_dir = os.path.join(JOBS_DIR, new_job_id)
        new_img_dir = os.path.join(new_job_dir, 'images')
        os.makedirs(new_img_dir, exist_ok=True)

        # Copy zip to new job dir (keeps the original intact for further retries)
        new_zip = os.path.join(new_job_dir, 'photos.zip')
        shutil.copy2(old_zip, new_zip)

        # Extract images
        IMG_EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
        with zipfile.ZipFile(new_zip, 'r') as zf:
            for member in zf.namelist():
                fname = os.path.basename(member)
                if not fname:
                    continue
                if os.path.splitext(fname.lower())[1] not in IMG_EXTS:
                    continue
                with zf.open(member) as src, \
                     open(os.path.join(new_img_dir, fname), 'wb') as dst:
                    dst.write(src.read())

        count = len(os.listdir(new_img_dir))
        if count < 10:
            shutil.rmtree(new_job_dir, ignore_errors=True)
            return jsonify(error=f'Only {count} photos in original job — need at least 10'), 400

        # Load scan metadata from the original job so retry uses the same settings
        old_meta_path = os.path.join(JOBS_DIR, job_id, 'meta.json')
        try:
            with open(old_meta_path) as _mf:
                old_meta = json.load(_mf)
            scan_type = old_meta.get('scan_type', 'halfWrap')
            body_part = old_meta.get('body_part', 'leg')
        except Exception:
            scan_type = 'halfWrap'
            body_part = 'leg'

        # Copy meta to new job dir
        meta = dict(scan_type=scan_type, body_part=body_part)
        with open(os.path.join(new_job_dir, 'meta.json'), 'w') as _mf:
            json.dump(meta, _mf)

        logging.info(f'[{new_job_id[:8]}] Retry of {job_id[:8]}: {count} photos, '
                     f'scan_type={scan_type} body_part={body_part}')
        set_job(new_job_id, 'queued', 0.0, f'Queued — {count} photos (retry)')
        enqueue_job(new_job_id, new_img_dir, new_job_dir, scan_type=scan_type, body_part=body_part)
        return jsonify(job_id=new_job_id, image_count=count), 202

    except Exception as e:
        logging.exception(f'[retry] {e}')
        return jsonify(error=str(e)), 500


@app.route('/logs')
def logs():
    """View recent server logs for debugging."""
    log_path = os.environ.get('LOG_FILE', '/workspace/server.log')
    try:
        with open(log_path) as f:
            lines = f.readlines()
        last = lines[-200:] if len(lines) > 200 else lines
        return '<pre>' + ''.join(last) + '</pre>', 200, {'Content-Type': 'text/html'}
    except:
        return 'No log file found', 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, threaded=True)
