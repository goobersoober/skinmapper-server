import os, uuid, zipfile, subprocess, threading, json, shutil, logging, traceback, time
from collections import deque
from flask import Flask, request, jsonify, send_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

JOBS_DIR = '/tmp/skinmapper_jobs'
MAX_CONCURRENT = 2          # max jobs running at once
MAX_QUEUE = 10              # max jobs waiting
MAX_PHOTOS = 60             # reject uploads with more than this
JOB_TTL = 3600              # keep results for 1 hour
MAX_DIM = 1200              # downscale longest edge to this

os.makedirs(JOBS_DIR, exist_ok=True)

# ── Job state ────────────────────────────────────────────────────────
jobs = {}
lock = threading.Lock()
queue = deque()
active_count = 0

def set_job(job_id, status, progress, message, result_url=None, error=None):
    with lock:
        jobs[job_id] = dict(
            status=status, progress=progress, message=message,
            result_url=result_url, error=error, updated=time.time()
        )

def get_job(job_id):
    with lock:
        return jobs.get(job_id)

# ── Job queue ────────────────────────────────────────────────────────
def enqueue_job(job_id, img_dir, job_dir):
    global active_count
    with lock:
        if active_count < MAX_CONCURRENT:
            active_count += 1
            threading.Thread(target=_run_and_release, args=(job_id, img_dir, job_dir), daemon=True).start()
        else:
            queue.append((job_id, img_dir, job_dir))
            pos = len(queue)
            set_job(job_id, 'queued', 0.0, f'In queue (position {pos})')

def _run_and_release(job_id, img_dir, job_dir):
    global active_count
    try:
        run_pipeline(job_id, img_dir, job_dir)
    finally:
        with lock:
            active_count -= 1
        _start_next()

def _start_next():
    global active_count
    with lock:
        if queue and active_count < MAX_CONCURRENT:
            active_count += 1
            job_id, img_dir, job_dir = queue.popleft()
            threading.Thread(target=_run_and_release, args=(job_id, img_dir, job_dir), daemon=True).start()

# ── Cleanup old jobs ────────────────────────────────────────────────
def cleanup_old_jobs():
    now = time.time()
    with lock:
        expired = [jid for jid, j in jobs.items() if now - j.get('updated', 0) > JOB_TTL]
        for jid in expired:
            del jobs[jid]
            d = os.path.join(JOBS_DIR, jid)
            shutil.rmtree(d, ignore_errors=True)
            logging.info(f'Cleaned up expired job {jid[:8]}')

# ── Pipeline ─────────────────────────────────────────────────────────
def run_pipeline(job_id, image_dir, job_dir):
    tag = job_id[:8]
    try:
        db      = os.path.join(job_dir, 'db.db')
        sparse  = os.path.join(job_dir, 'sparse')
        dense   = os.path.join(job_dir, 'dense')
        os.makedirs(sparse, exist_ok=True)
        os.makedirs(dense,  exist_ok=True)

        def run(cmd, progress, msg):
            set_job(job_id, 'processing', progress, msg)
            logging.info(f'[{tag}] {msg}')
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                err = r.stderr.strip() or r.stdout.strip() or '(no output)'
                if r.returncode < 0:
                    sig = -r.returncode
                    signames = {9: 'SIGKILL (out of memory)', 11: 'SIGSEGV (crash)', 6: 'SIGABRT (abort)'}
                    signame = signames.get(sig, f'signal {sig}')
                    err = f'Process killed by {signame}. {err}'
                logging.error(f'[{tag}] {msg} FAILED (exit {r.returncode}): {err[:500]}')
                raise RuntimeError(f'{msg} failed (exit {r.returncode}): {err[:800]}')
            logging.info(f'[{tag}] {msg} ✓')

        # Verify images
        imgs = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        logging.info(f'[{tag}] Pipeline starting with {len(imgs)} images')
        if not imgs:
            raise RuntimeError('No valid images found in upload')

        # Downscale images to save memory
        set_job(job_id, 'processing', 0.05, 'Preparing images…')
        from PIL import Image
        for fname in imgs:
            fpath = os.path.join(image_dir, fname)
            try:
                with Image.open(fpath) as im:
                    w, h = im.size
                    if max(w, h) > MAX_DIM:
                        scale = MAX_DIM / max(w, h)
                        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                        im.save(fpath, quality=85)
            except Exception as e:
                logging.warning(f'[{tag}] Could not resize {fname}: {e}')

        # 1 — Feature extraction
        run(['colmap', 'feature_extractor',
             '--database_path', db,
             '--image_path', image_dir,
             '--ImageReader.single_camera', '1',
             '--SiftExtraction.use_gpu', '0',
             '--SiftExtraction.max_image_size', str(MAX_DIM),
             '--SiftExtraction.max_num_features', '4096'],
            0.10, 'Extracting image features…')

        # 2 — Sequential matching (much less memory than exhaustive)
        run(['colmap', 'sequential_matcher',
             '--database_path', db,
             '--SiftMatching.use_gpu', '0',
             '--SequentialMatching.overlap', '10',
             '--SequentialMatching.loop_detection', '0'],
            0.25, 'Matching features across photos…')

        # 3 — Sparse SfM
        run(['colmap', 'mapper',
             '--database_path', db,
             '--image_path', image_dir,
             '--output_path', sparse,
             '--Mapper.num_threads', '2',
             '--Mapper.max_num_models', '1'],
            0.40, 'Reconstructing 3D structure…')

        sparse0 = os.path.join(sparse, '0')
        if not os.path.exists(sparse0):
            raise RuntimeError(
                'Could not reconstruct — take more overlapping photos with slow, steady movement.')

        # 4 — Undistort
        run(['colmap', 'image_undistorter',
             '--image_path', image_dir,
             '--input_path', sparse0,
             '--output_path', dense,
             '--output_type', 'COLMAP',
             '--max_image_size', str(MAX_DIM)],
            0.50, 'Undistorting images…')

        # 5 — Depth maps (patch-match stereo)
        run(['colmap', 'patch_match_stereo',
             '--workspace_path', dense,
             '--workspace_format', 'COLMAP',
             '--PatchMatchStereo.geom_consistency', 'true',
             '--PatchMatchStereo.max_image_size', str(MAX_DIM),
             '--PatchMatchStereo.window_radius', '3',
             '--PatchMatchStereo.num_iterations', '3'],
            0.62, 'Computing depth maps…')

        # 6 — Fuse into dense point cloud
        ply = os.path.join(job_dir, 'dense.ply')
        run(['colmap', 'stereo_fusion',
             '--workspace_path', dense,
             '--workspace_format', 'COLMAP',
             '--input_type', 'geometric',
             '--output_path', ply],
            0.72, 'Fusing into point cloud…')

        # 7 — Mesh with open3d
        set_job(job_id, 'processing', 0.78, 'Building mesh…')
        import open3d as o3d
        import numpy as np

        pcd = o3d.io.read_point_cloud(ply)
        pts = len(pcd.points)
        logging.info(f'[{tag}] Point cloud: {pts} points')

        if pts < 500:
            raise RuntimeError(
                f'Only {pts} points reconstructed — not enough for a mesh. '
                'Try taking more photos with more overlap.')

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(50)

        # Downsample if huge point cloud to save memory
        if pts > 200000:
            voxel = 0.002
            pcd = pcd.voxel_down_sample(voxel)
            logging.info(f'[{tag}] Downsampled to {len(pcd.points)} points')

        set_job(job_id, 'processing', 0.83, 'Creating surface…')
        poisson_depth = 8 if pts < 100000 else 9
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth)

        # Remove low-density vertices (Poisson artifacts at edges)
        densities = np.asarray(densities)
        threshold = np.quantile(densities, 0.05)
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        target_faces = min(50000, len(mesh.triangles))
        mesh = mesh.simplify_quadric_decimation(target_faces)
        mesh.compute_vertex_normals()

        set_job(job_id, 'processing', 0.90, 'Exporting mesh…')
        obj_path = os.path.join(job_dir, 'mesh.obj')
        o3d.io.write_triangle_mesh(obj_path, mesh, write_ascii=False)

        # 8 — Package as USDZ
        set_job(job_id, 'processing', 0.95, 'Packaging result…')
        usdz = os.path.join(job_dir, f'{job_id}.usdz')
        with zipfile.ZipFile(usdz, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(obj_path, 'mesh.obj')
            mtl = obj_path.replace('.obj', '.mtl')
            if os.path.exists(mtl):
                zf.write(mtl, 'mesh.mtl')
            for ext in ['.png', '.jpg', '.jpeg']:
                tex = obj_path.replace('.obj', ext)
                if os.path.exists(tex):
                    zf.write(tex, f'mesh{ext}')

        result_size = os.path.getsize(usdz)
        logging.info(f'[{tag}] Done! Result: {result_size} bytes, {len(mesh.triangles)} triangles')
        set_job(job_id, 'done', 1.0, 'Reconstruction complete!',
                result_url=f'/result/{job_id}')

    except Exception as e:
        logging.error(f'[{tag}] Pipeline error: {traceback.format_exc()}')
        set_job(job_id, 'error', 0, str(e), error=str(e))
    finally:
        # Clean up intermediate files, keep only the .usdz result
        for d in [image_dir, sparse, dense]:
            shutil.rmtree(d, ignore_errors=True)
        for f in [db, os.path.join(job_dir, 'dense.ply'), obj_path if 'obj_path' in dir() else '']:
            try: os.remove(f)
            except: pass


# ── Routes ───────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify(error='Upload too large. Please use fewer or smaller photos.'), 413

@app.errorhandler(500)
def server_error(e):
    original = getattr(e, 'original_exception', None)
    msg = str(original) if original else str(e)
    return jsonify(error=f'Server error: {msg}'), 500

@app.errorhandler(Exception)
def handle_exception(e):
    tb = traceback.format_exc()
    logging.error(f'Unhandled: {tb}')
    return jsonify(error=f'{type(e).__name__}: {str(e)}'), 500


@app.route('/health')
def health():
    cleanup_old_jobs()
    colmap_ok = shutil.which('colmap') is not None
    try:
        import open3d
        o3d_ok = True
    except ImportError:
        o3d_ok = False
    with lock:
        queued = len(queue)
        running = active_count
    return jsonify(status='ok', colmap=colmap_ok, open3d=o3d_ok,
                   version='2.0.0', active_jobs=running, queued_jobs=queued)


@app.route('/submit', methods=['POST'])
def submit():
    try:
        cleanup_old_jobs()

        # Check queue capacity
        with lock:
            total = active_count + len(queue)
        if total >= MAX_QUEUE:
            return jsonify(error='Server is busy. Please try again in a few minutes.'), 503

        logging.info(f'Submit: content_length={request.content_length}')
        if 'photos' not in request.files:
            return jsonify(error='No photos file in upload'), 400

        f = request.files['photos']
        job_id  = str(uuid.uuid4())
        job_dir = os.path.join(JOBS_DIR, job_id)
        img_dir = os.path.join(job_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)

        zip_path = os.path.join(job_dir, 'photos.zip')
        f.save(zip_path)
        logging.info(f'[{job_id[:8]}] Saved zip: {os.path.getsize(zip_path)} bytes')

        with zipfile.ZipFile(zip_path) as zf:
            for m in zf.namelist():
                name = os.path.basename(m)
                if name.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                    with zf.open(m) as src, open(os.path.join(img_dir, name), 'wb') as dst:
                        dst.write(src.read())
        os.remove(zip_path)

        count = len(os.listdir(img_dir))
        logging.info(f'[{job_id[:8]}] Extracted {count} images')

        if count < 10:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify(error=f'Only {count} photos — need at least 10 for a good scan.'), 400

        if count > MAX_PHOTOS:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify(error=f'{count} photos is too many — please use {MAX_PHOTOS} or fewer.'), 400

        set_job(job_id, 'queued', 0.0, f'Queued — {count} photos received')
        enqueue_job(job_id, img_dir, job_dir)
        return jsonify(job_id=job_id, image_count=count), 202

    except Exception as e:
        logging.error(f'Submit failed: {traceback.format_exc()}')
        return jsonify(error=f'Submit error: {type(e).__name__}: {str(e)}'), 500


@app.route('/status/<job_id>')
def status(job_id):
    job = get_job(job_id)
    if not job:
        usdz = os.path.join(JOBS_DIR, job_id, f'{job_id}.usdz')
        if os.path.exists(usdz):
            return jsonify(status='done', progress=1.0,
                           message='Complete', result_url=f'/result/{job_id}')
        return jsonify(error='Job not found'), 404
    return jsonify(job)


@app.route('/result/<job_id>')
def result(job_id):
    usdz = os.path.join(JOBS_DIR, job_id, f'{job_id}.usdz')
    if not os.path.exists(usdz):
        return jsonify(error='Result not found'), 404
    return send_file(usdz, mimetype='model/vnd.usdz+zip',
                     as_attachment=True, download_name='scan.usdz')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, threaded=True)
