"""
SkinMapper Server — GPU-accelerated photogrammetry pipeline
COLMAP (dense reconstruction) + open3d (meshing) + texture baking from photos
"""

import os, uuid, zipfile, subprocess, threading, json, shutil, logging, traceback, time, struct
from collections import deque
from flask import Flask, request, jsonify, send_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

JOBS_DIR = '/tmp/skinmapper_jobs'
MAX_CONCURRENT = 2
MAX_QUEUE = 10
MAX_PHOTOS = 60
JOB_TTL = 3600
MAX_DIM = 1600  # full resolution with GPU

os.makedirs(JOBS_DIR, exist_ok=True)

# ── Check GPU availability at startup ────────────────────────────────
def check_gpu():
    r = subprocess.run(['colmap', 'feature_extractor', '--help'], capture_output=True, text=True)
    # Also check nvidia-smi
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                         capture_output=True, text=True)
    gpu_name = gpu.stdout.strip() if gpu.returncode == 0 else 'No GPU detected'
    logging.info(f'GPU: {gpu_name}')
    return gpu.returncode == 0, gpu_name

HAS_GPU, GPU_NAME = check_gpu()

# ── Job state ────────────────────────────────────────────────────────
jobs = {}
lock = threading.Lock()
queue = deque()
active_count = 0

def set_job(job_id, status, progress, message, result_url=None, error=None):
    with lock:
        jobs[job_id] = dict(status=status, progress=progress, message=message,
                            result_url=result_url, error=error, updated=time.time())

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
            set_job(job_id, 'queued', 0.0, f'In queue (position {len(queue)})')

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

def cleanup_old_jobs():
    now = time.time()
    with lock:
        expired = [jid for jid, j in jobs.items() if now - j.get('updated', 0) > JOB_TTL]
        for jid in expired:
            del jobs[jid]
            shutil.rmtree(os.path.join(JOBS_DIR, jid), ignore_errors=True)


# ── Pipeline ─────────────────────────────────────────────────────────

def run_pipeline(job_id, image_dir, job_dir):
    tag = job_id[:8]
    use_gpu = '1' if HAS_GPU else '0'

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
                    if max(w, h) > MAX_DIM:
                        scale = MAX_DIM / max(w, h)
                        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                        im.save(fpath, quality=90)
            except Exception as e:
                logging.warning(f'[{tag}] Resize {fname}: {e}')

        # 1 — Feature extraction (CPU — COLMAP's SiftGPU needs OpenGL display)
        run(['colmap', 'feature_extractor',
             '--database_path', db,
             '--image_path', image_dir,
             '--ImageReader.single_camera', '1',
             '--SiftExtraction.use_gpu', '0',
             '--SiftExtraction.max_image_size', str(MAX_DIM),
             '--SiftExtraction.max_num_features', '8192'],
            0.08, 'Extracting features…')

        # 2 — Matching (CPU — same OpenGL issue)
        run(['colmap', 'exhaustive_matcher',
             '--database_path', db,
             '--SiftMatching.use_gpu', '0'],
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
            raise RuntimeError('Could not reconstruct — take more overlapping photos with slow, steady movement.')

        # 4 — Undistort images
        run(['colmap', 'image_undistorter',
             '--image_path', image_dir,
             '--input_path', sparse0,
             '--output_path', dense,
             '--output_type', 'COLMAP',
             '--max_image_size', str(MAX_DIM)],
            0.38, 'Undistorting images…')

        # 5 — Dense stereo (GPU accelerated)
        run(['colmap', 'patch_match_stereo',
             '--workspace_path', dense,
             '--workspace_format', 'COLMAP',
             '--PatchMatchStereo.geom_consistency', 'true',
             '--PatchMatchStereo.max_image_size', str(MAX_DIM)],
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

        # 7 — Mesh with open3d
        set_job(job_id, 'processing', 0.66, 'Building mesh…')
        import open3d as o3d
        import numpy as np

        pcd = o3d.io.read_point_cloud(ply_path)
        pts = len(pcd.points)
        has_colors = pcd.has_colors()
        logging.info(f'[{tag}] Dense cloud: {pts} points, has_colors={has_colors}')

        if pts < 500:
            raise RuntimeError(f'Only {pts} points — take more overlapping photos.')

        # Keep only the largest cluster (removes background/floor noise)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
        if len(labels) > 0 and labels.max() >= 0:
            largest = np.argmax(np.bincount(labels[labels >= 0]))
            mask = labels == largest
            pcd = pcd.select_by_index(np.where(mask)[0])
            logging.info(f'[{tag}] After clustering: {len(pcd.points)} points (was {pts})')

        # Save point cloud colors before meshing
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        logging.info(f'[{tag}] Point cloud colors available: {pcd_colors is not None}')

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

        if len(pcd.points) > 500000:
            pcd = pcd.voxel_down_sample(0.001)
            pcd_points = np.asarray(pcd.points)
            pcd_colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        set_job(job_id, 'processing', 0.72, 'Creating surface…')
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        densities = np.asarray(densities)
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))

        target = min(100000, len(mesh.triangles))
        if len(mesh.triangles) > target:
            mesh = mesh.simplify_quadric_decimation(target)
        mesh.compute_vertex_normals()

        logging.info(f'[{tag}] Mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris')
        if len(mesh.triangles) == 0:
            raise RuntimeError('Mesh generation failed — try more overlapping photos.')

        # 8 — Transfer point cloud colors to mesh vertices via nearest-neighbor
        set_job(job_id, 'processing', 0.76, 'Coloring mesh…')
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        vertex_colors = None
        if pcd_colors is not None and len(pcd_colors) > 0:
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            vertex_colors = np.zeros((len(vertices), 3))
            for i, v in enumerate(vertices):
                [_, idx, _] = pcd_tree.search_knn_vector_3d(v, 5)
                vertex_colors[i] = pcd_colors[idx].mean(axis=0)
            logging.info(f'[{tag}] Vertex colors: min={vertex_colors.min():.3f} max={vertex_colors.max():.3f}')

        # 9 — UV unwrap (cylindrical)
        set_job(job_id, 'processing', 0.78, 'UV unwrapping…')
        center = vertices.mean(axis=0)
        centered = vertices - center
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        axis = eigenvectors[:, np.argmax(eigenvalues)]
        ref = eigenvectors[:, np.argsort(eigenvalues)[-2]]

        along = centered @ axis
        radial = centered - np.outer(along, axis)
        u = np.arctan2(np.sum(radial * np.cross(axis, ref), axis=1),
                        np.sum(radial * ref, axis=1))
        u = (u + np.pi) / (2 * np.pi)

        v_min, v_max = along.min(), along.max()
        v = (along - v_min) / max(v_max - v_min, 1e-6)

        uvs = np.column_stack([u, v])

        # 10 — Bake texture by rasterizing colored triangles into UV space
        set_job(job_id, 'processing', 0.82, 'Baking skin texture…')
        texture = bake_texture_from_vertex_colors(
            vertices, triangles, uvs, vertex_colors, tag
        )

        # Normalize mesh to real-world scale (~0.35m longest axis = typical limb)
        bbox = vertices.max(axis=0) - vertices.min(axis=0)
        current_size = bbox.max()
        if current_size > 0:
            target_size = 0.35  # 35cm in meters
            scale = target_size / current_size
            vertices = vertices * scale
            logging.info(f'[{tag}] Normalized mesh: scale={scale:.4f}, bbox was {bbox}, now {bbox*scale}')

        # 10 — Export OBJ + MTL + texture as zip
        set_job(job_id, 'processing', 0.92, 'Packaging result…')
        result_dir = os.path.join(job_dir, 'result')
        os.makedirs(result_dir, exist_ok=True)

        tex_path = os.path.join(result_dir, 'mesh.png')
        if texture is not None:
            texture.save(tex_path)

        obj_path = os.path.join(result_dir, 'mesh.obj')
        mtl_path = os.path.join(result_dir, 'mesh.mtl')
        write_obj(vertices, triangles, uvs, obj_path, mtl_path, os.path.exists(tex_path))

        # Zip it all up
        zip_path = os.path.join(job_dir, f'{job_id}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            zf.write(obj_path, 'mesh.obj')
            zf.write(mtl_path, 'mesh.mtl')
            if os.path.exists(tex_path):
                zf.write(tex_path, 'mesh.png')

        result_size = os.path.getsize(zip_path)
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


# ── Texture baking from vertex colors ────────────────────────────────

def bake_texture_from_vertex_colors(vertices, triangles, uvs, vertex_colors, tag, tex_size=4096):
    """Rasterize vertex colors into UV texture space."""
    import numpy as np
    from PIL import Image, ImageFilter

    try:
        tex_arr = np.full((tex_size, tex_size, 3), 180, dtype=np.float32)  # skin-tone default
        weight_buf = np.zeros((tex_size, tex_size), dtype=np.float32)

        if vertex_colors is None:
            logging.warning(f'[{tag}] No vertex colors — returning skin-tone texture')
            return Image.fromarray(tex_arr.astype(np.uint8))

        # Colors from open3d are 0-1 float, convert to 0-255
        colors_255 = (vertex_colors * 255).clip(0, 255)

        logging.info(f'[{tag}] Rasterizing {len(triangles)} triangles into {tex_size}x{tex_size} texture')

        for tri_idx, tri in enumerate(triangles):
            v0, v1, v2 = tri

            # UV coords → pixel coords in texture
            uv0 = uvs[v0]; uv1 = uvs[v1]; uv2 = uvs[v2]
            tx0 = int(uv0[0] * (tex_size-1)); ty0 = int((1-uv0[1]) * (tex_size-1))
            tx1 = int(uv1[0] * (tex_size-1)); ty1 = int((1-uv1[1]) * (tex_size-1))
            tx2 = int(uv2[0] * (tex_size-1)); ty2 = int((1-uv2[1]) * (tex_size-1))

            # Bounding box of triangle in texture space
            min_x = max(0, min(tx0, tx1, tx2))
            max_x = min(tex_size-1, max(tx0, tx1, tx2))
            min_y = max(0, min(ty0, ty1, ty2))
            max_y = min(tex_size-1, max(ty0, ty1, ty2))

            # Skip degenerate or huge triangles (UV seam wrap-around)
            if max_x - min_x > tex_size // 2 or max_y - min_y > tex_size // 2:
                continue
            if max_x - min_x > 500 or max_y - min_y > 500:
                continue

            c0 = colors_255[v0]; c1 = colors_255[v1]; c2 = colors_255[v2]

            # Rasterize: for each pixel in bbox, check if inside triangle
            for py in range(min_y, max_y + 1):
                for px in range(min_x, max_x + 1):
                    # Barycentric coordinates
                    denom = (ty1-ty2)*(tx0-tx2) + (tx2-tx1)*(ty0-ty2)
                    if abs(denom) < 1e-10:
                        continue
                    w0 = ((ty1-ty2)*(px-tx2) + (tx2-tx1)*(py-ty2)) / denom
                    w1 = ((ty2-ty0)*(px-tx2) + (tx0-tx2)*(py-ty2)) / denom
                    w2 = 1.0 - w0 - w1

                    if w0 >= -0.001 and w1 >= -0.001 and w2 >= -0.001:
                        color = w0*c0 + w1*c1 + w2*c2
                        tex_arr[py, px] = color.clip(0, 255)
                        weight_buf[py, px] = 1.0

        painted = weight_buf > 0
        painted_pct = painted.sum() / painted.size * 100
        logging.info(f'[{tag}] Texture: {painted_pct:.1f}% painted from vertex colors')

        result = Image.fromarray(tex_arr.astype(np.uint8))

        # Dilate painted regions to fill gaps (push-pull)
        if painted_pct < 95:
            mask = Image.fromarray((weight_buf * 255).astype(np.uint8))
            for _ in range(20):
                blurred = result.filter(ImageFilter.BoxBlur(2))
                blurred_mask = mask.filter(ImageFilter.BoxBlur(2))
                result_arr = np.array(result).astype(float)
                blurred_arr = np.array(blurred).astype(float)
                mask_arr = np.array(mask).astype(float) / 255.0
                blurred_mask_arr = np.array(blurred_mask).astype(float) / 255.0
                # Fill unpainted with blurred
                unpainted = mask_arr < 0.5
                for c in range(3):
                    result_arr[:,:,c] = np.where(unpainted, blurred_arr[:,:,c], result_arr[:,:,c])
                mask_arr = np.maximum(mask_arr, blurred_mask_arr)
                result = Image.fromarray(result_arr.clip(0, 255).astype(np.uint8))
                mask = Image.fromarray((mask_arr * 255).clip(0, 255).astype(np.uint8))

        return result

    except Exception as e:
        logging.error(f'[{tag}] Texture bake failed: {traceback.format_exc()}')
        return None


# ── COLMAP binary readers ────────────────────────────────────────────

def read_colmap_cameras_bin(path):
    cameras = {}
    if not os.path.exists(path):
        return cameras
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            cam_id = struct.unpack('<I', f.read(4))[0]
            model_id = struct.unpack('<i', f.read(4))[0]
            w = struct.unpack('<Q', f.read(8))[0]
            h = struct.unpack('<Q', f.read(8))[0]
            np_map = {0:3, 1:4, 2:4, 3:5, 4:4, 5:5}
            num_p = np_map.get(model_id, 4)
            params = struct.unpack(f'<{num_p}d', f.read(8*num_p))
            cameras[cam_id] = {'w': w, 'h': h, 'params': params, 'model_id': model_id}
    return cameras


def read_colmap_images_bin(path):
    import numpy as np
    images = {}
    if not os.path.exists(path):
        return images
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            img_id = struct.unpack('<I', f.read(4))[0]
            qw, qx, qy, qz = struct.unpack('<4d', f.read(32))
            tx, ty, tz = struct.unpack('<3d', f.read(24))
            cam_id = struct.unpack('<I', f.read(4))[0]
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00': break
                name += c
            num_pts = struct.unpack('<Q', f.read(8))[0]
            f.read(num_pts * 24)
            # Quaternion to rotation matrix
            R = np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
            ])
            images[img_id] = {'name': name.decode(), 'camera_id': cam_id, 'R': R, 't': np.array([tx,ty,tz])}
    return images


# ── OBJ writer ───────────────────────────────────────────────────────

def write_obj(vertices, triangles, uvs, obj_path, mtl_path, has_texture):
    with open(mtl_path, 'w') as f:
        f.write("newmtl skin\nKa 0.2 0.2 0.2\nKd 0.8 0.8 0.8\nKs 0.0 0.0 0.0\n")
        if has_texture:
            f.write("map_Kd mesh.png\n")

    with open(obj_path, 'w') as f:
        f.write("mtllib mesh.mtl\nusemtl skin\n\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for tri in triangles:
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
    colmap_ok = shutil.which('colmap') is not None
    try:
        import open3d; o3d_ok = True
    except ImportError:
        o3d_ok = False
    with lock:
        q, a = len(queue), active_count
    return jsonify(status='ok', colmap=colmap_ok, open3d=o3d_ok, gpu=HAS_GPU,
                   gpu_name=GPU_NAME, version='3.2.0', active_jobs=a, queued_jobs=q)


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

        set_job(job_id, 'queued', 0.0, f'Queued — {count} photos')
        enqueue_job(job_id, img_dir, job_dir)
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
