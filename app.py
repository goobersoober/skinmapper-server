"""
SkinMapper Server — GPU-accelerated photogrammetry pipeline
COLMAP (dense reconstruction) + open3d (meshing) + texture baking from photos
"""

import os, uuid, zipfile, subprocess, threading, json, shutil, logging, traceback, time, struct
from collections import deque

# COLMAP requires a display on headless servers — force offscreen Qt
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ''

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

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2 — Point cloud → textured mesh (xatlas + vertex colors)
        # ═══════════════════════════════════════════════════════════════

        set_job(job_id, 'processing', 0.66, 'Loading point cloud…')
        import open3d as o3d
        import numpy as np
        import trimesh
        import xatlas
        import cv2
        from scipy.spatial import cKDTree

        pcd = o3d.io.read_point_cloud(ply_path)
        pts = len(pcd.points)
        has_colors = pcd.has_colors()
        logging.info(f'[{tag}] Dense cloud: {pts} points, has_colors={has_colors}')
        if pts < 500:
            raise RuntimeError(f'Only {pts} points — take more overlapping photos.')

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

        # DBSCAN: keep only the largest cluster (body part vs background)
        pts_arr = np.asarray(pcd.points)
        nn_dists = np.asarray(pcd.compute_nearest_neighbor_distance())
        eps = np.median(nn_dists) * 5
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=20, print_progress=False))
        if labels.max() >= 0:
            counts = np.bincount(labels[labels >= 0])
            largest = np.argmax(counts)
            pcd = pcd.select_by_index(np.where(labels == largest)[0])
            logging.info(f'[{tag}] Largest cluster: {len(pcd.points)} pts ({len(counts)} clusters)')

        # Save colors for later
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        # ── Step 8: Surface reconstruction ───────────────────────────
        set_job(job_id, 'processing', 0.72, 'Creating surface…')

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(30)

        # Ball-pivoting: only creates triangles where points exist
        nn_dists = np.asarray(pcd.compute_nearest_neighbor_distance())
        avg_dist = np.mean(nn_dists)
        radii = [avg_dist * 1.5, avg_dist * 3, avg_dist * 6]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        logging.info(f'[{tag}] Ball-pivot: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris')

        # Fallback to Poisson if ball-pivoting produces too little
        if len(mesh.triangles) < 500:
            logging.info(f'[{tag}] Ball-pivot insufficient, using Poisson')
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
            densities = np.asarray(densities)
            # Aggressive trim: remove bottom 15% by density (phantom surfaces)
            mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.15))

        # Remove small disconnected pieces
        tri_clusters, tri_counts, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        tri_counts = np.asarray(tri_counts)
        if len(tri_counts) > 1:
            keep = tri_clusters == tri_counts.argmax()
            mesh.remove_triangles_by_mask(~keep)
            mesh.remove_unreferenced_vertices()
            logging.info(f'[{tag}] Kept largest component: {len(mesh.triangles)} tris')

        # Decimate to reasonable size
        target_tris = min(50000, len(mesh.triangles))
        if len(mesh.triangles) > target_tris:
            mesh = mesh.simplify_quadric_decimation(target_tris)
        mesh.compute_vertex_normals()

        verts = np.asarray(mesh.vertices).astype(np.float32)
        faces = np.asarray(mesh.triangles).astype(np.uint32)
        logging.info(f'[{tag}] Final mesh: {len(verts)} verts, {len(faces)} tris')
        if len(faces) == 0:
            raise RuntimeError('Mesh generation failed — try more overlapping photos.')

        # ── Step 9: Transfer vertex colors from point cloud ──────────
        set_job(job_id, 'processing', 0.76, 'Coloring mesh…')

        if pcd_colors is not None and len(pcd_colors) > 0:
            tree = cKDTree(pcd_points)
            _, idx = tree.query(verts, k=3)
            vertex_colors = (pcd_colors[idx].mean(axis=1) * 255).clip(0, 255).astype(np.uint8)
            logging.info(f'[{tag}] Vertex colors: shape={vertex_colors.shape}')
        else:
            vertex_colors = np.full((len(verts), 3), 180, dtype=np.uint8)
            logging.warning(f'[{tag}] No point cloud colors — using placeholder')

        # ── Step 10: UV unwrap with xatlas (no overlapping UVs) ──────
        set_job(job_id, 'processing', 0.80, 'UV unwrapping (xatlas)…')

        vmapping, new_faces, uvs = xatlas.parametrize(verts, faces)
        # xatlas splits vertices at seams → remap colors
        new_verts = verts[vmapping]
        new_colors = vertex_colors[vmapping]
        logging.info(f'[{tag}] xatlas: {len(new_verts)} verts, {len(new_faces)} tris, '
                     f'UV range u=[{uvs[:,0].min():.3f},{uvs[:,0].max():.3f}] '
                     f'v=[{uvs[:,1].min():.3f},{uvs[:,1].max():.3f}]')

        # ── Step 11: Bake vertex colors → texture via rasterization ──
        set_job(job_id, 'processing', 0.85, 'Baking texture…')
        TEX_SIZE = 2048
        texture = bake_vertex_colors_to_texture(
            new_verts, new_faces, uvs, new_colors, TEX_SIZE, tag
        )

        # ── Step 12: Normalize scale + export ────────────────────────
        set_job(job_id, 'processing', 0.92, 'Packaging result…')

        # Normalize to real-world scale (~0.35m longest axis)
        bbox = new_verts.max(axis=0) - new_verts.min(axis=0)
        max_dim = bbox.max()
        if max_dim > 0:
            scale = 0.35 / max_dim
            new_verts = new_verts * scale
            logging.info(f'[{tag}] Normalized: scale={scale:.4f}')

        result_dir = os.path.join(job_dir, 'result')
        os.makedirs(result_dir, exist_ok=True)

        tex_path = os.path.join(result_dir, 'mesh.png')
        texture.save(tex_path)

        obj_path = os.path.join(result_dir, 'mesh.obj')
        mtl_path = os.path.join(result_dir, 'mesh.mtl')
        write_obj(new_verts, new_faces, uvs, obj_path, mtl_path)

        zip_path = os.path.join(job_dir, f'{job_id}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            zf.write(obj_path, 'mesh.obj')
            zf.write(mtl_path, 'mesh.mtl')
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
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
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
    colmap_ok = shutil.which('colmap') is not None
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
