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

        # Save colors for later
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        # ── Step 8: Surface reconstruction (Poisson + cleanup) ──────
        # Poisson creates a smooth continuous watertight surface.
        # The "back" of the limb (unscanned) gets hallucinated, but has very
        # low density — we remove it via aggressive density trimming.
        # This gives a clean continuous topology, equivalent to Blender's
        # Quadraflow remesh in the manual workflow.
        set_job(job_id, 'processing', 0.71, 'Estimating normals…')

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

        # Poisson reconstruction with depth=9 (good detail without huge memory)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, scale=1.1, linear_fit=False)
        densities = np.asarray(densities)
        logging.info(f'[{tag}] Poisson raw: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris')

        # No density trim — keep the full watertight Poisson mesh.
        # Trimming created holes → holes created hundreds of UV island fragments.
        # A closed watertight mesh produces far fewer UV islands → clean texture.
        # The unscanned back gets black (inpainted) texture, which is fine.
        logging.info(f'[{tag}] Poisson mesh (no trim): {len(mesh.triangles)} tris')

        # Smooth surface (Taubin preserves shape, Laplacian would shrink it)
        mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
        logging.info(f'[{tag}] Smoothed (Taubin x10)')

        # Cleanup non-manifold artifacts
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # Keep only largest connected component (drops floating fragments)
        tri_clusters, tri_counts, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        tri_counts = np.asarray(tri_counts)
        if len(tri_counts) > 1:
            keep = tri_clusters == tri_counts.argmax()
            mesh.remove_triangles_by_mask(~keep)
            mesh.remove_unreferenced_vertices()
            logging.info(f'[{tag}] Kept largest component: {len(mesh.triangles)} tris '
                         f'(of {len(tri_counts)} components)')

        # Decimate to ~30K tris — fewer triangles = fewer UV islands = better photo bake
        target_tris = min(30000, len(mesh.triangles))
        if len(mesh.triangles) > target_tris:
            mesh = mesh.simplify_quadric_decimation(target_tris)
            logging.info(f'[{tag}] Decimated to: {len(mesh.triangles)} tris')

        mesh.compute_vertex_normals()
        verts = np.asarray(mesh.vertices).astype(np.float32)
        faces = np.asarray(mesh.triangles).astype(np.uint32)
        logging.info(f'[{tag}] Final mesh: {len(verts)} verts, {len(faces)} tris')
        if len(faces) < 100:
            raise RuntimeError('Mesh too small after cleanup — take more overlapping photos.')

        # ── Step 9: UV unwrap with xatlas ───────────────────────────────
        # xatlas handles any body part shape (arms, legs, hands, shoulders).
        # With a watertight Poisson mesh (no holes), xatlas produces far
        # fewer UV islands than before — no hole-edges to fragment on.
        set_job(job_id, 'processing', 0.76, 'UV unwrapping…')

        vmapping, new_faces, uvs = xatlas.parametrize(verts, faces)
        new_verts = verts[vmapping]
        logging.info(f'[{tag}] xatlas: {len(new_verts)} verts, {len(new_faces)} tris, '
                     f'UV range u=[{uvs[:,0].min():.3f},{uvs[:,0].max():.3f}] '
                     f'v=[{uvs[:,1].min():.3f},{uvs[:,1].max():.3f}]')

        # ── Step 10: Bake texture from photos using COLMAP camera poses ──
        # This projects the original photos onto the UV map — same as Blender's
        # "Selected to Active" bake. Gives photographic-quality textures.
        set_job(job_id, 'processing', 0.82, 'Baking texture from photos…')
        TEX_SIZE = 2048
        dense_sparse = os.path.join(dense, 'sparse')
        undistorted_images = os.path.join(dense, 'images')

        texture = bake_texture_from_photos(
            new_verts, new_faces, uvs,
            undistorted_images, dense_sparse,
            TEX_SIZE, tag
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


def bake_texture_from_photos(verts, faces, uvs, image_dir, sparse_dir, tex_size, tag):
    """
    Project original photos onto the UV texture map using COLMAP camera poses.

    Uses per-texel BEST-CAMERA selection: for each UV pixel, pick the single
    camera with the highest face-to-camera dot product (most face-on view).
    This prevents color mixing between cameras (which causes the mosaic artefact)
    and matches how Reality Capture / Meshroom bake textures.

    Two-pass approach (no redundant image I/O):
      Pass 1 — geometry only: determine which camera wins each texel.
      Pass 2 — sampling: load each image once, write all its texels.
    """
    import cv2
    import numpy as np
    from PIL import Image

    try:
        cameras = _read_colmap_cameras_txt(os.path.join(sparse_dir, 'cameras.txt'))
        images  = _read_colmap_images_txt(os.path.join(sparse_dir, 'images.txt'))
        if not cameras or not images:
            logging.warning(f'[{tag}] No COLMAP cameras/images found — skipping photo bake')
            return None

        # Filter to images that exist on disk and have valid intrinsics
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
            logging.warning(f'[{tag}] No valid image files found on disk')
            return None

        verts_np = np.array(verts, dtype=np.float64)
        faces_np = np.array(faces, dtype=np.int32)
        uvs_np   = np.array(uvs,   dtype=np.float32)

        # ── Build UV→3D position + normal map ──────────────────────────
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

            n = np.cross(v1 - v0, v2 - v0)
            n_len = np.linalg.norm(n)
            if n_len < 1e-10:
                continue
            n = n / n_len

            # UV → pixel coords (flip V: UV origin bottom-left, image origin top-left)
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
        M = len(valid_tv)
        logging.info(f'[{tag}] UV rasterised: {M} texels')

        # ── Pass 1 — geometry only, no image I/O ───────────────────────
        # Goals:
        #   A. Per-UV-island camera assignment: every texel in an island
        #      uses the SAME camera → zero colour switching mid-island.
        #   B. Cache every camera's pixel coords per texel so Pass 2
        #      doesn't need to reproject.
        #   C. Count texels per camera to identify the dominant (reference)
        #      camera for colour normalisation.
        #
        # Why per-island (not per-texel)?
        #   Per-texel switches cameras frequently across a curved surface.
        #   Each switch = a visible seam if cameras differ in exposure/WB.
        #   Per-island = the entire UV patch is one photo crop → seamless
        #   within the patch, same as how RC / Meshroom bake textures.

        # --- Find UV islands (connected components of rasterised UV) ---
        num_islands, island_map = cv2.connectedComponents(has_data.astype(np.uint8))
        island_ids = island_map[valid_tv, valid_tu]   # (M,) — island per texel
        logging.info(f'[{tag}] xatlas UV has {num_islands - 1} islands')

        # --- Project all cameras, accumulate per-island scores ----------
        island_cam_score = np.zeros((num_islands, len(cam_list)), dtype=np.float64)
        cam_texel_count  = np.zeros(len(cam_list), dtype=np.int64)

        # Store pixel coords for every (texel, camera) — used in Pass 2.
        # Shape: (M, N_cams). For typical M≈1M and N_cams≈20 → ~80 MB, fine.
        all_img_x = np.zeros((M, len(cam_list)), dtype=np.float32)
        all_img_y = np.zeros((M, len(cam_list)), dtype=np.float32)

        for ci, c in enumerate(cam_list):
            R, t   = c['R'], c['t']
            cam    = c['cam']
            cam_pos = -R.T @ t

            pts_cam  = (R @ valid_pos.T).T + t
            in_front = pts_cam[:, 2] > 0.01

            fx, fy, cx_c, cy_c = cam['fx'], cam['fy'], cam['cx'], cam['cy']
            iw, ih = cam['w'], cam['h']
            iz    = np.where(in_front, 1.0 / (pts_cam[:, 2] + 1e-10), 0.0)
            img_x = fx * pts_cam[:, 0] * iz + cx_c
            img_y = fy * pts_cam[:, 1] * iz + cy_c

            in_bounds = (img_x >= 0) & (img_x < iw - 1) & \
                        (img_y >= 0) & (img_y < ih - 1)
            vd  = cam_pos - valid_pos
            dot = np.einsum('ij,ij->i', valid_n,
                            vd / (np.linalg.norm(vd, axis=1, keepdims=True) + 1e-10))

            visible = in_front & in_bounds & (dot > 0.1)
            np.add.at(island_cam_score[:, ci], island_ids[visible], dot[visible])
            cam_texel_count[ci] = int(visible.sum())

            all_img_x[:, ci] = img_x.astype(np.float32)
            all_img_y[:, ci] = img_y.astype(np.float32)

        # --- Per island: assign the camera with the highest total score --
        island_best = np.argmax(island_cam_score, axis=1)        # (num_islands,)
        island_best[island_cam_score.max(axis=1) <= 0] = -1      # uncovered islands

        best_cam_id  = island_best[island_ids].astype(np.int32)  # (M,)
        cam_idx_safe = np.clip(best_cam_id, 0, len(cam_list) - 1)
        best_img_x   = all_img_x[np.arange(M), cam_idx_safe].astype(np.float64)
        best_img_y   = all_img_y[np.arange(M), cam_idx_safe].astype(np.float64)
        best_cam_id[island_ids == 0] = -1   # background texels
        del all_img_x, all_img_y

        claimed = (best_cam_id >= 0).sum()
        logging.info(f'[{tag}] Island assignment: {(island_best>=0).sum()}/'
                     f'{num_islands-1} islands → {claimed}/{M} texels '
                     f'({claimed/M*100:.1f}%)')
        if claimed == 0:
            logging.warning(f'[{tag}] No texels claimed')
            return None

        # ── Pass 2 — load images, colour-normalise, sample ─────────────
        # Why colour normalisation?
        #   Different photos have different exposures / white-balances.
        #   Without correction, the boundary between two island-cameras
        #   is visible as a colour step even though the seam is geometrically
        #   correct. Normalising all cameras to the same statistics removes
        #   that step.
        #
        # Method: match each camera's per-channel mean + std to the
        # dominant camera (the one assigned the most texels = best-lit,
        # most face-on). This is the standard approach in remote sensing
        # and photogrammetry for multi-image colour balancing.

        ref_ci  = int(cam_texel_count.argmax())
        ref_img = cv2.cvtColor(cv2.imread(cam_list[ref_ci]['path']),
                               cv2.COLOR_BGR2RGB).astype(np.float32)
        # Compute reference stats on skin pixels only (not pure black/white)
        ref_flat = ref_img.reshape(-1, 3)
        skin_mask = (ref_flat.max(axis=1) > 20) & (ref_flat.max(axis=1) < 250)
        ref_mean = ref_flat[skin_mask].mean(axis=0) if skin_mask.sum() > 100 \
                   else ref_flat.mean(axis=0)
        ref_std  = ref_flat[skin_mask].std(axis=0)  + 1e-6 if skin_mask.sum() > 100 \
                   else ref_flat.std(axis=0) + 1e-6
        logging.info(f'[{tag}] Colour ref: {cam_list[ref_ci]["name"]} '
                     f'mean={ref_mean.round(1)} std={ref_std.round(1)}')

        result_rgb = np.zeros((M, 3), dtype=np.float32)

        for ci, c in enumerate(cam_list):
            mask = best_cam_id == ci
            if not mask.any():
                continue

            img_bgr = cv2.imread(c['path'])
            if img_bgr is None:
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

            # Colour normalisation: shift + scale per channel so this
            # camera's skin statistics match the reference camera.
            flat = img.reshape(-1, 3)
            sk   = (flat.max(axis=1) > 20) & (flat.max(axis=1) < 250)
            if sk.sum() > 100:
                c_mean = flat[sk].mean(axis=0)
                c_std  = flat[sk].std(axis=0) + 1e-6
            else:
                c_mean = flat.mean(axis=0)
                c_std  = flat.std(axis=0) + 1e-6
            img_norm = ((img - c_mean) / c_std * ref_std + ref_mean)
            img_norm = np.clip(img_norm, 0, 255)

            ih_i, iw_i = img_norm.shape[:2]
            sx = best_img_x[mask]; sy = best_img_y[mask]
            x0 = np.clip(sx.astype(int), 0, iw_i - 2)
            y0 = np.clip(sy.astype(int), 0, ih_i - 2)
            xf = sx - x0; yf = sy - y0

            sampled = (img_norm[y0,   x0  ] * ((1-xf)*(1-yf))[:, None] +
                       img_norm[y0,   x0+1] * (   xf *(1-yf))[:, None] +
                       img_norm[y0+1, x0  ] * ((1-xf)*   yf )[:, None] +
                       img_norm[y0+1, x0+1] * (   xf *   yf )[:, None])
            result_rgb[mask] = sampled / 255.0
            logging.info(f'[{tag}] {c["name"]}: {mask.sum()} texels '
                         f'(norm mean {c_mean.round(0)}→{ref_mean.round(0)})')

        # ── Compose + seam feathering + inpaint ────────────────────────
        result = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
        has_cam = best_cam_id >= 0
        result[valid_tv[has_cam], valid_tu[has_cam]] = \
            (result_rgb[has_cam] * 255).clip(0, 255).astype(np.uint8)

        # Seam feathering: blend a narrow band (~5 px) at island boundaries
        # so any residual colour step after normalisation is smoothed out.
        # We detect seams by finding has_data pixels whose neighbourhood
        # contains a different island label, then Gaussian-blur just those.
        cam_label = np.full((tex_size, tex_size), -1, dtype=np.int32)
        cam_label[valid_tv[has_cam], valid_tu[has_cam]] = best_cam_id[has_cam]
        kern = np.ones((9, 9), np.uint8)
        seam = np.zeros((tex_size, tex_size), dtype=bool)
        for ci in range(len(cam_list)):
            m = (cam_label == ci).astype(np.uint8)
            if not m.any():
                continue
            eroded = cv2.erode(m, kern)
            seam  |= (m.astype(bool)) & (~eroded.astype(bool))
        if seam.any():
            blurred = cv2.GaussianBlur(result, (0, 0), 3.0)
            result[seam] = blurred[seam]

        # Inpaint any remaining uncovered UV pixels
        inpaint_mask = (~has_data).astype(np.uint8) * 255
        no_cam_pix   = valid_tv[~has_cam], valid_tu[~has_cam]
        if len(no_cam_pix[0]):
            inpaint_mask[no_cam_pix] = 255
        if inpaint_mask.any():
            result = cv2.inpaint(result, inpaint_mask, 5, cv2.INPAINT_TELEA)

        pct = has_cam.sum() / M * 100
        logging.info(f'[{tag}] Bake complete: {pct:.1f}% of UV from photos')
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
