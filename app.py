import os, uuid, zipfile, subprocess, threading, json, shutil, logging, traceback
from flask import Flask, request, jsonify, send_file

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
JOBS_DIR = '/tmp/skinmapper_jobs'
os.makedirs(JOBS_DIR, exist_ok=True)

jobs = {}
lock = threading.Lock()

def set_job(job_id, status, progress, message, result_url=None, error=None):
    with lock:
        jobs[job_id] = dict(status=status, progress=progress,
                            message=message, result_url=result_url, error=error)

def run_pipeline(job_id, image_dir, job_dir):
    try:
        db      = os.path.join(job_dir, 'db.db')
        sparse  = os.path.join(job_dir, 'sparse')
        dense   = os.path.join(job_dir, 'dense')
        os.makedirs(sparse, exist_ok=True)
        os.makedirs(dense,  exist_ok=True)

        def run(cmd, progress, msg):
            set_job(job_id, 'processing', progress, msg)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(f'{msg} failed: {r.stderr[-400:]}')

        # 1 — Feature extraction
        run(['colmap', 'feature_extractor',
             '--database_path', db,
             '--image_path', image_dir,
             '--ImageReader.single_camera', '1',
             '--SiftExtraction.use_gpu', '0',
             '--SiftExtraction.max_image_size', '1600'],
            0.10, 'Extracting image features…')

        # 2 — Matching
        run(['colmap', 'exhaustive_matcher',
             '--database_path', db,
             '--SiftMatching.use_gpu', '0'],
            0.25, 'Matching features across photos…')

        # 3 — Sparse SfM
        run(['colmap', 'mapper',
             '--database_path', db,
             '--image_path', image_dir,
             '--output_path', sparse,
             '--Mapper.num_threads', '4'],
            0.40, 'Reconstructing 3D structure…')

        sparse0 = os.path.join(sparse, '0')
        if not os.path.exists(sparse0):
            raise RuntimeError(
                'Not enough matching features — please take more overlapping photos and try again.')

        # 4 — Undistort
        run(['colmap', 'image_undistorter',
             '--image_path', image_dir,
             '--input_path', sparse0,
             '--output_path', dense,
             '--output_type', 'COLMAP'],
            0.50, 'Undistorting images…')

        # 5 — Patch-match stereo depth maps
        run(['colmap', 'patch_match_stereo',
             '--workspace_path', dense,
             '--workspace_format', 'COLMAP',
             '--PatchMatchStereo.geom_consistency', 'true'],
            0.60, 'Computing depth maps…')

        # 6 — Stereo fusion → dense PLY
        ply = os.path.join(job_dir, 'dense.ply')
        run(['colmap', 'stereo_fusion',
             '--workspace_path', dense,
             '--workspace_format', 'COLMAP',
             '--input_type', 'geometric',
             '--output_path', ply],
            0.72, 'Fusing depth maps into point cloud…')

        # 7 — Mesh + texture with open3d
        set_job(job_id, 'processing', 0.78, 'Reconstructing mesh surface…')
        import open3d as o3d
        import numpy as np

        pcd = o3d.io.read_point_cloud(ply)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

        set_job(job_id, 'processing', 0.83, 'Building watertight mesh…')
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        # Crop to remove Poisson artefacts at edges
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
        mesh = mesh.simplify_quadric_decimation(50000)
        mesh.compute_vertex_normals()

        set_job(job_id, 'processing', 0.88, 'Baking texture…')
        obj_path = os.path.join(job_dir, 'mesh.obj')
        o3d.io.write_triangle_mesh(obj_path, mesh, write_ascii=False)

        # 8 — Package as USDZ (obj wrapped in zip with .usdz extension)
        # The iOS app's SCNScene loader reads this correctly
        set_job(job_id, 'processing', 0.94, 'Packaging result…')
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

        set_job(job_id, 'done', 1.0, 'Reconstruction complete!',
                result_url=f'/result/{job_id}')

    except Exception as e:
        set_job(job_id, 'error', 0, str(e), error=str(e))
    finally:
        for d in [image_dir, sparse, dense]:
            shutil.rmtree(d, ignore_errors=True)
        for f in [db, os.path.join(job_dir, 'dense.ply'), os.path.join(job_dir, 'mesh.obj')]:
            try: os.remove(f)
            except: pass


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
    import traceback
    tb = traceback.format_exc()
    return jsonify(error=f'{type(e).__name__}: {str(e)}', traceback=tb), 500

@app.route('/health')
def health():
    colmap_ok = shutil.which('colmap') is not None
    try:
        import open3d
        o3d_ok = True
    except ImportError:
        o3d_ok = False
    return jsonify(status='ok', colmap=colmap_ok, open3d=o3d_ok, version='1.1.0')


@app.route('/submit', methods=['POST'])
def submit():
    try:
        logging.info(f'Submit request: content_length={request.content_length}, content_type={request.content_type}')
        if 'photos' not in request.files:
            logging.error(f'No photos field. Form keys: {list(request.files.keys())}')
            return jsonify(error='No photos file'), 400
        f = request.files['photos']
        logging.info(f'Received file: {f.filename}, size approx {request.content_length}')

        job_id  = str(uuid.uuid4())
        job_dir = os.path.join(JOBS_DIR, job_id)
        img_dir = os.path.join(job_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)

        zip_path = os.path.join(job_dir, 'photos.zip')
        f.save(zip_path)
        zip_size = os.path.getsize(zip_path)
        logging.info(f'Saved zip: {zip_size} bytes')

        with zipfile.ZipFile(zip_path) as zf:
            for m in zf.namelist():
                name = os.path.basename(m)
                if name.lower().endswith(('.jpg','.jpeg','.png','.heic')):
                    with zf.open(m) as src, open(os.path.join(img_dir, name), 'wb') as dst:
                        dst.write(src.read())
        os.remove(zip_path)

        count = len(os.listdir(img_dir))
        logging.info(f'Extracted {count} images')
        if count < 10:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify(error=f'Only {count} images — need at least 10'), 400

        set_job(job_id, 'queued', 0.0, f'Queued — {count} photos received')
        threading.Thread(target=run_pipeline, args=(job_id, img_dir, job_dir), daemon=True).start()
        return jsonify(job_id=job_id, image_count=count), 202
    except Exception as e:
        logging.error(f'Submit failed: {traceback.format_exc()}')
        return jsonify(error=f'Submit error: {type(e).__name__}: {str(e)}'), 500


@app.route('/status/<job_id>')
def status(job_id):
    with lock:
        job = jobs.get(job_id)
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
