import os
import uuid
import zipfile
import subprocess
import threading
import json
import shutil
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/skinmapper_jobs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory job store (use Redis in production)
jobs = {}
jobs_lock = threading.Lock()

def update_job(job_id, status, progress, message, result_url=None, error=None):
    with jobs_lock:
        jobs[job_id] = {
            'status': status,
            'progress': progress,
            'message': message,
            'result_url': result_url,
            'error': error
        }

def run_colmap(job_id, image_dir, output_dir):
    """Full COLMAP + OpenMVS pipeline to produce a textured USDZ."""
    try:
        sparse_dir = os.path.join(output_dir, 'sparse')
        dense_dir  = os.path.join(output_dir, 'dense')
        mvs_dir    = os.path.join(output_dir, 'mvs')
        os.makedirs(sparse_dir, exist_ok=True)
        os.makedirs(dense_dir,  exist_ok=True)
        os.makedirs(mvs_dir,    exist_ok=True)

        def run(cmd, progress, msg):
            update_job(job_id, 'processing', progress, msg)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"{msg} failed:\n{result.stderr[-500:]}")
            return result

        # 1. Feature extraction
        run([
            'colmap', 'feature_extractor',
            '--database_path', os.path.join(output_dir, 'db.db'),
            '--image_path', image_dir,
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.use_gpu', '0',
            '--SiftExtraction.max_image_size', '1600'
        ], 0.10, 'Extracting image features…')

        # 2. Feature matching
        run([
            'colmap', 'exhaustive_matcher',
            '--database_path', os.path.join(output_dir, 'db.db'),
            '--SiftMatching.use_gpu', '0'
        ], 0.25, 'Matching features across photos…')

        # 3. Sparse reconstruction (Structure from Motion)
        run([
            'colmap', 'mapper',
            '--database_path', os.path.join(output_dir, 'db.db'),
            '--image_path', image_dir,
            '--output_path', sparse_dir,
            '--Mapper.num_threads', '4',
            '--Mapper.extract_colors', '1'
        ], 0.40, 'Reconstructing 3D structure…')

        sparse_model = os.path.join(sparse_dir, '0')
        if not os.path.exists(sparse_model):
            raise RuntimeError('Sparse reconstruction failed — not enough matching features. Try taking more overlapping photos.')

        # 4. Convert to TXT for OpenMVS
        run([
            'colmap', 'model_converter',
            '--input_path', sparse_model,
            '--output_path', sparse_model,
            '--output_type', 'TXT'
        ], 0.45, 'Converting model format…')

        # 5. Undistort images
        run([
            'colmap', 'image_undistorter',
            '--image_path', image_dir,
            '--input_path', sparse_model,
            '--output_path', dense_dir,
            '--output_type', 'COLMAP'
        ], 0.50, 'Undistorting images…')

        # 6. OpenMVS: dense point cloud
        mvs_scene = os.path.join(mvs_dir, 'scene.mvs')
        run([
            'InterfaceCOLMAP',
            '-i', dense_dir,
            '-o', mvs_scene
        ], 0.55, 'Building dense point cloud…')

        run([
            'DensifyPointCloud',
            '-i', mvs_scene,
            '-o', os.path.join(mvs_dir, 'scene_dense.mvs'),
            '--resolution-level', '1',
            '--number-views', '5'
        ], 0.65, 'Densifying point cloud…')

        # 7. Mesh reconstruction
        run([
            'ReconstructMesh',
            '-i', os.path.join(mvs_dir, 'scene_dense.mvs'),
            '-o', os.path.join(mvs_dir, 'scene_mesh.mvs')
        ], 0.75, 'Reconstructing mesh surface…')

        # 8. Texture baking
        run([
            'TextureMesh',
            '-i', os.path.join(mvs_dir, 'scene_mesh.mvs'),
            '-o', os.path.join(mvs_dir, 'scene_textured.mvs'),
            '--export-type', 'obj',
            '--decimate', '0.5'
        ], 0.85, 'Baking texture…')

        # 9. Convert OBJ → USDZ using Reality Converter (or Python fallback)
        obj_file = os.path.join(mvs_dir, 'scene_textured.obj')
        usdz_file = os.path.join(output_dir, f'{job_id}.usdz')

        if not os.path.exists(obj_file):
            # Try .ply fallback
            ply_file = os.path.join(mvs_dir, 'scene_dense.ply')
            if os.path.exists(ply_file):
                obj_file = convert_ply_to_obj(ply_file, mvs_dir)
            else:
                raise RuntimeError('Mesh export failed — no output geometry found.')

        convert_obj_to_usdz(obj_file, usdz_file)
        update_job(job_id, 'done', 1.0, 'Reconstruction complete!',
                   result_url=f'/result/{job_id}')

    except Exception as e:
        update_job(job_id, 'error', 0, str(e), error=str(e))
    finally:
        # Clean up working files, keep only the USDZ
        for d in [sparse_dir, dense_dir, mvs_dir, image_dir]:
            shutil.rmtree(d, ignore_errors=True)
        db = os.path.join(output_dir, 'db.db')
        if os.path.exists(db):
            os.remove(db)


def convert_ply_to_obj(ply_path, out_dir):
    """Convert PLY to OBJ using open3d."""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply_path)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    obj_path = os.path.join(out_dir, 'fallback.obj')
    o3d.io.write_triangle_mesh(obj_path, mesh)
    return obj_path


def convert_obj_to_usdz(obj_path, usdz_path):
    """Convert OBJ/MTL to USDZ using usd-core Python package."""
    try:
        from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
        import re

        # Parse OBJ
        verts, uvs, normals, faces = [], [], [], []
        mtl_file = None
        tex_file = None

        obj_dir = os.path.dirname(obj_path)
        with open(obj_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] == 'v':
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif parts[0] == 'vt':
                    uvs.append((float(parts[1]), float(parts[2])))
                elif parts[0] == 'vn':
                    normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif parts[0] == 'f':
                    face = []
                    for p in parts[1:]:
                        idx = [int(x)-1 if x else 0 for x in p.split('/')]
                        while len(idx) < 3: idx.append(0)
                        face.append(idx)
                    faces.append(face)
                elif parts[0] == 'mtllib':
                    mtl_file = os.path.join(obj_dir, parts[1])

        # Find texture from MTL
        if mtl_file and os.path.exists(mtl_file):
            with open(mtl_file) as f:
                for line in f:
                    if line.strip().lower().startswith('map_kd'):
                        tex_name = line.strip().split()[-1]
                        tex_file = os.path.join(obj_dir, tex_name)
                        break

        # Build USD
        stage = Usd.Stage.CreateNew(usdz_path.replace('.usdz', '.usda'))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        mesh_prim = UsdGeom.Mesh.Define(stage, '/Mesh')

        # Vertices
        flat_verts = [Gf.Vec3f(*v) for v in verts]
        mesh_prim.GetPointsAttr().Set(flat_verts)

        # Faces
        face_vert_counts = []
        face_vert_indices = []
        uv_indices = []
        for face in faces:
            face_vert_counts.append(len(face))
            for vi in face:
                face_vert_indices.append(vi[0])
                uv_indices.append(vi[1] if len(vi) > 1 else 0)
        mesh_prim.GetFaceVertexCountsAttr().Set(face_vert_counts)
        mesh_prim.GetFaceVertexIndicesAttr().Set(face_vert_indices)

        # UVs
        if uvs:
            st = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
                'st', Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.faceVarying
            )
            st.Set([Gf.Vec2f(*uv) for uv in uvs])
            st.SetIndices(uv_indices)

        stage.GetRootLayer().Save()

        # Package as USDZ
        usda_path = usdz_path.replace('.usdz', '.usda')
        subprocess.run([
            'usdzip', usdz_path, usda_path
        ] + ([tex_file] if tex_file and os.path.exists(tex_file) else []),
            check=True, capture_output=True
        )
        os.remove(usda_path)

    except Exception as e:
        # Fallback: just zip the OBJ as a USDZ-named archive the app can unpack
        with zipfile.ZipFile(usdz_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(obj_path, os.path.basename(obj_path))
            mtl = obj_path.replace('.obj', '.mtl')
            if os.path.exists(mtl): zf.write(mtl, os.path.basename(mtl))
            for ext in ['.png', '.jpg', '.jpeg']:
                tex = obj_path.replace('.obj', ext)
                if os.path.exists(tex):
                    zf.write(tex, os.path.basename(tex))


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    colmap_ok = shutil.which('colmap') is not None
    return jsonify({
        'status': 'ok',
        'colmap': colmap_ok,
        'version': '1.0.0'
    })


@app.route('/submit', methods=['POST'])
def submit():
    """Accept zipped photos, start reconstruction job."""
    if 'photos' not in request.files:
        return jsonify({'error': 'No photos file provided'}), 400

    file = request.files['photos']
    if not file.filename.endswith('.zip'):
        return jsonify({'error': 'Expected a .zip file of photos'}), 400

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(UPLOAD_FOLDER, job_id)
    image_dir = os.path.join(job_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    # Save and extract zip
    zip_path = os.path.join(job_dir, 'photos.zip')
    file.save(zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            name = os.path.basename(member)
            if name.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                with zf.open(member) as src, open(os.path.join(image_dir, name), 'wb') as dst:
                    dst.write(src.read())
    os.remove(zip_path)

    image_count = len(os.listdir(image_dir))
    if image_count < 10:
        shutil.rmtree(job_dir, ignore_errors=True)
        return jsonify({'error': f'Only {image_count} images found. Please provide at least 10.'}), 400

    # Start background reconstruction
    update_job(job_id, 'queued', 0.0, f'Job queued — {image_count} photos received')
    thread = threading.Thread(target=run_colmap, args=(job_id, image_dir, job_dir), daemon=True)
    thread.start()

    return jsonify({'job_id': job_id, 'image_count': image_count}), 202


@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    """Poll job status and progress."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        # Check if result file exists (server restart case)
        result_path = os.path.join(UPLOAD_FOLDER, job_id, f'{job_id}.usdz')
        if os.path.exists(result_path):
            return jsonify({'status': 'done', 'progress': 1.0,
                           'message': 'Complete', 'result_url': f'/result/{job_id}'})
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


@app.route('/result/<job_id>', methods=['GET'])
def result(job_id):
    """Download the completed USDZ."""
    usdz_path = os.path.join(UPLOAD_FOLDER, job_id, f'{job_id}.usdz')
    if not os.path.exists(usdz_path):
        return jsonify({'error': 'Result not found'}), 404
    return send_file(usdz_path, mimetype='model/vnd.usdz+zip',
                    as_attachment=True, download_name='scan.usdz')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, threaded=True)
