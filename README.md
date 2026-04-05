# SkinMapper Photogrammetry Server

Runs COLMAP + OpenMVS to reconstruct 3D meshes from photos.
Deployed on Railway — works on any device, no LiDAR required.

## Deploy to Railway (free)

1. Go to railway.app → New Project → Deploy from GitHub
2. Push this folder to a GitHub repo
3. Railway auto-detects the Dockerfile and deploys
4. Copy the generated URL (e.g. https://skinmapper-server.up.railway.app)
5. Paste it into the iOS app's ServerConfig.swift

## API

POST /submit          — upload photos.zip, get job_id back
GET  /status/{job_id} — poll progress (0.0 → 1.0)
GET  /result/{job_id} — download completed .usdz
GET  /health          — server health check
