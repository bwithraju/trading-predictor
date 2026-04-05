# Deployment Guide

## Docker Compose (Recommended for single-server)

```bash
# Development
docker compose up --build

# Production (detached)
docker compose build --target production
docker compose up -d
```

## Kubernetes

Apply the manifests in order:

```bash
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

Scale replicas:

```bash
kubectl scale deployment trading-predictor --replicas=4
```

## Environment Variables

Set secrets as Kubernetes Secrets or Docker environment variables. See `.env.example`.

## nginx

The included `nginx.conf` provides:
- Reverse proxy to the FastAPI app
- WebSocket upgrade support
- Health-check passthrough

```bash
# Run nginx alongside the app
docker run -v $(pwd)/nginx.conf:/etc/nginx/conf.d/default.conf nginx:alpine
```

## Publishing to Docker Hub

Push a tag to trigger the deploy workflow:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The `.github/workflows/deploy.yml` workflow builds and pushes to Docker Hub automatically using the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets.

## Database Backup

```bash
bash scripts/backup.sh
# Backups are stored in ./backups/
```
