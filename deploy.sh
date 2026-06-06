#!/usr/bin/env bash
# deploy.sh — build, push, and roll out d4gen to ECS Express Mode.
# Designed for an AWS Workshop Studio account (uses your current WSParticipantRole
# session; no IAM user / access keys needed).
set -euo pipefail

# ---- config (edit if names change) -----------------------------------------
REGION="us-west-2"
ACCOUNT="057629791221"
REPO="d4gen"
CLUSTER="default"
SERVICE="d4gen-c791"
# ----------------------------------------------------------------------------

REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
# Tag with the short git SHA when in a repo, else a timestamp. Always also tag :latest
# because the Express service tracks :latest and --force-new-deployment relies on it.
TAG="$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d-%H%M%S)"
IMAGE="${REGISTRY}/${REPO}"

echo ">> deploying ${IMAGE}:${TAG} to service ${SERVICE} (cluster ${CLUSTER}, ${REGION})"

# 1. Ensure the ECR repo exists (idempotent — ignore "already exists").
aws ecr describe-repositories --repository-names "${REPO}" --region "${REGION}" >/dev/null 2>&1 ||
	aws ecr create-repository --repository-name "${REPO}" --region "${REGION}" >/dev/null

# 2. Authenticate Docker to ECR.
aws ecr get-login-password --region "${REGION}" |
	docker login --username AWS --password-stdin "${REGISTRY}"

# 3. Build for the platform Fargate runs (amd64) and push both tags.
docker build --platform linux/amd64 -t "${IMAGE}:${TAG}" -t "${IMAGE}:latest" .
docker push "${IMAGE}:${TAG}"
docker push "${IMAGE}:latest"

# 4. Force the Express service to pull the new :latest and roll out.
aws ecs update-service \
	--cluster "${CLUSTER}" \
	--service "${SERVICE}" \
	--force-new-deployment \
	--region "${REGION}" >/dev/null

# 5. Block until the rollout stabilizes, then print the public URL.
echo ">> waiting for the service to stabilize (Ctrl-C to stop waiting; deploy still proceeds)..."
aws ecs wait services-stable \
	--cluster "${CLUSTER}" \
	--services "${SERVICE}" \
	--region "${REGION}"

URL="$(aws ecs describe-services \
	--cluster "${CLUSTER}" \
	--services "${SERVICE}" \
	--region "${REGION}" \
	--query 'services[0].deployments[0].serviceConnectConfiguration' \
	--output text 2>/dev/null || true)"

echo ">> deployed. Find the public URL on the Express Mode service page in the console,"
echo "   or via: aws ecs describe-services --cluster ${CLUSTER} --services ${SERVICE} --region ${REGION}"
