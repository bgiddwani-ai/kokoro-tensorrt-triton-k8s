#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
action="${1:-render}"
namespace="${NAMESPACE:-kokoro}"
image="${IMAGE:-ghcr.io/example/kokoro:latest}"
workers="${WORKERS:-1}"
mps="${MPS:-0}"
workspace="${WORKSPACE_MIB:-2048}"
max_tokens="${MAX_TOKENS:-500}"
max_frames="${MAX_FRAMES:-2000}"
cache_epoch="${CACHE_EPOCH:-1}"
storage_class="${STORAGE_CLASS:-}"
runtime_class="${RUNTIME_CLASS:-}"

[[ "$namespace" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]] || { echo "Invalid namespace" >&2; exit 2; }
[[ "$image" =~ ^[a-zA-Z0-9._/@:-]+$ ]] || { echo "Invalid container image" >&2; exit 2; }
for value in "$workers" "$workspace" "$max_tokens" "$max_frames" "$cache_epoch"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "Expected integer configuration" >&2; exit 2; }
done
[[ "$mps" == 0 || "$mps" == 1 ]] || { echo "MPS must be 0 or 1" >&2; exit 2; }
for value in "$storage_class" "$runtime_class"; do
    [[ -z "$value" || "$value" =~ ^[a-z0-9.-]+$ ]] || { echo "Invalid class name" >&2; exit 2; }
done
command -v kubectl >/dev/null || { echo "Install kubectl first" >&2; exit 2; }

temporary="$(mktemp -d "${repo_root}/.generated.XXXXXX")"
trap 'rm -rf "$temporary"' EXIT
cat > "$temporary/kustomization.yaml" <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: ${namespace}
resources:
  - ../k8s/base
configMapGenerator:
  - name: kokoro-settings
    behavior: merge
    literals:
      - KOKORO_WORKERS=${workers}
      - KOKORO_MPS=${mps}
      - KOKORO_WORKSPACE_MIB=${workspace}
      - KOKORO_MAX_TOKENS=${max_tokens}
      - KOKORO_MAX_FRAMES=${max_frames}
      - KOKORO_CACHE_EPOCH=${cache_epoch}
patches:
  - target:
      kind: Namespace
      name: kokoro
    patch: |-
      - op: replace
        path: /metadata/name
        value: ${namespace}
  - target:
      kind: Deployment
      name: kokoro
    patch: |-
      - op: replace
        path: /spec/template/spec/containers/0/image
        value: "${image}"
EOF
if [[ -n "$runtime_class" ]]; then
    cat >> "$temporary/kustomization.yaml" <<EOF
      - op: add
        path: /spec/template/spec/runtimeClassName
        value: ${runtime_class}
EOF
fi
if [[ -n "$storage_class" ]]; then
    cat >> "$temporary/kustomization.yaml" <<EOF
  - target:
      kind: PersistentVolumeClaim
      name: kokoro-data
    patch: |-
      - op: add
        path: /spec/storageClassName
        value: ${storage_class}
EOF
fi
case "$action" in
    render) kubectl kustomize "$temporary" ;;
    deploy)
        if [[ "$image" == ghcr.io/example/* ]]; then
            echo "Set IMAGE to a registry image your cluster can pull" >&2; exit 2
        fi
        kubectl kustomize "$temporary" | kubectl apply -f -
        kubectl -n "$namespace" rollout status deployment/kokoro --timeout=30m
        ;;
    forward) kubectl -n "$namespace" port-forward service/kokoro 8000:8000 8001:8001 8002:8002 ;;
    logs) kubectl -n "$namespace" logs -f deployment/kokoro ;;
    down) kubectl -n "$namespace" delete deployment/kokoro service/kokoro --ignore-not-found ;;
    *) echo "Usage: $0 {render|deploy|forward|logs|down}" >&2; exit 2 ;;
esac
