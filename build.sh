#!/usr/bin/env bash
# Build with a local Docker (the same images CI builds):
#   ./build.sh base <user>                 -> ghcr.io/<user>/pod-base-builder:cu128, pod-base:cu128 (torch 2.7.1)
#   ./build.sh base <user> cu128-torch280  -> ghcr.io/<user>/pod-base-builder:cu128-torch280, pod-base:cu128-torch280
#   ./build.sh base <user> cu130           -> ghcr.io/<user>/pod-base-builder:cu130, pod-base:cu130 (CUDA 13.0, torch 2.9.1)
#   ./build.sh lyra2 <user>                -> ghcr.io/<user>/pod-lyra2:cu128 (on pod-base:cu128)
#   ./build.sh comfy <user> [cu130]        -> ghcr.io/<user>/pod-comfy:cu128 (on pod-base:cu128-torch280) or :cu130
#   ./build.sh train <user>                -> ghcr.io/<user>/pod-train:cu128 (on pod-base:cu128-torch280)
#   ./build.sh lyra2-rp <user>             -> ghcr.io/<user>/pod-lyra2-rp:cu128 (on runpod/pytorch)
#   PUSH=1 ./build.sh ...                  -> also push (docker login ghcr.io first)
set -euo pipefail

usage="usage: ./build.sh <base|lyra2|comfy|train|lyra2-rp> <user> [variant]"
name=${1:?$usage}
user=${2:?$usage}
cd "$(dirname "$0")"

[ -f "$name/Dockerfile" ] || { echo "no such image: $name" >&2; exit 1; }

build() { # build <image:tag> <docker build args...>
  local image="ghcr.io/$user/$1"; shift
  docker build --platform linux/amd64 -t "$image" "$@"
  echo "built $image"
  if [ "${PUSH:-0}" = 1 ]; then docker push "$image"; fi
}

on_base() { # on_base <base tag> <image:tag> <dir> [docker build args...]
  local base=$1 image=$2 dir=$3; shift 3
  build "$image" \
    --build-arg "BUILDER=ghcr.io/$user/pod-base-builder:$base" \
    --build-arg "BASE=ghcr.io/$user/pod-base:$base" \
    "$@" "$dir"
}

case "$name" in
  base)
    variant=${3:-cu128}
    case "$variant" in
      cu128) torch=(--build-arg TORCH=2.7.1 --build-arg TORCHVISION=0.22.1) ;;
      cu128-torch280) torch=(--build-arg TORCH=2.8.0 --build-arg TORCHVISION=0.23.0) ;;
      cu130) torch=(--build-arg TORCH=2.9.1 --build-arg TORCHVISION=0.24.1 --build-arg TORCH_INDEX=cu130
                    --build-arg BUILDER_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04
                    --build-arg RUNTIME_IMAGE=nvidia/cuda:13.0.3-base-ubuntu22.04) ;;
      *) echo "unknown base variant: $variant (cu128, cu128-torch280 or cu130)" >&2; exit 1 ;;
    esac
    build "pod-base-builder:$variant" --target builder "${torch[@]}" base
    build "pod-base:$variant" --target runtime "${torch[@]}" base
    ;;
  lyra2) on_base cu128 pod-lyra2:cu128 lyra2 ;;
  comfy)
    case "${3:-cu128}" in
      cu128) on_base cu128-torch280 pod-comfy:cu128 comfy --build-arg TORCH_CUDA=12.8 ;;
      cu130) on_base cu130 pod-comfy:cu130 comfy --build-arg TORCH_CUDA=13.0 ;;
      *) echo "unknown comfy variant: $3 (cu128 or cu130)" >&2; exit 1 ;;
    esac ;;
  train) on_base cu128-torch280 pod-train:cu128 train ;;
  lyra2-rp) build pod-lyra2-rp:cu128 --build-context lyra2=lyra2 lyra2-rp ;;
esac
