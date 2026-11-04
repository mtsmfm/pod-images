#!/usr/bin/env bash
# Build with a local Docker (the same images CI builds):
#   ./build.sh base <user>                 -> ghcr.io/<user>/pod-base-builder:cu128, pod-base:cu128 (torch 2.7.1)
#   ./build.sh base <user> cu128-torch280  -> ghcr.io/<user>/pod-base-builder:cu128-torch280, pod-base:cu128-torch280
#   ./build.sh lyra2 <user>                -> ghcr.io/<user>/pod-lyra2:cu128 (on pod-base:cu128)
#   ./build.sh comfy <user>                -> ghcr.io/<user>/pod-comfy:cu128 (on pod-base:cu128-torch280)
#   ./build.sh train <user>                -> ghcr.io/<user>/pod-train:cu128 (on pod-base:cu128-torch280)
#   ./build.sh lyra2-rp <user>             -> ghcr.io/<user>/pod-lyra2-rp:cu128 (on runpod/pytorch)
#   PUSH=1 ./build.sh ...                  -> also push (docker login ghcr.io first)
set -euo pipefail

usage="usage: ./build.sh <base|lyra2|comfy|train|lyra2-rp> <user> [base variant]"
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

on_base() { # on_base <base tag> <image> <dir>
  build "$2:cu128" \
    --build-arg "BUILDER=ghcr.io/$user/pod-base-builder:$1" \
    --build-arg "BASE=ghcr.io/$user/pod-base:$1" \
    "$3"
}

case "$name" in
  base)
    variant=${3:-cu128}
    case "$variant" in
      cu128) torch=(--build-arg TORCH=2.7.1 --build-arg TORCHVISION=0.22.1) ;;
      cu128-torch280) torch=(--build-arg TORCH=2.8.0 --build-arg TORCHVISION=0.23.0) ;;
      *) echo "unknown base variant: $variant (cu128 or cu128-torch280)" >&2; exit 1 ;;
    esac
    build "pod-base-builder:$variant" --target builder "${torch[@]}" base
    build "pod-base:$variant" --target runtime "${torch[@]}" base
    ;;
  lyra2) on_base cu128 pod-lyra2 lyra2 ;;
  comfy) on_base cu128-torch280 pod-comfy comfy ;;
  train) on_base cu128-torch280 pod-train train ;;
  lyra2-rp) build pod-lyra2-rp:cu128 --build-context lyra2=lyra2 lyra2-rp ;;
esac
