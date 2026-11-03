#!/usr/bin/env bash
# Build with a local Docker:
#   ./build.sh base  <user>         -> ghcr.io/<user>/pod-base:cu128
#   ./build.sh lyra2 <user>         -> ghcr.io/<user>/pod-lyra2:cu128 (on top of ghcr.io/<user>/pod-base:cu128)
#   PUSH=1 ./build.sh base <user>   -> also push (docker login ghcr.io first)
set -euo pipefail

name=${1:?usage: ./build.sh <base|lyra2> <user>}
user=${2:?usage: ./build.sh <base|lyra2> <user>}
tag=${TAG:-cu128}
cd "$(dirname "$0")"

[ -f "$name/Dockerfile" ] || { echo "no such image: $name" >&2; exit 1; }

image="ghcr.io/$user/pod-$name:$tag"
args=()
if [ "$name" != base ]; then
  args+=(--build-arg "BASE=ghcr.io/$user/pod-base:$tag")
fi

docker build --platform linux/amd64 "${args[@]}" -t "$image" "$name"
echo "built $image"

if [ "${PUSH:-0}" = 1 ]; then
  docker push "$image"
fi
