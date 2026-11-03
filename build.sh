#!/usr/bin/env bash
# Build with a local Docker:
#   ./build.sh base  <user>         -> ghcr.io/<user>/pod-base-builder:cu128 and ghcr.io/<user>/pod-base:cu128
#   ./build.sh lyra2 <user>         -> ghcr.io/<user>/pod-lyra2:cu128 (on top of the two above)
#   PUSH=1 ./build.sh base <user>   -> also push (docker login ghcr.io first)
set -euo pipefail

name=${1:?usage: ./build.sh <base|lyra2> <user>}
user=${2:?usage: ./build.sh <base|lyra2> <user>}
tag=${TAG:-cu128}
cd "$(dirname "$0")"

[ -f "$name/Dockerfile" ] || { echo "no such image: $name" >&2; exit 1; }

build() { # build <image> <docker build args...>
  local image="ghcr.io/$user/$1:$tag"; shift
  docker build --platform linux/amd64 -t "$image" "$@"
  echo "built $image"
  if [ "${PUSH:-0}" = 1 ]; then docker push "$image"; fi
}

if [ "$name" = base ]; then
  build pod-base-builder --target builder base
  build pod-base --target runtime base
else
  build "pod-$name" \
    --build-arg "BUILDER=ghcr.io/$user/pod-base-builder:$tag" \
    --build-arg "BASE=ghcr.io/$user/pod-base:$tag" \
    "$name"
fi
