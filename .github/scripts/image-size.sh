#!/usr/bin/env bash
# Print the compressed size of an image (sum of the amd64 manifest's layer sizes).
#   .github/scripts/image-size.sh ghcr.io/OWNER/IMAGE:TAG ...
set -euo pipefail
for ref in "$@"; do
  raw=$(docker buildx imagetools inspect --raw "$ref")
  if jq -e '.manifests' >/dev/null <<<"$raw"; then
    digest=$(jq -r '[.manifests[] | select(.platform.architecture == "amd64")][0].digest' <<<"$raw")
    raw=$(docker buildx imagetools inspect --raw "${ref%%:*}@$digest")
  fi
  bytes=$(jq '[.layers[].size] | add' <<<"$raw")
  echo "$ref: $(numfmt --to=si --suffix=B "$bytes") compressed ($bytes bytes, $(jq '.layers | length' <<<"$raw") layers)"
done
