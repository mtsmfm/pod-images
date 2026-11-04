#!/usr/bin/env bash
# Print the compressed size of an image (sum of the amd64 manifest's layer sizes), one Markdown line each,
# also appended to $GITHUB_STEP_SUMMARY when set (in Actions).
#   [SIZE_LIMIT_GB=6] .github/scripts/image-size.sh ghcr.io/OWNER/IMAGE:TAG ...
# Above SIZE_LIMIT_GB (decimal GB; unset or 0 for no limit), also print a GitHub Actions warning.
set -euo pipefail
limit=${SIZE_LIMIT_GB:-0}
for ref in "$@"; do
  raw=$(docker buildx imagetools inspect --raw "$ref")
  if jq -e '.manifests' >/dev/null <<<"$raw"; then
    digest=$(jq -r '[.manifests[] | select(.platform.architecture == "amd64")][0].digest' <<<"$raw")
    raw=$(docker buildx imagetools inspect --raw "${ref%%:*}@$digest")
  fi
  bytes=$(jq '[.layers[].size] | add' <<<"$raw")
  gb=$(awk -v b="$bytes" 'BEGIN { printf "%.2f", b / 1e9 }')
  line="- \`$ref\`: **$gb GB** compressed ($bytes bytes, $(jq '.layers | length' <<<"$raw") layers)"
  if awk -v g="$gb" -v l="$limit" 'BEGIN { exit !(l > 0 && g > l) }'; then
    line="$line, above the $limit GB target"
    echo "::warning title=Image size::$ref is $gb GB compressed, above the $limit GB target"
  fi
  echo "$line"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then echo "$line" >> "$GITHUB_STEP_SUMMARY"; fi
done
