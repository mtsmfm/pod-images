#!/usr/bin/env bash
# Compare a zstd tag with the gzip tag of the same build (brief 7), as Markdown lines, also appended to
# $GITHUB_STEP_SUMMARY when set:
#   1. every layer of the zstd tag is application/vnd.oci.image.layer.v1.tar+zstd
#   2. rootfs.diff_ids (the uncompressed layers) are the same in both, in order
#   3. (optional) `docker pull` time of each, PULLS times (after `docker system prune -af` each time);
#      the zstd image is left pulled
#   .github/scripts/zstd-check.sh ghcr.io/OWNER/IMAGE:GZIP_TAG ghcr.io/OWNER/IMAGE:ZSTD_TAG [PULLS]
# Exits 1 when 1 or 2 fails.
set -euo pipefail
gzip_ref=$1 zstd_ref=$2 pulls=${3:-0}
out() { echo "$1"; if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then echo "$1" >> "$GITHUB_STEP_SUMMARY"; fi; }
manifest() { # the amd64 manifest of a tag (an index is resolved)
  local raw; raw=$(docker buildx imagetools inspect --raw "$1")
  if jq -e '.manifests' >/dev/null <<<"$raw"; then
    raw=$(docker buildx imagetools inspect --raw "${1%%:*}@$(jq -r '[.manifests[] | select(.platform.architecture == "amd64")][0].digest' <<<"$raw")")
  fi
  echo "$raw"
}
diff_ids() { docker buildx imagetools inspect "$1" --format '{{json .Image}}' | jq -r '(.["linux/amd64"] // .).rootfs.diff_ids[]'; }
ok=0

types=$(manifest "$zstd_ref" | jq -r '.layers[].mediaType' | sort | uniq -c | sed 's/^ *//')
out "- \`$zstd_ref\` layer types: $(tr '\n' ';' <<<"$types" | sed 's/;$//; s/;/; /g')"
if manifest "$zstd_ref" | jq -e '[.layers[].mediaType] | all(. == "application/vnd.oci.image.layer.v1.tar+zstd") | not' >/dev/null; then
  out "  - error: not every layer is zstd"; ok=1
fi

a=$(diff_ids "$gzip_ref") b=$(diff_ids "$zstd_ref")
if [ "$a" = "$b" ]; then
  out "  - diff_ids: the same as \`$gzip_ref\` ($(wc -l <<<"$a") layers)"
else
  bad=$(diff <(nl -v0 <<<"$a") <(nl -v0 <<<"$b") | sed -nE 's/^< +([0-9]+).*/\1/p' | tr '\n' ' ')
  out "  - error: diff_ids differ from \`$gzip_ref\` (layers $bad; counts $(wc -l <<<"$a") / $(wc -l <<<"$b"))"; ok=1
fi

for i in $(seq "$pulls"); do
  for ref in "$gzip_ref" "$zstd_ref"; do
    docker system prune -af >/dev/null
    start=$(date +%s.%N); docker pull -q "$ref" >/dev/null; end=$(date +%s.%N)
    out "  - pull $i of \`$ref\`: $(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.1f", e - s }') s"
  done
done
exit $ok
