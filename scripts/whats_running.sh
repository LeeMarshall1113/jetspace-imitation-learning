#!/usr/bin/env bash
# What are the live containers actually doing, and how far along?
cd "$(dirname "$0")/.."
echo "=== live containers ==="
docker ps --format '{{.ID}}  {{.Status}}  {{.Command}}'
echo
for log in /tmp/e6seeds.log /tmp/r1.log; do
    [ -f "$log" ] || continue
    echo "=== $log ($(wc -c < "$log") bytes, modified $(date -r "$log" +%H:%M:%S)) ==="
    grep -avE "UserWarning|self.blocks|it/s\]|^ *episode [0-9]" "$log" | tail -4
    echo
done
echo "=== progress ==="
printf "  E6 arms complete: %s\n" "$(ls cache/e6_h_*.json 2>/dev/null | wc -l)"
printf "  R1 episodes:      %s\n" "$(ls data/episodes/r1_push/episode_*.npz 2>/dev/null | wc -l)"
printf "  R1 latent caches: %s\n" "$(ls -d cache/latents/r1_push__* 2>/dev/null | wc -l)"
