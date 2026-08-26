#!/usr/bin/env bash
# Block until the running jetspace containers finish, then print their results.
#
# The session teardown killed the log-tailing wrappers but not the containers
# themselves -- `docker compose run` keeps going once started. Rather than
# restart hours of mesh rendering, this re-attaches to work already in flight.
cd "$(dirname "$0")/.."

MAX=${1:-14400}          # give up after 4h rather than hang forever
waited=0
while [ "$waited" -lt "$MAX" ]; do
    n=$(docker ps -q | wc -l)
    [ "$n" -eq 0 ] && break
    sleep 60
    waited=$((waited + 60))
done

echo "containers finished after ${waited}s (or timed out at ${MAX}s)"
echo

for log in /tmp/e6seeds.log /tmp/r1.log; do
    [ -f "$log" ] || continue
    echo "################ $log ################"
    grep -avE "UserWarning|self.blocks|it/s\]|^ *episode [0-9]|Container " "$log" | tail -45
    echo
done
