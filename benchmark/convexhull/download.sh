#!/bin/bash
# Downloads the instances listed in instances.txt into ./instances (run on a
# login node: compute nodes may have no internet access).
set -euo pipefail
cd "$(dirname "$0")"
url=https://axavier.org/UnitCommitment.jl/0.4/instances
while read -r name; do
    file="instances/$name.json.gz"
    [ -s "$file" ] && continue
    mkdir -p "$(dirname "$file")"
    curl -fsSL "$url/$name.json.gz" -o "$file"
done < instances.txt
