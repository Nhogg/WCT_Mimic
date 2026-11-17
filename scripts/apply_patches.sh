#!/usr/bin/env bash
# Apply our patches to third-party submodules (idempotent).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
apply() {  # <submodule dir> <patch>
  if git -C "$ROOT/$1" apply --reverse --check "$ROOT/$2" 2>/dev/null; then
    echo "already applied: $2"
  else
    git -C "$ROOT/$1" apply "$ROOT/$2" && echo "applied: $2"
  fi
}
apply third_party/mjlab patches/mjlab-local-motion-output.patch
