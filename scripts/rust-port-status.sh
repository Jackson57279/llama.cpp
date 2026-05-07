#!/usr/bin/env bash
set -euo pipefail

mode="${1:-report}"

if [[ "$mode" != "report" && "$mode" != "enforce" ]]; then
    echo "usage: $0 [report|enforce]" >&2
    exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mapfile -t files < <(
    find "$root" \
        -path "$root/.git" -prune -o \
        -path "$root/examples/simple-rust/target" -prune -o \
        -type f \( \
            -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o \
            -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' \
        \) -print | sort
)

printf 'C/C++ files remaining: %d\n' "${#files[@]}"

if [[ "${#files[@]}" -gt 0 ]]; then
    printf '%s\n' "${files[@]#$root/}" | sed -n '1,200p'
    if [[ "${#files[@]}" -gt 200 ]]; then
        printf '... %d more\n' "$((${#files[@]} - 200))"
    fi
fi

if [[ "$mode" == "enforce" && "${#files[@]}" -gt 0 ]]; then
    exit 1
fi
