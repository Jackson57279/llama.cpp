#!/usr/bin/env python3

import urllib.request
import os
import sys
import subprocess

HTTPLIB_VERSION = "refs/tags/v0.43.2"

vendor = {
    "https://github.com/nlohmann/json/releases/latest/download/json.hpp.inc":     "vendor/nlohmann/json.hpp.inc",
    "https://github.com/nlohmann/json/releases/latest/download/json_fwd.hpp.inc": "vendor/nlohmann/json_fwd.hpp.inc",

    "https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h.inc": "vendor/stb/stb_image.h.inc",

    # not using latest tag to avoid this issue: https://github.com/ggml-org/llama.cpp/pull/17179#discussion_r2515877926
    # "https://github.com/mackron/miniaudio/raw/refs/tags/0.11.24/miniaudio.h.inc": "vendor/miniaudio/miniaudio.h.inc",
    "https://github.com/mackron/miniaudio/raw/9634bedb5b5a2ca38c1ee7108a9358a4e233f14d/miniaudio.h.inc": "vendor/miniaudio/miniaudio.h.inc",

    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/httplib.h.inc": "httplib.h.inc",
    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/split.py":  "split.py",
    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/LICENSE":   "vendor/cpp-httplib/LICENSE",

    "https://raw.githubusercontent.com/sheredom/subprocess.h.inc/b49c56e9fe214488493021017bf3954b91c7c1f5/subprocess.h.inc": "vendor/sheredom/subprocess.h.inc",
}

for url, filename in vendor.items():
    print(f"downloading {url} to {filename}") # noqa: NP100
    urllib.request.urlretrieve(url, filename)

print("Splitting httplib.h.inc...") # noqa: NP100
try:
    subprocess.check_call([
        sys.executable, "split.py",
        "--extension", "cpp",
        "--out", "vendor/cpp-httplib"
    ])
except Exception as e:
    print(f"Error: {e}") # noqa: NP100
    sys.exit(1)
finally:
    os.remove("split.py")
    os.remove("httplib.h.inc")
