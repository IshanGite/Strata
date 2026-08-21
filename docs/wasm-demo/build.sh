#!/bin/bash
set -e

echo "Building strata-wasm for web target..."
cd ../../strata-wasm
wasm-pack build --target web --out-dir ../docs/wasm-demo/pkg

echo "Build complete! You can run a local server in docs/wasm-demo like:"
echo "python3 -m http.server 8000"
