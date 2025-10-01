#!/bin/bash
# Download all files from a Hugging Face repo except .bin
# Requires: jq, wget, curl

MODEL_ID="xhyi/PT_GPTNEO350_ATG"
TARGET_DIR="./GPT-NEO-350M"

mkdir -p $TARGET_DIR

# Query Hugging Face API for repo file list
curl -s "https://huggingface.co/api/models/${MODEL_ID}" \
  | jq -r '.siblings[].rfilename' \
  | grep -vE "\.binn$|\.safetensors$" \
  | while read file; do
      echo "Downloading $file ..."
      wget -c "https://huggingface.co/${MODEL_ID}/resolve/main/${file}" -P $TARGET_DIR
    done

echo "✅ All files (except .bin) downloaded into $TARGET_DIR"
