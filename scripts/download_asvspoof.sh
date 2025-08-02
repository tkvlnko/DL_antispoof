#!/usr/bin/env bash
set -e
DEST=data/ASVspoof2019
mkdir -p "$DEST"
cd "$DEST"

echo "======= downloading"
kaggle datasets download awsaf49/asvpoof-2019-dataset -p . --force

echo "======= unpacking…"
unzip -q asvspoof-2019-dataset.zip
rm asvspoof-2019-dataset.zip

mkdir -p LA
mv ASVspoof2019_LA*/* LA/
rm -r ASVspoof2019_LA*

echo "======= done: $(du -sh LA)"
