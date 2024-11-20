#!/bin/sh

node generate-static-files.js
if [ $? -ne 0 ]; then
  echo "Static file generation failed"
  exit 1
fi
node generate-sitemap.js
if [ $? -ne 0 ]; then
  echo "Site map generation failed"
  exit 1
fi
node change-markdown.js
if [ $? -ne 0 ]; then
  echo "Markdown extrapolation failed"
  exit 1
fi
git add --all