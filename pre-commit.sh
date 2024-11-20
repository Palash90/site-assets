#!/bin/sh
if node generate-static-files.js; then
  echo "Static file generated successfully"
else
  echo "Static file generation failed"
  exit 1
fi

node generate-sitemap.js
if [ $? -eq 0 ]; then
  echo "Node.js script executed successfully"
else
  echo "Sitemap generation failed"
  exit 1
fi

node change-markdown.js
if [ $? -eq 0 ]; then
  echo "Node.js script executed successfully"
else
  echo "Markdown variable extrapolation failed"
  exit 1
fi

git add --all