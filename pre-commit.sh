#!/bin/sh
node generate-static-files.js
if [ $? -eq 0 ]; then
  echo "Node.js script executed successfully"
else
  echo "Static file generation failed"
fi

node generate-sitemap.js
if [ $? -eq 0 ]; then
  echo "Node.js script executed successfully"
else
  echo "Sitemap generation failed"
fi

node change-markdown.js
if [ $? -eq 0 ]; then
  echo "Node.js script executed successfully"
else
  echo "Markdown variable extrapolation failed"
fi

git add --all