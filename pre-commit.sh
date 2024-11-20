#!/bin/sh
if node generate-static-files.js; then
  echo "Static file generated successfully"
else
  echo "Static file generation failed"
  exit 1
fi


if node generate-sitemap.js; then
  echo "Sitemap generated successfully"
else
  echo "Sitemap generation failed"
  exit 1
fi


if node change-markdown.js; then
  echo "Markdown variable extrapolated successfully"
else
  echo "Markdown variable extrapolation failed"
  exit 1
fi

git add --all