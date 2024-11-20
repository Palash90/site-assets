#!/bin/sh
set -e
node generate-static-files.js
node generate-sitemap.js
node change-markdown.js

git add --all