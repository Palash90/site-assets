#!/bin/sh
node generate-static-files.js
node generate-sitemap.js
git add --all