#!/bin/sh

node generate-static-files.js
echo $?
node generate-sitemap.js
node change-markdown.js

git add --all