#!/bin/sh

node generate-static-files.js
echo $?
node generate-sitemap.js
echo $?
node change-markdown.js

git add --all