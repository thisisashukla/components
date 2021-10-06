@echo off

set msg=%1
set branch=%2

git add .
git commit -a -m %~1
git push origin "%~2