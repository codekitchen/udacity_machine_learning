#!/bin/bash

# enable opengl forwarding for xquartz
defaults write org.macosforge.xquartz.X11 enable_iglx -bool true
open -a XQuartz
xhost + $(dinghy ip)
echo "open http://keras.docker/"
docker run -it --rm --name keras -e VIRTUAL_HOST=keras.docker:8888 -e DISPLAY=$(dinghy ip --host):0 -v=$(pwd):/srv keras
