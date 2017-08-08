#!/bin/bash

open -a XQuartz
xhost + $(dinghy ip)
echo "open http://keras.docker/"
docker run -it --rm -e VIRTUAL_HOST=keras.docker:8888 -e DISPLAY=$(dinghy ip --host):0 -v=$(pwd):/srv keras
