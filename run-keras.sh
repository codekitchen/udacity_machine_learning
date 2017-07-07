#!/bin/bash

echo "open http://keras.docker/"
docker run -it --rm -e VIRTUAL_HOST=keras.docker:8888 -v=$(pwd):/srv keras
