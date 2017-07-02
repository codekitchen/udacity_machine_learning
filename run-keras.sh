#!/bin/bash

echo "open http://keras.docker/"
docker run -it --rm -e VIRTUAL_HOST=keras.docker -e VIRTUAL_PORT=8888 -v=$(pwd)/notebooks:/srv gw000/keras-full
