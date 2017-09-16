FROM gw000/keras-full

RUN apt-get update && \
  apt-get install -yy python-opengl libgl1-mesa-dev mesa-utils cmake zlib1g-dev swig pandoc pandoc-citeproc texlive texlive-xetex

RUN pip install --upgrade 'scikit-learn<0.18' \
 && pip install 'seaborn' 'pygame' 'opencv-contrib-python' 'tqdm' 'gym[all]'

RUN update-glx --set glx /usr/lib/mesa-diverted