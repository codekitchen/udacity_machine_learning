FROM gw000/keras-full

RUN pip install --upgrade 'scikit-learn<0.18' \
 && pip install 'seaborn' 'pygame' 'opencv-contrib-python' 'tqdm'