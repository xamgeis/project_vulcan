FROM bamos/ubuntu-opencv-dlib-torch:ubuntu_14.04-opencv_2.4.11-dlib_19.0-torch_2016.07.12
MAINTAINER Brandon Amos <brandon.amos.cs@gmail.com>


# TODO: Should be added to opencv-dlib-torch image.
RUN ln -s /root/torch/install/bin/* /usr/local/bin

RUN apt-get update && \
        apt-get install -y \
        build-essential \
        curl \
        graphicsmagick \
        libssl-dev \
        libffi-dev \
        python-dev \
        python-pip \
        python-numpy \
        python-nose \
        python-scipy \
        python-pandas \
        python-protobuf \
        python-openssl \
        wget \
        unzip \
        cmake \
        git \
        pkg-config \
        libatlas-base-dev \
        gfortran \
        libjasper-dev \
        libgtk2.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libjasper-dev \
        libv4l-dev \
        liblapacke-dev \
        checkinstall \
        && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


ADD . /root/openface
RUN python -m pip install --upgrade --force pip
# end openface
RUN pip install numpy

WORKDIR /
RUN wget https://github.com/opencv/opencv/archive/3.2.0.zip -O opencv3.zip && \
    unzip -q opencv3.zip && mv /opencv-3.2.0 /opencv
RUN wget https://github.com/opencv/opencv_contrib/archive/3.2.0.zip -O opencv_contrib3.zip && \
    unzip -q opencv_contrib3.zip && mv /opencv_contrib-3.2.0 /opencv_contrib
RUN mkdir /opencv/build
WORKDIR /opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D BUILD_PYTHON_SUPPORT=ON \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D WITH_IPP=OFF \
    -D WITH_LAPACK=OFF \
    -D WITH_V4L=ON ..
RUN make -j4
RUN make install
RUN ldconfig

WORKDIR /
# openface#
RUN cd ~/openface && \
    ./models/get-models.sh && \
    pip2 install -r requirements.txt && \
    python2 setup.py install && \
    pip2 install --user --ignore-installed -r demos/web/requirements.txt && \
    pip2 install -r training/requirements.txt
# end openface#
RUN pip install flake8 pep8 --upgrade

# openface
EXPOSE 8000 9000
CMD /bin/bash -l -c '/root/openface/demos/web/start-servers.sh'



