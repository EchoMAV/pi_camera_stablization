#! /bin/bash
arch=$(uname -m)
if [[ $arch == x86_64* ]]; then
    echo "installing on x86_64 architecture"
elif [[ $arch == aarch64* ]]; then
    echo "installing on aarch64 architecture"
else
    echo "unrecognized architecture"
    exit 1
fi

apt update
apt install -y \
    cmake \
    git \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    libboost-dev \
    libboost-program-options-dev \
    libdrm-dev \
    libegl1-mesa-dev \
    libexif-dev \
    libglib2.0-dev \
    libgnutls28-dev \
    libgstreamer-plugins-bad1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libjpeg-dev \
    libopencv-dev \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libtiff5-dev \
    libx264-dev \
    meson \
    ninja-build \
    openssl \
    pybind11-dev \
    python3-jinja2 \
    python3-opencv \
    python3-pip \
    python3-ply \
    python3-yaml \
    qtbase5-dev \
    

git clone https://github.com/raspberrypi/libcamera.git /tmp/libcamera
git clone https://github.com/anholt/libepoxy.git /tmp/libepoxy
git clone https://github.com/raspberrypi/rpicam-apps.git /tmp/rpicam-apps

NUM_BUILD_PROCESSORS=1

originalDir=$(pwd)


cd /tmp/libcamera
meson setup build --buildtype=release -Dpipelines=rpi/vc4,rpi/pisp -Dipas=rpi/vc4,rpi/pisp -Dv4l2=true -Dgstreamer=enabled -Dtest=false -Dlc-compliance=disabled -Dcam=disabled -Dqcam=disabled -Ddocumentation=disabled -Dpycamera=enabled
#use -j 1 to limit compilation to one thread to avoid running out of memory
ninja -C build -j $NUM_BUILD_PROCESSORS
ninja -C build install

cd /tmp/libepoxy
mkdir _build
cd _build
meson
ninja -j $NUM_BUILD_PROCESSORS
ninja install

cd /tmp/rpicam-apps
meson setup build -Denable_libav=disabled -Denable_drm=enabled -Denable_egl=disabled -Denable_qt=disabled -Denable_opencv=enabled -Denable_tflite=disabled
meson compile -C build -j $NUM_BUILD_PROCESSORS
meson install -C build

ldconfig

cd $originalDir
# clean up the build directory if it exists
rm -rf build
mkdir build
cd build
# build the application and install it
cmake ..
make
make install

