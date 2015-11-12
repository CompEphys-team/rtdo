TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

# Sources compiled with nvcc
CUDA_SOURCES += kernel.cu

# Sources compiled with gcc
SOURCES += main.c

HEADERS += \
    gpgpu.h

INCLUDEPATH += \
    /usr/realtime/include \
    /usr/src/comedi/inc-wrap

LIBS += -L/usr/realtime/lib -lkcomedilxrt

LIBS += -pthread
QMAKE_CFLAGS += -pthread


# Add cuda sources to project overview
OTHER_FILES = $$CUDA_SOURCES


# nvcc setup
CUDA_DIR = /usr/local/cuda
INCLUDEPATH += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64
LIBS += -lcudart
CUDA_INC = $$join(INCLUDEPATH,'" -I "','-I "','"')
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj
cuda.commands = nvcc -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$CUDA_INC ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.dependency_type = TYPE_C
cuda.depend_command = nvcc -g -G -M $$CUDA_INC  ${QMAKE_FILE_NAME}
QMAKE_EXTRA_COMPILERS = cuda
