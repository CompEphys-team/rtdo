# NOTE: This file only serves as IDE setup.
# It is not intended for nor capable of building the project.

TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    ../vclamp/run.cc \
    ../vclamp/model/VClampGA.cu \
    converter.c \
    ../vclamp/generate.cc \
    rt.c \
    ../vclamp/model/HHVClamp_CODE/neuronFnct.cc \
    ../vclamp/model/HHVClamp_CODE/neuronKrnl.cc \
    ../vclamp/model/HHVClamp_CODE/runner.cc \
    ../vclamp/model/HHVClamp_CODE/runnerGPU.cc \
    ../vclamp/model/GA.cc \
    ../vclamp/model/HHVClamp.cc \
    ../vclamp/model/Wave.cc \
    ../vclamp/model/WaveGA.cu

HEADERS += \
    rt_helper.h \
    converter.h \
    rtdo_types.h \
    ../vclamp/model/helper.h \
    ../vclamp/model/HHVClamp_CODE/definitions.h \
    ../vclamp/model/HHVClampParameters.h \
    ../vclamp/model/VClampGA.h \
    ../vclamp/model/waveHelper.h

INCLUDEPATH += \
    /usr/realtime/include \
    /usr/src/comedi/inc-wrap \
    ../vclamp/model \
    /home/felix/genn/lib/include \
    /home/felix/genn/userproject/include \
    /usr/local/cuda/include
