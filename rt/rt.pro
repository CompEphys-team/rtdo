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
    options.c

HEADERS += \
    rt_helper.h \
    converter.h \
    options.h

INCLUDEPATH += \
    /usr/realtime/include \
    /usr/src/comedi/inc-wrap \
    ../vclamp/model
