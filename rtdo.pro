#-------------------------------------------------
# Author: Felix Kern
#
# Institute: School of Life Sciences
# University of Sussex
# Falmer, Brighton BN1 9QG, UK
#
# email to:  fbk21@sussex.ac.uk
#
# initial version: 2015-12-03
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = rtdo
TEMPLATE = app


SOURCES += core/main.cpp \
    gui/mainwindow.cpp \
    core/softrtdaq.c \
    core/rt.c \
    core/aothread.c \
    core/aithread.c \
    core/run.cc

OTHER_FILES += vclamp/model/GNUmakefile \
    vclamp/model/*.cc \
    vclamp/model/*.cu \
    vclamp/model/*.h

HEADERS  += \
    include/mainwindow.h \
    include/softrtdaq.h \
    include/types.h \
    include/globals.h \
    include/rt.h \
    include/run.h

FORMS    += \
    gui/mainwindow.ui

LIBS     += -L/usr/realtime/lib -lkcomedilxrt \
    -rdynamic -ldl \
    -pthread

INCLUDEPATH += \
    include \
    /usr/realtime/include \
    /usr/src/comedi/inc-wrap

QMAKE_CFLAGS += -Wno-unused-parameter
QMAKE_CXXFLAGS += -Wno-unused-parameter
