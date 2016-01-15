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

CONFIG(debug, release|debug):DEFINES += _DEBUG

SOURCES += core/main.cpp \
    gui/mainwindow.cpp \
    core/softrtdaq.c \
    core/rt.c \
    core/aothread.c \
    core/aithread.c \
    core/run.cc \
    gui/channelsetupdialog.cpp \
    gui/channeleditormodel.cpp \
    gui/devicechannelmodel.cpp \
    gui/devicerangemodel.cpp \
    gui/devicereferencemodel.cpp \
    gui/channellistmodel.cpp \
    gui/vclampsetupdialog.cpp \
    core/util.cpp \
    lib/src/tinystr.cpp \
    lib/src/tinyxml.cpp \
    lib/src/tinyxmlerror.cpp \
    lib/src/tinyxmlparser.cpp \
    core/config.cpp \
    core/xmlmodel.cpp \
    gui/wavegensetupdialog.cpp \
    gui/modelsetupdialog.cpp

OTHER_FILES += simulation/GNUmakefile \
    simulation/VClampGA.cu \
    simulation/VClampGA.h \
    simulation/helper.h \
    simulation/GA.cc \
    wavegen/GNUmakefile \
    wavegen/WaveGA.cu \
    wavegen/WaveGA.h \
    wavegen/waveHelper.h

HEADERS  += \
    include/mainwindow.h \
    include/softrtdaq.h \
    include/types.h \
    include/globals.h \
    include/rt.h \
    include/run.h \
    include/channelsetupdialog.h \
    include/channeleditormodel.h \
    include/devicechannelmodel.h \
    include/devicerangemodel.h \
    include/devicereferencemodel.h \
    include/channellistmodel.h \
    include/vclampsetupdialog.h \
    include/util.h \
    lib/include/tinystr.h \
    lib/include/tinyxml.h \
    include/config.h \
    include/xmlmodel.h \
    include/wavegensetupdialog.h \
    include/modelsetupdialog.h

FORMS    += \
    gui/mainwindow.ui \
    gui/channelsetupdialog.ui \
    gui/vclampsetupdialog.ui \
    gui/wavegensetupdialog.ui \
    gui/modelsetupdialog.ui

LIBS     += -rdynamic -ldl -pthread -L. -lRC_kcomedilxrt -lcomedi

INCLUDEPATH += \
    include \
    lib/include \
    /usr/realtime/include \
    /usr/src/comedi/inc-wrap

QMAKE_CFLAGS += -Wno-unused-parameter
QMAKE_CXXFLAGS += -Wno-unused-parameter -std=c++11

RTAI_LIBDIR = /usr/realtime/lib # Location of libkcomedilxrt.a
RC_HEADER = RC_rtai_comedi.h # Header file replacing rtai_comedi.h with a prefixed version

RC_syms.target = RC.syms
RC_syms.depends = $${RTAI_LIBDIR}/libkcomedilxrt.a
RC_syms.commands = nm $$RC_syms.depends | grep \'\\bcomedi\' | awk \'\$\$3 {print \$\$3 \" RC_\" \$\$3}\' > $$RC_syms.target

RC_header.target = $${OUT_PWD}/$$RC_HEADER
RC_header.commands = awk \'{print \"$${LITERAL_HASH}define \" \$\$0}\' $$RC_syms.target > $$RC_header.target; \
    echo \'$${LITERAL_HASH}include <rtai_comedi.h>\' >> $$RC_header.target; \
    awk \'{print \"$${LITERAL_HASH}undef \" \$\$1}\' $$RC_syms.target >> $$RC_header.target
RC_header.depends = RC_syms

RC_kcomedilxrt.target = libRC_kcomedilxrt.a
RC_kcomedilxrt.commands = objcopy --redefine-syms=$$RC_syms.target $${RTAI_LIBDIR}/libkcomedilxrt.a $$RC_kcomedilxrt.target
RC_kcomedilxrt.depends = RC_syms

RC_clean.commands = rm -rf $$RC_header.target $$RC_syms.target

QMAKE_EXTRA_TARGETS += RC_kcomedilxrt RC_syms RC_header RC_clean
PRE_TARGETDEPS += $$RC_header.target $$RC_kcomedilxrt.target
CLEAN_DEPS += RC_clean
