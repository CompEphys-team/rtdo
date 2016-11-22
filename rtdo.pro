#-------------------------------------------------
#
# Project created by QtCreator 2016-09-26T16:42:26
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = rtdo
TEMPLATE = app

CONFIG += c++11

INCLUDEPATH += \
    src/include

CONFIG(release, debug|release): DEFINES += NDEBUG
                          else: DEFINES +=  DEBUG

CONFIG(usedouble): DEFINES += USEDOUBLE

SOURCES += \
    src/core/main.cpp\
    src/gui/mainwindow.cpp \
    src/core/types.cpp \
    src/core/metamodel.cpp \
    src/core/kernelhelper.cpp

HEADERS  += \
    src/include/mainwindow.h \
    src/include/types.h \
    src/include/metamodel.h \
    src/include/kernelhelper.h

FORMS    += \
    src/gui/mainwindow.ui

LIBS     += -rdynamic -ldl

DEFINES += CORE_INCLUDE_PATH='\\"$${PWD}/src/include\\"'


# GeNN static library
libGENN.path = $${PWD}/lib/genn
libGENN.dir = $$libGENN.path/lib/lib
libGENN.target = $$libGENN.dir/libgenn.a
libGENN.commands = $(MAKE) -f $$libGENN.path/lib/GNUmakefile GENN_PATH=$$libGENN.path $$libGENN.target
cleanGENN.commands = $(MAKE) -f $$libGENN.path/lib/GNUmakefile GENN_PATH=$$libGENN.path clean_libgenn
QMAKE_EXTRA_TARGETS += libGENN cleanGENN
PRE_TARGETDEPS += $$libGENN.target
CLEAN_DEPS += cleanGENN
LIBS += -L$$libGENN.dir -lgenn
DEFINES += LOCAL_GENN_PATH='\\"$$libGENN.path\\"'


# Including GeNN generate* files in the main project
# This should largely mirror the generateAll target in genn/lib/GNUmakefile
INCLUDEPATH += \
    $$libGENN.path \
    $$libGENN.path/lib/include \
    $$(CUDA_PATH)/include
SOURCES += \
    $$libGENN.path/lib/src/generateCPU.cc \
    $$libGENN.path/lib/src/generateKernels.cc \
    $$libGENN.path/lib/src/generateRunner.cc \
    src/core/generateAllNoMain.cpp
DEFINES += NVCC=\\\"\"$$(CUDA_PATH)/bin/nvcc\"\\\"
contains(QMAKE_HOST.arch, x86_64): LIBS += -L$$(CUDA_PATH)/lib64
                             else: LIBS += -L$$(CUDA_PATH)/lib
LIBS += -lcuda -lcudart


# TinyXML2 lib
libTinyXML.path = $${PWD}/lib/tinyxml2
libTinyXML.target = $$libTinyXML.path/libtinyxml2.a
libTinyXML.commands = $(MAKE) -C $$libTinyXML.path staticlib
cleanTinyXML.commands = $(MAKE) -C $$libTinyXML.path clean
QMAKE_EXTRA_TARGETS += libTinyXML cleanTinyXML
PRE_TARGETDEPS += $$libTinyXML.target
CLEAN_DEPS += cleanTinyXML
LIBS += -L$$libTinyXML.path -ltinyxml2
INCLUDEPATH += $$libTinyXML.path
