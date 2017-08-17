#-------------------------------------------------
#
# Project created by QtCreator 2016-09-26T16:42:26
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = rtdo
TEMPLATE = app

CONFIG += c++11

INCLUDEPATH += \
    src/include \
    lib/randutils \
    lib/qcustomplot

CONFIG(release, debug|release): DEFINES += NDEBUG
                          else: DEFINES +=  DEBUG

CONFIG(usedouble): DEFINES += USEDOUBLE

SOURCES += \
    src/core/main.cpp\
    src/gui/mainwindow.cpp \
    src/drivers/daq.cpp \
    src/core/types.cpp \
    src/drivers/daqcache.cpp \
    src/drivers/comediconverter.cpp \
    src/core/metamodel.cpp \
    src/core/wavegen.cpp \
    src/core/wavegen_mutate.cpp \
    src/core/wavegen_search.cpp \
    src/core/wavegenlibrary.cpp \
    src/core/experimentlibrary.cpp \
    src/core/errorprofiler.cpp \
    src/core/supportcode.cpp \
    lib/qcustomplot/qcustomplot.cpp \
    src/core/util.cpp \
    src/gui/wavegendialog.cpp \
    src/gui/profiledialog.cpp \
    src/core/project.cpp \
    src/core/session.cpp \
    src/core/sessionlog.cpp \
    src/core/streamops.cpp \
    src/core/wavegenselector.cpp \
    src/core/sessionworker.cpp \
    src/gui/wavegenfitnessmapper.cpp \
    src/gui/profileplotter.cpp \
    src/core/wavesource.cpp \
    src/core/profilestats.cpp \
    src/core/wavesubsets.cpp \
    src/core/gafitter.cpp \
    src/core/wavesetcreator.cpp \
    src/gui/deckwidget.cpp \
    src/gui/gafitterwidget.cpp \
    src/gui/parameterfitplotter.cpp \
    src/gui/stimulationplotter.cpp \
    src/gui/stimulationgraph.cpp \
    src/gui/stimulationcreator.cpp \
    src/gui/gafittersettingsdialog.cpp \
    src/core/errorprofile.cpp \
    src/core/profilerlibrary.cpp \
    src/core/samplingprofiler.cpp \
    src/gui/samplingprofiledialog.cpp \
    src/gui/samplingprofileplotter.cpp \
    src/gui/rundatadialog.cpp \
    src/gui/projectsettingsdialog.cpp \
    src/gui/wavegendatadialog.cpp \
    src/gui/stimulationdatadialog.cpp \
    src/gui/wavegenprogressplotter.cpp \
    src/gui/responseplotter.cpp \
    src/gui/daqdialog.cpp \
    src/drivers/daqfilter.cpp \
    src/core/calibrator.cpp

HEADERS  += \
    src/include/mainwindow.h \
    src/include/daq.h \
    src/include/types.h \
    src/include/thread.h \
    src/include/conditionvariable.h \
    src/include/queue.h \
    src/include/daqcache.h \
    src/include/comediconverter.h \
    src/include/comedidaq.h \
    src/include/metamodel.h \
    src/include/wavegen.h \
    src/include/cuda_helper.h \
    src/include/multiarray.h \
    src/include/wavegenlibrary.h \
    src/include/experimentlibrary.h \
    src/include/errorprofiler.h \
    src/include/supportcode.h \
    src/include/util.h \
    lib/qcustomplot/qcustomplot.h \
    src/include/AP.h \
    src/include/wavegendialog.h \
    src/include/profiledialog.h \
    src/include/project.h \
    src/include/session.h \
    src/include/sessionlog.h \
    src/include/streamops.h \
    src/include/wavegenselector.h \
    src/include/sessionworker.h \
    src/include/wavegenfitnessmapper.h \
    src/include/profileplotter.h \
    src/include/wavesource.h \
    src/include/profilestats.h \
    src/include/colorbutton.h \
    src/include/wavesubsets.h \
    src/include/gafitter.h \
    src/include/wavesetcreator.h \
    src/include/deckwidget.h \
    src/include/gafitterwidget.h \
    src/include/parameterfitplotter.h \
    src/include/stimulationplotter.h \
    src/include/stimulationgraph.h \
    src/include/stimulationcreator.h \
    src/include/gafittersettingsdialog.h \
    src/include/errorprofile.h \
    src/include/profilerlibrary.h \
    src/include/samplingprofiler.h \
    src/include/samplingprofiledialog.h \
    src/include/samplingprofileplotter.h \
    src/include/rundatadialog.h \
    src/include/projectsettingsdialog.h \
    src/include/wavegendatadialog.h \
    src/include/stimulationdatadialog.h \
    src/include/wavegenprogressplotter.h \
    src/include/responseplotter.h \
    src/include/daqdialog.h \
    src/include/daqfilter.h \
    src/include/calibrator.h

FORMS    += \
    src/gui/mainwindow.ui \
    src/gui/wavegendialog.ui \
    src/gui/profiledialog.ui \
    src/gui/wavegenfitnessmapper.ui \
    src/gui/profileplotter.ui \
    src/gui/deckwidget.ui \
    src/gui/gafitterwidget.ui \
    src/gui/parameterfitplotter.ui \
    src/gui/stimulationplotter.ui \
    src/gui/stimulationcreator.ui \
    src/gui/gafittersettingsdialog.ui \
    src/gui/samplingprofiledialog.ui \
    src/gui/samplingprofileplotter.ui \
    src/gui/rundatadialog.ui \
    src/gui/projectsettingsdialog.ui \
    src/gui/wavegendatadialog.ui \
    src/gui/stimulationdatadialog.ui \
    src/gui/wavegenprogressplotter.ui \
    src/gui/responseplotter.ui \
    src/gui/daqdialog.ui

LIBS     += -rdynamic -ldl -lcomedi -lgsl

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


# RTAI/Comedi
realtime {
RTAI_LIBDIR = /usr/realtime/lib # Location of libkcomedilxrt.a
RC_HEADER = RC_rtai_comedi.h # Header file replacing rtai_comedi.h with a prefixed version

RC_syms.target = RC.syms
RC_syms.depends = $${RTAI_LIBDIR}/libkcomedilxrt.a
RC_syms.commands = nm $$RC_syms.depends | grep \'\\bcomedi\' | awk \'\$\$3 {print \$\$3 \" RC_\" \$\$3}\' > $$RC_syms.target

RC_header.target = $${OUT_PWD}/$$RC_HEADER
RC_header.depends = RC_syms
RC_header.commands = awk \'{print \"$${LITERAL_HASH}define \" \$\$0}\' $$RC_syms.target > $$RC_header.target; \
    echo \'$${LITERAL_HASH}ifdef __cplusplus\'                                         >> $$RC_header.target; \
    echo \'extern \"C\" {\'                                                            >> $$RC_header.target; \
    echo \'$${LITERAL_HASH}endif\'                                                     >> $$RC_header.target; \
    echo \'$${LITERAL_HASH}include <rtai_comedi.h>\'                                   >> $$RC_header.target; \
    echo \'$${LITERAL_HASH}ifdef __cplusplus\'                                         >> $$RC_header.target; \
    echo \'}\'                                                                         >> $$RC_header.target; \
    echo \'$${LITERAL_HASH}endif\'                                                     >> $$RC_header.target; \
    awk \'{print \"$${LITERAL_HASH}undef \" \$\$1}\' $$RC_syms.target                  >> $$RC_header.target

RC_kcomedilxrt.target = libRC_kcomedilxrt.a
RC_kcomedilxrt.depends = RC_syms
RC_kcomedilxrt.commands = objcopy --redefine-syms=$$RC_syms.target $$RC_syms.depends $$RC_kcomedilxrt.target

RC_clean.commands = rm -rf $$RC_header.target $$RC_syms.target $$RC_kcomedilxrt.target

QMAKE_EXTRA_TARGETS += RC_kcomedilxrt RC_syms RC_header RC_clean
PRE_TARGETDEPS += $$RC_header.target $$RC_kcomedilxrt.target
CLEAN_DEPS += RC_clean

DEFINES += BUILD_RT

QMAKE_CXXFLAGS += \
    -isystem /usr/realtime/include

SOURCES += \
    src/realtime/thread.cpp \
    src/realtime/conditionvariable.cpp \
    src/realtime/queue.cpp \
    src/drivers/rtcomedidaq.cpp

LIBS += -L. -lRC_kcomedilxrt -pthread

} else { # !realtime
SOURCES += \
    src/nonrealtime/thread.cpp \
    src/nonrealtime/conditionvariable.cpp \
    src/nonrealtime/queue.cpp \
    src/drivers/comedidaq.cpp
}

DISTFILES += \
    src/include/experimentlibrary.cu \
    src/include/wavegenlibrary.cu \
    src/include/profilerlibrary.cu
