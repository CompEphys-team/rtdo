##--------------------------------------------------------------------------
##   Author: Thomas Nowotny
##  
##   Institute: Center for Computational Neuroscience and Robotics
##              University of Sussex
##            Falmer, Brighton BN1 9QJ, UK 
##  
##   email to:  T.Nowotny@sussex.ac.uk
##  
##   initial version: 2010-02-07
##  
##--------------------------------------------------------------------------

EXECUTABLE	:= VClampGA
SOURCES		:= VClampGA.cu
NVCCFLAGS	+= --std c++11

RTAI_PATH	?= /usr/realtime
COMEDI_PATH	?= /usr/src/comedi

ifeq ($(RTDO),y)
# Build into a dynamically loadable library instead
    EXECUTABLE  := VClampGA.so
    CXXFLAGS	:= -fPIC
    NVCCFLAGS	+= --compiler-options "-fPIC"
    LINK_FLAGS	:= -shared -lc
    INCLUDE_FLAGS   += -I../../include \
			-I$(RTAI_PATH)/include \
			-I$(COMEDI_PATH)/inc-wrap
endif

include	$(GENN_PATH)/userproject/include/makefile_common_gnu.mk
