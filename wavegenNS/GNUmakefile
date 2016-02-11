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

EXECUTABLE	:= WaveGen.so
SOURCES		:= WaveGA.cu

RTAI_PATH	?= /usr/realtime
COMEDI_PATH	?= /usr/src/comedi

CXXFLAGS	:= -fPIC
NVCCFLAGS	+= --compiler-options "-fPIC" --std c++11
LINK_FLAGS	:= -shared -lc
INCLUDE_FLAGS   += -I../include \
		   -I$(RTAI_PATH)/include \
		   -I$(COMEDI_PATH)/inc-wrap

include	$(GENN_PATH)/userproject/include/makefile_common_gnu.mk
