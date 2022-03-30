
CC=g++
CFLAGS=-O2 -std=c++0x -I. -I/usr/local/include/opencv4
LIBDIRS= -L/usr/local/cuda/lib64 -L/usr/local/lib

CVLIBS = -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_video -lopencv_videoio -lnvcaffe_parser -lnvinfer -lopencv_features2d -lopencv_imgcodecs -lopencv_objdetect
LDFLAGS=$(LIBDIRS) -lm -lstdc++ $(CVLIBS) -lcuda -lcublas -lcurand -lcudart

OUTNAME_RELEASE = deserialize_model_to_get_engine
OUTNAME_DEBUG   = deserialize_model_to_get_engine_debug
EXTRA_DIRECTORIES = ../common
SAMPLE_DIR_NAME = $(shell basename $(dir $(abspath $(firstword $(MAKEFILE_LIST)))))
MAKEFILE ?= ../Makefile.config

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

all: deserialize_model_to_get_engine

deserialize_model_to_get_engine: deserialize_model_to_get_engine.o deserialize_model_to_get_engine.o
	gcc -o $@ $^ $(CFLAGS) $(LDFLAGS)
# include $(MAKEFILE)
