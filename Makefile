OUTNAME_RELEASE = deserialize_model_to_get_engine
OUTNAME_DEBUG   = deserialize_model_to_get_engine_debug
EXTRA_DIRECTORIES = ../common
SAMPLE_DIR_NAME = $(shell basename $(dir $(abspath $(firstword $(MAKEFILE_LIST)))))
MAKEFILE ?= ../Makefile.config
LIBDIRS= -L/usr/local/cuda/lib64 -L/home/ubuntu/opencv-3.1.0/lib



include $(MAKEFILE)
