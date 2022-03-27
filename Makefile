OUTNAME_RELEASE = deserialize_model_to_get_engine
OUTNAME_DEBUG   = deserialize_model_to_get_engine_debug
EXTRA_DIRECTORIES = ../common
SAMPLE_DIR_NAME = $(shell basename $(dir $(abspath $(firstword $(MAKEFILE_LIST)))))
MAKEFILE ?= ../Makefile.config
include $(MAKEFILE)
