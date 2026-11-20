OS       = $(shell uname)
ROCM     = /opt/rocm
HIP      = $(if $(filter Linux,$(OS)),$(if $(wildcard $(ROCM)/bin/hipcc),hip))
BACKENDS = blas cpu $(if $(filter Darwin,$(OS)),metal) $(HIP)
ASR      = whistle moonshine
LLM      = falcon qwen
TTS      = supertonic

FLAGS_Darwin = -fobjc-arc
FLAGS_Linux  = -D_GNU_SOURCE -D__sincosf=sincosf -I/usr/include/openblas
LIBS_Darwin  = -framework Accelerate -framework Metal -framework Foundation
LIBS_hip     = -L$(ROCM)/lib -lamdhip64 -Wl,-rpath,$(ROCM)/lib -lstdc++
LIBS_Linux   = -lopenblas -lgomp -lm $(LIBS_$(HIP))
CC      = clang -std=c2x -O3 -flto -march=native -Wall -Wextra -Wpedantic \
          $(FLAGS_$(OS))
LIBS    = $(LIBS_$(OS))
SHARED  = utils.c roles.c $(wildcard $(addsuffix .c,$(BACKENDS)) \
          $(addsuffix .m,$(BACKENDS))) $(if $(HIP),tmp/hip.o)
HEADERS = talk.h asr.h llm.h tts.h backend.h roles.h utils.h transformer.h
CHAT    = transformer.c sampler.c
ENGINES = $(addsuffix .c,$(ASR) $(LLM) $(TTS)) $(CHAT)
BACKS   = -D'BACKEND_LIST=$(foreach e,$(BACKENDS),X($(e)))'
LISTS   = $(BACKS) -D'ASR_LIST=$(foreach e,$(ASR),X($(e)))' \
          -D'LLM_LIST=$(foreach e,$(LLM),X($(e)))' \
          -D'TTS_LIST=$(foreach e,$(TTS),X($(e)))'

all: talk $(ASR) $(LLM) $(TTS) bench test

talk: $(HEADERS) talk.c $(ENGINES) $(SHARED)
	$(CC) -DNO_MAIN $(LISTS) talk.c $(ENGINES) $(SHARED) -o talk $(LIBS)

$(ASR): %: %.c $(HEADERS) $(SHARED)
	$(CC) $(BACKS) -D'ASR_LIST=X($@)' $< $(SHARED) -o $@ $(LIBS)

$(LLM): %: %.c $(HEADERS) $(CHAT) $(SHARED)
	$(CC) $(BACKS) -D'LLM_LIST=X($@)' $< $(CHAT) $(SHARED) -o $@ $(LIBS)

$(TTS): %: %.c $(HEADERS) $(SHARED)
	$(CC) $(BACKS) -D'TTS_LIST=X($@)' $< $(SHARED) -o $@ $(LIBS)

bench: bench.c $(HEADERS) $(addsuffix .c,$(LLM)) $(CHAT) $(SHARED)
	$(CC) -DNO_MAIN $(BACKS) -D'LLM_LIST=$(foreach e,$(LLM),X($(e)))' \
	    bench.c $(addsuffix .c,$(LLM)) $(CHAT) $(SHARED) -o bench $(LIBS)

test: test.c $(HEADERS) transformer.c $(SHARED)
	$(CC) -DNO_MAIN $(BACKS) test.c transformer.c $(SHARED) -o test $(LIBS)

tmp/hip.o: hip.cpp backend.h roles.h utils.h
	mkdir -p tmp
	$(ROCM)/bin/hipcc --rocm-path=$(ROCM) -I$(ROCM)/include -std=c++20 -O3 \
	    -Wall -Wextra -c hip.cpp -o tmp/hip.o
