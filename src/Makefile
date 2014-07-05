NVCC = nvcc
NVCCFLAGS = -Xcompiler="-Wall" -Xptxas="-v" --use_fast_math -arch=sm_20 -target-cpu-arch=x86 -O2
#--maxrregcount=20
# -g -G # for debugging
CC = g++
# _POSIX_C_SOURCE for clock_gettime(2)
CFLAGS = -Wall -pedantic -march=native -O2 -D_POSIX_C_SOURCE=199309
# -std=c99
# -g #for debugging
CPP = g++
CPPFLAGS = -Wall -pedantic -march=native -O2
# -g # for debugging
OBJECTS1 = libppm.o enc_cpu.o enc.o
OBJECTS2 = libppm.o enc_cpu.o ../tests/test_overhead.o
INCFLAGS = -I . 
LIBS = -lm -lrt 
LDFLAGS = 
PROGNAME1 = enc
PROGNAME2 = ../tests/test_overhead

all: $(PROGNAME1) $(PROGNAME2)

$(PROGNAME1): $(OBJECTS1)
	$(NVCC) -o $(PROGNAME1) $(OBJECTS1) $(LDFLAGS) $(LIBS)

$(PROGNAME2): $(OBJECTS2)
	$(NVCC) -o $(PROGNAME2) $(OBJECTS2) $(LDFLAGS) $(LIBS)

.SUFFIXES: # Delete the default suffixes
.SUFFIXES: .cu .c .h .cpp .hpp .o

.cu.o:
	$(NVCC) -o $@ -c $(NVCCFLAGS) $< $(INCFLAGS)

.c.o:
	$(CC) -o $@ -c $(CFLAGS) $< $(INCFLAGS)

.cpp.o:
	$(CPP) -o $@ -c $(CPPFLAGS) $< $(INCFLAGS)

count:
	wc *.cu *.c *.cpp *.h *.hpp

clean:
	rm -f *.o $(PROGNAME1) $(PROGNAME2)

rebuild: clean all

.PHONY: all
.PHONY: count
.PHONY: clean
.PHONY: rebuild

