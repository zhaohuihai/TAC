CC =	icpc     	

MKLROOT = /opt/intel/composer_xe_2013.5.198/mkl

MKLPATH = $(MKLROOT)/lib
MKLINCLUDE = $(MKLROOT)/include

CFLAGS = -openmp -I$(MKLROOT)/include


SRC =	main.cpp test.cpp
		

OBJS =	main.o	 test.o
		   
LIBS = -L$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm

TARGET   = TensorOperation

all: $(TARGET)
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LIBS)

.cpp.o:
	$(CC) -c $(CFLAGS) $<




clean:
	rm -f *.o core

