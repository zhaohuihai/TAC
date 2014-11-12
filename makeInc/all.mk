


SRC =	main.cpp \
        test.cpp \
        parameter.cpp \
        wave_function.cpp \
        operators.cpp \
        imaginary_time_evolution.cpp \
        renormalization_group.cpp \
        environment.cpp \
        Ising_square.cpp \
        transverse_Ising_square.cpp \
        Heisenberg_square.cpp
		

OBJS =	main.o	 \
		test.o   \
		parameter.o   \
		wave_function.o \
        operators.o   \
        imaginary_time_evolution.o \
        renormalization_group.o \
        environment.o \
        Ising_square.o \
        transverse_Ising_square.o \
        Heisenberg_square.o

TARGET   = PEPS

all: $(TARGET)
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LIBS)


