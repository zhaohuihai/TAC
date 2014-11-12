# specify boost header file directory
#BOOST_DIR := /Users/zhaohuihai/libraries/boost_1_55_0/

# setup for blas/lapack
BLAS_LAPACK_INC := -I/opt/intel/mkl/include/
#BLAS_LAPACK_LIB := -L/opt/intel/mkl/lib/ -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -lmkl_intel_thread 
BLAS_LAPACK_LIB := -L/opt/intel/mkl/lib/ -lmkl_intel_lp64 -lmkl_core -lpthread -lm -lmkl_intel_thread -liomp5

# set compiler
CC = icpc

# set compilation flags
CC_FLAGS := -O2 -parallel -openmp

# specify include directories
#GLOBAL_INC := -I./ -I$(BLAS_LAPACK_INC) -I$(BOOST_DIR)
GLOBAL_INC := -I./ -I$(BLAS_LAPACK_INC)

# create list of object files
objects := $(subst src/,obj/,$(subst .cpp,.o,$(wildcard src/*.cpp)))

# create list of dependency files
dependencies := $(subst src/,dep/,$(subst .cpp,.d,$(wildcard src/*.cpp)))

TARGET = bin/TO.exe

# when make is run with no arguments, create the executable
.PHONY : default
default : $(TARGET)

# for each object file foo.o, create a dependency file foo.d in the dep/ directory
$(dependencies) : dep/%.d : src/%.cpp
	@echo "updating dependencies for $<";
	@$(CC) $(CC_FLAGS) $(GLOBAL_INC) -MM $< | sed -e "s/$*\.o/obj\/$*.o dep\/$*.d/" > $@;

# read in the rules that list the dependencies for each object and dependency file
sinclude $(dependencies)

# compile each object file
$(objects) : obj/%.o : src/%.cpp
	$(CC) $(CC_FLAGS) $(GLOBAL_INC) -c $< -o $@;

# create the executable
$(TARGET) : $(objects)
	$(CC) $(CC_FLAGS) $(objects) $(BLAS_LAPACK_LIB) -lpthread -o $@;


# rule to clean up object and dependency files
.PHONY : clean
clean : 
	rm -f obj/*.o;
	rm -f dep/*.d;