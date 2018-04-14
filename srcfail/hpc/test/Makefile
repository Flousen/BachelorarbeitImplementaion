TEST_SRC   = $(wildcard *.cpp)
TEST_OBJ   = $(patsubst %.cpp,%.o,$(TEST_SRC))
TEST       = $(patsubst %.cpp,%,$(TEST_SRC))

################################################################################

CXX       = g++
CXXFLAGS  = -std=c++14
CXXFLAGS += -Wall -O3 -mavx -m64 -g
CPPFLAGS += -I../..

Arch := intel64
IntelDir := /opt/intel/compilers_and_libraries/linux
#IntelDir := /home/numerik/lehn/intel/
IncludeDir := $(IntelDir)/mkl/include
LibDirs := $(IntelDir)/lib/$(Arch) $(IntelDir)/mkl/lib/$(Arch)
LinkerFlag := -Wl,
Rpath := $(patsubst %,$(LinkerFlag)-rpath $(LinkerFlag)%,$(LibDirs))
Lpath := $(patsubst %,-L%,$(LibDirs))

LDFLAGS += $(Lpath) $(Rpath)
CPPFLAGS += -I$(IncludeDir) -DMKL_ILP64
LDLIBS += -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lm -lpthread

all:	$(TEST)

clean:
	$(RM) $(TEST) $(TEST_OBJ)
	$(RM) core
