# Compiler and flags
NVCC := nvcc
CFLAGS := -lcudart -lnccl

# Targets
TARGETS := server client

# Source files
SRCS := server.cc client.cc

# Object files
OBJS := $(SRCS:.cc=.o)

# Default target
all: $(TARGETS)

# Rule to build server
server: server.o
	$(NVCC) -o $@ $^ $(CFLAGS)

# Rule to build client
client: client.o
	$(NVCC) -o $@ $^ $(CFLAGS)

# Pattern rule to build object files
%.o: %.cc
	$(NVCC) -c $< -o $@

# Clean rule
clean:
	rm -rf $(TARGETS) $(OBJS)

.PHONY: all clean
