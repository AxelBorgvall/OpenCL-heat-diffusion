CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude
LDFLAGS = -lm -lOpenCL

SRC_DIR = src
OBJ_DIR = build
BIN_DIR = bin
EX_DIR=.

TARGET = $(EX_DIR)/diffusion
KERNEL_COMP = $(BIN_DIR)/comp_kern

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
OBJS := $(filter-out $(OBJ_DIR)/comp_kern.o, $(OBJS))

# Default target
all: $(BIN_DIR)/diffstep.bin $(TARGET)

# Link main executable
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Build kernel compiler
$(KERNEL_COMP): $(SRC_DIR)/comp_kern.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Compile OpenCL kernel into binary
$(BIN_DIR)/diffstep.bin: $(KERNEL_COMP) $(SRC_DIR)/diffstep.cl
	@echo "Compiling OpenCL kernel..."
	@$(KERNEL_COMP)

# Directory creation
$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

# Cleaning rules
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	rm $(TARGET) 

.PHONY: all clean
