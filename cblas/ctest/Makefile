# Compiler
CC = gcc

# OpenBLAS installation path
OPENBLAS_PATH = /opt/homebrew/opt/openblas

# Compiler flags
CFLAGS = -I$(OPENBLAS_PATH)/include

# Linker flags
LDFLAGS = -L$(OPENBLAS_PATH)/lib -lopenblas

# Target executable
TARGET = ctest

# Source files
SOURCES = ctest.c test_float.c test_double.c test_complex_float.c test_complex_double.c
OBJECTS = $(SOURCES:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

%.o: %.c cblas_test.h
	$(CC) -c $< $(CFLAGS)

run: $(TARGET)
	./$(TARGET)
	rm -f $(TARGET) *.o

clean:
	rm -f $(TARGET) *.o
