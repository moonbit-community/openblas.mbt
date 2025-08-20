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

all: $(TARGET)

$(TARGET): ctest.o
	$(CC) -o $(TARGET) ctest.o $(LDFLAGS)

main.o: ctest.c
	$(CC) -c ctest.c $(CFLAGS)

run: $(TARGET)
	./$(TARGET)
	rm -f $(TARGET) *.o

clean:
	rm -f $(TARGET) *.o
