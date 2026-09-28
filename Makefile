CC      = gcc
NASM    = nasm
CFLAGS  = -Wall -Wextra -O2 -g -Iinclude
NASMFLAGS = -f elf64 -g -F dwarf
LDFLAGS = -lm

.PHONY: all clean dirs

all: bin/norm_scalar bin/norm_vector

dirs:
	mkdir -p bin obj data

bin/norm_scalar: obj/driver.o obj/stats_scalar.o | dirs
	$(CC) $(CFLAGS) -o $@ obj/driver.o obj/stats_scalar.o $(LDFLAGS)

bin/norm_vector: obj/driver.o obj/stats_vector.o | dirs
	$(CC) $(CFLAGS) -o $@ obj/driver.o obj/stats_vector.o $(LDFLAGS)

obj/driver.o: main/driver.c include/stats.h | dirs
	$(CC) $(CFLAGS) -c main/driver.c -o $@

obj/stats_scalar.o: escalar/stats_scalar.asm | dirs
	$(NASM) $(NASMFLAGS) escalar/stats_scalar.asm -o $@

obj/stats_vector.o: vectorial/stats_vector.asm | dirs
	$(NASM) $(NASMFLAGS) vectorial/stats_vector.asm -o $@

clean:
	rm -rf obj bin
