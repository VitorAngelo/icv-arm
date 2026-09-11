# Makefile - IC_APRESENTACAOFINAL
#
#   make          compila o comparador com o compilador ARM (aarch64)
#   make rodar    roda as 10.000 imagens e grava os txt em ../resultados/
#   make clean    apaga o executavel
#
# No PC (x86) usa aarch64-linux-gnu-gcc e roda com qemu-aarch64.
# Na Raspberry Pi 4 usa o gcc da placa e roda direto.

ifeq ($(shell uname -m),aarch64)
CC       = gcc
EXEC     =
EXECUCAO = nativa
else
CC       = aarch64-linux-gnu-gcc
EXEC     = qemu-aarch64
EXECUCAO = emulada_qemu
endif

CFLAGS = -std=c11 -O2 -march=armv8-a+simd -fno-tree-vectorize -ffp-contract=off -Wall -Wextra -static
FONTES = comparador.c mlp_float.c mlp_int16.c mlp_int8.c mlp_neon_int8.c

comparador: $(FONTES) mlp.h
	$(CC) $(CFLAGS) -I. -I../dados -o $@ $(FONTES) -pthread

rodar: comparador
	mkdir -p ../resultados
	$(EXEC) ./comparador ../dados ../resultados $(EXECUCAO)

clean:
	rm -f comparador

.PHONY: rodar clean
