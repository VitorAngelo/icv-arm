# Makefile - compilacao dos modulos de inferencia e do orquestrador (x86, WSL).
#
# Uso:
#   make            compila ./orquestrador
#   make validar    compila e executa ./orquestrador todos
#   make clean      remove binario e objetos
#
# Flags:
#   -std=c11 -Wall -Wextra -Wpedantic   C padrao, com todos os avisos
#   -O2                                 otimizacao usual
#   -ffp-contract=off                   proibe fundir a*b + c em uma instrucao FMA.
#                                       FMA arredonda diferente e quebra a igualdade
#                                       bit a bit do modulo float (no aarch64, sem esta
#                                       flag, 10.000 de 10.000 logits divergem).
#   -I../modelo/exportado               pasta dos headers de pesos gerados pelo Python
#   -lm                                 biblioteca matematica, usada so pelo orquestrador
#                                       (p-valor de McNemar)

CC       = gcc
CPPFLAGS := -I../modelo/exportado
CFLAGS   = -std=c11 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off
LDFLAGS  =
LDLIBS   = -lm

# O modulo float e obrigatorio (referencia). int16 e int8 entram automaticamente
# quando mlp_int16.c e mlp_int8.c existirem nesta pasta; cada um liga uma
# definicao (-DMLP_COM_INT16, -DMLP_COM_INT8) que habilita o modulo no orquestrador.
OPCIONAIS := $(wildcard mlp_int16.c mlp_int8.c)
FONTES    := orquestrador.c mlp_float.c $(OPCIONAIS)
OBJETOS   := $(FONTES:.c=.o)
DEFINES   := $(if $(wildcard mlp_int16.c),-DMLP_COM_INT16) $(if $(wildcard mlp_int8.c),-DMLP_COM_INT8)

.PHONY: all validar clean

all: orquestrador

orquestrador: $(OBJETOS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Recompila o orquestrador quando um modulo opcional aparece ou muda.
orquestrador.o: orquestrador.c mlp.h $(OPCIONAIS)
	$(CC) $(CPPFLAGS) $(DEFINES) $(CFLAGS) -c -o $@ orquestrador.c

# Cada modulo depende do seu header de pesos: regenerar os pesos recompila o modulo.
mlp_%.o: mlp_%.c mlp.h ../modelo/exportado/mlp_pesos_%.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

validar: orquestrador
	./orquestrador todos

clean:
	rm -f orquestrador *.o
