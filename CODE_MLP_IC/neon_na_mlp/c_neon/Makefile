# Makefile - c_neon: kernels e inferencia NEON para aarch64 (Raspberry Pi 4).
#
# Compila de forma cruzada no WSL e executa no QEMU (correcao). Os tempos no
# QEMU sao de emulacao; desempenho so e medido na placa.
#
# Uso:
#   make                   compila todos os programas
#   make testar-kernel     compila e executa o teste isolado do kernel no QEMU
#   make validar-mlp       compila e executa a validacao da MLP int8 NEON no QEMU
#   make ensaiar-bench     executa o benchmark com poucos dados no QEMU (so funcionamento)
#   make verificar-asm     lista as instrucoes NEON de cada objeto (escalar deve ter nenhuma)
#   make clean             remove binarios e objetos
#
# Flags:
#   -march=armv8-a+simd    mesma arquitetura usada nos experimentos NEON da placa
#   -O2                    mesmo nivel de otimizacao dos experimentos
#   -static                a placa nao precisa das bibliotecas do WSL
#   -fno-tree-vectorize    so na referencia escalar: impede o GCC de vetoriza-la

CROSS    = aarch64-linux-gnu-
CC       = $(CROSS)gcc
OBJDUMP  = $(CROSS)objdump
OBJCOPY  = $(CROSS)objcopy
QEMU     = qemu-aarch64
DADOS    = ../modelo/exportado

MODULAR  = ../c_modular

CPPFLAGS = -I. -I$(MODULAR) -I$(DADOS)
CFLAGS   = -std=c11 -O2 -march=armv8-a+simd -Wall -Wextra -Wpedantic -ffp-contract=off
LDFLAGS  = -static
LDLIBS   = -pthread

PROGRAMAS = teste_dot_i8_neon valida_mlp_int8_neon bench_mlp_int8_neon

.PHONY: all testar-kernel validar-mlp ensaiar-bench verificar-asm clean

all: $(PROGRAMAS)

# --- kernels -----------------------------------------------------------------
dot_i8_neon.o: dot_i8_neon.c dot_i8_neon.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

dot_i8_ref.o: dot_i8_ref.c dot_i8_ref.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -fno-tree-vectorize -c -o $@ $<

# --- inferencia ---------------------------------------------------------------
mlp_int8_neon.o: mlp_int8_neon.c mlp_int8_neon.h dot_i8_neon.h $(MODULAR)/mlp.h $(DADOS)/mlp_pesos_int8.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

# Referencia: c_modular/mlp_int8.c compilado sem nenhuma alteracao.
mlp_int8_modular.o: $(MODULAR)/mlp_int8.c $(MODULAR)/mlp.h $(DADOS)/mlp_pesos_int8.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

# Variantes do benchmark: o mesmo c_modular/mlp_int8.c, sem alteracao,
# compilado duas vezes; objcopy renomeia o simbolo em cada objeto.
mlp_int8_escalar.o: $(MODULAR)/mlp_int8.c $(MODULAR)/mlp.h $(DADOS)/mlp_pesos_int8.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -fno-tree-vectorize -c -o $@.tmp $<
	$(OBJCOPY) --redefine-sym mlp_int8_inferir=mlp_int8_escalar_inferir $@.tmp $@
	rm -f $@.tmp

mlp_int8_autovet.o: $(MODULAR)/mlp_int8.c $(MODULAR)/mlp.h $(DADOS)/mlp_pesos_int8.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@.tmp $<
	$(OBJCOPY) --redefine-sym mlp_int8_inferir=mlp_int8_autovet_inferir $@.tmp $@
	rm -f $@.tmp

# --- programas ---------------------------------------------------------------
teste_dot_i8_neon: teste_dot_i8_neon.c dot_i8_neon.o dot_i8_ref.o dot_i8_neon.h dot_i8_ref.h $(DADOS)/mlp_pesos_int8.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ teste_dot_i8_neon.c dot_i8_neon.o dot_i8_ref.o

valida_mlp_int8_neon: valida_mlp_int8_neon.c mlp_int8_neon.o mlp_int8_modular.o dot_i8_neon.o dot_i8_ref.o mlp_int8_neon.h dot_i8_ref.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ valida_mlp_int8_neon.c mlp_int8_neon.o mlp_int8_modular.o dot_i8_neon.o dot_i8_ref.o

bench_mlp_int8_neon: bench_mlp_int8_neon.c mlp_int8_escalar.o mlp_int8_autovet.o mlp_int8_neon.o dot_i8_neon.o mlp_int8_neon.h mlp_int8_variantes.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ bench_mlp_int8_neon.c mlp_int8_escalar.o mlp_int8_autovet.o mlp_int8_neon.o dot_i8_neon.o $(LDLIBS)

# --- execucao e verificacoes -------------------------------------------------
testar-kernel: teste_dot_i8_neon
	$(QEMU) ./teste_dot_i8_neon $(DADOS)

validar-mlp: valida_mlp_int8_neon
	$(QEMU) ./valida_mlp_int8_neon $(DADOS)

ensaiar-bench: bench_mlp_int8_neon
	$(QEMU) ./bench_mlp_int8_neon 0 200 1 3 $(DADOS)

# Conta, por objeto, as instrucoes que operam registradores NEON (v0..v31).
verificar-asm: dot_i8_ref.o dot_i8_neon.o mlp_int8_escalar.o mlp_int8_autovet.o mlp_int8_neon.o
	@for o in $^; do \
	  printf '%-22s ' "$$o"; \
	  n=$$($(OBJDUMP) -d --no-show-raw-insn $$o | grep -cE '\bv[0-9]+\.'); \
	  if [ "$$n" = "0" ]; then echo "0 instrucoes NEON"; \
	  else $(OBJDUMP) -d --no-show-raw-insn $$o | grep -E '\bv[0-9]+\.' | awk '{print $$2}' | sort | uniq -c | tr -s ' ' | tr '\n' ' '; echo; fi; \
	done

clean:
	rm -f $(PROGRAMAS) *.o *.tmp
