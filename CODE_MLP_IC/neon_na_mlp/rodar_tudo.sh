#!/usr/bin/env bash
# rodar_tudo.sh - Compila e executa tudo em um PC x86 com Linux/WSL e grava os resultados.
#
# Uso: bash rodar_tudo.sh
# Requisitos: gcc, make, gcc-aarch64-linux-gnu, binutils-aarch64-linux-gnu, qemu-user
#
# 1. c_modular (x86 nativo): orquestrador com float32, int16 e int8
# 2. c_neon (aarch64 no QEMU): teste do kernel, validacao da MLP, assembly, ensaio do benchmark
# Saidas em resultados/. Tempos do QEMU sao de emulacao e nao medem desempenho.

set -u
AQUI="$(cd "$(dirname "$0")" && pwd)"
RES="$AQUI/resultados"
mkdir -p "$RES"
falhas=0

etapa() {
    local descricao=$1 arquivo=$2; shift 2
    printf '%-52s ' "$descricao"
    if "$@" > "$RES/$arquivo" 2>&1; then echo "ok"; else echo "FALHA (veja resultados/$arquivo)"; falhas=$((falhas + 1)); fi
}

echo "== ferramentas"
for c in gcc make aarch64-linux-gnu-gcc aarch64-linux-gnu-objcopy qemu-aarch64; do
    command -v "$c" > /dev/null || { echo "ERRO: $c nao instalado"; exit 1; }
done
{
    date '+%Y-%m-%d %H:%M:%S'
    uname -srm
    gcc --version | head -1
    aarch64-linux-gnu-gcc --version | head -1
    qemu-aarch64 --version | head -1
    lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)'
    echo
    cd "$AQUI" && md5sum c_modular/*.c c_modular/*.h c_modular/Makefile c_neon/*.c c_neon/*.h c_neon/Makefile modelo/exportado/*
} > "$RES/ambiente.txt" 2>&1

echo "== 1. c_modular (x86 nativo)"
cd "$AQUI/c_modular"
etapa "compilar c_modular" x86_c_modular_compilacao.txt sh -c 'make clean && make'
etapa "orquestrador todos (float32, int16, int8)" x86_c_modular_orquestrador.txt ./orquestrador todos

echo "== 2. c_neon (aarch64 emulado no QEMU)"
cd "$AQUI/c_neon"
etapa "compilar c_neon (cruzado, aarch64)" qemu_c_neon_compilacao.txt sh -c 'make clean && make'
etapa "teste isolado do kernel NEON" qemu_c_neon_teste_kernel.txt make --no-print-directory testar-kernel
etapa "validacao da MLP int8 NEON (10.000 imagens)" qemu_c_neon_validacao_mlp.txt make --no-print-directory validar-mlp
etapa "assembly (instrucoes NEON por objeto)" qemu_c_neon_assembly.txt make --no-print-directory verificar-asm
etapa "ensaio do benchmark (EMULACAO)" qemu_c_neon_ensaio_bench_EMULACAO.txt make --no-print-directory ensaiar-bench

{
    echo "RESUMO DA EXECUCAO - $(date '+%Y-%m-%d %H:%M:%S')"
    echo
    echo "[c_modular, x86 nativo]"
    sed -n '/COMPARACAO COM O PYTHON/,$p' "$RES/x86_c_modular_orquestrador.txt"
    echo
    echo "[c_neon, aarch64 emulado]"
    grep -E 'dot_i8_neon =|OK|FALHA|SUCESSO' "$RES/qemu_c_neon_teste_kernel.txt"
    grep -E 'OK|FALHA|SUCESSO|Acuracia|divergentes' "$RES/qemu_c_neon_validacao_mlp.txt"
    echo
    cat "$RES/qemu_c_neon_assembly.txt"
} > "$RES/resumo_execucao.txt" 2>&1

echo
if [ "$falhas" -eq 0 ]; then
    echo "SUCESSO: tudo executado. Resumo: resultados/resumo_execucao.txt"
else
    echo "FALHA em $falhas etapa(s). Veja os arquivos em resultados/."
fi
exit "$falhas"
