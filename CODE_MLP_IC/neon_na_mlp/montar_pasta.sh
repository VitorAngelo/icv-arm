#!/usr/bin/env bash
# montar_pasta.sh - Monta esta pasta com os arquivos exatamente como foram testados.
#
# Uso (no WSL, uma vez):
#   bash montar_pasta.sh [pasta_CODE_MLP_IC] [pasta_inferencia_neon]
#   padrao: /mnt/c/Users/Davi/Desktop/obsidian/arquivos_ufmg/MLP_IC/CODE_MLP_IC
#           /mnt/c/Users/Davi/Desktop/inferencia_neon   (so para copiar origem/)
#
# Copia, SEM ALTERAR:
#   CODE_MLP_IC/c_modular/          6 arquivos-fonte da MLP modular (float32, int16, int8)
#   CODE_MLP_IC/c_neon/             11 arquivos do NEON aplicado a MLP (versao inicial testada)
#   CODE_MLP_IC/modelo/exportado/   pesos, imagens de teste e referencias geradas em Python
#   inferencia_neon/origem/         experimentos NEON anteriores (fonte do kernel)
# Confere o md5 de cada codigo contra a versao testada e, no fim, executa rodar_tudo.sh.

set -euo pipefail
AQUI="$(cd "$(dirname "$0")" && pwd)"
PROJETO="${1:-/mnt/c/Users/Davi/Desktop/obsidian/arquivos_ufmg/MLP_IC/CODE_MLP_IC}"
INFNEON="${2:-/mnt/c/Users/Davi/Desktop/inferencia_neon}"
falhar() { echo "ERRO: $*" >&2; exit 1; }

mkdir -p "$AQUI/c_modular" "$AQUI/c_neon" "$AQUI/modelo/exportado" "$AQUI/origem/ICV_ARM_NEON" "$AQUI/origem/experimentos_neon_2"

echo "== copiando c_modular"
for f in mlp.h mlp_float.c mlp_int16.c mlp_int8.c orquestrador.c Makefile; do
    cp "$PROJETO/c_modular/$f" "$AQUI/c_modular/" || falhar "ausente: $PROJETO/c_modular/$f"
done

echo "== copiando c_neon (versao inicial)"
for f in Makefile dot_i8_neon.h dot_i8_neon.c dot_i8_ref.h dot_i8_ref.c teste_dot_i8_neon.c \
         mlp_int8_neon.h mlp_int8_neon.c valida_mlp_int8_neon.c mlp_int8_variantes.h bench_mlp_int8_neon.c; do
    cp "$PROJETO/c_neon/$f" "$AQUI/c_neon/" || falhar "ausente: $PROJETO/c_neon/$f"
done

echo "== copiando modelo/exportado"
for f in mlp_pesos_float.h mlp_pesos_int16.h mlp_pesos_int8.h teste_u8.bin ref_float32.bin \
         ref_int16.bin ref_int8.bin ref_python_pred.bin relatorio_exportacao.txt; do
    cp "$PROJETO/modelo/exportado/$f" "$AQUI/modelo/exportado/" || falhar "ausente: $PROJETO/modelo/exportado/$f"
done

echo "== copiando origem (experimentos NEON anteriores)"
for f in dot_neon.c dot_scalar.c relatorio_neon.txt; do
    cp "$INFNEON/origem/ICV_ARM_NEON/$f" "$AQUI/origem/ICV_ARM_NEON/" || falhar "ausente: $INFNEON/origem/ICV_ARM_NEON/$f"
done
for f in homogeneidade_v2.c resultado_homogeneidade.txt; do
    cp "$INFNEON/origem/experimentos_neon_2/$f" "$AQUI/origem/experimentos_neon_2/" || falhar "ausente: $INFNEON/origem/experimentos_neon_2/$f"
done

echo "== conferindo md5 (versoes testadas)"
cd "$AQUI"
md5sum -c --quiet <<'LISTA' || falhar "algum codigo difere da versao testada"
19f880e433a1d94423893dac97ceab16  c_modular/mlp.h
1f21bf7523bf5ad9b6f49faa1b7d8e12  c_modular/mlp_float.c
2d405d465a0f45728b88f4ba05f8dcfd  c_modular/mlp_int16.c
e8476fdec201ea0b99ac32acf1c30be4  c_modular/mlp_int8.c
96970be896806411b8b36816d4032975  c_modular/orquestrador.c
fe30e3df822c756d4358872acacd613b  c_modular/Makefile
bdabff464e619b5b3654fed8fcdc6970  c_neon/Makefile
5770778a7de52347a20fb40c517bf260  c_neon/dot_i8_neon.c
d99985be4b20c6ecc53bc32ced3b515b  c_neon/dot_i8_neon.h
69192c758bd4697f0132b00b8c1ca190  c_neon/dot_i8_ref.c
71b0d54eb730bd5ed96ca1ebe125e632  c_neon/dot_i8_ref.h
e8315a9014d02db4b112ea96cf15bae6  c_neon/teste_dot_i8_neon.c
733601d65bb6d1717699b3e7a1023737  c_neon/mlp_int8_neon.c
eb4545f6dd587b20bb4edf41943f31f6  c_neon/mlp_int8_neon.h
189919a81b2e04cc9f2f15c28d3ca104  c_neon/valida_mlp_int8_neon.c
6ae072b3df511bd77a28012fb8538601  c_neon/mlp_int8_variantes.h
3ceb20de1fa08bac51ceff950915fa85  c_neon/bench_mlp_int8_neon.c
LISTA
echo "   ok: todos os codigos sao as versoes testadas"

bash "$AQUI/rodar_tudo.sh"
