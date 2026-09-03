#!/bin/bash
# Para o processo imediatamente se houver falha de compilação
set -e 

echo "=================================================="
echo ">>> C Etapa 1 - Inferencia FLOAT32"
echo "=================================================="
gcc 01_inferencia_float.c -o 01_float -Wall -O3 -lm
./01_float

echo ""
echo "=================================================="
echo ">>> C Etapa 2 - Inferencia INT32"
echo "=================================================="
gcc 02_inferencia_int32.c -o 02_int32 -Wall -O3 -lm
./02_int32

echo ""
echo "=================================================="
echo ">>> C Etapa 3 - Inferencia INT8 (Pronto pro ARM)"
echo "=================================================="
gcc 03_inferencia_int8.c -o 03_int8 -Wall -O3 -lm
./03_int8

echo ""
echo "##################################################"
echo "# SUCESSO: As tres versoes rodaram perfeitamente."
echo "##################################################"
