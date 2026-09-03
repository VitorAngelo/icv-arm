#!/bin/bash
set -e 

echo ">>> 01. Inferencia FLOAT32 (Referencia)"
gcc 01_inferencia_float.c -o 01_float -Wall -O3 -lm
./01_float

echo -e "\n>>> 02. Inferencia INT32 (Com Float Hibrido)"
gcc 02_inferencia_int32.c -o 02_int32 -Wall -O3 -lm
./02_int32

echo -e "\n>>> 03. Inferencia INT8 (Com Float Hibrido)"
gcc 03_inferencia_int8.c -o 03_int8 -Wall -O3 -lm
./03_int8

echo -e "\n>>> 04. Inferencia INT32 PURO (Simulacao de Hardware)"
gcc 04_inferencia_int32_puro.c -o 04_int32_puro -Wall -O3 -lm
./04_int32_puro

echo -e "\n>>> 05. Inferencia INT8 PURO (Simulacao Estrita ARM)"
gcc 05_inferencia_int8_puro.c -o 05_int8_puro -Wall -O3 -lm
./05_int8_puro
