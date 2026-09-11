mlp_int8_neon_mnist
===================
MLP int8 com ARM NEON no MNIST, em C
Iniciacao Cientifica - Engenharia Eletrica - UFMG
Davi - Orientador: Prof. Vitor Angelo

O PROJETO
---------
Rodar as 10.000 imagens de teste do MNIST numa MLP 784-64-10 em C
(uma camada oculta com 64 neuronios, ativacao ReLU).

- Mostrar como fica o int8 em C.
- Mostrar como fica o int8 em C com NEON.
- Comparar os dois.
- Calcular a aceleracao NEON teorica em ARM para essa MLP.
- Saida no terminal: uma matriz de confusao comparando os dois e uma
  tabelinha comparando os dois com o int16 e o float em C.

Acuracia: medida como em testar_acuracia.c (acertos, acuracia, decisoes
alteradas em relacao ao float, matriz de confusao).
Tempo: medido com a minha funcao de tempo de homogeneidade_simples.c (warmup,
runs cronometrados juntos com CLOCK_PROCESS_CPUTIME_ID, divisao pelo numero de
runs, varios trials, media e mediana).

ESTRUTURA
---------
mlp_int8_neon_mnist/
  README.txt                 este arquivo
  codigos/
    mlp.h                    interface das MLPs
    mlp_float.c              MLP float32
    mlp_int16.c              MLP int16 (sem float)
    mlp_int8.c               MLP int8 (sem float)
    mlp_neon_int8.c          MLP int8 com o meu kernel NEON (vmull_s8 + vpadalq_s16)
    comparador.c             unico codigo de teste: roda tudo e grava os resultados
    Makefile
  dados/                     pesos e imagens (ver dados/LEIA.txt)
  resultados/                gerados pelo comparador
    comparacao_float_int16_int8.txt
    comparacao_int8_int8neon.txt

COMO RODAR
----------
No PC (Linux ou WSL) com compilador ARM e QEMU:
  sudo apt install -y make gcc-aarch64-linux-gnu qemu-user
  cd codigos
  make rodar

Na Raspberry Pi 4 o mesmo "make rodar" usa o gcc da placa e roda direto.

Leva cerca de 1 minuto no QEMU. No fim aparecem a tabela, a matriz de
confusao comparada, a aceleracao teorica e os dois txt em resultados/.

RESULTADO OBTIDO
----------------
versao      acertos  acuracia  dif_float  mediana_us/img  aceleracao
float32        9708    0.9708          0         293.743       0.34x
int16          9725    0.9725         36          99.079       1.00x
int8           9728    0.9728         47          98.930       1.00x
int8 NEON      9728    0.9728         47          35.231       2.81x

celulas diferentes entre int8 e int8 NEON: 0
imagens com logits diferentes entre int8 e int8 NEON : 0
RESULTADO: OK - int8 NEON identico ao int8 nas 10.000 imagens

Aceleracao teorica: 50.816 multiplicacoes por imagem viram 3.176 iteracoes
NEON de 16 multiplicacoes: limite de 16x nos produtos escalares; pela lei de
Amdahl, de 4,0x a 9,1x se 80% a 95% do tempo estiver nos produtos.

OBSERVACAO
----------
No PC o codigo ARM roda emulado pelo QEMU: acuracia, matriz de confusao e a
comparacao dos logits valem; o tempo e o da emulacao (execucao: emulada_qemu)
e nao representa a placa.
