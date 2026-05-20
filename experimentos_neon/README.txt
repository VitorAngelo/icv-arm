icv-arm - Experimentos de Caracterizacao do Pi 4 (Cortex-A72)
=============================================================

CONTEXTO
--------
Pergunta sobre o Pi 4: os 4 nucleos Cortex-A72 sao homogeneos
apenas em especificacao, ou tambem empiricamente? A partir disso, defini tres
experimentos independentes.


EXPERIMENTO 1 - homogeneidade.c
--------------------------------
Pergunta: os 4 nucleos tem performance igual na pratica?

O professor travou num unico core por causa do Orange Pi heterogeneo. O Pi 4 e
homogeneo em especificacao, mas especificacao nao e medicao. Temperatura diferente
por nucleo, historico de carga e compartilhamento do L2 podem criar assimetria
real. Este experimento pina a thread em cada core separadamente, despeja o cache
antes de cada medicao e compara escalar e NEON nos 4 nucleos.

Hipotese: variancia < 5% entre os cores, confirmando que no Pi 4 a fixacao de
nucleo e precaucao metodologica, nao necessidade como no Orange Pi.

Uso: ./homogeneidade <nucleo> <tamanho_kb>
     ./homogeneidade 0 16


EXPERIMENTO 2 - ordem_cache.c
-------------------------------
Pergunta: medir os tipos em sequencia contamina os resultados?

O codigo do professor mede sempre na ordem i8 -> i32 -> f32. Quando o i32 e medido,
os arrays ja percorreram o caminho de memoria e parte dos dados pode ainda estar
no cache, aquecendo a medicao seguinte. O speedup de 4.6x obtido no relatorio pode
conter uma fracao de cache aquecido misturada com o ganho aritmetico real.

Este experimento mede cada tipo duas vezes: uma fria (pos-flush) e uma quente
(imediata). O ratio frio/quente quantifica o quanto o cache estava interferindo.

Uso: ./ordem_cache <nucleo> <tamanho_kb>
     ./ordem_cache 0 16       (L1D)
     ./ordem_cache 0 256      (L2)
     ./ordem_cache 0 2048     (DRAM)


EXPERIMENTO 3 - regimes_memoria.c
-----------------------------------
Pergunta: o speedup do NEON e constante ou depende do regime de memoria?

O codigo do professor usa klen fixo, medindo num unico ponto do espaco. O que ele
nao mostra e como o speedup varia conforme o working set cresce alem do L1D e do
L2. NEON processa 16 elementos por ciclo, mas isso so se realiza se os dados
chegarem rapido o suficiente. Com dado na DRAM , escalar
e NEON ficam esperando igualmente e o speedup cai.

Varre de 1 KB a 4 MB em progressao x2 e reporta throughput e speedup por regime.

Uso: ./regimes_memoria <nucleo>
     ./regimes_memoria 0


COMPILAR (na placa)
--------------------
gcc -O2 -march=armv8-a+simd homogeneidade.c  -lpthread -o homogeneidade
gcc -O2 -march=armv8-a+simd ordem_cache.c    -lpthread -o ordem_cache
gcc -O2 -march=armv8-a+simd regimes_memoria.c -lpthread -o regimes_memoria
