# NEON aplicado à MLP int8 — MLP 784→64→10 (MNIST)

Iniciação Científica — Engenharia Elétrica — UFMG. Orientador: Prof. Vitor Ângelo.

---

## Resumo (modelo IEEE)

**Inferência Quantizada de uma MLP com ARM NEON: Especificação em Ponto Fixo e Validação Bit a Bit no Raspberry Pi 4**

*Davi (sobrenome), Vitor Ângelo (sobrenome) — Universidade Federal de Minas Gerais (UFMG), Belo Horizonte, Brasil*

***Resumo*** — Este trabalho implementa em C a inferência de uma rede perceptron multicamadas
784–64–10 com ativação ReLU, treinada com scikit-learn no conjunto MNIST, em três representações
numéricas (float32, int16 e int8), com o objetivo de acelerar a versão int8 com instruções SIMD
ARM NEON no processador Cortex-A72 do Raspberry Pi 4. A quantização simétrica por tensor foi
convertida em uma especificação exclusivamente inteira: tabela de consulta na entrada, acumulação
inteira e requantização por multiplicador inteiro e deslocamento de bits. As três implementações
reproduzem, bit a bit, referências geradas em Python; a acurácia foi 0,9708 em float32, 0,9725 em
int16 e 0,9728 em int8. Um núcleo NEON de produto escalar int8 (`vmull_s8` e `vpadalq_s16`),
desenvolvido e validado na placa em experimentos anteriores, foi aplicado sem alteração às duas
camadas da MLP. Em aarch64 emulado, a versão NEON produziu logits e matriz de confusão idênticos à
versão escalar nas 10.000 imagens de teste. Foi preparado um benchmark com núcleo fixo,
aquecimento, repetições e mediana, que compara as versões escalar, autovetorizada e NEON. A
medição de desempenho no Raspberry Pi 4 está em andamento.

***Palavras-chave*** — ARM NEON, quantização, inferência embarcada, SIMD, MLP, Raspberry Pi.

---

## 1. O que é esta pasta

Esta pasta junta **dois trabalhos que eu já tinha feito e testado** e mostra a aplicação de um ao outro:

1. **O meu NEON já pronto:** produto escalar int8 com intrinsics dos experimentos `ICV ARM NEON`
   (`dot_neon.c`, validado na placa) e `experimentos_neon_2` (`homogeneidade_v2.c`, com a
   metodologia de medição). Cópias em `origem/`.
2. **A minha MLP modular já pronta:** `c_modular/`, com inferência float32, int16 e int8 validada
   bit a bit contra o Python.
3. **O trabalho novo, `c_neon/`:** aplica o kernel NEON às duas camadas da MLP int8 **sem alterar
   nenhum dos dois**, e valida e mede o resultado.

## 2. De onde vem cada parte

| Arquivo | Origem | Alteração |
|---|---|---|
| `c_neon/dot_i8_neon.c` | `neon_dot_i8` de `origem/experimentos_neon_2/homogeneidade_v2.c` (mesmo algoritmo de `dot_product_int8_neon` em `origem/ICV_ARM_NEON/dot_neon.c`) | só o nome e a remoção de `static` |
| `c_neon/dot_i8_ref.c` | `dot_i8` de `homogeneidade_v2.c` (referência escalar do professor) | só o nome; acrescentada soma em 64 bits |
| `c_neon/bench_mlp_int8_neon.c` | metodologia de medição de `homogeneidade_v2.c` | adaptada para inferência por imagem |
| `c_modular/*` | minha MLP modular | nenhuma |
| `modelo/exportado/*` | gerado pelo meu `python/exporta_modelo.py` | nenhuma |

## 3. Estrutura

```
neon_na_mlp/
├── README.md                     este arquivo
├── rodar_tudo.sh                 compila, executa tudo e grava resultados/
├── c_modular/                    MLP modular: float32, int16, int8 e orquestrador (x86 ou ARM)
├── c_neon/
│   ├── dot_i8_neon.h/.c          meu kernel NEON (vmull_s8 + vpadalq_s16)
│   ├── dot_i8_ref.h/.c           produto escalar de referência (escalar puro)
│   ├── teste_dot_i8_neon.c       teste isolado do kernel contra a referência
│   ├── mlp_int8_neon.h/.c        MLP int8 usando o kernel nas camadas 784→64 e 64→10
│   ├── valida_mlp_int8_neon.c    NEON × escalar × Python, estágio por estágio
│   ├── mlp_int8_variantes.h      variantes escalar pura e autovetorizada de mlp_int8.c
│   ├── bench_mlp_int8_neon.c     benchmark escalar × autovetorizado × NEON
│   └── Makefile                  compilação cruzada para aarch64 e alvos de teste
├── modelo/exportado/             pesos, 10.000 imagens de teste e referências do Python
├── origem/                       meus experimentos NEON anteriores
└── resultados/                   saídas geradas por rodar_tudo.sh
```

## 4. Como rodar

Em um PC x86 com Linux ou WSL:

```bash
sudo apt install -y gcc make gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu qemu-user
bash rodar_tudo.sh
```

Ou etapa por etapa:

```bash
cd c_modular && make && ./orquestrador todos     # MLP float32/int16/int8, x86 nativo
cd ../c_neon && make                             # compila para ARM (aarch64)
make testar-kernel                               # kernel NEON x escalar (QEMU)
make validar-mlp                                 # MLP NEON x escalar x Python (QEMU)
make verificar-asm                               # instrucoes NEON geradas
make ensaiar-bench                               # benchmark curto (emulacao)
```

Na Raspberry Pi 4, com os binários ARM já compilados:

```bash
./teste_dot_i8_neon ../modelo/exportado
./valida_mlp_int8_neon ../modelo/exportado
./bench_mlp_int8_neon 3 10000 3 21     # nucleo imagens runs trials
```

O NEON não roda nativamente em x86. O **QEMU** executa o binário ARM no PC reproduzindo cada
instrução exatamente: serve para provar a **correção**. Os tempos no QEMU são de emulação e **não**
servem para desempenho.

## 5. Modelo e técnicas

| Item | O que é | Por quê |
|---|---|---|
| MNIST | dígitos 28×28 (0–255); 60.000 imagens de treino e 10.000 de teste | conjunto padrão; o teste não ajusta nada |
| StandardScaler | `x = (pixel − média) / desvio`, por pixel, calculado no treino | coloca as entradas na mesma escala |
| MLP 784→64→10 | `MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam", alpha=1e-4, batch_size=128, learning_rate_init=1e-3, max_iter=30, early_stopping=True, random_state=42)` | rede densa simples; o custo fica nas multiplicações matriz-vetor |
| **ReLU** | `h = max(a, 0)` na camada oculta | barata; em inteiros é só uma comparação; `max(0, s·A) = s·max(0, A)` para escala positiva, então pode ser aplicada no acumulador inteiro |
| Softmax / argmax | softmax só no treino; na inferência a classe é o índice do maior logit | a softmax é crescente e tem denominador comum: não muda o argmax |
| Quantização simétrica int8 | `T ≈ s·T_q`, `T_q ∈ [−127, 127]`; pesos com `s = max|W|/127`; ativações com o percentil 99,9 em 5.000 imagens de treino | zero real vira 0, sem deslocamento; o percentil evita que entradas extremas raras (até 419,6 desvios) desperdicem a faixa do int8 |
| Tabela de entrada `MLP8_LUT` | 784×256 valores int8 com padronização + quantização prontas | elimina 784 divisões em float por imagem; exata, pois o pixel só tem 256 valores |
| Bias `MLP8_B0` | `round(b0 / (s_x·s_W0))`, int32 | fica na mesma unidade da soma inteira |
| Requantização `M`, `N` | `h = min((max(A,0)·M + 2^(N−1)) >> N, 127)`, M = 1.132.510.016, N = 39 | troca o fator real `s_x·s_W0/s_h` por inteiros; `2^(N−1)` arredonda |
| int16 | mesmo método com 32767 | controle de precisão do int8 |
| Kernel NEON | 16 int8 por iteração; `vmull_s8` (int8×int8→int16) e `vpadalq_s16` (soma em pares acumulando em int32) | o produto não cabe em int8; o Cortex-A72 (ARMv8.0) não tem `vdotq_s32` |

## 6. Como os resultados foram medidos e por quê

### 6.1 Correção: comparação bit a bit

- **Como:** a mesma imagem passa pelas versões escalar e NEON, e cada valor é comparado exatamente:
  entrada quantizada, produto, produto + bias, ReLU, ativação, logits e classe. Os logits também são
  comparados com a referência do Python (`ref_int8.bin`). O kernel é testado antes, sozinho, contra
  o escalar e contra a soma exata em 64 bits.
- **Por quê:**
  - com aritmética inteira e sem estouro, o esperado é igualdade exata;
  - **a acurácia sozinha esconde diferenças:** float32 e int8 diferem em só 20 acertos, mas tomam
    **47 decisões diferentes**;
  - a comparação por estágio localiza onde uma divergência começa.

### 6.2 Acurácia e matriz de confusão

- **Acurácia** = acertos / 10.000.
- **Matriz de confusão:** a célula [real][previsto] conta quantas imagens de um dígito foram
  classificadas como outro; a diagonal são os acertos.
- **Precisão** da classe k: dos que a rede chamou de k, quantos eram k. **Revocação:** dos que eram
  k, quantos foram reconhecidos. **F1:** média harmônica das duas.
- **Por quê:** a matriz mostra *quais* dígitos a rede confunde. Ser idêntica entre NEON e escalar
  confirma, célula a célula, que as decisões são as mesmas.

### 6.3 Tempo no orquestrador (`c_modular`, x86): só indicativo

- **Como:** cada inferência é cronometrada com `CLOCK_MONOTONIC`; reporta a média por imagem em uma
  passagem pelas 10.000 imagens, sem fixar núcleo nem aquecer.
- **Por quê:** o objetivo do orquestrador é validar. O tempo só dá ordem de grandeza em x86 e varia
  entre execuções (float: 30 µs numa e 42 µs noutra). Não vale para a placa.

### 6.4 Tempo no benchmark NEON: método do professor

`c_neon/bench_mlp_int8_neon.c` segue `origem/experimentos_neon_2/homogeneidade_v2.c`:

| Decisão | Justificativa |
|---|---|
| Fixar a thread em um núcleo (`pthread_setaffinity_np`) | evita migração entre núcleos e perda de cache. No Pi 4 os núcleos se mostraram homogêneos (spread de 0,0% em `homogeneidade_v2`), mas fixar mantém as condições iguais |
| Carregar as imagens na RAM antes | leitura de arquivo é lenta e variável, e não é inferência |
| Aquecer: 1 passagem inicial e 1 inferência antes de cada bloco | a primeira execução paga falhas de página, cache fria e subida de frequência |
| `CLOCK_PROCESS_CPUTIME_ID` | conta só o tempo de CPU do processo, sem o tempo em que outro programa usou a CPU |
| Cronometrar `runs` passagens juntas e dividir | uma inferência dura microssegundos; blocos grandes diluem o custo do relógio |
| Vários trials (padrão 21) | uma medição isolada pode pegar uma interrupção do sistema |
| Mediana (com média e mínimo) | a mediana ignora picos raros; média muito acima da mediana indica picos |
| Girar a ordem das variantes a cada trial | `ordem_cache.c` mostrou que medir logo após outra variante pode favorecer a segunda (razão frio/quente de até 1,46) |
| Mesmas imagens nas três variantes | comparação justa |
| Três variantes: escalar puro, autovetorizado e NEON | o GCC com `-O2` já vetoriza o escalar (confirmado no assembly); sem `-fno-tree-vectorize`, compararíamos NEON manual com NEON automático |
| Portão de correção e soma de verificação | só mede se as variantes derem logits idênticos e confere que o trabalho medido é o mesmo |
| Registrar governador, frequência e temperatura | frequência e aquecimento alteram o tempo |
| Tempo por imagem; speedup = mediana do escalar / mediana do NEON | unidade comparável |

**Limitações desta medição:**
- todos os trials rodam num único processo;
- o argmax entra no tempo, igualmente para as três variantes;
- o kernel NEON é chamado por função (74 chamadas por imagem), o que pesa só na variante NEON.

## 7. Resultados medidos

### 7.1 MLP modular (x86 nativo, orquestrador)

| Versão | Acurácia | Logits bit a bit | Decisões diferentes do float | Tempo por imagem* |
|---|---|---|---|---|
| float32 | 0,9708 | idênticos | 0 | 29,9 µs |
| int16 | 0,9725 | idênticos | 36 | 7,7 µs |
| int8 | 0,9728 | idênticos | 47 | 3,2 µs |

\* x86 no WSL, uma passagem; indicativo. O mesmo código compilado para aarch64 também valida no QEMU.

### 7.2 Matriz de confusão do int8 (idêntica no NEON)

| real \ previsto | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** | **968** | 1 | 2 | 0 | 0 | 1 | 4 | 2 | 2 | 0 |
| **1** | 0 | **1125** | 3 | 0 | 0 | 1 | 2 | 1 | 3 | 0 |
| **2** | 3 | 2 | **995** | 7 | 4 | 0 | 3 | 4 | 13 | 1 |
| **3** | 1 | 0 | 3 | **982** | 0 | 7 | 1 | 6 | 5 | 5 |
| **4** | 0 | 0 | 9 | 1 | **959** | 0 | 4 | 1 | 3 | 5 |
| **5** | 3 | 0 | 1 | 13 | 3 | **857** | 4 | 0 | 9 | 2 |
| **6** | 3 | 3 | 4 | 1 | 5 | 5 | **935** | 0 | 2 | 0 |
| **7** | 1 | 4 | 14 | 3 | 4 | 0 | 0 | **991** | 3 | 8 |
| **8** | 3 | 2 | 2 | 4 | 4 | 3 | 3 | 3 | **948** | 2 |
| **9** | 3 | 4 | 1 | 5 | 12 | 1 | 0 | 5 | 10 | **968** |

Maiores confusões: 7→2 (14), 5→3 (13), 2→8 (13), 9→4 (12). A classe 8 tem a menor precisão
(0,9499), e a classe 1 o maior F1 (0,9886). A média macro do F1 é 0,9726.

### 7.3 Kernel NEON isolado (aarch64 no QEMU)

| Grupo | Casos | Diferenças |
|---|---|---|
| Vetor da placa (`srand(0)`, 16.384 elementos): **576483**, igual ao registrado na placa | 1 | 0 |
| Tamanhos múltiplos de 16, de 16 a 4096 | 5.120 | 0 |
| Extremos e impulsos (784 e 64) | 864 | 0 |
| Desalinhamento de memória | 800 | 0 |
| Vetores reais da camada 784→64 | 640.000 | 0 |
| Vetores reais da camada 64→10 | 100.000 | 0 |

### 7.4 MLP int8 com NEON (aarch64 no QEMU)

| Comparação nas 10.000 imagens | Valores | Diferenças |
|---|---|---|
| Entrada quantizada | 7.840.000 | 0 |
| Produto, produto + bias, ReLU, ativação (cada) | 640.000 | 0 |
| Logits | 100.000 | 0 |
| Classe prevista | 10.000 | 0 |
| Logits NEON × escalar (c_modular) | 100.000 | 0 |
| Logits NEON × Python | 100.000 | 0 |

Acurácia 0,9728 no escalar e no NEON; 0 classificações divergentes; `c_modular/mlp_int8.c` inalterado.

### 7.5 Código gerado (GCC 15)

| Objeto | Instruções NEON |
|---|---|
| escalar puro (`-fno-tree-vectorize`) | nenhuma |
| `mlp_int8.c` com `-O2` | 2 `smull`, 2 `smull2`, 4 `saddw`, 4 `saddw2` (vetorizado pelo GCC) |
| kernel manual | 2 `smull`, 2 `sadalp` |

## 8. Expectativas para a placa (hipóteses, a medir)

- **Correção: expectativa forte.** Mesmos logits bit a bit, acurácia 0,9728 e 0 divergências. A
  aritmética é inteira e o mesmo binário já validou no QEMU.
- **Kernel isolado, experimentos anteriores na placa:** `regimes_memoria` deu 5 a 6× (dados no L1D
  e no L2). `homogeneidade_v2` deu 2,2×; ali os flags não foram registrados, e o escalar levou
  0,45 ns por elemento, o que sugere comparação com um escalar já vetorizado.
- **Na MLP o ganho deve ser menor**, porque parte do tempo não é produto escalar: 784 leituras da
  tabela, 64 requantizações e 74 chamadas por imagem. Pela lei de Amdahl,
  `speedup = 1 / ((1 − f) + f / S)`:

| f \ S do kernel | 1,5× | 2,2× | 5,5× |
|---|---|---|---|
| 0,80 | 1,36× | 1,77× | 2,89× |
| 0,90 | 1,43× | 1,96× | 3,79× |
| 0,95 | 1,46× | 2,08× | 4,49× |

- **Faixas esperadas:** NEON contra escalar puro, de 3 a 4,5×; NEON contra autovetorizado, de 1,3 a
  2×, ou menos se as chamadas de função pesarem.
- **Memória:** os pesos da camada 1 (49 KB) e a tabela (200 KB) cabem no L2 de 1 MB do Cortex-A72.

## 9. Propostas de medição (próximos passos)

Estas análises **não** fazem parte dos resultados acima; são propostas para dar mais rigor.

| Proposta | O que é, em termos simples | Exemplo com os meus números |
|---|---|---|
| Intervalo de confiança de **Wilson** | 10.000 imagens são uma amostra; o intervalo de 95% dá a faixa plausível da acurácia verdadeira | int8: 0,9728 em [0,9694; 0,9758], cerca de ±0,3 ponto percentual |
| Teste de **McNemar** | compara dois modelos nas mesmas imagens usando só as discordâncias (só A acerta × só B acerta). Se fossem igualmente bons, cada discordância seria uma moeda justa; *p* é a chance do desequilíbrio observado acontecer por acaso | float × int8: 30 contra 10, *p* = 0,0022; int16 × int8: 7 contra 4, *p* = 0,55 (acaso). O orquestrador já imprime |
| Correção de **Bonferroni** | com muitos testes, o limiar de 5% é dividido pelo número de testes para evitar falsos "significativos" | 10 testes exigem *p* < 0,005 |
| **Ablação** do recorte | acrescentar uma transformação por vez ao modelo float para ver qual muda a acurácia | preliminar: só o recorte da entrada muda (0,9708 → 0,9725); a quantização não |
| Repetir o benchmark várias vezes | mede a variação entre execuções, e não só entre trials | 5 execuções independentes |
| Intervalo da mediana por **bootstrap** | reamostrar os trials para obter a faixa plausível da mediana e do speedup | — |
| Kernel isolado em 64, 784 e 16.384 | separa o ganho do kernel do ganho da MLP inteira | — |
| Entradas fora do MNIST | ruído, imagens invertidas e deslocadas, para testar a correção em casos extremos | — |
| Kernel `inline` | elimina as 74 chamadas de função por imagem sem mudar o corpo | — |

## 10. Limitações

- A correção do NEON foi verificada em emulação; ainda falta repetir na placa.
- Um único modelo treinado com uma semente.
- O kernel exige vetores com tamanho múltiplo de 16 (784 e 64 atendem).
- O teste do vetor da placa depende do `rand()` da glibc.
