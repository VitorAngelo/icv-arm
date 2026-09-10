# Inferência de MLP 784-64-10 (MNIST) em C: float32, int16 e int8

Iniciação Científica — Engenharia Elétrica — UFMG. Orientador: Prof. Vitor Ângelo.
Pasta: `CODE_MLP_IC/inferencia_modular/` (autocontida; a versão anterior está em `../c`).

Inferência de uma rede treinada em scikit-learn, reimplementada em C modular.
Os caminhos int16 e int8 não usam ponto flutuante. Os três módulos são
validados bit a bit contra referências geradas em Python, como base para a
aceleração com NEON no Raspberry Pi 4 (Cortex-A72).

## Execução rápida (Linux ou WSL, com gcc e make)

```bash
cd c_modular
make
./orquestrador todos
```

Resultado esperado: os três módulos com `VALIDADO (bit a bit)` e código de saída 0.

A documentação completa (método, fórmulas, origem dos pesos, resultados,
limitações e execução em ARM) está em **RELATORIO.txt**.
