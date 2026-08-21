"""Orquestrador — roda as três etapas em sequência, parando na primeira falha.

Executa, nesta ordem, ``treino.py``, ``referencia_float.py`` e ``quantiza.py``.
Cada etapa só é iniciada se a anterior terminar com sucesso (código de saída 0).
Ao final, imprime um resumo do que passou e do que falhou.

Portável: usa ``sys.executable`` (o Python do ambiente virtual ativo) e
``pathlib`` para os caminhos, funcionando igual no Windows e no Linux.

Uso (com o .venv ativo, a partir da raiz do projeto):
    python python/executar_tudo.py
"""

import subprocess
import sys
import time
from pathlib import Path

# Pasta onde este arquivo e os scripts das etapas estão (python/).
DIR_PYTHON = Path(__file__).resolve().parent

# As etapas, na ordem de execução. Cada par é (rótulo, nome do arquivo).
ETAPAS = [
    ("Etapa 1 - Treinar", "treino.py"),
    ("Etapa 2 - Referencia float", "referencia_float.py"),
    ("Etapa 3 - Quantizar int8", "quantiza.py"),
]


def rodar_etapa(rotulo, arquivo):
    """Executa um script de etapa e devolve se ele teve sucesso.

    O script é rodado com o MESMO interpretador Python que roda este
    orquestrador (``sys.executable``), garantindo que o ambiente virtual ativo
    seja usado tanto no Windows quanto no Linux. A saída do script aparece ao
    vivo no terminal, pois não é capturada. O sucesso é definido pelo código de
    saída: 0 significa que o script terminou sem erro.

    Antes de executar, verifica que o arquivo existe, dando uma mensagem clara
    caso contrário (em vez do erro genérico do interpretador).

    Parâmetros
    ----------
    rotulo : str
        Nome legível da etapa, usado nas mensagens.
    arquivo : str
        Nome do arquivo do script, dentro da pasta ``python/``.

    Retorna
    -------
    bool
        ``True`` se o script terminou com código 0; ``False`` caso contrário.
    """
    caminho = DIR_PYTHON / arquivo

    print("\n" + "=" * 70)
    print(f">>> {rotulo}  ({arquivo})")
    print("=" * 70)

    if not caminho.is_file():
        print(f"ERRO: arquivo nao encontrado: {caminho}")
        return False

    inicio = time.time()
    resultado = subprocess.run([sys.executable, str(caminho)])
    duracao = time.time() - inicio

    ok = resultado.returncode == 0
    estado = "OK" if ok else f"FALHOU (codigo {resultado.returncode})"
    print(f"\n--- {rotulo}: {estado} em {duracao:.1f}s ---")
    return ok


def main():
    """Roda as etapas em sequência, interrompendo na primeira falha.

    Retorna
    -------
    int
        Código de saída do processo: 0 se todas passaram, 1 se alguma falhou.
    """
    print("Iniciando a cadeia completa (treino -> referencia -> quantizacao).")

    inicio_total = time.time()
    concluidas = []
    for rotulo, arquivo in ETAPAS:
        if rodar_etapa(rotulo, arquivo):
            concluidas.append(rotulo)
            continue
        # Falhou: não faz sentido seguir para uma etapa que depende desta.
        print("\n" + "#" * 70)
        print(f"# INTERROMPIDO em '{rotulo}'.")
        print("# As etapas seguintes NAO foram executadas, pois dependem desta.")
        print("#" * 70)
        _resumo(concluidas, falhou=rotulo, duracao_total=time.time() - inicio_total)
        return 1

    print("\n" + "#" * 70)
    print("# CADEIA COMPLETA: todas as etapas passaram.")
    print("#" * 70)
    _resumo(concluidas, falhou=None, duracao_total=time.time() - inicio_total)
    return 0


def _resumo(concluidas, falhou, duracao_total):
    """Imprime o resumo final de etapas concluídas e da que falhou.

    Parâmetros
    ----------
    concluidas : list[str]
        Rótulos das etapas que terminaram com sucesso.
    falhou : str or None
        Rótulo da etapa que falhou, ou ``None`` se todas passaram.
    duracao_total : float
        Tempo total decorrido, em segundos.
    """
    print("\nResumo:")
    for rotulo, _ in ETAPAS:
        if rotulo in concluidas:
            print(f"  [OK]     {rotulo}")
        elif rotulo == falhou:
            print(f"  [FALHOU] {rotulo}")
        else:
            print(f"  [-]      {rotulo}  (nao executada)")
    print(f"\nTempo total: {duracao_total:.1f}s")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # Ctrl+C: encerra de forma limpa, sem despejar o traceback.
        print("\nInterrompido pelo usuario.")
        sys.exit(130)