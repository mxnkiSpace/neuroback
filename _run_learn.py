"""
Wrapper para learn.py.

Python 3.14 cambió el start method por defecto de multiprocessing a 'forkserver'
en Linux. learn.py corre el entrenamiento a nivel de módulo (sin guard
`if __name__ == '__main__':`), por lo que forkserver falla al reimportar el
módulo en los workers del DataLoader. Forzamos 'fork' para preservar el
comportamiento previo a 3.14 sin tener que editar learn.py.
"""
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import os
import runpy
import sys

script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learn.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
