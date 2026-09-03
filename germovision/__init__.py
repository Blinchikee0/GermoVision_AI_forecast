"""GermoVision 2.0 — система раннего предупреждения о клинически значимых
мутациях патогенов.

Полное описание проекта: GERMOVISION_2.0_MASTER.md
Разбивка на части: ARCHITECTURE_PARTS.md

Реализовано: Часть 1 (ядро).
"""

__version__ = "0.1.0"

from . import core

__all__ = ["core", "__version__"]
