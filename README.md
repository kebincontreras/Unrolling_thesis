# Unrolling_thesis

Repositorio limpio de la metodologia de unrolling guiado por fisica para precompensacion de miopia.

## Estructura principal

- `Main_Jorge_Physics.py`: entrenamiento en un solo canal.
- `Main_Jorge_Physics_RGB.py`: entrenamiento en color con tres PSF, una por canal.
- `project/`: modulos de datos, modelo, fisica, entrenamiento y metricas.
- `Image/`: imagenes de entrenamiento y validacion.
- `metodologia_unrolling_miopia.tex`: documento de metodologia.

## Ejecucion

Canal rojo:

```bash
source ~/mlenv/bin/activate && python Main_Jorge_Physics.py
```

RGB con tres PSF:

```bash
source ~/mlenv/bin/activate && python Main_Jorge_Physics_RGB.py
```
