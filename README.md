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

Si se ejecuta accidentalmente con `/usr/bin/python3`, `Main_Jorge_Physics_RGB.py`
intenta re-lanzarse automaticamente con `~/mlenv/bin/python` o
`~/miniconda3/bin/python3` siempre que alguno tenga las dependencias requeridas.

Tuning de hiperparametros RGB:

```bash
python hyperparameter_tuning_jorge_physics_rgb.py
```

Cada experimento usa `epochs=1000`, `1001`, `1002`, ... para separar carpetas.
El resumen acumulado queda en `outputs/hyperparameter_tuning_rgb/`.
