# Normalizador estadístico vectorizado (NASM + C)

Proyecto de curso: comparación escalar (SSE) vs vectorial (AVX2) de tres
kernels — `sum_array`, `compute_stats`, `normalize_array` — implementados
en NASM x86-64 y ejercitados desde un driver en C.

Mi parte (Gustavo) es **`vectorial/stats_vector.asm`** (versión AVX2),
ya implementada y probada. `escalar/stats_scalar.asm` se deja tal cual
el esqueleto del profesor (con los `TODO`) porque le corresponde a mi
compañero de equipo.

## Estructura

```
.
├── Makefile
├── include/stats.h
├── main/driver.c
├── escalar/stats_scalar.asm   # TODO (parte del companero)
├── vectorial/stats_vector.asm   # completo (mi parte, AVX2)
├── datos/gen_input.py
├── datos/verify_reference.py
├── gdb_session_ejemplo.txt       # sesion de GDB ya capturada (ver mas abajo)
└── data/                         # se crea al generar/ejecutar
```

## Compilar

```bash
make
```

Genera `bin/norm_scalar` y `bin/norm_vector`.

## Generar datos de prueba (incluye los casos borde)

```bash
mkdir -p data
python3 datos/gen_input.py 1000000 data/input.dat random 1
python3 datos/gen_input.py 8       data/input_small.dat    random 2
python3 datos/gen_input.py 1000    data/input_constant.dat constant
python3 datos/gen_input.py 0       data/input_empty.dat    random
python3 datos/gen_input.py 1       data/input_one.dat      random 3
python3 datos/gen_input.py 7       data/input_7.dat        random 4
python3 datos/gen_input.py 15      data/input_15.dat       random 5
python3 datos/gen_input.py 1001    data/input_1001.dat     random 6
python3 datos/gen_input.py 200     data/input_edge.dat     edge
```

## Ejecutar y verificar

```bash
./bin/norm_vector data/input_small.dat data/out_small.dat 5
python3 datos/verify_reference.py data/input_small.dat data/out_small.dat.stats.txt
```

`verify_reference.py` recalcula sum/mean/var/stddev/min/max en Python
puro a partir del mismo `.dat` de entrada y los compara contra el
resumen `*.stats.txt` que escribe `driver.c`, con tolerancia relativa
(más laxa para `var`/`stddev` porque el orden de suma vectorial cambia
el redondeo de punto flotante frente al escalar).

Ya lo corrí contra los 9 casos de arriba (incluyendo N=0, N=1, N<8,
N no múltiplo de 8, valores constantes y valores extremos) y todos
dieron `TODO CORRECTO`.

## Sesión de GDB (inspección de registros YMM)

El archivo `gdb_session_ejemplo.txt` ya trae una corrida capturada.
Para reproducirla:

```bash
gdb --args ./bin/norm_vector data/input_small.dat data/out_small.dat 1
(gdb) break normalize_array
(gdb) run
(gdb) print $xmm0.v4_float[0]     # mean, escalar, al entrar a la funcion
(gdb) print $xmm1.v4_float[0]     # stddev, escalar
(gdb) stepi                        # repetir hasta pasar las dos
(gdb) stepi                        # instrucciones vbroadcastss
(gdb) stepi
(gdb) stepi
(gdb) stepi
(gdb) print $ymm2.v8_float         # mean replicado en los 8 carriles
(gdb) print $ymm3.v8_float         # stddev replicado en los 8 carriles
(gdb) continue
(gdb) x/8fw $rsi                   # arreglo normalizado final (si aun no
                                    # ha retornado la funcion)
```

Esto es exactamente lo que pide la propuesta: ver con GDB cómo un
escalar (`xmm0`/`xmm1`) se convierte, vía `vbroadcastss`, en un vector
de 8 carriles idénticos (`ymm2`/`ymm3`) antes de aplicarse a los 8
floats de una sola pasada del bucle vectorial.

## Notas de implementación (stats_vector.asm)

- `compute_stats` hace una sola pasada de 8 en 8 acumulando
  `sum((x-mean)^2)` (para la varianza), `min` y `max` al mismo tiempo
  con `vsubps`/`vmulps`/`vaddps` y `vminps`/`vmaxps`, más reducción
  horizontal (patrón `vextractf128` + `vhaddps` para la varianza,
  `vextractf128` + shuffles para min/max) y un cierre escalar para el
  remanente `n % 8`.
- `normalize_array` vectoriza `(x-mean)/stddev` de 8 en 8; si
  `stddev == 0.0` copia el arreglo tal cual (bit a bit) para evitar la
  división por cero.
- Se usan instrucciones VEX escalares (`vaddss`, `vsubss`, etc.) en
  vez de las legacy SSE (`addss`, `subss`) para no mezclar
  codificaciones y evitar penalización de transición AVX/SSE.
