# Normalizador estadistico vectorizado (NASM + C)

Proyecto de curso: comparacion de una version escalar (SSE) contra una
version vectorizada (AVX2) de tres kernels de computo -- sum_array,
compute_stats y normalize_array -- implementados en NASM x86-64 y
ejercitados desde un mismo driver en C, respetando la convencion de llamada
System V AMD64 ABI.

## Estructura

    .
    |-- Makefile
    |-- README.md
    |-- .gitignore
    |-- include/
    |   `-- stats.h                  (firmas y contrato ABI compartido)
    |-- main/
    |   `-- driver.c                 (lee input, mide tiempo, llama a los kernels, escribe salida)
    |-- escalar/
    |   `-- stats_scalar.asm         (version escalar (SSE), completa)
    |-- vectorial/
    |   `-- stats_vector.asm         (version vectorial (AVX2), completa)
    |-- datos/
    |   |-- gen_input.py             (genera archivos .dat de prueba, incluye casos borde)
    |   `-- verify_reference.py      (verifica resultados contra NumPy/Python puro)
    |-- gdb_session_ejemplo.txt      (sesion de GDB capturada, inspeccion de registros YMM)
    `-- data/                        (se crea al generar/ejecutar, ignorado por git)

## Compilar

    make

Genera bin/norm_scalar y bin/norm_vector.

## Generar datos de prueba (incluye los casos borde)

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

## Ejecutar y verificar

    ./bin/norm_scalar data/input_small.dat data/out_small.dat
    python3 datos/verify_reference.py data/input_small.dat data/out_small.dat.stats.txt

    ./bin/norm_vector data/input_small.dat data/out_small_v.dat 5
    python3 datos/verify_reference.py data/input_small.dat data/out_small_v.dat.stats.txt

verify_reference.py recalcula sum/mean/var/stddev/min/max en Python puro a
partir del mismo .dat de entrada y los compara contra el resumen
*.stats.txt que escribe driver.c, con tolerancia relativa (mas laxa para
var/stddev, ya que el orden de suma vectorial cambia el redondeo de punto
flotante frente al escalar).

Verificado contra input_small.dat (N=8): TODO CORRECTO en ambas versiones.
Casos borde probados en la version escalar: N=0, N=1 y N=1000 con valores
constantes (sigma=0), todos correctos.

## Sesion de GDB (inspeccion de registros YMM)

El archivo gdb_session_ejemplo.txt trae una corrida capturada. Para
reproducirla:

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
    (gdb) x/8fw $rsi                   # arreglo normalizado final

Esto muestra como un escalar (xmm0/xmm1) se convierte, via vbroadcastss, en
un vector de 8 carriles identicos (ymm2/ymm3) antes de aplicarse a los 8
floats de una sola pasada del bucle vectorial.

## Notas de implementacion

### Version escalar (escalar/stats_scalar.asm)

- sum_array recorre el arreglo un elemento a la vez, acumulando en xmm0
  con movss/addss.
- compute_stats hace dos pasadas: la primera calcula la suma y la media;
  la segunda, usando la media ya calculada, acumula sum((x-mean)^2) para
  la varianza y lleva el minimo/maximo con comiss y saltos condicionales.
  Caso borde N=0: escribe 0.0 en mean/var/min/max sin dividir.
- normalize_array aplica (x-mean)/stddev elemento a elemento; si
  stddev == 0.0, copia el arreglo de entrada tal cual (copia de bits, sin
  operar) para evitar la division por cero.

### Version vectorial (vectorial/stats_vector.asm)

- sum_array procesa 8 floats por iteracion con AVX2 (vmovups/vaddps),
  reduce el acumulador de 8 carriles a un escalar (vextractf128 + vaddps +
  vhaddps), y cierra con un bucle escalar para el remanente n % 8.
- compute_stats hace una sola pasada vectorial de 8 en 8 acumulando
  sum((x-mean)^2), min y max simultaneamente con vsubps/vmulps/vaddps y
  vminps/vmaxps, mas reduccion horizontal (vextractf128 + vhaddps para la
  varianza; vextractf128 + shuffles para min/max) y un cierre escalar para
  el remanente.
- normalize_array vectoriza (x-mean)/stddev de 8 en 8; si stddev == 0.0
  copia el arreglo tal cual (bit a bit) para evitar la division por cero.
- Usa instrucciones VEX escalares (vaddss, vsubss, etc.) en vez de las
  legacy SSE (addss, subss) para no mezclar codificaciones y evitar la
  penalizacion de transicion AVX/SSE.
