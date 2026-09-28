"""
Verifica el resumen de estadisticos (<salida>.stats.txt, generado por
driver.c) contra una referencia calculada en Python puro, leyendo el
mismo archivo de entrada binario.

Uso:
    python3 verify_reference.py <input.dat> <output.dat.stats.txt> [tol]

    tol   tolerancia relativa para comparar floats (por defecto 1e-3;
          se usa una tolerancia mas laxa para 'var' porque el orden
          de suma vectorial/escalar difiere y el redondeo se acumula)
"""
import struct
import sys


def read_input(path):
    with open(path, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        arr = list(struct.unpack(f"<{n}f", f.read(4 * n))) if n > 0 else []
    return n, arr


def read_summary(path):
    stats = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            stats[key] = float(val) if key != "n" else int(val)
    return stats


def ref_stats(arr, n):
    if n == 0:
        return dict(sum=0.0, mean=0.0, var=0.0, stddev=0.0, min=0.0, max=0.0)
    total = sum(arr)
    mean = total / n
    var = sum((x - mean) ** 2 for x in arr) / n
    stddev = var ** 0.5
    return dict(sum=total, mean=mean, var=var, stddev=stddev,
                min=min(arr), max=max(arr))


def close(a, b, tol):
    return abs(a - b) <= tol * (1.0 + abs(a))


def main():
    if len(sys.argv) < 3:
        print(f"Uso: {sys.argv[0]} <input.dat> <output.dat.stats.txt> [tol]")
        sys.exit(1)

    input_path = sys.argv[1]
    summary_path = sys.argv[2]
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    var_tol = max(tol, 5e-3)  # la varianza acumula mas error de redondeo

    n_file, arr = read_input(input_path)
    got = read_summary(summary_path)
    ref = ref_stats(arr, n_file)

    if got.get("n") != n_file:
        print(f"FALLO: n en el resumen ({got.get('n')}) != n en el input ({n_file})")
        sys.exit(1)

    checks = [
        ("sum", tol), ("mean", tol), ("var", var_tol),
        ("stddev", var_tol), ("min", 1e-6), ("max", 1e-6),
    ]

    all_ok = True
    print(f"N = {n_file}")
    for key, t in checks:
        g = got.get(key)
        r = ref[key]
        ok = close(g, r, t)
        all_ok &= ok
        estado = "OK" if ok else "FALLO"
        print(f"  {key:8s} obtenido={g!r:>18} referencia={r!r:>18}  [{estado}]")

    print("\nRESULTADO:", "TODO CORRECTO" if all_ok else "HAY DIFERENCIAS")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
