; stats_vector.asm
; Version VECTORIZADA (AVX2, 8 floats por iteracion) de los
; kernels de computo. Misma ABI que la version escalar.
;
; Antes de compilar/ejecutar en su maquina, confirme soporte AVX2:
;   lscpu | grep avx2
;   cat /proc/cpuinfo | grep avx2
; =============================================================

    global sum_array
    global compute_stats
    global normalize_array

    section .text

; ---------------------------------------------------------------
; float sum_array(const float *arr, int n)
;   rdi = arr, esi = n -> retorna la suma en xmm0
;
; IMPLEMENTADA COMO EJEMPLO. Fijense especialmente en:
;   (1) como se calcula cuantos elementos entran en bucles de 8
;       ("and ecx, ~7" redondea n hacia abajo al multiplo de 8),
;   (2) la REDUCCION HORIZONTAL para pasar de 8 sumas parciales
;       (un YMM) a un unico escalar,
;   (3) el BUCLE ESCALAR DE CIERRE para el remanente (n % 8 != 0).
; Reutilicen este mismo patron en compute_stats y normalize_array.
; ---------------------------------------------------------------
sum_array:
    xor     eax, eax               ; eax = i = 0
    vxorps  ymm0, ymm0, ymm0       ; ymm0 = acumulador vectorial (8 carriles) = 0

    mov     ecx, esi
    and     ecx, ~7                ; ecx = n redondeado hacia abajo, multiplo de 8
    test    ecx, ecx
    jle     .sum_reduce

.sum_vec_loop:
    cmp     eax, ecx
    jge     .sum_reduce
    vmovups ymm1, [rdi + rax*4]    ; carga 8 floats (unaligned: siempre valido)
    vaddps  ymm0, ymm0, ymm1       ; acumula por carril
    add     eax, 8
    jmp     .sum_vec_loop

.sum_reduce:
    ; --- reduccion horizontal: 8 carriles de ymm0 -> un escalar ---
    vextractf128 xmm2, ymm0, 1     ; xmm2 = mitad alta (carriles 4-7)
    vaddps  xmm0, xmm0, xmm2       ; xmm0 = 4 sumas parciales (carriles 0-3 + 4-7)
    vhaddps xmm0, xmm0, xmm0       ; suma horizontal dentro de 128 bits
    vhaddps xmm0, xmm0, xmm0       ; xmm0[0] = suma total de los 8 carriles originales

.sum_scalar_tail:
    ; --- elementos sobrantes (n % 8), uno a la vez ---
    cmp     eax, esi
    jge     .sum_done
    vmovss  xmm1, [rdi + rax*4]
    vaddss  xmm0, xmm0, xmm1
    inc     eax
    jmp     .sum_scalar_tail

.sum_done:
    vzeroupper                     ; evita penalizacion de transicion AVX/SSE
    ret

; ---------------------------------------------------------------
; void compute_stats(const float *arr, int n,
;                     float *mean, float *var, float *min, float *max)
;   rdi = arr, esi = n, rdx = mean*, rcx = var*, r8 = min*, r9 = max*
;
; Estrategia:
;   1) mean = sum_array(arr, n) / n. Se guardan arr/n/los 4
;      punteros en registros callee-saved ANTES de llamar a
;      sum_array, porque la llamada destruye los caller-saved.
;   2) Segunda pasada vectorizada (8 floats/iter) que acumula
;      simultaneamente:
;        - sum((x-mean)^2)  -> para la varianza
;        - min vectorial (vminps) y max vectorial (vmaxps)
;      usando "broadcast" de mean a los 8 carriles.
;   3) Reduccion horizontal de los 3 acumuladores vectoriales
;      (mismo patron que sum_array para la suma; para min/max se
;      usan vminps/vmaxps + shuffles hasta quedar en 1 elemento).
;   4) Bucle escalar de cierre para el remanente (n % 8), tambien
;      alimentando var/min/max.
;   5) Caso borde n == 0: se escribe 0.0 en mean/var/min/max.
; ---------------------------------------------------------------
compute_stats:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    push    rbp
    sub     rsp, 8              ; alinea rsp a 16 antes de 'call sum_array'

    mov     rbx, rdi            ; rbx = arr
    mov     r12d, esi           ; r12d = n
    mov     r13, rdx            ; r13 = mean*
    mov     r14, rcx            ; r14 = var*
    mov     r15, r8             ; r15 = min*
    mov     rbp, r9             ; rbp = max*

    test    r12d, r12d
    jne     .cs_nonzero

    ; --- caso borde: n == 0 ---
    vxorps  xmm0, xmm0, xmm0
    vmovss  [r13], xmm0
    vmovss  [r14], xmm0
    vmovss  [r15], xmm0
    vmovss  [rbp], xmm0
    jmp     .cs_done

.cs_nonzero:
    ; --- mean = sum_array(arr, n) / n ---
    mov     rdi, rbx
    mov     esi, r12d
    call    sum_array           ; xmm0 = suma total

    vxorps  xmm7, xmm7, xmm7
    vcvtsi2ss xmm7, xmm7, r12d  ; xmm7 = (float) n
    vdivss  xmm0, xmm0, xmm7    ; xmm0 = mean
    vmovss  [r13], xmm0         ; guarda mean en *mean

    vbroadcastss ymm2, xmm0     ; ymm2 = mean en los 8 carriles

    ; --- inicializa acumuladores para la segunda pasada ---
    xor     eax, eax                ; eax = i = 0
    vxorps  ymm3, ymm3, ymm3        ; ymm3 = acumulador de (x-mean)^2
    vbroadcastss ymm4, [rbx]        ; ymm4 = min vectorial, init = arr[0]
    vbroadcastss ymm5, [rbx]        ; ymm5 = max vectorial, init = arr[0]

    mov     ecx, r12d
    and     ecx, ~7                 ; ecx = n redondeado a multiplo de 8
    test    ecx, ecx
    jle     .cs_scalar_tail

.cs_vec_loop:
    cmp     eax, ecx
    jge     .cs_reduce
    vmovups ymm1, [rbx + rax*4]     ; carga 8 floats
    vsubps  ymm6, ymm1, ymm2        ; diff = x - mean
    vmulps  ymm6, ymm6, ymm6        ; diff^2
    vaddps  ymm3, ymm3, ymm6        ; acumula varianza
    vminps  ymm4, ymm4, ymm1        ; min vectorial
    vmaxps  ymm5, ymm5, ymm1        ; max vectorial
    add     eax, 8
    jmp     .cs_vec_loop

.cs_reduce:
    ; --- reduccion horizontal de la varianza (ymm3 -> escalar) ---
    vextractf128 xmm7, ymm3, 1
    vaddps  xmm3, xmm3, xmm7
    vhaddps xmm3, xmm3, xmm3
    vhaddps xmm3, xmm3, xmm3        ; xmm3[0] = suma parcial (vectorial) de (x-mean)^2

    ; --- reduccion horizontal del min (ymm4 -> escalar) ---
    vextractf128 xmm7, ymm4, 1
    vminps  xmm4, xmm4, xmm7        ; 4 candidatos
    vshufps xmm7, xmm4, xmm4, 0xEE  ; duplica la mitad alta
    vminps  xmm4, xmm4, xmm7        ; 2 candidatos
    vshufps xmm7, xmm4, xmm4, 0x55  ; duplica el carril 1
    vminps  xmm4, xmm4, xmm7        ; xmm4[0] = min vectorial final

    ; --- reduccion horizontal del max (ymm5 -> escalar) ---
    vextractf128 xmm7, ymm5, 1
    vmaxps  xmm5, xmm5, xmm7
    vshufps xmm7, xmm5, xmm5, 0xEE
    vmaxps  xmm5, xmm5, xmm7
    vshufps xmm7, xmm5, xmm5, 0x55
    vmaxps  xmm5, xmm5, xmm7        ; xmm5[0] = max vectorial final

.cs_scalar_tail:
    ; --- elementos sobrantes (n % 8), uno a la vez ---
    ; (si n < 8, ecx == 0 y llegamos aqui directo desde arriba;
    ;  ymm3 sigue en 0 y ymm4/ymm5 siguen inicializados con arr[0],
    ;  asi que el bucle desde eax=0 procesa TODOS los elementos)
    cmp     eax, r12d
    jge     .cs_finish
    vmovss  xmm1, [rbx + rax*4]
    vsubss  xmm6, xmm1, xmm0        ; diff = x - mean (xmm0 = mean)
    vmulss  xmm6, xmm6, xmm6
    vaddss  xmm3, xmm3, xmm6
    vminss  xmm4, xmm4, xmm1
    vmaxss  xmm5, xmm5, xmm1
    inc     eax
    jmp     .cs_scalar_tail

.cs_finish:
    ; --- var = suma((x-mean)^2) / n ---
    vxorps  xmm7, xmm7, xmm7
    vcvtsi2ss xmm7, xmm7, r12d
    vdivss  xmm3, xmm3, xmm7
    vmovss  [r14], xmm3             ; guarda var
    vmovss  [r15], xmm4             ; guarda min
    vmovss  [rbp], xmm5             ; guarda max

.cs_done:
    add     rsp, 8
    pop     rbp
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    vzeroupper
    ret

; ---------------------------------------------------------------
; void normalize_array(const float *in, float *out, int n,
;                       float mean, float stddev)
;   rdi = in, rsi = out, edx = n, xmm0 = mean, xmm1 = stddev
;
;   out[i] = (in[i] - mean) / stddev
;   Caso borde: si stddev == 0.0, copie in[i] en out[i] tal cual.
;
; Estrategia: broadcast de mean/stddev a los 8 carriles, bucle
; vectorial de 8 en 8 (resta + division), bucle escalar de cierre
; para el remanente, y un camino separado de copia bit-exacta
; cuando stddev == 0.0.
; ---------------------------------------------------------------
normalize_array:
    vxorps   xmm2, xmm2, xmm2
    vucomiss xmm1, xmm2
    je       .na_copy               ; stddev == 0.0 -> copiar tal cual

    vbroadcastss ymm2, xmm0         ; ymm2 = mean en los 8 carriles
    vbroadcastss ymm3, xmm1         ; ymm3 = stddev en los 8 carriles

    xor     eax, eax                ; eax = i = 0
    mov     ecx, edx
    and     ecx, ~7                 ; ecx = n redondeado a multiplo de 8
    test    ecx, ecx
    jle     .na_scalar_tail

.na_vec_loop:
    cmp     eax, ecx
    jge     .na_scalar_tail
    vmovups ymm4, [rdi + rax*4]     ; carga 8 floats de entrada
    vsubps  ymm4, ymm4, ymm2        ; x - mean
    vdivps  ymm4, ymm4, ymm3        ; (x - mean) / stddev
    vmovups [rsi + rax*4], ymm4     ; guarda 8 floats de salida
    add     eax, 8
    jmp     .na_vec_loop

.na_scalar_tail:
    ; --- elementos sobrantes (n % 8), uno a la vez ---
    cmp     eax, edx
    jge     .na_done
    vmovss  xmm4, [rdi + rax*4]
    vsubss  xmm4, xmm4, xmm0        ; xmm0 = mean
    vdivss  xmm4, xmm4, xmm1        ; xmm1 = stddev
    vmovss  [rsi + rax*4], xmm4
    inc     eax
    jmp     .na_scalar_tail

.na_done:
    vzeroupper
    ret

.na_copy:
    ; --- stddev == 0.0: copia bit-exacta in -> out (evita div/0) ---
    xor     eax, eax
.na_copy_loop:
    cmp     eax, edx
    jge     .na_copy_done
    mov     ecx, [rdi + rax*4]      ; copia los 4 bytes tal cual (como entero)
    mov     [rsi + rax*4], ecx
    inc     eax
    jmp     .na_copy_loop
.na_copy_done:
    ret
