; stats_scalar.asm
; Version ESCALAR (referencia) de los kernels de computo.
;
; Convencion de llamada: System V AMD64 ABI
;   enteros/punteros: rdi, rsi, rdx, rcx, r8, r9
;   flotantes:        xmm0, xmm1, xmm2, ...
;   retorno float:    xmm0
;   callee-saved:     rbx, rbp, r12-r15 (si los usa, debe preservarlos)
; =============================================================

    global sum_array
    global compute_stats
    global normalize_array

    section .text

; ---------------------------------------------------------------
; float sum_array(const float *arr, int n)
;   rdi = arr, esi = n
;   retorna la suma en xmm0
; ---------------------------------------------------------------
sum_array:
    xor     eax, eax           ; eax = i = 0
    xorps   xmm0, xmm0         ; xmm0 = acumulador = 0.0

.sum_loop:
    cmp     eax, esi
    jge     .sum_done
    movss   xmm1, [rdi + rax*4]
    addss   xmm0, xmm1
    inc     eax
    jmp     .sum_loop

.sum_done:
    ret

; ---------------------------------------------------------------
; void compute_stats(const float *arr, int n,
;                     float *mean, float *var, float *min, float *max)
;   rdi = arr, esi = n, rdx = mean*, rcx = var*, r8 = min*, r9 = max*
;
;   var = varianza POBLACIONAL = sum((x - mean)^2) / n
;   Caso borde: si n == 0, escribe 0.0 en mean/var/min/max.
;
; Estrategia: dos pasadas por el arreglo.
;   Pasada 1: suma -> media.
;   Pasada 2: usando la media, acumula varianza y lleva min/max.
; ---------------------------------------------------------------
compute_stats:
    test    esi, esi
    jnz     .compute_nonzero

    ; --- caso borde: n == 0 ---
    xorps   xmm0, xmm0
    movss   [rdx], xmm0
    movss   [rcx], xmm0
    movss   [r8], xmm0
    movss   [r9], xmm0
    ret

.compute_nonzero:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    mov     r12, rdi        ; r12 = arr
    mov     r13d, esi       ; r13d = n
    mov     r14, rdx        ; r14 = mean*
    mov     r15, rcx        ; r15 = var*
    mov     rbx, r8         ; rbx = min*
    push    r9              ; guardamos max* en la pila (nos quedamos sin registros libres)

    ; ---- Pasada 1: suma y media ----
    xor     eax, eax
    xorps   xmm0, xmm0
.compute_sumloop:
    cmp     eax, r13d
    jge     .compute_sum_done
    movss   xmm1, [r12 + rax*4]
    addss   xmm0, xmm1
    inc     eax
    jmp     .compute_sumloop
.compute_sum_done:
    cvtsi2ss xmm1, r13d
    divss   xmm0, xmm1          ; xmm0 = mean
    movss   [r14], xmm0          ; *mean_ptr = mean
    movss   xmm7, xmm0           ; guardamos la media para la pasada 2

    ; ---- Pasada 2: varianza, min, max ----
    movss   xmm3, [r12]          ; min = arr[0]
    movss   xmm4, [r12]          ; max = arr[0]
    xorps   xmm5, xmm5            ; acumulador de varianza = 0.0

    xor     eax, eax
.compute_loop2:
    cmp     eax, r13d
    jge     .compute_loop2_done
    movss   xmm1, [r12 + rax*4]  ; xmm1 = arr[i]

    movss   xmm6, xmm1
    subss   xmm6, xmm7            ; xmm6 = x - mean
    mulss   xmm6, xmm6            ; xmm6 = (x - mean)^2
    addss   xmm5, xmm6            ; acumulamos

    comiss  xmm3, xmm1
    jbe     .compute_skip_min
    movss   xmm3, xmm1
.compute_skip_min:
    comiss  xmm4, xmm1
    jae     .compute_skip_max
    movss   xmm4, xmm1
.compute_skip_max:

    inc     eax
    jmp     .compute_loop2

.compute_loop2_done:
    cvtsi2ss xmm1, r13d
    divss   xmm5, xmm1            ; var = suma_cuadrados / n
    movss   [r15], xmm5            ; *var_ptr = var
    movss   [rbx], xmm3             ; *min_ptr = min

    pop     rax                     ; recuperamos max_ptr de la pila
    movss   [rax], xmm4              ; *max_ptr = max

    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; ---------------------------------------------------------------
; void normalize_array(const float *in, float *out, int n,
;                       float mean, float stddev)
;   rdi = in, rsi = out, edx = n, xmm0 = mean, xmm1 = stddev
;
;   out[i] = (in[i] - mean) / stddev
;   Caso borde: si stddev == 0.0, copia in[i] en out[i] tal cual.
; ---------------------------------------------------------------
normalize_array:
    movaps  xmm8, xmm0       ; guardamos mean en un registro libre
    movaps  xmm9, xmm1       ; guardamos stddev en un registro libre

    xorps   xmm2, xmm2
    comiss  xmm9, xmm2
    je      .normalize_copy   ; si stddev == 0.0, ir a copiar tal cual

    xor     eax, eax
.normalize_loop:
    cmp     eax, edx
    jge     .normalize_done
    movss   xmm3, [rdi + rax*4]
    subss   xmm3, xmm8
    divss   xmm3, xmm9
    movss   [rsi + rax*4], xmm3
    inc     eax
    jmp     .normalize_loop

.normalize_copy:
    xor     eax, eax
.normalize_copy_loop:
    cmp     eax, edx
    jge     .normalize_done
    mov     ecx, [rdi + rax*4]
    mov     [rsi + rax*4], ecx
    inc     eax
    jmp     .normalize_copy_loop

.normalize_done:
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
