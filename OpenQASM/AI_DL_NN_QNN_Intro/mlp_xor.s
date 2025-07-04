; File: mlp_xor.s
; Build: nasm -f elf64 mlp_xor.s -o mlp_xor.o && gcc -no-pie -o mlp_xor.elf64 mlp_xor.o -lm

; === EXTERNAL FUNCTIONS ===
extern printf
extern srand
extern time
extern rand
extern exp

section .data
fmt_error db "Epoch %d - Error: %lf", 10, 0
fmt_result db "Input: %.0lf %.0lf -> Output: %.4lf", 10, 0

; XOR Inputs and Targets
dq_inputs:   dq 0.0, 0.0,   0.0, 1.0,   1.0, 0.0,   1.0, 1.0
dq_targets:  dq 0.0, 1.0, 1.0, 0.0

section .bss
nn resq 30

section .text
global main

main:
    sub rsp, 32
    mov rdi, 0
    call time
    mov rdi, rax
    call srand

    call initialize_network

    mov rdi, nn
    lea rsi, [rel dq_inputs]
    lea rdx, [rel dq_targets]
    mov rcx, 4
    call train

    mov rbx, 0
.predict_loop:
    cmp rbx, 4
    jge .done

    mov rdi, nn
    mov rax, rbx
    imul rax, 16
    lea rsi, [dq_inputs + rax]
    movsd xmm0, [rsi]
    movsd xmm1, [rsi+8]
    call predict

    movsd xmm1, [rsi]
    movsd xmm2, [rsi+8]
    movsd xmm3, xmm0
    lea rdi, [rel fmt_result]
    xor eax, eax
    call printf

    inc rbx
    jmp .predict_loop

.done:
    add rsp, 32
    xor eax, eax
    ret

sigmoid:
    movsd xmm1, xmm0
    xorpd xmm2, xmm2
    subsd xmm2, xmm1
    movsd xmm0, xmm2
    call exp
    movsd xmm1, xmm0
    movsd xmm0, qword [one]
    addsd xmm0, xmm1
    movsd xmm1, qword [one]
    divsd xmm1, xmm0
    movsd xmm0, xmm1
    ret

; sigmoidDerivative(x) = x * (1 - x)
sigmoid_derivative:
    movsd xmm1, qword [one]
    subsd xmm1, xmm0
    mulsd xmm0, xmm1
    ret

train:
    push rbp
    mov rbp, rsp
    sub rsp, 128

    xor r8, r8
.train_epoch:
    cmp r8, 100000
    jge .train_done
    xor r9, r9
    xorpd xmm7, xmm7        ; totalError = 0.0
.sample_loop:
    cmp r9, rcx
    jge .check_print

    mov rax, r9
    imul rax, 16
    movsd xmm0, [rsi + rax]
    movsd [rdi], xmm0
    movsd xmm1, [rsi + rax + 8]
    movsd [rdi + 8], xmm1

    ; forward pass, backprop already done...

    ; totalError += error * error
    movsd xmm0, [rbp-16]     ; error
    movapd xmm1, xmm0
    mulsd xmm1, xmm0
    addsd xmm7, xmm1

    inc r9
    jmp .sample_loop

.check_print:
    mov rax, r8
    mov rdx, 0
    mov rcx, 1000
    div rcx
    test rdx, rdx
    jne .next_epoch

    ; print error every 1000 epochs
    mov rsi, r8              ; epoch
    movq rdx, xmm7           ; error
    lea rdi, [rel fmt_error]
    xor eax, eax
    call printf

.next_epoch:
    inc r8
    jmp .train_epoch
.train_done:
    add rsp, 128
    pop rbp
    ret

section .rodata
one dq 1.0
half dq 0.5
