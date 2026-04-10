	.arch armv8-a
	.file	"dot_scalar.c"
	.text
	.align	2
	.p2align 5,,15
	.global	dot_product_int8
	.type	dot_product_int8, %function
dot_product_int8:
.LFB22:
	.cfi_startproc
	mov	x4, x0
	cmp	w2, 0
	ble	.L8
	sub	w0, w2, #1
	cmp	w0, 14
	bls	.L9
	movi	v31.4s, 0
	and	x0, x2, 4294967280
	mov	x3, 0
	.p2align 5,,15
.L4:
	ldr	q30, [x4, x3]
	ldr	q29, [x1, x3]
	add	x3, x3, 16
	smull	v28.8h, v30.8b, v29.8b
	smull2	v30.8h, v30.16b, v29.16b
	saddw	v31.4s, v31.4s, v28.4h
	saddw2	v31.4s, v31.4s, v28.8h
	saddw	v31.4s, v31.4s, v30.4h
	saddw2	v31.4s, v31.4s, v30.8h
	cmp	x3, x0
	bne	.L4
	addv	s31, v31.4s
	and	w3, w2, -16
	fmov	w0, s31
	cmp	w2, w3
	beq	.L1
.L3:
	sxtw	x3, w3
	.p2align 5,,15
.L7:
	ldrsb	w6, [x4, x3]
	ldrsb	w5, [x1, x3]
	add	x3, x3, 1
	madd	w0, w6, w5, w0
	cmp	w2, w3
	bgt	.L7
.L1:
	ret
	.p2align 2,,3
.L8:
	mov	w0, 0
	ret
.L9:
	mov	w3, 0
	mov	w0, 0
	b	.L3
	.cfi_endproc
.LFE22:
	.size	dot_product_int8, .-dot_product_int8
	.align	2
	.p2align 5,,15
	.global	dot_product_int32
	.type	dot_product_int32, %function
dot_product_int32:
.LFB23:
	.cfi_startproc
	mov	x6, x0
	cmp	w2, 0
	ble	.L16
	ubfiz	x2, x2, 2, 32
	mov	x3, 0
	mov	w0, 0
	.p2align 5,,15
.L15:
	ldr	w5, [x6, x3]
	ldr	w4, [x1, x3]
	add	x3, x3, 4
	madd	w0, w5, w4, w0
	cmp	x2, x3
	bne	.L15
	ret
	.p2align 2,,3
.L16:
	mov	w0, 0
	ret
	.cfi_endproc
.LFE23:
	.size	dot_product_int32, .-dot_product_int32
	.align	2
	.p2align 5,,15
	.global	dot_product_float32
	.type	dot_product_float32, %function
dot_product_float32:
.LFB24:
	.cfi_startproc
	cmp	w2, 0
	ble	.L21
	movi	v0.2s, #0
	ubfiz	x2, x2, 2, 32
	mov	x3, 0
	.p2align 5,,15
.L20:
	ldr	s31, [x0, x3]
	ldr	s30, [x1, x3]
	add	x3, x3, 4
	fmadd	s0, s31, s30, s0
	cmp	x2, x3
	bne	.L20
	ret
	.p2align 2,,3
.L21:
	movi	v0.2s, #0
	ret
	.cfi_endproc
.LFE24:
	.size	dot_product_float32, .-dot_product_float32
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC0:
	.string	"malloc"
	.align	3
.LC1:
	.string	"%d %d %f\n"
	.section	.text.startup,"ax",@progbits
	.align	2
	.p2align 5,,15
	.global	main
	.type	main, %function
main:
.LFB25:
	.cfi_startproc
	stp	x29, x30, [sp, -80]!
	.cfi_def_cfa_offset 80
	.cfi_offset 29, -80
	.cfi_offset 30, -72
	mov	x0, 16384
	mov	x29, sp
	stp	x21, x22, [sp, 32]
	stp	x23, x24, [sp, 48]
	stp	x25, x26, [sp, 64]
	.cfi_offset 21, -48
	.cfi_offset 22, -40
	.cfi_offset 23, -32
	.cfi_offset 24, -24
	.cfi_offset 25, -16
	.cfi_offset 26, -8
	bl	malloc
	mov	x26, x0
	mov	x0, 16384
	bl	malloc
	mov	x25, x0
	mov	x0, 65536
	bl	malloc
	mov	x24, x0
	mov	x0, 65536
	bl	malloc
	mov	x23, x0
	mov	x0, 65536
	bl	malloc
	mov	x22, x0
	mov	x0, 65536
	bl	malloc
	cmp	x26, 0
	ccmp	x25, 0, 4, ne
	beq	.L24
	cmp	x24, 0
	ccmp	x23, 0, 4, ne
	beq	.L24
	cmp	x22, 0
	mov	x21, x0
	ccmp	x0, 0, 4, ne
	beq	.L24
	stp	x19, x20, [sp, 16]
	.cfi_offset 20, -56
	.cfi_offset 19, -64
	mov	x19, 0
	mov	w0, 0
	bl	srand
	.p2align 5,,15
.L27:
	bl	rand
	sxtb	w20, w0
	strb	w20, [x26, x19]
	bl	rand
	sxtb	w0, w0
	scvtf	s30, w20
	strb	w0, [x25, x19]
	scvtf	s31, w0
	str	w20, [x24, x19, lsl 2]
	str	w0, [x23, x19, lsl 2]
	str	s30, [x22, x19, lsl 2]
	str	s31, [x21, x19, lsl 2]
	add	x19, x19, 1
	cmp	x19, 16384
	bne	.L27
	movi	v31.4s, 0
	mov	x0, 0
	.p2align 5,,15
.L28:
	ldr	q30, [x26, x0]
	ldr	q29, [x25, x0]
	add	x0, x0, 16
	smull	v28.8h, v30.8b, v29.8b
	smull2	v30.8h, v30.16b, v29.16b
	saddw	v31.4s, v31.4s, v28.4h
	saddw2	v31.4s, v31.4s, v28.8h
	saddw	v31.4s, v31.4s, v30.4h
	saddw2	v31.4s, v31.4s, v30.8h
	cmp	x0, 16384
	bne	.L28
	addv	s30, v31.4s
	mov	x0, 0
	movi	v31.4s, 0
	.p2align 5,,15
.L29:
	ldr	q28, [x24, x0]
	ldr	q29, [x23, x0]
	add	x0, x0, 16
	mla	v31.4s, v28.4s, v29.4s
	cmp	x0, 65536
	bne	.L29
	addv	s28, v31.4s
	mov	x0, 0
	movi	v0.2s, #0
	.p2align 5,,15
.L30:
	ldr	q29, [x22, x0]
	ldr	q31, [x21, x0]
	add	x0, x0, 16
	fmul	v31.4s, v31.4s, v29.4s
	fadd	s0, s0, s31
	dup	s29, v31.s[1]
	fadd	s29, s29, s0
	dup	s0, v31.s[2]
	dup	s31, v31.s[3]
	fadd	s0, s0, s29
	fadd	s0, s0, s31
	cmp	x0, 65536
	bne	.L30
	fcvt	d0, s0
	fmov	w2, s28
	fmov	w1, s30
	adrp	x0, .LC1
	add	x0, x0, :lo12:.LC1
	bl	printf
	mov	x0, x26
	bl	free
	mov	x0, x25
	bl	free
	mov	x0, x24
	bl	free
	mov	x0, x23
	bl	free
	mov	x0, x22
	bl	free
	mov	x0, x21
	bl	free
	ldp	x19, x20, [sp, 16]
	.cfi_restore 20
	.cfi_restore 19
	mov	w0, 0
.L23:
	ldp	x21, x22, [sp, 32]
	ldp	x23, x24, [sp, 48]
	ldp	x25, x26, [sp, 64]
	ldp	x29, x30, [sp], 80
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_def_cfa_offset 0
	ret
.L24:
	.cfi_restore_state
	adrp	x0, .LC0
	add	x0, x0, :lo12:.LC0
	bl	perror
	mov	w0, 1
	b	.L23
	.cfi_endproc
.LFE25:
	.size	main, .-main
	.ident	"GCC: (GNU) 15.1.0"
	.section	.note.GNU-stack,"",@progbits
