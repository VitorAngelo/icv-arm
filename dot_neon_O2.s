	.arch armv8-a
	.file	"dot_neon.c"
	.text
	.align	2
	.p2align 5,,15
	.global	dot_product_int8
	.type	dot_product_int8, %function
dot_product_int8:
.LFB3908:
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
.LFE3908:
	.size	dot_product_int8, .-dot_product_int8
	.align	2
	.p2align 5,,15
	.global	dot_product_int32
	.type	dot_product_int32, %function
dot_product_int32:
.LFB3909:
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
.LFE3909:
	.size	dot_product_int32, .-dot_product_int32
	.align	2
	.p2align 5,,15
	.global	dot_product_float32
	.type	dot_product_float32, %function
dot_product_float32:
.LFB3910:
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
.LFE3910:
	.size	dot_product_float32, .-dot_product_float32
	.align	2
	.p2align 5,,15
	.global	dot_product_int8_neon
	.type	dot_product_int8_neon, %function
dot_product_int8_neon:
.LFB3911:
	.cfi_startproc
	cmp	w2, 0
	ble	.L26
	movi	v30.4s, 0
	mov	x3, 0
	.p2align 5,,15
.L25:
	ldr	d1, [x0, x3]
	add	x5, x0, x3
	ldr	d28, [x1, x3]
	add	x4, x1, x3
	add	x3, x3, 16
	ldr	d27, [x5, 8]
	smull	v28.8h, v1.8b, v28.8b
	ldr	d0, [x4, 8]
	smull	v0.8h, v27.8b, v0.8b
	sadalp	v30.4s, v28.8h
	sadalp	v30.4s, v0.8h
	cmp	w2, w3
	bgt	.L25
	dup	d29, v30.d[1]
	add	v31.2s, v29.2s, v30.2s
	addp	v31.2s, v31.2s, v31.2s
	umov	x0, v31.d[0]
	ret
	.p2align 2,,3
.L26:
	movi	v31.2s, 0
	addp	v31.2s, v31.2s, v31.2s
	umov	x0, v31.d[0]
	ret
	.cfi_endproc
.LFE3911:
	.size	dot_product_int8_neon, .-dot_product_int8_neon
	.align	2
	.p2align 5,,15
	.global	dot_product_int32_neon
	.type	dot_product_int32_neon, %function
dot_product_int32_neon:
.LFB3912:
	.cfi_startproc
	cmp	w2, 0
	ble	.L31
	movi	v30.4s, 0
	sub	w3, w2, #1
	mov	x2, 0
	lsr	w3, w3, 2
	add	w3, w3, 1
	ubfiz	x3, x3, 4, 31
	.p2align 5,,15
.L30:
	ldr	q29, [x0, x2]
	ldr	q0, [x1, x2]
	add	x2, x2, 16
	mla	v30.4s, v29.4s, v0.4s
	cmp	x3, x2
	bne	.L30
	dup	d28, v30.d[1]
	add	v31.2s, v28.2s, v30.2s
	addp	v31.2s, v31.2s, v31.2s
	umov	x0, v31.d[0]
	ret
	.p2align 2,,3
.L31:
	movi	v31.2s, 0
	addp	v31.2s, v31.2s, v31.2s
	umov	x0, v31.d[0]
	ret
	.cfi_endproc
.LFE3912:
	.size	dot_product_int32_neon, .-dot_product_int32_neon
	.align	2
	.p2align 5,,15
	.global	dot_product_float32_neon
	.type	dot_product_float32_neon, %function
dot_product_float32_neon:
.LFB3913:
	.cfi_startproc
	cmp	w2, 0
	ble	.L36
	movi	v30.4s, 0
	sub	w3, w2, #1
	mov	x2, 0
	lsr	w3, w3, 2
	add	w3, w3, 1
	ubfiz	x3, x3, 4, 31
	.p2align 5,,15
.L35:
	ldr	q31, [x0, x2]
	ldr	q29, [x1, x2]
	add	x2, x2, 16
	fmul	v31.4s, v31.4s, v29.4s
	fadd	v30.4s, v30.4s, v31.4s
	cmp	x3, x2
	bne	.L35
	dup	d0, v30.d[1]
	fadd	v0.2s, v0.2s, v30.2s
	faddp	v0.2s, v0.2s, v0.2s
	ret
	.p2align 2,,3
.L36:
	movi	v0.2s, 0
	faddp	v0.2s, v0.2s, v0.2s
	ret
	.cfi_endproc
.LFE3913:
	.size	dot_product_float32_neon, .-dot_product_float32_neon
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC0:
	.string	"OK"
	.align	3
.LC1:
	.string	"DIFF"
	.align	3
.LC2:
	.string	"malloc"
	.align	3
.LC3:
	.string	"ok?"
	.align	3
.LC4:
	.string	"neon"
	.align	3
.LC5:
	.string	"escalar"
	.align	3
.LC6:
	.string	"funcao"
	.align	3
.LC7:
	.string	"%-20s %12s %12s %6s\n"
	.align	3
.LC8:
	.string	"int8"
	.align	3
.LC9:
	.string	"%-20s %12d %12d %6s\n"
	.align	3
.LC10:
	.string	"int32"
	.align	3
.LC11:
	.string	"ver"
	.align	3
.LC12:
	.string	"float32"
	.align	3
.LC13:
	.string	"%-20s %12.1f %12.1f %6s\n"
	.section	.text.startup,"ax",@progbits
	.align	2
	.p2align 5,,15
	.global	main
	.type	main, %function
main:
.LFB3914:
	.cfi_startproc
	stp	x29, x30, [sp, -144]!
	.cfi_def_cfa_offset 144
	.cfi_offset 29, -144
	.cfi_offset 30, -136
	mov	x0, 16384
	mov	x29, sp
	stp	x21, x22, [sp, 32]
	stp	x23, x24, [sp, 48]
	stp	x25, x26, [sp, 64]
	.cfi_offset 21, -112
	.cfi_offset 22, -104
	.cfi_offset 23, -96
	.cfi_offset 24, -88
	.cfi_offset 25, -80
	.cfi_offset 26, -72
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
	beq	.L39
	cmp	x24, 0
	ccmp	x23, 0, 4, ne
	beq	.L39
	cmp	x22, 0
	mov	x21, x0
	ccmp	x0, 0, 4, ne
	beq	.L39
	mov	w0, 0
	stp	x19, x20, [sp, 16]
	.cfi_offset 20, -120
	.cfi_offset 19, -128
	mov	x19, 0
	stp	x27, x28, [sp, 80]
	.cfi_offset 28, -56
	.cfi_offset 27, -64
	stp	d13, d14, [sp, 96]
	.cfi_offset 78, -40
	.cfi_offset 77, -48
	str	d15, [sp, 112]
	.cfi_offset 79, -32
	bl	srand
	.p2align 5,,15
.L42:
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
	bne	.L42
	movi	v31.4s, 0
	mov	x0, 0
	.p2align 5,,15
.L43:
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
	bne	.L43
	addv	s31, v31.4s
	mov	x0, 0
	fmov	w28, s31
	movi	v31.4s, 0
	.p2align 5,,15
.L44:
	ldr	q29, [x23, x0]
	ldr	q30, [x24, x0]
	add	x0, x0, 16
	mla	v31.4s, v29.4s, v30.4s
	cmp	x0, 65536
	bne	.L44
	addv	s31, v31.4s
	mov	x0, 0
	movi	v15.2s, #0
	str	s31, [sp, 136]
	.p2align 5,,15
.L45:
	ldr	q30, [x22, x0]
	ldr	q31, [x21, x0]
	add	x0, x0, 16
	fmul	v31.4s, v31.4s, v30.4s
	fadd	s15, s15, s31
	dup	s30, v31.s[1]
	fadd	s30, s30, s15
	dup	s15, v31.s[2]
	dup	s31, v31.s[3]
	fadd	s15, s15, s30
	fadd	s15, s15, s31
	cmp	x0, 65536
	bne	.L45
	mov	x1, x25
	mov	x0, x26
	mov	w2, 16384
	bl	dot_product_int8_neon
	movi	v31.4s, 0
	mov	w27, w0
	mov	x0, 0
	.p2align 5,,15
.L46:
	ldr	q29, [x24, x0]
	ldr	q30, [x23, x0]
	add	x0, x0, 16
	mla	v31.4s, v29.4s, v30.4s
	cmp	x0, 65536
	bne	.L46
	dup	d14, v31.d[1]
	mov	x0, 0
	movi	v30.4s, 0
	add	v14.2s, v14.2s, v31.2s
	addp	v14.2s, v14.2s, v14.2s
	str	s14, [sp, 140]
	.p2align 5,,15
.L47:
	ldr	q31, [x22, x0]
	ldr	q29, [x21, x0]
	add	x0, x0, 16
	fmul	v31.4s, v31.4s, v29.4s
	fadd	v30.4s, v30.4s, v31.4s
	cmp	x0, 65536
	bne	.L47
	dup	d1, v30.d[1]
	adrp	x4, .LC3
	adrp	x3, .LC4
	add	x4, x4, :lo12:.LC3
	add	x3, x3, :lo12:.LC4
	adrp	x2, .LC5
	adrp	x1, .LC6
	add	x2, x2, :lo12:.LC5
	fadd	v1.2s, v1.2s, v30.2s
	add	x1, x1, :lo12:.LC6
	adrp	x0, .LC7
	add	x0, x0, :lo12:.LC7
	faddp	v13.2s, v1.2s, v1.2s
	bl	printf
	cmp	w27, w28
	adrp	x5, .LC0
	adrp	x6, .LC1
	add	x19, x5, :lo12:.LC0
	add	x20, x6, :lo12:.LC1
	adrp	x1, .LC8
	add	x1, x1, :lo12:.LC8
	csel	x4, x20, x19, ne
	mov	w2, w28
	mov	w3, w27
	adrp	x28, .LC9
	add	x0, x28, :lo12:.LC9
	bl	printf
	fmov	w1, s14
	ldp	w0, w3, [sp, 136]
	cmp	w0, w1
	mov	w2, w0
	csel	x4, x20, x19, ne
	add	x0, x28, :lo12:.LC9
	adrp	x1, .LC10
	add	x1, x1, :lo12:.LC10
	bl	printf
	fcvt	d1, s13
	fcvt	d0, s15
	adrp	x2, .LC11
	adrp	x1, .LC12
	add	x2, x2, :lo12:.LC11
	add	x1, x1, :lo12:.LC12
	adrp	x0, .LC13
	add	x0, x0, :lo12:.LC13
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
	ldp	x27, x28, [sp, 80]
	.cfi_restore 28
	.cfi_restore 27
	ldp	d13, d14, [sp, 96]
	.cfi_restore 78
	.cfi_restore 77
	ldr	d15, [sp, 112]
	.cfi_restore 79
.L38:
	ldp	x21, x22, [sp, 32]
	ldp	x23, x24, [sp, 48]
	ldp	x25, x26, [sp, 64]
	ldp	x29, x30, [sp], 144
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
.L39:
	.cfi_restore_state
	adrp	x0, .LC2
	add	x0, x0, :lo12:.LC2
	bl	perror
	mov	w0, 1
	b	.L38
	.cfi_endproc
.LFE3914:
	.size	main, .-main
	.ident	"GCC: (GNU) 15.1.0"
	.section	.note.GNU-stack,"",@progbits
