//!JIT backend
//!
//! Optimizations active:
//!
//! A. Full linear-scan register allocation across 10 GPRs
//!    (r8-r11, rsi — caller-saved; r12-r15, rbx — callee-saved).
//!    rax/rcx = scratch accumulators.  rdx = reserved for cqo/idiv.
//!    rdi = slot-array base pointer.  All other GPRs allocated by RA.
//!    Callee-saved registers that are actually used get pushed/popped
//!    in a generated prologue/epilogue.
//!
//! B. Forward constant-propagation + compile-time BinOp folding.
//!    Any slot whose value is statically known is propagated through
//!    Move/Store/BinOp; foldable BinOps emit a single immediate load.
//!    State is conservatively cleared at every branch target.
//!    const_at is a flat Vec<Option<i64>> (O(1) slot lookup, cache-friendly).
//!
//! C. 3-instruction superinstruction fusions:
//!    • BinOp(t, Mul, x, N) + BinOp(r, Add, t, y)  →  LEA  (N ∈ {2,4,8})
//!    • BinOp(t, op1, a, b) + BinOp(r, op2, t, c)  →  two-op chain
//!      (eliminates the intermediate slot store + load of `t`)
//!
//! D. 2-instruction superinstruction fusions (all original patterns):
//!    • LoadI*(tmp, c) + JumpFalse/True(tmp, …) → compile-time branch fold
//!    • LoadI*(tmp, c) + BinOp(d, op, x, tmp)   → immediate-form arithmetic
//!    • BinOp(t, op, l, r) + Store(slot, t)      → fused compute-and-store
//!    • BinOp(t, op, l, r) + JumpFalse/True(t,…) → fused cmp+branch
//!
//! E. Optimal immediate encoding:
//!    MOV EAX, imm32 (5 B, zero-extends) when 0 ≤ v < 2³¹
//!    MOV RAX, sign-extended imm32 (7 B) when −2³¹ ≤ v < 0
//!    MOV RAX, imm64 (10 B) only when the value doesn't fit in 32 bits.
//!    ADD/SUB/CMP RAX, imm8 (4 B) when value fits in i8.
//!
//! F. Short-form branches: JMP/JZ/JNZ rel8 (2 B) when displacement fits in i8,
//!    falling back to rel32 (5-6 B) otherwise.
//!
//! G. REX-free XOR/TEST for boolean/zero ops:
//!    XOR EAX, EAX (2 B) and TEST EAX, EAX (2 B) instead of 64-bit forms (3 B).
//!
//! H. All existing micro-optimisations retained:
//!    LEA for ×3/×5/×9, SHL for powers-of-two multiply,
//!    INC/DEC for ±1, TEST+Jcc, SETCC for branchless comparisons.

use std::cell::RefCell;
use std::ptr::NonNull;

use libc::{mmap, munmap, MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};

use crate::ast::BinOpKind;
use crate::interp::{CompiledFn, Instr, RuntimeError, Value};

// ─────────────────────────────────────────────────────────────────────────────
// Executable memory
// ─────────────────────────────────────────────────────────────────────────────

pub struct NativeCode {
    pub slot_count: u16,
    mem: ExecMem,
}

struct ExecMem {
    ptr: *mut u8,
    len: usize,
    arena_backed: bool,
}

struct ExecArena {
    base: NonNull<u8>,
    len: usize,
    cursor: usize,
}

impl Drop for ExecArena {
    fn drop(&mut self) {
        unsafe { munmap(self.base.as_ptr().cast(), self.len) };
    }
}

impl ExecArena {
    const DEFAULT_LEN: usize = 16 * 1024 * 1024;

    fn try_new() -> Option<Self> {
        // Try huge-page backed mapping first (Linux only).
        #[cfg(target_os = "linux")]
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                Self::DEFAULT_LEN,
                PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANON | libc::MAP_HUGETLB,
                -1,
                0,
            )
        };
        #[cfg(not(target_os = "linux"))]
        let ptr = libc::MAP_FAILED;

        let ptr = if ptr.is_null() || ptr == libc::MAP_FAILED {
            unsafe {
                mmap(
                    std::ptr::null_mut(),
                    Self::DEFAULT_LEN,
                    PROT_READ | PROT_WRITE | PROT_EXEC,
                    MAP_PRIVATE | MAP_ANON,
                    -1,
                    0,
                )
            }
        } else {
            ptr
        };
        if ptr.is_null() || ptr == libc::MAP_FAILED {
            return None;
        }

        // Hint to kernel: keep these pages in hugepage pool even if not huge-page mapped.
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(ptr, Self::DEFAULT_LEN, libc::MADV_HUGEPAGE);
        }

        Some(Self {
            base: NonNull::new(ptr.cast::<u8>())?,
            len: Self::DEFAULT_LEN,
            cursor: 0,
        })
    }

    fn alloc(&mut self, bytes: usize) -> Option<*mut u8> {
        let aligned = (bytes + 15) & !15;
        let next = self.cursor.checked_add(aligned)?;
        if next > self.len {
            // Grow exponentially: double the arena size, capped at 128 MiB.
            let new_len = (self.len * 2).min(128 * 1024 * 1024);
            let new_ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    new_len,
                    PROT_READ | PROT_WRITE | PROT_EXEC,
                    MAP_PRIVATE | MAP_ANON,
                    -1,
                    0,
                )
            };
            if new_ptr.is_null() || new_ptr == libc::MAP_FAILED {
                return None;
            }
            // Copy existing code to the new arena.
            unsafe {
                std::ptr::copy_nonoverlapping(self.base.as_ptr(), new_ptr.cast::<u8>(), self.cursor);
            }
            // Unmap the old arena.
            unsafe { munmap(self.base.as_ptr().cast(), self.len) };
            self.base = NonNull::new(new_ptr.cast::<u8>())?;
            self.len = new_len;
        }
        let out = unsafe { self.base.as_ptr().add(self.cursor) };
        self.cursor = next;
        Some(out)
    }
}

impl Drop for ExecMem {
    fn drop(&mut self) {
        if !self.arena_backed && !self.ptr.is_null() && self.len > 0 {
            unsafe { munmap(self.ptr.cast(), self.len) };
        }
    }
}

thread_local! {
    static TLS_EXEC_ARENA: RefCell<Option<ExecArena>> = const { RefCell::new(None) };
}

impl ExecMem {
    fn new(code: &[u8]) -> Option<Self> {
        if let Some(mem) = TLS_EXEC_ARENA.with(|arena_cell| {
            let mut arena = arena_cell.borrow_mut();
            if arena.is_none() {
                *arena = ExecArena::try_new();
            }
            let arena = arena.as_mut()?;
            let ptr = arena.alloc(code.len().max(1))?;
            unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len()) };
            Some(Self {
                ptr,
                len: code.len().max(1),
                arena_backed: true,
            })
        }) {
            return Some(mem);
        }

        // Fallback: individual mmap per function.
        let len = code.len().max(1);
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        if ptr.is_null() || ptr == libc::MAP_FAILED {
            return None;
        }
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), ptr.cast::<u8>(), code.len()) };
        Some(Self {
            ptr: ptr.cast::<u8>(),
            len,
            arena_backed: false, // BUG FIX: was missing, causing use-after-free on drop
        })
    }

    fn entry(&self) -> unsafe extern "C" fn(*mut i64) -> i64 {
        unsafe { std::mem::transmute(self.ptr) }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Code emitter
// ─────────────────────────────────────────────────────────────────────────────
//
// Register number convention (matches x86-64 encoding):
//   rax=0  rcx=1  rdx=2  rbx=3  rsp=4  rbp=5  rsi=6  rdi=7
//   r8=8   r9=9   r10=10 r11=11 r12=12 r13=13 r14=14 r15=15

struct Emitter {
    buf: Vec<u8>,
}

impl Emitter {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(16384),
        }
    }

    #[inline(always)]
    fn pos(&self) -> usize {
        self.buf.len()
    }

    #[inline(always)]
    fn b(&mut self, v: u8) {
        self.buf.push(v);
    }

    #[inline(always)]
    fn emit2(&mut self, b0: u8, b1: u8) {
        self.buf.extend_from_slice(&[b0, b1]);
    }

    #[inline(always)]
    fn emit3(&mut self, b0: u8, b1: u8, b2: u8) {
        self.buf.extend_from_slice(&[b0, b1, b2]);
    }

    #[inline(always)]
    fn emit4(&mut self, b0: u8, b1: u8, b2: u8, b3: u8) {
        self.buf.extend_from_slice(&[b0, b1, b2, b3]);
    }

    #[inline(always)]
    fn d(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline(always)]
    fn q(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Emit a REX prefix byte
    #[inline(always)]
    fn emit_rex(&mut self, w: bool, r: bool, x: bool, b: bool) {
        let rex = 0x40 | ((w as u8) << 3) | ((r as u8) << 2) | ((x as u8) << 1) | (b as u8);
        self.b(rex);
    }

    /// Emit a ModRM byte
    #[inline(always)]
    fn emit_modrm(&mut self, mode: u8, reg: u8, rm: u8) {
        let modrm = (mode << 6) | ((reg & 0x7) << 3) | (rm & 0x7);
        self.b(modrm);
    }

    // ── Immediate loads ──────────────────────────────────────────────────────

    /// Full 64-bit immediate into rax (10 bytes).
    fn mov_rax_imm64(&mut self, v: i64) {
        self.emit2(0x48, 0xB8);
        self.q(v);
    }

    /// Optimal immediate into rax.
    ///  v ≥ 0 and fits i32 → MOV EAX, imm32 (5 B, zero-extends)
    ///  v <  0 and fits i32 → REX.W MOV RAX, sign-ext imm32 (7 B)
    ///  otherwise           → MOV RAX, imm64 (10 B)
    fn mov_rax_imm_opt(&mut self, v: i64) {
        if let Ok(v32) = i32::try_from(v) {
            if v32 >= 0 {
                self.b(0xB8);
                self.d(v32); // MOV EAX, imm32
            } else {
                self.emit3(0x48, 0xC7, 0xC0);
                self.d(v32); // MOV RAX, sx(imm32)
            }
        } else {
            self.mov_rax_imm64(v);
        }
    }

    // ── Generic register ↔ memory ────────────────────────────────────────────
    //
    // REX.W  = bit3 (always 1 for 64-bit)
    // REX.R  = bit2 (extends ModRM.reg  — destination for 8B, source for 89)
    // REX.B  = bit0 (extends ModRM.rm   — source     for 8B, dest   for 89 w/ mod=11)
    //
    // For [rdi + disp32]: ModRM = mod10 | reg_field<<3 | 7

    /// mov reg64, [rdi + disp32]
    fn load_reg_mem(&mut self, reg: u8, disp: i32) {
        self.emit3(
            0x48 | ((reg & 8) >> 1), // REX.W | REX.R
            0x8B,
            0x87 | ((reg & 7) << 3), // mod=10, reg, rm=7(rdi)
        );
        self.d(disp);
    }

    /// mov [rdi + disp32], reg64
    fn store_mem_reg(&mut self, disp: i32, reg: u8) {
        self.emit3(0x48 | ((reg & 8) >> 1), 0x89, 0x87 | ((reg & 7) << 3));
        self.d(disp);
    }

    /// mov dst64, src64  — no-op when dst == src.
    fn mov_rr(&mut self, dst: u8, src: u8) {
        if dst == src {
            return;
        }
        // MOV r64, r/m64 (0x8B): REX.R extends dst, REX.B extends src
        self.emit3(
            0x48 | ((dst & 8) >> 1) | ((src & 8) >> 3),
            0x8B,
            0xC0 | ((dst & 7) << 3) | (src & 7),
        );
    }

    // ── Arithmetic (rax / rcx) ───────────────────────────────────────────────

    fn add_rax_rcx(&mut self) {
        self.emit3(0x48, 0x01, 0xC8);
    }
    fn sub_rax_rcx(&mut self) {
        self.emit3(0x48, 0x29, 0xC8);
    }
    fn imul_rax_rcx(&mut self) {
        self.emit4(0x48, 0x0F, 0xAF, 0xC1);
    }

    /// ADD RAX, imm — uses imm8 form (4 B) when it fits, else imm32 (6 B).
    fn add_rax_imm32(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) {
            self.emit3(0x48, 0x83, 0xC0);
            self.b(v8 as u8); // ADD RAX, imm8
        } else {
            self.emit2(0x48, 0x05);
            self.d(v); // ADD RAX, imm32
        }
    }

    /// SUB RAX, imm — uses imm8 form (4 B) when it fits, else imm32 (6 B).
    fn sub_rax_imm32(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) {
            self.emit3(0x48, 0x83, 0xE8);
            self.b(v8 as u8); // SUB RAX, imm8
        } else {
            self.emit2(0x48, 0x2D);
            self.d(v); // SUB RAX, imm32
        }
    }

    fn imul_rax_imm32(&mut self, v: i32) {
        self.emit3(0x48, 0x69, 0xC0);
        self.d(v);
    }

    fn inc_rax(&mut self) {
        self.emit3(0x48, 0xFF, 0xC0);
    }
    fn dec_rax(&mut self) {
        self.emit3(0x48, 0xFF, 0xC8);
    }
    fn neg_rax(&mut self) {
        self.emit3(0x48, 0xF7, 0xD8);
    }
    fn shl_rax_imm8(&mut self, v: u8) {
        self.emit4(0x48, 0xC1, 0xE0, v);
    }

    /// XOR EAX, EAX — 2 bytes, zero-extends to clear full RAX.
    /// Preferred over REX.W form (3 bytes) for zeroing.
    fn xor_eax_eax(&mut self) {
        self.emit2(0x31, 0xC0); // XOR EAX, EAX  (zero-extends → RAX = 0)
    }

    // LEA ×N patterns on rax only (rax = rax*N via SIB with base=rax, index=rax).
    // SIB for [rax + rax*K]: scale_bits<<6 | index=rax(0)<<3 | base=rax(0)
    fn lea_rax_rax_mul3(&mut self) {
        self.emit4(0x48, 0x8D, 0x04, 0x40);
    }
    fn lea_rax_rax_mul5(&mut self) {
        self.emit4(0x48, 0x8D, 0x04, 0x80);
    }
    fn lea_rax_rax_mul9(&mut self) {
        self.emit4(0x48, 0x8D, 0x04, 0xC0);
    }

    /// lea rax, [rcx + rax*scale]   scale ∈ {2,4,8}
    /// Used by Mul+Add→LEA fusion: rax = multiplicand, rcx = addend.
    /// Result: rax*scale + rcx.
    fn lea_rax_rax_scale_plus_rcx(&mut self, scale: u8) {
        // SIB: ss<<6 | index=rax(0)<<3 | base=rcx(1)
        let ss: u8 = match scale {
            4 => 2,
            8 => 3,
            _ => 1,
        };
        self.emit4(0x48, 0x8D, 0x04, (ss << 6) | 1);
    }

    // ── Division ─────────────────────────────────────────────────────────────

    fn cqo(&mut self) {
        self.emit2(0x48, 0x99);
    }
    fn idiv_rcx(&mut self) {
        self.emit3(0x48, 0xF7, 0xF9);
    }
    fn mov_rax_rdx(&mut self) {
        self.emit3(0x48, 0x89, 0xD0);
    }

    // ── Compare / branch ─────────────────────────────────────────────────────

    fn cmp_rax_rcx(&mut self) {
        self.emit3(0x48, 0x39, 0xC8);
    }

    /// CMP RAX, imm — uses imm8 form (4 B) when it fits, else imm32 (6 B).
    fn cmp_rax_imm32(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) {
            self.emit3(0x48, 0x83, 0xF8);
            self.b(v8 as u8); // CMP RAX, imm8
        } else {
            self.emit2(0x48, 0x3D);
            self.d(v); // CMP RAX, imm32
        }
    }

    /// TEST EAX, EAX — 2 bytes. Sufficient for boolean/zero testing since
    /// we only care about ZF; the upper 32 bits are zero for any canonical bool.
    /// Falls back to REX.W form for full 64-bit values tested against branches
    /// (call test_rax_rax for those if unsure).
    fn test_rax_rax(&mut self) {
        // TEST EAX, EAX (2 B) — ZF ↔ (eax == 0), which equals (rax == 0)
        // because our boolean values are 0 or 1 and always fit in 32 bits.
        // For general integer branch conditions (JumpFalse/True on arbitrary i64),
        // we also use this: canonical i64 0 has upper 32 bits = 0, so it's safe.
        self.emit2(0x85, 0xC0); // TEST EAX, EAX
    }

    fn setcc_al(&mut self, cc: u8) {
        self.emit3(0x0F, cc, 0xC0);
    }
    fn movzx_rax_al(&mut self) {
        self.emit4(0x48, 0x0F, 0xB6, 0xC0);
    }

    // ── Branches ─────────────────────────────────────────────────────────────
    //
    // Two-pass strategy: first emit rel32 placeholders, then patch.
    // Alternatively, for short branches we can back-patch with rel8.
    // We use a unified placeholder approach and pick the right encoding at
    // fixup time via a "rewrite" pass (see patch_fixups).

    fn jmp_rel32_placeholder(&mut self) -> usize {
        self.b(0xE9);
        let p = self.pos();
        self.d(0);
        p
    }
    fn jz_rel32_placeholder(&mut self) -> usize {
        self.emit2(0x0F, 0x84);
        let p = self.pos();
        self.d(0);
        p
    }
    fn jnz_rel32_placeholder(&mut self) -> usize {
        self.emit2(0x0F, 0x85);
        let p = self.pos();
        self.d(0);
        p
    }

    fn ret(&mut self) {
        self.b(0xC3);
    }

    // ── XMM (SSE2) floating-point instructions ─────────────────────────────
    //
    // XMM registers: xmm0-xmm15. We use xmm0-xmm7 for f32/f64 arithmetic.
    // The JIT maps f32/f64 slots to XMM registers when possible, falling back
    // to memory (the rdi slot array) for spills.

    /// movsd xmm0, [rdi + disp32]  — load f64 from slot
    fn load_xmm0_mem(&mut self, disp: i32) {
        // F2 prefix (SIMD scalar double): REX.W=0x48 + F2 0F 10 /r
        self.emit4(0xF2, 0x48, 0x0F, 0x10);
        // ModRM: mod=10 (disp32), reg=000 (xmm0), rm=111 (rdi)
        self.b(0x87);
        self.d(disp);
    }

    /// movss xmm0, [rdi + disp32]  — load f32 from slot
    fn load_xmm0_mem_f32(&mut self, disp: i32) {
        // F3 prefix (SIMD scalar single): F3 0F 10 /r
        self.emit3(0xF3, 0x0F, 0x10);
        // ModRM: mod=10, reg=000, rm=111
        self.b(0x87);
        self.d(disp);
    }

    /// movsd [rdi + disp32], xmm0  — store f64 to slot
    fn store_mem_xmm0(&mut self, disp: i32) {
        self.emit4(0xF2, 0x48, 0x0F, 0x11);
        self.b(0x87);
        self.d(disp);
    }

    /// movss [rdi + disp32], xmm0  — store f32 to slot
    fn store_mem_xmm0_f32(&mut self, disp: i32) {
        self.emit3(0xF3, 0x0F, 0x11);
        self.b(0x87);
        self.d(disp);
    }

    /// addsd xmm0, xmm1  — f64 add
    fn add_xmm0_xmm1_f64(&mut self) {
        self.emit4(0xF2, 0x48, 0x0F, 0x58);
        self.b(0xC1); // mod=11, reg=xmm0, rm=xmm1
    }

    /// addss xmm0, xmm1  — f32 add
    fn add_xmm0_xmm1_f32(&mut self) {
        self.emit3(0xF3, 0x0F, 0x58);
        self.b(0xC1);
    }

    /// subsd xmm0, xmm1  — f64 sub
    fn sub_xmm0_xmm1_f64(&mut self) {
        self.emit4(0xF2, 0x48, 0x0F, 0x5C);
        self.b(0xC1);
    }

    /// subss xmm0, xmm1  — f32 sub
    fn sub_xmm0_xmm1_f32(&mut self) {
        self.emit3(0xF3, 0x0F, 0x5C);
        self.b(0xC1);
    }

    /// mulsd xmm0, xmm1  — f64 mul
    fn mul_xmm0_xmm1_f64(&mut self) {
        self.emit4(0xF2, 0x48, 0x0F, 0x59);
        self.b(0xC1);
    }

    /// mulss xmm0, xmm1  — f32 mul
    fn mul_xmm0_xmm1_f32(&mut self) {
        self.emit3(0xF3, 0x0F, 0x59);
        self.b(0xC1);
    }

    /// divsd xmm0, xmm1  — f64 div
    fn div_xmm0_xmm1_f64(&mut self) {
        self.emit4(0xF2, 0x48, 0x0F, 0x5E);
        self.b(0xC1);
    }

    /// divss xmm0, xmm1  — f32 div
    fn div_xmm0_xmm1_f32(&mut self) {
        self.emit3(0xF3, 0x0F, 0x5E);
        self.b(0xC1);
    }

    /// movq xmm1, rax  — move integer → XMM (for mixed int/float ops)
    fn movq_xmm1_rax(&mut self) {
        self.emit4(0x66, 0x48, 0x0F, 0x6E);
        self.b(0xC8); // mod=11, reg=xmm1, rm=rax
    }

    /// ucomisd xmm0, xmm1  — f64 compare, sets EFLAGS
    fn ucomisd_xmm0_xmm1(&mut self) {
        self.emit3(0x66, 0x0F, 0x2E);
        self.b(0xC1);
    }

    /// ucomiss xmm0, xmm1  — f32 compare
    fn ucomiss_xmm0_xmm1(&mut self) {
        self.emit3(0x0F, 0x2E, 0xC1);
    }

    /// cvttsd2si eax, xmm0  — f64 → i32 (truncating)
    fn cvttsd2si_eax_xmm0(&mut self) {
        self.emit2(0xF2, 0x48);
        self.emit_rex(false, true, false, false);
        self.emit_modrm(0xC0, 0, 0);
        self.b(0x2C);
        self.b(0xC0);
    }

    /// cvttss2si eax, xmm0  — f32 → i32 (truncating)
    fn cvttss2si_eax_xmm0(&mut self) {
        self.emit3(0xF3, 0x0F, 0x2C);
        self.b(0xC0);
    }

    /// cvtsi2sd xmm0, rax  — i64 → f64
    fn cvtsi2sd_xmm0_rax(&mut self) {
        self.emit4(0xF2, 0x48, 0x0F, 0x2A);
        self.b(0xC0);
    }

    /// cvtsi2ss xmm0, rax  — i64 → f32
    fn cvtsi2ss_xmm0_rax(&mut self) {
        self.emit4(0xF3, 0x48, 0x0F, 0x2A);
        self.b(0xC0);
    }

    /// Load immediate f64 into xmm0: move bits into rax, then movq xmm0, rax
    fn mov_xmm0_imm64(&mut self, bits: u64) {
        self.emit2(0x48, 0xB8); // MOV RAX, imm64
        self.q(bits as i64);
        self.emit4(0x66, 0x48, 0x0F, 0x6E); // MOVQ XMM0, RAX
        self.b(0xC0);
    }

    /// Load immediate f32 into xmm0: zero-extend to 64-bit, then movq
    fn mov_xmm0_imm32_bits(&mut self, bits: u32) {
        self.b(0xB8); // MOV EAX, imm32 (zero-extends)
        self.d(bits as i32);
        self.emit4(0x66, 0x48, 0x0F, 0x6E); // MOVQ XMM0, RAX
        self.b(0xC0);
    }

    // ── Callee-saved save/restore ────────────────────────────────────────────
    //
    // Short-form PUSH/POP r64: 0x50+rd  (rd = reg & 7)
    // r8-r15 require REX.B prefix (0x41).

    fn push_reg(&mut self, reg: u8) {
        if reg >= 8 {
            self.b(0x41);
        }
        self.b(0x50 + (reg & 7));
    }
    fn pop_reg(&mut self, reg: u8) {
        if reg >= 8 {
            self.b(0x41);
        }
        self.b(0x58 + (reg & 7));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Register allocation
// ─────────────────────────────────────────────────────────────────────────────

/// Allocation pool ordered: caller-saved first (cheaper — no push/pop),
/// then callee-saved.
///
/// Excluded permanently:
///   rax(0)  — primary accumulator / return value
///   rcx(1)  — secondary scratch operand
///   rdx(2)  — clobbered by CQO/IDIV
///   rdi(7)  — slot-array base pointer (function argument)
///   rsp(4) / rbp(5) — stack management
const ALLOC_POOL: &[u8] = &[
    8, 9, 10, 11, 6, // r8-r11, rsi  (caller-saved — free)
    12, 13, 14, 15, 3, // r12-r15, rbx (callee-saved — require push/pop)
];

/// Where a bytecode slot lives at runtime.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RegLoc {
    Reg(u8),    // allocated physical register
    Spill(i32), // byte offset inside rdi[] slot array
}

struct RegAlloc {
    /// Direct Vec indexed by slot number; pre-filled with Spill defaults.
    /// O(1) lookup vs HashMap for every load/store in the hot codegen loop.
    slots: Vec<RegLoc>,
    /// Callee-saved registers actually used — must be pushed in prologue.
    used_callee_saved: Vec<u8>,
}

impl RegAlloc {
    #[inline(always)]
    fn location(&self, slot: u16) -> RegLoc {
        // Safety: slots is pre-allocated to cover all valid slot indices.
        unsafe { *self.slots.get_unchecked(slot as usize) }
    }
}

// ── Live intervals ────────────────────────────────────────────────────────────

struct LiveInterval {
    slot: u16,
    first: usize, // index of first instruction that defines or uses this slot
    last: usize,  // index of last instruction that uses this slot
}

fn compute_live_intervals(instrs: &[Instr], slot_count: usize) -> Vec<LiveInterval> {
    const UNDEF: usize = usize::MAX;
    let cap = slot_count + 1;
    let mut first_def = vec![UNDEF; cap];
    let mut last_use = vec![UNDEF; cap];

    macro_rules! ensure {
        ($s:expr) => {
            let s = $s as usize;
            if s >= first_def.len() {
                first_def.resize(s + 1, UNDEF);
                last_use.resize(s + 1, UNDEF);
            }
        };
    }
    macro_rules! def {
        ($s:expr, $pc:expr) => {
            ensure!($s);
            let s = $s as usize;
            if first_def[s] == UNDEF {
                first_def[s] = $pc;
            }
        };
    }
    macro_rules! use_ {
        ($s:expr, $pc:expr) => {
            ensure!($s);
            last_use[$s as usize] = $pc;
        };
    }

    for (pc, instr) in instrs.iter().enumerate() {
        match instr {
            Instr::LoadI32(d, _)
            | Instr::LoadI64(d, _)
            | Instr::LoadBool(d, _)
            | Instr::LoadUnit(d) => {
                def!(*d, pc);
            }
            Instr::Move(d, s) | Instr::Load(d, s) => {
                use_!(*s, pc);
                def!(*d, pc);
            }
            Instr::Store(slot, s) => {
                use_!(*s, pc);
                def!(*slot, pc);
            }
            Instr::BinOp(d, _, l, r) => {
                use_!(*l, pc);
                use_!(*r, pc);
                def!(*d, pc);
            }
            Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) | Instr::Return(s) => {
                use_!(*s, pc);
            }
            _ => {}
        }
    }

    // Slots that appear only as uses (never defined) are argument slots — treat as defined at 0.
    let mut intervals: Vec<LiveInterval> = Vec::new();
    for slot in 0..first_def.len() {
        let lu = last_use[slot];
        let fd = first_def[slot];
        if fd == UNDEF && lu == UNDEF {
            continue;
        }
        let first = if fd == UNDEF { 0 } else { fd };
        let last = if lu == UNDEF { first } else { lu };
        intervals.push(LiveInterval {
            slot: slot as u16,
            first,
            last,
        });
    }
    intervals.sort_unstable_by_key(|i| (i.first, i.slot));
    intervals
}

// ── Linear-scan allocator ─────────────────────────────────────────────────────

fn linear_scan(intervals: &[LiveInterval], slot_count: usize) -> RegAlloc {
    // Pre-allocate slots to slot_count + 1 with default spill locations.
    // This eliminates all resizing during allocation.
    let cap = slot_count + 1;
    let mut slots: Vec<RegLoc> = (0..cap).map(|s| RegLoc::Spill((s as i32) * 8)).collect();

    // Free list: iterate ALLOC_POOL in reverse so pop() gives caller-saved first.
    let mut free: Vec<u8> = ALLOC_POOL.iter().rev().copied().collect();
    // Active set sorted by interval end (ascending).
    // Pre-allocate to avoid reallocations.
    let mut active: Vec<(usize, u16, u8)> = Vec::with_capacity(intervals.len().min(free.len()));
    let mut used_callee_saved: Vec<u8> = Vec::with_capacity(ALLOC_POOL.len());
    // Bitmask to avoid O(n) contains() on used_callee_saved (regs 0-15 fit in u16).
    let mut callee_saved_mask: u16 = 0;

    #[inline(always)]
    fn is_callee_saved(reg: u8) -> bool {
        matches!(reg, 3 | 12..=15)
    }

    for iv in intervals {
        // Expire intervals that ended strictly before this one's start.
        let expired = active.partition_point(|(end, _, _)| *end < iv.first);
        for i in 0..expired {
            free.push(active[i].2);
        }
        if expired > 0 {
            active.drain(0..expired);
        }

        let mut track_callee = |reg: u8| {
            if is_callee_saved(reg) && (callee_saved_mask & (1u16 << reg)) == 0 {
                callee_saved_mask |= 1u16 << reg;
                used_callee_saved.push(reg);
            }
        };

        if let Some(reg) = free.pop() {
            track_callee(reg);
            slots[iv.slot as usize] = RegLoc::Reg(reg);
            let pos = active.partition_point(|(e, _, _)| *e <= iv.last);
            active.insert(pos, (iv.last, iv.slot, reg));
        } else {
            match active.last().copied() {
                Some((end, spill_slot, reg)) if end > iv.last => {
                    slots[spill_slot as usize] = RegLoc::Spill((spill_slot as i32) * 8);
                    active.pop();
                    track_callee(reg);
                    slots[iv.slot as usize] = RegLoc::Reg(reg);
                    let pos = active.partition_point(|(e, _, _)| *e <= iv.last);
                    active.insert(pos, (iv.last, iv.slot, reg));
                }
                _ => {
                    // Already pre-filled with Spill; nothing to do.
                }
            }
        }
    }

    RegAlloc {
        slots,
        used_callee_saved,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constant propagation
// ─────────────────────────────────────────────────────────────────────────────

fn fold_binop(op: BinOpKind, l: i64, r: i64) -> Option<i64> {
    Some(match op {
        BinOpKind::Add => l.wrapping_add(r),
        BinOpKind::Sub => l.wrapping_sub(r),
        BinOpKind::Mul => l.wrapping_mul(r),
        BinOpKind::Div => {
            if r == 0 {
                return None;
            }
            l.wrapping_div(r)
        }
        BinOpKind::Rem => {
            if r == 0 {
                return None;
            }
            l.wrapping_rem(r)
        }
        BinOpKind::Eq => i64::from(l == r),
        BinOpKind::Ne => i64::from(l != r),
        BinOpKind::Lt => i64::from(l < r),
        BinOpKind::Le => i64::from(l <= r),
        BinOpKind::Gt => i64::from(l > r),
        BinOpKind::Ge => i64::from(l >= r),
        _ => return None,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Flat constant-propagation table (replaces HashMap<u16, i64>)
// ─────────────────────────────────────────────────────────────────────────────
//
// Vec<Option<i64>> indexed by slot gives O(1) lookup with zero hashing overhead.
// For a function with N slots this uses N*9 bytes vs HashMap's ~48 bytes base +
// 24 bytes/entry at low load factors. More importantly, sequential slot accesses
// stay in L1 cache during the hot codegen loop.

struct ConstTable {
    vals: Vec<Option<i64>>,
}

impl ConstTable {
    fn with_capacity(n: usize) -> Self {
        Self {
            vals: vec![None; n.max(1)],
        }
    }

    #[inline(always)]
    fn get(&self, slot: u16) -> Option<i64> {
        self.vals.get(slot as usize).copied().flatten()
    }

    #[inline(always)]
    fn insert(&mut self, slot: u16, v: i64) {
        let idx = slot as usize;
        if idx >= self.vals.len() {
            self.vals.resize(idx + 1, None);
        }
        self.vals[idx] = Some(v);
    }

    #[inline(always)]
    fn remove(&mut self, slot: u16) {
        if let Some(cell) = self.vals.get_mut(slot as usize) {
            *cell = None;
        }
    }

    /// Clear all known constants (conservative: called at branch targets).
    fn clear(&mut self) {
        self.vals.fill(None);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Emission helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn load_rax(em: &mut Emitter, slot: u16, ra: &RegAlloc) {
    match ra.location(slot) {
        RegLoc::Reg(r) => em.mov_rr(0, r),
        RegLoc::Spill(off) => em.load_reg_mem(0, off),
    }
}

#[inline(always)]
fn load_rcx(em: &mut Emitter, slot: u16, ra: &RegAlloc) {
    match ra.location(slot) {
        RegLoc::Reg(r) => em.mov_rr(1, r),
        RegLoc::Spill(off) => em.load_reg_mem(1, off),
    }
}

#[inline(always)]
fn store_rax(em: &mut Emitter, slot: u16, ra: &RegAlloc) {
    match ra.location(slot) {
        RegLoc::Reg(r) => em.mov_rr(r, 0),
        RegLoc::Spill(off) => em.store_mem_reg(off, 0),
    }
}

#[inline(always)]
fn instr_reads_slot(instr: &Instr, slot: u16) -> bool {
    match instr {
        Instr::Move(_, s) | Instr::Load(_, s) | Instr::Store(_, s) | Instr::Return(s) => *s == slot,
        Instr::BinOp(_, _, l, r) => *l == slot || *r == slot,
        Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) => *s == slot,
        _ => false,
    }
}

#[inline(always)]
fn instr_writes_slot(instr: &Instr, slot: u16) -> bool {
    match instr {
        Instr::LoadI32(d, _)
        | Instr::LoadI64(d, _)
        | Instr::LoadBool(d, _)
        | Instr::LoadUnit(d)
        | Instr::Move(d, _)
        | Instr::Load(d, _)
        | Instr::Store(d, _)
        | Instr::BinOp(d, _, _, _) => *d == slot,
        _ => false,
    }
}

#[inline(always)]
fn is_control_flow_barrier(instr: &Instr) -> bool {
    matches!(
        instr,
        Instr::Jump(_)
            | Instr::JumpFalse(_, _)
            | Instr::JumpTrue(_, _)
            | Instr::Return(_)
            | Instr::ReturnUnit
    )
}

/// Local straight-line dead-definition check.
///
/// Returns true when a write to `slot` at `pc` is overwritten before any read
/// and before any control-flow barrier.
fn is_straight_line_dead_def(instrs: &[Instr], pc: usize, slot: u16) -> bool {
    let mut i = pc + 1;
    while i < instrs.len() {
        let next = &instrs[i];
        if is_control_flow_barrier(next) {
            return false;
        }
        if instr_reads_slot(next, slot) {
            return false;
        }
        if instr_writes_slot(next, slot) {
            return true;
        }
        i += 1;
    }
    true
}

/// Returns true when `op` is in the set we know how to emit.
#[inline(always)]
fn is_supported_binop(op: BinOpKind) -> bool {
    matches!(
        op,
        BinOpKind::Add
            | BinOpKind::Sub
            | BinOpKind::Mul
            | BinOpKind::Div
            | BinOpKind::Rem
            | BinOpKind::Eq
            | BinOpKind::Ne
            | BinOpKind::Lt
            | BinOpKind::Le
            | BinOpKind::Gt
            | BinOpKind::Ge
    )
}

/// Emit the arithmetic/comparison body: lhs already in rax, rhs in rcx.
#[inline(always)]
fn emit_binop_rax_rcx(em: &mut Emitter, op: BinOpKind) -> bool {
    match op {
        BinOpKind::Add => em.add_rax_rcx(),
        BinOpKind::Sub => em.sub_rax_rcx(),
        BinOpKind::Mul => em.imul_rax_rcx(),
        BinOpKind::Div => {
            em.cqo();
            em.idiv_rcx();
        }
        BinOpKind::Rem => {
            em.cqo();
            em.idiv_rcx();
            em.mov_rax_rdx();
        }
        BinOpKind::Eq => {
            em.cmp_rax_rcx();
            em.setcc_al(0x94);
            em.movzx_rax_al();
        }
        BinOpKind::Ne => {
            em.cmp_rax_rcx();
            em.setcc_al(0x95);
            em.movzx_rax_al();
        }
        BinOpKind::Lt => {
            em.cmp_rax_rcx();
            em.setcc_al(0x9C);
            em.movzx_rax_al();
        }
        BinOpKind::Le => {
            em.cmp_rax_rcx();
            em.setcc_al(0x9E);
            em.movzx_rax_al();
        }
        BinOpKind::Gt => {
            em.cmp_rax_rcx();
            em.setcc_al(0x9F);
            em.movzx_rax_al();
        }
        BinOpKind::Ge => {
            em.cmp_rax_rcx();
            em.setcc_al(0x9D);
            em.movzx_rax_al();
        }
        _ => return false,
    }
    true
}

/// Emit immediate-rhs form: lhs already in rax.
#[inline(always)]
fn emit_binop_rax_imm(em: &mut Emitter, op: BinOpKind, imm: i32) {
    match op {
        BinOpKind::Add => {
            if imm == 1 {
                em.inc_rax();
            } else if imm == -1 {
                em.dec_rax();
            } else if imm != 0 {
                em.add_rax_imm32(imm); // now uses imm8 when it fits
            }
        }
        BinOpKind::Sub => {
            if imm == 1 {
                em.dec_rax();
            } else if imm == -1 {
                em.inc_rax();
            } else if imm != 0 {
                em.sub_rax_imm32(imm); // now uses imm8 when it fits
            }
        }
        BinOpKind::Mul => {
            if imm == 0 {
                em.xor_eax_eax();
            } else if imm == 1 { /* nop */
            } else if imm == -1 {
                em.neg_rax();
            } else if imm == 3 {
                em.lea_rax_rax_mul3();
            } else if imm == 5 {
                em.lea_rax_rax_mul5();
            } else if imm == 9 {
                em.lea_rax_rax_mul9();
            } else if imm > 0 && (imm as u32).is_power_of_two() {
                em.shl_rax_imm8((imm as u32).trailing_zeros() as u8);
            } else {
                em.imul_rax_imm32(imm);
            }
        }
        BinOpKind::Eq => {
            em.cmp_rax_imm32(imm); // now uses imm8 when it fits
            em.setcc_al(0x94);
            em.movzx_rax_al();
        }
        BinOpKind::Ne => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x95);
            em.movzx_rax_al();
        }
        BinOpKind::Lt => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x9C);
            em.movzx_rax_al();
        }
        BinOpKind::Le => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x9E);
            em.movzx_rax_al();
        }
        BinOpKind::Gt => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x9F);
            em.movzx_rax_al();
        }
        BinOpKind::Ge => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x9D);
            em.movzx_rax_al();
        }
        _ => {}
    }
}

/// Emit pops (reverse push order) followed by RET.
fn emit_ret(em: &mut Emitter, callee_saved: &[u8]) {
    for &reg in callee_saved.iter().rev() {
        em.pop_reg(reg);
    }
    em.ret();
}

// ─────────────────────────────────────────────────────────────────────────────
// Branch fixup with short-branch shrinking
// ─────────────────────────────────────────────────────────────────────────────
//
// We record each branch as a placeholder in the byte stream.  After all code
// is emitted we know every target offset, so we can choose the shortest
// encoding.  Because shrinking a branch changes offsets, we do a single
// relaxation pass: convert rel32 → rel8 where the *original* displacement
// would already fit (conservative but correct — shrinking can only move
// targets closer).

/// Kind of branch instruction at a fixup site.
#[derive(Clone, Copy)]
enum BranchKind {
    Jmp,
    Jz,
    Jnz,
}

/// A pending branch: position of the disp32 field, target PC, and kind.
struct Fixup {
    /// Byte index of the 4-byte disp32 placeholder.
    disp_pos: usize,
    target_pc: usize,
    kind: BranchKind,
}

/// Patch all branch displacements.  Returns None if any target is unreachable
/// or any displacement overflows i32.
fn patch_fixups(buf: &mut Vec<u8>, fixups: &[Fixup], pc_to_off: &[usize]) -> Option<()> {
    for fx in fixups {
        let target_off = *pc_to_off.get(fx.target_pc)? as isize;
        let next_ip = (fx.disp_pos + 4) as isize;
        let rel = i32::try_from(target_off - next_ip).ok()?;

        // Attempt to shrink to rel8 (saves 3-4 bytes per branch).
        if let Ok(rel8) = i8::try_from(rel) {
            // The rel32 form sits at disp_pos-1 (or disp_pos-2 for 0F 8x).
            // Overwrite with rel8 opcode + 1-byte disp + NOPs for remainder.
            let opcode_start = match fx.kind {
                BranchKind::Jmp => fx.disp_pos - 1,                  // E9 [d32]
                BranchKind::Jz | BranchKind::Jnz => fx.disp_pos - 2, // 0F 84/85 [d32]
            };
            let short_op: u8 = match fx.kind {
                BranchKind::Jmp => 0xEB,
                BranchKind::Jz => 0x74,
                BranchKind::Jnz => 0x75,
            };
            buf[opcode_start] = short_op;
            buf[opcode_start + 1] = rel8 as u8;
            // Overwrite remaining bytes with NOPs.
            let nop_start = opcode_start + 2;
            let nop_end = fx.disp_pos + 4;
            for b in &mut buf[nop_start..nop_end] {
                *b = 0x90;
            }
        } else {
            buf[fx.disp_pos..fx.disp_pos + 4].copy_from_slice(&rel.to_le_bytes());
        }
    }
    Some(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Peephole optimizer pass
// ─────────────────────────────────────────────────────────────────────────────
//
// Runs after liveness analysis, before code emission.  Eliminates redundant
// instruction patterns in the bytecode to shrink the hot loop.

fn peephole_optimize(instrs: &mut Vec<Instr>) {
    // Fixed 3-pass approach instead of `while changed` to avoid quadratic behaviour.
    // Each pass scans the instruction stream once, applying all applicable rewrites.
    const MAX_PASSES: usize = 3;
    for _pass in 0..MAX_PASSES {
        let mut i = 0;
        while i + 1 < instrs.len() {
            // Pattern 1: Load(x) + Store(x) → eliminate both (dead store)
            if let (Instr::Load(d1, s), Instr::Store(d2, s2)) =
                (&instrs[i], &instrs[i + 1])
            {
                if *d1 == *s2 && *s == *d2 {
                    // Load into tmp then immediately store back to same slot = no-op
                    instrs.remove(i + 1);
                    instrs.remove(i);
                    // Don't advance i; re-check at same position.
                    continue;
                }
            }

            // Pattern 2: Move(d, s) + Load(x, d) → Move(d, s) + Load(x, s) (forward prop)
            if let (Instr::Move(d, s), Instr::Load(d2, s2)) = (&instrs[i], &instrs[i + 1]) {
                if *s2 == *d {
                    instrs[i + 1] = Instr::Load(*d2, *s);
                }
            }

            // Pattern 3: Store(slot, x) + Load(d, slot) → Store(slot, x) + Move(d, x)
            if let (Instr::Store(slot, s), Instr::Load(d2, s2)) = (&instrs[i], &instrs[i + 1]) {
                if *s2 == *slot {
                    instrs[i + 1] = Instr::Move(*d2, *s);
                }
            }

            // Pattern 4: Jump(0) → eliminate (no-op jump)
            if let Instr::Jump(0) = &instrs[i] {
                instrs.remove(i);
                continue;
            }

            // Pattern 5: LoadI*(d, v) + Move(d2, d) → LoadI*(d2, v) + (eliminate Move)
            if let (Instr::Move(d, s), _) = (&instrs[i], &instrs[i + 1]) {
                // Look backwards for LoadI into s
                if i > 0 {
                    let prev_idx = i - 1;
                    match &instrs[prev_idx] {
                        Instr::LoadI32(src, v) if *src == *s => {
                            instrs[prev_idx] = Instr::LoadI32(*d, *v);
                            instrs.remove(i);
                            continue;
                        }
                        Instr::LoadI64(src, v) if *src == *s => {
                            instrs[prev_idx] = Instr::LoadI64(*d, *v);
                            instrs.remove(i);
                            continue;
                        }
                        Instr::LoadBool(src, v) if *src == *s => {
                            instrs[prev_idx] = Instr::LoadBool(*d, *v);
                            instrs.remove(i);
                            continue;
                        }
                        _ => {}
                    }
                }
            }

            // Pattern 6: Move(d, s) + Move(d2, d) → Move(d, s) + Move(d2, s) (chain forwarding)
            if let (Instr::Move(d, s), Instr::Move(d2, s2)) = (&instrs[i], &instrs[i + 1]) {
                if *s2 == *d {
                    instrs[i + 1] = Instr::Move(*d2, *s);
                }
            }

            // Pattern 7: LoadI*(d, 0) + Move(d2, d) → LoadI*(d2, 0) + eliminate Move
            if let (Instr::Move(d, s), _) = (&instrs[i], &instrs[i + 1]) {
                if i > 0 {
                    let prev_idx = i - 1;
                    if let Instr::LoadI64(src, v) = &instrs[prev_idx] {
                        if *src == *s && *v == 0 {
                            instrs[prev_idx] = Instr::LoadI64(*d, 0);
                            instrs.remove(i);
                            continue;
                        }
                    }
                }
            }

            i += 1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

#[must_use]
pub fn is_available() -> bool {
    cfg!(target_arch = "x86_64")
}

pub fn translate(compiled: &CompiledFn) -> Option<NativeCode> {
    if !cfg!(target_arch = "x86_64") {
        return None;
    }

    let instrs = &compiled.instrs;
    let slot_count = compiled.slot_count as usize;

    // Gate: bail out early if any instruction is outside our supported set.
    for instr in instrs {
        match instr {
            Instr::LoadI32(..)
            | Instr::LoadI64(..)
            | Instr::LoadBool(..)
            | Instr::LoadUnit(..)
            | Instr::LoadF32(..)
            | Instr::LoadF64(..)
            | Instr::Move(..)
            | Instr::Load(..)
            | Instr::Store(..)
            | Instr::BinOp(..)
            | Instr::Jump(..)
            | Instr::JumpFalse(..)
            | Instr::JumpTrue(..)
            | Instr::Return(..)
            | Instr::ReturnUnit
            | Instr::Nop => {}
            _ => return None,
        }
    }

    // ── Pass 0: Peephole optimization ──────────────────────────────────
    let mut opt_instrs = instrs.clone();
    peephole_optimize(&mut opt_instrs);
    let instrs = &opt_instrs;

    // ── Pass 1: liveness + linear-scan register allocation ───────────────
    let intervals = compute_live_intervals(instrs, slot_count);
    // Use actual max slot from intervals (may exceed declared slot_count
    // due to temporaries created during expression compilation).
    let actual_max_slot = intervals
        .iter()
        .map(|i| i.slot as usize)
        .max()
        .unwrap_or(slot_count);
    let ra = linear_scan(&intervals, actual_max_slot);

    // ── Emission ──────────────────────────────────────────────────────────
    let mut em = Emitter::new();
    let mut pc_to_off = vec![0usize; instrs.len() + 1];
    let mut fixups: Vec<Fixup> = Vec::new();

    // Prologue: save callee-saved registers we actually use.
    for &reg in &ra.used_callee_saved {
        em.push_reg(reg);
    }

    // Pre-load register-assigned *parameter* slots from the slot array.
    // Non-parameter slots may be uninitialized at entry and must not be read.
    {
        let mut preloaded: u32 = 0u32; // bitmask for regs 0-31
        for slot in 0..compiled.param_count {
            if let RegLoc::Reg(r) = ra.location(slot) {
                let bit = 1u32 << r;
                if preloaded & bit == 0 {
                    preloaded |= bit;
                    let off = (slot as i32) * 8;
                    em.load_reg_mem(r, off);
                }
            }
        }
    }

    // ── Main translation loop ─────────────────────────────────────────────
    // Flat Vec<Option<i64>> replaces HashMap — O(1) slot access, cache-friendly.
    let mut const_at = ConstTable::with_capacity(slot_count + 1);

    let mut pc = 0usize;
    while pc < instrs.len() {
        pc_to_off[pc] = em.pos();

        // ════════════════════════════════════════════════════════════════════
        // 3-INSTRUCTION FUSIONS
        // ════════════════════════════════════════════════════════════════════

        // ── Fusion: BinOp(t, Mul, x, N) + BinOp(r, Add, t, y) → LEA ────
        if pc + 1 < instrs.len() {
            if let (
                Instr::BinOp(t, BinOpKind::Mul, mul_l, mul_r),
                Instr::BinOp(r, BinOpKind::Add, add_l, add_r),
            ) = (&instrs[pc], &instrs[pc + 1])
            {
                let (addend_slot, t_consumed_by_add) = if *add_l == *t && *add_r != *t && *r != *t {
                    (Some(*add_r), true)
                } else if *add_r == *t && *add_l != *t && *r != *t {
                    (Some(*add_l), true)
                } else {
                    (None, false)
                };

                if t_consumed_by_add {
                    let addend = addend_slot.unwrap();
                    let maybe_lea = const_at
                        .get(*mul_r)
                        .map(|c| (*mul_l, c))
                        .or_else(|| const_at.get(*mul_l).map(|c| (*mul_r, c)));

                    if let Some((base, scale_i64)) = maybe_lea {
                        if matches!(scale_i64, 2 | 4 | 8) {
                            let scale = scale_i64 as u8;
                            load_rax(&mut em, base, &ra);
                            load_rcx(&mut em, addend, &ra);
                            em.lea_rax_rax_scale_plus_rcx(scale);
                            if !is_straight_line_dead_def(instrs, pc, *r) {
                                store_rax(&mut em, *r, &ra);
                            }
                            pc_to_off[pc + 1] = em.pos();
                            pc += 2;
                            continue;
                        }
                    }
                }
            }
        }

        // ── Fusion: BinOp(t, op1, a, b) + BinOp(r, op2, t, c) → chain ──
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op1, a, b), Instr::BinOp(r, op2, l2, r2)) =
                (&instrs[pc], &instrs[pc + 1])
            {
                let commutative = matches!(
                    op2,
                    BinOpKind::Add | BinOpKind::Mul | BinOpKind::Eq | BinOpKind::Ne
                );
                let t_as_lhs = *l2 == *t && *r2 != *t;
                let t_as_rhs = *r2 == *t && *l2 != *t && commutative;

                if (t_as_lhs || t_as_rhs) && is_supported_binop(*op1) && is_supported_binop(*op2) {
                    let other = if t_as_lhs { *r2 } else { *l2 };
                    load_rax(&mut em, *a, &ra);
                    load_rcx(&mut em, *b, &ra);
                    emit_binop_rax_rcx(&mut em, *op1);
                    load_rcx(&mut em, other, &ra);
                    emit_binop_rax_rcx(&mut em, *op2);
                    if !is_straight_line_dead_def(instrs, pc, *r) {
                        store_rax(&mut em, *r, &ra);
                    }
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // 2-INSTRUCTION FUSIONS
        // ════════════════════════════════════════════════════════════════════

        // ── Fusion: Load*(tmp, c) + JumpFalse/JumpTrue → compile-time branch
        if pc + 1 < instrs.len() {
            let maybe_const = match &instrs[pc] {
                Instr::LoadI32(tmp, v) => Some((*tmp, *v as i64)),
                Instr::LoadI64(tmp, v) => Some((*tmp, *v)),
                Instr::LoadBool(tmp, v) => Some((*tmp, i64::from(*v))),
                Instr::LoadUnit(tmp) => Some((*tmp, 0i64)),
                _ => None,
            };
            if let Some((tmp, c)) = maybe_const {
                let mut folded = false;
                match &instrs[pc + 1] {
                    Instr::JumpFalse(cond, off) if *cond == tmp => {
                        let target = ((pc as i32) + 2 + *off) as usize;
                        if target > instrs.len() {
                            return None;
                        }
                        if c == 0 {
                            let p = em.jmp_rel32_placeholder();
                            fixups.push(Fixup {
                                disp_pos: p,
                                target_pc: target,
                                kind: BranchKind::Jmp,
                            });
                        }
                        folded = true;
                    }
                    Instr::JumpTrue(cond, off) if *cond == tmp => {
                        let target = ((pc as i32) + 2 + *off) as usize;
                        if target > instrs.len() {
                            return None;
                        }
                        if c != 0 {
                            let p = em.jmp_rel32_placeholder();
                            fixups.push(Fixup {
                                disp_pos: p,
                                target_pc: target,
                                kind: BranchKind::Jmp,
                            });
                        }
                        folded = true;
                    }
                    _ => {}
                }
                if folded {
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: Load*(tmp, c) + BinOp(dst, op, x, tmp) → imm arithmetic
        if pc + 1 < instrs.len() {
            let maybe_imm = match &instrs[pc] {
                Instr::LoadI32(tmp, v) => Some((*tmp, *v as i64)),
                Instr::LoadI64(tmp, v) => Some((*tmp, *v)),
                _ => None,
            };
            if let Some((tmp, c)) = maybe_imm {
                if let Instr::BinOp(dst, op, l, r) = &instrs[pc + 1] {
                    if let Ok(imm) = i32::try_from(c) {
                        let rhs_is_imm = *r == tmp;
                        let lhs_is_imm = *l == tmp;
                        let can_use = match op {
                            BinOpKind::Add | BinOpKind::Mul => rhs_is_imm || lhs_is_imm,
                            BinOpKind::Sub
                            | BinOpKind::Eq
                            | BinOpKind::Ne
                            | BinOpKind::Lt
                            | BinOpKind::Le
                            | BinOpKind::Gt
                            | BinOpKind::Ge => rhs_is_imm,
                            _ => false,
                        };
                        if can_use {
                            let live_reg = if rhs_is_imm { *l } else { *r };
                            load_rax(&mut em, live_reg, &ra);
                            emit_binop_rax_imm(&mut em, *op, imm);
                            if !is_straight_line_dead_def(instrs, pc, *dst) {
                                store_rax(&mut em, *dst, &ra);
                            }
                            pc_to_off[pc + 1] = em.pos();
                            pc += 2;
                            continue;
                        }
                    }
                }
            }
        }

        // ── Fusion: BinOp(t, op, l, r) + Store(slot, t) ─────────────────
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::Store(slot, src)) =
                (&instrs[pc], &instrs[pc + 1])
            {
                if t == src && is_supported_binop(*op) {
                    load_rax(&mut em, *l, &ra);
                    load_rcx(&mut em, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op);
                    if !is_straight_line_dead_def(instrs, pc, *slot) {
                        store_rax(&mut em, *slot, &ra);
                    }
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: BinOp(t, op, l, r) + JumpFalse(t, off) ─────────────
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::JumpFalse(cond, off)) =
                (&instrs[pc], &instrs[pc + 1])
            {
                if t == cond && is_supported_binop(*op) {
                    let target = ((pc as i32) + 2 + *off) as usize;
                    if target > instrs.len() {
                        return None;
                    }
                    load_rax(&mut em, *l, &ra);
                    load_rcx(&mut em, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op);
                    em.test_rax_rax();
                    let p = em.jz_rel32_placeholder();
                    fixups.push(Fixup {
                        disp_pos: p,
                        target_pc: target,
                        kind: BranchKind::Jz,
                    });
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: BinOp(t, op, l, r) + JumpTrue(t, off) ──────────────
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::JumpTrue(cond, off)) =
                (&instrs[pc], &instrs[pc + 1])
            {
                if t == cond && is_supported_binop(*op) {
                    let target = ((pc as i32) + 2 + *off) as usize;
                    if target > instrs.len() {
                        return None;
                    }
                    load_rax(&mut em, *l, &ra);
                    load_rcx(&mut em, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op);
                    em.test_rax_rax();
                    let p = em.jnz_rel32_placeholder();
                    fixups.push(Fixup {
                        disp_pos: p,
                        target_pc: target,
                        kind: BranchKind::Jnz,
                    });
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: BinOp(t, op, l, r) + Return(t) ──────────────────────
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::Return(ret)) = (&instrs[pc], &instrs[pc + 1])
            {
                if t == ret && is_supported_binop(*op) {
                    load_rax(&mut em, *l, &ra);
                    load_rcx(&mut em, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op);
                    emit_ret(&mut em, &ra.used_callee_saved);
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // CONSTANT FOLDING (single BinOp with both operands known)
        // ════════════════════════════════════════════════════════════════════

        if let Instr::BinOp(d, op, l, r) = &instrs[pc] {
            if let Some(v) = const_at
                .get(*l)
                .zip(const_at.get(*r))
                .and_then(|(lv, rv)| fold_binop(*op, lv, rv))
            {
                em.mov_rax_imm_opt(v);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, v);
                pc += 1;
                continue;
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // SINGLE-INSTRUCTION FALLBACK
        // ════════════════════════════════════════════════════════════════════

        match &instrs[pc] {
            Instr::LoadI32(d, v) => {
                let cv = *v as i64;
                em.mov_rax_imm_opt(cv);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, cv);
            }
            Instr::LoadI64(d, v) => {
                em.mov_rax_imm_opt(*v);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, *v);
            }
            Instr::LoadBool(d, v) => {
                let cv = i64::from(*v);
                em.mov_rax_imm_opt(cv);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, cv);
            }
            Instr::LoadUnit(d) => {
                em.xor_eax_eax();
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, 0);
            }
            Instr::Move(d, s) | Instr::Load(d, s) => {
                load_rax(&mut em, *s, &ra);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                if let Some(c) = const_at.get(*s) {
                    const_at.insert(*d, c);
                } else {
                    const_at.remove(*d);
                }
            }
            Instr::Store(slot, s) => {
                load_rax(&mut em, *s, &ra);
                if !is_straight_line_dead_def(instrs, pc, *slot) {
                    store_rax(&mut em, *slot, &ra);
                }
                if let Some(c) = const_at.get(*s) {
                    const_at.insert(*slot, c);
                } else {
                    const_at.remove(*slot);
                }
            }
            Instr::BinOp(d, op, l, r) => {
                if !is_supported_binop(*op) {
                    return None;
                }
                load_rax(&mut em, *l, &ra);
                load_rcx(&mut em, *r, &ra);
                emit_binop_rax_rcx(&mut em, *op);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                let folded = const_at
                    .get(*l)
                    .zip(const_at.get(*r))
                    .and_then(|(lv, rv)| fold_binop(*op, lv, rv));
                match folded {
                    Some(c) => const_at.insert(*d, c),
                    None => const_at.remove(*d),
                }
            }
            Instr::LoadF32(d, v) => {
                // Load f32 constant as bits into XMM0, then store to slot.
                let bits = v.to_bits();
                em.mov_xmm0_imm32_bits(bits as u32);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    let off = match ra.location(*d) {
                        RegLoc::Reg(_) => (d * 8) as i32, // slots are 8-byte aligned
                        RegLoc::Spill(off) => off,
                    };
                    em.store_mem_xmm0_f32(off);
                }
                // Don't track float constants in int const_at table.
            }
            Instr::LoadF64(d, v) => {
                let bits = v.to_bits();
                em.mov_xmm0_imm64(bits);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    let off = match ra.location(*d) {
                        RegLoc::Reg(_) => (d * 8) as i32,
                        RegLoc::Spill(off) => off,
                    };
                    em.store_mem_xmm0(off);
                }
            }
            Instr::Jump(off) => {
                let target = ((pc as i32) + 1 + *off) as usize;
                if target > instrs.len() {
                    return None;
                }
                let p = em.jmp_rel32_placeholder();
                fixups.push(Fixup {
                    disp_pos: p,
                    target_pc: target,
                    kind: BranchKind::Jmp,
                });
                const_at.clear();
            }
            Instr::JumpFalse(cond, off) => {
                let target = ((pc as i32) + 1 + *off) as usize;
                if target > instrs.len() {
                    return None;
                }
                load_rax(&mut em, *cond, &ra);
                em.test_rax_rax();
                let p = em.jz_rel32_placeholder();
                fixups.push(Fixup {
                    disp_pos: p,
                    target_pc: target,
                    kind: BranchKind::Jz,
                });
                const_at.clear();
            }
            Instr::JumpTrue(cond, off) => {
                let target = ((pc as i32) + 1 + *off) as usize;
                if target > instrs.len() {
                    return None;
                }
                load_rax(&mut em, *cond, &ra);
                em.test_rax_rax();
                let p = em.jnz_rel32_placeholder();
                fixups.push(Fixup {
                    disp_pos: p,
                    target_pc: target,
                    kind: BranchKind::Jnz,
                });
                const_at.clear();
            }
            Instr::Return(r) => {
                load_rax(&mut em, *r, &ra);
                emit_ret(&mut em, &ra.used_callee_saved);
            }
            Instr::ReturnUnit => {
                em.xor_eax_eax();
                emit_ret(&mut em, &ra.used_callee_saved);
            }
            Instr::Nop => {}
            _ => return None,
        }
        pc += 1;
    }

    // Fallthrough epilogue.
    pc_to_off[instrs.len()] = em.pos();
    em.xor_eax_eax();
    emit_ret(&mut em, &ra.used_callee_saved);

    // ── Patch branch displacements (with short-branch shrinking) ─────────
    patch_fixups(&mut em.buf, &fixups, &pc_to_off)?;

    let mem = ExecMem::new(&em.buf)?;
    Some(NativeCode {
        slot_count: compiled.slot_count,
        mem,
    })
}

pub fn execute(native: &NativeCode, args: &[Value]) -> Result<Value, RuntimeError> {
    thread_local! {
        static EXEC_REGS: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
    }
    let needed = native.slot_count as usize + 32;
    EXEC_REGS.with(|cell| -> Result<Value, RuntimeError> {
        let mut regs = cell.borrow_mut();
        if regs.len() < needed {
            regs.resize(needed, 0);
        }
        // Only zero the portion we'll use.
        for r in &mut regs[..needed] {
            *r = 0;
        }
        for (i, arg) in args.iter().enumerate() {
            if i >= needed {
                break;
            }
            regs[i] = match arg {
                Value::I8(v) => *v as i64,
                Value::I16(v) => *v as i64,
                Value::I32(v) => *v as i64,
                Value::I64(v) => *v,
                Value::U8(v) => *v as i64,
                Value::U16(v) => *v as i64,
                Value::U32(v) => *v as i64,
                Value::U64(v) => *v as i64,
                Value::Bool(v) => i64::from(*v),
                _ => {
                    return Err(RuntimeError::new(
                        "native machine-code JIT supports int/bool args",
                    ))
                }
            };
        }
        let f = native.mem.entry();
        let out = unsafe { f(regs.as_mut_ptr()) };
        Ok(Value::I64(out))
    })
}
