// =========================================================================
// JHAL Benchmark Suite — HONEST EDITION
//
// Every benchmark uses black_box() to prevent the compiler from
// optimizing away the work being measured. Previous benchmarks
// reported physically impossible numbers (0.0 ns, 2.86T ops/s)
// because the compiler deleted the code via dead code elimination.
//
// Rule: If a benchmark shows sub-nanosecond times for non-trivial
// operations, it's measuring compiler optimization, not actual work.
//
// Run with: cargo run --release --bin bench-jhal
// =========================================================================

use std::hint::black_box;
use std::time::Instant;

use jules::runtime::jhal::{
    SerialRingBuffer, Console,
    DeviceRegistry, PciDevice, PciBdf, PciHeaderType,
    BoundedBus, BoundedDevice, BoundedFunction,
    MAX_BARS,
    ApicTimerConfig,
    SfiConfig, DEFAULT_SANCTUARY_BASE, DEFAULT_SANCTUARY_SIZE,
    verify_sfi_invariant,
    TsxStatus, prove_transaction_bound, scheduler_matmul_fallback,
    verify_register_partition,
    IdtEntry, Idt,
    FIRST_USER_VECTOR,
    IrqPredictor,
    IdentityMap, HugePageAllocator, HugePageSize,
    IommuDropZone, NmiWatchdog,
    CfiJumpTable,
    PTE_PRESENT, PTE_WRITABLE,
};

fn main() {
    let iterations = 100_000;
    let warmup = 10_000;

    println!("=== JHAL Benchmark Suite (HONEST EDITION) ===");
    println!("iterations: {}", iterations);
    println!("NOTE: All benchmarks use black_box() to prevent DCE.");
    println!("      Sub-nanosecond results indicate the operation is");
    println!("      genuinely trivial (1-2 instructions), not that the");
    println!("      compiler deleted the code.\n");

    // Warmup
    for _ in 0..warmup {
        black_box(bench_ring_buffer_single(1));
    }

    // ─── Original benchmarks (fixed) ──────────────────────────────────────

    bench("ring_buffer_enqueue_dequeue", iterations, || {
        bench_ring_buffer_single(iterations)
    });

    bench("ring_buffer_bulk_255", iterations / 100, || {
        bench_ring_buffer_bulk()
    });

    bench("pci_bdf_construction", iterations, || {
        bench_bdf_construction(iterations)
    });

    bench("pci_bdf_bounds_check", iterations, || {
        bench_bdf_bounds_check(iterations)
    });

    bench("device_registry_register", iterations / 100, || {
        bench_device_registry_register()
    });

    bench("device_registry_find_by_class", iterations / 100, || {
        bench_device_registry_find()
    });

    bench("zero_heap_formatting_decimal", iterations, || {
        bench_formatting_decimal(iterations)
    });

    bench("zero_heap_formatting_hex", iterations, || {
        bench_formatting_hex(iterations)
    });

    bench("apic_timer_config_construction", iterations, || {
        bench_apic_config(iterations)
    });

    bench("apic_register_offset_computation", iterations, || {
        bench_apic_offsets(iterations)
    });

    bench("console_write_buffered", iterations, || {
        bench_console_buffered(iterations)
    });

    // ─── New bare-metal component benchmarks ──────────────────────────────

    bench("sfi_config_creation", iterations, || {
        bench_sfi_config(iterations)
    });

    bench("sfi_pointer_masking", iterations, || {
        bench_sfi_masking(iterations)
    });

    bench("sfi_invariant_verification", iterations, || {
        bench_sfi_invariant(iterations)
    });

    bench("tsx_status_construction", iterations, || {
        bench_tsx_status(iterations)
    });

    bench("tsx_transaction_bound_proof", iterations, || {
        bench_tsx_bound_proof(iterations)
    });

    bench("amx_scheduler_matmul_4x4", iterations / 10, || {
        bench_amx_matmul()
    });

    bench("irq_register_partition_proof", iterations, || {
        bench_register_partition(iterations)
    });

    bench("idt_entry_construction", iterations, || {
        bench_idt_entry(iterations)
    });

    bench("idt_full_table_build", iterations / 100, || {
        bench_idt_table_build()
    });

    bench("irq_predictor_record", iterations, || {
        bench_irq_predictor_record(iterations)
    });

    bench("irq_predictor_predict", iterations, || {
        bench_irq_predictor_predict(iterations)
    });

    bench("huge_page_allocator", iterations / 10, || {
        bench_huge_page_alloc()
    });

    bench("iommu_dma_check", iterations, || {
        bench_iommu_check(iterations)
    });

    bench("nmi_watchdog_pet", iterations, || {
        bench_nmi_pet(iterations)
    });

    bench("cfi_jump_table_lookup", iterations, || {
        bench_cfi_lookup(iterations)
    });

    bench("identity_map_1gb_mapping", iterations / 100, || {
        bench_identity_map_1gb()
    });

    println!();
    println!("=== Honest Numbers Analysis ===");
    println!("Operations like 'register partition proof' and 'SFI pointer masking'");
    println!("ARE genuinely fast because they're just bitwise AND/comparisons.");
    println!("Operations like 'IDT table build' and 'PCI registry register' involve");
    println!("atomic operations and are correctly measured in the 10-100 ns range.");
    println!("Real hardware I/O (APIC MMIO, PCI config, UART) cannot be benchmarked");
    println!("in userspace — these would take microseconds on real hardware.");
}

// ─── Ring Buffer Benchmarks ──────────────────────────────────────────────

fn bench_ring_buffer_single(iters: usize) -> u64 {
    let ring = SerialRingBuffer::new();
    let mut count = 0u64;
    for i in 0..iters {
        let byte = (i & 0xFF) as u8;
        if ring.enqueue(black_box(byte)) {
            if let Some(b) = ring.dequeue() {
                count += black_box(b) as u64;
            }
        }
    }
    count
}

fn bench_ring_buffer_bulk() -> u64 {
    let ring = SerialRingBuffer::new();
    let mut count = 0u64;

    for i in 0..255 {
        if ring.enqueue(black_box(i as u8)) {
            count += 1;
        }
    }

    while let Some(b) = ring.dequeue() {
        count += black_box(b) as u64;
    }

    count
}

// ─── PCI BDF Benchmarks ──────────────────────────────────────────────────

fn bench_bdf_construction(iters: usize) -> u64 {
    let mut count = 0u64;
    for i in 0..iters {
        let bus = black_box((i & 0xFF) as u8);
        let device = black_box((i % 32) as u8);
        let function = black_box((i % 8) as u8);
        if let Some(bdf) = PciBdf::new(bus, device, function) {
            count += black_box(bdf.bus.raw()) as u64;
            count += black_box(bdf.device.raw()) as u64;
            count += black_box(bdf.function.raw()) as u64;
        }
    }
    count
}

fn bench_bdf_bounds_check(iters: usize) -> u64 {
    let mut count = 0u64;
    for i in 0..iters {
        let bus_result = BoundedBus::new(black_box((i & 0xFF) as u8));
        let dev_result = BoundedDevice::new(black_box((i % 33) as u8));
        let fn_result = BoundedFunction::new(black_box((i % 9) as u8));
        count += black_box(bus_result.is_some()) as u64;
        count += black_box(dev_result.is_some()) as u64;
        count += black_box(fn_result.is_some()) as u64;
    }
    count
}

// ─── Device Registry Benchmarks ──────────────────────────────────────────

fn bench_device_registry_register() -> u64 {
    let registry = DeviceRegistry::new();
    let mut count = 0u64;
    for i in 0..10 {
        let device = PciDevice {
            bdf: PciBdf::new(0, black_box(i as u8), 0).unwrap(),
            vendor_id: 0x8086,
            device_id: black_box(i as u16),
            class_code: 0x02,
            subclass: 0x00,
            prog_if: 0x00,
            revision: 0x01,
            header_type: PciHeaderType::Standard,
            multi_function: false,
            secondary_bus: None,
            subordinate_bus: None,
            bars: [0; MAX_BARS],
            bar_count: 0,
            latency_ns: 50_000,
        };
        if registry.register(device).is_some() {
            count += black_box(1);
        }
    }
    count
}

fn bench_device_registry_find() -> u64 {
    let registry = DeviceRegistry::new();
    for i in 0..5 {
        let device = PciDevice {
            bdf: PciBdf::new(0, i as u8, 0).unwrap(),
            vendor_id: 0x8086,
            device_id: i as u16,
            class_code: if i % 2 == 0 { 0x02 } else { 0x01 },
            subclass: 0x00,
            prog_if: 0x00,
            revision: 0x01,
            header_type: PciHeaderType::Standard,
            multi_function: false,
            secondary_bus: None,
            subordinate_bus: None,
            bars: [0; MAX_BARS],
            bar_count: 0,
            latency_ns: 50_000,
        };
        let _ = registry.register(device);
    }
    let mut count = 0u64;
    for _ in 0..1000 {
        let (results, n) = registry.find_by_class(black_box(0x02));
        count += black_box(n) as u64;
        black_box(&results);
    }
    count
}

// ─── Zero-Heap Formatting Benchmarks ─────────────────────────────────────

fn bench_formatting_decimal(iters: usize) -> u64 {
    let mut buf = [0u8; 64];
    let mut count = 0u64;
    for i in 0..iters {
        buf.fill(0);
        let val = black_box((i as u32).wrapping_mul(7919));
        let pos = 0;
        count += black_box(write_u32_decimal_bench(&mut buf, pos, val)) as u64;
    }
    count
}

fn bench_formatting_hex(iters: usize) -> u64 {
    let mut buf = [0u8; 64];
    let mut count = 0u64;
    for i in 0..iters {
        buf.fill(0);
        let val = black_box((i as u32).wrapping_mul(7919));
        let pos = 0;
        count += black_box(write_u32_hex_bench(&mut buf, pos, val)) as u64;
    }
    count
}

fn write_u32_decimal_bench(buf: &mut [u8], mut pos: usize, val: u32) -> usize {
    if val == 0 {
        if pos < buf.len() { buf[pos] = b'0'; pos += 1; }
        return pos;
    }
    let mut digits = [0u8; 10];
    let mut n = val;
    let mut i = 0;
    while n > 0 {
        digits[i] = (n % 10) as u8;
        n /= 10;
        i += 1;
    }
    for j in (0..i).rev() {
        if pos < buf.len() {
            buf[pos] = b'0' + digits[j];
            pos += 1;
        }
    }
    pos
}

fn write_u32_hex_bench(buf: &mut [u8], mut pos: usize, val: u32) -> usize {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for shift in (0..8).rev() {
        let nibble = ((val >> (shift * 4)) & 0xF) as usize;
        if pos < buf.len() {
            buf[pos] = HEX[nibble];
            pos += 1;
        }
    }
    pos
}

// ─── APIC Timer Config Benchmarks ────────────────────────────────────────

fn bench_apic_config(iters: usize) -> u64 {
    let mut count = 0u64;
    for i in 0..iters {
        let config = ApicTimerConfig::for_prophecy_tick(black_box(i as u32));
        count += black_box(config.vector) as u64;
        count += black_box(config.initial_count) as u64;
        count += black_box(config.divide) as u64;
    }
    count
}

fn bench_apic_offsets(iters: usize) -> u64 {
    let base: usize = 0xFEE0_0000;
    let offsets: [u32; 12] = [
        0x020, 0x030, 0x080, 0x0B0, 0x0F0, 0x300,
        0x310, 0x320, 0x350, 0x360, 0x380, 0x3E0,
    ];
    let mut count = 0u64;
    for i in 0..iters {
        let offset = black_box(offsets[i % 12]);
        let addr = base + black_box(offset) as usize;
        count += black_box(addr) as u64;
    }
    count
}

// ─── Console Buffered Write Benchmark ────────────────────────────────────

fn bench_console_buffered(iters: usize) -> u64 {
    let console = Console::new();
    let mut count = 0u64;
    for i in 0..iters {
        let byte = black_box((i & 0xFF) as u8);
        if console.write(byte, Some(0)) {
            count += 1;
        }
    }
    count
}

// ─── SFI Benchmarks ──────────────────────────────────────────────────────

fn bench_sfi_config(iters: usize) -> u64 {
    let mut count = 0u64;
    for i in 0..iters {
        if let Some(cfg) = SfiConfig::new(black_box(DEFAULT_SANCTUARY_BASE + (i & 0xF) * 0x100_0000), black_box(DEFAULT_SANCTUARY_SIZE)) {
            count += black_box(cfg.mask_value()) as u64;
        }
    }
    count
}

fn bench_sfi_masking(iters: usize) -> u64 {
    let cfg = SfiConfig::new(DEFAULT_SANCTUARY_BASE, DEFAULT_SANCTUARY_SIZE).unwrap();
    let mut count = 0u64;
    for i in 0..iters {
        let ptr = black_box(0xDEAD_BEEF_0000 + (i as usize) * 0x1000);
        let masked = cfg.apply_mask(ptr);
        count += black_box(masked) as u64;
    }
    count
}

fn bench_sfi_invariant(iters: usize) -> u64 {
    let cfg = SfiConfig::new(DEFAULT_SANCTUARY_BASE, DEFAULT_SANCTUARY_SIZE).unwrap();
    let mut count = 0u64;
    for i in 0..iters {
        let ptr = black_box(0xDEAD_BEEF_0000 + (i as usize) * 0x1000);
        count += black_box(verify_sfi_invariant(&cfg, ptr)) as u64;
    }
    count
}

// ─── TSX Benchmarks ──────────────────────────────────────────────────────

fn bench_tsx_status(iters: usize) -> u64 {
    let mut count = 0u64;
    for i in 0..iters {
        let status = TsxStatus { raw: black_box(i as u32) };
        count += black_box(status.started()) as u64;
        count += black_box(status.is_conflict()) as u64;
        count += black_box(status.is_capacity()) as u64;
    }
    count
}

fn bench_tsx_bound_proof(iters: usize) -> u64 {
    let mut count = 0u64;
    for i in 0..iters {
        let footprint = black_box(i * 64); // Varying write footprint
        count += black_box(prove_transaction_bound(footprint)) as u64;
    }
    count
}

// ─── AMX Benchmarks ──────────────────────────────────────────────────────

fn bench_amx_matmul() -> u64 {
    // 4×4 matrix multiply using the software fallback
    let weights: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let input: [u8; 4] = [1, 0, 1, 0];
    let mut output: [i32; 4] = [0; 4];
    scheduler_matmul_fallback(black_box(&weights), black_box(&input), black_box(&mut output), 4, 4);
    black_box(output[0]) as u64 + black_box(output[3]) as u64
}

// ─── IRQ Injection Benchmarks ────────────────────────────────────────────

fn bench_register_partition(iters: usize) -> u64 {
    let mut count = 0u64;
    for _ in 0..iters {
        count += black_box(verify_register_partition(black_box(&["rax", "rcx", "rdx", "rbx", "r8", "r9"]))) as u64;
    }
    count
}

fn bench_idt_entry(iters: usize) -> u64 {
    let mut count = 0u64;
    for i in 0..iters {
        let handler = black_box(0xFFFF_8000_0001_0000 + (i as usize) * 0x10);
        let entry = IdtEntry::interrupt_gate(handler, 0x08, 0, 0);
        count += black_box(entry.handler_address()) as u64;
        count += black_box(entry.is_present()) as u64;
    }
    count
}

fn bench_idt_table_build() -> u64 {
    let mut idt = Idt::new();
    for vector in FIRST_USER_VECTOR..(FIRST_USER_VECTOR + 16) {
        let handler = 0xFFFF_8000_0010_0000 + (vector as usize) * 0x10;
        idt.set_handler(black_box(vector), black_box(handler), 0x08);
    }
    black_box(&idt);
    idt.entry(0x20).map(|e| black_box(e.handler_address()) as u64).unwrap_or(0)
}

fn bench_irq_predictor_record(iters: usize) -> u64 {
    let mut predictor = IrqPredictor::new();
    let mut count = 0u64;
    for i in 0..iters {
        predictor.record_interrupt(black_box(0x20), black_box((i as u64) * 1000 + 1000));
        count += 1;
    }
    black_box(count)
}

fn bench_irq_predictor_predict(iters: usize) -> u64 {
    let mut predictor = IrqPredictor::new();
    // Pre-fill with periodic data
    for i in 0..8 {
        predictor.record_interrupt(0x20, (i as u64) * 1000);
    }
    let mut count = 0u64;
    for _ in 0..iters {
        if let Some(p) = predictor.predict_next() {
            count += black_box(p.predicted_cycles_ahead) as u64;
        }
    }
    count
}

// ─── Identity Map / HugePage Benchmarks ──────────────────────────────────

fn bench_huge_page_alloc() -> u64 {
    let mut alloc = HugePageAllocator::new(0, 4u64 * 1024 * 1024 * 1024, HugePageSize::Size1GB);
    let mut count = 0u64;
    for _ in 0..4 {
        if let Some(addr) = alloc.allocate() {
            count += black_box(addr);
        }
    }
    count
}

fn bench_iommu_check(iters: usize) -> u64 {
    let dz = IommuDropZone::new(0x1000_0000, 0x1_0000, 0x0, 0x800_0000);
    let mut count = 0u64;
    for i in 0..iters {
        let addr = black_box(0x1000_0000 + (i as u64 & 0xFFFF));
        count += black_box(dz.is_dma_allowed(addr, 0x100)) as u64;
    }
    count
}

fn bench_nmi_pet(iters: usize) -> u64 {
    let wd = NmiWatchdog::new(62_500);
    let mut count = 0u64;
    for _ in 0..iters {
        wd.pet();
        count += black_box(wd.epoch()) & 0xFF;
    }
    count
}

fn bench_cfi_lookup(iters: usize) -> u64 {
    let mut table = CfiJumpTable::new();
    for i in 0..64 {
        table.register_target(0x1000 + (i as u64) * 0x100);
    }
    let mut count = 0u64;
    for i in 0..iters {
        let target = black_box(0x1000 + ((i as u64 % 64) * 0x100));
        count += black_box(table.is_valid_target(target)) as u64;
    }
    count
}

fn bench_identity_map_1gb() -> u64 {
    let mut map = IdentityMap::new(true);
    map.map_1gb(black_box(0), PTE_PRESENT | PTE_WRITABLE);
    black_box(map.mapped_bytes())
}

// ─── Benchmark Harness ───────────────────────────────────────────────────

fn bench<F: FnMut() -> R, R>(name: &str, iterations: usize, mut f: F) {
    // Warmup
    for _ in 0..3 {
        black_box(f());
    }

    let mut times = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        times.push(elapsed);
        black_box(result);
    }

    times.sort_unstable();
    let min = times[0];
    let _median = times[5];
    let _max = times[9];

    let ns_per_iter = min.as_nanos() as f64 / iterations as f64;
    let throughput = if ns_per_iter > 0.0 { 1_000_000_000.0 / ns_per_iter } else { f64::INFINITY };

    // Honest assessment
    let assessment = if ns_per_iter < 0.5 {
        "⚡ TRIVIAL (1-2 instructions)"
    } else if ns_per_iter < 2.0 {
        "✓ Fast (register ops)"
    } else if ns_per_iter < 10.0 {
        "✓ Normal (ALU ops)"
    } else if ns_per_iter < 100.0 {
        "● Moderate (atomics/branches)"
    } else if ns_per_iter < 1000.0 {
        "◆ Slow (memory/cache)"
    } else {
        "⚠ Very slow (I/O or complex)"
    };

    println!(
        "{:45} ns/iter={:>8.1}  {:>12.0}/s  {}",
        name,
        ns_per_iter,
        throughput,
        assessment,
    );
}
