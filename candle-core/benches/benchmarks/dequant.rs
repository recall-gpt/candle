use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{
    quantized::{self, GgmlDType},
    Device, Result, Tensor,
};
use criterion::{criterion_group, Criterion};
use std::time::Instant;

fn run_dequant_f32(qtensor: &quantized::QTensor, device: &Device) {
    let _ = qtensor.dequantize(device).unwrap();
}

fn run_dequant_f16(qtensor: &quantized::QTensor, device: &Device) {
    let _ = qtensor.dequantize_f16(device).unwrap();
}

fn bench_dequant(c: &mut Criterion, device: &Device) -> Result<()> {
    let shape = (64, 128);
    let src = Tensor::randn(0f32, 1f32, shape, device)?;
    let qtensor = quantized::QTensor::quantize(&src, GgmlDType::Q8_0)?;

    let mut group = c.benchmark_group(device.bench_name("dequant_q8_0"));
    group.bench_function("f32", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                run_dequant_f32(&qtensor, device);
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.bench_function("f16", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                run_dequant_f16(&qtensor, device);
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();

    Ok(())
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        bench_dequant(c, &device).unwrap();
    }
}

criterion_group!(benches, criterion_benchmark);
