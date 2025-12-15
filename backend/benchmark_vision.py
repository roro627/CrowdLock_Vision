import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from backend.core.analytics.pipeline import VisionPipeline
from backend.core.detectors.yolo import YoloPersonDetector


def generate_synthetic_frame(width: int, height: int, num_people: int = 5) -> np.ndarray:
    """Generates a black frame with moving white rectangles simulating people."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(num_people):
        x = np.random.randint(0, width - 50)
        y = np.random.randint(0, height - 100)
        cv2.rectangle(frame, (x, y), (x + 50, y + 100), (255, 255, 255), -1)
    return frame


def run_benchmark(
    device: str,
    duration_sec: int = 10,
    resolution: tuple[int, int] = (1920, 1080),
    inference_width: int = 640,
    optimize: bool = False,
):
    print(f"\n{'='*50}")
    print(f"Running Benchmark on DEVICE: {device.upper()}")
    print(f"{'='*50}")
    print(f"Resolution: {resolution}")
    print(f"Inference Width: {inference_width}")
    print(f"Duration: {duration_sec}s")
    print(f"Optimize: {optimize}")

    model_name = "yolov8n.pt"
    if optimize and device == "cpu":
        onnx_path = "yolov8n.onnx"
        if os.path.exists(onnx_path):
            print(f"Found existing optimized model: {onnx_path}")
            model_name = onnx_path
        else:
            print("Exporting model to ONNX for CPU optimization...")
            model = YOLO(model_name)
            model.export(format="onnx", device="cpu")
            model_name = onnx_path
            print(f"Model exported to {model_name}")

    # Initialize pipeline
    try:
        # ONNX models should not receive .to(device)
        detector = YoloPersonDetector(
            model_name=model_name,
            device=None if model_name.endswith(".onnx") else device,
            task="detect",
        )
        pipeline = VisionPipeline(detector=detector)
    except Exception as e:
        import traceback

        print(f"Error initializing pipeline on {device}: {e}")
        with open("benchmark_error.txt", "w") as f:
            f.write(traceback.format_exc())
        return

    print("Pipeline initialized. Starting warmup...")

    # Warmup
    warmup_frame = generate_synthetic_frame(*resolution)
    for _ in range(10):
        pipeline.process(warmup_frame, inference_width=inference_width)

    print("Warmup complete. Starting benchmark...")

    frame_count = 0
    latencies = []
    start_time = time.time()
    end_time = start_time + duration_sec

    while time.time() < end_time:
        frame = generate_synthetic_frame(*resolution)

        t0 = time.time()
        pipeline.process(frame, inference_width=inference_width)
        t1 = time.time()

        latencies.append((t1 - t0) * 1000)  # ms
        frame_count += 1

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time

    latencies = np.array(latencies)
    print(f"\nResults for {device.upper()} (Optimize={optimize}):")
    print(f"Total Frames: {frame_count}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print("Latency (ms):")
    print(f"  Min: {np.min(latencies):.2f}")
    print(f"  Max: {np.max(latencies):.2f}")
    print(f"  Avg: {np.mean(latencies):.2f}")
    print(f"  P95: {np.percentile(latencies, 95):.2f}")
    print(f"  P99: {np.percentile(latencies, 99):.2f}")

    results = {
        "device": device,
        "optimize": optimize,
        "resolution": list(resolution),
        "inference_width": inference_width,
        "fps": float(avg_fps),
        "latency_avg_ms": float(np.mean(latencies)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "frames": frame_count,
        "duration_sec": duration_sec,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CrowdLock Vision Pipeline")
    parser.add_argument(
        "--duration", type=int, default=10, help="Benchmark duration in seconds per device"
    )
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--inference-width", type=int, default=640, help="Inference width")
    parser.add_argument(
        "--optimize", action="store_true", help="Enable model optimization (ONNX for CPU)"
    )
    parser.add_argument(
        "--device", type=str, default="all", help="Device to run on (cpu, cuda, or all)"
    )

    args = parser.parse_args()

    devices = []
    if args.device == "all":
        devices.append("cpu")
        if torch.cuda.is_available():
            devices.append("cuda")
        else:
            print("\nCUDA not available. Skipping GPU benchmark.")
    elif args.device == "cuda":
        if torch.cuda.is_available():
            devices.append("cuda")
        else:
            print("\nCUDA not available.")
    else:
        devices.append(args.device)

    all_results = []
    for device in devices:
        # Only optimize CPU for now
        do_optimize = args.optimize and device == "cpu"
        res = run_benchmark(
            device=device,
            duration_sec=args.duration,
            resolution=(args.width, args.height),
            inference_width=args.inference_width,
            optimize=do_optimize,
        )
        if res:
            all_results.append(res)

    if all_results:
        with open("benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=4)
        print("Saved results to benchmark_results.json")
