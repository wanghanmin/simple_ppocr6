"""OCR a local image with simple_ppocr6 (CLI example)."""

import argparse
import os
import sys
import time

import cv2

from simple_ppocr6 import simple_ppocr6


def main():
    parser = argparse.ArgumentParser(description="Run PP-OCRv6 ONNX OCR on an image")
    parser.add_argument(
        "image",
        nargs="?",
        default="example.jpg",
        help="image path (default: example.jpg)",
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "medium"],
        default="small",
        help="model preset (default: small)",
    )
    parser.add_argument("--gpu", action="store_true", help="use CUDAExecutionProvider")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return 1

    load_start = time.perf_counter()
    ocr = simple_ppocr6(model_size=args.model_size, use_gpu=args.gpu)
    load_time = time.perf_counter() - load_start

    start = time.perf_counter()
    ocr.run(args.image)
    ocr_time = time.perf_counter() - start

    provider = ocr.det_net.get_providers()[0]
    n_det = len(ocr.all_det_boxes) if ocr.all_det_boxes is not None else 0
    n_rec = len(ocr.results or [])

    print(f"Image: {args.image}")
    print(f"Model: {args.model_size} | Provider: {provider}")
    print(f"Load: {load_time:.3f}s | OCR: {ocr_time:.3f}s | Total: {load_time + ocr_time:.3f}s")
    print(f"Boxes: det={n_det} rec={n_rec}")
    print()

    for item in ocr.results or []:
        print(item["text"], item["rec_pos"])

    ocr.displaybox("simple-ppocr6 result")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
