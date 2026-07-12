"""Windows clipboard screenshot OCR example (simple_ppocr6)."""

import argparse
import sys
import time
import traceback
import warnings

import cv2
import numpy as np
import win32clipboard

import simple_ppocr6

warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_clip_dib():
    try:
        win32clipboard.OpenClipboard()
        try:
            if not win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_DIB):
                return None
            clipdata = win32clipboard.GetClipboardData(win32clipboard.CF_DIB)
            bmpfile = bytearray([0x42, 0x4D])
            bmpfile.extend(len(clipdata).to_bytes(4, "little"))
            bmpfile.extend([0x00, 0x00, 0x00, 0x00])
            bmpfile.extend((14 + 40 + 0).to_bytes(4, "little"))
            bmpfile.extend(clipdata)
            return cv2.imdecode(np.frombuffer(bmpfile, dtype=np.uint8), cv2.IMREAD_COLOR)
        finally:
            win32clipboard.CloseClipboard()
    except Exception:
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="OCR image from Windows clipboard")
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "medium"],
        default="small",
        help="model preset (default: small)",
    )
    parser.add_argument("--gpu", dest="use_gpu", action="store_true", default=True)
    parser.add_argument("--cpu", dest="use_gpu", action="store_false")
    args = parser.parse_args()
    use_gpu = args.use_gpu

    img = get_clip_dib()
    if img is None:
        print("No image in clipboard. Copy a screenshot first (Win+Shift+S).")
        return 1

    h, w = img.shape[:2]
    print(f"Clipboard: {w}x{h}px | model={args.model_size} | gpu={use_gpu}")

    t0 = time.perf_counter()
    ocr = simple_ppocr6.simple_ppocr6(model_size=args.model_size, use_gpu=use_gpu)
    load_t = time.perf_counter() - t0

    t1 = time.perf_counter()
    ocr.run(img)
    ocr_t = time.perf_counter() - t1

    results = ocr.results or []
    print(
        f"Load: {load_t:.2f}s | OCR: {ocr_t:.2f}s | "
        f"boxes={len(results)} | provider={ocr.det_net.get_providers()[0]}"
    )
    for item in results:
        pts = ", ".join(f"({int(c[0])},{int(c[1])})" for c in item["rec_pos"])
        print(f"[{item['text']}] {pts}")

    cv2.namedWindow("ocr_result", cv2.WINDOW_GUI_EXPANDED)
    ocr.displaybox("ocr_result")
    cv2.waitKey()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
