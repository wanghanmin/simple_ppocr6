"""Download PP-OCRv6 ONNX models from HuggingFace into models/."""

import argparse
import os
import shutil

from huggingface_hub import hf_hub_download

SIZES = {
    "tiny": [
        ("PaddlePaddle/PP-OCRv6_tiny_det_onnx", "inference.onnx", "ppocr6_t_det.onnx"),
        ("PaddlePaddle/PP-OCRv6_tiny_rec_onnx", "inference.onnx", "ppocr6_t_rec.onnx"),
        ("PaddlePaddle/PP-LCNet_x0_25_textline_ori_onnx", "inference.onnx", "ppocr6_textline_ori.onnx"),
        ("PaddlePaddle/PP-OCRv6_tiny_rec_onnx", "inference.yml", "ppocr6_t_rec.yml"),
    ],
    "small": [
        ("PaddlePaddle/PP-OCRv6_small_det_onnx", "inference.onnx", "ppocr6_s_det.onnx"),
        ("PaddlePaddle/PP-OCRv6_small_rec_onnx", "inference.onnx", "ppocr6_s_rec.onnx"),
        ("PaddlePaddle/PP-LCNet_x0_25_textline_ori_onnx", "inference.onnx", "ppocr6_textline_ori.onnx"),
        ("PaddlePaddle/PP-OCRv6_small_rec_onnx", "inference.yml", "ppocr6_s_rec.yml"),
    ],
    "medium": [
        ("PaddlePaddle/PP-OCRv6_medium_det_onnx", "inference.onnx", "ppocr6_m_det.onnx"),
        ("PaddlePaddle/PP-OCRv6_medium_rec_onnx", "inference.onnx", "ppocr6_m_rec.onnx"),
        ("PaddlePaddle/PP-LCNet_x0_25_textline_ori_onnx", "inference.onnx", "ppocr6_textline_ori.onnx"),
        ("PaddlePaddle/PP-OCRv6_medium_rec_onnx", "inference.yml", "ppocr6_m_rec.yml"),
    ],
}

DICT_NAMES = {
    "tiny": "ppocr6_t_dict.txt",
    "small": "ppocr6_s_dict.txt",
    "medium": "ppocr6_dict.txt",
}


def extract_dict(yml_path, dict_path):
    import yaml

    with open(yml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    chars = cfg["PostProcess"]["character_dict"]
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("\n".join(chars) + "\n")
    print(f"Dictionary: {len(chars)} chars -> {dict_path}")


def download_size(size, output_dir):
    seen = set()
    yml_path = None
    for repo, fname, dest_name in SIZES[size]:
        if dest_name in seen:
            continue
        seen.add(dest_name)
        dest = os.path.join(output_dir, dest_name)
        if os.path.isfile(dest):
            print(f"Skip existing {dest}")
            continue
        print(f"Downloading {repo}/{fname} ...")
        cached = hf_hub_download(repo_id=repo, filename=fname)
        shutil.copy2(cached, dest)
        print(f"  -> {dest} ({os.path.getsize(dest) / 1024 / 1024:.2f} MB)")
        if dest_name.endswith("_rec.yml"):
            yml_path = dest

    if yml_path:
        dict_path = os.path.join(output_dir, DICT_NAMES[size])
        extract_dict(yml_path, dict_path)
        os.remove(yml_path)


def main():
    parser = argparse.ArgumentParser(description="Download PP-OCRv6 ONNX models")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument(
        "--size",
        default="medium",
        choices=["tiny", "small", "medium", "all"],
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sizes = list(SIZES) if args.size == "all" else [args.size]
    for size in sizes:
        print(f"\n=== {size} ===")
        download_size(size, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
