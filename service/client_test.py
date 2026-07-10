import argparse
import base64
import os

import requests


def save_base64_image(b64_str, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--image", default="data/samples/bus.jpg")
    parser.add_argument("--out_dir", default="outputs/service")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.image, "rb") as f:
        files = {"file": (os.path.basename(args.image), f, "image/jpeg")}
        response = requests.post(args.url, files=files, timeout=120)

    print("HTTP status:", response.status_code)

    if response.status_code != 200:
        print(response.text)
        return

    data = response.json()

    print("num_detections:", data["num_detections"])
    print("latency_ms:", data["latency_ms"])
    print("providers:", data["model_info"]["providers"])

    print("\nDetections:")
    for det in data["detections"]:
        print(det)

    result_path = os.path.join(args.out_dir, "fastapi_result.jpg")
    mask_path = os.path.join(args.out_dir, "fastapi_mask.png")

    save_base64_image(data["result_image_base64"], result_path)
    save_base64_image(data["mask_image_base64"], mask_path)

    print("\nSaved:", result_path)
    print("Saved:", mask_path)


if __name__ == "__main__":
    main()
