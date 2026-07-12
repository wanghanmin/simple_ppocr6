import cv2
import time
import base64
import numpy as np
import argparse
from flask import Flask, request, jsonify
from simple_ppocr6 import simple_ppocr6
import threading

# Initialize Flask app
app = Flask(__name__)

# Initialize OCR model (global instance, reused for all requests)
# Will be reinitialized with command line args in __main__
ocr = None
ocr_lock = threading.Lock()  # Thread safety for concurrent requests

@app.route('/ocr', methods=['POST'])
def ocr_service():
    total_start = time.time()
    
    try:
        # Get request data
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Invalid request, 'image' field is required."}), 400

        # Decode base64 image
        decode_start = time.time()
        image_base64 = data["image"]
        try:
            image_bytes = base64.b64decode(image_base64)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "Failed to decode image from base64."}), 400
        except Exception as e:
            return jsonify({"error": f"Image decoding failed: {str(e)}"}), 400
        decode_time = time.time() - decode_start

        # Run OCR with thread lock for safety
        ocr_start = time.time()
        with ocr_lock:
            ocr.run(img)
            # Format results immediately within lock to avoid data race
            ocr_results = [
                {
                    "text": item['text'],
                    "bounding_box": item['rec_pos'].tolist()
                }
                for item in ocr.results
            ]
        ocr_time = time.time() - ocr_start
        
        total_time = time.time() - total_start

        # Return results with detailed timing
        return jsonify({
            "processing_time": ocr_time,
            "decode_time": decode_time,
            "total_time": total_time,
            "results": ocr_results
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='OCR HTTP Service - Production-ready API for text recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python ocr-http-service.py
  python ocr-http-service.py --port 8080
  python ocr-http-service.py --host 127.0.0.1 --port 8080 --gpu
  python ocr-http-service.py --host 0.0.0.0 --port 8080 --threads 8 --gpu
''')
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                        metavar='HOST',
                        help='host address to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=11005, 
                        metavar='PORT',
                        help='port number to bind (default: 11005)')
    parser.add_argument('--threads', type=int, default=4, 
                        metavar='N',
                        help='number of worker threads (default: 4)')
    parser.add_argument('--gpu', action='store_true', 
                        help='enable GPU acceleration (default: CPU mode)')
    args = parser.parse_args()
    
    # Reinitialize OCR model with GPU setting if specified
    if args.gpu:
        print(f"Using GPU mode...")
        ocr = simple_ppocr6(use_gpu=True)
    else:
        print(f"Using CPU mode...")
        ocr = simple_ppocr6(use_gpu=False)
    
    print(f"OCR model loaded successfully.")
    
    # Start Flask service with optimized settings
    from waitress import serve
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"Threads: {args.threads}")
    # Use waitress for production (better performance than Flask dev server)
    serve(app, host=args.host, port=args.port, threads=args.threads)
