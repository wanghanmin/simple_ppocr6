"""PP-OCRv6 ONNXRuntime OCR (no PaddleOCR framework).

Default model_size='medium' (bundled in repo). Optional: 'tiny' / 'small' via download_models.py.

Full-screen optional speed modes (default: recognize every detected box):
  det runs on the whole image (limit_side_len=960); rec/ori crops from original pixels.
  Set auto_large_image_opt=True to skip low-score / excess boxes on images >= 1920px.

  ocr.run(img)                                    # full det + rec (default)
  ocr.run(img, focus_rect=(x1, y1, x2, y2))     # optional: rec subset only
  ocr.run(img, rec=False)                         # det only -> all_det_boxes

  After run(): all_det_boxes = every det quad; results = recognized text (score>0.6).
"""

import os
import cv2
import numpy as np
import onnxruntime
import sys
import gc
import math
import traceback
import copy
import locale

def _cuda_provider_options(algo=None):
    """ORT 1.20+ maps cudnn_conv_algo_search DEFAULT -> cuDNN FALLBACK (very slow).

    det/ori/rec: HEURISTIC (override SIMPLE_PPOCR6_CUDNN_CONV_ALGO /
    SIMPLE_PPOCR6_CUDNN_CONV_ALGO_REC). Avoid DEFAULT on ORT 1.20+.
    """
    if algo is None:
        algo = os.environ.get('SIMPLE_PPOCR6_CUDNN_CONV_ALGO', 'HEURISTIC').upper()
    if algo not in ('HEURISTIC', 'EXHAUSTIVE', 'DEFAULT'):
        algo = 'HEURISTIC'
    return {
        'arena_extend_strategy': 'kSameAsRequested',
        'cudnn_conv_algo_search': algo,
        'cudnn_conv_use_max_workspace': '1',
    }


def _onnx_providers(use_gpu, session='det', model_size='medium'):
    if not use_gpu:
        return ['CPUExecutionProvider']
    if session == 'rec':
        default_rec_algo = 'HEURISTIC'
        rec_algo = os.environ.get(
            'SIMPLE_PPOCR6_CUDNN_CONV_ALGO_REC',
            default_rec_algo,
        ).upper()
        if rec_algo not in ('HEURISTIC', 'EXHAUSTIVE', 'DEFAULT'):
            rec_algo = default_rec_algo
        cuda_opts = _cuda_provider_options(rec_algo)
    else:
        cuda_opts = _cuda_provider_options()
    return [('CUDAExecutionProvider', cuda_opts), 'CPUExecutionProvider']


class simple_ppocr6():
    MODEL_PRESETS = {
        'tiny': {
            'det': r'models/ppocr6_t_det.onnx',
            'rec': r'models/ppocr6_t_rec.onnx',
            'dict': r'models/ppocr6_t_dict.txt',
        },
        'small': {
            'det': r'models/ppocr6_s_det.onnx',
            'rec': r'models/ppocr6_s_rec.onnx',
            'dict': r'models/ppocr6_s_dict.txt',
        },
        'medium': {
            'det': r'models/ppocr6_m_det.onnx',
            'rec': r'models/ppocr6_m_rec.onnx',
            'dict': r'models/ppocr6_dict.txt',
        },
    }

    def __init__(
        self,
        ppocr6_onnx_det=None,
        ppocr6_onnx_ori=r'models/ppocr6_textline_ori.onnx',
        ppocr6_onnx_rec=None,
        ppocr6_dict=None,
        model_size='medium',
        use_gpu=False,
    ):
        preset = self.MODEL_PRESETS.get(model_size, self.MODEL_PRESETS['medium'])
        ppocr6_onnx_det = ppocr6_onnx_det or preset['det']
        ppocr6_onnx_rec = ppocr6_onnx_rec or preset['rec']
        ppocr6_dict = ppocr6_dict or preset['dict']
        self.model_size = model_size
        self.use_gpu = use_gpu
        self.modal_ready = False
        self.dev_mode = False
        self.limit_side_len = 960
        self.scale = np.float32(1. / 255.)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self._det_mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        self._det_std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        # PP-OCRv6 medium det post-process (inference.yml)
        self.det_db_thresh = 0.2
        self.use_dilation = False
        self.det_db_box_thresh = 0.45
        self.det_db_unclip_ratio = 1.4
        self.det_max_candidates = 3000
        self.max_batch_size = 10
        self.use_textline_orientation = True
        self.ori_batch_num = 32 if use_gpu else 6
        self.ori_thresh = 0.9
        self.label_list = ["0_degree", "180_degree"]
        self.rec_image_shape = [3, 48, 320]
        self.max_text_length = 25
        self.rec_batch_num = 32 if use_gpu else 6
        # Optional full-screen speed cap (off by default — full det + rec).
        self.large_image_side = 1920
        self.large_image_max_rec_boxes = 96
        self.large_image_min_det_score = 0.55
        self.auto_large_image_opt = False
        self.all_det_boxes = []
        self.dt_boxes = []
        self.lang = self.checklanguage()
        self.det_net = None
        self.rec_net = None
        self.ori_net = None

        try:
            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            if use_gpu:
                options.intra_op_num_threads = 1
                det_providers = _onnx_providers(True, 'det', model_size)
                rec_providers = _onnx_providers(True, 'rec', model_size)
                ori_providers = _onnx_providers(True, 'ori', model_size)
            else:
                det_providers = rec_providers = ori_providers = ['CPUExecutionProvider']
            self.det_net = onnxruntime.InferenceSession(
                ppocr6_onnx_det, sess_options=options, providers=det_providers)
            self.rec_net = onnxruntime.InferenceSession(
                ppocr6_onnx_rec, sess_options=options, providers=rec_providers)
            self.ori_net = onnxruntime.InferenceSession(
                ppocr6_onnx_ori, sess_options=options, providers=ori_providers)
            self.logger(("Model loaded", "模型已经装载")[self.lang])
            self.det_net_input_name = self.det_net.get_inputs()[0].name

            self.character_str = []
            with open(ppocr6_dict, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
                self.character_str.append(" ")
            self.dict_character = list(self.character_str)
            self.dict_character = ["blank"] + self.dict_character
            self.character = self.dict_character
            self.logger(("Initialization complete", "初始化完成")[self.lang])
            self.modal_ready = True
        except Exception as e:
            print(("=== Detailed Exception ===", "=== 详细异常信息 ===")[self.lang])
            print(f"{('Exception Type', '异常类型')[self.lang]}: {type(e).__name__}")
            print(f"{('Exception Message', '异常信息')[self.lang]}: {str(e)}")
            print(f"\n{('=== Stack Trace ===', '=== 堆栈跟踪 ===')[self.lang]}")
            traceback.print_exc()
            raise

    def checklanguage(self):
        try:
            lang, _ = locale.getlocale()
            if lang is None:
                locale.setlocale(locale.LC_ALL, '')
                lang, _ = locale.getlocale()
            if lang:
                lang = lang.lower()
                if 'chinese' in lang or 'zh' in lang:
                    return 1
        except Exception:
            return 0
        return 0

    def unload_model(self):
        if self.det_net is not None:
            del self.det_net
            self.det_net = None
        if self.rec_net is not None:
            del self.rec_net
            self.rec_net = None
        if self.ori_net is not None:
            del self.ori_net
            self.ori_net = None
        self.logger(("Model unloaded", "模型已经卸载")[self.lang])
        if gc is not None and hasattr(gc, 'collect'):
            gc.collect()

    @staticmethod
    def get_mini_boxes(contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1],
            points[index_2],
            points[index_3],
            points[index_4],
        ]
        return box, min(bounding_box[1])

    @staticmethod
    def box_score_fast(bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin: ymax + 1, xmin: xmax + 1], mask)[0]

    @staticmethod
    def unclip(box, unclip_ratio):
        box = np.array(box).astype(np.float32)
        area = cv2.contourArea(box)
        length = cv2.arcLength(box, True)
        distance = area * unclip_ratio / (length + 1e-6)
        edges = np.roll(box, -1, axis=0) - box
        edge_lengths = np.sqrt(np.sum(edges**2, axis=1))
        norm_edges = edges / (edge_lengths[:, None] + 1e-6)
        v_side = np.column_stack((norm_edges[:, 1], -norm_edges[:, 0]))
        v_vertex = v_side + np.roll(v_side, 1, axis=0)
        cos_theta = np.sum(v_side * np.roll(v_side, 1, axis=0), axis=1)
        scale = np.sqrt(2 / (1 + cos_theta + 1e-6))
        scale = np.clip(scale, 0, 2)
        expanded_box = box + v_vertex * (distance * scale[:, None])
        return expanded_box.astype(np.int32)

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        max_candidates = self.det_max_candidates
        min_size = 3
        unclip_ratio = self.det_db_unclip_ratio

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(outs) == 3:
            _, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))

            if self.det_db_box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width
            )
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    @staticmethod
    def order_points_clockwise(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    @staticmethod
    def clip_det_res(points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape, scores=None):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        scores_new = [] if scores is not None else None
        for idx, box in enumerate(dt_boxes):
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
            if scores_new is not None:
                scores_new.append(float(scores[idx]))
        dt_boxes = np.array(dt_boxes_new)
        if scores_new is not None:
            return dt_boxes, scores_new
        return dt_boxes

    def _is_large_image(self, height, width):
        return max(height, width) >= self.large_image_side

    @staticmethod
    def _box_center_in_rect(box, rect):
        x1, y1, x2, y2 = rect
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        cx = float(box[:, 0].mean())
        cy = float(box[:, 1].mean())
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _select_rec_indices(
        self,
        dt_boxes,
        det_scores,
        image_shape,
        focus_rect=None,
        max_rec_boxes=None,
        min_det_score=None,
    ):
        n = len(dt_boxes)
        if n == 0:
            return []
        h, w = image_shape[0:2]
        if (
            self.auto_large_image_opt
            and self._is_large_image(h, w)
            and min_det_score is None
        ):
            min_det_score = self.large_image_min_det_score
        if (
            self.auto_large_image_opt
            and self._is_large_image(h, w)
            and max_rec_boxes is None
        ):
            max_rec_boxes = self.large_image_max_rec_boxes

        indices = list(range(n))
        if min_det_score is not None and det_scores is not None:
            indices = [i for i in indices if det_scores[i] >= min_det_score]
        if focus_rect is not None:
            indices = [
                i for i in indices
                if self._box_center_in_rect(dt_boxes[i], focus_rect)
            ]
        if max_rec_boxes is not None and len(indices) > max_rec_boxes:
            indices.sort(key=lambda i: det_scores[i] if det_scores else 0, reverse=True)
            indices = indices[:max_rec_boxes]
            indices.sort()
        return indices

    def sorted_boxes(self, dt_boxes):
        num_boxes = dt_boxes.shape[0]
        indexed_boxes = [(dt_boxes[i], i) for i in range(num_boxes)]
        sorted_indexed_boxes = sorted(
            indexed_boxes, key=lambda x: (x[0][0][1], x[0][0][0])
        )

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(
                        sorted_indexed_boxes[j + 1][0][0][1]
                        - sorted_indexed_boxes[j][0][0][1]
                ) < 10 and (
                        sorted_indexed_boxes[j + 1][0][0][0]
                        < sorted_indexed_boxes[j][0][0][0]
                ):
                    tmp = sorted_indexed_boxes[j]
                    sorted_indexed_boxes[j] = sorted_indexed_boxes[j + 1]
                    sorted_indexed_boxes[j + 1] = tmp
                else:
                    break

        sorted_boxes_list = [item[0] for item in sorted_indexed_boxes]
        sort_indices = [item[1] for item in sorted_indexed_boxes]
        return sorted_boxes_list, sort_indices

    def get_rotate_crop_image(self, img, points):
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def resize_norm_img_textline_ori(self, img):
        """PP-LCNet textline orientation: fixed 160x80, ImageNet normalize."""
        resized = cv2.resize(img, (160, 80)).astype("float32")
        shape = (1, 1, 3)
        para_mean = np.array(self.mean).reshape(shape).astype("float32")
        para_std = np.array(self.std).reshape(shape).astype("float32")
        normalized = (resized * self.scale - para_mean) / para_std
        return normalized.transpose(2, 0, 1)

    def resize_norm_img_rec(self, img, max_wh_ratio):
        rec_image_shape = self.rec_image_shape
        imgC, imgH, imgW = rec_image_shape

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))

        w = self.rec_net.get_inputs()[0].shape[3:][0]
        if isinstance(w, str):
            pass
        elif w is not None and w > 0:
            imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def _global_max_wh_ratio(self, img_list):
        imgC, imgH, imgW = self.rec_image_shape[:3]
        max_ratio = imgW / imgH
        for img_rec in img_list:
            h, w = img_rec.shape[:2]
            max_ratio = max(max_ratio, w / float(h))
        return max_ratio

    def _rec_single_batch_limit(self):
        if self.use_gpu:
            return 128 if self.model_size == 'medium' else self.rec_batch_num
        return self.rec_batch_num

    def _fixed_rec_chunk_size(self):
        if self.use_gpu:
            return 64 if self.model_size == 'medium' else self.rec_batch_num
        return self.rec_batch_num

    def _run_rec_batches(self, img_list, indices, global_max_wh_ratio):
        """Run rec with fixed-width tensors; pad GPU chunks for stable cuDNN shapes."""
        img_num = len(img_list)
        rec_res = [["", 0.0]] * img_num
        if img_num == 0:
            return rec_res
        rec_input_name = self.rec_net.get_inputs()[0].name
        single_limit = self._rec_single_batch_limit()
        norm_cache = [
            self.resize_norm_img_rec(img_list[indices[i]], global_max_wh_ratio)
            for i in range(img_num)
        ]

        def _infer(batch_tensors, count):
            batch = np.stack(batch_tensors, axis=0)
            preds_rec = self.rec_net.run(None, {rec_input_name: batch})[0]
            return self._decode_rec_outputs(preds_rec, count)

        if img_num <= single_limit:
            decoded = _infer(norm_cache, img_num)
            for rno, rec_item in enumerate(decoded):
                rec_res[indices[rno]] = rec_item
            return rec_res

        chunk = self._fixed_rec_chunk_size()
        for beg in range(0, img_num, chunk):
            end = min(img_num, beg + chunk)
            chunk_tensors = norm_cache[beg:end]
            count = end - beg
            if self.use_gpu and count < chunk:
                pad_shape = chunk_tensors[0].shape
                chunk_tensors = chunk_tensors + [
                    np.zeros(pad_shape, dtype=np.float32)
                    for _ in range(chunk - count)
                ]
            decoded = _infer(chunk_tensors, count)
            for rno, rec_item in enumerate(decoded):
                rec_res[indices[beg + rno]] = rec_item
        return rec_res

    def _decode_rec_outputs(self, preds_rec, count):
        if isinstance(preds_rec, (tuple, list)):
            preds_rec = preds_rec[-1]
        preds_idx = preds_rec.argmax(axis=2)
        preds_prob = preds_rec.max(axis=2)
        result_list = []
        ignored_tokens = [0]
        for batch_idx in range(count):
            selection = np.ones(len(preds_idx[batch_idx]), dtype=bool)
            for ignored_token in ignored_tokens:
                selection &= preds_idx[batch_idx] != ignored_token
            char_list = [
                self.character[text_id]
                for text_id in preds_idx[batch_idx][selection]
            ]
            if preds_prob is not None:
                conf_list = preds_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]
            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_bounding_box(self, points):
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

    def logger(self, info):
        if self.dev_mode:
            print(info)

    def __del__(self):
        self.unload_model()

    def run(
        self,
        img,
        det=True,
        rec=True,
        cls=True,
        focus_rect=None,
        max_rec_boxes=None,
        min_det_score=None,
    ):
        try:
            self.dt_boxes = []
            self.all_det_boxes = []
            if isinstance(img, str) and os.path.isfile(img):
                self.img = cv2.imread(img)
            elif isinstance(img, np.ndarray):
                self.img = img
            elif isinstance(img, (bytes, bytearray)):
                nparr = np.frombuffer(img, np.uint8)
                self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                raise ValueError("无法解码图像数据")

            h, w, c = self.img.shape
            if max(h, w) > self.limit_side_len:
                if h > w:
                    ratio = float(self.limit_side_len) / h
                else:
                    ratio = float(self.limit_side_len) / w
            else:
                ratio = 1.0
            resize_h = int(h * ratio)
            resize_w = int(w * ratio)
            resize_h = max(int(round(resize_h / 32) * 32), 32)
            resize_w = max(int(round(resize_w / 32) * 32), 32)
            ratio_h = resize_h / float(h)
            ratio_w = resize_w / float(w)
            img_resize = cv2.resize(self.img, (int(resize_w), int(resize_h)))
            shape_list = np.expand_dims(np.array([h, w, ratio_h, ratio_w]), axis=0)

            img_det = (img_resize.astype("float32") * self.scale - self._det_mean) / self._det_std
            img_chw = img_det.transpose((2, 0, 1))
            img_ret = np.expand_dims(img_chw, axis=0)
            outputs = self.det_net.run(None, {self.det_net_input_name: img_ret})
            pred = outputs[0][:, 0, :, :]
            segmentation = pred > self.det_db_thresh
            boxes_batch = []
            for batch_index in range(pred.shape[0]):
                src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
                mask = segmentation[batch_index]
                boxes, scores = self.boxes_from_bitmap(
                    pred[batch_index], mask, src_w, src_h
                )
                boxes_batch.append({"points": boxes, "scores": scores})
            dt_boxes = boxes_batch[0]["points"]
            det_scores = boxes_batch[0]["scores"]
            dt_boxes, det_scores = self.filter_tag_det_res(
                dt_boxes, self.img.shape, det_scores
            )
            self.all_det_boxes = dt_boxes.copy() if len(dt_boxes) else np.array([])

            if not rec:
                self.dt_boxes = list(dt_boxes) if len(dt_boxes) else []
                self.results = []
                return self.results

            rec_indices = self._select_rec_indices(
                dt_boxes,
                det_scores,
                self.img.shape,
                focus_rect=focus_rect,
                max_rec_boxes=max_rec_boxes,
                min_det_score=min_det_score,
            )
            if len(rec_indices) == 0:
                self.results = []
                return self.results

            dt_boxes = dt_boxes[rec_indices]
            if det_scores is not None:
                det_scores = [det_scores[i] for i in rec_indices]

            img_crop_list = []
            dt_boxes, sort_indices = self.sorted_boxes(dt_boxes)
            for bno in range(len(dt_boxes)):
                img_crop = self.get_rotate_crop_image(self.img, dt_boxes[bno])
                img_crop_list.append(img_crop)

            img_list = img_crop_list
            if self.use_textline_orientation and cls:
                img_num = len(img_list)
                width_list = [img_ori.shape[1] / float(img_ori.shape[0]) for img_ori in img_list]
                indices = np.argsort(np.array(width_list))
                ori_res = [["", 0.0]] * img_num
                ori_input_name = self.ori_net.get_inputs()[0].name
                if img_num > 0:
                    norm_ori_batch = np.stack(
                        [self.resize_norm_img_textline_ori(img_list[indices[i]]) for i in range(img_num)],
                        axis=0,
                    )
                    prob_out = self.ori_net.run(None, {ori_input_name: norm_ori_batch})[0]
                    pred_idxs = prob_out.argmax(axis=1)
                    for i, idx in enumerate(pred_idxs):
                        label = self.label_list[idx]
                        score = prob_out[i, idx]
                        ori_res[indices[i]] = [label, score]
                        if "180" in label and score > self.ori_thresh:
                            img_list[indices[i]] = cv2.rotate(img_list[indices[i]], 1)

            img_num = len(img_list)
            width_list = [img_rec.shape[1] / float(img_rec.shape[0]) for img_rec in img_list]
            indices = np.argsort(np.array(width_list))
            global_max_wh_ratio = self._global_max_wh_ratio(img_list)
            rec_res = self._run_rec_batches(img_list, indices, global_max_wh_ratio)

            self.results = []
            for i, (text, score) in enumerate(rec_res):
                if score > 0.6:
                    self.results.append({
                        'text': text,
                        'rec_pos': self.get_bounding_box(dt_boxes[i])
                    })
            self.dt_boxes = [item['rec_pos'] for item in self.results]
        except Exception:
            traceback.print_exc()
            self.results = []
            return None

    def displaybox(self, winname):
        try:
            box_img = self.img.copy()
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            for i, box in enumerate(self.dt_boxes):
                color = colors[i % len(colors)]
                points_array = np.array(box, dtype=np.int32)
                cv2.polylines(box_img, [points_array], isClosed=True, color=color, thickness=2)
            cv2.imshow(winname, box_img)
        except Exception:
            traceback.print_exc()
            return None


if __name__ == "__main__":
    print("This is simple_ppocr6 src file")
