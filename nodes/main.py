"""promboard.wan22-animate — Wan 2.2 Animate(동작 옮기기 · 인물 바꾸기) 노드 코드. 🔴 검증 안 됨: 이 코드는 실제로 돌려 보지 않았다.

노드 하나(wan-animate)가 캐릭터 그림 · 움직임 영상(필수)과 인물 마스크 영상(선택)을 받는다.
  동작 옮기기(animate)   캐릭터 그림 속 인물이 움직임 영상 속 사람의 몸짓 · 표정대로 움직이는 영상(배경은 캐릭터 그림)
  인물 바꾸기(replace)   움직임 영상 속 사람을 캐릭터 그림의 인물로 바꾼 영상(배경 · 조명은 원래 영상). 인물 마스크 영상이 필요하다
                        (흰색 = 바꿀 사람 — PromBoard 마스킹 팩의 '마스크 영상'이나 YOLO 플러그인의 흑백 마스크 영상을 그대로 잇는다)

전처리는 Wan2.2 공식 코드(wan/modules/animate/preprocess, Apache-2.0)를 nodes/preprocess/ 에 그대로 옮겨 쓴다.
  - 사람 찾기 YOLOv10m · 전신 관절(얼굴 · 손 포함 133점) ViTPose-H — 공식 Wan-AI/Wan2.2-Animate-14B 저장소 process_checkpoint 의 ONNX
  - 관절 그림(src_pose) · 얼굴 512×512 자른 영상(src_face) · (바꾸기) 마스크를 키운 것과 배경(src_bg · src_mask)을 공식 함수로 만든다
  - 공식 바꾸기 전처리는 SAM2 로 마스크를 만들지만 여기서는 이어진 마스크 영상을 쓴다(SAM2 · decord · moviepy 를 받지 않으려고).
    키우기 값은 공식 기본(iterations 3, k 7, w_len 1, h_len 1).
  - '몸 비율 맞추기'(retarget)는 공식 기본 방식(FLUX Kontext 없이)이다. 캐릭터 그림과 영상 첫 프레임의 사람이 모두 정면으로 팔다리를
    편 자세일 때만 맞는다(공식 안내).
생성: diffusers WanAnimatePipeline. 변환기는 설정 '경량화'로 고른 QuantStack GGUF Q8_0 · Q4_K_M(공식 Animate-14B 변환,
  '자동'은 앱이 이 PC 그래픽카드로 풀어 넘긴다 — 노드 코드는 '자동'을 받지 않는다), 글 인코더 UMT5-XXL(Wan2.2 공식, 5B 저장소와
  같은 파일) · CLIP 그림 인코더 · Wan2.1 VAE 는 공식 Wan-AI/Wan2.2-Animate-14B-Diffusers. 공식 예시 값: 20단계, guidance 1.0,
  77프레임 구간(segment_frame_length), 출력 30fps(전처리 fps).

노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import gc
import math
import os
import random
import time

DIFFUSERS = "Wan-AI/Wan2.2-Animate-14B-Diffusers"
TEXT_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"  # 같은 UMT5 파일(sha256 같음) — 다른 Wan 플러그인과 같이 쓴다
GGUF = {  # 설정 quantization → 변환기 파일(manifest requires.models 의 when 과 짝)
    "gguf-q8": "QuantStack/Wan2.2-Animate-14B-GGUF/Wan2.2-Animate-14B-Q8_0.gguf",
    "gguf-q4": "QuantStack/Wan2.2-Animate-14B-GGUF/Wan2.2-Animate-14B-Q4_K_M.gguf",
}
DET = "Wan-AI/Wan2.2-Animate-14B/process_checkpoint/det/yolov10m.onnx"
POSE = "Wan-AI/Wan2.2-Animate-14B/process_checkpoint/pose2d/vitpose_h_wholebody.onnx/end2end.onnx"
AREAS = {"720p": 1280 * 720, "480p": 832 * 480}
FPS = 30  # 공식 전처리 기본 fps
MAX_SECONDS = 20
STEPS = 20
GUIDANCE = 1.0
SEGMENT = 77
DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
TIME_BUDGET_SEC = 1050  # PromBoard 실행기 노드 한 번 900초 + 적재 몫 180초

_CACHE = {"pipe": None, "quant": None, "pose": None}


def _vram_mb():
    import torch

    return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024) if torch.cuda.is_available() else 0


def _free_cuda():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _repo_dir(ctx, repo, probe):
    path = os.path.normpath(ctx.model_path(f"{repo}/{probe}"))
    for _ in probe.split("/"):
        path = os.path.dirname(path)
    return path


def _read_video(path, max_seconds):
    """영상 → (RGB 프레임 목록 (h, w, 3) uint8, fps). 화면 회전 정보는 돌려서 보이는 방향으로."""
    import av
    import numpy as np

    frames = []
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("움직임 영상에 영상이 없습니다.")
        stream = container.streams.video[0]
        rate = float(stream.average_rate or stream.guessed_rate or 30)
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="rgb24")
            rot = int(round((getattr(frame, "rotation", 0) or 0) / 90.0)) % 4
            if rot:
                arr = np.ascontiguousarray(np.rot90(arr, rot))
            frames.append(arr)
            if len(frames) / rate > max_seconds:
                raise ValueError(f"움직임 영상이 너무 깁니다. {max_seconds}초까지 받습니다 — 잘라서 이어 주세요.")
    if not frames:
        raise ValueError("움직임 영상에서 프레임을 읽지 못했습니다.")
    return frames, rate


def _preprocess(ctx, progress, refer, video_path, mask_path, area, replace, retarget):
    """공식 process_pipepline.ProcessPipeline.__call__ 과 같은 순서로 (포즈 그림, 얼굴 그림, 배경, 마스크, 캐릭터 그림) PIL 목록을 만든다."""
    import numpy as np
    from PIL import Image

    from .preprocess.human_visualization import draw_aapose_by_meta_new
    from .preprocess.pose2d import Pose2d
    from .preprocess.pose2d_utils import AAPoseMeta
    from .preprocess.retarget_pose import get_retarget_pose
    from .preprocess.utils import get_aug_mask, get_face_bboxes, get_frame_indices, get_mask_body_img, padding_resize, resize_by_area
    import cv2

    if _CACHE["pose"] is None:
        progress(0.0, "load-model")
        _CACHE["pose"] = Pose2d(checkpoint=ctx.model_path(POSE), detector_checkpoint=ctx.model_path(DET), device="cpu")
    pose2d = _CACHE["pose"]
    progress(0.02, "read")
    raw, video_fps = _read_video(video_path, MAX_SECONDS)
    target = max(1, int(len(raw) / video_fps * FPS))
    frames = [raw[i] for i in get_frame_indices(len(raw), video_fps, target, FPS)]
    if replace:
        frames = [resize_by_area(f, area, divisor=16) for f in frames]
    height, width = frames[0].shape[:2]
    ctx.check_cancel()
    progress(0.05, "pose")
    metas = pose2d(frames)
    ctx.check_cancel()
    faces = []
    for idx, meta in enumerate(metas):
        x1, x2, y1, y2 = get_face_bboxes(meta["keypoints_face"][:, :2], scale=1.3, image_shape=(height, width))
        faces.append(cv2.resize(frames[idx][y1:y2, x1:x2], (512, 512)))
    refer_np = np.asarray(refer)
    bg_images, mask_images = None, None
    if replace:
        refer_np = padding_resize(refer_np, height, width)
        tpl_metas = [AAPoseMeta.from_humanapi_meta(m) for m in metas]
        cond = [draw_aapose_by_meta_new(np.zeros_like(refer_np), m) for m in tpl_metas]
        masks_raw, _ = _read_video(mask_path, MAX_SECONDS)
        masks_raw = [masks_raw[i] for i in get_frame_indices(len(masks_raw), video_fps, target, FPS)]
        bg_images, mask_images = [], []
        for frame, m in zip(frames, masks_raw):
            m = cv2.resize(m[..., 0], (width, height), interpolation=cv2.INTER_NEAREST)
            m = (m >= 128).astype(np.uint8)
            _, m = get_mask_body_img(frame, m, iterations=3, k=7)
            m = get_aug_mask(m, w_len=1, h_len=1)
            bg_images.append(Image.fromarray((frame * (1 - m[:, :, None])).astype(np.uint8)))
            mask_images.append(Image.fromarray(np.stack([m * 255] * 3, axis=2).astype(np.uint8)))
    else:
        refer_np = resize_by_area(refer_np, area, divisor=16)
        if retarget:
            refer_meta = pose2d([refer_np])[0]
            tpl_metas = get_retarget_pose(metas[0], refer_meta, metas, None, None)
            cond = [draw_aapose_by_meta_new(np.zeros_like(refer_np), m) for m in tpl_metas]
        else:
            tpl_metas = [AAPoseMeta.from_humanapi_meta(m) for m in metas]
            cond = [
                padding_resize(draw_aapose_by_meta_new(np.zeros_like(frames[0]), m), refer_np.shape[0], refer_np.shape[1])
                for m in tpl_metas
            ]
    to_pil = lambda arr: Image.fromarray(np.asarray(arr, dtype=np.uint8))
    return [to_pil(c) for c in cond], [to_pil(f) for f in faces], bg_images, mask_images, to_pil(refer_np)


def _pipeline(ctx, progress, quant):
    import torch
    from diffusers import AutoencoderKLWan, GGUFQuantizationConfig, UniPCMultistepScheduler, WanAnimatePipeline, WanAnimateTransformer3DModel
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

    if _CACHE["pipe"] is not None and _CACHE["quant"] == quant:
        return _CACHE["pipe"]
    _CACHE.update(pipe=None, quant=None)
    _free_cuda()
    progress(0.0, "load-model")
    root = _repo_dir(ctx, DIFFUSERS, "model_index.json")
    text = _repo_dir(ctx, TEXT_REPO, "text_encoder/config.json")
    transformer = WanAnimateTransformer3DModel.from_single_file(
        ctx.model_path(GGUF[quant]),
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=root,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    ctx.check_cancel()
    pipe = WanAnimatePipeline(
        tokenizer=AutoTokenizer.from_pretrained(os.path.join(text, "tokenizer")),
        text_encoder=UMT5EncoderModel.from_pretrained(os.path.join(text, "text_encoder"), torch_dtype=torch.bfloat16),
        vae=AutoencoderKLWan.from_pretrained(os.path.join(root, "vae"), torch_dtype=torch.float32),
        scheduler=UniPCMultistepScheduler.from_pretrained(os.path.join(root, "scheduler")),
        image_processor=CLIPImageProcessor.from_pretrained(os.path.join(root, "image_processor")),
        image_encoder=CLIPVisionModel.from_pretrained(os.path.join(root, "image_encoder"), torch_dtype=torch.bfloat16),
        transformer=transformer,
    )
    # 그래픽카드가 작으면(24GB 미만) 변환기는 블록 하나씩, 글 · 그림 인코더는 층(leaf) 단위로 올렸다 내리고 VAE 는 그래픽카드에 둔다.
    # 블록 단위(block_level)는 바로 아래 자식의 ModuleList 만 묶어, 층이 한 단계 깊은 UMT5 · CLIP 은 통째로 올라간다.
    # 크면 부품 단위로.
    if _vram_mb() < 24_000:
        from diffusers.hooks import apply_group_offloading

        cuda, cpu = torch.device("cuda"), torch.device("cpu")
        pipe.transformer.enable_group_offload(onload_device=cuda, offload_device=cpu, offload_type="block_level", num_blocks_per_group=1)
        for name in ("text_encoder", "image_encoder"):
            apply_group_offloading(getattr(pipe, name), onload_device=cuda, offload_device=cpu, offload_type="leaf_level")
        pipe.vae.to(cuda)
    else:
        pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    _CACHE.update(pipe=pipe, quant=quant)
    return pipe


def wan_animate(ctx, inputs, settings):
    import torch
    from PIL import Image, ImageOps

    t0 = time.monotonic()
    if ctx.device != "cuda":
        raise ValueError("Wan 2.2 Animate 는 NVIDIA 그래픽카드(CUDA)에서만 돕니다. 이 PC 에서는 그래픽카드를 찾지 못했습니다.")
    replace = settings["mode"] == "replace"
    if replace and not inputs.get("mask"):
        raise ValueError("'인물 바꾸기'는 인물 마스크 영상(흰색 = 바꿀 사람)이 있어야 합니다. 마스킹 팩의 마스크 영상을 이어 주세요.")
    with Image.open(inputs["character"]) as im:
        im.load()
        refer = ImageOps.exif_transpose(im).convert("RGB")
    area = AREAS[settings["resolution"]]
    seed = int(settings["seed"])
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    prompt = (inputs.get("prompt") or "").strip() or "视频中的人在做动作"  # 공식 generate.py 의 Animate 기본 프롬프트
    pre = _ProgressRange(ctx, 0.0, 0.2)
    try:
        cond, faces, bg, masks, refer_img = _preprocess(
            ctx, pre, refer, inputs["video"], inputs.get("mask"), area, replace, settings["retarget"] == "on" and not replace
        )
        width, height = refer_img.size
        ctx.log(f"preprocess {len(cond)} frames {width}x{height} in {time.monotonic() - t0:.1f}s")
        pipe = _pipeline(ctx, _ProgressRange(ctx, 0.2, 0.25), settings["quantization"])
        segments = max(1, math.ceil((len(cond) - 1) / (SEGMENT - 1)))
        total_steps = segments * STEPS
        done = [0]
        denoise = _ProgressRange(ctx, 0.25, 0.95)
        loop_start = [None]

        def on_step(p, i, t, kwargs):
            ctx.check_cancel()
            now = time.monotonic()
            if loop_start[0] is None:
                loop_start[0] = now
            done[0] += 1
            denoise(done[0] / total_steps, "denoise")
            if done[0] >= 2:
                per = (now - loop_start[0]) / max(1, done[0] - 1)
                estimate = (now - t0) + per * (total_steps - done[0]) * 1.15
                if estimate > TIME_BUDGET_SEC:
                    raise TimeoutError(
                        f"이 PC 에서는 약 {math.ceil(estimate / 60)}분이 걸려 한 번에 돌 수 있는 시간(약 {TIME_BUDGET_SEC // 60}분)을 넘습니다."
                        " 움직임 영상을 짧게 자르거나 해상도를 480P 로 낮춰 주세요."
                    )
            return kwargs

        call = dict(
            image=refer_img,
            pose_video=cond,
            face_video=faces,
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE,
            height=height,
            width=width,
            segment_frame_length=SEGMENT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            mode="replace" if replace else "animate",
            generator=torch.Generator(device="cpu").manual_seed(seed),
            output_type="np",
            callback_on_step_end=on_step,
        )
        if replace:
            call.update(background_video=bg, mask_video=masks)
        frames = pipe(**call).frames[0]
        ctx.check_cancel()
        ctx.progress(0.96, "encode")
        out = ctx.out_path("wan-animate.mp4")
        _write_mp4(out, frames, FPS)
        pose_out = ctx.out_path("pose.mp4")
        import numpy as np

        _write_mp4(pose_out, np.stack([np.asarray(c, dtype=np.float32) / 255.0 for c in cond]), FPS)
    except torch.OutOfMemoryError:
        oom = True
    else:
        oom = False
    finally:
        _free_cuda()
    if oom:
        # except 블록 밖에서 정리하고 새 오류를 낸다 — 블록 안에서 내면 원래 오류(와 그 프레임이 쥔 그래픽카드 텐서)가
        # 새 오류의 __context__ 로 남아 사이드카가 다음 실행까지 메모리를 놓지 못한다.
        _CACHE.update(pipe=None, quant=None)
        _free_cuda()
        raise RuntimeError(
            "그래픽카드 메모리가 모자라 멈췄습니다. 해상도를 480P 로 낮추거나 다른 프로그램(게임 · 다른 AI)을 닫아 주세요."
        )
    ctx.progress(1.0, "done")
    return {"video": out, "pose": pose_out}


class _ProgressRange:
    def __init__(self, ctx, lo, hi):
        self.ctx, self.lo, self.hi, self.last = ctx, lo, hi, lo

    def __call__(self, fraction, stage="work"):
        value = max(self.last, self.lo + (self.hi - self.lo) * max(0.0, min(1.0, fraction)))
        self.last = value
        self.ctx.progress(value, stage)


def _write_mp4(path, frames, fps):
    """frames: (n, h, w, 3) float 0~1 → H.264 yuv420p mp4(소리 없음). 크기는 16 의 배수라 짝수다."""
    import av
    import numpy as np

    n, h, w, _ = frames.shape
    container = av.open(path, "w", options={"movflags": "+faststart"})
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16", "preset": "medium"}
        for i in range(n):
            arr = (np.clip(frames[i], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()


NODES = {"wan-animate": wan_animate}
