# Wan 2.2 Animate — 동작 옮기기 · 인물 바꾸기

PromBoard 커뮤니티 플러그인입니다. 캔버스에 **Wan 2.2 Animate** 노드 하나를 더합니다. 알리바바 Wan 팀이 공개한 Wan2.2-Animate-14B 로 두 가지를 합니다.

- **동작 옮기기**: 캐릭터 그림 속 인물이 움직임 영상 속 사람의 몸짓 · 표정대로 움직이는 영상을 만듭니다. 배경은 캐릭터 그림을 따릅니다.
- **인물 바꾸기**: 움직임 영상의 배경 · 조명은 그대로 두고 사람만 캐릭터 그림의 인물로 바꿉니다. 바꿀 사람을 흰색으로 칠한 **인물 마스크 영상**을 이어야 합니다(PromBoard 마스킹 팩의 '마스크 영상', YOLO 플러그인의 흑백 마스크 영상).

> 🔴 **검증 안 됨**: 아직 실제로 돌려 보지 않았습니다. 이 플러그인을 만든 PC(그래픽카드 12GB)로는 14B 모델을 PromBoard 의 한 번 실행 시간(약 15분) 안에 돌릴 근거가 없어 시험하지 않았습니다. 문제가 있으면 [이슈](https://github.com/promboard-app/wan22-animate/issues)로 알려 주세요.

## 입력과 결과

| 포트 | 종류 | 뜻 |
|---|---|---|
| 캐릭터 그림 | 그림(필수) | 움직일 인물. 정면 전신 그림이 가장 잘 맞습니다 |
| 움직임 영상 | 영상(필수) | 사람 한 명이 움직이는 영상, 20초까지 |
| 인물 마스크 영상 | 영상(인물 바꾸기에 필수) | 흰색 = 바꿀 사람, 검정 = 그대로 둘 곳 |
| 프롬프트 | 글(선택) | 비우면 공식 기본 프롬프트를 씁니다 |
| 결과 영상 | 영상 | 30fps mp4(H.264), 소리 없음 |
| 뽑은 관절 영상 | 영상 | 모델에 넣은 관절 그림(확인용) |

- 설정: 모드(동작 옮기기 · 인물 바꾸기), 몸 비율 맞추기(끄기 · 켜기, 동작 옮기기에서만), 해상도(720P · 480P, 넓이 기준), 경량화(아래), 시드.
- 생성 값은 Wan 공식 설정(`wan/configs/wan_animate_14B.py`)과 같습니다: 20단계, guidance 1.0, 77프레임 구간, 30fps.

## 받는 것과 용량 (처음 실행할 때 한 번, 약 27GB · Q8 은 약 35GB)

| 파일 | 출처 | 크기 |
|---|---|---|
| 변환기 Wan2.2-Animate-14B GGUF Q4_K_M 또는 Q8_0 (설정 '경량화'로 고른 것만) | [QuantStack/Wan2.2-Animate-14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF) | 11.5GB · 18.7GB |
| 글 인코더 UMT5-XXL · 토크나이저 | [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (Animate 저장소와 같은 파일 — Wan 2.2 플러그인과 같이 씁니다) | 11.4GB |
| CLIP 그림 인코더 · Wan2.1 VAE · 설정 | [Wan-AI/Wan2.2-Animate-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B-Diffusers) | 1.8GB |
| 전처리 YOLOv10m · ViTPose-H 전신 | [Wan-AI/Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) `process_checkpoint/` | 2.6GB |

- 경량화: 처음 값 **자동**은 그래픽카드 32GB 이상이면 GGUF Q8, 그 밖(그래픽카드를 모르는 PC 포함)은 GGUF Q4 를 고르고 그 판만 받습니다. 24GB 이상 그래픽카드는 변환기를 통째로 올려 Q8(18.7GB)은 32GB 부터 넉넉하고, 그보다 작은 그래픽카드는 변환기를 블록 하나씩 올리므로 판이 RAM 사용량 · 받기 크기만 바꿉니다. 두 판 모두 텐서 이름이 diffusers 변환과 맞는 것을 GGUF 머리로 확인했고, 실제로 돌려 보지는 않았습니다.

## 필요한 사양

- NVIDIA 그래픽카드 전용(CUDA). 공식 최소 VRAM 수치가 없어, 실제로 돌려 보지 않은 상태로 **24GB 이상**을 권장 값으로 적었습니다(PromBoard 에서 '고성능'으로 보입니다). 24GB 보다 작은 그래픽카드에서는 변환기를 블록 하나씩, 글 · 그림 인코더를 층 단위로 올렸다 내리고 VAE 는 그래픽카드에 둡니다.
- RAM 64GB 권장(변환기 · 글 인코더를 RAM 에 둡니다).

## 라이선스와 조건

- 플러그인 코드: Apache-2.0([LICENSE](LICENSE)).
- `nodes/preprocess/` 는 Wan2.2 공식 전처리 코드(Apache-2.0, [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) `wan/modules/animate/preprocess`)를 옮긴 것입니다. 바꾼 것은 패키지 안 import 경로뿐입니다(`nodes/preprocess/LICENSE-Wan2.2.txt`).
- 모델 가중치: Wan2.2-Animate-14B 는 Apache-2.0 입니다. 매출 상한 · 지역 제한 · 사용 승인 절차가 없습니다. 만든 영상의 권리와 책임은 사용자에게 있습니다(모델 카드).
- 공식 전처리와 다른 점: 인물 바꾸기의 마스크를 SAM2 로 만들지 않고 이어진 마스크 영상을 씁니다(키우기 값은 공식 기본 iterations 3 · k 7). 몸 비율 맞추기는 FLUX Kontext 를 쓰지 않는 공식 기본 방식입니다.

## 설치

1. PromBoard 앱 → 플러그인 매니저 → **커뮤니티 플러그인** → 둘러보기에서 설치합니다.
2. 코드가 든 플러그인이라 **제한 모드**를 끄고 플러그인을 켭니다.
3. 처음 실행할 때 실행 준비가 파이썬 패키지(diffusers · transformers · onnxruntime · OpenCV 등)와 모델(약 27GB, Q8 은 약 35GB)을 받습니다.

## 플러그인 만드는 사람을 위한 메모

```
manifest.json            플러그인 정보 · 노드 정의 · 모델(requires.models)
nodes/main.py            NODES = {"wan-animate": 함수} — 공식 전처리 순서대로 관절 · 얼굴 · (바꾸기) 마스크를 만들고 WanAnimatePipeline 으로 생성
nodes/preprocess/        Wan2.2 공식 전처리 코드(Apache-2.0)
nodes/requirements*.txt  받을 패키지 · 운영체제별 해시 잠금 파일(Windows · Linux)
```

- 잠금 파일 다시 만들기: `uv pip compile nodes/requirements.in --generate-hashes --python-version 3.12 --python-platform x86_64-pc-windows-msvc --index-url https://pypi.org/simple --exclude-newer 2026-10-25T00:00:00Z` (리눅스는 `x86_64-manylinux_2_28`). torch 줄은 앱이 빼고 앱의 torch 2.8.0 을 먼저 깝니다(`"torch": "host"`).
- ViTPose-H ONNX 는 가중치가 파일 394개로 나뉘어 있어(external data) 모델 항목 두 개로 나눠 받습니다(항목 하나에 파일 200개까지).

---

## English

A PromBoard community plugin. It adds one **Wan 2.2 Animate** node (`Wan 2.2 Animate`) that runs Alibaba's open Wan2.2-Animate-14B:

- **Animate**: the person in the character image moves like the person in the driving video (body and face). The background follows the character image.
- **Replace**: keeps the driving video's background and lighting and replaces only the person with the character. Needs a **person mask video** (white = the person to replace), e.g. the PromBoard masking pack's mask video or the YOLO plugin's mask video.

> 🔴 **Unverified**: not run yet. The 12 GB card this plugin was built on gives no reason to expect the 14B model to finish within PromBoard's single-run limit (about 15 minutes), so it was not tested. Please report problems in [issues](https://github.com/promboard-app/wan22-animate/issues).

- Inputs: character image (required), driving video (required, one person, up to 20 s), person mask video (required for replace), prompt (optional; the official default prompt is used when empty). Outputs: result video (30 fps H.264 MP4, no audio) and the extracted pose video.
- Settings: mode, retarget body proportions (animate only), resolution by area (720P · 480P), quantization (Auto = GGUF Q8 on 32 GB+ GPUs, GGUF Q4_K_M otherwise; only the chosen file is downloaded), seed. Generation values follow the official config: 20 steps, guidance 1.0, 77-frame segments, 30 fps.
- Downloads (once, about 27 GB with Q4 / 35 GB with Q8): Animate-14B GGUF Q4_K_M or Q8_0 transformer (QuantStack, 11.5 / 18.7 GB), UMT5-XXL text encoder from Wan-AI/Wan2.2-TI2V-5B-Diffusers (same file as the Animate repo, shared with the Wan 2.2 plugin, 11.4 GB), CLIP image encoder · Wan2.1 VAE · configs from Wan-AI/Wan2.2-Animate-14B-Diffusers (1.8 GB), and the official preprocessing models YOLOv10m · ViTPose-H wholebody from Wan-AI/Wan2.2-Animate-14B (2.6 GB).
- Requirements: NVIDIA GPU (CUDA) only. There is no official minimum VRAM figure; **24 GB or more** is a recommendation, not a measurement (shown as high-end in PromBoard). On smaller cards the transformer is offloaded block by block, the text and image encoders layer by layer, and the VAE stays on the GPU. 64 GB RAM recommended.
- **Install**: PromBoard → Plugin Manager → **Community plugins** → Browse → install. Turn off **Restricted mode** and enable the plugin. The first run prepares the Python packages and the models (about 27 GB, or 35 GB with Q8).

The plugin texts in the app are Korean only for now — the plugin manifest takes a single language.

## License

Plugin code: Apache-2.0 — see [LICENSE](LICENSE). `nodes/preprocess/` is the official Wan2.2 preprocessing code (Apache-2.0, [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)), changed only in its package-relative imports. The Wan2.2-Animate-14B weights are Apache-2.0 with no revenue cap, regional restriction or access gate, and are downloaded from Hugging Face, not bundled. Differences from the official preprocessing: replace mode takes the connected mask video instead of running SAM2 (mask dilation uses the official defaults), and retargeting is the official basic method without FLUX Kontext.
