"""与部署位置无关的业务常量。

这里只放「协议/能力本身」的固定值；凡是可能因机器而变的路径、端口、开关，
全部在 :mod:`localai.config` 里。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

# 与 IndexTTS 上游支持的语言一致。
SUPPORTED_LANGUAGES = ("zh", "en", "ja", "es", "ar")

# 主权重文件（与 engine/engine.py、IndexTTS 节点保持一致）。
REQUIRED_MODEL_FILES = (
    "config.yaml",
    "gpt.pth",
    "s2mel.pth",
    "codec.pth",
    "wav2vec2bert_stats.pt",
    "feat1.pt",
    "feat2.pt",
    "multilingual_zh_ja_yue_char_del.tiktoken",
)

# 独立引擎还需要这些辅助模型；缺了上游会尝试联网下载。
REQUIRED_AUX_FILES = (
    r"hf_cache\w2v-bert-2.0\config.json",
    r"hf_cache\w2v-bert-2.0\model.safetensors",
    r"hf_cache\w2v-bert-2.0\preprocessor_config.json",
    r"hf_cache\campplus_cn_common.bin",
    r"hf_cache\bigvgan\config.json",
    r"hf_cache\bigvgan\bigvgan_generator.pt",
)

# TTS 单条请求文本上限（与旧接口的 pydantic 校验一致）。
TTS_MAX_TEXT = 5000

# ---------------------------------------------------------------------------
# 音频 / 媒体格式
# ---------------------------------------------------------------------------

# 参考音色与音频上传接受的扩展名。
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac",
}

# ffmpeg 能解的音频容器。
MEDIA_AUDIO_EXTENSIONS = AUDIO_EXTENSIONS | {".wma", ".amr", ".aiff", ".aif", ".ape", ".m4b"}

# ffmpeg 能解的视频容器。
MEDIA_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".m4v", ".ts",
    ".mpg", ".mpeg", ".wmv", ".3gp", ".rmvb", ".vob",
}

# /asr 与 /media/* 统一接受的扩展名。
MEDIA_EXTENSIONS = MEDIA_AUDIO_EXTENSIONS | MEDIA_VIDEO_EXTENSIONS

# 视频上传上限 200MB；音色/声纹上传上限 50MB。
MEDIA_MAX_BYTES = 200 * 1024 * 1024
AUDIO_MAX_BYTES = 50 * 1024 * 1024

# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

ASR_LANGUAGES = ("auto", "zh", "en", "ja", "ko", "yue", "nospeech")

#: 识别引擎：SenseVoice（默认，带声纹）与 faster-whisper（原生时间戳）。
ASR_ENGINES = ("sensevoice", "whisper")
#: 请求里可用的 engine 取值；``auto`` 表示按「是否要时间戳 / 配置默认」决定。
ASR_ENGINE_CHOICES = ("sensevoice", "whisper", "auto")


#: faster-whisper 支持的档位（与 engine/whisper_engine.py 保持一致）。
WHISPER_SIZES = ("tiny", "base", "small", "medium", "large-v3")

#: 各档权重的显存初始估算（MB）；之后由显存管家按实测增量校准。
WHISPER_ESTIMATE_MB = {
    "tiny": 150,
    "base": 250,
    "small": 600,
    "medium": 1600,
    "large-v3": 3500,
}

# ---------------------------------------------------------------------------
# 翻译（NLLB-200）
# ---------------------------------------------------------------------------

#: 单条翻译文本上限（字符）。
TRANSLATE_MAX_TEXT = 5000
#: 单批 texts 条数上限。
TRANSLATE_BATCH_MAX = 500
#: 单批 texts 总字符上限（200k）。
TRANSLATE_BATCH_CHARS = 200_000
#: 一次推理最多多少块（防止单条超长文本切出过多块）。
TRANSLATE_MAX_CHUNKS = 2000
#: 翻译引擎可选的实现名（目前只有 NLLB，保留 engine 字段便于以后扩展）。
TRANSLATE_ENGINES = ("nllb",)
#: NLLB-200-distilled-600M 的显存初始估算（MB）；之后由显存管家按实测校准。
TRANSLATE_ESTIMATE_MB = 1500

# ---------------------------------------------------------------------------
# 人声分离
# ---------------------------------------------------------------------------

SEPARATE_MODELS = ("htdemucs", "htdemucs_ft", "htdemucs_6s")

SEPARATE_STEMS = {
    "htdemucs": ("vocals", "drums", "bass", "other"),
    "htdemucs_ft": ("vocals", "drums", "bass", "other"),
    "htdemucs_6s": ("vocals", "drums", "bass", "other", "guitar", "piano"),
}

# ---------------------------------------------------------------------------
# 模型下载（ModelScope 优先）
# ---------------------------------------------------------------------------

MODEL_SPECS = {
    "tts": {
        "label": "IndexTTS-2.5",
        "modelscope": "IndexTeam/IndexTTS-2.5",
        "huggingface": "IndexTeam/IndexTTS-2.5",
        "required": REQUIRED_MODEL_FILES,
    },
    "asr": {
        "label": "SenseVoiceSmall",
        "modelscope": "iic/SenseVoiceSmall",
        "huggingface": "FunAudioLLM/SenseVoiceSmall",
        "required": (
            "model.pt",
            "config.yaml",
            "configuration.json",
            "am.mvn",
            "tokens.json",
            "chn_jpn_yue_eng_ko_spectok.bpe.model",
        ),
    },
    "speaker": {
        "label": "campplus",
        "modelscope": "iic/speech_campplus_sv_zh-cn_16k-common",
        "huggingface": "funasr/campplus",
        "required": ("campplus_cn_common.bin", "config.yaml"),
    },
    "separate": {
        "label": "Demucs htdemucs / htdemucs_6s",
        "modelscope": None,
        "huggingface": None,
        "required": (),
    },
}
