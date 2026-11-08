"""翻译能力单测：语言码 / 长文本切分与拼回 / 批量顺序 / 超限 / 自动检测。

默认 **不加载 2.4GB 权重**：
- 纯函数（语言码、切分、拼回）直接测；
- 批量翻译用 monkeypatch 替换 ``ensure_loaded`` 与 ``_translate_chunks``，
  只验证切分、顺序、条数、参数传递；
- 真实模型端到端验证放在 ``@pytest.mark.slow``（默认跳过），
  实际跑过的记录见 ``docs/SERVICE_API.md`` 的实测表格。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from localai.core.translate import normalize_engine, normalize_texts
from localai.engines._native import import_engine_module


@pytest.fixture(scope="module")
def engine():
    """原生翻译引擎模块（只 import，不加载模型、不联网）。"""
    return import_engine_module("translate_engine")


# ---------------------------------------------------------------------------
# 语言码
# ---------------------------------------------------------------------------


def test_normalize_lang_aliases(engine) -> None:
    assert engine.normalize_lang("zh") == "zho_Hans"
    assert engine.normalize_lang("zh-cn") == "zho_Hans"
    assert engine.normalize_lang("zh-tw") == "zho_Hant"
    assert engine.normalize_lang("en") == "eng_Latn"
    assert engine.normalize_lang("ja") == "jpn_Jpan"
    assert engine.normalize_lang("ko") == "kor_Hang"
    assert engine.normalize_lang("fr") == "fra_Latn"
    # 直接给 NLLB 码也要能用
    assert engine.normalize_lang("zho_Hans") == "zho_Hans"
    assert engine.normalize_lang("eng-latn") == "eng_Latn"
    # auto 只在允许时原样返回
    assert engine.normalize_lang("auto", allow_auto=True) == "auto"
    with pytest.raises(engine.LangError):
        engine.normalize_lang("auto")


def test_normalize_lang_unknown_lists_supported(engine) -> None:
    with pytest.raises(engine.LangError) as info:
        engine.normalize_lang("klingon")
    message = str(info.value)
    assert "不支持的语言" in message
    assert "zho_Hans" in message and "eng_Latn" in message
    assert "zho_Hans" in info.value.supported


def test_detected_iso_maps_to_nllb(engine) -> None:
    assert engine.ISO_TO_NLLB["zh-cn"] == "zho_Hans"
    assert engine.ISO_TO_NLLB["zh-tw"] == "zho_Hant"
    assert engine.ISO_TO_NLLB["en"] == "eng_Latn"
    assert engine.ISO_TO_NLLB["ja"] == "jpn_Jpan"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("今天天气不错，我们出去走走吧。明天可能会下雨，记得带伞。", "zho_Hans"),
        ("This is a short English sentence for language detection.", "eng_Latn"),
        ("これは日本語の文章です。言語検出のテストをしています。", "jpn_Jpan"),
        ("Ceci est une phrase française pour tester la détection de langue.", "fra_Latn"),
    ],
)
def test_detect_language_common(engine, text: str, expected: str) -> None:
    assert engine.detect_language(text) == expected


# ---------------------------------------------------------------------------
# 长文本切分 / 拼回
# ---------------------------------------------------------------------------


def test_split_plan_keeps_paragraphs(engine) -> None:
    text = "第一段第一句。第一段第二句！\n\n第二段只有一句。\n第三段：还有一句；最后一句。"
    plan = engine.split_plan(text, max_len=400)
    text_blocks = [block for block in plan if block.kind == "text"]
    sep_blocks = [block for block in plan if block.kind == "sep"]
    assert len(text_blocks) == 3, "三段必须都在"
    assert [sep.text for sep in sep_blocks] == ["\n\n", "\n"]
    chunks = engine.plan_chunks(plan)
    # 段落没丢、句子没漏（切分不会删字符）
    assert "第一段第一句" in "".join(chunks)
    assert "第二段只有一句" in "".join(chunks)
    assert "最后一句" in "".join(chunks)


def test_split_plan_respects_max_len(engine) -> None:
    text = "这是第一句。这是第二句！" + "很长的一段话" * 30 + "。\n\nTail paragraph."
    plan = engine.split_plan(text, max_len=40)  # 默认按字符保守估算
    chunks = engine.plan_chunks(plan)
    assert chunks
    assert all(len(chunk) <= 40 for chunk in chunks), [len(c) for c in chunks]


def test_long_sentence_is_hard_split_without_loss(engine) -> None:
    text = "a" * 123
    plan = engine.split_plan(text, max_len=50)
    chunks = engine.plan_chunks(plan)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert "".join(chunks) == text, "硬切后拼回必须一字不差"


def test_render_plan_preserves_structure(engine) -> None:
    text = "第一段第一句。第一段第二句。\n\n第二段。"
    # 用 1 个字符当 max_len，确保每个句子都被切开
    plan = engine.split_plan(text, max_len=8)
    count = len(engine.plan_chunks(plan))
    translated = [f"[{index}]" for index in range(count)]
    rendered = engine.render_plan(plan, translated, joiner="")
    assert rendered.startswith("[0]")
    assert "\n\n" in rendered, "段落之间的换行必须保留"
    assert rendered.count("[") == count, "每个块都要出现在结果里"
    # 中文目标不加空格；英文目标用空格连接
    spaced = engine.render_plan(plan, translated, joiner=" ")
    assert spaced == "[0] [1]\n\n[2]", spaced
    with pytest.raises(engine.TranslateError):
        engine.render_plan(plan, translated[:-1])


# ---------------------------------------------------------------------------
# 批量翻译（mock 掉模型，只测顺序与参数）
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """只要支持 ``tokenizer(text).input_ids`` 就够了。"""

    src_lang = ""

    def __call__(self, text: str, **_: object) -> SimpleNamespace:
        return SimpleNamespace(input_ids=list(range(len(text))))


@pytest.fixture
def fake_model(engine, monkeypatch):
    calls: dict[str, object] = {}

    def fake_ensure_loaded():
        return None, _FakeTokenizer()

    def fake_translate_chunks(
        chunks,
        src_lang,
        tgt_lang,
        *,
        batch_size,
        num_beams,
        max_len,
    ):
        calls.update(
            {
                "chunks": list(chunks),
                "src": src_lang,
                "tgt": tgt_lang,
                "batch_size": batch_size,
                "num_beams": num_beams,
                "max_len": max_len,
            }
        )
        return [f"<{chunk}>" for chunk in chunks]

    monkeypatch.setattr(engine, "ensure_loaded", fake_ensure_loaded)
    monkeypatch.setattr(engine, "_translate_chunks", fake_translate_chunks)
    return calls


def test_translate_batch_order_and_count(engine, fake_model) -> None:
    texts = ["甲。", "乙。", "丙。"]
    result = engine.translate(
        texts, src_lang="zh", tgt_lang="en", batch_size=4, num_beams=2
    )
    assert result == ["<甲。>", "<乙。>", "<丙。>"]
    assert fake_model["src"] == "zho_Hans"
    assert fake_model["tgt"] == "eng_Latn"
    assert fake_model["batch_size"] == 4
    assert fake_model["num_beams"] == 2


def test_translate_documents_reports_chunks(engine, fake_model) -> None:
    texts = ["短句。", "这是一个很长的句子" * 10]
    docs = engine.translate_documents(texts, src_lang="zh", tgt_lang="en")
    assert [doc["chunks"] for doc in docs] == [1, 1]
    assert docs[0]["text"] == "<短句。>"
    assert docs[0]["source"] == "zho_Hans"


def test_translate_source_auto_uses_detection(engine, fake_model) -> None:
    result = engine.translate(["今天天气不错。"], src_lang="auto", tgt_lang="en")
    assert result == ["<今天天气不错。>"]
    assert fake_model["src"] == "zho_Hans"


def test_translate_same_language_returns_input(engine) -> None:
    texts = ["原样返回。", ""]
    assert engine.translate(texts, src_lang="zh", tgt_lang="zh") == texts


# ---------------------------------------------------------------------------
# 接口参数上限（超限必须是 400，而不是 422）
# ---------------------------------------------------------------------------


def test_normalize_texts_limits() -> None:
    assert normalize_texts("你好", single=True) == ["你好"]
    assert normalize_texts(["a", "b"]) == ["a", "b"]
    with pytest.raises(ValueError, match="上限 5000"):
        normalize_texts("字" * 5001, single=True)
    with pytest.raises(ValueError, match="500 条"):
        normalize_texts(["x"] * 501)
    with pytest.raises(ValueError, match="200000 字"):
        normalize_texts(["x" * 5000] * 41)  # 205k 字
    with pytest.raises(ValueError, match="不能为空"):
        normalize_texts("   ", single=True)
    with pytest.raises(ValueError, match="不支持"):
        normalize_engine("google")
    assert normalize_engine(None) == "nllb"


# ---------------------------------------------------------------------------
# 真实模型（默认跳过；-m slow 才会跑）
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_translate_zh_to_en(engine) -> None:
    status = engine.model_status()
    if not status.get("ready"):
        pytest.skip(f"NLLB 权重未准备：{status.get('model_dir')}")
    try:
        result = engine.translate(
            ["今天天气不错，我们出去走走吧。"], src_lang="zh", tgt_lang="en"
        )
    finally:
        engine.unload()
    assert result and result[0]
    assert any(word in result[0].lower() for word in ("weather", "walk", "go"))
