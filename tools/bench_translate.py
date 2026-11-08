"""NLLB 翻译能力实测脚本（质量 / 速度 / 长文本结构 / 显存）。

用法::

    E:\\LocalAIService\\engine\\.venv\\Scripts\\python.exe E:\\LocalAIService\\tools\\bench_translate.py

结果同时打印并写到 ``outputs/translate_bench.json``（运行产物，不入库）。
这个脚本只用于人工抽检与文档表格，不属于服务运行时依赖。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

import translate_engine as te  # noqa: E402

OUTPUT = ROOT / "outputs" / "translate_bench.json"

# 20 条短句：口语 / 书面 / 术语混合，中英日三种目标都从这批里取。
ZH_SENTENCES = [
    "今天天气不错，我们出去走走吧。",
    "你吃饭了吗？我有点饿了。",
    "这个方案我觉得还得再讨论一下。",
    "他昨天加班到很晚，今天状态不太好。",
    "请把这份文件打印三份，下午开会要用。",
    "人工智能正在深刻改变我们的生活方式。",
    "该系统采用分布式架构，支持水平扩展。",
    "请在提交代码前运行单元测试，确保没有回归。",
    "这家餐厅的招牌菜是红烧肉，味道很地道。",
    "不好意思，我可能听错了，你能再说一遍吗？",
    "由于天气原因，航班将延误大约两个小时。",
    "学习一门新语言需要长期坚持和大量练习。",
    "合同期满后，双方可以协商续签。",
    "这次实验的结果与理论预测基本一致。",
    "请把空调温度调高一点，我有点冷。",
    "他从小就喜欢拆装各种机械装置。",
    "城市轨道交通的客流量在早晚高峰达到峰值。",
    "我们需要在成本和性能之间找到平衡点。",
    "这部电影的叙事节奏很慢，但画面非常美。",
    "如果有任何问题，请随时联系客服。",
]

EN_SENTENCES = [
    "The weather is nice today; let's go for a walk.",
    "Have you eaten yet? I'm a little hungry.",
    "I think this proposal needs further discussion.",
    "He worked late yesterday and isn't at his best today.",
    "Please print three copies of this document for the afternoon meeting.",
    "Artificial intelligence is profoundly changing the way we live.",
    "The system uses a distributed architecture and supports horizontal scaling.",
    "Please run the unit tests before committing code to make sure nothing regressed.",
    "The restaurant's signature dish is braised pork, and it tastes authentic.",
    "Sorry, I may have misheard you. Could you say that again?",
    "The flight will be delayed by about two hours due to weather.",
    "Learning a new language requires long-term persistence and plenty of practice.",
    "After the contract expires, both parties may negotiate a renewal.",
    "The experimental results are basically consistent with theoretical predictions.",
    "Please turn the air conditioning up a bit; I'm a little cold.",
    "He has enjoyed taking apart and reassembling mechanical devices since childhood.",
    "Urban rail transit passenger flow peaks during the morning and evening rush hours.",
    "We need to find a balance between cost and performance.",
    "The film's narrative pace is slow, but the visuals are beautiful.",
    "If you have any questions, please contact customer service at any time.",
]

# 三段较长中文（300~800 字），用空行分段，验证长文本切分与结构保留。
LONG_DOCS = [
    (
        "人工智能的发展正在改变许多行业的运作方式。过去，人们只能依靠人工完成重复性的"
        "数据处理工作，如今越来越多的流程可以由模型自动完成。不过，技术并不是万能的，"
        "它更像是一种工具，需要与人的判断结合起来，才能真正产生价值。\n\n"
        "在具体落地时，团队往往会遇到数据质量、隐私保护和系统集成等问题。数据质量决定了"
        "模型效果的上限，如果训练数据存在偏差，模型输出的结果也会带上同样的偏差。因此，"
        "在项目开始之前，花时间梳理数据来源、清洗规则和标注标准，通常比匆忙训练一个模型"
        "更有意义。\n\n"
        "此外，部署之后的持续监控同样重要。模型的效果会随着输入分布的变化而衰减，只有建立"
        "起完善的评估和反馈机制，才能及时发现问题并迭代改进。对于使用者来说，理解系统的"
        "能力边界，比盲目相信模型给出的每一个答案更加重要。"
    ),
    (
        "这家公司成立于二〇一五年，最初只有十几名员工，办公地点在一栋老旧写字楼的顶层。"
        "创始团队当时判断，中小企业对数字化工具的需求会持续增长，但市场上缺少价格合理、"
        "上手简单的产品。于是他们决定从最基础的库存管理切入，先做一个足够小、但足够好用的"
        "模块。\n\n"
        "产品上线后的第一年并不顺利。客户抱怨界面复杂，销售团队也说不清楚产品到底解决了"
        "什么问题。团队花了几个月时间，一家一家地拜访客户，把反馈记在本子上，再逐条改进。"
        "到了第三年，续费率终于稳定在百分之九十以上，公司也开始进入盈利阶段。\n\n"
        "回顾这段经历，创始人常说，最重要的并不是技术有多先进，而是愿不愿意认真听客户说话。"
        "很多看起来不起眼的小问题，恰恰是用户每天都会遇到的痛点。把这些问题解决好，产品"
        "自然就有了生命力。"
    ),
    (
        "研究人员在实验中发现，睡眠不足会显著影响人的注意力和记忆能力。参与者被要求在"
        "连续两天只睡四小时的情况下完成一系列认知任务，结果显示，他们的反应速度平均下降了"
        "约百分之二十，错误率则明显上升。更有意思的是，参与者普遍高估了自己的表现，认为"
        "自己仍然能够集中注意力。\n\n"
        "进一步的分析表明，睡眠不足对大脑前额叶区域的影响尤为明显。这个区域负责计划、"
        "决策和抑制冲动，当它得不到充分休息时，人更容易做出短视的选择。研究团队因此建议，"
        "在需要高度专注的工作之前，应当优先保证充足的睡眠，而不是依赖咖啡因强行提神。\n\n"
        "当然，这项研究也有局限。样本量相对较小，实验周期较短，而且参与者的年龄集中在"
        "二十到三十岁之间。未来的研究需要纳入更多不同年龄和职业的人群，才能判断这些结论"
        "是否具有普遍性。"
    ),
]


def vram_mb() -> tuple[float, float]:
    import torch

    if not torch.cuda.is_available():
        return (0.0, 0.0)
    free, total = torch.cuda.mem_get_info()
    return (round(free / 1024**2, 0), round(total / 1024**2, 0))


def count_paragraphs(text: str) -> int:
    return len([part for part in text.split("\n\n") if part.strip()])


def count_sentences(text: str) -> int:
    return len(
        [
            sentence
            for sentence, _gap in te._split_sentences(text)
            if sentence.strip()
        ]
    )


def main() -> int:
    results: dict[str, object] = {}

    print("模型状态（加载前）：", json.dumps(te.model_status(), ensure_ascii=False))
    free_before, total = vram_mb()
    print(f"加载前显存：空闲 {free_before:.0f} / {total:.0f} MB")

    started = time.perf_counter()
    te.ensure_loaded()
    load_seconds = time.perf_counter() - started
    free_loaded, _ = vram_mb()
    print(
        f"加载耗时 {load_seconds:.2f}s；加载后空闲 {free_loaded:.0f} MB，"
        f"增量 {free_before - free_loaded:.0f} MB"
    )
    results["load_seconds"] = round(load_seconds, 2)
    results["vram"] = {
        "total_mb": total,
        "before_free_mb": free_before,
        "after_load_free_mb": free_loaded,
        "delta_mb": round(free_before - free_loaded),
        "torch_allocated_mb": round(
            __import__("torch").cuda.memory_allocated() / 1024**2
        ),
    }

    # ---- 单条延迟 ----
    single_text = ZH_SENTENCES[0]
    te.translate([single_text], "zho_Hans", "eng_Latn", num_beams=2)
    single_times = []
    for _ in range(3):
        start = time.perf_counter()
        te.translate([single_text], "zho_Hans", "eng_Latn", num_beams=2)
        single_times.append(time.perf_counter() - start)
    results["single_latency"] = {
        "runs": [round(item, 3) for item in single_times],
        "avg_ms": round(sum(single_times) / len(single_times) * 1000),
    }
    print("单条延迟：", results["single_latency"])

    # ---- 20 句 × beams 1/2/4 ----
    beam_rows = {}
    for beams in (1, 2, 4):
        start = time.perf_counter()
        translated = te.translate(
            ZH_SENTENCES, "zho_Hans", "eng_Latn", batch_size=8, num_beams=beams
        )
        elapsed = time.perf_counter() - start
        beam_rows[str(beams)] = {
            "elapsed_s": round(elapsed, 3),
            "sentences_per_s": round(len(ZH_SENTENCES) / elapsed, 2),
        }
        if beams == 2:
            results["quality_zh_en"] = translated
        if beams == 4:
            results["quality_zh_en_beams4_sample"] = translated[:3]
        print(f"20 句 beams={beams}：{elapsed:.3f}s，{len(ZH_SENTENCES)/elapsed:.2f} 句/s")
    results["beam_bench"] = beam_rows

    # ---- 质量：英->中、中->日 ----
    start = time.perf_counter()
    results["quality_en_zh"] = te.translate(
        EN_SENTENCES, "eng_Latn", "zho_Hans", batch_size=8, num_beams=2
    )
    print(f"英->中 20 句：{time.perf_counter()-start:.3f}s")
    start = time.perf_counter()
    results["quality_zh_ja"] = te.translate(
        ZH_SENTENCES, "zho_Hans", "jpn_Jpan", batch_size=8, num_beams=2
    )
    print(f"中->日 20 句：{time.perf_counter()-start:.3f}s")

    # ---- 长文本结构 ----
    long_rows = []
    for index, doc in enumerate(LONG_DOCS, start=1):
        plan = te.split_plan(doc, max_len=400, count_tokens=te._token_counter(te._tokenizer))
        chunks = te.plan_chunks(plan)
        start = time.perf_counter()
        document = te.translate_documents(
            [doc], "zho_Hans", "eng_Latn", max_len=400, batch_size=8, num_beams=2
        )[0]
        elapsed = time.perf_counter() - start
        out = document["text"]
        row = {
            "doc": index,
            "chars": len(doc),
            "paragraphs_in": count_paragraphs(doc),
            "paragraphs_out": count_paragraphs(out),
            "sentences_in": count_sentences(doc),
            "newlines_in": doc.count("\n\n"),
            "newlines_out": out.count("\n\n"),
            "chunks": document["chunks"],
            "elapsed_s": round(elapsed, 3),
            "chars_out": len(out),
            "output": out,
        }
        # 段落结构必须一致：输出仍然有相同的空行分段数。
        row["structure_ok"] = row["paragraphs_in"] == row["paragraphs_out"]
        long_rows.append(row)
        print(
            f"长文 {index}：{len(doc)} 字 -> {document['chunks']} 块 / {elapsed:.2f}s；"
            f"段落 {row['paragraphs_in']}->{row['paragraphs_out']}，"
            f"空行 {row['newlines_in']}->{row['newlines_out']}，结构保持={row['structure_ok']}"
        )
    results["long_docs"] = long_rows

    # ---- 500 条吞吐 ----
    batch_500 = (ZH_SENTENCES * 25)[:500]
    start = time.perf_counter()
    out_500 = te.translate(
        batch_500, "zho_Hans", "eng_Latn", batch_size=8, num_beams=2
    )
    elapsed_500 = time.perf_counter() - start
    assert len(out_500) == 500 and all(out_500)
    free_after_batch, _ = vram_mb()
    results["batch_500"] = {
        "count": len(out_500),
        "elapsed_s": round(elapsed_500, 3),
        "items_per_s": round(500 / elapsed_500, 2),
        "chars_in": sum(len(text) for text in batch_500),
        "chars_per_s": round(
            sum(len(text) for text in batch_500) / elapsed_500, 1
        ),
        "vram_free_mb": free_after_batch,
        "vram_delta_after_batch_mb": round(free_loaded - free_after_batch),
    }
    print(
        f"500 条：{elapsed_500:.2f}s，{500/elapsed_500:.2f} 条/s；"
        f"批次后空闲 {free_after_batch:.0f} MB（相对加载后 {free_loaded - free_after_batch:+.0f} MB）"
    )

    # ---- 卸载 ----
    te.unload()
    time.sleep(1.0)
    free_after_unload, _ = vram_mb()
    results["vram"]["after_unload_free_mb"] = free_after_unload
    results["vram"]["unload_restored_mb"] = round(free_after_unload - free_loaded)
    print(f"卸载后空闲 {free_after_unload:.0f} MB（相对加载后 +{free_after_unload-free_loaded:.0f} MB）")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("结果写入：", OUTPUT)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
