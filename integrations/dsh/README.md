# DSH 语音插件（占位）

> **本目录只有文档，没有实现代码。**
> 第二层（DSH 插件）现在只定契约，不在本仓库实现。

## 计划

未来的 DSH 语音插件放在独立仓库/独立包中，只通过 HTTP 调用本服务层：

- 自动朗读助手回复：`POST /tts`（长文本分段、串行、失败降级）；
- 语音条转文字：`POST /asr`（服务端自动转 16k）；
- 声纹向量：`POST /speakers/embed`（只返回向量，不判断是谁）；
- 声音选择/音色管理：`GET /voices`、`POST /api/v1/upload`；
- 能力发现与状态：`GET /capabilities`、`GET /health`、`GET /vram`。

完整约定见 [`../../docs/PLUGIN_CONTRACT.md`](../../docs/PLUGIN_CONTRACT.md)。

## 明确不做的事

- 不 import torch/funasr/demucs/index_tts；
- 不加载/卸载模型，不管理显存；
- 不直接读写 `models/`、`voices/`、`outputs/`；
- 不在服务不可用时阻塞 UI 或无限重试。

## 当前状态

- 服务层已就绪（v2.2.0，API v1），接口契约稳定；
- 插件层尚未开始实现；
- 实现时以 `docs/PLUGIN_CONTRACT.md` 为准，如需改契约先改文档。
