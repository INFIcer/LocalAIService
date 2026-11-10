"""日志编码探测的回归测试。

背景：service.out.log 曾经因为被截断/稀疏写留下一大片 NUL 空洞，尾部读取
窗口正好落进空洞，旧实现只看「NUL 占比 > 25%」就判定成 UTF-16，整段日志解
成 \\x00 乱码，WebUI 的「最近日志」面板跟着全变乱码。
"""

from localai.admin.system import _decode_log


def test_utf8_with_nul_hole_is_not_misread_as_utf16() -> None:
    data = "前一段时间\n".encode() + b"\x00" * 5000 + "后面的日志\n".encode()

    text, encoding = _decode_log(data)

    assert encoding == "utf-8"
    assert "\x00" not in text
    assert "前一段时间" in text
    assert "后面的日志" in text


def test_utf8_hole_only_window() -> None:
    data = b"\x00" * 4096 + "尾部日志\n".encode()

    text, encoding = _decode_log(data)

    assert encoding == "utf-8"
    assert text.strip() == "尾部日志"


def test_utf16_le_without_bom_still_detected() -> None:
    data = "INFO start\nINFO ok\n".encode("utf-16-le")

    text, encoding = _decode_log(data)

    assert encoding == "utf-16-le"
    assert "INFO start" in text


def test_utf16_be_without_bom_still_detected() -> None:
    data = "INFO start\nINFO ok\n".encode("utf-16-be")

    text, encoding = _decode_log(data)

    assert encoding == "utf-16-be"
    assert "INFO start" in text


def test_utf16_with_bom() -> None:
    data = "INFO start\n".encode("utf-16-le")

    _, encoding = _decode_log(data)

    assert encoding == "utf-16-le"


def test_utf8_with_bom() -> None:
    data = "日志\n".encode("utf-8-sig")

    text, encoding = _decode_log(data)

    assert encoding == "utf-8-sig"
    assert text.strip() == "日志"


def test_gbk_fallback() -> None:
    data = "合成完成\n".encode("gbk")

    text, encoding = _decode_log(data)

    assert encoding == "gbk"
    assert "合成完成" in text


def test_utf8_tail_cut_inside_multibyte_char() -> None:
    raw = "任务完成\n".encode()
    data = raw[1:]  # 模拟尾部窗口从某个中文字符中间开始

    text, encoding = _decode_log(data)

    assert encoding == "utf-8"
    # 被切断的第一个字丢掉，后面的内容不能变乱码。
    assert text.strip() == "务完成"
