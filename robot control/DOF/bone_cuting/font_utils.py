"""
Polyscope 中文字体工具

解决 Polyscope (Dear ImGui) 中文显示为问号的问题。

注意：由于 Polyscope Python 绑定的 ImGui 接口限制，
中文字体加载可能无法完全解决所有中文显示问题。
因此，建议同时将 UI 中的名称改为英文，确保无乱码。

用法：
    from font_utils import setup_chinese_font
    setup_chinese_font()
"""

import os
import sys


def _find_chinese_font():
    """
    查找系统中的中文字体文件。

    Returns:
        (font_name, font_path) 元组，如果找不到返回 (None, None)
    """
    if sys.platform == "win32":
        font_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
        candidates = [
            ("SimHei", os.path.join(font_dir, "simhei.ttf")),
            ("Microsoft YaHei", os.path.join(font_dir, "msyh.ttc")),
            ("Microsoft YaHei", os.path.join(font_dir, "msyh.ttf")),
            ("SimSun", os.path.join(font_dir, "simsun.ttc")),
            ("SimSun", os.path.join(font_dir, "simsun.ttf")),
        ]
    elif sys.platform == "darwin":
        font_dir = "/System/Library/Fonts"
        candidates = [
            ("PingFang SC", os.path.join(font_dir, "PingFang.ttc")),
            ("Heiti SC", os.path.join(font_dir, "STHeiti Medium.ttc")),
            ("Songti SC", os.path.join(font_dir, "STSongti-SC-Regular.otf")),
        ]
    else:
        font_dir = "/usr/share/fonts"
        candidates = [
            ("Noto Sans CJK SC", os.path.join(font_dir, "opentype/noto/NotoSansCJK-Regular.ttc")),
            ("WenQuanYi Zen Hei", os.path.join(font_dir, "truetype/wqy/wqy-zenhei.ttc")),
            ("Source Han Sans CN", os.path.join(font_dir, "opentype/source-han-sans/SourceHanSansSC-Regular.otf")),
        ]

    for name, path in candidates:
        if os.path.exists(path):
            return name, path

    return None, None


def setup_chinese_font(font_size=15.0, verbose=False):
    """
    设置 Polyscope 的中文字体。

    在 ps.init() 之后、ps.show() 之前调用。

    注意：由于 Polyscope Python 绑定限制，中文字体加载可能无法完全
    解决所有中文显示问题。建议同时使用英文名称命名 UI 元素。

    Args:
        font_size: 字体大小（像素）
        verbose: 是否打印详细日志

    Returns:
        bool: 是否成功设置中文字体回调
    """
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError:
        if verbose:
            print("[font_utils] Polyscope 未安装，跳过字体设置")
        return False

    font_name, font_path = _find_chinese_font()
    if font_path is None:
        if verbose:
            print("[font_utils] 未找到系统中文字体，跳过字体设置")
        return False

    def _prepare_fonts(*args):
        """字体准备回调"""
        try:
            io = psim.GetIO()
            fonts = io.Fonts

            chinese_font = fonts.AddFontFromFileTTF(font_path, font_size)
            if chinese_font is not None and chinese_font.IsLoaded():
                try:
                    io.FontDefault = chinese_font
                    if verbose:
                        print(f"[font_utils] 已加载中文字体: {font_name}")
                except Exception:
                    if verbose:
                        print(f"[font_utils] 字体加载成功，但设置默认字体失败: {font_name}")
            else:
                if verbose:
                    print(f"[font_utils] 字体加载失败: {font_name}")
        except Exception as e:
            if verbose:
                print(f"[font_utils] 字体设置异常: {e}")

    try:
        ps.set_prepare_imgui_fonts_callback(_prepare_fonts)
        if verbose:
            print(f"[font_utils] 字体回调已设置: {font_name}")
        return True
    except Exception as e:
        if verbose:
            print(f"[font_utils] 设置字体回调失败: {e}")
        return False
