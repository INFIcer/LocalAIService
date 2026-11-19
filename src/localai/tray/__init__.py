"""Windows 通知区托盘启动器（零依赖实现）。

- actions: 纯动作层（URL、菜单契约、探测、进程收尾），可单测；
- winicon: ctypes 直调 Shell_NotifyIcon 的 UI 层，只在 Windows 上导入；
- app: 入口，负责配置、日志、单实例与依赖装配。
"""
