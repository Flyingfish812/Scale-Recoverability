"""
Luna — Scale-Resolved Field Reconstruction Evaluation Framework
==============================================================

公共层 (Common Layer): 核心数据结构、小波操作、POD 分解、模型定义、配置管理。
不包含任何可执行脚本逻辑，只提供可复用的函数和类。

Layer architecture:
    luna/core       — 基础类型与常量
    luna/data       — 数据 IO、数据集注册、掩码生成
    luna/wavelet    — 小波变换、频带分解、误差度量
    luna/pod        — POD 分解、投影、频带 POD
    luna/models     — 神经网络模型定义 (VCNN, Ridge, MLP)
    luna/config     — 配置 schema 与加载器
"""

__version__ = "2.0.0"
