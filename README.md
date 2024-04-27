# 智能营销助手


## 📝目录

- [📖 简介](#-简介)
- [🚀 更新](#-更新)
- [🧾 任务](#-任务)
- [🛠️ 使用方法](#-使用方法)

## 📖 简介

智能营销助手，主要功能是基于与用户的对话，了解用户的需求，以此推荐符合用户的商品。目前是基于 InternLM2-Chat-1_8B 模型，通过 XTuner 微调。后续计划基于 InternLM2-Chat-7B 模型进行微调。

如果您觉得这个项目还不错，欢迎⭐Star，让更多的人发现它！

## 🚀 更新

[2024.04.25] 智能营销助手第一版部署上线  🚀 [智能营销助手GPU版](https://openxlab.org.cn/apps/detail/AI-Labs/IntelligentSalesAssistant)  🚀 [智能营销助手CPU版](https://openxlab.org.cn/apps/detail/AI-Labs/IntelligentSalesAssistant-CPU) 🚀


## 🧾 任务

- [x] 文本对话功能
- [x] 语音对话功能
- [ ] 多模态数据处理
  - [x] 使用浦语·灵笔基于商品平铺图生成营销文案
- [ ] Lagent工具调用
- [ ] RAG检索
- [ ] 模型持续微调

## 🛠️ 使用方法

### 环境准备

#### 基础环境准备

```bash
apt-get install -y tree tmux libaio-dev ffmpeg git-lfs sqlite3
```

#### 虚拟环境准备

```bash
# 创建虚拟环境
studio-conda -t sales -o internlm-base
# 激活虚拟环境
conda activate sales
# 安装必要依赖
pip install -r requirements.txt
```

> 待继续完成~

## 💕 致谢

<div align="center">

***感谢 上海人工智能实验室 组织的书生·浦语大模型实战营学习活动 和 提供的强大算力支持~***

***感谢 OpenXLab 对项目部署的算力支持~***

***感谢 浦语小助手 对项目的支持~***

[**InternStudio**](https://studio.intern-ai.org.cn/)、[**Tutorial**](https://github.com/InternLM/tutorial)、[**XTuner**](https://github.com/InternLM/xtuner)、[**InternLM-XComposer2**](https://github.com/InternLM/InternLM-XComposer)、[**Lagent**](https://github.com/InternLM/lagent)、[**InternLM-Math**](https://github.com/InternLM/InternLM-Math)

</div>