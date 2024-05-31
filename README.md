# 智能营销助手


## 📝目录

- [📖 简介](#-简介)
- [🚀 更新](#-更新)
- [🧾 任务](#-任务)
- [🛠️ 使用方法](#-使用方法)
  - [环境准备](#-环境准备)
    - [基础环境准备](#-基础环境准备)
    - [虚拟环境准备](#-虚拟环境准备)
  - [系统运行](#-系统运行)
- [💕 致谢](#-致谢)

## 📖 简介

众所周知，获客、活客、留客是电商行业的三大难题，谁拥有跟客户最佳的沟通方式，谁就拥有客户。

随着用户消费逐渐转移至线上，电商行业面临以下一些问题：

* 用户交流体验差
* 商品推荐不精准
* 客户转化率低
* 退换货频率高
* 物流成本高

在这样的背景下，未来销售的引擎——大模型加持的智能营销助手就诞生了。

它能够与用户的对话，了解用户的需求，基于多模态的AIGC生成能力，持续输出更符合用户消费习惯的文本、图片和视频等营销内容，推荐符合用户的商品，将营销与经营结合。



如果您觉得这个项目还不错，欢迎⭐Star，让更多的人发现它！

## 🚀 更新

🚀 [智能营销助手GPU版](https://openxlab.org.cn/apps/detail/AI-Labs/IntelligentSalesAssistant) 🚀 [智能营销助手CPU版](https://openxlab.org.cn/apps/detail/AI-Labs/IntelligentSalesAssistant-CPU) 🚀  

[2024.05.15] 语音合成功能、商品咨询功能  
[2024.05.10] 图片生成功能  
[2024.05.05] 基于协同过滤的商品推荐  
[2024.05.01] 基于商品平铺图生成营销文案、 商品信息录入、基于商品信息创建知识库  
[2024.04.25] 智能营销助手第一版部署上线  


## 🧾 任务

- [x] 文本对话功能
- [x] 语音对话功能
- [x] 图片生成功能
- [x] 商品管理、基于商品信息的知识库创建
- [ ] 多模态数据处理
  - [x] 使用浦语·灵笔基于商品平铺图生成营销文案
- [x] 基于协同过滤的商品推荐
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

### 系统运行

```bash
streamlit run app.py
```

> 待继续完成~

## 💕 致谢

<div align="center">

***感谢 上海人工智能实验室 组织的书生·浦语大模型实战营学习活动 和 提供的强大算力支持~***

***感谢 OpenXLab 对项目部署的算力支持~***

***感谢 浦语小助手 对项目的支持~***

[**InternStudio**](https://studio.intern-ai.org.cn/)、[**Tutorial**](https://github.com/InternLM/tutorial)、[**XTuner**](https://github.com/InternLM/xtuner)、[**InternLM-XComposer2**](https://github.com/InternLM/InternLM-XComposer)、[**Lagent**](https://github.com/InternLM/lagent)、[**InternLM-Math**](https://github.com/InternLM/InternLM-Math)

</div>