<img width="1453" height="963" alt="image" src="https://github.com/user-attachments/assets/ab0927a9-1a2e-40fb-8398-230871b4edfa" />

**Sydney International Student AI Assistant**

一个面向澳洲留学生日常生活场景的 AI 助手，基于 **RAG（检索增强生成）+ 多模块 Agent 架构**，支持交通、手机套餐、医疗保险及大学事务等咨询，并具备多轮对话能力。

---

## 🚀 项目功能

* 📱 手机套餐推荐（预算 / 流量 / prepaid / postpaid）
* 🚇 公共交通指导（Opal 卡、优惠、出行建议）
* 🏥 医疗保险咨询（OSHC、就医流程）
* 🎓 学校事务问答（政策、流程等）
* 🤖 多轮对话支持（上下文记忆）
* 📚 多知识库语义检索问答

---

## 🧠 技术架构

核心架构：

用户问题
→ 意图路由（Intent Router）
→ 知识库检索（FAISS + Embedding）
→ LLM 推理生成回答
→ Web Demo 展示

技术栈：

* Python + Streamlit（前端展示）
* FAISS 向量数据库
* SentenceTransformers Embedding
* DashScope / LLM API
* Session Memory 多轮对话
* GitHub 项目管理与部署

---

## ⚙️ 本地运行方法

### 1️⃣ 克隆项目

```bash
git clone https://github.com/你的用户名/sydney-student-agent.git
cd sydney-student-agent
```

### 2️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 3️⃣ 配置 API Key（可选）

创建 `.env` 文件：

```
DASHSCOPE_API_KEY=你的key
```

### 4️⃣ 启动项目

```bash
streamlit run app.py
```

---

## 📂 知识库结构

```
data/
 ├ transport/
 ├ phone/
 ├ healthcare/
 └ uni/
```

向量索引文件已忽略 Git 上传。

---

## 🧪 示例提问

* 悉尼30刀以内手机套餐推荐？
* Opal 卡学生优惠怎么办？
* OSHC 医疗保险怎么报销？
* 学校政策相关问题？

---

## 📌 后续优化方向

* 更稳定的 Intent Router
* 更丰富知识库数据
* Demo 云端部署优化
* 引用来源展示
* Agent 工作流优化

---

## 👩‍💻 作者

Bellis




