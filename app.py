import os
import glob
import json
from dataclasses import dataclass
from typing import List, Tuple
import dashscope


import numpy as np
import streamlit as st
from dotenv import load_dotenv

import faiss
from sentence_transformers import SentenceTransformer

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_module" not in st.session_state:
    st.session_state.last_module = "transport"

if "slot_state" not in st.session_state:
    st.session_state.slot_state = {}

# Optional LLM
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


load_dotenv()

INDEX_DIR = "embeddings"

MODULES = {
    "transport": "data/transport",
    "phone": "data/phone",
    "healthcare": "data/healthcare",
    "uni": "data/uni",
}

MODULE_SYSTEM_PROMPTS = {
    "transport": "你是悉尼公共交通专家。回答要给可操作步骤、卡种/费用/注意事项，尽量用要点列出。",
    "phone": "你是澳洲手机卡顾问。先问清需求（预算/流量/覆盖/是否合约），再给推荐方案对比：Optus/Telstra/Vodafone + prepaid/postpaid。",
    "healthcare": "你是留学生医疗与OSHC顾问。解释流程、报销步骤、GP/急诊区别、紧急电话，给清晰步骤和注意事项。",
    "uni": "你是UNSW学生事务顾问。回答要包含规则点、可能后果、建议做法、下一步联系对象（Student Services等）。",
}

MODULE_REQUIREMENTS = {
    "transport": {
        "required": ["起点", "终点"],
        "optional": ["是否携带行李", "时间(几点出发)", "预算"],
        "question": "我可以帮你规划路线～先告诉我：1) 从哪里出发？2) 要去哪里？（可选：出发时间/预算）"
    },
    "phone": {
        "required": ["预算", "流量需求", "prepaid_or_postpaid"],
        "optional": ["是否需要国际通话", "是否经常偏远地区", "是否要eSIM"],
        "question": "我可以帮你选手机套餐～先确认：1) 预算（每月上限）2) 大概需要多少GB/月 3) 想要预付prepaid还是合约postpaid？（可选：国际通话/eSIM/偏远地区）"
    },
    "healthcare": {
        "required": ["你是OSHC还是Medicare", "需求类型(GP/急诊/报销)"],
        "optional": ["所在区", "是否紧急"],
        "question": "我可以帮你梳理就医/报销～先确认：1) 你是OSHC学生保险还是Medicare？2) 你要解决的是看GP/急诊/报销哪个？"
    },
    "uni": {
        "required": ["学校", "问题类型(挂科/特殊考虑/学术诚信等)"],
        "optional": ["课程代码", "截止日期"],
        "question": "我可以按学校规则帮你判断～先说：1) 你哪个学校（UNSW？）2) 属于哪类问题（特殊考虑/挂科/抄袭/出勤等）？"
    },
}

MODULE_DISPLAY = {
    "transport": "🚆 交通专家",
    "phone": "📱 手机卡专家",
    "healthcare": "🏥 医疗专家",
    "uni": "🎓 学业事务专家"
}


LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class DocChunk:
    source: str
    text: str

def route_intent(query: str) -> str:
    q = query.lower()

    if any(k in q for k in ["sim", "telstra", "optus", "vodafone", "手机卡", "流量", "套餐"]):
        return "phone"
    if any(k in q for k in ["oshc", "gp", "bulk billing", "000", "医疗", "看病", "保险", "急诊"]):
        return "healthcare"
    if any(k in q for k in ["attendance", "special consideration", "plagiarism", "appeal", "出勤", "延期", "学术诚信", "申诉"]):
        return "uni"
    return "transport"

INTENT_LABELS = {
    "transport": "public transport, opal card, train bus ferry, airport to city",
    "phone": "sim card, mobile plan, optus telstra vodafone prepaid postpaid",
    "healthcare": "oshc insurance, gp doctor, medicare bulk billing 000",
    "uni": "attendance, academic policy, special consideration, plagiarism"
}

SLOTS = {
  "phone": ["预算上限(每月$)", "大概需要多少GB", "prepaid还是postpaid"],
  "transport": ["出发地/目的地", "是否需要Opal", "是否周末/高峰"],
  "healthcare": ["是否OSHC", "症状紧急程度", "是否需要GP/急诊"],
  "uni": ["学校/课程", "问题类型(出勤/学术诚信/延期)", "截止时间"]
}

def route_intent_semantic(query: str, embedder):
    q_vec = embedder.encode([query])[0]
    best_k, best_score = None, -1.0

    for k, desc in INTENT_LABELS.items():
        d_vec = embedder.encode([desc])[0]
        score = float(np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec) + 1e-9))
        if score > best_score:
            best_k, best_score = k, score

    return (best_k or "transport"), best_score

def is_opal_card_question(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["opal", "opal卡", "办卡", "充值", "top up", "concession", "student fare", "优惠", "学生票", "卡丢了", "挂失"])

def need_clarify(module: str, query: str) -> bool:
    # ✅ transport: Opal 办卡类不要求起点/终点
    if module == "transport" and is_opal_card_question(query):
        return False

    req = MODULE_REQUIREMENTS.get(module, {})
    required = req.get("required", [])

    for r in required:
        if r in query:
            continue
        # phone 兜底逻辑保留...
        if module == "phone":
            ...
        return True
    return False




def split_markdown(text: str) -> List[str]:
    # 简单切块：按空行分段，过滤太短段落
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    # 再过滤特别短的段（少于 40 字符）
    parts = [p for p in parts if len(p) >= 40]
    return parts

def load_docs(data_dir: str) -> List[DocChunk]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.md")))
    chunks: List[DocChunk] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            full = f.read().strip()
        if not full:
            continue

        parts = split_markdown(full)
        for i, p in enumerate(parts):
            chunks.append(
                DocChunk(
                    source=f"{os.path.basename(fp)}#chunk{i+1}",
                    text=f"(Source: {os.path.basename(fp)})\n{p}",
                )
            )
    return chunks



def embed_texts(model, texts):
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype="float32")


def build_or_load_index(module_name: str, chunks: List[DocChunk]):
    os.makedirs(INDEX_DIR, exist_ok=True)

    index_path = os.path.join(INDEX_DIR, f"{module_name}.faiss")
    meta_path = os.path.join(INDEX_DIR, f"{module_name}_meta.json")

    # load if exists
    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        embedder = SentenceTransformer(LOCAL_EMBED_MODEL, device="cpu")
        return index, meta, embedder

    # build
    embedder = SentenceTransformer(LOCAL_EMBED_MODEL)
    texts = [c.text for c in chunks]
    vectors = embedder.encode(texts, normalize_embeddings=True)
    vectors = np.array(vectors).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    meta = [{"source": c.source, "text": c.text} for c in chunks]

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return index, meta, embedder



def retrieve(query, index, meta, embedder, k=3):
    qv = embed_texts(embedder, [query])
    scores, ids = index.search(qv, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        item = meta[idx]
        results.append((float(score), item["source"], item["text"]))

    return results


def answer_with_llm(query, contexts, out_lang, module):
    provider = os.getenv("LLM_PROVIDER", "dashscope").strip().lower()

    if provider != "dashscope":
        return ""  # 你暂时只用 dashscope 就先这样

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return "❌ 未检测到 DASHSCOPE_API_KEY，请检查 .env 或环境变量。"

    dashscope.api_key = api_key
    model = os.getenv("DASHSCOPE_MODEL", "qwen-turbo").strip()

    base_prompt = "你是悉尼留学生生活助手。请用清晰要点回答，必要时分步骤。"
    system_prompt = MODULE_SYSTEM_PROMPTS.get(module, base_prompt)

    ctx_text = "\n\n".join([f"[{src}] {txt}" for _, src, txt in contexts]) if contexts else "（无检索结果）"

    user_prompt = f"""用户问题：{query}

可用资料（RAG检索）：
{ctx_text}

输出语言：{out_lang}
请基于资料回答；资料不足就明确说明，并给出下一步建议。"""

    messages = [{"role": "system", "content": system_prompt}]

    # 注入最近3轮（6条），但要保证格式正确
    hist = st.session_state.get("chat_history", [])
    if hist:
        messages += hist[-6:]

    messages.append({"role": "user", "content": user_prompt})

    resp = dashscope.Generation.call(
        model=model,
        messages=messages,
        result_format="message",
        temperature=0.2,
    )

    try:
        content = resp["output"]["choices"][0]["message"]["content"].strip()
        return content if content else "（LLM 返回空内容）"
    except Exception:
        return f"❌ DashScope 返回异常：{resp}"






def fallback_answer(contexts, out_lang):
    if not contexts:
        return "当前知识库没有相关内容，请先补充 data/transport 下的文档。"

    top = contexts[0]  # (score, src, text)
    score, src, text = top

    if out_lang == "中文":
        return (
            "【本地模式】未配置 API Key，所以无法进行高质量翻译/改写。我先把最相关的资料片段给你：\n\n"
            f"来源：{src}\n\n{text}"
        )
    else:
        return (
            "[Local mode] No API key configured, showing the most relevant retrieved text:\n\n"
            f"Source: {src}\n\n{text}"
        )

if "last_module" not in st.session_state:
    st.session_state.last_module = "transport"

current_module = st.session_state.last_module
# ---------------- UI ----------------

st.set_page_config(page_title="Sydney Student Agent", layout="wide")

st.title(f"Sydney International Student AI Assistant")



st.caption("RAG: FAISS + SentenceTransformers")

# =========================
# Chat 历史显示区域
# =========================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

embedder = SentenceTransformer(LOCAL_EMBED_MODEL)

indexes = {}
metas = {}

for module_name, data_dir in MODULES.items():
    docs_m = load_docs(data_dir)

    # ✅ 关键：把每个模块读到多少 chunk 打出来
    st.sidebar.write(f"📚 {module_name}: chunks={len(docs_m)} dir={data_dir}")

    if not docs_m:
        continue

    idx, meta_m, _ = build_or_load_index(module_name, docs_m)
    indexes[module_name] = idx
    metas[module_name] = meta_m

# ✅ 关键：把最终建好的索引模块列表打印出来
st.sidebar.write("✅ indexes keys:", list(indexes.keys()))



with st.sidebar:
    st.json(st.session_state.slot_state)
    st.subheader("系统状态")

    provider = os.getenv("LLM_PROVIDER", "dashscope").strip().lower()
    st.write("LLM_PROVIDER:", provider)

    if provider == "dashscope":
        has_key = bool(os.getenv("DASHSCOPE_API_KEY", "").strip())
        st.write("DASHSCOPE_API_KEY:", "✅ 已配置" if has_key else "❌ 未配置")
    else:
        has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
        st.write("OPENAI_API_KEY:", "✅ 已配置" if has_key else "❌ 未配置")

    st.subheader("输出语言 / Output Language")
    out_lang = st.selectbox(
        "选择回答语言",
        ["中文", "English", "Français", "日本語", "한국어"],
        index=0
    )

    if st.button("重建向量索引"):
        # 清理所有模块的索引
        for module_name in MODULES.keys():
            index_path = os.path.join(INDEX_DIR, f"{module_name}.faiss")
            meta_path = os.path.join(INDEX_DIR, f"{module_name}_meta.json")
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
        
        st.success("索引已重建，请刷新页面")



query = st.chat_input("输入问题（例如：怎么办 Opal 卡？Optus 套餐哪个好？）")

if query:
    query = query.strip()
    if not query:
        st.warning("请输入问题")
        st.stop()

    # ===== 1) Router 决策（这里一定会定义 module）=====
    module, score = route_intent_semantic(query, embedder)
    st.info(f"🧭 Router 选择模块: {module} (score={score:.3f})")

    AMBIGUOUS = ["优惠","折扣","多少钱","价格","怎么弄","怎么办","怎么做","需要什么","材料","流程"]
    last = st.session_state.get("last_module")

    if last and (len(query) <= 8 or any(k in query for k in AMBIGUOUS)):
        module = last
        st.info(f"↩️ 模糊问题，沿用上次模块: {module}")

    if score < 0.20:
        module = route_intent(query)
        st.info(f"🪝 语义分数低，关键词兜底: {module}")

    # ===== 2) need_clarify（如果要追问，直接 stop）=====
    if need_clarify(module, query):
        st.session_state.last_module = module
        st.warning(MODULE_REQUIREMENTS[module]["question"])
        st.stop()

    # ===== 3) 模块必须有索引 =====
    if module not in indexes:
        st.error(f"模块 '{module}' 没有索引。当前可用模块: {list(indexes.keys())}")
        st.stop()

    # ✅ 先记录 last_module（避免后面 stop 时丢失）
    st.session_state.last_module = module

    # ===== 4) Slot Filling（只对 phone 启用）=====
    if "slot_state" not in st.session_state:
        st.session_state.slot_state = {}

    if module == "phone":
        if "phone" not in st.session_state.slot_state:
            st.session_state.slot_state["phone"] = {k: "" for k in SLOTS["phone"]}

        def update_slots_from_text(text):
            s = st.session_state.slot_state["phone"]
            t = text.lower()
            if any(x in t for x in ["$", "aud", "刀", "以内", "以下", "预算"]):
                s["预算上限(每月$)"] = text
            if "gb" in t or "流量" in text:
                s["大概需要多少GB"] = text
            if any(x in t for x in ["prepaid", "postpaid", "预付", "合约"]):
                s["prepaid还是postpaid"] = text

        update_slots_from_text(query)

        missing = [k for k, v in st.session_state.slot_state["phone"].items() if not v]
        if missing:
            ask = "我需要你补充：\n" + "\n".join([f"- {m}" for m in missing[:2]])
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": ask})
            st.stop()

    # ===== 5) RAG 检索 =====
    contexts = retrieve(query, indexes[module], metas[module], embedder)

    st.subheader("🔎 RAG 检索结果")
    for s, src, txt in contexts:
        with st.expander(f"{src} | score={s:.3f}"):
            st.write(txt)

    # ===== 6) 生成答案 =====
    st.subheader("🤖 Agent 回答")
    answer = answer_with_llm(query, contexts, out_lang, module)
    st.write(answer if answer else "（未生成答案）")

    # ===== 7) 写入历史（严格两条）=====
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer if answer else "（未生成答案）"})





