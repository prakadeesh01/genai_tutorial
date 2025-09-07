# 1.RAG Pipeline with Neo4j + Vector Store

---

## 📌 Pipeline Flowchart

**Legend**  
- `[PRECOMPUTE]` = done once before queries  
- `[RUNTIME]` = happens per user query  
- ✨ = LLM call  

---

```text
                [PRECOMPUTE]
                Raw documents (wiki, pdf, etc.)
                        │
                        ├─► LLMGraphTransformer (✨, ingestion LLM)
                        │      converts docs → GraphDocuments (nodes+edges)
                        │
                        └─► Insert GraphDocuments into Neo4j Graph
                            (nodes, relationships, properties)
                            │
                            └─► Neo4jVector.from_existing_graph(...)
                                    → Vector index created over graph nodes (embeddings stored)
                                    (vector_index ready for similarity_search)
                --------------------------------------------------------------------------------
                [RUNTIME]   <--- user query (and optional chat_history) arrives
                --------------------------------------------------------------------------------
                User input:
                {"question": "...", "chat_history": [...]?}
                        │
                        ▼
                RunnableParallel (two parallel branches):
                ┌──────────────────────────────────────────────────────────────┐
                │ "question" branch: passthrough → provides the question text   │
                └──────────────────────────────────────────────────────────────┘
                        │
                ┌──────────────────────────────────────────────────────────────┐
                │ "context" branch: (_search_query | retriever)                │
                │  Step A: _search_query                                        │
                │    ├─ if chat_history exists: format chat history → prompt → │
                │    │  LLM condense (✨) → produce standalone_question         │
                │    └─ else: standalone_question = input question (passthrough)│
                │                                                                │
                │  Step B: retriever (runs with the standalone_question)         │
                │    ├─ entity_chain.invoke(standalone_question)                │
                │    │     → Entity extraction via LLM.with_structured_output (✨)
                │    │     → returns entities.names = [ "Elizabeth I", ... ]     │
                │    │                                                            │
                │    ├─ For each extracted entity:                                │
                │    │     generate_full_text_query(entity)                       │
                │    │     CALL db.index.fulltext.queryNodes('entity', query)     │
                │    │     YIELD node,score                                        │
                │    │     SUBQUERY: MATCH (node)-[r:!MENTIONS]->(neighbor) /      │
                │    │               MATCH (node)<-[r:!MENTIONS]-(neighbor)        │
                │    │     → FORMAT neighbor relationships as triplet strings     │
                │    │     **(Triplets are produced here, at runtime, from the graph)**
                │    │                                                            │
                │    └─ In parallel inside retriever:                              │
                │          vector_index.similarity_search(standalone_question)     │
                │          → returns unstructured docs (page_content)              │
                │                                                                    │
                │    └─ Combine: structured triplet strings + unstructured doc text │
                │          → final context string (returned by retriever)          │
                └──────────────────────────────────────────────────────────────┘
                        │
                        ▼
                RunnableParallel collects:
                {
                "context": "<structured triplets + unstructured docs>",
                "question": "<standalone_question>"
                }
                        │
                        ▼
                Final Prompt Template:
                "Answer the question based only on the following context:
                {context}
                Question: {question}
                Answer:"
                        │
                        ▼
                LLM answer generation (✨)  — final LLM call
                        │
                        ▼
                StrOutputParser → final string answer returned to caller
```

# Quick Clarifications on Retriever + LLM Flow

---

## 🔹 Triplets
- Triplets are produced **at runtime inside the retriever** when you call the full-text index and run the subquery that formats neighbor relationships.  
- They are **not** being created on-the-fly by the LLM at that moment — instead, you query the graph and format the relationships into strings.

---

## 🔹 Vector Index
- Created earlier with `Neo4jVector.from_existing_graph(...)`.  
- Contains embeddings of the **graph nodes/documents** that existed at indexing time.  
- This index is later used for `similarity_search(...)` at runtime.

---

## 🔹 LLM Invocations (Ingestion + Runtime)

### ✨ Ingestion LLM
- `LLMGraphTransformer.convert_to_graph_documents`  
- Runs at **precompute stage** when you first build the graph.

### ✨ Entity Extraction LLM
- `entity_chain.invoke(...)`  
- Runs at **runtime** to extract entities from the question.

### ✨ Condense-Question LLM
- Used when chat history exists.  
- Runs at **runtime** to produce a standalone question.

### ✨ Final-Answer LLM
- Runs at **runtime** to generate the final answer from combined context.

---

## ✅ Summary
- **Ingestion:** 1 LLM call (graph conversion).  
- **Query-time:** Up to 3 LLM calls (entity extraction, condense question, final answer).  
- Overall: LLM is invoked several times across the lifecycle.  

<br><br>



# 2.Unified Retriever with Structured + Unstructured Data

If you want a single unified retriever (structured + unstructured) **before knowing the question**, you need to pre-store the structured knowledge in text form + embeddings, not generate it on-the-fly.

---

## How to do that

### 1. Precompute all triples from the graph

```python
triples = []
for node in graph.nodes():
    for rel, neighbor in node.relationships.items():
        triples.append(f"{node.id} - {rel} -> {neighbor.id}")

```

### 2. Combine triples with document text

```python
from langchain.schema import Document

all_docs = []

# Add precomputed triples as "documents"
for triple in triples:
    all_docs.append(Document(page_content=triple))

# Add original document text
for doc in raw_documents:
    all_docs.append(doc)
```

### 3. Embed everything into the vector store

```python
vector_index.add_documents(all_docs)
```

<br><br>

## 3.Modern Vector RAG (Neo4j as a vector store) 
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import wikipedia

# --- Init embeddings + LLM ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY
)
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# --- Get document ---
page = wikipedia.page("Elizabeth I")
docs = [page.content]

# --- Create vector index in Neo4j ---
vector_store = Neo4jVector.from_texts(
    texts=docs,
    embedding=embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="elizabeth1_vector",
    node_label="Document",
    text_node_property="text",
)

# --- Build retriever ---
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --- Prompt template ---
template = """You are a helpful historian.
Use the context below to answer the user question.
Keep answers concise and fact-based.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# --- LCEL chain ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Run query ---
print(rag_chain.invoke("Who were Elizabeth I's main advisors?"))
```

<br><br>

# 4.🔎 Techniques in Your Pipeline

## ✅ Techniques You Already Use

### 1. Hybrid Retrieval (Structured + Unstructured)
You combine:
- **Structured retrieval** from Neo4j (triplets via entity chain + full-text).  
- **Unstructured retrieval** via vector similarity search.  
- Then merge results into a single context.  

✔ This is a strong form of **hybrid search**.

---

### 2. Query Rewriting (Condense Prompt)
- If chat history exists, you run a **condense-question LLM → standalone question**.  
- This is a standard **query rewriting** technique, sometimes called *question condensation*.  

---

### 3. Entity-based Retrieval
- You explicitly extract entities with `structured_output LLM`.  
- Entities drive the graph full-text search.  

✔ This is **entity-level retrieval**, often considered advanced.

---

### 4. Multi-Stage LLM Orchestration
- Specialized LLM calls for: ingestion, entity extraction, condense, final answer.  

✔ This is a **modular RAG pipeline**, not a monolithic one.  

---

## ❌ Techniques You Don’t Use Yet (but could add)

### 1. Fusion (e.g., Reciprocal Rank Fusion, RRF)
- Right now, you just **concatenate structured + unstructured results**.  
- Fusion means **ranking and merging results from multiple retrievers** into a unified ranked list (e.g., RRF, weighted fusion).  

---

### 2. Re-ranking (Cross-Encoders / LLM Re-ranker)
- After retrieval, you don’t apply a **reranker**.  
- Reranking means using a **smaller model (e.g., BERT cross-encoder)** or even an LLM to **sort results by relevance** before passing to the final answer generator.  

---

### 3. Dynamic Retrieval Strategies (Retriever Router)
- Right now you **always use both structured + unstructured**.  
- Advanced pipelines sometimes use a **router LLM** that decides:  
  - “This question is about entities → use graph.”  
  - “This is semantic → use vector DB.”  
  - Or both.  

---

### 4. Self-Reflection / Verification
- Final answer is generated once.  
- Some pipelines make the LLM **critique/revise the answer** (self-consistency, fact-checking against retrieved context).  
