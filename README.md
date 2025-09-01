# GenAI Tutorial

This repository contains my implementation of various Generative AI projects based on the YouTube playlist **"Generative AI from Basic to Advance"** by Sunny Savita. The playlist is dedicated to learning Generative AI step by step, covering foundational concepts, LangChain, LlamaIndex, RAG, and advanced AI pipelines.

## What I Learned

From these tutorials, I have implemented:

- Generative AI basics: text generation, embeddings, and vector search
- LangChain and LlamaIndex integration for RAG pipelines
- Querying and retrieving information from vector databases (MongoDB Atlas, FAISS)
- Using embeddings and advanced search techniques for QA over documents
- Incorporating multi-modal inputs like images in RAG pipelines
- Fine-tuning and leveraging Google Gemini API for LLM responses
- End-to-end RAG applications combining document retrieval, embeddings, and LLM generation

## Notes

- **API Choice:** I have used **Google Gemini API** instead of OpenAI API. Google Gemini API is free, whereas OpenAI API is a paid service.  
- **Package Updates:** Most of the LangChain and LlamaIndex code from the tutorials has changed over time due to package updates and deprecations. I have adapted those changes in my code. Future updates may require further modifications.  
- **References:** All tutorials and guidance are credited to Sunny Savita’s YouTube playlist: [Generative AI from Basic to Advance](https://www.youtube.com/playlist?list=PLQxDHpeGU14AIu52l2OlIJs9z94yTFR_t)


> **Note:** The `.gitignore` is configured to exclude temporary and sensitive files like `logs/`, `__pycache__/`, `.env`, and virtual environments (`rag_qa_gemini_llama_venv`) to keep the repo clean.

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/prakadeesh01/genai_tutorial.git
cd genai_tutorial
```

2. Install dependencies (suggested in a virtual environment):

```bash
pip install -r requirements.txt
```

3. Set up your environment variables (.env) with your Google Gemini API key.

4. Open the notebooks in Colab or Jupyter and run each cell sequentially.

## Contributing

This repository is based on tutorial guidance but contains my adaptations.  
Feel free to submit improvements, fixes, or updates for package compatibility.

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

