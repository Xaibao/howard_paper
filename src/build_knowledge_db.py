"""
一次性腳本：把 knowledge_base/ 下所有文件建成 ChromaDB 向量資料庫
執行：conda activate spectral && python src/build_knowledge_db.py
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = Path(__file__).parent
KB_DIR   = BASE_DIR / "knowledge_base"
DB_DIR   = BASE_DIR / "fog_expert_db"

def load_documents():
    docs = []
    for f in KB_DIR.iterdir():
        try:
            if f.suffix == ".pdf":
                loader = PyPDFLoader(str(f))
            elif f.suffix == ".txt":
                loader = TextLoader(str(f), encoding="utf-8")
            else:
                continue
            loaded = loader.load()
            docs.extend(loaded)
            print(f"  載入：{f.name}（{len(loaded)} 頁/段）")
        except Exception as e:
            print(f"  跳過 {f.name}：{e}")
    return docs

def main():
    print("=== 建立 FOG 知識庫向量資料庫 ===")
    print(f"知識庫路徑：{KB_DIR}")
    print(f"資料庫路徑：{DB_DIR}")

    docs = load_documents()
    print(f"\n共載入 {len(docs)} 份文件段落")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks   = splitter.split_documents(docs)
    print(f"切分成 {len(chunks)} 個 chunks")

    print("\n載入 Embedding 模型（all-MiniLM-L6-v2）...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("寫入 ChromaDB...")
    DB_DIR.mkdir(exist_ok=True)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )
    print(f"\n完成！資料庫已存至 {DB_DIR}，共 {db._collection.count()} 個向量。")

if __name__ == "__main__":
    main()
