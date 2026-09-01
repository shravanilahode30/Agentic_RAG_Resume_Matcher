import chromadb

def load_resume_chunks(filepath="resume.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.split("\n")
    chunks = []
    current_section = "Summary"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        # header = short line ending in ':'
        if stripped.endswith(":") and len(stripped.split()) <= 4:
            if current_lines:
                chunks.append((current_section, "\n".join(current_lines).strip()))
            current_section = stripped.rstrip(":")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        chunks.append((current_section, "\n".join(current_lines).strip()))

    return chunks


def build_collection():
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection(name="resume")
    except Exception:
        pass

    collection = client.create_collection(name="resume")

    chunks = load_resume_chunks()
    documents = [content for _, content in chunks if content]
    ids = [f"chunk_{i}" for i in range(len(documents))]
    metadatas = [{"section": section} for section, content in chunks if content]

    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    print(f"Ingested {len(documents)} chunks into ChromaDB:")
    for section, content in chunks:
        if content:
            print(f"  - {section} ({len(content)} chars)")


if __name__ == "__main__":
    build_collection()