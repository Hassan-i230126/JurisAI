import chromadb
client = chromadb.PersistentClient(path='data/chroma_db')
if client.list_collections():
    collection = client.get_collection('juris_legal_corpus')
    print(f'Count: {collection.count()}')
    res = collection.get(limit=1)
    if res and res['metadatas']:
        print(f'Sample Metadata: {res["metadatas"][0]}')
        print(f'Sample Doc: {res["documents"][0][:100]}')
