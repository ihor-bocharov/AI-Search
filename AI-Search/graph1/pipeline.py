import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core import KnowledgeGraphIndex, StorageContext, Settings
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.openai import OpenAI

def run_pipeline(questions: list[str]):
    #Settings
    data_dir = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\data\\paul_graham_essay"

    username = "neo4j"
    password = os.getenv("NEO4J_PASS")
    url = "neo4j+s://824e6c44.databases.neo4j.io"
    database = "neo4j"

    #Init
    graph_store = Neo4jGraphStore(
        username=username,
        password=password,
        url=url,
        database=database,
    )
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    Settings.llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo")
    Settings.chunk_size = 512

    #Index pipeline
    reader = SimpleDirectoryReader(input_dir=data_dir)
    documents = reader.load_data()

    index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=2,
        include_embeddings=True,
    )

    #Query pipeline
    query_engine = index.as_query_engine(
        llm=Settings.llm,
        include_text=True,
        response_mode="tree_summarize",
        embedding_mode="hybrid",
        similarity_top_k=5,
    )

    print("Graph 1", end='\n')
          
    for question in questions:
        response = query_engine.query(question)
        print(question, end='\n')
        print(response, end='\n')
        print(end='\n')

    print("Graph 1 done", end='\n')