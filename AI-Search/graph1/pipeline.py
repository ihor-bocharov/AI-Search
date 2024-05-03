from llama_index.core import SimpleDirectoryReader
from llama_index.core import KnowledgeGraphIndex, StorageContext, Settings, load_graph_from_storage
from llama_index.core.graph_stores import SimpleGraphStore

def run_pipeline(questions: list[str], load_from_storage: bool):
    # Settings
    data_dir = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\data\\paul_graham_essay"
    graph_context_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\graph-context-store"

    # Init

    # Index pipeline
    if load_from_storage:
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=graph_context_store_path)
        # load index
        index = load_graph_from_storage(storage_context, root_id="vector_index")
    else:
        reader = SimpleDirectoryReader(input_dir=data_dir)
        documents = reader.load_data()

        graph_store = SimpleGraphStore()
        storage_context = StorageContext.from_defaults(graph_store=graph_store)

        index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            max_triplets_per_chunk=2,
            include_embeddings=True,
        )

        index.set_index_id("vector_index")
        index.storage_context.persist(graph_context_store_path)

    # Query pipeline
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