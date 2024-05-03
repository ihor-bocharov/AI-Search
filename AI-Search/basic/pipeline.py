from llama_index.core import SimpleDirectoryReader, Settings, StorageContext, VectorStoreIndex, load_index_from_storage

def run_pipeline(questions: list[str], load_from_storage: bool):
    # Settings
    data_dir = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\data\\paul_graham_essay"
    basic_context_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\basic-context-store"

    # Init

    # Index pipeline
    if load_from_storage:
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=basic_context_store_path)
        # load index
        index = load_index_from_storage(storage_context, index_id="vector_index")
    else:
        reader = SimpleDirectoryReader(input_dir=data_dir)
        documents = reader.load_data()

        index = VectorStoreIndex.from_documents(documents)

        index.set_index_id("vector_index")
        index.storage_context.persist(basic_context_store_path)

    #Query pipeline
    query_engine = index.as_query_engine()

    print("Basic", end='\n')
          
    for question in questions:
        response = query_engine.query(question)
        print(question, end='\n')
        print(response, end='\n')
        print(end='\n')

    print("Basic done", end='\n')