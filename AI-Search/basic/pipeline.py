import os
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from deeplake.core.vectorstore import DeepLakeVectorStore


def run_pipeline(questions: list[str]):
    #Settings
    data_dir = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\data\\paul_graham_essay"
    dataset_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\vector-store2"

    #Init
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    Settings.llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo")

    #Index pipeline
    reader = SimpleDirectoryReader(input_dir=data_dir)
    documents = reader.load_data()

    vector_store = DeepLakeVectorStore(path=dataset_path, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        llm=Settings.llm
    )

    #Query pipeline
    query_engine = index.as_query_engine()

    print("Basic", end='\n')
          
    for question in questions:
        response = query_engine.query(question)
        print(question, end='\n')
        print(response, end='\n')
        print(end='\n')

    print("Basic done", end='\n')