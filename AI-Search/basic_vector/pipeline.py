from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage, Settings
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

def run_pipeline(questions: list[str], load_from_storage: bool):
    # Settings
    data_dir = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\data\\paul_graham_essay"
    basic_context_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\basic-context-store"

    # Init
    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)

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

    print("Basic vector started", end='\n')
          
    for question in questions:
        response = query_engine.query(question)
        print("Q : " + question, end='\n')
        print("A : " + str(response), end='\n')

        faithfulness_eval_result = faithfulness_evaluator.evaluate_response(response=response)
        print("Evaluating Response Faithfulness : " + str(faithfulness_eval_result.passing), end='\n')
        
        relevancy_eval_result = relevancy_evaluator.evaluate_response(query=question, response=response)
        print("Evaluating Response Relevancy : " + str(relevancy_eval_result), end='\n')

        print("----------", end='\n')

    print("Basic vector finished", end='\n')
    print("==========", end='\n')