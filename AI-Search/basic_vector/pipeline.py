from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage, Settings
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index_client import EvalQuestionResult
from llama_index.core.schema import NodeWithScore

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
    retriever = index.as_retriever(similarity_top_k=2)

    print("==================================================", end='\n')
    print("Basic vector started", end="\n\n")
          
    for question in questions:
        print("--------------------------------------------------", end="\n\n")

        response = query_engine.query(question)
        print("Q : " + question, end='\n')
        print("A : " + str(response), end="\n\n")
  
        print("Retrieved nodes ->", end="\n")
        retrieved_nodes = retriever.retrieve(question)
        for node in retrieved_nodes:
            print()
            display_node(node)

        print()
        print("*** Response Evaluation ***", end='\n')
        faithfulness_eval_result = faithfulness_evaluator.evaluate_response(response=response)
        display_evaluation_result(faithfulness_eval_result)
        print()

        relevancy_eval_result = relevancy_evaluator.evaluate_response(query=question, response=response)
        display_evaluation_result(relevancy_eval_result)

    print()
    print("Basic vector finished", end='\n')
    print("==================================================", end="\n\n")

def display_evaluation_result(result: EvalQuestionResult):
    print("Evaluating Response Relevancy -> ", end='\n')
    #print("Query : " + str(relevancy_eval_result.query), end='\n')
    #print("Contexts : " + str(relevancy_eval_result.contexts), end='\n')
    #print("Response : " + str(relevancy_eval_result.response), end='\n')
    print("Passing : " + str(result.passing), end='\n')
    print("Feedback : " + str(result.feedback), end='\n')
    print("Score : " + str(result.score), end='\n')
    print("Pairwise source : " + str(result.pairwise_source), end='\n')
    print("Invalid result : " + str(result.invalid_result), end='\n')
    print("Invalid reason : " + str(result.invalid_reason), end='\n')

def display_node(node: NodeWithScore):
    print("Node ID  : " + str(node.node_id ), end='\n')
    print("Score    : " + str(node.score ), end='\n')
    print("Text     : " + str(node.text ), end='\n')
    #print("Metadata : " + str(node.metadata ), end='\n')