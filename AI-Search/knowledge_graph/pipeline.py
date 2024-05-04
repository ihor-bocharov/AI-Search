from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, StorageContext, Settings, load_graph_from_storage
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from helpers.display_helper import display_node, display_evaluation_result

def run_pipeline(questions: list[str], load_from_storage: bool):
    # Settings
    data_dir = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\data\\paul_graham_essay"
    graph_context_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\graph-context-store"

    # Init
    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)

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
        similarity_top_k=2,
    )

    print("==================================================", end='\n')
    print("Knowledge Graph started", end="\n\n")
          
    for question in questions:
        print("--------------------------------------------------", end="\n\n")

        response = query_engine.query(question)
        print("Q : " + question, end='\n')
        print("A : " + str(response), end='\n')

        print("Retrieved nodes ->", end="\n")
        for node in response.source_nodes:
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
    print("Knowledge Graph finished", end='\n')
    print("==================================================", end="\n\n")