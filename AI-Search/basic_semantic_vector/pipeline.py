from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from helpers.display_helper import display_node, display_evaluation_result
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from helpers.display_helper import display_chunks, print_log

def run_pipeline(questions: list[str], load_from_storage: bool):
    # Settings
    data_dir = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\data\\paul_graham_essay"
    basic_context_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\basic-semantic-context-store\\1"
    semantic_context_store_path = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\basic-semantic-context-store\\2"

    # Init
    embed_model = OpenAIEmbedding()
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model
    )
    base_splitter = SentenceSplitter(chunk_size=512)

    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)

    # Index pipeline
    if load_from_storage:
        storage_context_base = StorageContext.from_defaults(persist_dir=basic_context_store_path)
        index_base = load_index_from_storage(storage_context_base, index_id="vector_index")

        storage_context = StorageContext.from_defaults(persist_dir=semantic_context_store_path)
        index = load_index_from_storage(storage_context, index_id="vector_index")
    else:
        reader = SimpleDirectoryReader(input_dir=data_dir)
        documents = reader.load_data()

        nodes = splitter.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)
        display_chunks(nodes)

        nodes_base = base_splitter.get_nodes_from_documents(documents)
        index_base = VectorStoreIndex(nodes_base)
        display_chunks(nodes_base)
 
        # Store
        index_base.set_index_id("vector_index")
        index_base.storage_context.persist(basic_context_store_path)

        index.set_index_id("vector_index")
        index.storage_context.persist(semantic_context_store_path)

    # Query pipeline
    print_log("*** Semantic Splitter ***", end='\n')
    query_engine = index.as_query_engine()
    retriever = index.as_retriever(similarity_top_k=2)
    query(query_engine, retriever, questions, faithfulness_evaluator, relevancy_evaluator)

    print_log("*** Base Splitter ***", end='\n')
    query_engine_base = index_base.as_query_engine()
    retriever_base = index_base.as_retriever(similarity_top_k=2)
    query(query_engine_base, retriever_base, questions, faithfulness_evaluator, relevancy_evaluator)

def query(query_engine: BaseQueryEngine, retriever: BaseRetriever, questions: list[str], faithfulness_evaluator: FaithfulnessEvaluator, relevancy_evaluator: RelevancyEvaluator):
    print_log("==================================================", end='\n')
    print_log("Vector pipeline started", end="\n\n")
          
    for question in questions:
        print_log("--------------------------------------------------", end="\n\n")

        response = query_engine.query(question)
        print_log("Q : " + question, end='\n')
        print_log("A : " + str(response), end="\n\n")
  
        print_log("Retrieved nodes ->", end="\n")
        retrieved_nodes = retriever.retrieve(question)
        for node in retrieved_nodes:
            print_log("", end='\n')
            display_node(node)

        print_log("", end='\n')
        print_log("*** Response Evaluation ***", end='\n')
        faithfulness_eval_result = faithfulness_evaluator.evaluate_response(response=response)
        display_evaluation_result(faithfulness_eval_result)
        print_log("", end='\n')

        relevancy_eval_result = relevancy_evaluator.evaluate_response(query=question, response=response)
        display_evaluation_result(relevancy_eval_result)

    print_log("", end='\n')
    print_log("Vector pipeline finished", end='\n')
    print_log("==================================================", end="\n\n")

