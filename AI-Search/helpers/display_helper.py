from llama_index_client import EvalQuestionResult
from llama_index.core.schema import NodeWithScore, BaseModel
import logging

def display_evaluation_result(result: EvalQuestionResult):
    logging.info("Evaluating Response Relevancy -> ", end='\n')
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

def display_chunks(nodes: list[BaseModel]):
    print("Chunks ->", end="\n")
    for node in nodes:
        print()
        print(node.get_content())