from llama_index_client import EvalQuestionResult
from llama_index.core.schema import NodeWithScore, BaseModel
import logging

def display_evaluation_result(result: EvalQuestionResult):
    print_log("Evaluating Response Relevancy -> ", end='\n')
    #print_log("Query : " + str(relevancy_eval_result.query), end='\n')
    #print_log("Contexts : " + str(relevancy_eval_result.contexts), end='\n')
    #print_log("Response : " + str(relevancy_eval_result.response), end='\n')
    print_log("Passing : " + str(result.passing), end='\n')
    print_log("Feedback : " + str(result.feedback), end='\n')
    print_log("Score : " + str(result.score), end='\n')
    print_log("Pairwise source : " + str(result.pairwise_source), end='\n')
    print_log("Invalid result : " + str(result.invalid_result), end='\n')
    print_log("Invalid reason : " + str(result.invalid_reason), end='\n')

def display_node(node: NodeWithScore):
    print_log("Node ID  : " + str(node.node_id ), end='\n')
    print_log("Score    : " + str(node.score ), end='\n')
    print_log("Text     : " + str(node.text ), end='\n')
    #print_log("Metadata : " + str(node.metadata ), end='\n')

def display_chunks(nodes: list[BaseModel]):
    print_log("Chunks ->", end="\n")
    for node in nodes:
        print_log("", end='\n')
        print_log(node.get_content())

def print_log(text: str, end: str, display: bool = True, log: bool = True):
    if display:
        print(text, end=end)

    if log:
        logging.info(msg=text)