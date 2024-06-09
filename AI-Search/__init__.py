from basic_vector.pipeline import run_pipeline as run_basic_vector_pipeline
from basic_semantic_vector.pipeline import run_pipeline as run_basic_semantic_vector_pipeline
from knowledge_graph.pipeline import run_pipeline as run_graph_pipeline
from metadata_filtering.pipeline import run_pipeline as run_metadata_pipeline
from agentic.pipeline import run_pipeline as run_agentic_pipeline
from agentic.pipeline import generate_questions as generate_agentic_questions

from dotenv import load_dotenv
import logging
import sys
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, ServiceContext
import llama_index.core
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, TokenCountingHandler
import helpers.file_helper as file_helper
import tiktoken

# pip-review --local --interactive

def main():

    load_dotenv('../.env')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    #logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    #logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    file_handler = logging.FileHandler('trace.txt')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    #model = "gpt-3.5-turbo-0125"
    #model="gpt-4-turbo"
    model="gpt-4-0125-preview"

    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(model).encode
    )

    llama_index.core.set_global_handler("simple")

    # Tracing
    #llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    #Settings.callback_manager = CallbackManager([llama_debug])

    Settings.callback_manager = CallbackManager([token_counter])

    Settings.llm = OpenAI(temperature=0.0, model=model)
    Settings.service_context = ServiceContext.from_defaults(llm=Settings.llm)

    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    Settings.context_window = 4096
    Settings.num_output = 256

    #questions = ["Who is John McCarthy?", "Who is Sam Altman?"]
    questions = ["What is the User-Agent?"]

    #run_basic_vector_pipeline(questions, load_from_storage=False)
    #run_basic_semantic_vector_pipeline(questions, load_from_storage=True)
    #run_graph_pipeline(questions, load_from_storage=True)
    run_metadata_pipeline()

    # Agentic RAG
    questions_data_dir = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\docs.llamaindex.ai\\questions"
    file_questions_name = "scenario-based.txt"
    #questions = file_helper.load_list_from_file(questions_data_dir, file_questions_name)
    questions = [
        #"How does llamaindex.io compare to Elasticsearch?",
        #"What are the main components of llamaindex.io?"
        #"Tell me about LlamaIndex connectors",
        "From the documentation what is the best way to get started with LlamaIndex?",
        #"What is pinecone?"
        ]
    #generate_agentic_questions()
    #run_agentic_pipeline(questions, False, token_counter)
    
if __name__ == "__main__":
    main()