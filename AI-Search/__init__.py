from agentic.pipeline import run_pipeline as run_agentic_pipeline

from dotenv import load_dotenv
import sys, os, logging, datetime
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

    log_dir = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\logs"
    log_file_name_pattern = "{:04}-{:02}-{:02}_{:02}-{:02}-{:02}_log.txt"
    console_file_name_pattern = "{:04}-{:02}-{:02}_{:02}-{:02}-{:02}_console.txt"

    now = datetime.datetime.now()

    full_file_name = os.path.join(log_dir, console_file_name_pattern.format(now.year, now.month, now.day, now.hour, now.minute, now.second))
    sys.stdout = open(file=full_file_name, mode="w", encoding="utf-8")

    full_file_name = os.path.join(log_dir, log_file_name_pattern.format(now.year, now.month, now.day, now.hour, now.minute, now.second))
    file_handler = logging.FileHandler(full_file_name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    model = "gpt-3.5-turbo"
    #model="gpt-4o"

    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(model).encode
    )

    llama_index.core.set_global_handler("simple")

    # Tracing
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)

    Settings.callback_manager = CallbackManager([token_counter, llama_debug])

    Settings.llm = OpenAI(temperature=0.0, model=model)
    Settings.service_context = ServiceContext.from_defaults(llm=Settings.llm)

    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    Settings.context_window = 4096
    Settings.num_output = 256

    # Agentic RAG
    questions_data_dir = "C:\\Users\\ihor.k.bocharov\\Documents\\GitHub\\AI-Search\\persistent\\docs.llamaindex.ai\\questions"
    #file_questions_name = "scenario-based.txt"
    #file_questions_name = "take_2_from_all_categories.txt"
    #file_questions_name = "take_1_from_all_categories.txt"
    #questions = file_helper.load_list_from_file(questions_data_dir, file_questions_name)

    file_questions_name = "inline"
    questions = [
        "How would you migrate an existing indexing system to llamaindex.io?"
        #"How does llamaindex.io compare to Elasticsearch?",
        #"What are the main components of llamaindex.io?"
        #"Tell me about LlamaIndex connectors",
        #"From the documentation what is the best way to get started with LlamaIndex?",
        #"What is pinecone?"
        ]

    logging.info("Questions : " + file_questions_name)
    logging.info("Generation Model : " + model)
    run_agentic_pipeline(questions, True, token_counter)

    sys.stdout.close()
    
if __name__ == "__main__":
    main()