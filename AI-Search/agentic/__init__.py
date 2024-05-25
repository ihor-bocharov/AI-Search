# https://github.com/cobusgreyling/LlamaIndex/blob/d8902482a247c76c7902ded143a875d5580f072a/Agentic_RAG_Multi_Document_Agents-v1.ipynb
# https://www.tomshardware.com/how-to/use-wget-download-files-command-line
# wget -e robots=off --recursive --no-clobber --page-requisites --html-extension --convert-links --restrict-file-names=windows --domains docs.llamaindex.ai --no-parent https://docs.llamaindex.ai/en/latest/

from agentic.pipeline import run_pipeline

def main():
    run_pipeline()

if __name__ == "__main__":
    main()