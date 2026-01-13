import sys

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import MetaData

from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser # type: ignore
from logger import GLOBAL_LOGGER as log

class DocumentAnalyzer:

    def __init__(self):
        try:
            self.loader=ModelLoader()
            self.llm=self.loader.load_llm()

            self.parser=JsonOutputParser(pydantic_object=MetaData)
            self.fixing_parser=OutputFixingParser.from_llm(parser=self.parser,llm=self.llm)

            self.prompt = PROMPT_REGISTRY["document_analysis"]

            log.info("DocumentAnalyzer initialized successfully")
        except Exception as e:
            log.error(f"Error initializing DocumentAnalyzer:{e}")
            raise DocumentPortalException("Error in DocumentAnalyzer Initialization", sys)

    def analyze_document(self, document_text:str):
        try:
            chain = self.prompt | self.llm | self.fixing_parser
            
            log.info("Meta-data analysis chain initialized")
            print("#################################################################")
            print("doc_text::::::!!!!!!!!!!!!!", len(document_text))
            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            })

            log.info("Metadata extraction successful", keys=list(response.keys()))
            
            return response

        except Exception as e:
            log.error("Metadata analysis failed", error=str(e))
            raise DocumentPortalException("Metadata extraction failed",sys)
    
# da=DocumentAnalyzer()