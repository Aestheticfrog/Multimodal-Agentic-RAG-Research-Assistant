"""LLM Provider Initialization with automatic 429 Rate-Limit Model Failover Pool."""
import os
import logging
from typing import Any, List
from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
logger = logging.getLogger("researchpilot.models")

# Pool of active Gemini models with free-tier quotas
GEMINI_MODEL_POOL = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3-flash",
    "gemini-flash-latest",
]


class ResilientGeminiLLM(Runnable):
    """A resilient LLM wrapper that automatically catches 429 rate limit errors
    and rotates through a pool of available Gemini models seamlessly.
    """

    def __init__(self, primary_model: str = "gemini-3.5-flash-lite", temperature: float = 0.2):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not found.")
        self.temperature = temperature
        self.pool: List[str] = list(GEMINI_MODEL_POOL)
        if primary_model in self.pool:
            self.pool.remove(primary_model)
            self.pool.insert(0, primary_model)

    def invoke(self, input_data: Any, config: Any = None, **kwargs) -> Any:
        last_exception = None
        for model in self.pool:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model,
                    temperature=self.temperature,
                    google_api_key=self.api_key,
                )
                return llm.invoke(input_data, config=config, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                    logger.warning(f"Model '{model}' rate limited (429). Auto-rotating to next model candidate...")
                    last_exception = e
                    continue
                else:
                    raise e
        if last_exception:
            raise last_exception

    def with_structured_output(self, schema: Any, **kwargs):
        """Supports structured output with automatic model rotation."""
        parent = self

        class StructuredWrapper(Runnable):
            def invoke(self, input_data: Any, config: Any = None, **chain_kwargs) -> Any:
                last_exc = None
                for model in parent.pool:
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model=model,
                            temperature=parent.temperature,
                            google_api_key=parent.api_key,
                        )
                        structured_llm = llm.with_structured_output(schema, **kwargs)
                        return structured_llm.invoke(input_data, config=config, **chain_kwargs)
                    except Exception as e:
                        err_str = str(e).lower()
                        if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                            logger.warning(f"Model '{model}' rate limited (429) during structured output. Auto-rotating...")
                            last_exc = e
                            continue
                        else:
                            raise e
                if last_exc:
                    raise last_exc

        return StructuredWrapper()


def get_gemini_llm(model_name: str = "gemini-3.5-flash-lite", temperature: float = 0.2) -> ResilientGeminiLLM:
    """Returns a ResilientGeminiLLM instance with automatic rate-limit failover pool."""
    return ResilientGeminiLLM(primary_model=model_name, temperature=temperature)


def get_gemini_embeddings(model_name: str = "models/text-embedding-004"):
    """Initialize and return Google Gemini Embeddings."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not found. "
            "Please set it in your .env file."
        )

    formatted_model = model_name if model_name.startswith("models/") else f"models/{model_name}"

    return GoogleGenerativeAIEmbeddings(
        model=formatted_model,
        google_api_key=api_key,
    )
