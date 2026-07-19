import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.helpers import load_config
from src.prompts.prompt_templates import get_template


def test_load_config_keys():
    """
    Unit test to verify our config.yaml file exists
    and contains the crucial key "db_name"
    """
    config = load_config()

    assert config is not None
    assert "db_name" in config


def test_rag_prompt_variabels():
    """
    Unit test to ensure our prompt template contains
    exact blank spaces needed by our application
    """
    prompt = get_template()

    assert "context" in prompt.input_variables
    assert "question" in prompt.input_variables


def test_invalid_chunk_size():
    """
    Unit tests the config file has chunking releated values
    """
    config = load_config()
    assert config is not None
    assert "chunk_size" in config
    assert config["chunk_size"] > 0
    assert "chunk_overlap" in config
    assert config["chunk_overlap"] > 0
