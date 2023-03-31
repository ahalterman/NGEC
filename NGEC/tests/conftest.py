from ..actor_resolution import ActorResolver
from ..formatter import Formatter
import pytest
import spacy
from NGEC import AttributeModel

@pytest.fixture(scope='session', autouse=True)
def ag():
    return ActorResolver(base_path="./assets/")

@pytest.fixture(scope='session', autouse=True)
def nlp():
    return spacy.load("en_core_web_trf")

@pytest.fixture(scope='session', autouse=True)
def am():
    return AttributeModel(model_dir = "./assets/PROP-SQuAD-trained-tinybert-6l-768d-squad2220302-1457",
                    expand_actors=True,
                    silent=False)

