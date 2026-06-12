import pytest
import yaml

from fund_flow.synthetic.abm_data import generate_synthetic_book
from fund_flow.synthetic.predictor_data import generate_predictor_panel


@pytest.fixture(scope="session")
def cfg():
    with open("config/default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def predictor_df(cfg):
    return generate_predictor_panel(cfg)


@pytest.fixture(scope="session")
def abm_book(cfg):
    return generate_synthetic_book(cfg)
