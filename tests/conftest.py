import pytest

from tle.config import load_config, problem_spec
from tle.hydraulics.simulator import Simulator
from tle.scenarios.space import base_scenario


@pytest.fixture(scope="session")
def cfg():
    return load_config("configs/base_net3.yaml")


@pytest.fixture(scope="session")
def spec(cfg):
    return problem_spec(cfg)


@pytest.fixture(scope="session")
def sim(spec):
    s = Simulator(spec)
    yield s
    s.close()


@pytest.fixture(scope="session")
def base(cfg):
    return base_scenario(cfg)
