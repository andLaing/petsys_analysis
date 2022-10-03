import os
import pytest

@pytest.fixture(scope = 'session')
def TEST_DATA_DIR():
    return os.path.join(os.environ['PWD'], "pet_code/test_data/")


@pytest.fixture(scope = 'session')
def DUMMY_EVT():
    return [[703, 10, 1270956643124, 14.73399], [702, 10, 1270956642920, 13.79702],
            [701, 10, 1270956643593, 12.95290], [700, 10, 1270956644837, 10.41906],
            [699, 10, 1270956642785,  8.63039], [687, 10, 1270956643239,  8.29962],
            [686, 10, 1270956642228,  5.73080]]


@pytest.fixture(scope = 'session')
def TMP_OUT(tmp_path_factory):
    return tmp_path_factory.mktemp('cry_tests')


# @pytest.fixture(scope = 'session')
# def TMP_TXT(tmp_path_factory):
#     fn = tmp_path_factory.mktemp("map") / "notyaml.txt"
#     fn.write_text("I'm not a mapping file")
#     return fn
@pytest.fixture(scope = 'session')
def TMP_TXT(TMP_OUT):
    fn = TMP_OUT / "notyaml.txt"
    fn.write_text("I'm not a mapping file")
    return fn