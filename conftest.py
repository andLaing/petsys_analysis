import os
import pytest

@pytest.fixture(scope = 'session')
def TEST_DATA_DIR():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "pet_code", "test_data")


@pytest.fixture(scope = 'session')
def DUMMY_EVT():
    return ([[700, 'ENERGY', 1271020681580, 11.84212], [682,   'TIME', 1271020680335,  6.41114],
             [640,   'TIME', 1271020703627,  6.26117], [643,   'TIME', 1271020680098,  5.19882],
             [701, 'ENERGY', 1271020686027,  4.45831], [703, 'ENERGY', 1271020688208,  4.17891],
             [686, 'ENERGY', 1271020686206,  4.16571], [693, 'ENERGY', 1271020690939,  3.93042],
             [691, 'ENERGY', 1271020690039,  3.88229], [687, 'ENERGY', 1271020686584,  3.82647],
             [689, 'ENERGY', 1271020689086,  3.82441], [702, 'ENERGY', 1271020689312,  3.69508],
             [692, 'ENERGY', 1271020687401,  3.67538], [699, 'ENERGY', 1271020687785,  3.45453],
             [696, 'ENERGY', 1271020688655,  2.99942], [697, 'ENERGY', 1271020689490,  2.30935],
             [681,   'TIME', 1271020701875,  0.95076], [683,   'TIME', 1271020694885,  0.38728]],
            [[116, 'ENERGY', 1271020687770, 20.09876], [117, 'ENERGY', 1271020686919, 18.76999],
             [115, 'ENERGY', 1271020688324, 14.76801], [118, 'ENERGY', 1271020687029, 14.30905],
             [ 64,   'TIME', 1271020680320, 13.67074], [113, 'ENERGY', 1271020688170, 12.35866],
             [120, 'ENERGY', 1271020687748, 10.73928], [119, 'ENERGY', 1271020687019, 10.60250],
             [121, 'ENERGY', 1271020688278,  8.21954], [ 67,   'TIME', 1271020681006,  3.45661],
             [ 65,   'TIME', 1271020681840,  3.38235], [ 66,   'TIME', 1271020686165,  3.11285],
             [112,   'TIME', 1271020700412,  1.88958], [ 69,   'TIME', 1271020704470,  1.68284]])


@pytest.fixture(scope = 'session')
def DUMMY_SM():
    return [[703, 'ENERGY', 1270956643124, 14.73399], [702, 'ENERGY', 1270956642920, 13.79702],
            [701, 'ENERGY', 1270956643593, 12.95290], [700, 'ENERGY', 1270956644837, 10.41906],
            [699, 'ENERGY', 1270956642785,  8.63039], [687, 'ENERGY', 1270956643239,  8.29962],
            [686, 'ENERGY', 1270956642228,  5.73080]]


@pytest.fixture(scope = 'session')
def TMP_OUT(tmp_path_factory):
    return tmp_path_factory.mktemp('cry_tests')


@pytest.fixture(scope = 'session')
def TMP_TXT(TMP_OUT):
    fn = TMP_OUT / "notyaml.txt"
    fn.write_text("I'm not a mapping file")
    return fn