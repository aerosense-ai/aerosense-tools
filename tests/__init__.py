import os
import datetime as dt

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(__file__))
TEST_START_TIME = dt.datetime(2020, 5, 4)
TEST_DURATION = dt.timedelta(seconds=1)
TEST_SAMPLING_STEP = dt.timedelta(seconds=0.1)
