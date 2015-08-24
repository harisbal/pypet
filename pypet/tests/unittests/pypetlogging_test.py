__author__ = 'Robert Meyer'

import sys
if (sys.version_info < (2, 7, 0)):
    import unittest2 as unittest
else:
    import unittest

try:
    import cPickle as pickle
except ImportError:
    import pickle

from pypet.pypetlogging import LoggingManager
from pypet.tests.testutils.ioutils import get_log_config, run_suite, parse_args
from pypet.utils.comparisons import nested_equal

class FakeTraj(object):
    def __init__(self):
        self.environment_name = 'env'
        self.name = 'traj'

    def wildcard(self, card):
        return 'Ladida'


class LoggingManagerTest(unittest.TestCase):

    tags = 'logging', 'unittest', 'pickle'

    def test_pickling(self):
        manager = LoggingManager(log_config=get_log_config(), log_stdout=True)
        manager.extract_replacements(FakeTraj())
        manager.check_log_config()
        manager.make_logging_handlers_and_tools()
        dump = pickle.dumps(manager)
        new_manager = pickle.loads(dump)
        manager.finalize()


if __name__ == '__main__':
    opt_args = parse_args()
    run_suite(**opt_args)
