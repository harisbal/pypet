__author__ = 'Robert Meyer'

import sys
import os

import numpy as np

from pypet.trajectory import Trajectory
from pypet.tests.testutils.ioutils import make_temp_dir, remove_data, run_suite, parse_args
from pypet.utils import comparisons as comp


try:
    import cPickle as pickle # will fail under python 3
except ImportError:
    import pickle

if (sys.version_info < (2, 7, 0)):
    import unittest2 as unittest
else:
    import unittest

class AnnotationsTest(unittest.TestCase):

    tags = 'unittest', 'annotations'

    def setUp(self):

        self.filename = make_temp_dir(os.path.join('experiments','tests','HDF5','annotations.hdf5'))

        self.traj = Trajectory(name='Annotations', filename = self.filename)

        self.traj.add_result('testres', 42)

        self.traj.add_parameter('testgroup.testparam', 42)

        self.make_annotations()

        self.add_annotations(self.traj)

        pred = lambda x: 'config' not in x.full_name

        x = len([node for node in self.traj.iter_nodes(recursive=True, predicate=pred)])
        self.assertTrue(x == 5, '%s != %s' % (str(x), str(5)))

    def tearDown(self):
        remove_data()

    def make_annotations(self):

        self.annotations={}

        self.annotations['dict']={'33':12,'kkk':[1,2,'h'], 3:{'a':42.0}}
        self.annotations['list']= [self.annotations['dict'],33]
        self.annotations['string'] = 'string'
        self.annotations['integer'] = 42
        self.annotations['tuple']=(3,4,5)
        self.annotations['Numpy_Data'] = np.array(['fff','ddd'])
        self.annotations[0] = 7777

    def add_annotations(self, traj):
        funcs = 5

        for idx,node in enumerate([traj] + [node for node in traj.iter_nodes(recursive=True)]):
            for name in self.annotations:
                anno = self.annotations[name]
                if name == 0:
                    node.set_annotations(anno)
                    node.annotations.set(anno)
                elif idx % funcs == 0:
                    node.set_annotations(**{name:anno})
                elif idx % funcs == 1:
                    node.annotations.set(**{name:anno})
                elif idx % funcs == 2:
                    node.annotations.set_single(name,anno)
                elif idx % funcs == 3:
                    setattr(node.annotations,name, anno)
                elif idx % funcs == 4:
                    node.annotations[name]=anno

    def test_annotations_insert(self):

        for idx,node in \
                enumerate([self.traj] + [node for node in self.traj.iter_nodes(recursive=True)]):
            for name in self.annotations:
                anno = self.annotations[name]
                node_anno = node.annotations[name]
                self.assertTrue(comp.nested_equal(anno, node_anno),
                                                  '%s != %s' % (str(anno), str(node_anno)))

    def test_pickling(self):
        dump = pickle.dumps(self.traj)

        del self.traj

        self.traj = pickle.loads(dump)

        self.test_annotations_insert()


    def test_storage_and_loading(self):
        self.traj.store()

        traj_name = self.traj.name

        del self.traj

        self.traj = Trajectory(filename=self.filename)

        self.traj.load(name=traj_name, load_parameters=2, load_derived_parameters=2,
                         load_results=2, load_other_data=2)

        self.test_annotations_insert()


    def test_attribute_deletion(self):
        for node in self.traj.iter_nodes(recursive=True):
            name_list=[name for name in node.annotations]
            for name in name_list:
                delattr(node.annotations, name)

            self.assertTrue(node.annotations.is_empty())

    def test_item_deletion(self):
        for node in self.traj.iter_nodes(recursive=True):
            name_list=[name for name in node.annotations]
            for name in name_list:
                del node.annotations[name]

            self.assertTrue(node.annotations.is_empty())

    def test_get_item(self):
        for node in self.traj.iter_nodes(recursive=True):
            for key, val1 in node.annotations.to_dict().items():
                val2 = node.annotations[key]
                self.assertTrue(comp.nested_equal(val1,val2))

    def test_get_item_no_copy(self):
        for node in self.traj.iter_nodes(recursive=True):
            for key, val1 in node.annotations.to_dict(copy=False).items():
                val2 = node.annotations[key]
                self.assertTrue(comp.nested_equal(val1,val2))

    @staticmethod
    def dict_to_str(dictionary):
        resstr = ''
        new_dict={}
        for key, val in dictionary.items():
            if key == 0:
                key = 'annotation'
            new_dict[key]=val

        for key in sorted(new_dict.keys()):
            resstr+='%s=%s; ' % (key,str(new_dict[key]))
        return resstr[:-2]

    def test_to_str(self):
        dict_str = self.dict_to_str(self.annotations)
        for node in self.traj.iter_nodes(recursive=True):
            ann_str = node.ann_to_str()

            self.assertTrue(not ann_str.endswith(' ') or not ann_str.endswith(','))

            for name in self.annotations:
                if name==0:
                    name = 'annotation'
                self.assertTrue(name in ann_str)

            self.assertEqual(dict_str, ann_str, '%s!=%s' % (dict_str, ann_str))

            ann_str = str(node.annotations)
            self.assertEqual(dict_str, ann_str, '%s!=%s' % (dict_str, ann_str))

    def test_single_get_and_getattr_and_setattr(self):

        self.traj.add_parameter('test2', 42)

        self.traj.get('test2').annotations.test = 4

        self.assertTrue(self.traj.get('test2').annotations.test, 4)

        self.assertTrue(self.traj.get('test2').annotations.get(), 4)

    def test_get_annotations(self):
        key_list = list(self.annotations.keys())
        for node in self.traj.iter_nodes(recursive=True):
            for name in self.annotations:
                self.assertTrue(comp.nested_equal(self.annotations[name],
                                                  node.get_annotations(name)))

            val_list = node.get_annotations(*key_list)

            for idx, val in enumerate(val_list):
                self.assertTrue(comp.nested_equal(self.annotations[key_list[idx]], val))

    def test_f_get_errors(self):
        for node in self.traj.iter_nodes(recursive=True):
            with self.assertRaises(ValueError):
                node.annotations.get()

            with self.assertRaises(AttributeError):
                node.annotations.get('gdsdfd')

        testparam = self.traj.add_parameter('ggg',343)

        with self.assertRaises(AttributeError):
            testparam.annotations.get()

    def test_f_set_numbering(self):
        int_list = list(range(10))
        for node in self.traj.iter_nodes(recursive=True):
            node.annotations.set(*int_list)

            self.assertEqual(node.annotations.get(*int_list), tuple(int_list))

            for integer in int_list:
                if integer == 0:
                    name = 'annotation'
                else:
                    name = 'annotation_%d' % integer

                self.assertTrue(name in node.annotations)


if __name__ == '__main__':
    opt_args = parse_args()
    run_suite(**opt_args)