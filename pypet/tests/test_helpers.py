__author__ = 'Robert Meyer'

from pypet.utils.comparisons import results_equal,parameters_equal
from pypet.utils.helpful_functions import nested_equal, nest_dictionary, flatten_dictionary
from pypet.parameter import Parameter, PickleParameter, BaseResult, ArrayParameter, PickleResult, BaseParameter
import scipy.sparse as spsp
import os
import logging
import unittest
import shutil
import numpy as np
import pandas as pd

import tempfile

TEMPDIR = 'temp_folder_for_pypet_tests/'
''' Temporary directory for the hdf5 files'''
REMOVE=True
''' Whether or not to remove the temporary directory after the tests'''

actual_tempdir=''
''' Actual temp dir, maybe in tests folder or in `tempfile.gettempdir()`'''

user_tempdir=''
'''If the user specifies in run all test a folder, this variable will be used'''

def make_temp_file(filename):
    global actual_tempdir
    global user_tempdir
    global TEMPDIR
    try:

        if not (user_tempdir == '' or user_tempdir is None) and actual_tempdir=='':
        #     actual_tempdir=TEMPDIR
        #  elif actual_tempdir=='':
            actual_tempdir=user_tempdir

        if not os.path.isdir(actual_tempdir):
            os.mkdir(actual_tempdir)

        return os.path.join(actual_tempdir,filename)
    except OSError:
        logging.getLogger('').error('Cannot create a temp file in the specified folder will'
                                    ' use pythons gettempdir method!')
        actual_tempdir = os.path.join(tempfile.gettempdir(),TEMPDIR)
        return os.path.join(actual_tempdir,filename)
    except:
        logging.getLogger('').error('Could not create a directory. Sorry cannot run them')
        raise

def run_tests(remove=None, folder=None):

    if remove is None:
        remove = REMOVE

    global user_tempdir
    user_tempdir=folder

    global actual_tempdir
    try:
        unittest.main()
    finally:
        if remove:
            shutil.rmtree(actual_tempdir,True)

def create_param_dict(param_dict):
    '''Fills a dictionary with some parameters that can be put into a trajectory.
    '''
    param_dict['Normal'] = {}
    param_dict['Numpy'] = {}
    param_dict['Sparse'] ={}
    param_dict['Numpy_2D'] = {}
    param_dict['Numpy_3D'] = {}
    param_dict['Tuples'] ={}

    normal_dict = param_dict['Normal']
    normal_dict['string'] = 'Im a test string!'
    normal_dict['int'] = 42
    normal_dict['double'] = 42.42
    normal_dict['bool'] =True
    normal_dict['trial'] = 0

    numpy_dict=param_dict['Numpy']
    numpy_dict['string'] = np.array(['Uno', 'Dos', 'Tres'])
    numpy_dict['int'] = np.array([1,2,3,4])
    numpy_dict['double'] = np.array([1.0,2.0,3.0,4.0])
    numpy_dict['bool'] = np.array([True,False, True])

    param_dict['Numpy_2D']['double'] = np.array([[1.0,2.0],[3.0,4.0]])
    param_dict['Numpy_3D']['double'] = np.array([[[1.0,2.0],[3.0,4.0]],[[3.0,-3.0],[42.0,41.0]]])

    spsparse_csc = spsp.csc_matrix((2222,22))
    spsparse_csc[1,2] = 44.6

    spsparse_csr = spsp.csr_matrix((2222,22))
    spsparse_csr[1,3] = 44.7

    spsparse_lil = spsp.lil_matrix((2222,22))
    spsparse_lil[3,2] = 44.5

    param_dict['Sparse']['lil_mat'] = spsparse_lil
    param_dict['Sparse']['csc_mat'] = spsparse_csc
    param_dict['Sparse']['csr_mat'] = spsparse_csr

    param_dict['Tuples']['int'] = (1,2,3)
    param_dict['Tuples']['float'] = (44.4,42.1,3.)
    param_dict['Tuples']['str'] = ('1','2wei','dr3i')


def add_params(traj,param_dict):
    '''Adds parameters to a trajectory
    '''
    flat_dict = flatten_dictionary(param_dict,'.')

    for key, val in flat_dict.items():
        if isinstance(val, (np.ndarray,list, tuple)):
            traj.f_add_parameter(ArrayParameter,key,val, )
        elif isinstance(val, (int,str,bool,float)):
            traj.f_add_parameter(Parameter,key,val, comment='Im a comment!')
        elif spsp.isspmatrix(val):
            traj.f_add_parameter(PickleParameter,key,val).v_annotations.f_set(
                **{'Name':key,'Val' :str(val),'Favorite_Numbers:':[1,2,3],
                                 'Second_Fav':np.array([43.0,43.0])})
        else:
            raise RuntimeError('You shall not pass, %s is %s!' % (str(val),str(type(val))))


    traj.f_add_derived_parameter('Another.String', 'Hi, how are you?')
    traj.f_add_result('Peter_Jackson',np.str(['is','full','of','suboptimal ideas']),comment='Only my opinion bro!',)



def simple_calculations(traj, arg1, simple_kwarg):


        # all_mat = traj.csc_mat + traj.lil_mat + traj.csr_mat
        Normal_int= traj.Normal.int
        Sum= np.sum(traj.Numpy.double)

        # result_mat = all_mat * Normal_int * Sum * arg1 * simple_kwarg



        my_dict = {}

        my_dict2={}
        for key, val in traj.parameters.f_to_dict(fast_access=True,short_names=False).items():
            if 'trial' in key:
                continue
            newkey = key.replace('.','_')
            my_dict[newkey] = str(val)
            my_dict2[newkey] = [str(val)+' juhu!']

        my_dict['__FLOAT'] = 44.0
        my_dict['__INT'] = 66
        my_dict['__NPINT'] = np.int_(55)
        my_dict['__INTaRRAy'] = np.array([1,2,3])
        my_dict['__FLOATaRRAy'] = np.array([1.0,2.0,41.0])
        my_dict['__STRaRRAy'] = np.array(['sds','aea','sf'])

        keys = sorted(traj.f_to_dict(short_names=False).keys())
        for idx,key in enumerate(keys):
            keys[idx] = key.replace('.','_')

        traj.f_add_result('List.Of.Keys', dict1=my_dict, dict2=my_dict2)
        traj.f_add_result('DictsNFrame', keys=keys, comment='A dict!')
        traj.f_add_result('ResMatrix',np.array([1.2,2.3]))
        #traj.f_add_derived_parameter('All.To.String', str(traj.f_to_dict(fast_access=True,short_names=False)))

        myframe = pd.DataFrame(data ={'TC1':[1,2,3],'TC2':['Waaa',np.nan,''],'TC3':[1.2,42.2,np.nan]})

        traj.f_get('DictsNFrame').f_set(myframe)

        traj.f_add_result('IStore.SimpleThings',1.0,3,np.float32(5.0), 'Iamstring',(1,2,3),[4,5,6],zwei=2)
        traj.f_add_derived_parameter('mega',33, comment='It is huuuuge!')

        #traj.f_add_result('PickleTerror', result_type=PickleResult, test=traj.SimpleThings)

class TrajectoryComparator(unittest.TestCase):

    def compare_trajectories(self,traj1,traj2):

        old_items = traj1.f_to_dict(fast_access=False)
        new_items = traj2.f_to_dict(fast_access=False)



        self.assertEqual(len(old_items),len(new_items))
        for key,item in new_items.items():
            old_item = old_items[key]
            if key.startswith('config'):
                continue

            if isinstance(item, BaseParameter):
                self.assertTrue(parameters_equal(item,old_item),
                                'For key %s: %s not equal to %s' %(key,str(old_item),str(item)))
            elif isinstance(item,BaseResult):
                self.assertTrue(results_equal(item, old_item),
                                'For key %s: %s not equal to %s' %(key,str(old_item),str(item)))
            else:
                raise RuntimeError('You shall not pass')

            self.assertTrue(nested_equal(item.v_annotations,old_item.v_annotations),'%s != %s' %
                        (item.v_annotations.f_ann_to_str(),old_item.v_annotations.f_ann_to_str()))
