import pymongo
from bson import Binary
import os
import arctic
import arctic.exceptions as aexc
try:
    import cPickle as pickle
except ImportError:
    import pickle

from pypet.storageservice import StorageService
from pypet.pypetlogging import HasLogger

import numpy as np
from pandas import DataFrame, Series, Panel, Panel4D, HDFStore

import pypet.compat as compat
from pypet.utils.decorators import retry
import pypet.utils.ptcompat as ptcompat
import pypet.pypetconstants as pypetconstants
import pypet.pypetexceptions as pex
from pypet._version import __version__ as VERSION
from pypet.parameter import ObjectTable, Parameter
import pypet.naturalnaming as nn
from pypet.pypetlogging import HasLogger, DisableAllLogging
from pypet.storageservice import NodeProcessingTimer, HDF5StorageService


MAX_NAME_LENGTH = 32
TREE_COLL = 'tree'
INFO_COLL = 'info'
RUN_COLL = 'runs'
DATA_COLL = 'data'


class MongoStorageService(StorageService, HasLogger):
    def __init__(self, mongo_host='localhost', mongo_port=27017,
                 mongo_db = None, trajectory=None,
                 max_bulk_length=0, protocol=2,
                 display_time = 20,
                 overwrite_db=False,
                 client_kwargs=None):
        self._set_logger()
        if client_kwargs is None:
            client_kwargs = {}
        self._client_kwargs = client_kwargs
        self._mongo_host = mongo_host
        self._mongo_port = mongo_port

        self._protocol = protocol
        self._max_bulk_length = max_bulk_length
        self._info_coll = None
        self._tree_coll = None
        self._run_coll = None
        self._arctic_lib = None
        self._mode = None
        self._traj_name = None
        self._traj_index = None
        self._traj_stump = None
        self._db_name = None
        self._db = None
        self._bulk = []

        if mongo_db is None and trajectory is not None:
            mongo_db = trajectory.v_name.lower()[:MAX_NAME_LENGTH]
        elif mongo_db is None:
            mongo_db = 'experiments'

        self._create_client()
        self._srvc_set_db_name(mongo_db)
        self._is_open = False
        self._display_time = display_time

        if overwrite_db and self._db_name in self._client.database_names():
            self._client.drop_database(self._db_name)

    def _create_client(self):
        self._client = pymongo.MongoClient(self._mongo_host, self._mongo_port,
                                           connect=False, **self._client_kwargs)
        self._arctic = arctic.Arctic(self._client)

    def __getstate__(self):
        result = super(MongoStorageService, self).__getstate__()
        result['_client'] = None
        result['_arctic'] = None
        result['_info_coll'] = None
        result['_tree_coll'] = None
        result['_run_coll'] = None
        result['_arctic_lib'] = None
        result['_db'] = None
        return result

    def __setstate__(self, statedict):
        super(MongoStorageService, self).__setstate__(statedict)
        self._create_client()

    CLASS_NAME = 'class_name'
    ''' Name of a parameter or result class, is converted to a constructor'''
    COMMENT = 'comment'
    ''' Comment of parameter or result'''
    LENGTH = 'length'
    ''' Length of a parameter if it is explored, no longer in use, only for backwards
    compatibility'''
    LEAF = 'leaf'
    ''' Whether an hdf5 node is a leaf node'''
    ANNOTATIONS = 'annotations'
    '''Annotations entry'''
    GROUPS = 'groups'
    '''Children entry'''
    LINKS = 'links'
    '''Links entry'''
    LINK = 'link'
    '''If elem is link'''
    LEAVES = 'leaves'
    '''Leaves entry'''
    DATA = 'data'
    '''data entry'''

    INFO = 'info'
    '''Info entry'''
    EXPLORATIONS = 'explorations'
    '''Explorations entry'''


    PICKLE_TYPES = (tuple, list, dict, ObjectTable, compat.bytes_type)
    BINARY = 'binary'
    MATRIX = 'matrix'

    @property
    def is_open(self):
        """ Normally the file is opened and closed after each insertion.

        However, the storage service may provide the option to keep the store open and signals
        this via this property.

        """
        return self._is_open

    @property
    def multiproc_safe(self):
        """MongoDB is multiproc sage"""
        return True

    def store(self, msg, stuff_to_store, *args, **kwargs):
        """ Stores a particular item to disk.

        The storage service always accepts these parameters:

        :param trajectory_name: Name or current trajectory and name of top node in hdf5 file

        :param filename: Name of the hdf5 file

        :param file_title: If file needs to be created, assigns a title to the file.


        The following messages (first argument msg) are understood and the following arguments
        can be provided in combination with the message:

            * :const:`pypet.pypetconstants.PREPARE_MERGE` ('PREPARE_MERGE'):

                Called to prepare a trajectory for merging, see also 'MERGE' below.

                Will also be called if merging cannot happen within the same hdf5 file.
                Stores already enlarged parameters and updates meta information.

                :param stuff_to_store: Trajectory that is about to be extended by another one

                :param changed_parameters:

                    List containing all parameters that were enlarged due to merging

                :param old_length:

                    Old length of trajectory before merge

            * :const:`pypet.pypetconstants.MERGE` ('MERGE')

                Note that before merging within HDF5 file, the storage service will be called
                with msg='PREPARE_MERGE' before, see above.

                Raises a ValueError if the two trajectories are not stored within the very
                same hdf5 file. Then the current trajectory needs to perform the merge slowly
                item by item.

                Merges two trajectories, parameters are:

                :param stuff_to_store: The trajectory data is merged into

                :param other_trajectory_name: Name of the other trajectory

                :param rename_dict:

                    Dictionary containing the old result and derived parameter names in the
                    other trajectory and their new names in the current trajectory.

                :param move_nodes:

                    Whether to move the nodes from the other to the current trajectory

                :param delete_trajectory:

                    Whether to delete the other trajectory after merging.

            * :const:`pypet.pypetconstants.BACKUP` ('BACKUP')

                :param stuff_to_store: Trajectory to be backed up

                :param backup_filename:

                    Name of file where to store the backup. If None the backup file will be in
                    the same folder as your hdf5 file and named 'backup_XXXXX.hdf5'
                    where 'XXXXX' is the name of your current trajectory.

            * :const:`pypet.pypetconstants.TRAJECTORY` ('TRAJECTORY')

                Stores the whole trajectory

                :param stuff_to_store: The trajectory to be stored

                :param only_init:

                    If you just want to initialise the store. If yes, only meta information about
                    the trajectory is stored and none of the nodes/leaves within the trajectory.

                :param store_data:

                    How to store data, the following settings are understood:

                     :const:`pypet.pypetconstants.STORE_NOTHING`: (0)

                        Nothing is stored

                    :const:`pypet.pypetconstants.STORE_DATA_SKIPPING`: (1)

                        Data of not already stored nodes is stored

                    :const:`pypet.pypetconstants.STORE_DATA`: (2)

                        Data of all nodes is stored. However, existing data on disk is left
                        untouched.

                    :const:`pypet.pypetconstants.OVERWRITE_DATA`: (3)

                        Data of all nodes is stored and data on disk is overwritten.
                        May lead to fragmentation of the HDF5 file. The user is adviced
                        to recompress the file manually later on.

            * :const:`pypet.pypetconstants.SINGLE_RUN` ('SINGLE_RUN')

                :param stuff_to_store: The trajectory

                :param store_data: How to store data see above

                :param store_final: If final meta info should be stored

            * :const:`pypet.pypetconstants.LEAF`

                Stores a parameter or result

                Note that everything that is supported by the storage service and that is
                stored to disk will be perfectly recovered.
                For instance, you store a tuple of numpy 32 bit integers, you will get a tuple
                of numpy 32 bit integers after loading independent of the platform!

                :param stuff_to_sore: Result or parameter to store

                    In order to determine what to store, the function '_store' of the parameter or
                    result is called. This function returns a dictionary with name keys and data to
                    store as values. In order to determine how to store the data, the storage flags
                    are considered, see below.

                    The function '_store' has to return a dictionary containing values only from
                    the following objects:

                        * python natives (int, long, str, bool, float, complex),

                        *
                            numpy natives, arrays and matrices of type np.int8-64, np.uint8-64,
                            np.float32-64, np.complex, np.str

                        *

                            python lists and tuples of the previous types
                            (python natives + numpy natives and arrays)
                            Lists and tuples are not allowed to be nested and must be
                            homogeneous, i.e. only contain data of one particular type.
                            Only integers, or only floats, etc.

                        *

                            python dictionaries of the previous types (not nested!), data can be
                            heterogeneous, keys must be strings. For example, one key-value-pair
                            of string and int and one key-value pair of string and float, and so
                            on.

                        * pandas DataFrames_

                        * :class:`~pypet.parameter.ObjectTable`

                    .. _DataFrames: http://pandas.pydata.org/pandas-docs/dev/dsintro.html#dataframe

                    The keys from the '_store' dictionaries determine how the data will be named
                    in the hdf5 file.

                :param store_data:

                    How to store the data, see above for a descitpion.

                :param overwrite:

                    Can be used if parts of a leaf should be replaced. Either a list of
                    HDF5 names or `True` if this should account for all.

            * :const:`pypet.pypetconstants.DELETE` ('DELETE')

                Removes an item from disk. Empty group nodes, results and non-explored
                parameters can be removed.

                :param stuff_to_store: The item to be removed.

                :param delete_only:

                    Potential list of parts of a leaf node that should be deleted.

                :param remove_from_item:

                    If `delete_only` is used, whether deleted nodes should also be erased
                    from the leaf nodes themseleves.

                :param recursive:

                    If you want to delete a group node you can recursively delete all its
                    children.

            * :const:`pypet.pypetconstants.GROUP` ('GROUP')

                :param stuff_to_store: The group to store

                :param store_data: How to store data

                :param recursive: To recursively load everything below.

                :param max_depth:

                    Maximum depth in case of recursion. `None` for no limit.

            * :const:`pypet.pypetconstants.TREE`

                Stores a single node or a full subtree

                :param stuff_to_store: Node to store

                :param store_data: How to store data

                :param recursive: Whether to store recursively the whole sub-tree

                :param max_depth:

                    Maximum depth in case of recursion. `None` for no limit.

            * :const:`pypet.pypetconstants.DELETE_LINK`

                Deletes a link from hard drive

                :param name: The full colon separated name of the link

            * :const:`pypet.pypetconstants.LIST`

                .. _store-lists:

                Stores several items at once

                :param stuff_to_store:

                    Iterable whose items are to be stored. Iterable must contain tuples,
                    for example `[(msg1,item1,arg1,kwargs1),(msg2,item2,arg2,kwargs2),...]`

            * :const:`pypet.pypetconstants.ACCESS_DATA`

                Requests and manipulates data within the storage.
                Storage must be open.

                :param stuff_to_store:

                    A colon separated name to the data path

                :param item_name:

                    The name of the data item to interact with

                :param request:

                    A functional request in form of a string

                :param args:

                    Positional arguments passed to the reques

                :param kwargs:

                    Keyword arguments passed to the request


            * :const:`pypet.pypetconstants.CLOSE_FILE`

                Closes the mongo connection.

                :param stuff_to_store: ``None``


        :raises: NoSuchServiceError if message or data is not understood

        """
        opened = False
        try:

            opened = self._srvc_opening_routine('a', msg, kwargs)

            if msg == pypetconstants.MERGE:
                self._trj_merge_trajectories(*args, **kwargs)

            elif msg == pypetconstants.BACKUP:
                self._trj_backup_trajectory(stuff_to_store, *args, **kwargs)

            elif msg == pypetconstants.PREPARE_MERGE:
                self._trj_prepare_merge(stuff_to_store, *args, **kwargs)

            elif msg == pypetconstants.TRAJECTORY:
                self._trj_store_trajectory(stuff_to_store, *args, **kwargs)

            elif msg == pypetconstants.SINGLE_RUN:
                self._srn_store_single_run(stuff_to_store, *args, **kwargs)

            elif msg in pypetconstants.LEAF:
                self._prm_store_parameter_or_result(stuff_to_store, *args, **kwargs)

            elif msg == pypetconstants.DELETE:
                self._all_delete_parameter_or_result_or_group(stuff_to_store, *args, **kwargs)

            elif msg == pypetconstants.GROUP:
                self._grp_store_group(stuff_to_store, *args, **kwargs)

            elif msg == pypetconstants.TREE:
                self._tree_store_sub_branch(stuff_to_store, *args, **kwargs)

            elif msg == pypetconstants.DELETE_LINK:
                self._lnk_delete_link(stuff_to_store, *args, **kwargs)

            elif msg == pypetconstants.LIST:
                self._srvc_store_several_items(stuff_to_store, *args, **kwargs)

            elif msg == pypetconstants.ACCESS_DATA:
                return self._hdf5_interact_with_data(stuff_to_store, *args, **kwargs)

            else:
                raise pex.NoSuchServiceError('I do not know how to handle `%s`' % msg)

        except:
            self._logger.error('Failed storing `%s`' % str(stuff_to_store))
            raise
        finally:
            self._srvc_closing_routine(opened)


    def load(self, msg, stuff_to_load, *args, **kwargs):
        """Loads a particular item from disk.

        The storage service always accepts these parameters:

        :param trajectory_name: Name of current trajectory and name of top node in hdf5 file.

        :param filename: Name of the hdf5 file


        The following messages (first argument msg) are understood and the following arguments
        can be provided in combination with the message:

            * :const:`pypet.pypetconstants.TRAJECTORY` ('TRAJECTORY')

                Loads a trajectory.

                :param stuff_to_load: The trajectory

                :param as_new: Whether to load trajectory as new

                :param load_parameters: How to load parameters and config

                :param load_derived_parameters: How to load derived parameters

                :param load_results: How to load results

                :param force: Force load in case there is a pypet version mismatch

                You can specify how to load the parameters, derived parameters and results
                as follows:

                :const:`pypet.pypetconstants.LOAD_NOTHING`: (0)

                    Nothing is loaded

                :const:`pypet.pypetconstants.LOAD_SKELETON`: (1)

                    The skeleton including annotations are loaded, i.e. the items are empty.
                    Non-empty items in RAM are left untouched.

                :const:`pypet.pypetconstants.LOAD_DATA`: (2)

                    The whole data is loaded.
                    Only empty or in RAM non-existing instance are filled with the
                    data found on disk.

                :const:`pypet.pypetconstants.OVERWRITE_DATA`: (3)

                    The whole data is loaded.
                    If items that are to be loaded are already in RAM and not empty,
                    they are emptied and new data is loaded from disk.

            * :const:`pypet.pypetconstants.LEAF` ('LEAF')

                Loads a parameter or result.

                :param stuff_to_load: The item to be loaded

                :param load_data: How to load data

                :param load_only:

                    If you load a result, you can partially load it and ignore the
                    rest of the data. Just specify the name of the data you want to load.
                    You can also provide a list,
                    for example `load_only='spikes'`, `load_only=['spikes','membrane_potential']`.

                    Issues a warning if items cannot be found.

                :param load_except:

                    If you load a result you can partially load in and specify items
                    that should NOT be loaded here. You cannot use `load_except` and
                    `load_only` at the same time.

            * :const:`pypet.pyetconstants.GROUP`

                Loads a group a node (comment and annotations)

                :param recursive:

                    Recursively loads everything below

                :param load_data:

                    How to load stuff if ``recursive=True``
                    accepted values as above for loading the trajectory

                :param max_depth:

                    Maximum depth in case of recursion. `None` for no limit.

            * :const:`pypet.pypetconstants.TREE` ('TREE')

                Loads a whole subtree

                :param stuff_to_load: The parent node (!) not the one where loading starts!

                :param child_name: Name of child node that should be loaded

                :param recursive: Whether to load recursively the subtree below child

                :param load_data:

                    How to load stuff, accepted values as above for loading the trajectory

                :param max_depth:

                    Maximum depth in case of recursion. `None` for no limit.

                :param trajectory: The trajectory object

            * :const:`pypet.pypetconstants.LIST` ('LIST')

                Analogous to :ref:`storing lists <store-lists>`

        :raises:

            NoSuchServiceError if message or data is not understood

            DataNotInStorageError if data to be loaded cannot be found on disk

        """
        opened = False
        try:
            opened = self._srvc_opening_routine('r', msg, kwargs)

            if msg == pypetconstants.TRAJECTORY:
                self._trj_load_trajectory(stuff_to_load, *args, **kwargs)

            elif msg == pypetconstants.LEAF:
                self._prm_load_parameter_or_result(stuff_to_load, *args, **kwargs)

            elif msg == pypetconstants.GROUP:
                self._grp_load_group(stuff_to_load, *args, **kwargs)

            elif msg == pypetconstants.TREE:
                self._tree_load_sub_branch(stuff_to_load, *args, **kwargs)

            elif msg == pypetconstants.LIST:
                self._srvc_load_several_items(stuff_to_load, *args, **kwargs)

            else:
                raise pex.NoSuchServiceError('I do not know how to handle `%s`' % msg)

        except aexc.NoDataFoundException as exc:
            self._logger.error('Failed loading  `%s`' % str(stuff_to_load))
            raise pex.DataNotInStorageError(repr(exc))
        except:
            self._logger.error('Failed loading  `%s`' % str(stuff_to_load))
            raise
        finally:
            self._srvc_closing_routine(opened)

    def _trj_store_trajectory(self, traj, only_init=False,
                          store_data=pypetconstants.STORE_DATA,
                          max_depth=None):
        """ Stores a trajectory to an hdf5 file

        Stores all groups, parameters and results

        """
        if not only_init:
            self._logger.info('Start storing Trajectory `%s`.' % self._traj_name)
        else:
            self._logger.info('Initialising storage or updating meta data of Trajectory `%s`.' %
                              self._traj_name)
            store_data = pypetconstants.STORE_NOTHING

        # In case we accidentally chose a trajectory name that already exist
        # We do not want to mess up the stored trajectory but raise an Error
        if not traj._stored and self._info_coll.count() > 0:
            raise RuntimeError('You want to store a completely new trajectory with name'
                               ' `%s` but this trajectory is already found in DB `%s`' %
                               (traj.v_name, self._db_name))

        # Store meta information
        self._trj_store_meta_data(traj)

        # Store group data
        self._grp_store_group(traj, with_links=False)

        # # Store recursively the config subtree
        # self._tree_store_recursively(pypetconstants.LEAF,traj.config,self._trajectory_group)

        if store_data in (pypetconstants.STORE_DATA_SKIPPING,
                          pypetconstants.STORE_DATA,
                          pypetconstants.OVERWRITE_DATA):

            counter = 0
            maximum_display_other = 10
            name_set = set(['parameters', 'config', 'derived_parameters', 'results'])

            for child_name in traj._children:

                if child_name in name_set:
                    self._logger.info('Storing branch `%s`.' % child_name)
                else:
                    if counter < maximum_display_other:
                        self._logger.info('Storing branch/node `%s`.' % child_name)
                    elif counter == maximum_display_other:
                        self._logger.info('To many branches or nodes at root for display. '
                                          'I will not inform you about storing anymore. '
                                          'Branches are stored silently in the background. '
                                          'Do not worry, I will not freeze! Pinky promise!!!')
                    counter += 1

                # Store recursively the elements
                self._tree_store_sub_branch(traj, child_name, store_data=store_data,
                                            with_links=True, recursive=True, max_depth=max_depth)

            self._logger.info('Finished storing Trajectory `%s`.' % self._traj_name)
        else:
            self._logger.info('Finished init or meta data update for `%s`.' %
                              self._traj_name)
        traj._stored = True

    def _trj_fill_run_table(self, traj, start, stop):
        """Fills the `run` overview table with information.

        Will also update new information.

        """

        rows = []
        updated_run_information = traj._updated_run_information
        for idx in compat.xrange(start, stop):
            info_dict = traj._run_information[traj._single_run_ids[idx]]
            insert_dict = dict(_id = info_dict['idx'])
            insert_dict.update(info_dict)
            rows.append(pymongo.InsertOne(insert_dict))
            updated_run_information.discard(idx)

        if rows:
            self._run_coll.bulk_write(rows)

        # Store all runs that are updated and that have not been stored yet
        rows = []
        indices = []
        for idx in updated_run_information:
            info_dict = traj.f_get_run_information(idx, copy=False)
            # insert_dict = dict(_id = info_dict['idx'])
            # insert_dict.update(info_dict)
            insert_dict = info_dict.copy()
            rows.append(pymongo.UpdateOne({'_id': info_dict['idx']},
                                    {'$set': insert_dict}))
            indices.append(idx)

        if rows:
            self._run_coll.bulk_write(rows)

        traj._updated_run_information = set()

    def _srvc_store_several_items(self, iterable, *args, **kwargs):
        """Stores several items from an iterable

        Iterables are supposed to be of a format like `[(msg, item, args, kwarg),...]`
        If `args` and `kwargs` are not part of a tuple, they are taken from the
        current `args` and `kwargs` provided to this function.

        """
        for input_tuple in iterable:
            msg = input_tuple[0]
            item = input_tuple[1]
            if len(input_tuple) > 2:
                args = input_tuple[2]
            if len(input_tuple) > 3:
                kwargs = input_tuple[3]
            if len(input_tuple) > 4:
                raise RuntimeError('You shall not pass!')

            self.store(msg, item, *args, **kwargs)

    def _srvc_load_several_items(self, iterable, *args, **kwargs):
        """Loads several items from an iterable

        Iterables are supposed to be of a format like `[(msg, item, args, kwarg),...]`
        If `args` and `kwargs` are not part of a tuple, they are taken from the
        current `args` and `kwargs` provided to this function.

        """
        for input_tuple in iterable:
            msg = input_tuple[0]
            item = input_tuple[1]
            if len(input_tuple) > 2:
                args = input_tuple[2]
            if len(input_tuple) > 3:
                kwargs = input_tuple[3]
            if len(input_tuple) > 4:
                raise RuntimeError('You shall not pass!')

            self.load(msg, item, *args, **kwargs)

    def _srvc_opening_routine(self, mode, msg=None, kwargs=()):
        """Opens an hdf5 file for reading or writing

        The file is only opened if it has not been opened before (i.e. `self._hdf5file is None`).

        :param mode:

            'a' for appending

            'r' for reading

                Unfortunately, pandas currently does not work with read-only mode.
                Thus, if mode is chosen to be 'r', the file will still be opened in
                append mode.

        :param msg:

            Message provided to `load` or `store`. Only considered to check if a trajectory
            was stored before.

        :param kwargs:

            Arguments to extract file information from

        :return:

            `True` if file is opened

            `False` if the file was already open before calling this function

        """
        self._mode = mode
        self._srvc_extract_file_information(kwargs)

        if not self.is_open:
            if 'a' in mode:

                if self._db_name is not None:
                    if self._db_name not in self._client.database_names():
                        # If we want to store individual items we we have to check if the
                        # trajectory has been stored before
                        if not msg == pypetconstants.TRAJECTORY:
                            raise ValueError('Your trajectory cannot be found in the database, '
                                             'please use >>traj.f_store()<< '
                                             'before storing anything else.')
                        else:
                            self._arctic.initialize_library(self._db_name + '.' +
                                                            self._traj_stump + '_' + DATA_COLL)
                else:
                    raise ValueError('I don`t know which trajectory to load')
                self._logger.debug('Opening MongoDB `%s` in mode `a` with trajectory `%s`' %
                                   (self._db_name, self._traj_name))

            elif mode == 'r':
                if self._traj_name is not None and self._traj_index is not None:
                    raise ValueError('Please specify either a name of a trajectory or an index, '
                                     'but not both at the same time.')

                self._logger.debug('Opening MongoDB `%s` in mode `r` with trajectory `%s`' %
                                   (self._db_name, self._traj_name))

            else:
                raise RuntimeError('You shall not pass!')

            # Keep a reference to the top trajectory node
            self._db = self._client[self._db_name]
            self._arctic_lib = self._arctic[self._db_name + '.' + self._traj_stump + '_'
                                            + DATA_COLL]
            self._tree_coll = self._db[self._traj_stump + '_' + TREE_COLL]
            self._info_coll = self._db[self._traj_stump + '_' + INFO_COLL]
            self._run_coll = self._db[self._traj_stump + '_' + RUN_COLL]

            self._node_processing_timer = NodeProcessingTimer(display_time=self._display_time,
                                                              logger_name=self._logger.name)
            self._is_open = True
            return True

    def _srvc_set_db_name(self, db_name):
        if db_name.startswith(self._arctic.DB_PREFIX):
            self._db_name = db_name
        else:
            self._db_name = self._arctic.DB_PREFIX + '_' + db_name

    def _srvc_extract_file_information(self, kwargs):
        """Extracts file information from kwargs.

        Note that `kwargs` is not passed as `**kwargs` in order to also
        `pop` the elements on the level of the function calling `_srvc_extract_file_information`.

        """
        if 'mongo_db' in kwargs:
            self._srvc_set_db_name(kwargs.pop('mongo_db'))

        if 'trajectory_name' in kwargs:
            traj_name = kwargs.pop('trajectory_name')
            if self._traj_name is not None and (traj_name != self._traj_name):
                self._srvc_closing_routine(True)
            self._traj_name = traj_name
            self._traj_stump = self._traj_name.lower()[:MAX_NAME_LENGTH]
            if self._db_name is None:
                self._srvc_set_db_name(self._traj_stump)

        if 'trajectory_index' in kwargs:
            self._traj_index = kwargs.pop('trajectory_index')
            if self._traj_index is not None:
                all_dbs = self._client.database_names()
                all_trajs = [x.endswith(TREE_COLL) for x in all_dbs]
                self._traj_stump = sorted(all_trajs)[self._traj_index].split('_'+TREE_COLL)[0]
                self._traj_name = self._traj_stump + '...'

    def _srvc_closing_routine(self, opened):
        """Routine to close an hdf5 file

        The file is closed only when `closing=True`. `closing=True` means that
        the file was opened in the current highest recursion level. This prevents re-opening
        and closing of the file if `store` or `load` are called recursively.

        """
        if opened:
            self._srvc_flush_tree_db()
            self._client.close()
            self._client._topology._pid = None
            self._info_coll = None
            self._tree_coll = None
            self._arctic_lib = None
            self._mode = None
            self._traj_name = None
            self._db = None
            self._traj_stump = None
            self._is_open = False
            self._traj_index = None

    @retry(9, Exception, 0.01, 'pypet.retry')
    def _retry_write(self):
        try:
            self._tree_coll.bulk_write(self._bulk)
        except Exception as exc:
            raise

    def _srvc_flush_tree_db(self):
        if self._bulk:
            try:
                self._retry_write()
            except:
                self._logger.error('Bulk write error with bul `%s`' % str(self._bulk))
                raise
            self._bulk = []

    def _srvc_update_db(self, entry, _id, how='$setOnInsert', upsert=True):
        if entry:
            add = {how: entry}
        else:
            add = {how: {'_': '_'}}
        u = pymongo.UpdateOne({'_id': _id}, add, upsert=upsert)
        self._bulk.append(u)
        if len(self._bulk) > self._max_bulk_length:
            self._srvc_flush_tree_db()

    def _trj_store_meta_data(self, traj):
        """ Stores general information about the trajectory in the hdf5file.

        The `info` table will contain the name of the trajectory, it's timestamp, a comment,
        the length (aka the number of single runs), and the current version number of pypet.

        Also prepares the desired overview tables and fills the `run` table with dummies.

        """

        # Description of the `info` table

        descriptiondict = {#'_id' : self.INFO,
                           'name': traj.v_name,
                           'time': traj.v_time,
                           'timestamp': traj.v_timestamp,
                           'comment': traj.v_comment,
                           'length': len(traj),
                           'version': traj.v_version,
                           'python': traj.v_python}
        # 'loaded_from' : pt.StringCol(pypetconstants.HDF5_STRCOL_MAX_LOCATION_LENGTH)}
        self._info_coll.update_one({'_id': self.INFO}, {'$set': descriptiondict}, upsert=True)

        # Fill table with dummy entries starting from the current table size
        actual_rows = self._run_coll.count()
        self._trj_fill_run_table(traj, actual_rows, len(traj._run_information))

        # Store the list of explored paramters
        self._trj_store_explorations(traj)

    def _trj_load_exploration(self, traj):
        """Recalls names of all explored parameters"""
        explorations_entry = self._info_coll.find_one({'_id': self.EXPLORATIONS})
        if explorations_entry is not None:
            explorations_list = explorations_entry[self.EXPLORATIONS]
            for param_name in explorations_list:
                param_name = str(param_name)
                if param_name not in traj._explored_parameters:
                    traj._explored_parameters[param_name] = None

    def _trj_store_explorations(self, traj):
        """Stores a all explored parameter names for internal recall"""
        nexplored = len(traj._explored_parameters)
        if nexplored > 0:
            explorations_entry = self._info_coll.find_one({'_id': self.EXPLORATIONS})
            if explorations_entry is not None:
                explored_list = explorations_entry[self.EXPLORATIONS]
                if len(explored_list) != nexplored:
                    self._info_coll.delete_one({'_id': self.EXPLORATIONS})
            explored_list = compat.listkeys(traj._explored_parameters)
            self._info_coll.update_one( {'_id': self.EXPLORATIONS},  {'$set':
                                                        {#'_id': self.EXPLORATIONS,
                                                       self.EXPLORATIONS: explored_list}},
                                                        upsert=True)

    def _tree_store_sub_branch(self, traj_node, branch_name,
                               store_data=pypetconstants.STORE_DATA,
                               with_links=True,
                               recursive=False,
                               max_depth=None):
        """Stores data starting from a node along a branch and starts recursively loading
        all data at end of branch.

        :param traj_node: The node where storing starts

        :param branch_name:

            A branch along which storing progresses. Colon Notation is used:
            'group1.group2.group3' loads 'group1', then 'group2', then 'group3', and then finally
            recursively all children and children's children below 'group3'.

        :param store_data: How data should be stored

        :param with_links: If links should be stored

        :param recursive:

            If the rest of the tree should be recursively stored

        :param max_depth:

            Maximum depth to store

        """
        if store_data == pypetconstants.STORE_NOTHING:
            return

        if max_depth is None:
            max_depth = float('inf')

        node_entry = self._tree_coll.find_one({'_id': traj_node.v_full_name})
        if node_entry is None:
            # Get parent hdf5 node
            location = traj_node.v_full_name
            self._logger.debug('Cannot store `%s` the parental hdf5 node with location `%s` does '
                                     'not exist on disk.' %
                                     (traj_node.v_name, location))
            if traj_node.v_is_leaf:
                self._logger.error('Cannot store `%s` the parental hdf5 '
                                   'node with locations `%s` does '
                                   'not exist on disk! The child '
                                   'you want to store is a leaf node,'
                                   'that cannot be stored without '
                                   'the parental node existing on '
                                   'disk.' % (traj_node.v_name, location))
                raise
            else:
                self._logger.debug('I will try to store the path from trajectory root to '
                                     'the child now.')

                self._tree_store_sub_branch(traj_node._nn_interface._root_instance,
                                            traj_node.v_full_name + '.' + branch_name,
                                            store_data=store_data, with_links=with_links,
                                            recursive=recursive,
                                            max_depth=max_depth + traj_node.v_depth)
                return

        current_depth = 1

        split_names = branch_name.split('.')

        leaf_name = split_names.pop()

        for name in split_names:
            if current_depth > max_depth:
                return
            # Store along a branch
            self._tree_store_nodes_dfs(traj_node, name, store_data=store_data,
                                       with_links=with_links,
                                       recursive=False, max_depth=max_depth,
                                       current_depth=current_depth)
            current_depth += 1

            traj_node = traj_node._children[name]

        # Store final group and recursively everything below it
        if current_depth <= max_depth:
            self._tree_store_nodes_dfs(traj_node, leaf_name, store_data=store_data,
                               with_links=with_links, recursive=recursive,
                               max_depth=max_depth, current_depth=current_depth)

    def _grp_store_group(self, traj_group, store_data=pypetconstants.STORE_DATA,
                         with_links=True, recursive=False, max_depth=None,
                         parent_name=None):
        """Stores a group node.

        For group nodes only annotations and comments need to be stored.

        """
        if store_data == pypetconstants.STORE_NOTHING:
            return
        elif store_data == pypetconstants.STORE_DATA_SKIPPING and traj_group._stored:
            self._logger.debug('Already found `%s` on disk I will not store it!' %
                                   traj_group.v_full_name)
        elif not recursive:

            overwrite = store_data == pypetconstants.OVERWRITE_DATA

            if overwrite:
                option = '$set'
            else:
                option = '$setOnInsert'

            data = {}#'_id': traj_group.v_full_name}

            if type(traj_group) not in (nn.NNGroupNode, nn.ConfigGroup, nn.ParameterGroup,
                                             nn.DerivedParameterGroup, nn.ResultGroup):
                data[self.CLASS_NAME] = traj_group.f_get_class_name()

            if traj_group.v_comment != '':
                data[self.COMMENT] = traj_group.v_comment

            data = self._ann_store_annotations(traj_group, data)

            self._srvc_update_db(data, _id=traj_group.v_full_name, how=option)
            traj_group._stored = True

            if not traj_group.v_is_root:
                if parent_name is None:
                    parent_name = traj_group.v_location

                self._srvc_update_db(entry={self.GROUPS: traj_group.v_name},
                                             _id= parent_name,
                                             how='$addToSet')

            # Signal completed node loading
            self._node_processing_timer.signal_update()

        if recursive:
            parent_traj_group = traj_group.f_get_parent()

            self._tree_store_nodes_dfs(parent_traj_group, traj_group.v_name, store_data=store_data,
                                       with_links=with_links, recursive=recursive,
                                       max_depth=max_depth, current_depth=0)

    def _ann_store_annotations(self, traj_node, data):
        """Stores annotations into data."""

        if not traj_node.v_annotations.f_is_empty():
            data[self.ANNOTATIONS] = Binary(pickle.dumps(traj_node.v_annotations.f_to_dict(),
                                                  protocol=self._protocol))

        return data

    def _tree_store_nodes_dfs(self, parent_traj_node, name, store_data, with_links, recursive,
                          max_depth, current_depth):
        """Stores a node to hdf5 and if desired stores recursively everything below it.

        :param parent_traj_node: The parental node
        :param name: Name of node to be stored
        :param store_data: How to store data
        :param with_links: If links should be stored
        :param recursive: Whether to store recursively the subtree
        :param max_depth: Maximum recursion depth in tree
        :param current_depth: Current depth

        """
        if max_depth is None:
            max_depth = float('inf')

        store_list = [(parent_traj_node, name, current_depth)]

        while store_list:
            parent_traj_node, name, current_depth = store_list.pop()

            # Check if we create a link
            if name in parent_traj_node._links:
                if with_links:
                    self._tree_store_link(parent_traj_node, name)
                continue

            traj_node = parent_traj_node._children[name]

            if traj_node.v_is_leaf:
                self._prm_store_parameter_or_result(traj_node, store_data=store_data,
                                                    parent_name=parent_traj_node.v_full_name)

            else:
                self._grp_store_group(traj_node, store_data=store_data, with_links=with_links,
                                      recursive=False, max_depth=max_depth,
                                      parent_name=parent_traj_node.v_full_name)

                if recursive and current_depth < max_depth:
                    for child in compat.iterkeys(traj_node._children):
                        store_list.append((traj_node, child, current_depth + 1))

    def _tree_store_link(self, node_in_traj, link):
        """Creates a soft link.

        :param node_in_traj: parental node
        :param store_data: how to store data
        :param link: name of link
        """

        self._srvc_flush_tree_db()
        linked_traj_node = node_in_traj._links[link]
        linking_name = linked_traj_node.v_full_name
        entry = self._tree_coll.find_one({'_id' : linking_name})
        if entry is None:
            self._logger.debug('Could not store link `%s` under `%s` immediately, '
                               'need to store `%s` first. '
                               'Will store the link right after.' % (link,
                                                                     node_in_traj.v_full_name,
                                                                     linked_traj_node.v_full_name))
            root = node_in_traj._nn_interface._root_instance
            self._tree_store_sub_branch(root, linked_traj_node.v_full_name,
                                        store_data=pypetconstants.STORE_DATA_SKIPPING,
                                        with_links=False, recursive=False)
        if node_in_traj.v_is_root:
            full_name = link
        else:
            full_name = node_in_traj.v_full_name + '.' + link
        self._srvc_update_db(entry={#'_id': full_name,
                                    self.LINK: linking_name},
                             _id = full_name)
        self._srvc_update_db(entry={self.LINKS: link},
                                         _id= node_in_traj.v_full_name,
                                         how='$addToSet')

    def _prm_store_parameter_or_result(self,
                                       instance,
                                       store_data=pypetconstants.STORE_DATA,
                                       overwrite=None,
                                       with_links=False,
                                       recursive=False,
                                       parent_name=None,
                                       **kwargs):
        """Stores a parameter or result to hdf5.

        :param instance:

            The instance to be stored

        :param store_data:

            How to store data

        :param overwrite:

            Instructions how to overwrite data

        :param with_links:

            Placeholder because leaves have no links

        :param recursive:

            Placeholder, because leaves have no children

        """
        if store_data == pypetconstants.STORE_NOTHING:
            return
        elif store_data == pypetconstants.STORE_DATA_SKIPPING and instance._stored:
            self._logger.debug('Already found `%s` on disk I will not store it!' %
                                   instance.v_full_name)
            return
        elif store_data == pypetconstants.OVERWRITE_DATA:
            if not overwrite:
                overwrite = True

        fullname = instance.v_full_name
        self._logger.debug('Storing `%s`.' % fullname)

        # kwargs_flags = {} # Dictionary to change settings
        # old_kwargs = {}
        store_dict = {}

        try:
            # Get the data to store from the instance
            if not instance.f_is_empty():
                store_dict = instance._store()

            if overwrite:
                entry = self._tree_coll.find_one({'_id': instance.v_full_name})
                if isinstance(overwrite, compat.base_type):
                    overwrite = [overwrite]

                if overwrite is True:
                    if entry is not None:
                        to_delete = [instance.v_full_name + '.' + x for x in entry[self.DATA]]
                        self._all_delete_parameter_or_result_or_group(instance,
                                                                      delete_only=to_delete)

                elif isinstance(overwrite, (list, tuple)):
                    overwrite_set = set(overwrite)
                    key_set = set(store_dict.keys())

                    stuff_not_to_be_overwritten = overwrite_set - key_set

                    if overwrite != 'v_annotations' and len(stuff_not_to_be_overwritten) > 0:
                        self._logger.warning('Cannot overwrite `%s`, these items are not supposed to '
                                             'be stored by the leaf node.' %
                                             str(stuff_not_to_be_overwritten))

                    stuff_to_overwrite = overwrite_set & key_set
                    if len(stuff_to_overwrite) > 0:
                        self._all_delete_parameter_or_result_or_group(instance,
                                                                      delete_only=list(
                                                                          stuff_to_overwrite))
                else:
                    raise ValueError('Your value of overwrite `%s` is not understood. '
                                     'Please pass `True` of a list of strings to fine grain '
                                     'overwriting.' % str(overwrite))

            # Store meta information and annotations
            if overwrite:
                option = '$set'
            else:
                option = '$setOnInsert'

            data = {#'_id': instance.v_full_name,
                    self.CLASS_NAME: instance.f_get_class_name(),
                    self.LEAF: True}

            if instance.v_comment != '':
                data[self.COMMENT] = instance.v_comment

            if overwrite != 'v_annotations':

                data = self._ann_store_annotations(instance, data)

            self._srvc_update_db(data, _id=instance.v_full_name, how=option)

            if overwrite == 'v_annotations':
                data = self._ann_store_annotations(instance, {})
                self._srvc_update_db(data, _id=instance.v_full_name, how='$set')

            instance._stored = True

            if parent_name is None:
                parent_name = instance.v_location
            self._srvc_update_db(entry={self.LEAVES: instance.v_name},
                                         _id= parent_name,
                                         how='$addToSet')

            self._prm_store_from_dict(fullname, store_dict)

            #self._logger.debug('Finished Storing `%s`.' % fullname)
            # Signal completed node loading
            self._node_processing_timer.signal_update()

        except:
            # I anything fails, we want to remove the data of the parameter again
            self._logger.error(
                'Failed storing leaf `%s`. I will remove the data I added  again.' % fullname)
            # Delete data
            self._all_delete_parameter_or_result_or_group(instance)
            raise

    def _prm_store_from_dict(self, fullname, store_dict):
        """Stores a `store_dict`"""
        for key in store_dict:
            self._srvc_update_db({self.DATA: key}, _id = fullname, how='$addToSet')
        self._srvc_flush_tree_db()
        for key, data_to_store in store_dict.items():
            name = fullname + '.' + key
            if type(data_to_store) in self.PICKLE_TYPES:
                data_to_store = Binary(pickle.dumps(data_to_store, protocol=self._protocol))
                metadata = {self.BINARY: True}
            elif type(data_to_store) is np.matrix:
                metadata = {self.MATRIX: True}
            else:
                metadata = None
            try:
                self._arctic_lib.write(name, data_to_store, metadata=metadata)
            except:
                raise

    def _all_delete_parameter_or_result_or_group(self, instance,
                                                 delete_only=None,
                                                 remove_from_item=False,
                                                 recursive=False,
                                                 entry=None):
        """Removes a parameter or result or group from the hdf5 file.

        :param instance: Instance to be removed

        :param delete_only:

            List of elements if you only want to delete parts of a leaf node. Note that this
            needs to list the names of the hdf5 subnodes. BE CAREFUL if you erase parts of a leaf.
            Erasing partly happens at your own risk, it might be the case that you can
            no longer reconstruct the leaf from the leftovers!

        :param remove_from_item:

            If using `delete_only` and `remove_from_item=True` after deletion the data item is
            also removed from the `instance`.


        """
        split_name = instance.v_location.split('.')
        if entry is None:
            entry = self._tree_coll.find_one({'_id': instance.v_full_name})
        if entry is None:
            self._logger.warning('Could not delete `%s. Entry not found!' % instance.v_full_name)
            return

        self._srvc_flush_tree_db()


        if delete_only is None:
            if instance.v_is_group and not recursive and (len(entry.get(self.GROUPS,{})) +
                    len(entry.get(self.LEAVES,{})) + len(entry.get(self.LINKS, {})) != 0):
                    raise TypeError('You cannot remove the group `%s`, it has children, please '
                                    'use `recursive=True` to enforce removal.' %
                                    instance.v_full_name)
            if instance.v_is_leaf:
                for elem in entry.get(self.DATA, []):
                    self._arctic_lib.delete(instance.v_full_name + '.' + elem)
                self._tree_coll.update_one({'_id': instance.v_location},
                                       {'$pull': {self.LEAVES: instance.v_name}})
            else:
                self._tree_coll.update_one({'_id': instance.v_location},
                                       {'$pull': {self.GROUPS: instance.v_name}})
            self._tree_coll.delete_one({'_id': instance.v_full_name})

        else:
            if not instance.v_is_leaf:
                raise ValueError('You can only choose `delete_only` mode for leafs.')

            if isinstance(delete_only, compat.base_type):
                delete_only = [delete_only]

            for delete_item in delete_only:
                if (remove_from_item and
                        hasattr(instance, '__contains__') and
                        hasattr(instance, '__delattr__') and
                            delete_item in instance):
                    delattr(instance, delete_item)

                self._arctic_lib.delete(instance.v_full_name + '.' + delete_item)
                self._tree_coll.update_one({'_id': instance.v_full_name}, {'$pull': {self.DATA:
                                                                                delete_item}})
                # if deletion.deleted_count == 0:
                #     self._logger.warning('Could not delete `%s` from `%s`. Entry not found!' %
                #                          (delete_item, instance.v_full_name))

    @staticmethod
    def _prm_set_recall_object_table(data):
        """Stores original data type to hdf5 node attributes for preserving the data type.

        :param data:

            Data to be stored

        """
        dtypes = {}
        for col in data:
            strtype = type(data[col][0]).__name__
            if not strtype in pypetconstants.PARAMETERTYPEDICT:
                raise TypeError('I do not know how to handle coulumn `%s` its type is `%s`.' %
                            (str(col), repr(type(data))))
            dtypes[col] = strtype
        meta_data = {MongoStorageService.COLL_OBJ_TABLE: dtypes}
        return meta_data

    @staticmethod
    def _prm_recall_obj_table(obj_table, meta_data):
        dtypes = meta_data[MongoStorageService.COLL_OBJ_TABLE]
        res = {}
        for key, val in dtypes.items():
            dtype = pypetconstants.PARAMETERTYPEDICT[val]
            res[key] = [dtype(x) for x in obj_table[key]]
            del obj_table[key]

        return ObjectTable(data=res)

    def _srn_store_single_run(self, traj,
                              recursive=True,
                              store_data=pypetconstants.STORE_DATA,
                              max_depth=None):
        """ Stores a single run instance to disk (only meta data)"""

        if store_data != pypetconstants.STORE_NOTHING:
            self._logger.debug('Storing Data of single run `%s`.' % traj.v_crun)
            if max_depth is None:
                max_depth = float('inf')
            for name_pair in traj._new_nodes:
                _, name = name_pair
                parent_group, child_node = traj._new_nodes[name_pair]
                if not child_node._stored:
                    self._tree_store_sub_branch(parent_group, name,
                                          store_data=store_data,
                                          with_links=True,
                                          recursive=recursive,
                                          max_depth=max_depth - child_node.v_depth)
            for name_pair in traj._new_links:
                _, link = name_pair
                parent_group, _ = traj._new_links[name_pair]
                self._tree_store_sub_branch(parent_group, link,
                                            store_data=store_data,
                                            with_links=True,
                                            recursive=recursive,
                                            max_depth=max_depth - parent_group.v_depth - 1)


    def _tree_load_nodes_dfs(self, parent_traj_node, load_data, with_links, recursive,
                             max_depth, current_depth, trajectory, as_new, child_name):
        """Loads a node from hdf5 file and if desired recursively everything below

        :param parent_traj_node: The parent node whose child should be loaded
        :param load_data: How to load the data
        :param with_links: If links should be loaded
        :param recursive: Whether loading recursively below entry
        :param max_depth: Maximum depth
        :param current_depth: Current depth
        :param trajectory: The trajectory object
        :param as_new: If trajectory is loaded as new
        :param entry: The db entry containing the child to be loaded

        """
        if max_depth is None:
            max_depth = float('inf')

        loading_list = [(parent_traj_node, current_depth, child_name)]

        while loading_list:
            parent_traj_node, current_depth, child_name = loading_list.pop()
            if parent_traj_node.v_is_root:
                full_name = child_name
            else:
                full_name = parent_traj_node.v_full_name + '.' + child_name
            entry = self._tree_coll.find_one({'_id': full_name})

            if entry is None:
                raise pex.DataNotInStorageError('Could not find `%s`!' % str(full_name))

            is_link = self.LINK in entry
            if is_link: # TOFO
                if with_links:
                    # We end up here when auto-loading a soft link
                    self._tree_load_link(parent_traj_node, load_data=load_data, traj=trajectory,
                                         as_new=as_new, entry=entry, link=child_name)
                continue

            is_leaf = self.LEAF in entry
            in_trajectory = child_name in parent_traj_node._children

            if is_leaf:
                # In case we have a leaf node, we need to check if we have to create a new
                # parameter or result

                if in_trajectory:
                    instance = parent_traj_node._children[child_name]
                # Otherwise we need to create a new instance
                else:
                    instance = self._tree_create_leaf(child_name, trajectory, entry)

                    # Add the instance to the trajectory tree
                    parent_traj_node._add_leaf_from_storage(args=(instance,), kwargs={})

                self._prm_load_parameter_or_result(instance, load_data=load_data)
                if as_new:
                    instance._stored = False

            else:
                if in_trajectory:
                    traj_group = parent_traj_node._children[child_name]

                    if load_data == pypetconstants.OVERWRITE_DATA:
                        traj_group.v_annotations.f_empty()
                        traj_group.v_comment = ''
                else:
                    if self.CLASS_NAME in entry:
                        class_name = entry[self.CLASS_NAME]
                        class_constructor = trajectory._create_class(class_name)
                        instance = trajectory._construct_instance(class_constructor, child_name)
                        args = (instance,)
                    else:
                        args = (child_name,)
                    # If the group does not exist create it'
                    traj_group = parent_traj_node._add_group_from_storage(args=args, kwargs={})

                # Load annotations and comment
                self._grp_load_group(traj_group, load_data=load_data, with_links=with_links,
                                     recursive=False, max_depth=max_depth,
                                     _traj=trajectory, _as_new=as_new,
                                     _entry=entry)

                if recursive and current_depth < max_depth:
                    new_depth = current_depth + 1
                    for what in (self.GROUPS, self.LEAVES, self.LINKS):
                        for child in entry.get(what, []):
                            loading_list.append((traj_group, new_depth, child))

    def _tree_create_leaf(self, name, trajectory, entry):
        """ Creates a new pypet leaf instance.

        Returns the leaf and if it is an explored parameter the length of the range.

        """
        class_name = entry[self.CLASS_NAME]
        # Create the instance with the appropriate constructor
        class_constructor = trajectory._create_class(class_name)
        instance = trajectory._construct_instance(class_constructor, name)
        return instance

    def _grp_load_group(self, traj_group, load_data=pypetconstants.LOAD_DATA, with_links=True,
                        recursive=False, max_depth=None,
                        _traj=None, _as_new=False, _entry=None):
        """Loads a group node and potentially everything recursively below"""
        if _entry is None:
            _entry = self._tree_coll.find_one({'_id': traj_group.v_full_name})
            if _entry is None:
                raise pex.DataNotInStorageError('Could not find `%s` in DB!' %
                                                traj_group.v_full_name)

        if recursive:
            parent_traj_node = traj_group.f_get_parent()
            self._tree_load_nodes_dfs(parent_traj_node, load_data=load_data, with_links=with_links,
                                      recursive=recursive, max_depth=max_depth,
                                      current_depth=0,
                                      trajectory=_traj, as_new=_as_new,
                                      child_name=traj_group.v_name)
        else:
            if load_data == pypetconstants.LOAD_NOTHING:
                return

            elif load_data == pypetconstants.OVERWRITE_DATA:
                traj_group.v_annotations.f_empty()
                traj_group.v_comment = ''

            self._all_load_skeleton(traj_group, _entry)
            traj_group._stored = not _as_new

            # Signal completed node loading
            self._node_processing_timer.signal_update()

    def _all_load_skeleton(self, traj_node, entry):
        """Reloads skeleton data of a tree node"""
        if traj_node.v_annotations.f_is_empty():
            self._ann_load_annotations(traj_node, entry)
        if traj_node.v_comment == '':
            comment = entry.get(self.COMMENT, '')
            traj_node.v_comment = comment

    def _ann_load_annotations(self, item_with_annotations, entry):
        """Loads annotations from disk."""
        annotated = self.ANNOTATIONS in entry
        if annotated:
            annotations = item_with_annotations.v_annotations
            # You can only load into non-empty annotations, to prevent overwriting data in RAM
            if not annotations.f_is_empty():
                raise TypeError('Loading into non-empty annotations!')
            anno_dict = pickle.loads(entry[self.ANNOTATIONS])
            annotations.f_set(**anno_dict)

    def _tree_load_link(self, new_traj_node, load_data, traj, as_new, entry, link):
        """ Loads a link

        :param new_traj_node: Node in traj containing link
        :param load_data: How to load data in the linked node
        :param traj: The trajectory
        :param as_new: If data in linked node should be loaded as new
        :param entry: The link db entry

        """
        full_name = entry[self.LINK]

        if (not link in new_traj_node._links or
                    load_data==pypetconstants.OVERWRITE_DATA):

            if not full_name in traj:
                try:
                    self._tree_load_sub_branch(traj, full_name,
                                               load_data=pypetconstants.LOAD_SKELETON,
                                               with_links=False, recursive=False, _trajectory=traj,
                                               _as_new=as_new)
                except pex.DataNotInStorageError:
                    self._logger.error('Linked node not found will remove link `%s` under `%s`!'
                                       % (link, new_traj_node.v_full_name))
                    self._srvc_flush_tree_db()
                    self._tree_coll.update_one({'_id': new_traj_node.v_full_name},
                                           {'$pull': {self.LINKS: link}})
                    self._tree_coll.delete_one(entry)
                    return

            if (load_data == pypetconstants.OVERWRITE_DATA and
                        link in new_traj_node._links):
                new_traj_node.f_remove_link(link)
            if not link in new_traj_node._links:
                new_traj_node._nn_interface._add_generic(new_traj_node,
                                                            type_name=nn.LINK,
                                                            group_type_name=nn.GROUP,
                                                            args=(link,
                                                                  traj.f_get(full_name)),
                                                            kwargs={},
                                                            add_prefix=False,
                                                            check_naming=False)
            else:
                raise RuntimeError('You shall not pass!')


    def _tree_load_sub_branch(self, traj_node, branch_name,
                              load_data=pypetconstants.LOAD_DATA,
                              with_links=True, recursive=False,
                              max_depth=None, _trajectory=None,
                              _as_new=False):
        """Loads data starting from a node along a branch and starts recursively loading
        all data at end of branch.

        :param traj_node: The node from where loading starts

        :param branch_name:

            A branch along which loading progresses. Colon Notation is used:
            'group1.group2.group3' loads 'group1', then 'group2', then 'group3' and then finally
            recursively all children and children's children below 'group3'

        :param load_data:

            How to load the data


        :param with_links:

            If links should be loaded

        :param recursive:

            If loading recursively

        :param max_depth:

            The maximum depth to load the tree

        :param _trajectory:

            The trajectory

        :param _as_new:

            If trajectory is loaded as new

        """
        if load_data == pypetconstants.LOAD_NOTHING:
            return

        if max_depth is None:
            max_depth = float('inf')

        if _trajectory is None:
            _trajectory = traj_node.v_root

        split_names = branch_name.split('.')

        final_group_name = split_names.pop()

        current_depth = 1

        for name in split_names:
            if current_depth > max_depth:
                return
            # First load along the branch
            self._tree_load_nodes_dfs(traj_node, load_data=load_data, with_links=with_links,
                                      recursive=False, max_depth=max_depth,
                                      current_depth=current_depth,
                                      trajectory=_trajectory, as_new=_as_new,
                                      child_name=name)
            current_depth += 1

            traj_node = traj_node._children[name]

        if current_depth <= max_depth:
            # Then load recursively all data in the last group and below
            self._tree_load_nodes_dfs(traj_node, load_data=load_data, with_links=with_links,
                                      recursive=recursive, max_depth=max_depth,
                                      current_depth=current_depth, trajectory=_trajectory,
                                      as_new=_as_new,
                                      child_name=final_group_name)

    def _prm_load_parameter_or_result(self, instance,
                                      load_data=pypetconstants.LOAD_DATA,
                                      load_only=None,
                                      load_except=None,
                                      with_links=False,
                                      recursive=False,
                                      max_depth=None,
                                      _entry=None,):
        """Loads a parameter or result from disk.

        :param instance:

            Empty parameter or result instance

        :param load_data:

            How to load stuff

        :param load_only:

            List of data keys if only parts of a result should be loaded

        :param load_except:

            List of data key that should NOT be loaded.

        :param with_links:

            Placeholder, because leaves have no links

        :param recursive:

            Dummy variable, no-op because leaves have no children

        :param max_depth:

            Dummy variable, no-op because leaves have no children

        :param _entry:

            The corresponding DB entry of the instance

        """
        if load_data == pypetconstants.LOAD_NOTHING:
            return

        if _entry is None:
            _entry = self._tree_coll.find_one({'_id': instance.v_full_name})
            if _entry is None:
                raise pex.DataNotInStorageError('Could not find `%s` '
                                                'in DB' % instance.v_full_name)

        if load_data == pypetconstants.OVERWRITE_DATA:
            if instance.v_is_parameter and instance.v_locked:
                self._logger.debug('Parameter `%s` is locked, I will skip loading.' %
                                     instance.v_full_name)
                return
            instance.f_empty()
            instance.v_annotations.f_empty()
            instance.v_comment = ''

        self._all_load_skeleton(instance, _entry)
        instance._stored = True

        # If load only is just a name and not a list of names, turn it into a 1 element list
        if isinstance(load_only, compat.base_type):
            load_only = [load_only]
        if isinstance(load_except, compat.base_type):
            load_except = [load_except]

        if load_data == pypetconstants.LOAD_SKELETON:
            # We only load skeleton if asked for it and thus only
            # signal completed node loading
            self._node_processing_timer.signal_update()
            return
        elif load_only is not None:
            if load_except is not None:
                raise ValueError('Please use either `load_only` or `load_except` and not '
                             'both at the same time.')
            elif instance.v_is_parameter and instance.v_locked:
                raise pex.ParameterLockedException('Parameter `%s` is locked, '
                                                   'I will skip loading.' %
                                                    instance.v_full_name)
            self._logger.debug('I am in load only mode, I will only load %s.' %
                               str(load_only))
            load_only = set(load_only)
        elif load_except is not None:
            if instance.v_is_parameter and instance.v_locked:
                raise pex.ParameterLockedException('Parameter `%s` is locked, '
                                                   'I will skip loading.' %
                                                    instance.v_full_name)
            self._logger.debug('I am in load except mode, I will load everything except %s.' %
                               str(load_except))
            # We do not want to modify the original list
            load_except = set(load_except)
        elif not instance.f_is_empty():
            # We only load data if the instance is empty or we specified load_only or
            # load_except and thus only
            # signal completed node loading
            self._node_processing_timer.signal_update()
            return

        full_name = instance.v_full_name
        self._logger.debug('Loading data of %s' % full_name)

        load_dict = {}  # Dict that will be used to keep all data for loading the parameter or
        # result

        self._prm_load_into_dict(full_name=full_name,
                                 load_dict=load_dict,
                                 entry=_entry,
                                 instance=instance,
                                 load_only=load_only,
                                 load_except=load_except)

        if load_only is not None:
            # Check if all data in `load_only` was actually found in the hdf5 file
            if len(load_only) > 0:
                self._logger.warning('You marked %s for load only, '
                                     'but I cannot find these for `%s`' %
                                     (str(load_only), full_name))
        elif load_except is not None:
            if len(load_except) > 0:
                self._logger.warning(('You marked `%s` for not loading, but these were not part '
                                      'of `%s` anyway.' % (str(load_except), full_name)))

        # Finally tell the parameter or result to load the data, if there was any ;-)
        if load_dict:
            try:
                instance._load(load_dict)
                if instance.v_is_parameter:
                    # Lock parameter as soon as data is loaded
                    instance.f_lock()
            except:
                self._logger.error(
                    'Error while reconstructing data of leaf `%s`.' % full_name)
                raise

        # Signal completed node loading
        self._node_processing_timer.signal_update()

    def _prm_load_into_dict(self, full_name, load_dict, entry, instance,
                            load_only, load_except):
        """Loads into dictionary"""
        self._srvc_flush_tree_db()
        for load_name in entry.get(self.DATA, []):

            if load_only is not None:
                if load_name not in load_only:
                    continue
                else:
                    load_only.remove(load_name)

            elif load_except is not None:
                if load_name in load_except:
                    load_except.remove(load_name)
                    continue

            full_load_name = full_name + '.' + load_name
            load_elem = self._arctic_lib.read(full_load_name)
            to_load = load_elem.data
            meta_data = load_elem.metadata
            if meta_data is not None:
                if self.BINARY in meta_data:
                    to_load = pickle.loads(to_load)
                elif self.MATRIX in meta_data:
                    to_load = np.matrix(to_load)

            load_dict[load_name] = to_load

    def _trj_load_trajectory(self, traj, as_new, load_parameters, load_derived_parameters,
                             load_results, load_other_data, recursive, max_depth,
                             with_run_information, force):
        """Loads a single trajectory from a given file.


        :param traj: The trajectory

        :param as_new: Whether to load trajectory as new

        :param load_parameters: How to load parameters and config

        :param load_derived_parameters: How to load derived parameters

        :param load_results: How to load results

        :param load_other_data: How to load anything not within the four subbranches

        :param recursive: If data should be loaded recursively

        :param max_depth: Maximum depth of loading

        :param with_run_information:

            If run information should be loaded

        :param force: Force load in case there is a pypet version mismatch

        You can specify how to load the parameters, derived parameters and results
        as follows:

        :const:`pypet.pypetconstants.LOAD_NOTHING`: (0)

            Nothing is loaded

        :const:`pypet.pypetconstants.LOAD_SKELETON`: (1)

            The skeleton including annotations are loaded, i.e. the items are empty.
            Non-empty items in RAM are left untouched.

        :const:`pypet.pypetconstants.LOAD_DATA`: (2)

            The whole data is loaded.
            Only empty or in RAM non-existing instance are filled with the
            data found on disk.

        :const:`pypet.pypetconstants.OVERWRITE_DATA`: (3)

            The whole data is loaded.
            If items that are to be loaded are already in RAM and not empty,
            they are emptied and new data is loaded from disk.


        If `as_new=True` the old trajectory is loaded into the new one, only parameters can be
        loaded. If `as_new=False` the current trajectory is completely replaced by the one
        on disk, i.e. the name from disk, the timestamp, etc. are assigned to `traj`.

        """
        # Some validity checks, if `as_new` is used correctly
        if (as_new and (load_derived_parameters != pypetconstants.LOAD_NOTHING or
                                load_results != pypetconstants.LOAD_NOTHING or
                                load_other_data != pypetconstants.LOAD_NOTHING)):
            raise ValueError('You cannot load a trajectory as new and load the derived '
                             'parameters and results. Only parameters are allowed.')

        if as_new and load_parameters != pypetconstants.LOAD_DATA:
            raise ValueError('You cannot load the trajectory as new and not load the data of '
                             'the parameters.')

        loadconstants = (pypetconstants.LOAD_NOTHING, pypetconstants.LOAD_SKELETON,
                         pypetconstants.LOAD_DATA, pypetconstants.OVERWRITE_DATA)

        if not (load_parameters in loadconstants and load_derived_parameters in loadconstants and
                        load_results in loadconstants and load_other_data in loadconstants):
            raise ValueError('Please give a valid option on how to load data. Options for '
                             '`load_parameter`, `load_derived_parameters`, `load_results`, '
                             'and `load_other_data` are %s. See function documentation for '
                             'the semantics of the values.' % str(loadconstants))

        traj._stored = not as_new

        # Loads meta data like the name, timestamps etc.
        # load_data is only used here to determine how to load the annotations
        load_data = max(load_parameters, load_derived_parameters, load_results, load_other_data)
        self._trj_load_meta_data(traj, load_data, as_new, with_run_information, force)

        if (load_parameters != pypetconstants.LOAD_NOTHING or
                load_derived_parameters != pypetconstants.LOAD_NOTHING or
                load_results != pypetconstants.LOAD_NOTHING or
                load_other_data != pypetconstants.LOAD_NOTHING):
            self._logger.info('Loading trajectory `%s`.' % traj.v_name)
        else:
            self._logger.info('Checked meta data of trajectory `%s`.' % traj.v_name)
            return

        maximum_display_other = 10
        counter = 0

        traj_entry = self._tree_coll.find_one({'_id': ''})
        for kdx, children in enumerate([traj_entry.get(self.GROUPS, []),
                         traj_entry.get(self.LEAVES, []),
                         traj_entry.get(self.LINKS, [])]):
            for child_name in children:

                if kdx == 2:
                    child_name, _ = child_name

                load_subbranch = True
                if child_name == 'config':
                    if as_new:
                        loading = pypetconstants.LOAD_NOTHING
                    else:
                        # If the trajectory is loaded as new, we don't care about old config stuff
                        # and only load the parameters
                        loading = load_parameters
                elif child_name == 'parameters':
                    loading = load_parameters
                elif child_name == 'results':
                    loading = load_results
                elif child_name == 'derived_parameters':
                    loading = load_derived_parameters
                elif child_name == 'overview':
                    continue
                else:
                    loading = load_other_data
                    load_subbranch = False

                if loading == pypetconstants.LOAD_NOTHING:
                    continue

                if load_subbranch:
                    # Load the subbranches recursively
                    self._logger.info('Loading branch `%s` in mode `%s`.' %
                                          (child_name, str(loading)))
                else:
                    if counter < maximum_display_other:
                        self._logger.info(
                            'Loading branch/node `%s` in mode `%s`.' % (child_name, str(loading)))
                    elif counter == maximum_display_other:
                        self._logger.info('To many branchs or nodes at root for display. '
                                          'I will not inform you about loading anymore. '
                                          'Branches are loaded silently '
                                          'in the background. Do not worry, '
                                          'I will not freeze! Pinky promise!!!')
                    counter += 1

                self._tree_load_sub_branch(traj, child_name, load_data=loading, with_links=True,
                                     recursive=recursive,
                                     max_depth=max_depth,
                                     _trajectory=traj, _as_new=as_new)

    def _trj_load_meta_data(self, traj, load_data, as_new, with_run_information, force):
        """Loads meta information about the trajectory

        Checks if the version number does not differ from current pypet version
        Loads, comment, timestamp, name, version from disk in case trajectory is not loaded
        as new. Updates the run information as well.

        """
        info_entry = self._info_coll.find_one({'_id': self.INFO})

        try:
            version = str(info_entry['version'])
        except (IndexError, ValueError) as ke:
            self._logger.error('Could not check version due to: %s' % str(ke))
            version = '`COULD NOT BE LOADED`'

        try:
            python = str(info_entry['python'])
        except (IndexError, ValueError) as ke:
            self._logger.error('Could not check version due to: %s' % str(ke))
            python = '`COULD NOT BE LOADED`'

        self._trj_check_version(version, python, force)

        # Load the skeleton information
        self._grp_load_group(traj, load_data=load_data,
                             with_links=False, recursive=False, _traj=traj,
                             _as_new=as_new)

        if as_new:
            length = int(info_entry['length'])
            for irun in range(length):
                traj._add_run_info(irun)
        else:
            traj._comment = str(info_entry['comment'])
            traj._timestamp = float(info_entry['timestamp'])
            traj._trajectory_timestamp = traj._timestamp
            traj._time = str(info_entry['time'])
            traj._trajectory_time = traj._time
            traj._name = str(info_entry['name'])
            traj._traj_name = traj._name
            traj._version = version
            traj._python = python

            self._traj_name = traj._name

            if with_run_information:
                for entry in self._run_coll.find():
                    name = str(entry['name'])
                    idx = int(entry['idx'])
                    timestamp = float(entry['timestamp'])
                    time_ = str(entry['time'])
                    completed = int(entry['completed'])
                    summary = str(entry['parameter_summary'])
                    hexsha = str(entry['short_environment_hexsha'])

                    runtime = str(entry['runtime'])
                    finish_timestamp = float(entry['finish_timestamp'])

                    info_dict = {'idx': idx,
                                 'timestamp': timestamp,
                                 'finish_timestamp': finish_timestamp,
                                 'runtime': runtime,
                                 'time': time_,
                                 'completed': completed,
                                 'name': name,
                                 'parameter_summary': summary,
                                 'short_environment_hexsha': hexsha}

                    traj._add_run_info(**info_dict)
            else:
                traj._length = self._run_coll.count()

        # Load explorations
        self._trj_load_exploration(traj)

    def _trj_check_version(self, version, python, force):
        """Checks for version mismatch

        Raises a VersionMismatchError if version of loaded trajectory and current pypet version
        do not match. In case of `force=True` error is not raised only a warning is emitted.

        """
        curr_python = compat.python_version_string

        if (version != VERSION or curr_python != python) and not force:
            raise pex.VersionMismatchError('Current pypet version is %s used under python %s '
                                           '  but your trajectory'
                                           ' was created with version %s and python %s.'
                                           ' Use >>force=True<< to perform your load regardless'
                                           ' of version mismatch.' %
                                           (VERSION, curr_python, version, python))
        elif version != VERSION or curr_python != python:
            self._logger.warning('Current pypet version is %s with python %s but your trajectory'
                                 ' was created with version %s under python %s.'
                                 ' Yet, you enforced the load, so I will'
                                 ' handle the trajectory despite the'
                                 ' version mismatch.' %
                                 (VERSION, curr_python, version, python))