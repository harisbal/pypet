import pymongo
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
import pypet.utils.ptcompat as ptcompat
import pypet.pypetconstants as pypetconstants
import pypet.pypetexceptions as pex
from pypet._version import __version__ as VERSION
from pypet.parameter import ObjectTable, Parameter
import pypet.naturalnaming as nn
from pypet.pypetlogging import HasLogger, DisableAllLogging
from pypet.storageservice import NodeProcessingTimer


MAX_NAME_LENGTH = 50
TREE_COLL = 'tree'
INFO_COLL = 'info'
RUN_COLL = 'runs'
DATA_COLL = 'data'


class MongoStorageService(StorageService, HasLogger):
    def __init__(self, mongo_host='localhost', mongo_port=27017):
        self._set_logger()
        if isinstance(mongo_host, compat.base_type):
            self._mongo_host = mongo_host
            self._mongo_port = mongo_port
            self._client = pymongo.MongoClient(mongo_host, mongo_port, connect=False)
        else:
            self._client = mongo_host
            self._mongo_host = None
            self._mongo_port = None
        self._arctic = arctic.Arctic(self._client)
        self._info_coll = None
        self._tree_coll = None
        self._run_coll = None
        self._arctic_lib = None
        self._mode = None
        self._traj_name = None
        self._traj_stump = None
        self._db_name = None
        self._db = None
        self._is_open = False

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
    GROUPS_AND_LEAVES = 'groups_leaves'
    '''Children entry'''
    LINKS = 'links'
    '''Links entry'''

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

                :param store_flags: Flags describing how to store data.

                        :const:`~pypet.HDF5StorageService.ARRAY` ('ARRAY')

                            Store stuff as array

                        :const:`~pypet.HDF5StorageService.CARRAY` ('CARRAY')

                            Store stuff as carray

                        :const:`~pypet.HDF5StorageService.TABLE` ('TABLE')

                            Store stuff as pytable

                        :const:`~pypet.HDF5StorageService.DICT` ('DICT')

                            Store stuff as pytable but reconstructs it later as dictionary
                            on loading

                        :const:`~pypet.HDF%StorageService.FRAME` ('FRAME')

                            Store stuff as pandas data frame

                    Storage flags can also be provided by the parameters and results themselves
                    if they implement a function '_store_flags' that returns a dictionary
                    with the names of the data to store as keys and the flags as values.

                    If no storage flags are provided, they are automatically inferred from the
                    data. See :const:`pypet.HDF5StorageService.TYPE_FLAG_MAPPING` for the mapping
                    from type to flag.

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
        try:

            self._srvc_opening_routine('a', msg, kwargs)

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


    def load(self, msg, stuff_to_load, *args, **kwargs):
        """Loads a particular item from disk.

        The storage service always accepts these parameters:

        :param trajectory_name: Name of current trajectory and name of top node in hdf5 file.

        :param trajectory_index:

            If no `trajectory_name` is provided, you can specify an integer index.
            The trajectory at the index position in the hdf5 file is considered to loaded.
            Negative indices are also possible for reverse indexing.

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
        try:
            self._srvc_opening_routine('r', msg, kwargs)

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

    def _trj_store_trajectory(self, traj, only_init=False,
                          store_data=pypetconstants.STORE_DATA,
                          max_depth=None):
        """ Stores a trajectory to an hdf5 file

        Stores all groups, parameters and results

        """
        if not only_init:
            self._logger.info('Start storing Trajectory `%s`.' % self._trajectory_name)
        else:
            self._logger.info('Initialising storage or updating meta data of Trajectory `%s`.' %
                              self._trajectory_name)
            store_data = pypetconstants.STORE_NOTHING

        # In case we accidentally chose a trajectory name that already exist
        # We do not want to mess up the stored trajectory but raise an Error
        if not traj._stored and self._info_coll is not None:
            raise RuntimeError('You want to store a completely new trajectory with name'
                               ' `%s` but this trajectory is already found in file `%s`' %
                               (traj.v_name, self._filename))

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

            self._logger.info('Finished storing Trajectory `%s`.' % self._trajectory_name)
        else:
            self._logger.info('Finished init or meta data update for `%s`.' %
                              self._traj_name)
        traj._stored = True

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
                    if self._db_name in self._client.database_names():
                        # If we want to store individual items we we have to check if the
                        # trajectory has been stored before
                        if not msg == pypetconstants.TRAJECTORY:
                            raise ValueError('Your trajectory cannot be found in the database, '
                                             'please use >>traj.f_store()<< '
                                             'before storing anything else.')
                        else:
                            self._arctic.initialize_library(self._db_name + '.' +
                                                            self._traj_stump + DATA_COLL)
                else:
                    raise ValueError('I don`t know which trajectory to load')
                self._logger.debug('Opening MongoDB `%s` in mode `a` with trajectory `%s`' %
                                   (self._db_name, self._traj_name))

            elif mode == 'r':

                if self._trajectory_name is not None:
                    # Otherwise pick the trajectory group by name
                    if not '/' + self._trajectory_name in self._hdf5file:
                        raise ValueError('File %s does not contain trajectory %s.'
                                         % (self._filename, self._trajectory_name))

                else:
                    raise ValueError('Please specify a name of a trajectory to load, '
                                     'otherwise I cannot open the databse.')

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

    def _set_db_name(self, db_name):
        if db_name.startswith(self._arctic.DB_PREFIX):
            self._db_name = self._db_name
        else:
            self._db_name = self._arctic.DB_PREFIX + '_' + db_name

    def _srvc_extract_file_information(self, kwargs):
        """Extracts file information from kwargs.

        Note that `kwargs` is not passed as `**kwargs` in order to also
        `pop` the elements on the level of the function calling `_srvc_extract_file_information`.

        """
        if 'db_name' in kwargs:
            self._set_db_name(kwargs.pop('db_name'))

        if 'trajectory_name' in kwargs:
            traj_name = kwargs.pop('trajectory_name')
            if self._traj_name is not None and (traj_name != self._traj_name):
                self._srvc_closing_routine()
            self._traj_name = traj_name
            self._traj_stump = self._traj_name.lower()[:MAX_NAME_LENGTH]
            if self._db_name is None:
                self._db_name._set_db_name(self._traj_stump)

    def _srvc_closing_routine(self):
        """Routine to close an hdf5 file

        The file is closed only when `closing=True`. `closing=True` means that
        the file was opened in the current highest recursion level. This prevents re-opening
        and closing of the file if `store` or `load` are called recursively.

        """
        self._client.close()
        self._info_coll = None
        self._tree_coll = None
        self._arctic_lib = None
        self._mode = None
        self._traj_name = None
        self._db_name = None
        self._db = None
        self._traj_stump = None
        self._is_open = False

    @staticmethod
    def _srvc_set_on_insert(coll, entry, how='$setOnInsert'):
        coll.update_one({'_id': entry['_id_']}, {how:  entry}, upsert=True)

    def _trj_store_meta_data(self, traj):
        """ Stores general information about the trajectory in the hdf5file.

        The `info` table will contain the name of the trajectory, it's timestamp, a comment,
        the length (aka the number of single runs), and the current version number of pypet.

        Also prepares the desired overview tables and fills the `run` table with dummies.

        """

        # Description of the `info` table

        descriptiondict = {'_id' : 'info',
                           'name': traj.v_name,
                           'time': traj.v_time,
                           'timestamp': traj.v_timestamp,
                           'comment': traj.v_comment,
                           'length': len(traj),
                           'version': traj.v_version,
                           'python': traj.v_python}
        # 'loaded_from' : pt.StringCol(pypetconstants.HDF5_STRCOL_MAX_LOCATION_LENGTH)}
        self._info_coll.update_one({'_id': 'info'}, {'$set': descriptiondict}, upsert=True)

        # # Description of the `run` table
        # rundescription_dict = {'name': pt.StringCol(pypetconstants.HDF5_STRCOL_MAX_NAME_LENGTH,
        #                                             pos=1),
        #                        'time': pt.StringCol(len(traj.v_time), pos=2),
        #                        'timestamp': pt.FloatCol(pos=3),
        #                        'idx': pt.IntCol(pos=0),
        #                        'completed': pt.IntCol(pos=8),
        #                        'parameter_summary': pt.StringCol(
        #                            pypetconstants.HDF5_STRCOL_MAX_COMMENT_LENGTH,
        #                            pos=6),
        #                        'short_environment_hexsha': pt.StringCol(7, pos=7),
        #                        'finish_timestamp': pt.FloatCol(pos=4),
        #                        'runtime': pt.StringCol(
        #                            pypetconstants.HDF5_STRCOL_MAX_RUNTIME_LENGTH,
        #                            pos=5)}

        # Store the list of explored paramters
        self._trj_store_explorations(traj)

    def _trj_load_exploration(self, traj):
        """Recalls names of all explored parameters"""
        explorations_entry = self._info_coll.find_one({'_id': 'explorations'})
        explorations_list = explorations_entry['explorations']
        for param_name in explorations_list:
            if param_name not in traj._explored_parameters:
                traj._explored_parameters[param_name] = None

    def _trj_store_explorations(self, traj):
        """Stores a all explored parameter names for internal recall"""
        nexplored = len(traj._explored_parameters)
        if nexplored > 0:
            if hasattr(self._overview_group, 'explorations'):
                explorations_entry = self._info_coll.find_one({'_id': 'explorations'})
                if explorations_entry is not None:
                    explored_list = explorations_entry['explorations']
                    if len(explored_list) != nexplored:
                        self._info_coll.delete_one({'_id': 'explorations'})
            explored_list = compat.listkeys(traj._explored_parameters)
            self._srvc_set_on_insert(self._info_coll, {'_id': 'explorations',
                                                       'explorations': explored_list})

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

            hdf5_group = getattr(hdf5_group, name)

        # Store final group and recursively everything below it
        if current_depth <= max_depth:
            self._tree_store_nodes_dfs(traj_node, leaf_name, store_data=store_data,
                               with_links=with_links, recursive=recursive,
                               max_depth=max_depth, current_depth=current_depth)

    def _grp_store_group(self, traj_group, store_data=pypetconstants.STORE_DATA,
                         with_links=True, recursive=False, max_depth=None):
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

            data = {'_id': traj_group.v_full_name,
                    self.CLASS_NAME: traj_group.f_get_class_name()}

            if traj_group.v_comment != '':
                option = '$set'
                data[self.COMMENT] = traj_group.v_comment

            if not traj_group.v_annotations.f_is_empty():
                option = '$set'
                annotations = traj_group.v_annotations.f_to_dict()
                entry = self._tree_coll.find_one({'_id': traj_group.v_full_name})
                if entry is None:
                    data[self.ANNOTATIONS] = annotations
                else:
                    if self.ANNOTATIONS in entry:
                        old_annotations = entry[self.ANNOTATIONS]
                    else:
                        old_annotations = {}
                    for key in annotations:
                        if key not in old_annotations:
                            old_annotations[key] = annotations[key]
                    data[self.ANNOTATIONS] = old_annotations

            self._srvc_set_on_insert(data, how=option)
            traj_group._stored = True

            # Signal completed node loading
            self._node_processing_timer.signal_update()

        if recursive:
            parent_traj_group = traj_group.f_get_parent()

            self._tree_store_nodes_dfs(parent_traj_group, traj_group.v_name, store_data=store_data,
                                       with_links=with_links, recursive=recursive,
                                       max_depth=max_depth, current_depth=0)

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
                    self._tree_coll.update_one({'_id': parent_traj_node.v_fullname},
                                               {'addToSet'}) #TODO HERE!!!
                continue

            traj_node = parent_traj_node._children[name]

            if traj_node.v_is_leaf:
                self._prm_store_parameter_or_result(traj_node, store_data=store_data)

            else:
                self._grp_store_group(traj_node, store_data=store_data, with_links=with_links,
                                      recursive=False, max_depth=max_depth)

                if recursive and current_depth < max_depth:
                    for child in compat.iterkeys(traj_node._children):
                        store_list.append((traj_node, child, current_depth + 1))

