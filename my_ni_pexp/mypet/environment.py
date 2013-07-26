'''
Created on 03.06.2013

@author: robert
'''
from Cython.Runtime.refnanny import loglevel

from mypet.mplogging import StreamToLogger
from mypet.trajectory import Trajectory, SingleRun
import os
import sys
import logging
import time
import datetime
import multiprocessing as multip
import traceback
from mypet.storageservice import HDF5StorageService


def _single_run(args):

    try:
        traj=args[0] 
        logpath=args[1] 
        lock=args[2] 
        runfunc=args[3] 
        total_runs = args[4]
        runparams = args[5]
        kwrunparams = args[6]
    
        assert isinstance(traj, SingleRun)
        root = logging.getLogger()
        n = traj.get_n()
        #If the logger has no handler, add one:
        #print root.handlers
        if len(root.handlers)<3:
            
            #print 'do i come here?'

            filename = 'process%03d.txt' % n
            h=logging.FileHandler(filename=logpath+'/'+filename)
            f = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s %(message)s')
            h.setFormatter(f)
            root.addHandler(h)
    
            #Redirect standard out and error to the file
            outstl = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
            sys.stdout = outstl

            errstl = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)
            sys.stderr = errstl
            
            
        outstl = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
        sys.stdout = outstl

        
    
        root.info('\n--------------------------------\n Starting single run #%d of %d \n--------------------------------' % (n,total_runs))
        result =runfunc(traj,*runparams,**kwrunparams)
        root.info('Storing Parameters')
        traj.store(lock)
        root.info('Storing Parameters Finished')
        root.info('\n--------------------------------\n Finished single run #%d of %d \n--------------------------------' % (n,total_runs))
        return result

    except:
        errstr = "\n\n########## ERROR ##########\n"+"".join(traceback.format_exception(*sys.exc_info()))+"\n"
        logging.getLogger('STDERR').error(errstr)
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
 

    
    
class Environment(object):
    ''' The environment to run a parameter exploration.
    '''
    
    def __init__(self, trajectoryname, filename, filetitle='Experiment', dynamicly_imported_classes=[], logfolder='../log/'):
        
        #Acquiring the current time
        init_time = time.time()
        thetime = datetime.datetime.fromtimestamp(init_time).strftime('%Y_%m_%d_%Hh%Mm%Ss');
        
        # Logging
        self._logpath = os.path.join(logfolder,trajectoryname+'_'+thetime)
        
        if not os.path.isdir(self._logpath):
            os.makedirs(self._logpath)
        
        
        f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
        h=logging.FileHandler(filename=self._logpath+'/main.txt')
        #sh = logging.StreamHandler(sys.stdout)
        root = logging.getLogger()
        root.addHandler(h)



        for handler in root.handlers:
            handler.setFormatter(f)
        self._logger = logging.getLogger('mypet.environment.Environment')

        # Creating the Trajectory

        self._traj = Trajectory(trajectoryname, dynamicly_imported_classes,init_time)

        storage_service = HDF5StorageService(filename, filetitle,self._traj.get_name() )
        self._traj.set_storage_service(storage_service)

        # Adding some default configuration
        self._traj.ac('logpath', self._logpath).lock()
        self._traj.ac('ncores',1)
        self._traj.ac('multiproc',False)


        self._logger.debug('Environment initialized.')

        
    def get_trajectory(self):
        return self._traj
    


    def run(self, runfunc, *runparams,**kwrunparams):
        

        #Prepares the trajecotry for running

        
        multiproc = self._traj.get('Config.multiproc').return_default()

        self._traj.prepare_experiment()
        if multiproc:
            
            lock = multip.Manager().Lock()
           
            ncores = multiproc = self._traj.get('Config.ncores').return_default()
            
            mpool = multip.Pool(ncores)
        
            print '----------------------------------------'
            print 'Starting run in parallel with %d cores.' % ncores
            print '----------------------------------------'
            
            iterator = ((self._traj.make_single_run(n),self._logpath,lock,runfunc,len(self._traj),runparams,kwrunparams) for n in xrange(len(self._traj)))
        
            results = mpool.imap(_single_run,iterator)
            
            mpool.close()
            mpool.join()
            print '----------------------------------------'
            print 'Finished run in parallel with %d cores.' % ncores
            print '----------------------------------------'
            
            return results
        else:
            
            results = [_single_run((self._traj.make_single_run(n),self._logpath,None,runfunc,len(self._traj),runparams,kwrunparams)) for n in xrange(len(self._traj))]
            return results
                
        
        
        