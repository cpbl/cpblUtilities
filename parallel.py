#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Following is for running functions in parallel.
We want to 
 - monitor progress of a bunch of function calls, running in parallel
 - capture the output of each function call. This is a problem because Queues normally break if they get too full. Thus we regularly empty them.
 - Close the queues as functions finish. This is key because otherwise the OS shuts us down for using too many open files.

"""
__author__ = "Chris Barrington-Leigh"
class pWrapper(): # Maybe if I enclose this in a class, the Garbage Collection will work better?
    def __init__(self,thefunc,theArgs=None,thekwargs=None,delay=None,name=None):
        self.callfunc=thefunc
        self.callargs=theArgs
        self.callkwargs=thekwargs
        self.calldelay=delay  # Or should this be dealt with elsewhere?
        self.name=name  # Or should this be dealt with elsewhere?
        self.gotQueue=None
        self.started=False
        self.running=False
        self.finished=False
        self.exitcode='dns'
        self.is_alive='dns' # For internal use only. Present "running"
        self.queue=0
    @staticmethod
    def functionWrapper(fff,que,theArgs=None,thekwargs=None,delay=None): #add a argument to function for assigning a queue
        if delay:
            from time import sleep
            sleep(delay)
        funcName='(built-in function)' if not hasattr(fff,'func_name') else fff.func_name
        #print 'MULTIPROCESSING: Launching %s in parallel '%funcName
        theArgs=theArgs if theArgs is not None else []
        kwargs=thekwargs if thekwargs is not None  else {}
             
        returnVal=que.put(fff(*theArgs,**kwargs))
        print 'MULTIPROCESSING: Finished %s in parallel! '%funcName
        return(returnVal) #this hsould be 0.
    def start(self):
        import multiprocessing as mp
        assert self.started==False
        self.queue=mp.Queue()

        self.thejob=mp.Process(target=self.functionWrapper, args=[self.callfunc, self.queue,self.callargs,self.callkwargs],)
        #if self.calldelay:
        #    from time import sleep
        #    sleep(self.calldelay)
        self.thejob.start()
        funcName='(built-in function)' if not hasattr(self.callfunc,'func_name') else self.callfunc.func_name
        print('MULTIPROCESSING: Launching %s in parallel %s'%(funcName,self.name))
        self.started=True
        self.running=True
    def status(self):
        if self.started is False:
            return('dns')
        if self.finished:
            return({0:'0',1:'failed'}.get(self.exitcode,self.exitcode))
        assert self.running
        self.is_alive=self.thejob.is_alive()
        cleanup=  self.is_alive not in ['dns',1]
        assert self.running
        # Update/empty queue
        if not self.queue.empty():
            if self.gotQueue is None:
                self.gotQueue=self.queue.get()
            else:
                self.gotQueue+=self.queue.get()
        # Terminate the job, close the queue, try to initiate Garbage Collecting in order to avoid "Too many open files"
        if cleanup: # The following is intended to get arround OSError: [Errno 24] Too many open files.  But it does not. What more can I do to garbage clean the completed queues and jobs?
            self.cleanup()
        return('running')
    def cleanup(self):
        self.exitcode=self.thejob.exitcode
        self.thejob.join()
        self.thejob.terminate()
        self.queue.close()
        self.thejob=None
        #del job
        self.queue=None
        self.finished=True
        self.running=False
    def queuestatus(self):
        if self.queue in [0]:
            return('dns') # Did not start yet
        if self.queue is None:
            return('') # Closed
        return('empty'*self.queue.empty()  + 'full'*self.queue.full() )


###########################################################################################
###
def  runFunctionsInParallel(listOf_FuncAndArgLists,kwargs=None,names=None,parallel=None,offsetsSeconds=None,expectNonzeroExit=False,maxAtOnce=None,showFinished=20,  maxFilesAtOnce=None):
    ###
    #######################################################################################
    """
    Chris Barrington-Leigh, 2011-2014+

    Take a list of lists like [function, args, kwargs]. Run those functions in parallel, wait for them all to finish, and return a tuple of (return codes, return values), each a list in order.

This implements a piecemeal collection of return values from the functions, since otherwise the pipes get stuck (!) and the processes cannot finish.

listOf_FuncAndArgLists: a list of lists like [function, args, kwargs], specifying the set of functions to be launched in parallel.  If an element is just a function, rather than a list, then it is assumed to have no arguments. ie args and kwargs can be filled in where both, or kwargs, are missing.

names: an optional list of names for the processes.

offsetsSeconds: delay some functions' start times

expectNonzeroExit: Normally, we should not proceed if any function exists with a failed exit code? So the functions that get passed here should return nonzero only if they fail.

parallel: If only one function is given or if parallel is False, this will run the functions in serial.

maxAtOnce: if nonzero, it will queue jobs, adding more only when some are finished

maxFilesAtOnce: Set this below your user limit for how many files you can have open at once. Jobs and Queues are cleaned up as we go, but may lag the behind the finishing of jobs. So setting this as high as possible will increase the speed of the batch.
    If you leave this as None, we ignore the constraint (since there seems still to be some way that queues can escape cleanup).

showFinished= (int) . : Specifies the maximum number of successfully finished jobs to show in reports (before the last, which should always show them all).

2013 Feb: when there's a change in the statuses, update again immediately rather than sleeping.

2013July: You can now pass os.system or etc to this as the function, with no need for a wrapper: I made use of hasattr(builtinfunction,'func_name') to check for a name.

Bug:
 - "too many files open" if more than ~1000 jobs are given (or whatever is set as your user-level limit for open files).  Function should be rewritten so that the Queues are only created when the instance is being launched. Right now, all queues are created at once at the beginning. [Done: 2015Nov. In testing.]

    """
    import numpy as np
    from os import nice
    import gc # Effort to close files (queues) when done... 

    if parallel is None or parallel is True: # Use parallel only when we have many processing cores (well, here, more than 8)
        from multiprocessing import  cpu_count
        parallel=cpu_count() >2

    if not listOf_FuncAndArgLists:
        return([]) # list of functions to run was empty.

    if offsetsSeconds is None:
        offsetsSeconds=0

    # If no argument list is given, make one:
    listOf_FuncAndArgLists=[faal if isinstance(faal,list) else [faal,[],{}] for faal in listOf_FuncAndArgLists]
    listOf_FuncAndArgLists=[faal+[{}] if len(faal)==2 else faal for faal in listOf_FuncAndArgLists]
    listOf_FuncAndArgLists=[faal+[[],{}] if len(faal)==1 else faal for faal in listOf_FuncAndArgLists]
    kwargs=kwargs if kwargs else [faal[2] for faal in listOf_FuncAndArgLists]

    if len(listOf_FuncAndArgLists)>1000:
        pass
        #raise (""" Sorry, until the bug above is solved, you must limit this to ~1000 processes in the list""")
    
    if len(listOf_FuncAndArgLists)==1:
        parallel=False

    if parallel is False:
        print('++++++++++++++++++++++  DOING FUNCTIONS SEQUENTIALLY ---------------- (parallel=False in runFunctionsInParallel)')
        returnVals=[fffargs[0](*(fffargs[1]),**(fffargs[2]))  for iffargs,fffargs in enumerate(listOf_FuncAndArgLists)]
        assert expectNonzeroExit or not any(returnVals)
        return(returnVals)


    if names is None:
        names=[None for fff in listOf_FuncAndArgLists]
    names=[names[iii] if names[iii] is not None else fff[0].func_name for iii,fff in enumerate(listOf_FuncAndArgLists)]
        
    assert len(names)==len(listOf_FuncAndArgLists)

    def reportStatus(status, exitcodes,names,showmax,showsuccessful=np.inf):#jobs):
        ishowable=range(min(len(status), showmax))
        istarted=[ii for ii in range(len(status)) if  status[ii] not in ['dns']]
        isuccess=[ii for ii in ishowable if status[ii] in ['success',0,'0']]
        irunning=[ii for ii in range(len(status)) if  status[ii] in ['running']]
        earliestSuccess= -1 if len(isuccess)<showsuccessful else isuccess[::-1][showsuccessful-1]
        if 0:
            print(showmax,showsuccessful,earliestSuccess)
            print(len(isuccess)-showsuccessful)
        tableFormatString='%'+str(max([len(name) for name in names]))+'s:\t%10s\t%10s\t%s()'
        print('\n'+'-'*75+'\n'+ tableFormatString%('Job','Status','Queue','Func',)+ '\n'+'-'*75)
        # Check that we aren't going to show more *successfully finished* jobs than we're allowed: Find index of nth-last successful one. That is, if the limit binds, we should show the latest N=showsuccessful ones only.
        print('\n'.join([tableFormatString%(names[ii],status[ii], queuestatus[ii], '(built-in function)' if not hasattr(listOf_FuncAndArgLists[ii][0],'func_name') else listOf_FuncAndArgLists[ii][0].func_name) for ii in ishowable if status[ii] not in ['success',0,'0'] or ii>=earliestSuccess  ]))
        if len(isuccess)>showsuccessful: # We don't hide failed jobs, but we do sometimes skip older successful jobs
            print('%d job%s running. %d other jobs finished successfully.'%(len(irunning), 's'*(len(irunning)!=1), len(isuccess)-showsuccessful))
        else:
            print('%d job%s running.' % (len(irunning),'s'*(len(irunning)!=1)))
        if len(status)>len(istarted):
            print('%d more jobs waiting for their turn to start...'%(len(status)-len(istarted))) ##len(sjobs)-len(djobs)))
        #print('%d open queues...'%len(queues))
        print('-'*75+'\n')
        #return([exitcode(job) for ii,job in enumerate(sjobs)])

    def emptyQueues():#jobs,queues,gotQueues):
        for ii,job in enumerate(jobs):
            if ii not in queues or not isinstance(queues[ii],mp.queues.Queue):
                continue
            cleanup= exitcode(job)==0
            
            if not queues[ii].empty():
                if ii in gotQueues:
                    gotQueues[ii]+=queues[ii].get()
                else:
                    gotQueues[ii]=queues[ii].get()
            if cleanup: # The following is intended to get arround OSError: [Errno 24] Too many open files.  But it does not. What more can I do to garbage clean the completed queues and jobs?
                job.join()
                job.terminate()
                queues[ii].close()
                """
        print('Joined job %d'%ii)
        job.terminate()
        print('Terminated job %d'%ii)
        queues[ii].close()
                """
                job=None
                #del job
                queues[ii]=None
                #del queues[ii] # This seems key. Before, when I kept queues in a list, deleting the item wasn't good enough.
                #print('                       Cleaning up/closing queue for job %d'%ii)
                

    if maxFilesAtOnce is None:
        pass # maxFilesAtOnce =max(10*maxAtOnce,100) 
    if maxAtOnce is None:
        maxAtOnce=max(1,cpu_count()-1)  #np.inf
    else:
        maxAtOnce=max(min(cpu_count()-2,maxAtOnce),1)  #np.inf
    # For initial set of launched processes, stagger them with a spacing of the offsetSeconds.
    delays=list((  (np.arange(len(listOf_FuncAndArgLists))-1) * ( np.arange(len(listOf_FuncAndArgLists))<maxAtOnce  ) + 1 )* offsetsSeconds)
    nice(10) # Add 10 to the niceness of this process (POSIX only)
    jobs = [pWrapper(funcArgs[0],funcArgs[1],funcArgs[2],delays[iii],names[iii]) for iii,funcArgs in enumerate(listOf_FuncAndArgLists)]
    # Let's never create a loop variable which takes on the value of an element of the above list. Always instead dereference the list using an index.  So no local variables take on the value of a job. (In addition, the job class is supposed to clean itself up when a job is done running).

    istart=maxAtOnce if maxAtOnce<len(jobs) else len(jobs)
    status=[None for                 ii,fff in enumerate(listOf_FuncAndArgLists)]
    exitcodes=[None for                 ii,fff in enumerate(listOf_FuncAndArgLists)]
    queuestatus=[None for                 ii,fff in enumerate(listOf_FuncAndArgLists)]
    for ijob in range(istart):#enumerate(jobs[:istart]):
        jobs[ijob].start() # Launch them all

    import time
    from math import sqrt
    timeElapsed=0
    ##n=1 # obselete
    gotQueues=dict()

    def updateStatus():
        for ii in range(len(jobs)):
            if status[ii] not in ['failed','success','0',0,1,'1']: 
                status[ii]=jobs[ii].status()
                exitcodes[ii]=jobs[ii].exitcode
                queuestatus[ii]=jobs[ii].queuestatus()
            if status[ii] not in ['dns','running',None] and ii not in gotQueues:
                    gotQueues[ii]=jobs[ii].gotQueue
                    #jobs[ii].destroy()
                    jobs[ii]=None
                    gc.collect()
    updateStatus()
    reportStatus(status, exitcodes,names,istart,showFinished) # This is not necessary; we can leave it to the first loop, below, to report. But for debug, this shows the initial batch.

    """ Now, wait for all the jobs to finish.  Allow for everything to finish quickly, at the beginning. 
    """
    while any([status[ijj]=='running' for  ijj in range(len(jobs))]) or istart<len(jobs):
        sleepTime=5*(timeElapsed>2) + np.log(1.5+timeElapsed)/2 
        #print('DEBUG: ',n,newStatus,lastStatus,sleepTime)
        if timeElapsed>0:
            time.sleep(1+sleepTime) # Wait a while before next update. Slow down updates for really long runs.
        timeElapsed+=sleepTime
        # Add any extra jobs needed to reach the maximum allowed:
        newjobs=0
        while istart<len(jobs) and sum([status[ijj] in ['running'] for ijj in range(len(jobs))]) < maxAtOnce:#  and (maxFilesAtOnce is None or len([qq for qq in queues if qq is not None])< maxFilesAtOnce):
            #print len(queues), maxAtOnce
            #print [is_alive(jj) for jj in jobs]

            ##print istart, len(jobs), sum([jj.is_alive() for jj in jobs]),  maxAtOnce
            jobs[istart].start() #=jstart(jobs[istart])
            newjobs+=1
            updateStatus()
            if newjobs>=maxAtOnce:
                reportStatus(status, exitcodes,names,istart,showFinished) #istart)#jobs)
                newjobs=0
            istart+=1
            timeElapse=.01

        updateStatus()
        reportStatus(status, exitcodes,names,istart,showFinished) #istart)#jobs)
        #emptyQueues()#jobs,queues,gotQueues)

    #for job in jobs: job.join() # Wait for them all to finish... Hm, Is this needed to get at the Queues?

    # And now, collect any remaining buffered outputs (queues):
    #emptyQueues()
    #for ii,job in enumerate(jobs):
    #    if ii not in gotQueues:
    #        gotQueues[ii]=None

    # Give final report of exit statuses?
    updateStatus()
    reportStatus(status, exitcodes,names,np.inf)
    if any(exitcodes):
        print('MULTIPROCESSING: Parallel processing batch set did not ALL succeed successfully ('+' '.join(names)+')')
        assert expectNonzeroExit  # one of the functions you called failed.
        return(False)
    else:
        print('MULTIPROCESSING: Apparent success of all functions ('+' '.join(names)+')')
    return(exitcodes,[gotQueues[ii] for ii in range(len(jobs))])






def breaktest(): # The following demonstrates how to clean up jobs and queues (the queues was key) to avoid the OSError of too many files open. But why does this not work, above? Because there's still a pointer in the list of queues? No, 
    def dummy(inv,que):
        que.put(inv)
        return(0)
    from multiprocessing import Process, Queue, cpu_count
    nTest=1800
    queues=[None for ii in range(nTest)]
    jobs=[None for ii in range(nTest)]#[Process(target=dummy, args=[ii,queues[ii]]) for ii in range(nTest)]
    #for ii,job in enumerate(jobs):
    for ii in range(nTest):#,job in enumerate(jobs):
        queues[ii]=Queue()
        job=Process(target=dummy, args=[ii,queues[ii]])
        job.start()
        print('Started job %d'%ii)
        job.join()
        print('Joined job %d'%ii)
        job.terminate()
        print('Terminated job %d'%ii)
        queues[ii].close()
        queues[ii]=None #  This line does it!
        


def testParallel():
    import numpy as np

    # Demo longer jobs, since other demos' jobs finish too quickly on some platforms
    def doodle4():
        ll=np.random.randint(7)+3
        i=0
        while i<10**ll:
            i=i+1
        return(i)
    nTest=20
    runFunctionsInParallel([doodle4 for ii in range(nTest)],names=[str(ii) for ii in range(nTest)], offsetsSeconds=None, maxAtOnce=10, showFinished=5)


    # Test use of kwargs
    def doodle1(jj, a=None, b=None):
        i=0 + len(str(a))+len(str(b))
        while i<1e2:
            i=i+1
        return(jj)
    nTest=10
    runFunctionsInParallel([[doodle1,[ii],{'a':5,'b':10}] for ii in range(nTest)],names=[str(ii) for ii in range(nTest)], offsetsSeconds=0.2, maxAtOnce=40, parallel=True, expectNonzeroExit=True)

    # Demo simpler use, function takes no arguments
    def doodle2():
        i=0
        while i<1e9:
            i=i+1
        return(i)
    nTest=100
    runFunctionsInParallel([doodle2 for ii in range(nTest)],names=[str(ii) for ii in range(nTest)], offsetsSeconds=0.2, maxAtOnce=10, showFinished=5)

    
    # Test use of large number of jobs, enough to make some systems get upset without our countermeasures
    def doodle3(jj, a=None, b=None):
        i=0
        while 0:#i<1e2:
            i=i+1
        return(jj)
    nTest=2700
    runFunctionsInParallel([[doodle3,[ii],{'a':5,'b':10}] for ii in range(nTest)],names=[str(ii) for ii in range(nTest)], offsetsSeconds=0.2, maxAtOnce=40, parallel=True, expectNonzeroExit=True)



