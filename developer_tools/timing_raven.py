import sys
import re
import os
import numpy as np

class Job:
  def __init__(self):
    self.step         = None
    self.jid          = None # prefix, identifier, prefix
    self.cid          = None # Step counter number
    self.tot_elapsed  = None
    self.status       = None
    # step
    self.step_submit  = None
    self.step_collect = None
    self.step_finish  = None
    # job handler
    self.jh_onQ       = None
    self.jh_start     = None
    self.jh_finish    = None
    # model
    self.m_start      = None
    self.m_run        = None
    self.m_finish     = None
    self.m_return     = None

  def finalize(self):
    self.tot_elapsed = self.step_collect - self.step_submit

  def printme(self):
    print 'Job:',self
    print '  JID :',self.jid
    print '  CID :',self.cid
    print '  Step:',self.step
    print '    Submit :',self.step_submit
    print '    Collect:',self.step_collect
    print '    Elapsed:',self.step_collect - self.step_submit


#### READER ####
class TimingHandler:
  def __init__(self,fname,skipClean=False):
    self.fname        = fname
    self.working_jobs = []
    self.done_jobs    = []
    self.failed_jobs  = []
    if not skipClean:
      self.cleanLogFile()
    self.loadLog()

  #### "main" ####
  def loadLog(self):
    print 'Loading jobs from log file ...'
    with open(self.fname,'r') as logfile:
      for l,line in enumerate(logfile):
        if 'TIMING' not in line or line.strip()=='':
          continue
        # timing line template: ( ### sec) Module : DEBUG -> TIMING Entity "name" action: key "value" key "value"
        # pre is after TIMING up to colon, kw is key word entries after colon
        try:
          pre,kw = line.split('->')[1].split(':')
        except ValueError as e:
          print 'Errored on line',l+1
          raise e
        #print 'pre:',pre
        pre = pre.strip()[6:]

        # time is the time stamp from raven
        time = float(line.split('sec')[0].strip('( '))

        # determine entity giving timing
        in_step = 'STEP' in pre
        in_jobhandler = 'JobHandler' in pre
        in_model = 'Model' in pre

        # action  is the part right before the colon
        action = pre.split(' ')[-1]

        # obj_name is name of entity doing the timing
        obj_name = pre.split('"')[1]

        # kw are the entries after the colon to describe who is being run
        kw = self.getKW(kw.strip())

        # case: entity
        if in_step:
          self.readStepAction(action,obj_name,kw,time)
        elif in_jobhandler:
          self.readJobHandlerAction(action,obj_name,kw,time)
        elif in_model:
          self.readModelAction(action,obj_name,kw,time)
    print 'There were {} finished, {} unfinished, and {} failed jobs collected.'.format(len(self.done_jobs),len(self.working_jobs),len(self.failed_jobs))

  #### reading actions ####
  def readStepAction(self,action,obj_name,kw,time):
    if action == 'submitting':
      job = Job()
      job.step = obj_name
      job.jid = kw['jobID'] # job ID isn't necessarily an integer
      job.step_submit = time
      self.working_jobs.append(job)
    elif action == 'collected':
      jid = kw['JobID']
      job = self.findJob(jid)
      if job is None:
        raise KeyError,'UNFOUND BUT COLLECTED: '+jid
      job.step_collect = time
      job.cid = int(kw['StepCounter'])
      job.status = kw['Status']
    elif action == 'collectedDone':
      jid = kw['JobID']
      job = self.findJob(jid)
      job.step_finish = time
      job.finalize()
      self.working_jobs.remove(job)
      if job.status == 'PASS':
        self.done_jobs.append(job)
      else:
        self.failed_jobs.append(job)

  def readJobHandlerAction(self,action,obj_name,kw,time):
    jid = kw['jobID']
    job = self.findJob(jid)
    if job is None:
      #print 'Untracked job:',jid
      return
    if action in ['putOnNormalQueue','putOnClientQueue']:
      job.jh_onQ = time
    elif action in ['startingNormalRun','startingClientRun']:
      job.jh_start = time
    elif action == 'finishedWaitCollect':
      job.jh_finish = time
    return

  def readModelAction(self,action,obj_name,kw,time):
    jid = kw['jobID']
    job = self.findJob(jid)
    if job is None:
      return
    if action == 'evaluateSampleStart':
      job.m_start = time
    elif action == 'evaluateSampleRun':
      job.m_run = time
    elif action == 'evaluateSampleFinish':
      job.m_finish = time
    elif action == 'evaluateSampleReturn':
      job.m_return = time

  #### utilities ####
  def findJob(self,jid):
    for mjob in self.working_jobs:
      if mjob.jid == jid:
        return mjob
    return None

  def getKW(self,kw):
    entries = list(e.strip() for e in kw.split('"'))[:-1]
    return dict((entries[2*i],entries[2*i+1]) for i in range(len(entries)/2))

  def cleanLogFile(self):
    print 'Cleaning log file ...'
    #### Accidental double-lines split into individual lines
    new = open(self.fname+'_new','w')
    old = open(self.fname,'r')
    # line starts with "("
    # followed by 1 or more spaces
    # followed by a float (numbers plus decimal)
    # followed by a space
    # followed by "sec)"
    line_start = r'[(] *[0-9]*[.][0-9]* sec[)]'
    for l,line in enumerate(old):
      line = line.strip() # take off newlines
      if re.search(line_start,line):
        # special cleanup
        for starter in ['/apps/local/','randomUtils: ','(            ) UTILS']:
          if starter in line:
            line = line.split(starter)[0]
        beginners = list(re.finditer(line_start,line))
        if len(beginners)>1:
          for b in range(len(beginners[:-1])):
            start = beginners[b].start()
            end = beginners[b+1].start()
            new.writelines(line[start:end]+'\n')
          start = beginners[-1].start()
          end = len(line)
          new.writelines(line[start:end]+'\n')
        else:
          new.writelines(line+'\n')
      else:
        pass
        #new.writelines(line+'\n')
    new.close()
    old.close()
    os.system('cp {}_new {}'.format(self.fname,self.fname))
              
  def plotJobs(self):
    if len(self.done_jobs) == 0:
      print 'No jobs to show!'
      return
    # choose your adventure
    #plt.figure()
    #self.__plotElapsed()
    #plt.figure()
    #self.__plotMarkers()
    #plt.figure()
    #self.__plotLines()
    plt.figure()
    self.__plotStacked()
    #plt.figure()
    #self.__plotModelStacked()
    #plt.figure()
    #self.__plotActivity()
    print 'Showing plots ...'
    plt.show()

  def __plotElapsed(self):
    """
      Plots the elapsed time on Y for each run on X (sorted by submission time)
    """
    ### ELAPSED
    print 'Plotting elapsed ...'
    timings       = list(job.step_collect - job.step_submit for job in self.done_jobs)
    time_running  = list(job.jh_finish    - job.jh_start    for job in self.done_jobs)
    time_startq   = list(job.jh_start     - job.step_submit for job in self.done_jobs)
    time_collectq = list(job.jh_finish    - job.step_submit for job in self.done_jobs)
    submit = list(job.step_submit for job in self.done_jobs)
    print 'Global run time mean :',np.average(timings)
    print 'Global run time sigma:',np.std(timings)
    half = int(len(timings)/2)
    p10 = int(len(timings)/10)
    print ''
    print 'First 1/2 run time mean:',np.average(timings[:half])
    print 'Last  1/2 run time mean:',np.average(timings[-half:])
    print ''
    print 'First 1/10 run time mean:',np.average(timings[:p10])
    print 'Last  1/10 run time mean:',np.average(timings[-p10:])
    plt.plot(submit,timings,      'b.',label='total'       )
    plt.plot(submit,time_running ,'k.',label='running'     )
    plt.plot(submit,time_startq  ,'g.',label='startQueue'  )
    plt.plot(submit,time_collectq,'r.',label='collectQueue')
    plt.legend(loc=0)
    plt.xlabel('Time run submitted (s from RAVEN start)')
    plt.ylabel('Total elapsed time (s)')
    plt.title('Submission to Collection timings by run, from '+self.fname)
    plt.savefig('{}_timings.png'.format(self.fname.split('.')[0]), dpi=300)

  def __plotLines(self):
    """
      Y is run (by submission time), X is time, each line is divided into colors for parts of run
    """
    print 'Plotting lines ...'
    self.done_jobs.sort(key = lambda x:x.step_submit)
    colors = [               'b',        'g',          'k',            'r']
    for j,job in enumerate(self.done_jobs):
      steps = [job.step_submit, job.jh_onQ, job.jh_start, job.jh_finish, job.step_collect]
      j += 1
      # step.submit, jh.Q, jh.run, jh.finish, step.collect
      for i in range(len(steps)-1):
        plt.plot([steps[i], steps[i+1]], [j,j], '-', color=colors[i])
    plt.ylabel('Run Number (sorted by submission time)')
    plt.xlabel('RAVEN Time (s)')
    plt.title('Submitted (b), On Queue (g), Running (k), WaitCollect (r)')
    plt.ylim(0,len(self.done_jobs)+1)
    plt.savefig('{}_lines.png'.format(self.fname.split('.')[0]), dpi=300)

  def __plotMarkers(self):
    """
      Y is run (by submission time), X is time, each line has markers for milestones
    """
    print 'Plotting markers ...'
    if len(self.done_jobs) == 0:
      print 'No jobs to show!'
      return
    self.done_jobs.sort(key = lambda x:x.step_submit)
    for j,job in enumerate(self.done_jobs):
      j += 1
      # step.submit, jh.Q, jh.run, jh.finish, step.collect
      # whole line
      plt.plot([job.step_submit,job.step_collect],[j,j],'r-o',zorder=1)
      # marker points
      kw = {'verticalalignment':'center', 'horizontalalignment':'center'}#, 'zorder':10}
      plt.text(job.jh_onQ,       j, 'Q', zorder=10, **kw)
      plt.text(job.jh_start,     j, 'S', zorder=10, **kw)
      plt.text(job.jh_finish,    j, 'F', zorder=10, **kw)
      plt.text(job.step_collect, j, 'C', zorder=10, **kw)
    plt.ylabel('Run Number (sorted by submission time)')
    plt.xlabel('RAVEN Time (s)')
    plt.title('[Q]ueue, [S]tart, [F]inish, [C]ollect times')
    plt.ylim(0,len(self.done_jobs)+1)

  def __plotStacked(self):
    """
      Plots stacked columns showing how much cumulative time is spent in each part, by color.
    """
    print 'Plotting stacked ...'
    print ' ... collecting data ...'
    time_startq   = np.array(list(job.jh_start     - job.step_submit  for job in self.done_jobs))
    time_running  = np.array(list(job.jh_finish    - job.jh_start     for job in self.done_jobs))
    time_collectq = np.array(list(job.step_collect - job.jh_finish    for job in self.done_jobs))
    time_store    = np.array(list(job.step_finish  - job.step_collect for job in self.done_jobs))
    time_total    = np.array(list(job.step_finish  - job.step_submit  for job in self.done_jobs))

    style = 'line'
    print ' ... constructing plots ...'
    if style == 'bar':
      # FIXME needs updating!!! # bar plots are surprisingly slow
      xs = range(len(self.done_jobs))
      print ' ... ... stack start ...'
      p_start = plt.bar(xs,time_startq, linewidth=0, color='g',label='queue')
      print ' ... ... stack run ...'
      p_run = plt.bar(xs,time_running, bottom=time_startq, linewidth=0, color='k',label='run')
      print ' ... ... stack collect ...'
      p_collect = plt.bar(xs,time_collectq, bottom=time_startq+time_running, linewidth=0, color='r',label='collect')
      plt.legend((p_start[0],p_run[0],p_collect[0]), ('Start','Run','Collect'))
    elif style == 'line':
      # are line plots faster? -> not really
      colors = ['b','g','k','r','c']
      names = ['submit','queue','run','collect','store']
      kw = {'linewidth':1}
      for j,job in enumerate(self.done_jobs):
        steps = np.array([job.step_submit, job.jh_onQ, job.jh_start, job.jh_finish, job.step_collect, job.step_finish]) - job.step_submit
        for i in range(len(steps)-1):
          if j == 0:
            plt.plot([j,j],[steps[i],steps[i+1]], '-', color=colors[i], label=names[i], **kw)
          else:
            plt.plot([j,j],[steps[i],steps[i+1]], '-', color=colors[i], **kw)
    # either style ...
    plt.legend(loc=0)
    plt.ylabel('Time (s)')
    plt.xlabel('Run Number (order collected)')
    plt.title('Job time spent, stacked')
    print ' ... saving ...'
    plt.savefig('{}_stacked.png'.format(self.fname.split('.')[0]), dpi=300)

    # set up individual figures
    fig_start,   ax_start   = plt.subplots()
    fig_run,     ax_run     = plt.subplots()
    fig_collect, ax_collect = plt.subplots()
    fig_store,   ax_store   = plt.subplots()

    # plot separate parts
    print ' ... ... indiv ...'
    if style == 'bar':
      # TODO needs updating
      ax_start.bar(xs,time_startq,linewidth=0,color='g',label='start')
      ax_run.bar(xs,time_running,linewidth=0,color='k',label='run')
      ax_collect.bar(xs,time_collectq,linewidth=0,color='r',label='collect')
    elif style=='line':
      for j,job in enumerate(self.done_jobs):
        if j == 0:
          ax_start.  plot([j],[job.jh_start    -job.step_submit ],'g.',label='start'  )
          ax_run.    plot([j],[job.jh_finish   -job.jh_start    ],'k.',label='run'    )
          ax_collect.plot([j],[job.step_collect-job.jh_finish   ],'r.',label='collect')
          ax_store.  plot([j],[job.step_finish -job.step_collect],'c.',label='store')
        else:
          ax_start.  plot([j],[job.jh_start    -job.step_submit ],'g.')
          ax_run.    plot([j],[job.jh_finish   -job.jh_start    ],'k.')
          ax_collect.plot([j],[job.step_collect-job.jh_finish   ],'r.')
          ax_store.  plot([j],[job.step_finish -job.step_collect],'c.')

    for ax in [ax_start,ax_run,ax_collect,ax_store]:
      ax.set_ylabel('Time (s)')
      ax.set_xlabel('Run Number (order collected)')
      #ax.set_ylim(0,400)
      ax.legend(loc=0)
    fig_start  .savefig('{}_start.png'  .format(self.fname.split('.')[0]), dpi=300)
    fig_run    .savefig('{}_running.png'.format(self.fname.split('.')[0]), dpi=300)
    fig_collect.savefig('{}_collect.png'.format(self.fname.split('.')[0]), dpi=300)
    fig_store  .savefig('{}_store.png'  .format(self.fname.split('.')[0]), dpi=300)

  def __plotModelStacked(self):
    """
      Plots stacked columns showing how much cumulative time is spent in each MODEL part, by color.
    """
    print 'Plotting model stacked ...'
    style = 'line'

    # check for None, disable for speed
    #bad = False
    #for job in self.done_jobs:
    #  if None in [job.m_start, job.m_run, job.m_finish, job.m_return]:
    #    bad = True
    #    print 'Job:',job.jid,[job.m_start, job.m_run, job.m_finish, job.m_return]
    #if bad:
    #  raise IOError('Missing data!')

    print ' ... constructing plots ...'
    if style == 'bar':
      print ' ... collecting data ...'
      time_running  = np.array(list(job.m_finish - job.m_run    for job in self.done_jobs))
      time_startq   = np.array(list(job.m_run    - job.m_start  for job in self.done_jobs))
      time_collectq = np.array(list(job.m_return - job.m_finish for job in self.done_jobs))
      time_total    = np.array(list(job.m_return - job.m_start  for job in self.done_jobs))
      # bar plots are surprisingly slow
      xs = range(len(self.done_jobs))
      print ' ... ... stack start ...'
      p_start = plt.bar(xs,time_startq, linewidth=0, color='g',label='queue')
      print ' ... ... stack run ...'
      p_run = plt.bar(xs,time_running, bottom=time_startq, linewidth=0, color='k',label='run')
      print ' ... ... stack collect ...'
      p_collect = plt.bar(xs,time_collectq, bottom=time_startq+time_running, linewidth=0, color='r',label='collect')
      plt.legend((p_start[0],p_run[0],p_collect[0]), ('Start','Run','Collect'))
    elif style == 'line':
      # are line plots faster? -> not really
      colors = ['c','g','k','r','y']
      names = ['remote','startup','run','finish','return']
      kw = {'linewidth':1}
      for j,job in enumerate(self.done_jobs):
        steps = np.array([job.jh_start,job.m_start, job.m_run, job.m_finish, job.m_return, job.jh_finish ]) - job.jh_start
        for i in range(len(steps)-1):
          if j == 0:
            plt.plot([j,j],[steps[i],steps[i+1]], '-', color=colors[i], label=names[i], **kw)
          else:
            plt.plot([j,j],[steps[i],steps[i+1]], '-', color=colors[i], **kw)
    # either style ...
    plt.legend(loc=0)
    plt.ylabel('Time (s)')
    plt.xlabel('Run Number (order collected)')
    plt.title('MODEL time spent, stacked')
    print ' ... saving ...'
    plt.savefig('{}_model_stacked.png'.format(self.fname.split('.')[0]), dpi=300)

    # set up individual figures
    fig_remote,  ax_remote  = plt.subplots()
    fig_start,   ax_start   = plt.subplots()
    fig_run,     ax_run     = plt.subplots()
    fig_collect, ax_collect = plt.subplots()
    fig_return,  ax_return  = plt.subplots()

    # plot separate parts
    print ' ... ... indiv ...'
    if style == 'bar':
      ax_start.bar(xs,time_startq,linewidth=0,color='g',label='start')
      ax_run.bar(xs,time_running,linewidth=0,color='k',label='run')
      ax_collect.bar(xs,time_collectq,linewidth=0,color='r',label='collect')
    elif style=='line':
      for j,job in enumerate(self.done_jobs):
        if j == 0:
          ax_remote. plot([j],[job.m_start  -job.jh_start],'c.',label='remote')
          ax_start.  plot([j],[job.m_run    -job.m_start ],'g.',label='startup')
          ax_run.    plot([j],[job.m_finish -job.m_run   ],'k.',label='run'    )
          ax_collect.plot([j],[job.m_return -job.m_finish],'r.',label='finish' )
          ax_return. plot([j],[job.jh_finish-job.m_return],'y.',label='return' )
        else:
          ax_remote. plot([j],[job.m_start  -job.jh_start],'c.')
          ax_start.  plot([j],[job.m_run    -job.m_start ],'g.')
          ax_run.    plot([j],[job.m_finish -job.m_run   ],'k.')
          ax_collect.plot([j],[job.m_return -job.m_finish],'r.')
          ax_return. plot([j],[job.jh_finish-job.m_return],'y.')

    for ax in [ax_remote,ax_start,ax_run,ax_collect,ax_return]:
      ax.set_ylabel('Time (s)')
      ax.set_xlabel('Run Number (order collected)')
      #ax.set_ylim(0,400)
      ax.set_title('MODEL timing')
      ax.legend(loc=0)
    print ' ... ... saving indiv ...'
    fig_remote .savefig('{}_model_remote.png' .format(self.fname.split('.')[0]), dpi=300)
    fig_start  .savefig('{}_model_start.png'  .format(self.fname.split('.')[0]), dpi=300)
    fig_run    .savefig('{}_model_running.png'.format(self.fname.split('.')[0]), dpi=300)
    fig_collect.savefig('{}_model_collect.png'.format(self.fname.split('.')[0]), dpi=300)
    fig_return .savefig('{}_model_return.png' .format(self.fname.split('.')[0]), dpi=300)

  def __plotActivity(self):
    """
    Over each second of the simulation, plots what each job is up to (ignoring 'not started' and 'done')
    """
    print 'Plotting activity ...'
    print ' ... collecting data ...'
    longest = int(np.ceil(self.done_jobs[-1].step_collect+0.5))
    xs = range(longest+1)
    N = len(xs)
    submitted = np.zeros(N)
    running = np.zeros(N)
    finished = np.zeros(N)
    capfac = np.zeros(N)
    for sec in xs:
      if longest % (longest/100) == 0:
        print '... ... sec:',sec,'\r',
      for job in self.done_jobs:
        # if not started or already done, skip
        if sec < job.step_submit : continue
        if sec > job.step_collect: continue
        # sort activity
        if   sec < job.jh_start : submitted[sec] += 1
        elif sec < job.jh_finish: running[sec] += 1
        else                    : finished[sec] += 1
      capfac[sec] = running[sec]/81.0 # FIXME hard-coded batch size
    capfac_tot = 1./(N*81.) * np.sum(running) # FIXME hard-coded batch size
    print ' -> Capacity factor, mean:',capfac_tot
    print '                      std:',np.std(capfac)
    print '                      min:',np.min(capfac)
    print '                      max:',np.max(capfac)
    print ' ... plotting ...'
    print ' ... ... start ...'
    plt.plot(xs,submitted,'g.',label='start queue'  )
    print ' ... ... run ...'
    plt.plot(xs,running,  'k.',label='running'      )
    print ' ... ... collect ...'
    plt.plot(xs,finished, 'r.',label='collect queue')
    plt.title('RAVEN activity, snapshots every second')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Number of Jobs in Activity')
    plt.legend(loc=0)
    print ' ... saving ...'
    plt.savefig('{}_activity.png'.format(self.fname.split('.')[0]), dpi=300)
    # capacity factor
    plt.figure()
    plt.plot(xs,capfac,'b.')
    plt.title('Capacity Factor snapshots every second')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Capacity Factor (jobs running / batch size')
    plt.savefig('{}_capfac.png'.format(self.fname.split('.')[0]), dpi=300)
      


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  logname = sys.argv[1]
  if '--skipClean' in sys.argv:
    skip = True
  else:
    skip = False
  handler = TimingHandler(logname,skip)
  handler.plotJobs()
  #timings = []
  #for job in done_jobs:
  #  timings.append((job.step_collect - job.step_submit,job.step_submit))



