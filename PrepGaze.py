

##################################################################################################################
#                                                                                                                #
# EYE TRACKING DATA PREPROCESSING FOR GROWTH CURVE ANALYSIS                                                      #
# (C) NADJA ALTHAUS, 2022 https://github.com/nadjaalthaus                                                        #
#                                                                                                                #
# Developed for preprocessing EyeLink sample reports for growth curve analysis                                   #
# for Althaus, Kotzor, Schuster & Lahiri (2022). Distinct orthography boosts morphophonological discrimination:  #
# Vowel raising in Bengali verb inflections. Cognition, 222, 104963.                                             #
# https://doi.org/10.1016/j.cognition.2021.104963                                                                #
#                                                                                                                #
##################################################################################################################

import numpy as np
import pandas as pd
import re



##################################################
#                                                #
# aux functions for long format preprocessing    #
#                                                #
##################################################

#convert x into float if possible or return nan
def nanfloat(x):
    try:
        y=float(x)
    except ValueError as e:
        y=np.nan
    return y


#convert x into int if possible or return nan
def nanint(x):
    try:
        y=int(x)
    except ValueError as e:
        y=np.nan
    return y





def strip_brackets(s):
    m=re.search('^\[(.*)\]$',s)
    if m:
        return m.group(1)
    else:
        return s


#######################
#                     #
#   core functions    #
#                     #
#######################

def downsample(filename,sd,instruct_filename):
    """Performs downsampling for gaze data from EyeLink sample report
           Writes output file with suffix <stem>_ds.csv where <stem> is from filename

          Parameters
          ----------
          filename : csv file (Eye Link sample report)

          sd: target sample duration in ms - number of ms one sample will contain
          instruct_filename : is a csv file that contains one row per header variable where
                column 1 is the variable name (corresponding to columns in <filename>
                column 2 specifies what downsampling does with information in that column
                    'mean' calculates the mean over all entries in the new sample (must be numeric)
                    'first' uses the value of the first entry (goes for string and numeric)
                    'sum' adds up all entries (must be numeric)
                    'any' will find the first nonempty value even if it is not in the first slot (strings and numeric)
                    'maj' will assign the value that the majority of entries fall into



          """


    stemre=re.match('(.*)\.csv$', filename)
    stem=stemre.group(1)

    InData=pd.read_csv(filename)


    #determine which eye was tracked, currently code handles monocular tracking only
    try:
        Eye=InData.loc[0,'EYE_TRACKED']

        if Eye=='Left':
            Eye='LEFT'
            unusedEye='RIGHT'
        elif Eye=='Right':
            Eye='RIGHT'
            unusedEye='LEFT'

    except:
        if 'RIGHT_GAZE_X' in InData.columns:
            if not 'LEFT_GAZE_X'in InData.columns:
                Eye='RIGHT'
            else:
                print('both eyes found, unsure')
                return
        elif 'LEFT_GAZE_X' in InData.columns:
            if not 'RIGHT_GAZE_X'in InData.columns:
                Eye='LEFT'
            else:
                print('both eyes found, unsure')
                return
        else:
            print('no eyes found, error')
            return

    InData['TRIAL_INDEX']=pd.to_numeric(InData['TRIAL_INDEX'],errors='coerce')
    if Eye=='LEFT':
        print('left eye assigned')
        InData['FIX_INDEX']=pd.to_numeric(InData['LEFT_FIX_INDEX'],errors='coerce')
        unusedEye='RIGHT'

    elif Eye=='RIGHT':
        print('right eye assigned')
        InData['FIX_INDEX']=pd.to_numeric(InData['RIGHT_FIX_INDEX'],errors='coerce')
        unusedEye = 'LEFT'


    #read Instruction file
    Instruct=pd.read_csv(instruct_filename)

    #prepare data according to Instruction file
    for i in range(Instruct.shape[0]):

        #ignore vars that were not exported
        if not Instruct.loc[i,'VarName'] in InData.columns:
            continue

        #for variables that will be averaged across the slot, make sure the data are numeric
        if Instruct.loc[i,'SampleAction']=='mean':
            InData[Instruct.loc[i,'VarName']]=pd.to_numeric(InData[Instruct.loc[i,'VarName']],errors='coerce')

        #for columns with sparse information there will be lots of empty rows, which is marked in output with a single dot
        #replace with empty string
        if Instruct.loc[i,'SampleAction']=='any':
            InData[Instruct.loc[i,'VarName']]=InData[Instruct.loc[i,'VarName']].replace('.','')

        #conflate eye-specific columns
        eye_re= re.search(re.compile('^'+Eye+'\_(.*)$'),Instruct.loc[i,'VarName'])
        if eye_re:
            #rename, e.g. LEFT_GAZE_X to GAZE_X
            InData.rename(columns={eye_re.group(0):eye_re.group(1)})
            #remove anything else ending in GAZE_X, e.g. RIGHT_GAZE_X
            if Eye=='LEFT':
                unusedEye='RIGHT'
            elif Eye=='RIGHT':
                unusedEye='LEFT'
            for c in InData.columns:
                if re.search(re.compile('^'+unusedEye+eye_re.group(1)+'$'),c):
                    InData=InData.drop(columns=c)




    #divide InData into trials and fixations
    #necessary to avoid an interval for downsampling spanning a trial boundary
    maxtrials=max(InData['TRIAL_INDEX'])


    OutDataDict={}

    for i in range(1,maxtrials+1):

        InTrial=InData[InData['TRIAL_INDEX']==i]
        InTrial=InTrial.reset_index(drop=True)

        maxfixations=0

        counter=0
        for k in range(0,InTrial.shape[0],sd):

            InSlot=InTrial.loc[k:k+sd-1]

            InSlot=InSlot.reset_index(drop=True)

            if InSlot.shape[0]==0:
                break

            for m in range(Instruct.shape[0]):


                varname=Instruct.loc[m,'VarName']
                outvarname=varname

                #make sure that input file has that variable
                if not varname in InSlot.columns:
                    continue


                s_action=Instruct.loc[m,'SampleAction']

                #don't include if unused eye, rename if relevant to used eye
                if re.search(re.compile('^'+unusedEye+'.*$'), varname):
                    continue


                eye_match=re.search(re.compile('^'+Eye+'\_(.*)$'), varname)

                if eye_match:

                    outvarname=eye_match.group(1)


                # variables for which the first sample is used in downsampling
                if s_action=='first':

                    OutDataDict.setdefault(outvarname,[]).append(InSlot.loc[0,varname])
                #where the instruction is to average across all samples in interval
                elif Instruct.loc[m,'SampleAction']=='mean':
                    OutDataDict.setdefault(outvarname,[]).append(np.mean(InSlot.loc[0:InSlot.shape[0],varname]))
                #where the instruction is to find 'any', i.e. the first nonempty entry in the interval
                elif Instruct.loc[m,'SampleAction']=='any':

                    found=False
                    for n in range(InSlot.shape[0]):
                        if not found and len(InSlot.loc[n,varname])>0:

                            OutDataDict.setdefault(outvarname,[]).append(InSlot.loc[n,varname])

                            found = True
                    if not found:
                        OutDataDict.setdefault(outvarname,[]).append('')

                #where the instruction is to use the majority entry
                elif Instruct.loc[m,'SampleAction']=='maj':
                    counts=pd.value_counts(InSlot[varname])
                    maj_entry=counts.idxmax()
                    OutDataDict.setdefault(outvarname,[]).append(maj_entry)


            counter=counter+1


    #write the output file
    OutData=pd.DataFrame(OutDataDict)
    OutData.to_csv(stem+'_ds.csv', index=False)




def clean_ds_data(filename, subjrep=''):
    """
        Call AFTER downsample
        Performs cleaning/auxiliary processes:
        - AOI to number from list
        - rename the column 'RECORDING_SESSION_LABEL' to 'SUBJECTNO'
        - add a unique trial identifier (subjectno_trialno) as column 'TR_ID'
        - drop redundant columns
        - add a Time column and a Slot column to be used in growth curve analysis
        - rename subject numbers if listed in file given as input arg subjrep
        writes output file adding suffix '_cln'

              Parameters
              ----------
              filename : csv file (Eye Link sample report / downsampled)

              subjrep : file name for subject replacement



              """




    dsdata=pd.read_csv(filename)

    #check if subject numbers need to be replaced, and if so, then replace
    if len(subjrep)>0:
        subjno_replacement=pd.read_csv(subjrep)
        try:
            for i in range(subjno_replacement.shape[0]):

                infile_to_replace=subjno_replacement.loc[i,'Filename']

                toreplacere=re.search('(.*).csv', infile_to_replace)
                if toreplacere:
                    repstem=toreplacere.group(1)


                infilestemre=re.search('.*/(.*)\_ds.csv', filename)
                if infilestemre:
                    stem=infilestemre.group(1)


                if stem==repstem:
                    print('REPLACING')
                    dsdata['RECORDING_SESSION_LABEL']=subjno_replacement.loc[i,'AssignedNo']
        except:
            print('no subj number to be replaced here')



    #strip brackets from AOI output if necessary
    try:
        dsdata['INTEREST_AREAS']=dsdata['INTEREST_AREAS'].apply(strip_brackets)
        dsdata['INTEREST_AREAS']=pd.to_numeric(dsdata['INTEREST_AREAS'],errors='coerce')
    except:
        print('no INTEREST_AREAS to strip')


    dsdata['SUBJECTNO']=dsdata['RECORDING_SESSION_LABEL']
    dsdata=dsdata.drop(columns='RECORDING_SESSION_LABEL')

    #drop columns that are not used
    ColsToDrop=['TRIAL_LABEL','Trial_Index_','Trial_Recycled_', 'SAMPLE_INPUT']

    for c in ColsToDrop:
        try:
            dsdata=dsdata.drop(columns=c)

        except:
            print('skipped drop:'+c)

    #construct a TRIAL_ID variable that will be unique even when all subjects collated
    dsdata['temp']=dsdata['SUBJECTNO'].astype(str) +'_'
    dsdata['TRIAL_ID']=dsdata['temp'].astype(str)+dsdata['TRIAL_INDEX'].astype(str)
    dsdata=dsdata.drop(columns='temp')

    #calculate a TIME variable which is time since the trial start, and a Slot variable that counts slots from the trial start
    dsdata['TIME']=dsdata['TIMESTAMP']-dsdata['TRIAL_START_TIME']
    sd=dsdata.loc[1,'TIME']-dsdata.loc[0,'TIME']
    dsdata['Slot']=dsdata['TIME']/sd


    #identify the file stem, then write the output file
    fsre=re.search('^(.*).csv$',filename)
    if fsre:
        filestem=fsre.group(1)
    else:
        filestem=filename

    dsdata.to_csv(filestem+'_cln.csv', index=False)


def add_TarDisColumn(filename, aoi_criteria):
    """
           Call AFTER downsample, clean_ds
           Adds a column 'TarDis' which labels each sample as falling on target or distracter
           In the current version this is specific to setups with a top and a bottom AOI
           #TODO make this more parametric
           Note: This function relies on the variable tarloc which is -1 if the target in this trial is at the top, and 1 if the target
           is at the bottom. In the Bengali Vowel Raising study this was used to control target/distracter location in ExperimentBuilder
           which means we could export it as part of the sample reports, but could in theory be added afterwards.

           Writes output to csv file with suffix _tardis

                 Parameters
                 ----------
                 filename : csv file (Eye Link sample report / downsampled / cleaned)

                 aoi_criteria: dictionary with keys top_y_max, top_y_min, bottom_y_min, bottom_y_max, top_x_max etc.
                                values are corresponding pixel values



                 """

    dsdata=pd.read_csv(filename)

    TarDis=[]

    top_y_max = aoi_criteria['top_y_max']
    top_y_min = aoi_criteria['top_y_min']
    top_x_min = aoi_criteria['top_x_min']
    top_x_max = aoi_criteria['top_x_max']
    bottom_y_max = aoi_criteria['bottom_y_max']
    bottom_y_min = aoi_criteria['bottom_y_min']
    bottom_x_min = aoi_criteria['bottom_x_min']
    bottom_x_max = aoi_criteria['bottom_x_max']

    for i in range(dsdata.shape[0]):

        tarloc=dsdata.loc[i,'targetloc']
        tarloc=pd.to_numeric(tarloc, errors='coerce')
        gaze_x=dsdata.loc[i,'GAZE_X']
        gaze_y=dsdata.loc[i,'GAZE_Y']
        fix_idx=pd.to_numeric(dsdata.loc[i,'FIX_INDEX'], errors='coerce')


        label=np.nan

        if tarloc<0:

            #target at top

            if gaze_x>top_x_min and gaze_x<top_x_max and gaze_y>top_y_min and gaze_y<top_y_max:
                if ~np.isnan(fix_idx):
                    label=1

            elif gaze_x>bottom_x_min and gaze_x<bottom_x_max and gaze_y>bottom_y_min and gaze_y<bottom_y_max:
                if ~np.isnan(fix_idx):
                    label=-1

        else:
            #target at bottom, distracter at top
            if gaze_x>top_x_min and gaze_x<top_x_max and gaze_y>top_y_min and gaze_y<top_y_max:
                if ~np.isnan(fix_idx):
                    label=-1
            elif gaze_x>bottom_x_min and gaze_x<bottom_x_max and gaze_y>bottom_y_min and gaze_y<bottom_y_max:
                if ~np.isnan(fix_idx):
                    label=1

        TarDis.append(label)
    dsdata['TarDis']=TarDis


    fsre=re.search('^(.*).csv$',filename)
    if fsre:
        filestem=fsre.group(1)
    else:
        filestem=filename
    dsdata.to_csv(filestem+'_tardis.csv', index=False)








def add_LookIndex(filename, maxgap=200, aoicol='TarDis'):
    """
           Call AFTER downsample, clean_ds, add_TarDisColumn
           Adds a column counting Looks 'LOOK_INDEX', where subsequent fixations are counted as the same 'look' as long as the gap is
           smaller than maxgap (default 200ms)

           writes output file adding suffix '_lk'

                 Parameters
                 ----------
                 filename : csv file (Eye Link sample report / downsampled / cleaned / with TarDis column added)

                 maxgap: maximum gap (in ms) between samples in same AOI for sample still to be counted as part of the same Look

                 aoicol: label of column to be used to check whether sample is in AOI in order to perform Look assignment
                        (default: use TarDis column, which contains 1 for target, -1 for distracter)
                        needs to contain numeric AOI labels #TODO make this more flexible

                 """



    dsdata=pd.read_csv(filename)

    trials=pd.unique(dsdata['TRIAL_INDEX'])
    Look=[] #for output

    #iterate through trials
    for t in trials:
        this_trial=dsdata[dsdata['TRIAL_INDEX']==t]
        this_trial=this_trial.reset_index(drop=True)
        endt=0 #start at time t=0

        curr_aoi=np.nan #initialise curr_aoi
        lookno=1 # start counting looks at 1

        for i in range(this_trial.shape[0]):

            current_timestamp=this_trial.loc[i,'TIMESTAMP']-this_trial.loc[i,'TRIAL_START_TIME']
            gap=current_timestamp-endt
            this_sample_aoi=this_trial.loc[i,aoicol] #the assigned AOI label

            #make sure this is in numeric format
            if type(this_sample_aoi) is str:

                if len(this_sample_aoi)==0 or this_sample_aoi=='.':

                    this_sample_aoi=np.nan
                else:

                    aoinumre=re.search('[0-9]+', this_sample_aoi)
                    if aoinumre:

                        this_sample_aoi=int(aoinumre.group(0))


            lookval=np.nan

            if ~np.isnan(this_sample_aoi):
                #this sample is assigned to an AOI

                if np.isnan(curr_aoi): #this is the first AOI encountered
                    curr_aoi=this_sample_aoi
                    lookval=lookno
                    endt=current_timestamp

                #this sample is in the same AOI as previously encountered
                elif this_sample_aoi==curr_aoi:

                    #is the gap since the last marked sample small enough?
                    if gap<maxgap:
                        lookval=lookno
                        endt=current_timestamp
                    else:
                        lookno=lookno+1
                        lookval=lookno
                        endt=current_timestamp

                #this sample is assigned to a different aoi compared to the previous one
                else:
                    lookno=lookno+1
                    curr_aoi=this_sample_aoi
                    lookval=lookno
                    endt=current_timestamp
            Look.append(lookval)

    #Add looks as a column and write output file
    dsdata['LOOK']=Look

    fsre=re.search('^(.*).csv$',filename)
    if fsre:
        filestem=fsre.group(1)
    else:
        filestem=filename
    dsdata.to_csv(filestem+'_lk.csv', index=False)




#add First and NSamp to downsampled data - the latter 2 to be used in modelling with odds ratio
#must already have LOOK info and TarDis column and wide filename
def add_OddsRatioInfo(filename):
    """
              Call AFTER downsample, clean_ds, add_TarDisColumn, add_LookIndex
              Adds columns First, NSamp and Dir1 that will later serve to calculate the odds ratio for growth curve analysis
              Must have columns LOOK and TarDis
              'First' will have value 1 if the AOI is the one that the first fixation in the trial was landed on, 0 otherwise
              'NSamp' will have value 1 if the sample was inside an AOI, 0 otherwise
              'Dir1' lists the direction of the first look of the present trial

              writes output file adding suffix '_frst'

                    Parameters
                    ----------
                    filename : csv file (Eye Link sample report / downsampled / cleaned / TarDis added/ LOOK added)


                    """

    dsdata=pd.read_csv(filename)
    unique_trials=pd.unique(dsdata['TRIAL_ID'])
    First=np.array([])

    NSamples=np.array([])
    Dir1=np.array([])

    #iterate over trials
    for t in unique_trials:

        trial_data=dsdata[dsdata['TRIAL_ID']==t]
        trial_data=trial_data.reset_index(drop=True)

        #find the slice of samples containing the first fixation
        firstfix=trial_data[trial_data['LOOK']==1]



        firstfix=firstfix.reset_index(drop=True)

        if firstfix.shape[0]==0:
            this_trial_first=np.zeros(trial_data.shape[0])
            this_trial_nsamp=np.zeros(trial_data.shape[0])
            this_trial_dir1=np.zeros(trial_data.shape[0])
        else:
            #find the direction of the first look
            dir1=firstfix.loc[0,'TarDis']

            #find all fixations directed at that item, create columns First and NSamp for the present trial
            this_trial_first=(trial_data['TarDis']==dir1)
            this_trial_first=np.array(this_trial_first)
            this_trial_first=this_trial_first.astype(int)

            this_trial_nsamp=(abs(trial_data['TarDis'])==1)
            this_trial_nsamp=np.array(this_trial_nsamp)
            this_trial_nsamp=this_trial_nsamp.astype(int)

            this_trial_dir1=np.ones(trial_data.shape[0])*dir1

        #append to collate all trials
        First=np.concatenate((First,this_trial_first))

        NSamples=np.concatenate((NSamples,this_trial_nsamp))

        Dir1=np.concatenate((Dir1,this_trial_dir1))

    #append columns
    dsdata['First']=First
    dsdata['NSamples']=NSamples
    dsdata['Dir1']=Dir1


    #write output file
    fsre=re.search('^(.*).csv$',filename)
    if fsre:
        filestem=fsre.group(1)
    else:
        filestem=filename
    dsdata.to_csv(filestem+'_frst.csv', index=False)







##################################
#                                #
#  AUX FUNCTIONS FOR WIDE FORMAT #
#                                #
##################################


def get_ith_in_list(L,i):
    """
        Returns a list of the ith element in each list in L, with NaN if len of that list was <i

        Returns [TargetSlots,DisSlots], both dataframes

        Parameters
        ----------
        L : list of lists
        i: index (int)
                   """

    if not type(L)==list:
        return []
    out_L=[]
    for l in L:
        if len(l)>i:
            out_L.append(l[i])
        else:
            out_L.append(np.nan)
    return out_L





#return lists Tar (LT directed at target in ms), Dis (LT directed at distracter in ms), Tarprop (targetproportion)

def get_tardis(data, aoi_criteria):
    """
       calculate LT directed at target/distracter and target proportion

            Returns Tar (LT directed at target in ms), Dis (LT directed at distracter in ms), Tarprop (proportion of target looking)

            Parameters
            ----------
            data : dataframe (long format)

            aoi_criteria: dictionary with keys top_y_max, top_y_min, bottom_y_min, bottom_y_max, top_x_max etc.
                                 values are corresponding pixel values

            """

    sampledur=data.loc[1,'TIMESTAMP']-data.loc[0, 'TIMESTAMP']

    Tar=[]
    Dis=[]
    Tarprop=[]
    trialno=pd.unique(data['TRIAL_INDEX'])

    for i in trialno:

        this_trial=data[data['TRIAL_INDEX']==i]
        this_trial=this_trial.reset_index(drop=True)

        [TargetSlots, DisSlots]=get_target_slots(this_trial, aoi_criteria)


        TarSum=TargetSlots.shape[0]*sampledur

        DisSum=DisSlots.shape[0]*sampledur
        Tar.append(TarSum)
        Dis.append(DisSum)
        Tarprop.append(TarSum/(TarSum+DisSum))
    return[Tar,Dis,Tarprop]



def get_target_slots(this_trial, aoi_criteria):
        """
               Return slices corresponding to target and distracter

               Returns [TargetSlots,DisSlots], both dataframes

               Parameters
               ----------
               this_trial : dataframe (slice of downsampled long format, only containing one trial)

               aoi_criteria: dictionary with keys top_y_max, top_y_min, bottom_y_min, bottom_y_max, top_x_max etc.
                                    values are corresponding pixel values

               """

        #identify rows with samples directed at target

        tarloc=this_trial.loc[0,'targetloc']
        try:
            tarloc=int(tarloc)
        except:
            TargetSlots=pd.DataFrame({})
            DisSlots=pd.DataFrame({})
            return [TargetSlots, DisSlots]

        if tarloc<0:

            #target is at the top

            top_y_max=aoi_criteria['top_y_max']
            top_y_min=aoi_criteria['top_y_min']
            top_x_min=aoi_criteria['top_x_min']
            top_x_max=aoi_criteria['top_x_max']

            TargetSlots=this_trial[this_trial['GAZE_Y']<top_y_max]

            TargetSlots=TargetSlots.reset_index(drop=True)
            TargetSlots=TargetSlots[TargetSlots['GAZE_Y']>top_y_min]
            TargetSlots=TargetSlots.reset_index(drop=True)

            TargetSlots=TargetSlots[TargetSlots['GAZE_X']>top_x_min]
            TargetSlots=TargetSlots.reset_index(drop=True)
            TargetSlots=TargetSlots[TargetSlots['GAZE_X']<top_x_max]
            TargetSlots=TargetSlots.reset_index(drop=True)

            #distracter is at the bottom
            bottom_y_min=aoi_criteria['bottom_y_min']
            bottom_y_max=aoi_criteria['bottom_y_max']
            bottom_x_min=aoi_criteria['bottom_x_min']
            bottom_x_max=aoi_criteria['bottom_x_max']

            DisSlots=this_trial[this_trial['GAZE_Y']>bottom_y_min]
            DisSlots=DisSlots.reset_index(drop=True)
            DisSlots=DisSlots[DisSlots['GAZE_Y']<bottom_y_max]
            DisSlots=DisSlots.reset_index(drop=True)
            DisSlots=DisSlots[DisSlots['GAZE_X']>bottom_x_min]
            DisSlots=DisSlots.reset_index(drop=True)
            DisSlots=DisSlots[DisSlots['GAZE_X']<bottom_x_max]
            DisSlots=DisSlots.reset_index(drop=True)

        else:

            #target is at the bottom
            top_y_max=aoi_criteria['top_y_max']
            top_x_min=aoi_criteria['top_x_min']
            top_x_max=aoi_criteria['top_x_max']
            top_y_min=aoi_criteria['top_y_min']

            DisSlots=this_trial[this_trial['GAZE_Y']<top_y_max]
            DisSlots=DisSlots.reset_index(drop=True)
            DisSlots=DisSlots[DisSlots['GAZE_Y']>top_y_min]
            DisSlots=DisSlots.reset_index(drop=True)
            DisSlots=DisSlots[DisSlots['GAZE_X']>top_x_min]
            DisSlots=DisSlots.reset_index(drop=True)
            DisSlots=DisSlots[DisSlots['GAZE_X']<top_x_max]
            DisSlots=DisSlots.reset_index(drop=True)

            #distracter is at the bottom
            bottom_y_min=aoi_criteria['bottom_y_min']
            bottom_x_min=aoi_criteria['bottom_x_min']
            bottom_x_max=aoi_criteria['bottom_x_max']
            bottom_y_max=aoi_criteria['bottom_y_max']
            TargetSlots=this_trial[this_trial['GAZE_Y']>bottom_y_min]
            TargetSlots=TargetSlots.reset_index(drop=True)
            TargetSlots=TargetSlots[TargetSlots['GAZE_Y']<bottom_y_max]
            TargetSlots=TargetSlots.reset_index(drop=True)
            TargetSlots=TargetSlots[TargetSlots['GAZE_X']>bottom_x_min]
            TargetSlots=TargetSlots.reset_index(drop=True)
            TargetSlots=TargetSlots[TargetSlots['GAZE_X']<bottom_x_max]
            TargetSlots=TargetSlots.reset_index(drop=True)
        return [TargetSlots,DisSlots]




def get_accuracy_rt(data,responsevarname,buttonmap):
    """
        Determine manual accuracy for each trial

        Returns lists accuracy and RT in list of length 2

        Parameters
        ----------
        data : dataframe (downsampled long format)

        responsevarname: the name of the column that has the button response eg. KBRESPONSE

        buttonmap: dictionary mapping button values (as in DataViewer output) to target locations -1,1


        """



    trialno=pd.unique(data['TRIAL_INDEX'])

    accuracy=[]
    RT=[]
    for i in trialno:

        acc=0
        this_trial=data[data['TRIAL_INDEX']==i]
        this_trial=this_trial.reset_index(drop=True)

        #print(this_trial.shape[0])
        if this_trial.shape[0]>0:

            manresponse=this_trial.loc[0,responsevarname]
            manresponse=pd.to_numeric(manresponse,errors='coerce')

            if manresponse>0 and buttonmap[manresponse]==this_trial.loc[0,'targetloc']:
                acc=1
        accuracy.append(acc)
        rt=pd.to_numeric(this_trial.loc[0, 'RESPONSETIME'], errors='coerce')- pd.to_numeric(this_trial.loc[0,'DISPLAYSTARTTIME'],errors='coerce')

        RT.append(rt)
    return [accuracy, RT]



def get_firstlook(data, aoi_criteria):
    """
           Determine first look information for each trial
           latency is based on first sample that is in an aoi that isn't called FIXATION #TODO make this more generic

           Returns [Latency,Dir1], both lists

           Parameters
           ----------
           data : dataframe (downsampled long format)

           aoi_criteria: dictionary with keys top_y_max, top_y_min, bottom_y_min, bottom_y_max, top_x_max etc.
                                values are corresponding pixel values

           """

    Dir1=[]
    Latency=[]

    #get list of trials
    trialno=pd.unique(data['TRIAL_INDEX'])

    #iterate over all trials
    for i in trialno:

        this_trial=data[data['TRIAL_INDEX']==i]

        #slice corresponding to this trial
        this_trial=this_trial.reset_index(drop=True)


        #get slices corresponding to target and distracter looking on this trial
        [TargetSlots, DisSlots]=get_target_slots(this_trial, aoi_criteria)


        #determine latency to target and distracter
        if TargetSlots.shape[0]>0:
            target_minslot=min(TargetSlots['TIMESTAMP']-TargetSlots['TRIAL_START_TIME'])
        else:
            target_minslot=np.nan

        if DisSlots.shape[0]>0:
            distracter_minslot=min(DisSlots['TIMESTAMP']-DisSlots['TRIAL_START_TIME'])
        else:
            distracter_minslot=np.nan


        #determine direction of first look
        if np.isnan(target_minslot) and not np.isnan(distracter_minslot):
            Dir1.append(-1)
            Latency.append(distracter_minslot)
        elif not np.isnan(target_minslot) and np.isnan(distracter_minslot):
            Dir1.append(1)
            Latency.append(target_minslot)
        elif np.isnan(target_minslot) and np.isnan(distracter_minslot):
            Dir1.append(np.nan)
            Latency.append(np.nan)
        elif target_minslot<distracter_minslot:
            Dir1.append(1)
            Latency.append(target_minslot)
        elif distracter_minslot < target_minslot:
            Dir1.append(-1)
            Latency.append(distracter_minslot)

        else:
            Dir1.append(np.nan)
            Latency.append(np.nan)


    return [Latency,Dir1]




def get_look_durations(data):
    """
        run after add_LookIndex
        Creates a list of durations of looks for every trial; length of list corresponds to number of looks in that trial
        Uses the LOOK_INDEX column
        Returns LookDur, list of lists: one list of durations for each trial

        Parameters
        ----------
        data : dataframe (downsampled long format)

               """

    sample_duration=data.loc[1,'TIMESTAMP']-data.loc[0,'TIMESTAMP']
    trialno=pd.unique(data['TRIAL_INDEX'])
    LookDur=[]
    for i in trialno:
        this_trial=data[data['TRIAL_INDEX']==i]
        this_trial=this_trial.reset_index(drop=True)
        LDur=this_trial['LOOK'].value_counts()

        LDur=LDur.sort_index()
        LDur=LDur*sample_duration

        LookDur.append(LDur.tolist())

    return LookDur



##################################
#                                #
#  CONVERT TO WIDE FORMAT        #
#                                #
##################################

def wideByTrial(filename, instruct_filename, aoi_criteria, buttonmap):
    """
                  Produce a wide format with one row per trial

                  writes output file adding suffix '_frst'

                        Parameters
                        ----------
                        filename : csv file (Eye Link sample report / downsampled / cleaned / TarDis added/ LOOK added / First, NSamp, Dir1 added)
                        instruct_filename: is a csv file that contains one row per header variable where
                            column 1 is the variable name (corresponding to columns in <filename>
                            column 2 specifies what downsampling does with information in that column
                            'mean' calculates the mean over all entries in the new sample (must be numeric)
                            'first' uses the value of the first entry (goes for string and numeric)
                            'sum' adds up all entries (must be numeric)
                            'any' will find the first nonempty value even if it is not in the first slot (strings and numeric)
                            'maj' will assign the value that the majority of entries fall into

                        aoi_criteria: dictionary with keys top_y_max, top_y_min, bottom_y_min, bottom_y_max, top_x_max etc.
                                values are corresponding pixel values
                        buttonmap: dictionary mapping button values (as in DataViewer output) to target locations -1,1


                        """


    #read data
    fsre=re.search('^(.*).csv$',filename)
    if fsre:
        filestem=fsre.group(1)
    else:
        filestem=filename

    dsdata = pd.read_csv(filename)

    #read instructions
    Instruct=pd.read_csv(instruct_filename)

    #find all unique trials
    trialno=pd.unique(dsdata['TRIAL_INDEX'])

    #which columns to use in the wide output
    ColumnsForWide=Instruct[Instruct['IncludeInWide']==True]

    #columnname was changed
    ColumnsForWide['VarName']=ColumnsForWide['VarName'].replace('RECORDING_SESSION_LABEL', 'SUBJECTNO')
    ColumnsForWide=ColumnsForWide.append({'VarName':'TRIAL_ID', 'SampleAction':'', 'IncludeInWide':True}, ignore_index=True)

    TrialInfoDict={}

    MaxLk=np.array([])
    for tno in trialno:

        trialdata=dsdata[dsdata['TRIAL_INDEX'] == tno]
        trialdata=trialdata.reset_index(drop=True)

        MaxLk=np.append(MaxLk,np.max(trialdata['LOOK']))
        for vn in ColumnsForWide['VarName']:


            if vn in trialdata.columns:

                TrialInfoDict.setdefault(vn,[]).append(trialdata.loc[0,vn])




    TrialInfoWide=pd.DataFrame(TrialInfoDict)


    #get lists with accuracy and RT for each trial and add to data
    [Accuracy, RT]=get_accuracy_rt(dsdata, 'KBRESPONSE', buttonmap)
    TrialInfoWide['Accuracy']=Accuracy
    TrialInfoWide['RT']=RT

    #get lists with latency and direction of first look for each trial and add to data
    [Latency, Dir1]=get_firstlook(dsdata, aoi_criteria)
    TrialInfoWide['Dir1']=Dir1
    TrialInfoWide['Latency']=Latency

    #get target/distracter looking and target proportion
    [Tar,Dis,Tarprop]=get_tardis(dsdata, aoi_criteria)


    #add info columns and write output file
    TrialInfoWide['TarLT']=Tar
    TrialInfoWide['DisLT']=Dis
    TrialInfoWide['TarProp']=Tarprop
    #actually this is about looks, not fixations
    LookDur=get_look_durations(dsdata)


    TrialInfoWide['Dur1']=get_ith_in_list(LookDur,0)
    TrialInfoWide['Dur2']=get_ith_in_list(LookDur,1)
    TrialInfoWide['NumFix']=pd.Series(MaxLk)



    TrialInfoWide.to_csv(filestem+'_wide'+'.csv', index=False)


