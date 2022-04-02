##################################################################################################################
#                                                                                                                #
# EYE TRACKING DATA PREPROCESSING FOR GROWTH CURVE ANALYSIS                                                      #
# (C) NADJA ALTHAUS, 2022 https://github.com/nadjaalthaus                                                        #
#                                                                                                                #
# Developed for preprocessing EyeLink sample reports for growth curve analysis                                   #
# for Althaus, Kotzor, Schuster & Lahiri (2022). Distinct orthography boosts morphophonological discrimination:  #
# Vowel raising in Bengali verb inflections. Cognition, 222, 104963.                                             #
#  https://doi.org/10.1016/j.cognition.2021.104963                                                               #
#                                                                                                                #
##################################################################################################################



import PrepGaze as pg
import os, re
import pandas as pd







#directory with raw files
datadir='./Data'
#info only: data in this folder were exported from DataViewer as sample reports with prefix sr_VRFC
#this has to be manually adjusted in regex further below
#dataprefix='sr_VRFC'



#screen 1024x1280
#centre 512x640
#words were displayed vertically above / below central fixation cross
#use 100 px from 512 as AOI, i.e. y coordinate <412 or >612


#file holding instructions about how to treat individual variables during downsampling, e.g. use first value / use average value etc.
Instructionsfile=datadir+'/Instructions.csv'

#initialise data frame that will hold all data (wide format), one trial per line
WData=pd.DataFrame()

#switches to easily enforce re-doing individual preprocessing steps;
#if set to False then these steps are only run if a file with the corresponding name does not yet exist in datadir
force_downsample=True
force_clean=True
force_tardis=True #tardis: tar / dis - labels samples as falling on target or distracter
force_look=True
force_first=True
force_wide=True


#criteria for assignment of a sample to an area of interest
#has to be adjusted for individual experimental setup
#used by add_TarDisColumn
aoi_criteria={}
aoi_criteria['top_y_max']=412 #must be lower than this threshold
aoi_criteria['top_y_min']=212
aoi_criteria['bottom_y_min']=612#must be higher than this threshold
aoi_criteria['bottom_y_max']=812
aoi_criteria['top_x_min']=480
aoi_criteria['top_x_max']=800
aoi_criteria['bottom_x_min']=480
aoi_criteria['bottom_x_max']=800


#dictionary for correct assignment of manual response buttons
#has to be adjusted for individual experimental setup
buttonmap={}
#buttons here are 6 and 7, corresponding to 'down' and 'up', which is target locs 1, -1
buttonmap={}
buttonmap[6]=1
buttonmap[7]=-1



#make a list of all files in datadir
D=os.listdir(datadir)
D.sort()

#iterate through list of files
for entry in D:

    #check whether the file is a sample export from the relevant study, e.g. sr_VRFC1.csv
    #this regex has to be manually adjusted to match data from the individual study
    r=re.match('^(sr\_VRFC[0-9]*).csv$',entry)
    if r:
        print(entry)
        # keep the filestem from the regex to construct output filenames
        stem = r.group(1)

        # The final experimental procedure starts at participant 19 because we added experimenter-triggered trial start
        #after realising that subjects did not always fixate the central fixation cross
        numberselect=re.search('sr\_[a-zA-Z]+([0-9]+).csv$',entry)


        if numberselect:
            filenumstr=numberselect.group(1)
            filenum=int(filenumstr)

        if filenum<19:
            continue


        # downsampling
        if force_downsample or not os.path.isfile(datadir+'/'+stem+'_ds.csv'):
            print('downsample...')
            pg.downsample(datadir+'/'+stem+'.csv', 20, Instructionsfile)
        else:
            print('downsampled version exists, skip downsample...')

        #cleaning the file
        if force_clean or not os.path.isfile(datadir+'/'+stem+'_ds_cln.csv'):
            print('clean...')
            pg.clean_ds_data(datadir+'/'+stem+'_ds.csv')
        else:
            print('clean version exists, skip cleaning...')

        #add AOI info (target/distracter)
        if force_tardis or not os.path.isfile(datadir+'/'+stem+'_ds_cln_tardis.csv'):
            print('target and distracter labelling...')
            pg.add_TarDisColumn(datadir+'/'+stem+ '_ds_cln.csv', aoi_criteria)
        else:
            print('tardis version exists, skip tardis...')

        #group into numbered[CHECK] 'looks': continuous set of samples in the same AOI (with maximum gap maxgap, here 200 ms)
        if force_look or not os.path.isfile(datadir+'/'+stem+'_ds_cln_tardis_lk.csv'):
            print('add looks...')
            pg.add_LookIndex(datadir+'/'+stem+ '_ds_cln_tardis.csv', maxgap=200,aoicol='TarDis')
        else:
            print('version with LOOK exists, skip adding...')

        #For time course analysis on the basis of direction of first look: label each look to indicate whether it falls on the item that
        #was first looked at, will be used to determine odds ratio for growth curve analysis in R
        if force_first or not os.path.isfile(datadir+'/'+stem+'_ds_cln_tardis_lk_frst.csv'):
            print('add oddsratio columns...')
            print(datadir+'/'+stem+'_ds_cln_tardis_lk_frst.csv')
            pg.add_OddsRatioInfo(datadir+'/'+stem+ '_ds_cln_tardis_lk.csv')
        else:
            print(datadir+'/'+stem+'_ds_cln_tardis_lk_frst.csv')
            print('version with oddsratio exists, skip oddsratio...')




        #create a wide format version of the data
        if force_wide or not os.path.isfile(datadir+'/'+stem+'_ds_cln_tardis_lk_frst_wide.csv'):
            print('wide format...')
            pg.wideByTrial(datadir+'/'+ stem +'_ds_cln_tardis_lk_frst.csv', Instructionsfile, aoi_criteria, buttonmap)
        else:
            print('wide format exists, skip formatting...')


        #append to data frame that contains all wide data
        wdata=pd.read_csv(datadir+'/'+ stem +'_ds_cln_tardis_lk_frst_wide.csv')
        if WData.shape[0]==0:
            WData=wdata
        else:
            print('appending...')
            WData=WData.append(wdata, ignore_index=True)

#WData contains all data, one trial per line
print('***')
print('Total number of subjects:'+str(len(pd.unique(WData['SUBJECTNO']))))
print('***')
print('Strip off practice trials ...')
WData=WData[WData['condition']!='Practice']
WData.reset_index(drop=True)
print('Number of trials for analysis: '+ str(WData.shape[0]))
WData.to_csv(datadir+'/VRFCwide.csv')
