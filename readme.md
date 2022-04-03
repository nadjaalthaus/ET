EYE TRACKING DATA PREPROCESSING FOR GROWTH CURVE ANALYSIS 

(C) Nadja Althaus

These scripts were developed for the eye tracking time course analysis (multi-level models/growth curve)
presented in **Althaus, Kotzor, Schuster & Lahiri (2022)**. Distinct orthography boosts morphophonological discrimination: 
Vowel raising in Bengali verb inflections. *Cognition, 222*, 104963.  
They are intended for "sample report" exports from EyeLink/DataViewer.  Exported files need to be placed in subdirectory ./Data. 

The experiment is a visual world paradigm with two visual targets (match=target vs. nonmatch=distracter). On each trial, 
the participant hears a fragment of a word and has to decide which of two words (presented visually) the fragment corresponds to.
The analysis for which scripts are provided for is the time course analysis using (multilevel) growth curves, 
i.e. fitting polynomials to the looking patterns over time using glmer (cf. Mirman, 2014).
The fundamental insight is that participants will fixate longer on the target than on the distracter, particularly on the
very first fixation -- provided they can discriminate between the two. In particular therefore the curves for targets on which
the target was fixated first vs. those on which the distracter was fixated first diverge rapidly for easily decidable trials,
but less so for trials where the decision is harder, and not at all if there is no correct answer (i.e. both visual items
are potential matches as in the No Diff condition in the paper). Here we provide code for the preprocessing (in Python) and
growth curve modelling (in R) of such gaze data. As detailed in the paper, we modeled the effects of direction of first fixation vs.
effects of condition separately, due to non-convergence of models containing both. The files Dir1Effects.Rmd and ConditionEffects_MatchFirst.Rmd 
demonstrate these, respectively (for Distracter-First trials, the procedure is equivalent). 

The scripts will have to be adapted slightly for new studies, in particular if using different settings/parameters.
