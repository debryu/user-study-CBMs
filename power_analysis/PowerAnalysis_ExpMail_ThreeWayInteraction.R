# ******** PREPARE R SESSION ******** ----

# This part is heavily based on the scripts for the analyses of Sebben & Ullrich (2021).
# Can conditionals explain explanations? A modus ponens model of B because A. Cognition
# See for example https://osf.io/wnczr/

# WARNING: 
# The following line of code will remove all objects from the workspace. 
# Please consider saving your workspace.
rm(list=ls())

# WARNING: 
# All currently loaded packages will be detached and unloaded.
# Packages needed to run this script will be loaded. 
# Missing packages will be downloaded and installed.
# Also, the working directory will be automatically 
# set to the location of this script.
# If you did not alter the structure of the folder containing this script 
# and the data, in most cases you should be able to run this script without any 
# manual adjustments in a current version of RStudio.

# Detach and unload packages ----
if(!is.null(sessionInfo()$otherPkgs)){
  invisible(lapply(paste('package:', names(sessionInfo()$otherPkgs), sep = ""), 
                   detach, character.only = T, unload = T))
}

# Download and install missing packages ----
requiredPackages <- c(
  
  "rio",
  "lme4",
  "simr",
  "car",
  "tidyverse"
)

missingPackages <- requiredPackages[!requiredPackages 
                                    %in% installed.packages()[ ,"Package"]]

if(length(missingPackages) > 0){
  install.packages(missingPackages)
}

# Load required packages ----
invisible(lapply(requiredPackages, require, character.only = T))

# Set working directory to source file location ----
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) 

# Set seed to ensure replicability of the results ----
set.seed(1234)


remove(list = ls())


# ******** Results ******** ----

# To see the results without running the full script, run this

load("Results_SampleSize_WithNone_340.RData")
resultsPower # Each value referes to one of the sample size tested (see possibleSampleSize below)

# Parameters used:
# nRepetitions = 1000
# possibleNumberTrials = c(10)
# possibleSampleSize = c(320, 340, 360, 380)
# seed = 1234



# ******** Sample Size Estimation ******** ----

# Run model on pilot data to get estimate of effect sizes

dataPilot <- import("dataLongComplete.xlsx")

dataPilot$ExperimentalCondition = factor(dataPilot$ExperimentalCondition, levels = c("BlackBox", "None"))
contrasts(dataPilot$ExperimentalCondition) = contr.sum(2)

dataPilot$GroundTruth = factor(dataPilot$GroundTruth, levels = c("fraudulent", "legitimate"))
contrasts(dataPilot$GroundTruth) = contr.sum(2)

dataPilot$IsModelAnswerCorrect = factor(dataPilot$IsModelAnswerCorrect, levels = c("0", "1"))
contrasts(dataPilot$IsModelAnswerCorrect) = contr.sum(2)


pilotMixedModel <- glmer(IsParticipantAnswerCorrect ~ ExperimentalCondition * GroundTruth * IsModelAnswerCorrect + 
                                 (1 | ID) + (1 | StimID),
                               family = binomial(link="logit"),
                               control = glmerControl(optimizer = "bobyqa"), nAGQ = 1,
                               data = dataPilot)

pilotResults <- summary(pilotMixedModel)

vc <- VarCorr(pilotMixedModel)
var_ID    <- as.numeric(VarCorr(pilotMixedModel)$StimID) # set the same as StimID because, for insufficient data, it cannot be estimated
var_StimID <- as.numeric(VarCorr(pilotMixedModel)$StimID)






# A priori power analysis: sample to detect the effect of interaction between valence of the outcome experienced and age group on information seeking behavior

# Script based on
# Green, P. and MacLeod, C.J. (2016), SIMR: an R package for power analysis of generalized linear mixed models by simulation. Methods Ecol Evol, 7: 493-498. https://doi.org/10.1111/2041-210X.12504
# Kumle, L., Võ, M.LH. & Draschkow, D. Estimating power in (generalized) linear mixed models: An open introduction and tutorial in R. Behav Res 53, 2528–2543 (2021). https://doi.org/10.3758/s13428-021-01546-0

# Magnitude of effect sizes determined using https://www.escal.site/

computeSampleSize <- function(nRepetitions, numberTrials, possibleSampleSize, seed){

  set.seed(seed)
  
  powerResults = vector(length = length(possibleSampleSize)[1])
  failedIterations = vector(length = length(possibleSampleSize)[1])
  
  for(m in 1:length(possibleNumberTrials)) {
      for(i in 1:length(possibleSampleSize)){
        
        pValuesRepository = vector(length = nRepetitions[1])
        failedIterationsRepository = vector(length = nRepetitions[1])
        
        for(o in 1:nRepetitions){
          
          #### ********** Create artificial dataset ********** ####
          
          # Subject IDs
          subject_ID <- rep(1:possibleSampleSize[i], times=1, each=possibleNumberTrials[m])
          
          # Experimental condition (between)
          experimentalCondition <- rep(c("None", "BlackBox", "CBM_Fixed", "CBM_Interactive"), each = (possibleSampleSize[i]/4)*possibleNumberTrials[m])
          
          # Ground truth label (within)
          groundTruthLabel <- rep(c("Legitimate", "Fraudulent"), each = possibleNumberTrials[m]/2, times = possibleSampleSize[i])
          
          # Model prediction (within; Considering a model with 75% accuracy)
          accuracy = 0.8
          modelPrediction <- rep(
            c(rep("Correct", each = round(possibleNumberTrials[m]*accuracy/2)), 
              rep("Wrong", each = round(possibleNumberTrials[m]*(1-accuracy)/2))), 
            times = possibleSampleSize[i]*2)
          
          # Model concepts (within)
          modelConcepts <- rep(
            c(rep("Correct", each = 2), 
              rep("Wrong", each = 2),
              rep("Mixed", each = 1),
              rep("Correct", each = 3), 
              rep("Wrong", each = 1),
              rep("Mixed", each = 1)), 
            times = possibleSampleSize[i])
          
          # Create dataset
          artificial_data <- data.frame(Subject = subject_ID, 
                                        ExperimentalCondition = experimentalCondition,
                                        CorrectClassification = groundTruthLabel,
                                        ModelClassification = modelPrediction,
                                        ModelConcepts = modelConcepts)
          
          # ID and Difficulty
          experimentalConditions = c("None", "BlackBox", "CBM_Fixed", "CBM_Interactive")
          
          for(a in experimentalConditions) {
            
            
            urn_Fraudulent_ConceptCorrect_PredictionCorrect = replicate(possibleSampleSize[i], sample(c("1", "2", "3", 
                                                                                                        "4", "5", "6")), 
                                                                        simplify = FALSE) |> unlist()
            rows <- artificial_data$ExperimentalCondition == a &
              artificial_data$CorrectClassification == "Fraudulent" &
              artificial_data$ModelConcepts == "Correct" &
              artificial_data$ModelClassification == "Correct"
            artificial_data[rows, "StimID"] <- urn_Fraudulent_ConceptCorrect_PredictionCorrect[1:sum(rows)]

            
            urn_Fraudulent_ConceptWrong_PredictionCorrect = replicate(possibleSampleSize[i], sample(c("7", "8", "9", 
                                                                                                      "10", "11", "12")), 
                                                                      simplify = FALSE) |> unlist()
            rows <- artificial_data$ExperimentalCondition == a &
              artificial_data$CorrectClassification == "Fraudulent" &
              artificial_data$ModelConcepts == "Wrong" &
              artificial_data$ModelClassification == "Correct"
            artificial_data[rows, "StimID"] <- urn_Fraudulent_ConceptWrong_PredictionCorrect[1:sum(rows)]
            
            
            urn_Fraudulent_ConceptMixed_PredictionWrong = replicate(possibleSampleSize[i], sample(c("13", "14", 
                                                                                                    "15")), 
                                                                    simplify = FALSE) |> unlist()
            rows <- artificial_data$ExperimentalCondition == a &
              artificial_data$CorrectClassification == "Fraudulent" &
              artificial_data$ModelConcepts == "Mixed" &
              artificial_data$ModelClassification == "Wrong"
            artificial_data[rows, "StimID"] <- urn_Fraudulent_ConceptMixed_PredictionWrong[1:sum(rows)]
            
            
            urn_Legitimate_ConceptCorrect_PredictionCorrect = replicate(possibleSampleSize[i], sample(c("17", "18", 
                                                                                                        "19", "20")), 
                                                                        simplify = FALSE) |> unlist()
            rows <- artificial_data$ExperimentalCondition == a &
              artificial_data$CorrectClassification == "Legitimate" &
              artificial_data$ModelConcepts == "Correct" &
              artificial_data$ModelClassification == "Correct"
            artificial_data[rows, "StimID"] <- urn_Legitimate_ConceptCorrect_PredictionCorrect[1:sum(rows)]
            
            
            urn_Legitimate_ConceptWrong_PredictionCorrect = replicate(possibleSampleSize[i], sample(c("21", "22", "23", "24",
                                                                                                      "25", "26", "27", "28")), 
                                                                      simplify = FALSE) |> unlist()
            rows <- artificial_data$ExperimentalCondition == a &
              artificial_data$CorrectClassification == "Legitimate" &
              artificial_data$ModelConcepts == "Wrong" &
              artificial_data$ModelClassification == "Correct"
            artificial_data[rows, "StimID"] <- urn_Legitimate_ConceptWrong_PredictionCorrect[1:sum(rows)]
            
            
            urn_Legitimate_ConceptMixed_PredictionWrong = replicate(possibleSampleSize[i], sample(c("30",
                                                                                                    "31", "32")), 
                                                                    simplify = FALSE) |> unlist()
            rows <- artificial_data$ExperimentalCondition == a &
              artificial_data$CorrectClassification == "Legitimate" &
              artificial_data$ModelConcepts == "Mixed" &
              artificial_data$ModelClassification == "Wrong"
            artificial_data[rows, "StimID"] <- urn_Legitimate_ConceptMixed_PredictionWrong[1:sum(rows)]
            
          }
            
          
          easyImages = c("1", "2", "3",
                         "7", "8", "9", 
                         "13", "15", 
                         "17", "18", 
                         "21", "22", "23", "24",
                         "29")
          
          conceptsCorrectImages = c("1", "2", "3", "4", "5", "6",
                                    "13", "14",
                                    "17", "18", "19", "20",
                                    "29", "30")
          
          
          artificial_data["Difficulty"] <- ifelse(artificial_data$StimID %in% easyImages, "Easy", "Difficult")
          artificial_data["ModelConcepts_Final"] <- ifelse(artificial_data$StimID %in% conceptsCorrectImages, "Correct", "Wrong")

          artificial_data$ExperimentalCondition = factor(artificial_data$ExperimentalCondition, levels = c("CBM_Fixed", "CBM_Interactive", "BlackBox", "None"))
          contrasts(artificial_data$ExperimentalCondition) = contr.sum(4)
          
          artificial_data$CorrectClassification = factor(artificial_data$CorrectClassification, levels = c("Fraudulent", "Legitimate"))
          contrasts(artificial_data$CorrectClassification) = contr.sum(2)
          
          artificial_data$ModelClassification = factor(artificial_data$ModelClassification, levels = c("Wrong", "Correct"))
          contrasts(artificial_data$ModelClassification) = contr.sum(2)
          
          artificial_data$ModelConcepts_Final = factor(artificial_data$ModelConcepts_Final, levels = c("Wrong", "Correct"))
          contrasts(artificial_data$ModelConcepts_Final) = contr.sum(2)
          
          artificial_data$Difficulty = factor(artificial_data$Difficulty, levels = c("Difficult", "Easy"))
          contrasts(artificial_data$Difficulty) = contr.sum(2)
          
          artificial_data$StimID = as.factor(artificial_data$StimID)
          
          artificial_data$Subject = as.factor(artificial_data$Subject)
          
          
          #### ********** Model parameters ********** ####
          
          #### Specify beta coefficients for fixed effects ####
          
          # DV (to be computed): 0 = Participant's classification is wrong; 1 = Participant's classification is correct)
          # Effect size converter: https://www.escal.site/
          # logOdds = 0.64 -> OR = 1.89 (d = 0.35)
          # logOdds = 0.54 -> OR = 1.72 (d = 0.30)
          # logOdds = 0.45 -> OR = 1.57 (d = 0.25)
          # logOdds = 0.36 -> OR = 1.44 (d = 0.20)
          # logOdds = 0.27 -> OR = 1.31 (d = 0.15)
          
          # All effects divided by three to be conservative

          fixed_effects <-  c(
            
            # Intercept
            qlogis(0.80),   # Overall accuracy we could expect in the three supported conditions)
            
            
            # Main effects
            abs(round(pilotResults$coefficients[2, 1], 2))/1.5,  # Main effect experimental condition (BlackBox --> improvement compared to overall mean)
            abs(round(pilotResults$coefficients[2, 1], 2))/1.5,  # Main effect experimental condition (CBM_Fixed --> improvement compared to overall mean)
            abs(round(pilotResults$coefficients[2, 1], 2))/1.5,  # Main effect experimental condition (CBM_Interactive --> improvement compared to overall mean, similar to CBM_Fixed)
            round(pilotResults$coefficients[3, 1], 2)/1.5,  # Main effect correct classification (Fraudulent --> maybe more likely to be classified erroneously as legitimate)
            round(pilotResults$coefficients[4, 1], 2)/1.5,  # Main effect model classification (Wrong --> wrong classifications are more likely to mislead participants)
            
            
            # Two-way interactions
            abs(round(pilotResults$coefficients[5, 1], 2))/1.5,  # experimental condition * correct classification (BlackBox:Fraudulent --> improvement compared to overall mean)
            abs(round(pilotResults$coefficients[5, 1], 2))/1.5,  # experimental condition * correct classification (CBM_Fixed:Fraudulent --> improvement compared to overall mean)
            abs(round(pilotResults$coefficients[5, 1], 2))/1.5,  # experimental condition * correct classification (CBM_Interactive:Fraudulent --> improvement compared to overall mean)
            abs(round(pilotResults$coefficients[6, 1], 2))/1.5,  # experimental condition * model classification (BlackBox:Wrong --> improvement compared to overall mean)
            abs(round(pilotResults$coefficients[6, 1], 2))/1.5,  # experimental condition * model classification (CBM_Fixed:Wrong --> improvement compared to overall mean)
            abs(round(pilotResults$coefficients[6, 1], 2))/1.5,  # experimental condition * model classification (CBM_Interactive:Wrong --> improvement compared to overall mean)
            round(pilotResults$coefficients[7, 1], 2)/1.5,  # Correct classification* model classification (Fraudulent:Wrong --> impaired performance compared to overall mean)
            
            # Three-way interactions
            abs(round(pilotResults$coefficients[8, 1], 2))/1.5,  # experimental condition * correct classification * model classification (BlackBox:Fraudulent:Wrong --> improvement compared to overall mean)
            abs(round(pilotResults$coefficients[8, 1], 2))/1.5,  # experimental condition * correct classification * model classification (CBM_Fixed:Fraudulent:Wrong --> improvement compared to overall mean)
            abs(round(pilotResults$coefficients[8, 1], 2))/1.5  # experimental condition * correct classification * model classification (CBM_Interactive:Fraudulent:Wrong --> improvement compared to overall mean)
          
          )
  
          #### Specify random intercept variance ####
          random_variance = list(round(var_ID, 2), round(var_StimID, 2))        # Random intercept for subjects, distribution, and scenario (test different possible values)
          
          
          #### ********** Create model and dependent variable ********** ####
          
          # create GLMM
          artificial_glmer <- makeGlmer(formula = DV ~ ExperimentalCondition * CorrectClassification * ModelClassification + 
                                          (1 | Subject) + (1 | StimID),
                                        family = "binomial", fixef = fixed_effects,
                                        VarCorr = random_variance, data = artificial_data)
          
          artificial_data$DV = artificial_glmer@resp$`.->y`
          
          
          #### ********** Power analysis ********** ####
          
          # Function to handle warning and errors: if it happens, drop that result from the power computation
          
          handleFailureConverge <- function() {
            fitModel <- tryCatch(
              
              ########################################################
              # Try part: define the expression(s) you want to "try" #
              ########################################################
              
              {

                artificial_mixedModel <- glmer(DV ~ ExperimentalCondition * CorrectClassification * ModelClassification + 
                                                 (1 | Subject) + (1 | StimID),
                                               family = binomial(link="logit"),
                                               control = glmerControl(optimizer = "bobyqa"), nAGQ = 1,
                                               data = artificial_data)
                
                # The following  code is used to assess how convergence differs using different optimizer. 
                # It showed that fixed effects were similar across optimizer, so the estimations should be reliable regardless of the optimizer used
                
                # all_fits <- allFit(artificial_mixedModel)
                # summary(all_fits)
                
                summary(artificial_mixedModel)
                
                aov = Anova(artificial_mixedModel, type = "III")

                pValuesRepository[o] <<- aov$`Pr(>Chisq)`[8]
                failedIterationsRepository[o] <<- 0
                
              },
              
              ########################################################################
              # Condition handler part: define how you want conditions to be handled #
              ########################################################################
              
              # Handler when a warning occurs:
              warning = function(cond) {
                message(paste("Warning was fired: value set to 1 to be conservative"))
                pValuesRepository[o] <<- 1
                failedIterationsRepository[o] <<- 1
                
              },
              
              # Handler when an error occurs:
              error = function(cond) {
                message(paste("Error was fired: value set to 1 to be conservative"))
                pValuesRepository[o] <<- 1
                failedIterationsRepository[o] <<- 1
              },
  
  
              ###############################################
              # Final part: define what should happen AFTER #
              # everything has been tried and/or handled    #
              ###############################################
  
              finally = {
                #message(paste("Proceding to the next iteration"))
              }
            )    
            return(fitModel)
          }
          
          handleFailureConverge()
          print(paste0("o: ", o))
        }
          
        pValuesRepository <- na.omit(pValuesRepository)
        powerThisSample = sum(pValuesRepository < .05)/length(pValuesRepository)
        failedIterationsThisSample = sum(failedIterationsRepository)
          
        powerResults[i] = powerThisSample
        failedIterations[i] = failedIterationsThisSample

        # Print these messages just to check progression of the simulation process
        print(paste0("m: ", m))
        print(paste0("i: ", i))
        print(paste0("Succesful iterations for this sample: ", nRepetitions - failedIterationsThisSample))
  
        pValuesRepository = vector()
        failedIterationsRepository = vector()
        
      }
  }
  
  resultsPower = rbind(powerResults)
  resultsFailedIterations = rbind(failedIterations)
  return(list(resultsPower = resultsPower, resultsFailedIterations = resultsFailedIterations))

}


nRepetitions = 300
possibleNumberTrials = c(10)
possibleSampleSize = c(320, 340, 360, 380)
seed = 1234

resultsSampleSize <- computeSampleSize(nRepetitions, numberTrials, possibleSampleSize, seed)

resultsPower = resultsSampleSize$resultsPower
iterationsFailed = resultsSampleSize$resultsFailedIterations

save.image("Results_SampleSize_WithNone_340.RData")





