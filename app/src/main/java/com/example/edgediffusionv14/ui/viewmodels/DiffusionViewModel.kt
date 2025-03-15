package com.example.edgediffusionv14.ui.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import com.example.edgediffusionv14.diffusion.models.MinimalCLIPTokenizer
import com.example.edgediffusionv14.diffusion.DiffusionPipeline
import com.example.edgediffusionv14.diffusion.models.PNDMScheduler

class DiffusionViewModel(application: Application) : AndroidViewModel(application) {

    // Lazily initialize tokenizer to avoid creating it until needed
    val tokenizer by lazy {
        MinimalCLIPTokenizer(
            context = getApplication<Application>().applicationContext,
            vocabFileName = "diffusion/vocab.json",
            mergesFileName = "diffusion/merges.txt"
        )
    }

    val scheduler by lazy {
        PNDMScheduler(
            betaSchedule = "scaled_linear",
            betaStart = 0.00085,
            betaEnd = 0.012,
            timestepSpacing = "leading",
            stepsOffset = 1,
            skipPrkSteps = true,
            numTrainTimesteps = 1000,
            predictionType = "epsilon"
        )
    }

    val diffusionPipeline by lazy {
        DiffusionPipeline(
            context = getApplication<Application>().applicationContext,
            tokenizer = tokenizer,
            scheduler = scheduler,
        )
    }
    // Other state and functions for app...
}