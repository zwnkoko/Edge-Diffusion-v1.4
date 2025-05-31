package com.example.edgediffusionv14.ui.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import com.example.edgediffusionv14.diffusion.models.MinimalCLIPTokenizer
import com.example.edgediffusionv14.diffusion.DiffusionPipeline
import com.example.edgediffusionv14.diffusion.models.PNDMScheduler
import com.example.edgediffusionv14.network.ApiClient
import com.example.edgediffusionv14.network.models.PromptRequest
import com.example.edgediffusionv14.network.models.RewriteResponse
import com.example.edgediffusionv14.network.models.StatusResponse
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

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

    fun makeApiRequest(prompt: String, callback: (String?) -> Unit) {
        val request = PromptRequest(prompt)

        ApiClient.apiService.postData(request).enqueue(object : Callback<RewriteResponse> {
            override fun onResponse(call: Call<RewriteResponse>, response: Response<RewriteResponse>) {
                if (response.isSuccessful) {
                    val rewrittenPrompt = response.body()?.rewrite
                    println("API Response: $rewrittenPrompt")
                    callback(rewrittenPrompt)
                } else {
                    println("API Error: ${response.code()}")
                    callback(null)
                }
            }

            override fun onFailure(call: Call<RewriteResponse>, t: Throwable) {
                println("Network Error: ${t.message}")
                callback(null)
            }
        })
    }

    fun checkLlmAvailability(callback: (Boolean) -> Unit) {
        ApiClient.apiService.getData().enqueue(object : Callback<StatusResponse> {
            override fun onResponse(call: Call<StatusResponse>, response: Response<StatusResponse>) {
                if (response.isSuccessful) {
                    val isAvailable = response.body()?.status == true
                    println("LLM Availability: $isAvailable")
                    callback(isAvailable)
                } else {
                    println("LLM Status Error: ${response.code()}")
                    callback(false)
                }
            }

            override fun onFailure(call: Call<StatusResponse>, t: Throwable) {
                println("LLM Status Check Failed: ${t.message}")
                callback(false)
            }
        })
    }

    // Other state and functions for app...

}