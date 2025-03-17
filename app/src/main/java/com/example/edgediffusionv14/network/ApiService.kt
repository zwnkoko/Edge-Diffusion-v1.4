package com.example.edgediffusionv14.network
import com.example.edgediffusionv14.network.models.PromptRequest
import com.example.edgediffusionv14.network.models.RewriteResponse
import com.example.edgediffusionv14.network.models.StatusResponse
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST


interface ApiService {

    @POST("rewrite_prompt")
    fun postData(@Body request: PromptRequest): Call<RewriteResponse>

    @GET("status")
    fun getData(): Call<StatusResponse>

}