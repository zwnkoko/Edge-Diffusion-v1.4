package com.example.edgediffusionv14.network.models

data class PromptRequest(val prompt: String)

data class RewriteResponse(val rewrite: String)

data class StatusResponse(val status: Boolean)