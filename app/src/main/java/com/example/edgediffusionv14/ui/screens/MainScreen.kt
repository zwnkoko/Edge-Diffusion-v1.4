package com.example.edgediffusionv14.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.edgediffusionv14.ui.components.ImageDisplay
import com.example.edgediffusionv14.ui.components.PromptField
import com.example.edgediffusionv14.ui.components.SegmentedControl
import com.example.edgediffusionv14.ui.components.StepControl
import com.example.edgediffusionv14.ui.components.StatusCheck
import com.example.edgediffusionv14.ui.viewmodels.DiffusionViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.graphics.Bitmap
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Switch
import androidx.compose.material3.TextField

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    viewModel: DiffusionViewModel = viewModel()
) {
    val seedOption = listOf("x86 Seed 0", "Custom" )
    var selectedSeed by remember { mutableStateOf("Custom") }
    var denoiseSteps by remember { mutableIntStateOf(20) }
    var promptText by remember { mutableStateOf("") }
    var statusProgress by remember { mutableStateOf(listOf<String>("Status...ready")) }
    val surfaceColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f)
    var generatedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var onsiteLLMAvailable by remember { mutableStateOf<Boolean?>(false) }
    var showNegativePrompt by remember { mutableStateOf(false) }
    var negativePromptText by remember { mutableStateOf("") }
    var showErrorDialog by remember { mutableStateOf(false) }
    var customSeed by remember { mutableStateOf("42") }
    val diffusionPipeline = viewModel.diffusionPipeline


    viewModel.checkLlmAvailability { isAvailable ->
        onsiteLLMAvailable = isAvailable
    }

    fun generateImage() {
        if(promptText.isEmpty()){
            showErrorDialog = true
            return
        }

        // To reset UI
        generatedBitmap = null
//        statusProgress = listOf("Status...ready")
        println("Generating image with prompt: $promptText")
        statusProgress = statusProgress + listOf("Encoding prompt...")

        // Launch a coroutine for background processing
        viewModel.viewModelScope.launch {
            // Run background thread to prevent UI thread blocking
            val (encodedPromptData, encodedPromptShape)  = withContext(Dispatchers.IO) {
                diffusionPipeline.encodePrompt(promptText, negativePromptText)
            }
            println("encodedPromptData: $encodedPromptData")
            // Execution returns back to main thread here
            if (encodedPromptData == null || encodedPromptShape == null) {
                statusProgress = statusProgress + listOf("Encoding prompt... failed")
                return@launch
            }
            statusProgress = statusProgress + listOf("Generating image...")
            statusProgress = statusProgress + listOf("Fetching UNET file...")


            // Generate image with progress updates and get the final result
            val bitmap = withContext(Dispatchers.IO) {
                diffusionPipeline.generateImage(
                    encodedPrompt = encodedPromptData,
                    encodedPromptShape = encodedPromptShape,
                    numSteps = denoiseSteps,
                    progressCallback = { step: Int, totalSteps: Int ->
                        // Update UI on the main thread
                        launch(Dispatchers.Main) {
                            statusProgress = if(step == -1 && totalSteps == -1){
                                statusProgress + "VAE - Converting latent space to image..."
                            } else{
                                statusProgress + "Generating step ${step}/${totalSteps}"
                            }

                        }
                    },
                    randomSeed = if (selectedSeed != "Custom") null else customSeed.toLong()
                )
            }

            generatedBitmap = bitmap

        }
    }

    fun rewritePrompt(){
        statusProgress = statusProgress + listOf("Rewriting prompt...")
        viewModel.makeApiRequest(promptText) { result ->
            if (result != null) {
                // Use the rewritten prompt here
                println("Got rewritten prompt: $result")
                statusProgress = statusProgress + listOf("Prompt rewritten")
                promptText = result
            } else {
                // Handle error case
                println("Failed to get rewritten prompt")
                statusProgress = statusProgress + listOf("Prompt rewrite failed")
            }
        }

    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "Edge Diffusion v1.4",
                        style = MaterialTheme.typography.titleLarge
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    titleContentColor = MaterialTheme.colorScheme.onPrimary
                )
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .background(MaterialTheme.colorScheme.background),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.Start
            ) {
                StatusCheck(
                    label = "On-site LLM Availability",
                    isActive = onsiteLLMAvailable,
                )
            }
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.End
            ) {
                //  Denoise loop selector
                Card(
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(containerColor = surfaceColor),
                    modifier = Modifier
                        .border(
                            width = 1.dp,
                            color = MaterialTheme.colorScheme.outlineVariant,
                            shape = RoundedCornerShape(24.dp)
                        )
                        .height(48.dp)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxHeight()
                            .padding(horizontal = 12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        StepControl(
                            label = "Denoise Steps",
                            denoisingSteps = denoiseSteps,
                            onValueChange = { newValue -> denoiseSteps = newValue },
                        )
                    }
                }


                Spacer(modifier = Modifier.weight(1f))

                // Seed mode selector
                Card(
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(containerColor = surfaceColor),
                    modifier = Modifier
                        .border(
                            width = 1.dp,
                            color = MaterialTheme.colorScheme.outlineVariant,
                            shape = RoundedCornerShape(24.dp)
                        )
                        .height(48.dp)
                ) {
                    SegmentedControl(
                        options = seedOption,
                        selectedOption = selectedSeed,
                        onValueChange = { option -> selectedSeed = option },
                        textStyle = MaterialTheme.typography.bodySmall
                    )
                    // Show text field for custom seed when "Seed 0" is selected

                }
            }


            // Image display area
            Card(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(16.dp),
                shape = RoundedCornerShape(24.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
                colors = CardDefaults.cardColors(containerColor = surfaceColor)
            ) {
                if (generatedBitmap == null) {
                    Row (
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.Center,
                    ) {
                        Text(
                            text = "Progress Status",
                            style = MaterialTheme.typography.titleLarge,
                            modifier = Modifier.padding(16.dp)

                        )
                    }
                    Box( modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .verticalScroll(rememberScrollState())
                    ){
                        Text(
                            text = statusProgress.joinToString("\n"),
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.padding(16.dp)
                        )
                    }
                } else {
                    ImageDisplay(
                        modifier = Modifier.fillMaxSize(),
                        placeHolderText = "Generated image will appear here",
                        bitmap = generatedBitmap
                    )
                }
            }
            Row (
                verticalAlignment = Alignment.CenterVertically ,

                modifier = Modifier.padding(horizontal = 28.dp)){
                if (selectedSeed == "Custom") {
                        Text(
                            text = "Enter seed: ",
                            style = MaterialTheme.typography.titleMedium,
                        )
                        TextField(
                            value = customSeed,
                            onValueChange = { value ->
                                // Only allow numeric input
                                if (value.isEmpty() || value.all { it.isDigit() }) {
                                    customSeed = value
                                }
                            },
                            modifier = Modifier.width(60.dp).height(45.dp),
                            textStyle = MaterialTheme.typography.bodySmall,
                            singleLine = true,
                            placeholder = {
                                Text(
                                    "Seed",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            },

                        )
                    Spacer(modifier = Modifier.width(16.dp))

                }
                Text(
                    text = "Negative prompt: ",
                    style = MaterialTheme.typography.titleMedium,

                )
                Switch(
                    checked = showNegativePrompt,
                    onCheckedChange = { showNegativePrompt = it },

                )
            }

            // Prompt input
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                shape = RoundedCornerShape(28.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
                colors = CardDefaults.cardColors(containerColor = surfaceColor)
            ) {


                if(! showNegativePrompt){
                    // positive prompt field
                    PromptField(
                        modifier = Modifier.padding(12.dp),
                        rewriteBtn = true,
                        rewriteBtnStatus = onsiteLLMAvailable == true,
                        placeHolderText = "Describe image to generate",
                        promptText = promptText,
                        onPromptChange = { newPrompt -> promptText = newPrompt },
                        onRewriteClick = { rewritePrompt() },
                        onSubmit =  { generateImage() }
                    )
                } else {
                    // negative prompt field
                    PromptField(
                        modifier = Modifier.padding(12.dp),
                        rewriteBtn = false,
                        rewriteBtnStatus = false,
                        placeHolderText = "Enter things to exclude from image",
                        promptText = negativePromptText,
                        onPromptChange = { negativePrompt -> negativePromptText = negativePrompt },
                        onRewriteClick = {},
                        onSubmit =  { generateImage() }
                    )
                }





            }
        }
        if (showErrorDialog) {
            androidx.compose.material3.AlertDialog(
                onDismissRequest = { showErrorDialog = false },
                title = { Text("Error") },
                text = { Text("Please enter a prompt before generating an image") },
                confirmButton = {
                    androidx.compose.material3.TextButton(
                        onClick = { showErrorDialog = false }
                    ) {
                        Text("OK")
                    }
                }
            )
        }
    }
}