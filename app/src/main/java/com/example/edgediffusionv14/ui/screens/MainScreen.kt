package com.example.edgediffusionv14.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
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
import com.example.edgediffusionv14.ui.viewmodels.DiffusionViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.graphics.Bitmap

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    viewModel: DiffusionViewModel = viewModel()
) {
    val deviceOptions = listOf("CPU", "GPU")
    var selectedProcessor by remember { mutableStateOf("CPU") }
    var denoiseSteps by remember { mutableIntStateOf(20) }
    var promptText by remember { mutableStateOf("") }
    var statusProgress by remember { mutableStateOf(listOf<String>("Status...ready")) }
    val surfaceColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f)
    var generatedBitmap by remember { mutableStateOf<Bitmap?>(null) }

    val diffusionPipeline = viewModel.diffusionPipeline

    fun generateImage() {
        println("Generating image with prompt: $promptText")
        statusProgress = statusProgress + listOf("Encoding prompt...")

        // Launch a coroutine for background processing
        viewModel.viewModelScope.launch {
            // Run background thread to prevent UI thread blocking
            val (encodedPromptData, encodedPromptShape)  = withContext(Dispatchers.IO) {
                diffusionPipeline.encodePrompt(promptText)
            }

            // Execution returns back to mainthread here
            if (encodedPromptData == null || encodedPromptShape == null) {
                statusProgress = statusProgress + listOf("Encoding prompt... failed")
                return@launch
            }
            statusProgress = statusProgress + listOf("Encoding prompt... complete")
            statusProgress = statusProgress + listOf("Generating image...")
            statusProgress = statusProgress + listOf("Fetching UNET file...")

            val bitmap = withContext(Dispatchers.IO) {
                diffusionPipeline.generateImage(
                    encodedPrompt = encodedPromptData,
                    encodedPromptShape = encodedPromptShape,
                    numSteps = denoiseSteps,
                )
            }
            statusProgress = statusProgress + listOf("Image generation complete")
            generatedBitmap = bitmap

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

                // Processing mode selector (CPU/GPU)
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
                        options = deviceOptions,
                        selectedOption = selectedProcessor,
                        onValueChange = { option -> selectedProcessor = option },
                    )
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

                    Text(
                        text = statusProgress.joinToString("\n"),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(16.dp)
                    )
                } else {
                    ImageDisplay(
                        modifier = Modifier.fillMaxSize(),
                        placeHolderText = "Generated image will appear here",
                        bitmap = generatedBitmap
                    )
                }
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
                PromptField(
                    modifier = Modifier.padding(12.dp),
                    rewriteBtn = true,
                    placeHolderText = "Describe image to generate",
                    promptText = promptText,
                    onPromptChange = { newPrompt -> promptText = newPrompt },
                    onRewriteClick = { println("Rewrite clicked") },
                    onSubmit =  { generateImage() }
                )
            }
        }
    }
}