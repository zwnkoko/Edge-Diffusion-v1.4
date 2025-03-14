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
import com.example.edgediffusionv14.ui.components.ImageDisplay
import com.example.edgediffusionv14.ui.components.PromptField
import com.example.edgediffusionv14.ui.components.SegmentedControl
import com.example.edgediffusionv14.ui.components.StepControl

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen() {
    val deviceOptions = listOf("CPU", "GPU")
    var selectedProcessor by remember { mutableStateOf("CPU") }
    var denoiseSteps by remember { mutableIntStateOf(30) }
    var promptText by remember { mutableStateOf("") }
    val surfaceColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f)

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
                    modifier = Modifier.border(
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
                ImageDisplay(
                    modifier = Modifier.background(
                        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.7f)
                    ),
                    placeHolderText = "Generated image will appear here",
                    imagePath = null
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
                PromptField(
                    modifier = Modifier.padding(12.dp),
                    rewriteBtn = true,
                    placeHolderText = "Describe image to generate",
                    promptText = promptText,
                    onPromptChange = { newPrompt -> promptText = newPrompt },
                    onRewriteClick = { println("Rewrite clicked") },
                    onSubmit = { println(promptText) }
                )
            }
        }
    }
}