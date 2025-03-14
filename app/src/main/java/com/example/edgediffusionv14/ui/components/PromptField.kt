package com.example.edgediffusionv14.ui.components

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.FloatingActionButtonDefaults

@Composable
fun PromptField(
    modifier: Modifier = Modifier,
    rewriteBtn: Boolean,
    placeHolderText: String,
    promptText: String,
    onPromptChange: (String) -> Unit,
    onRewriteClick: () -> Unit,
    onSubmit: () -> Unit,
    maxLines: Int = 4,
    btnSize: Int = 48
) {
    Row(modifier = modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        // Prompt text field
        OutlinedTextField(
            value = promptText,
            onValueChange = onPromptChange,
            modifier = Modifier
                .weight(1f)
                .heightIn(min = 56.dp, max = 120.dp),
            placeholder = { Text(text = placeHolderText) },
            shape = RoundedCornerShape(24.dp),
            singleLine = false,
            maxLines = maxLines,
            colors = TextFieldDefaults.colors(
                focusedContainerColor = Color.Transparent,
                unfocusedContainerColor = Color.Transparent,
                disabledContainerColor = Color.Transparent,
                focusedIndicatorColor = MaterialTheme.colorScheme.primary,
                unfocusedIndicatorColor = MaterialTheme.colorScheme.outlineVariant
            )
        )

        // Rewrite button
        if(rewriteBtn) {
            FloatingActionButton(
                modifier = Modifier
                    .padding(start = 12.dp)
                    .size(btnSize.dp),
                containerColor = MaterialTheme.colorScheme.secondary,
                contentColor = MaterialTheme.colorScheme.onSecondary,
                shape = RoundedCornerShape(16.dp),
                elevation = FloatingActionButtonDefaults.elevation(
                    defaultElevation = 2.dp,
                    pressedElevation = 4.dp
                ),
                onClick = onRewriteClick
            ) {
                Icon(
                    imageVector = Icons.Default.AutoFixHigh,
                    contentDescription = "Rewrite Prompt"
                )
            }
        }

        // Submit button
        FloatingActionButton(
            onClick = onSubmit,
            modifier = Modifier
                .padding(start = 12.dp)
                .size(btnSize.dp),
            containerColor = MaterialTheme.colorScheme.primary,
            contentColor = MaterialTheme.colorScheme.onPrimary,
            shape = RoundedCornerShape(16.dp),
            elevation = FloatingActionButtonDefaults.elevation(
                defaultElevation = 2.dp,
                pressedElevation = 4.dp
            )
        ) {
            Icon(
                imageVector = Icons.Default.ArrowUpward,
                contentDescription = "Generate"
            )
        }
    }
}