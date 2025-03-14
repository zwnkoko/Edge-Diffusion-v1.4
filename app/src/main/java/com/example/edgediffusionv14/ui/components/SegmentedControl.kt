package com.example.edgediffusionv14.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

@Composable
fun SegmentedControl(
    modifier: Modifier = Modifier,
    boxPadding: Modifier = Modifier.padding(horizontal = 16.dp),
    options: List<String>,
    selectedOption: String,
    onValueChange: (String) -> Unit
) {
    Row(modifier = modifier.padding(4.dp).fillMaxHeight()) {
        options.forEach { option ->
            val isSelected = option == selectedOption
            val backgroundColor = if (isSelected)
                MaterialTheme.colorScheme.primaryContainer
            else
                Color.Transparent
            val textColor = if (isSelected)
                MaterialTheme.colorScheme.onPrimaryContainer
            else
                MaterialTheme.colorScheme.onSurface
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .background(backgroundColor)
                    .clickable {
                        onValueChange(option)
                    }
                    .fillMaxHeight()
                    .then(boxPadding),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = option,
                    color = textColor
                )
            }
        }
    }
}