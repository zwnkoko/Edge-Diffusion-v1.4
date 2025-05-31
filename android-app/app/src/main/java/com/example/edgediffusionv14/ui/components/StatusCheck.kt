package com.example.edgediffusionv14.ui.components

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

@Composable
fun StatusCheck(
    label: String,
    isActive: Boolean?,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier.padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        var statusColor = Color.Gray
        var statusText = "Checking"

        Text(
            text = "$label:",
            fontWeight = FontWeight.Medium
        )
        Spacer(modifier = Modifier.width(8.dp))

        if(isActive != null){
            statusColor = if (isActive) Color.Green else Color.Red
            statusText = if (isActive) "Active" else "Inactive"
        }


        Row(
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Colored indicator circle
            androidx.compose.foundation.Canvas(
                modifier = Modifier.size(10.dp)
            ) {
                drawCircle(color = statusColor)
            }

            Spacer(modifier = Modifier.width(4.dp))

            Text(
                text = statusText,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium
            )
        }
    }
}