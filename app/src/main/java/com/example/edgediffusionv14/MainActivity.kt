package com.example.edgediffusionv14

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.example.edgediffusionv14.ui.theme.EdgeDiffusionV14Theme
import com.example.edgediffusionv14.ui.screens.MainScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            EdgeDiffusionV14Theme {
                MainScreen()
            }
        }
    }
}

