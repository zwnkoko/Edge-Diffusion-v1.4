package com.example.edgediffusionv14

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import com.example.edgediffusionv14.ui.theme.EdgeDiffusionV14Theme
import com.example.edgediffusionv14.ui.screens.MainScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            EdgeDiffusionV14Theme {
                if (allPermissionsGranted()) {
                    MainScreen()
                } else {
                    requestPermissions()
                }

            }
        }
    }
    /**
     * Request storage permissions for Android 11+ (API 30+)
     * - Requests MANAGE_ALL_FILES_ACCESS permission via Settings
     */
    private fun requestPermissions() {
        // Android 11+ requires MANAGE_ALL_FILES_ACCESS_PERMISSION via Settings
        try {
            val uri = Uri.parse("package:${packageName}")
            val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, uri)
            requestPermissionLauncher.launch(intent)
        } catch (e: Exception) {
            Log.e("MainActivity", "Error launching settings for all files access", e)
            Toast.makeText(
                this,
                "Unable to request storage permissions. Please enable them manually in Settings.",
                Toast.LENGTH_LONG
            ).show()
            finish()
        }
    }

    /**
     * Handles the result from the settings activity for Android 11+ permissions
     */
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { _ ->
        if (Environment.isExternalStorageManager()) {
            // Refresh the UI to show the main content since permission is granted
            setContent {
                EdgeDiffusionV14Theme {
                    MainScreen()
                }
            }
        } else {
            // User denied the permission - inform them and exit
            showPermissionDeniedMessage()
        }
    }

    /**
     * Checks if all required storage permissions are granted
     * @return true if MANAGE_ALL_FILES_ACCESS permission is granted
     */
    private fun allPermissionsGranted() = Environment.isExternalStorageManager()

    /**
     * Shows a toast message and finishes the activity when permission is denied
     */
    private fun showPermissionDeniedMessage() {
        Toast.makeText(
            this,
            "Storage permission is required for this app to function.",
            Toast.LENGTH_LONG
        ).show()
        finish()
    }

}
