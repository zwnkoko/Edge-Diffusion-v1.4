package com.example.edgediffusionv14.diffusion.utils

import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import android.content.Context


object FileUtils {

    /**
     * Gets the absolute path to the file, whether on external storage or in assets.
     *
     * @param context The application context.
     * @param assetPath The path to the asset within the assets directory, or the filename if in external storage.
     * @return The absolute path to the file.
     * @throws IOException If an error occurs accessing the file.
     */
    @Throws(IOException::class)
    fun getFilePath(context: Context, assetPath: String): String {
        return assetFilePath(context, assetPath)
    }

    @Throws(IOException::class)
    fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        file.parentFile?.mkdirs()
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { `is` ->
            FileOutputStream(file).use { os ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (`is`.read(buffer).also { read = it } != -1) {
                    os.write(buffer, 0, read)
                }
                os.flush()
            }
            return file.absolutePath
        }
    }


    /**
     * Utility function to fetch the UNET asset file path.
     *
     * @return Absolute path of the file as a String, or null if the file does not exist.
     */

    fun fetchUnetAssetPath(): String? {
        val unetPath = "/storage/0000-0000/UNET/unet_onnx/"
        val unetFileName = "unet_onnx.onnx"

        // Create the File object
        val file = File(unetPath, unetFileName)

        // Check if the file exists
        return if (file.exists()) {
            Log.d("FileAccess", "UNET file exists at: ${file.absolutePath}")
            file.absolutePath
        } else {
            Log.e("FileAccess", "UNET file not found at: ${file.absolutePath}")
            null
        }
    }

}