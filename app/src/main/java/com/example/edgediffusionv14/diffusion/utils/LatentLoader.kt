package com.example.edgediffusionv14.diffusion.utils

import android.content.Context
import android.util.Log
import org.pytorch.Tensor
import java.io.DataInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder


object LatentUtil {

    fun loadLatentsFromFile(context: Context, fileName: String): Tensor {
        val batchSize = 1
        val channels = 4
        val height = 64
        val width = 64
        val numElements = batchSize * channels * height * width
        val latentsFloatArray = FloatArray(numElements)

        try {
            DataInputStream(context.assets.open(fileName)).use { dataInputStream ->
                val byteBuffer = ByteBuffer.allocate(numElements * 4) // 4 bytes per float
                byteBuffer.order(ByteOrder.LITTLE_ENDIAN) // Specify little-endian order

                dataInputStream.readFully(byteBuffer.array()) // Read all bytes at once

                byteBuffer.asFloatBuffer().get(latentsFloatArray) // Get floats from buffer
            }
        } catch (e: IOException) {
            Log.e("LatentUtils", "Error loading latents from file: $fileName", e)
            throw e
        }

        return Tensor.fromBlob(
            latentsFloatArray,
            longArrayOf(batchSize.toLong(), channels.toLong(), height.toLong(), width.toLong())
        )
    }



}