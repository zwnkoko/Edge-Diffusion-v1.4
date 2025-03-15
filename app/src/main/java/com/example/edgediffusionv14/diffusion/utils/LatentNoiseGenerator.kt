package com.example.edgediffusionv14.diffusion.utils

import kotlin.random.Random

object LatentNoiseGenerator {
    fun generateGaussianNoise(batch: Int, channels: Int, height: Int, width: Int): Array<Array<Array<FloatArray>>> {
        val random = java.util.Random()
        return Array(batch) {
            Array(channels) {
                Array(height) {
                    FloatArray(width) { random.nextGaussian().toFloat() } // Standard Normal Distribution
                }
            }
        }
    }
}
