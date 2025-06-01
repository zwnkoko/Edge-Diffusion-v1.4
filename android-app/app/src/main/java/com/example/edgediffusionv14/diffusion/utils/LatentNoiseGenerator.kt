package com.example.edgediffusionv14.diffusion.utils

object LatentNoiseGenerator {
    fun generateGaussianNoise(
        batch: Int,
        channels: Int,
        height: Int,
        width: Int,
        seed: Long? = null
    ): Array<Array<Array<FloatArray>>> {
        // Use provided seed if available, otherwise use random seed
        val random = if (seed != null) {
            java.util.Random(seed)
        } else {
            java.util.Random()
        }

        return Array(batch) {
            Array(channels) {
                Array(height) {
                    FloatArray(width) { random.nextGaussian().toFloat() }
                }
            }
        }
    }
}