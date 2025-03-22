    package com.example.edgediffusionv14.diffusion.models
    
    import org.pytorch.Tensor
    import java.lang.IllegalArgumentException
    import kotlin.math.pow
    
    /**
     * Extension functions for PyTorch Tensor operations.
     * These functions provide simplified arithmetic operations for Tensors.
     */
    
    // Private extension function to convert Tensor to List<Float>
    private fun Tensor.toNumList(): List<Float> { // Assume float32
        return dataAsFloatArray.toList()
    }
    
    // Extension function to get shape of tensor
    fun Tensor.getShape(): LongArray {
        return shape()
    }
    
    // Extension function to add two tensors
    fun Tensor.add(other: Tensor, scale: Float = 1.0f): Tensor {
        val result = this.toNumList().zip(other.toNumList()) { a, b -> (a + b * scale) }
        return Tensor.fromBlob(result.toFloatArray(), this.getShape())
    }
    
    // Extension function to subtract tensor
    fun Tensor.sub(other: Tensor, scale: Float = 1.0f): Tensor {
        val result = this.toNumList().zip(other.toNumList()) { a, b -> (a - b * scale) }
        return Tensor.fromBlob(result.toFloatArray(), this.getShape())
    }
    
    // Extension function to multiply two tensors element-wise
    fun Tensor.mul(other: Tensor): Tensor {
        val result = this.toNumList().zip(other.toNumList()) { a, b -> a * b }
        return Tensor.fromBlob(result.toFloatArray(), this.getShape())
    }
    
    // Extension function to multiply tensor by scalar
    fun Tensor.mul(scale: Float): Tensor {
        val result = this.toNumList().map { it * scale }
        return Tensor.fromBlob(result.toFloatArray(), this.getShape())
    }
    
    // Extension function to divide two tensors element-wise
    fun Tensor.div(other: Tensor): Tensor {
        val result = this.toNumList().zip(other.toNumList()) { a, b -> a / b }
        return Tensor.fromBlob(result.toFloatArray(), this.getShape())
    }
    
    // Extension function to divide tensor by scalar
    fun Tensor.div(scale: Float): Tensor {
        val result = this.toNumList().map { it / scale }
        return Tensor.fromBlob(result.toFloatArray(), this.getShape())
    }
    
    /**
     * PNDMScheduler - Implements the Pseudo Numerical Methods for Diffusion Models scheduler.
     * This scheduler uses a combination of Runge-Kutta and PLMS (Pseudo Linear Multistep) methods
     * to improve the sampling process for diffusion models.
     *
     * @param betaSchedule Schedule for beta parameter ("scaled_linear" etc.)
     * @param betaStart Starting value for beta
     * @param betaEnd Ending value for beta
     * @param numTrainTimesteps Number of training timesteps
     * @param timestepSpacing How to space timesteps ("leading" etc.)
     * @param stepsOffset Offset for steps
     * @param skipPrkSteps Whether to skip Runge-Kutta steps
     * @param predictionType Type of prediction ("v_prediction" etc.)
     * @param setAlphaToOne Whether to set final alpha to one
     */
    class PNDMScheduler(
        private val betaSchedule: String,
        private val betaStart: Double,
        private val betaEnd: Double,
        private val numTrainTimesteps: Int,
        private val timestepSpacing: String,
        private val stepsOffset: Int,
        private val skipPrkSteps: Boolean,
        private val predictionType: String,
        private val setAlphaToOne: Boolean = false
    ) {
        // Add a callback function property
        var onIntermediateSampleCallback: ((Tensor, Int) -> Unit)? = null
    
        // Arrays for diffusion process parameters
        private val betas: FloatArray
        private val alphas: FloatArray
        private val alphasCumprod: FloatArray
        private val finalAlphaCumprod: Float
    
        // Running state values
        private var curModelOutput: Tensor? = null
        private var counter = 0
        private var curSample: Tensor? = null
        private val ets: MutableList<Tensor> = mutableListOf()
    
        // Timestep-related variables
        private var numInferenceSteps: Int? = null
        private lateinit var _timesteps: IntArray
        private var prkTimesteps: IntArray? = null
        private var plmsTimesteps: IntArray? = null
        private var timesteps: Tensor? = null
    
        /**
         * Initialize the scheduler parameters based on the provided configuration.
         */
        init {
            if (betaSchedule == "scaled_linear") {
                // Initialize diffusion parameters
                betas = FloatArray(numTrainTimesteps)
                alphas = FloatArray(numTrainTimesteps)
                alphasCumprod = FloatArray(numTrainTimesteps)
    
                // Calculate beta, alpha, and cumulative product of alphas for each timestep
                for (i in 0 until numTrainTimesteps) {
                    betas[i] = (betaStart.pow(0.5) + (betaEnd.pow(0.5) - betaStart.pow(0.5)) * i / (numTrainTimesteps - 1)).toFloat().pow(2)
                    alphas[i] = 1.0f - betas[i]
                    alphasCumprod[i] = if (i == 0) alphas[i] else alphasCumprod[i - 1] * alphas[i]
                }
                finalAlphaCumprod = if (setAlphaToOne) 1.0f else alphasCumprod[0]
            } else {
                throw IllegalArgumentException("Unsupported beta schedule: $betaSchedule")
            }
        }
    
        /**
         * Set the timesteps for inference.
         * Configures the schedule based on the number of inference steps.
         *
         * @param numInferenceSteps Number of inference steps to use
         */
        fun setTimesteps(numInferenceSteps: Int) {
            this.numInferenceSteps = numInferenceSteps
    
            if (timestepSpacing == "leading") {
                val stepRatio = numTrainTimesteps / numInferenceSteps
                _timesteps = (0 until numInferenceSteps).map { (it * stepRatio) }.toIntArray()
                _timesteps = _timesteps.map { it + stepsOffset }.toIntArray()
            }
    
            if (skipPrkSteps) {
                prkTimesteps = intArrayOf()
                plmsTimesteps = (_timesteps.dropLast(1) + _timesteps.takeLast(2).take(1) + _timesteps.takeLast(1)).reversed().toIntArray()
    
                val timestepsArray = (prkTimesteps!! + plmsTimesteps!!).map { it.toLong() }.toLongArray()
                timesteps = Tensor.fromBlob(timestepsArray, longArrayOf(timestepsArray.size.toLong()))
    
                // Reset state
                ets.clear()
                counter = 0
                curModelOutput = null
            }
        }
    
        /**
         * Main step function that chooses between PRK and PLMS methods.
         *
         * @param modelOutput Output from the diffusion model
         * @param timestep Current timestep
         * @param sample Current sample tensor
         * @return Updated sample tensor after the step
         */
        fun step(modelOutput: Tensor, timestep: Int, sample: Tensor): Tensor {
            return if (counter < (prkTimesteps?.size ?: 0) && !skipPrkSteps) {
                stepPrk(modelOutput, timestep, sample)
            } else {
                stepPlms(modelOutput, timestep, sample)
            }
        }
    
        /**
         * Perform a Pseudo Runge-Kutta step.
         * Implements a 4th order Runge-Kutta-like method for the diffusion process.
         *
         * @param modelOutput Output from the diffusion model
         * @param timestep Current timestep
         * @param sample Current sample tensor
         * @return Updated sample tensor after the PRK step
         */
        private fun stepPrk(modelOutput: Tensor, timestep: Int, sample: Tensor): Tensor {
            val diffToPrev = if (counter % 2 == 0) 0 else numTrainTimesteps / numInferenceSteps!! / 2
            val prevTimestep = timestep - diffToPrev
            val currentTimestep = prkTimesteps!![counter / 4 * 4]
    
            // Handle different stages of the 4-stage Runge-Kutta process
            if (counter % 4 == 0) {
                // First stage - initialize or accumulate
                if (curModelOutput == null) {
                    curModelOutput = modelOutput.mul(1.0f / 6.0f)
                } else {
                    curModelOutput = curModelOutput!!.add(modelOutput.mul(1.0f / 6.0f))
                }
                ets.add(modelOutput)
                curSample = sample
            } else if ((counter - 1) % 4 == 0) {
                // Second stage - accumulate with weight 1/3
                curModelOutput = curModelOutput!!.add(modelOutput.mul(1.0f / 3.0f))
            } else if ((counter - 2) % 4 == 0) {
                // Third stage - accumulate with weight 1/3
                curModelOutput = curModelOutput!!.add(modelOutput.mul(1.0f / 3.0f))
            } else if ((counter - 3) % 4 == 0) {
                // Fourth stage - finalize and update
                val curModelOutputCopy = curModelOutput!!.add(modelOutput.mul(1.0f / 6.0f))
                curModelOutput = null // Reset curModelOutput after using it
                val prevSample = getPrevSample(curSample!!, currentTimestep, prevTimestep, curModelOutputCopy)
                // Call the callback with intermediate result
                onIntermediateSampleCallback?.invoke(prevSample, counter)
                counter += 1
                return prevSample
            }
    
            // Return intermediary result during RK steps
            val curSample = curSample ?: sample
            val prevSample = getPrevSample(curSample, currentTimestep, prevTimestep, modelOutput)
            counter += 1
            return prevSample
        }
    
        /**
         * Perform a Pseudo Linear Multistep (PLMS) step.
         * Uses previous model outputs to improve the accuracy of the step.
         *
         * @param modelOutput Output from the diffusion model
         * @param timestep Current timestep
         * @param sample Current sample tensor
         * @return Updated sample tensor after the PLMS step
         */
        private fun stepPlms(modelOutput: Tensor, timestep: Int, sample: Tensor): Tensor {
            var prevTimestep = timestep - numTrainTimesteps / numInferenceSteps!!
    
            // Manage model output history
            if (counter != 1) {
                while (ets.size > 3) {
                    ets.removeAt(0)
                }
                ets.add(modelOutput)
            } else {
                prevTimestep = timestep
                // commented out: timestep += numTrainTimesteps / numInferenceSteps!! // Causes error on real device
            }
    
            // Apply different order methods based on available history
            val modelOutput = when {
                ets.size == 1 && counter == 0 -> {
                    // First order - use model output directly
                    modelOutput
                }
                ets.size == 1 && counter == 1 -> {
                    // Also first order but with averaging
                    ets.last().mul(0.5f).add(modelOutput.mul(0.5f))
                }
                ets.size == 2 -> {
                    // Second order method
                    ets.last().mul(3f).sub(ets[ets.size - 2].mul(1f)).mul(0.5f)
                }
                ets.size == 3 -> {
                    // Third order method
                    ets.last().mul(23f / 12f).sub(ets[ets.size - 2].mul(16f / 12f)).add(ets[ets.size - 3].mul(5f / 12f))
                }
                else -> {
                    // Fourth order method
                    ets.last().mul(55f / 24f).sub(ets[ets.size - 2].mul(59f / 24f)).add(ets[ets.size - 3].mul(37f / 24f)).sub(ets[ets.size - 4].mul(9f / 24f))
                }
            }
    
            // Calculate previous sample and increment counter
            val prevSample = getPrevSample(sample, timestep, prevTimestep, modelOutput)
            counter += 1
    
            return prevSample
        }
    
        /**
         * Calculate the previous sample based on current sample and model output.
         * Uses the diffusion process parameters to determine the update.
         *
         * @param sample Current sample tensor
         * @param timestep Current timestep
         * @param prevTimestep Previous timestep
         * @param modelOutput Model output tensor
         * @return Previous sample tensor
         */
        private fun getPrevSample(sample: Tensor, timestep: Int, prevTimestep: Int, modelOutput: Tensor): Tensor {
            // Get alpha and beta values for current and previous timesteps
            val alphaProdT = alphasCumprod[timestep]
            val alphaProdTPrev = if (prevTimestep >= 0) alphasCumprod[prevTimestep] else finalAlphaCumprod
            val betaProdT = 1 - alphaProdT
            val betaProdTPrev = 1 - alphaProdTPrev
    
            // Calculate coefficients for the update
            val sampleCoeff = (alphaProdTPrev.toDouble() / alphaProdT.toDouble()).pow(0.5).toFloat()
            val modelOutputDenomCoeff = (alphaProdT.toDouble() * betaProdTPrev.toDouble().pow(0.5) +
                    (alphaProdT.toDouble() * betaProdT.toDouble() * alphaProdTPrev.toDouble()).pow(0.5)).toFloat()
    
            // Handle different prediction types
            var modelOutputValue = modelOutput
            if (predictionType == "v_prediction") {
                modelOutputValue = modelOutput.mul(alphaProdT.toDouble().pow(0.5).toFloat())
                    .add(sample.mul(betaProdT.toDouble().pow(0.5).toFloat()))
            }
    
            // Calculate previous sample
            val prevSample = sample.mul(sampleCoeff)
                .sub(modelOutputValue.mul((alphaProdTPrev - alphaProdT) / modelOutputDenomCoeff))
    
            return prevSample
        }
    
        /**
         * Get the timesteps to be used for the diffusion process.
         *
         * @return Array of timesteps or null if not set
         */
        fun getTimeSteps(): IntArray? {
            return timesteps?.let { tensorToIntegerArray(it) }
        }
    
        /**
         * Convert a tensor to an integer array.
         *
         * @param tensor Input tensor
         * @return Integer array representation
         */
        private fun tensorToIntegerArray(tensor: Tensor): IntArray {
            return tensor.dataAsLongArray.map { it.toInt() }.toIntArray()
        }
    }