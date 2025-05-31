package com.example.edgediffusionv14.diffusion

import android.content.Context
import android.util.Log
import com.example.edgediffusionv14.diffusion.models.MinimalCLIPTokenizer
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import com.example.edgediffusionv14.diffusion.utils.FileUtils
import com.example.edgediffusionv14.diffusion.models.PNDMScheduler
import com.example.edgediffusionv14.diffusion.utils.LatentUtil
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import android.graphics.Bitmap
import android.graphics.Color
import java.nio.FloatBuffer
import java.nio.LongBuffer
import com.example.edgediffusionv14.diffusion.utils.LatentNoiseGenerator.generateGaussianNoise

/**
 * Helper function to flatten a 4D array into a 1D list
 */
private fun Array<Array<Array<FloatArray>>>.flatten(): List<Float> {
    val result = mutableListOf<Float>()
    for (batch in this) {
        for (channel in batch) {
            for (row in channel) {
                for (value in row) {
                    result.add(value)
                }
            }
        }
    }
    return result
}

/**
 * Stable Diffusion pipeline implementation for on-device text-to-image generation
 *
 * This pipeline handles the full text-to-image generation process:
 * 1. Text encoding (via CLIP)
 * 2. Denoising diffusion (via U-Net)
 * 3. Latent decoding (via VAE)
 */
class DiffusionPipeline(
    private val context: Context,
    private val tokenizer: MinimalCLIPTokenizer,
    private val batchSize: Int = 1,
    private val sequenceLength: Int = 77,
    private val scheduler: PNDMScheduler,
) {

    /**
     * Encodes the text prompt and negative prompt into embeddings
     *
     * @param prompt The text prompt to generate an image from
     * @param negativePrompt Optional negative prompt to guide what not to include
     * @return Pair of encoded embeddings (as FloatArray) and the shape information
     */
    fun encodePrompt(
        prompt: String,
        negativePrompt: String = "",
    ): Pair<FloatArray?, LongArray?> {
        tokenizer.clearCache()

        // 1. Tokenize and encode the prompts
        val promptList = listOf(prompt)
        val uncondPromptList = listOf(negativePrompt)

        val encodedPrompt = tokenizer(
            promptList,
            padding = "max_length",
            maxLength = sequenceLength,
            truncation = true
        )
        val uncondEncodedPrompt = tokenizer(
            uncondPromptList,
            padding = "max_length",
            maxLength = sequenceLength,
            truncation = true
        )

        // 2. Get input IDs and convert to LongArray
        val encodedInputIds = encodedPrompt["input_ids"] as List<*>
        val uncondEncodedInputIds = uncondEncodedPrompt["input_ids"] as List<*>

        val inputIds = encodedInputIds.map { subList ->
            if (subList is List<*>) {
                subList.filterIsInstance<Int>().map { it.toLong() }.toLongArray()
            } else {
                longArrayOf()
            }
        }.toTypedArray()

        val uncondInputIds = uncondEncodedInputIds.map { subList ->
            if (subList is List<*>) {
                subList.filterIsInstance<Int>().map { it.toLong() }.toLongArray()
            } else {
                longArrayOf()
            }
        }.toTypedArray()

        // 3. Load text encoder model
        val module = Module.load(FileUtils.assetFilePath(context, "diffusion/text_encoder_pt.pt"))

        // 4. Create the input tensors for both prompts
        val inputTensor = Tensor.fromBlob(inputIds[0], longArrayOf(1, inputIds[0].size.toLong()))
        val uncondInputTensor = Tensor.fromBlob(uncondInputIds[0], longArrayOf(1, uncondInputIds[0].size.toLong()))

        // 5. Run text encoder inference
        val textEmbeddings = module.forward(IValue.from(inputTensor)).toTensor()
        val uncondTextEmbeddings = module.forward(IValue.from(uncondInputTensor)).toTensor()

        val textEmbeddingsFloat = textEmbeddings.dataAsFloatArray
        val uncondTextEmbeddingsFloat = uncondTextEmbeddings.dataAsFloatArray

        // 6. Merge negative and positive embeddings for classifier-free guidance
        val embeddingSize = textEmbeddingsFloat.size / (batchSize * sequenceLength)
        val mergedEmbeddings = FloatArray(textEmbeddingsFloat.size + uncondTextEmbeddingsFloat.size)

        // Copy the unconditional embeddings first, then the conditional ones
        System.arraycopy(uncondTextEmbeddingsFloat, 0, mergedEmbeddings, 0, uncondTextEmbeddingsFloat.size)
        System.arraycopy(textEmbeddingsFloat, 0, mergedEmbeddings, uncondTextEmbeddingsFloat.size, textEmbeddingsFloat.size)

        // 7. Create a merged tensor for classifier-free guidance
        val mergedTensor = Tensor.fromBlob(
            mergedEmbeddings,
            longArrayOf(2, sequenceLength.toLong(), embeddingSize.toLong()) // Shape: [2, 77, embeddingSize]
        )
        val mergedTensorFloat = mergedTensor.dataAsFloatArray

        return Pair(mergedTensorFloat, mergedTensor.shape())
    }

    /**
     * Generates an image from the encoded prompt using diffusion
     *
     * @param encodedPrompt The encoded prompt embeddings from encodePrompt
     * @param encodedPromptShape Shape information for the encoded prompt
     * @param numSteps Number of denoising steps to perform
     * @param progressCallback Optional callback for progress updates
     * @param randomSeed Optional seed for reproducible generation
     * @return The generated image as a Bitmap, or null if generation failed
     */
    fun generateImage(
        encodedPrompt: FloatArray,
        encodedPromptShape: LongArray,
        numSteps: Int,
        progressCallback: (Int, Int) -> Unit = { _, _ -> },
        randomSeed: Long?,
    ): Bitmap? {
        // Initialize latent tensor either from random noise or a file
        var latentTensor: Tensor = if (randomSeed != null) {
            // Generate random noise with given seed for reproducibility
            val noiseArray = generateGaussianNoise(1, 4, 64, 64, randomSeed)
            val flattenedNoise = noiseArray.flatten().toFloatArray()
            Tensor.fromBlob(flattenedNoise, longArrayOf(1, 4, 64, 64))
        } else {
            // Use pre-defined latents from file
            LatentUtil.loadLatentsFromFile(context, "diffusion/latents.bin")
        }

        // Set up diffusion scheduler
        scheduler.setTimesteps(numSteps)
        val timeSteps = scheduler.getTimeSteps() ?: return null

        // Initialize UNet ONNX runtime
        val unetPath = FileUtils.fetchUnetAssetPath()
        if (unetPath == null) {
            Log.e("UNETSession", "Failed to initialize UNET session: File not found")
            return null
        }

        val ortEnv = OrtEnvironment.getEnvironment()
        val ortSession = ortEnv.createSession(unetPath, OrtSession.SessionOptions())
        Log.d("UNETSession", "Initialized UNET session with file: $unetPath")

        // Get initial data from latent tensor
        var data = latentTensor.dataAsFloatArray

        // Prepare for batch processing (doubled for classifier-free guidance)
        val shape = latentTensor.shape()
        val newBatchSize = shape[0] * 2 // Double the batch size for guidance
        val newShape = LongArray(shape.size) { index ->
            if (index == 0) newBatchSize.toLong() else shape[index]
        }

        // Set guidance scale for classifier-free guidance
        val guidanceScale = 7.5f

        // Main diffusion loop
        var counter = 1
        for (timestep in timeSteps) {
            // Update progress
            progressCallback(counter, timeSteps.size)
            counter++

            // 1. Create doubled batch for classifier-free guidance
            val combinedData = FloatArray(data.size * 2)
            System.arraycopy(data, 0, combinedData, 0, data.size)
            System.arraycopy(data, 0, combinedData, data.size, data.size)
            val latentModelInput = Tensor.fromBlob(combinedData, newShape)
            val latentModelInputFloat = latentModelInput.dataAsFloatArray

            // 2. Prepare inputs for UNet
            val latentBuffer = FloatBuffer.wrap(latentModelInputFloat)
            val combinedBuffer = FloatBuffer.wrap(encodedPrompt)

            // Create ONNX tensors
            val latentOnnxTensor = OnnxTensor.createTensor(ortEnv, latentBuffer, newShape)

            // Create timestep tensor (as scalar)
            val timestepScalarValue = timestep.toLong()
            val timestepBuffer = LongBuffer.allocate(1)
            timestepBuffer.put(timestepScalarValue)
            timestepBuffer.rewind()
            val timestepOnnxTensor = OnnxTensor.createTensor(
                ortEnv,
                timestepBuffer,
                longArrayOf() // Empty shape for scalar
            )

            // Create text embedding tensor
            val combinedOnnxTensor = OnnxTensor.createTensor(ortEnv, combinedBuffer, encodedPromptShape)

            // 3. Run UNet inference
            val inputs = mapOf(
                "sample" to latentOnnxTensor,
                "timestep" to timestepOnnxTensor,
                "encoder_hidden_states" to combinedOnnxTensor
            )

            val output = ortSession.run(inputs)

            // 4. Process UNet output
            val noisePredTensor = output?.get(0) as OnnxTensor
            val noisePredShape = noisePredTensor.info.shape

            // 5. Process noise prediction for classifier-free guidance
            val noisePredBuffer = noisePredTensor.floatBuffer
            val batchSize = noisePredShape[0].toInt()
            val channelSize = noisePredShape[1].toInt()
            val height = noisePredShape[2].toInt()
            val width = noisePredShape[3].toInt()
            val channelDataSize = channelSize * height * width
            val splitSize = batchSize / 2

            // Split into unconditional and conditional noise predictions
            val noisePredUncondBuffer = FloatBuffer.allocate(splitSize * channelDataSize)
            val noisePredTextBuffer = FloatBuffer.allocate(splitSize * channelDataSize)

            // Extract unconditional and conditional parts
            for (i in 0 until splitSize) {
                noisePredBuffer.position(i * channelDataSize)
                val tempUncondBuffer = noisePredBuffer.slice()
                tempUncondBuffer.limit(channelDataSize)
                noisePredUncondBuffer.put(tempUncondBuffer)
            }
            for (i in splitSize until batchSize) {
                noisePredBuffer.position(i * channelDataSize)
                val tempTextBuffer = noisePredBuffer.slice()
                tempTextBuffer.limit(channelDataSize)
                noisePredTextBuffer.put(tempTextBuffer)
            }
            noisePredUncondBuffer.rewind()
            noisePredTextBuffer.rewind()

            // Convert to arrays
            val noisePredUncond = FloatArray(noisePredUncondBuffer.remaining())
            noisePredUncondBuffer.get(noisePredUncond)
            val noisePredText = FloatArray(noisePredTextBuffer.remaining())
            noisePredTextBuffer.get(noisePredText)

            // 6. Apply classifier-free guidance formula
            val guidedNoisePred = noisePredUncond.mapIndexed { index, uncond ->
                uncond + guidanceScale * (noisePredText[index] - uncond)
            }.toFloatArray()

            // 7. Create tensor for guided prediction
            val guidedNoisePredTensor = Tensor.fromBlob(
                guidedNoisePred,
                longArrayOf(splitSize.toLong(), channelSize.toLong(), height.toLong(), width.toLong())
            )

            // 8. Compute the previous noisy sample using the scheduler
            latentTensor = scheduler.step(
                guidedNoisePredTensor,
                timestep,
                latentTensor
            )

            // Update data for next iteration
            data = latentTensor.dataAsFloatArray
        }

        // Clean up ONNX resources
        ortSession.close()

        // Signal completion
        progressCallback(-1, -1)

        // --- VAE Decoding phase ---

        // 1. Scale the latents for VAE
        val scalingFactor = 1 / 0.18215f
        val scaledVAEData = data.map { it * scalingFactor }.toFloatArray()

        // 2. Set up VAE decoder
        val vaePath = FileUtils.getFilePath(context, "diffusion/vae_onnx.onnx")
        val vaeOrtSession = ortEnv.createSession(vaePath, OrtSession.SessionOptions())

        // 3. Create tensor for VAE input
        val vaeLatentTensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(scaledVAEData),
            latentTensor.shape()
        )

        // 4. Prepare VAE input
        val vaeInputs = mapOf(
            vaeOrtSession.inputNames.first() to vaeLatentTensor
        )

        // 5. Run VAE inference
        val vaeOutput = vaeOrtSession.run(vaeInputs)

        // 6. Process VAE output
        val vaeOutputTensor = vaeOutput[0] as? OnnxTensor
        val vaeOutputShape = vaeOutputTensor?.info?.shape
        val vaeOutputData = vaeOutputTensor?.floatBuffer

        // Clean up resources
        vaeOrtSession.close()
        ortEnv.close()

        // 7. Convert output to bitmap image
        if (vaeOutputData != null && vaeOutputShape != null) {
            return convertToImage(vaeOutputData, vaeOutputShape)
        }

        return null
    }

    /**
     * Converts VAE output tensor to a Bitmap image
     *
     * @param data The VAE output data as a FloatBuffer
     * @param shape The shape of the VAE output tensor
     * @return The generated image as a Bitmap
     */
    private fun convertToImage(data: FloatBuffer, shape: LongArray): Bitmap {
        // Extract dimensions from shape (NCHW format)
        val batchSize = shape[0].toInt()
        val channels = shape[1].toInt()
        val height = shape[2].toInt()
        val width = shape[3].toInt()

        require(batchSize == 1) { "Only batch size 1 is supported" }

        // Create bitmap for the output image
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Convert buffer to array for easier access
        val dataArray = FloatArray(data.remaining())
        data.get(dataArray)
        data.rewind()

        val numPixels = height * width

        // Process each pixel
        for (h in 0 until height) {
            for (w in 0 until width) {
                val pixelIndex = h * width + w

                when (channels) {
                    3 -> {
                        // RGB format - calculate indices for each channel in NCHW layout
                        val rIndex = pixelIndex
                        val gIndex = numPixels + pixelIndex
                        val bIndex = 2 * numPixels + pixelIndex

                        val rValue = dataArray[rIndex]
                        val gValue = dataArray[gIndex]
                        val bValue = dataArray[bIndex]

                        // Normalize from [-1, 1] to [0, 255]
                        val r = ((rValue * 0.5f + 0.5f).coerceIn(0f, 1f) * 255).toInt()
                        val g = ((gValue * 0.5f + 0.5f).coerceIn(0f, 1f) * 255).toInt()
                        val b = ((bValue * 0.5f + 0.5f).coerceIn(0f, 1f) * 255).toInt()

                        bitmap.setPixel(w, h, Color.rgb(r, g, b))
                    }
                    1 -> {
                        // Grayscale format
                        val grayValue = dataArray[pixelIndex]
                        val gray = ((grayValue * 0.5f + 0.5f).coerceIn(0f, 1f) * 255).toInt()
                        bitmap.setPixel(w, h, Color.rgb(gray, gray, gray))
                    }
                    else -> throw IllegalArgumentException("Unsupported number of channels: $channels")
                }
            }
        }

        return bitmap
    }
}