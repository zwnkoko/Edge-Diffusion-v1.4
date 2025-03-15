package com.example.edgediffusionv14.diffusion

import android.content.Context
import android.util.Log
import com.example.edgediffusionv14.diffusion.models.MinimalCLIPTokenizer
import kotlin.collections.map
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

// Add this outside your class, at the top level of the file
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

class DiffusionPipeline (
    private val context: Context,

    // Tokenizer for the text prompt
    private val tokenizer: MinimalCLIPTokenizer,
    private val batchSize: Int = 1,
    private val sequenceLength : Int = 77,

    // Scheduler for the diffusion model
    private val scheduler: PNDMScheduler,
){

    fun encodePrompt(
        prompt: String
    ): Pair<FloatArray?, LongArray?> {
       tokenizer.clearCache()

        //1. Tokenize and encode the text prompt
        var promptList =listOf(prompt)
        var uncondPromptList = listOf("")

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

        // 2. Get input IDs and convert to LongArray (keeping batch dimension)
        val encodedInputIds = encodedPrompt["input_ids"] as List<*>
        val uncondEncodedInputIds = uncondEncodedPrompt["input_ids"] as List<*>

//        println("Encoded input IDs: $encodedInputIds")
//        println("Unconditional Encoded input IDs: $uncondEncodedInputIds")

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


        // 3. Load encoder model
        val module = Module.load(FileUtils.assetFilePath(context, "diffusion/text_encoder_pt.pt"))

        // 4. Create the input tensor (for the first batch)
        val inputTensor = Tensor.fromBlob(inputIds[0], longArrayOf(1, inputIds[0].size.toLong()))
        val uncondInputTensor = Tensor.fromBlob(uncondInputIds[0], longArrayOf(1, uncondInputIds[0].size.toLong()))

        // 5. Run inference
        val textEmbeddings = module.forward(IValue.from(inputTensor)).toTensor()
        val uncondTextEmbeddings = module.forward(IValue.from(uncondInputTensor)).toTensor()

        val textEmbeddingsFloat = textEmbeddings.dataAsFloatArray
        val uncondTextEmbeddingsFloat = uncondTextEmbeddings.dataAsFloatArray

        // 6. Process the output
//        println("Inference Output (first 10 values): ${textEmbeddingsFloat.take(10).joinToString()}")
//        println("Inference Output (last 10 values): ${textEmbeddingsFloat.takeLast(10).joinToString()}")

        val embeddingSize =  textEmbeddingsFloat.size / (batchSize * sequenceLength)
        val mergedEmbeddings = FloatArray(textEmbeddingsFloat.size + uncondTextEmbeddingsFloat.size)

        System.arraycopy(uncondTextEmbeddingsFloat, 0, mergedEmbeddings, 0, uncondTextEmbeddingsFloat.size)
        System.arraycopy(textEmbeddingsFloat, 0, mergedEmbeddings, uncondTextEmbeddingsFloat.size, textEmbeddingsFloat.size)

        // 3. Create a new tensor from the combined embeddings
        val mergedTensor = Tensor.fromBlob(
            mergedEmbeddings,
            longArrayOf(2, sequenceLength.toLong(), embeddingSize.toLong()) // New shape: [2, 77, embeddingSize]
        )
        val mergedTensorFloat = mergedTensor.dataAsFloatArray

//        println("Merged Tensor (first 10 values): ${mergedTensorFloat.take(10).joinToString()}")

        return Pair(mergedTensorFloat, mergedTensor.shape())

    }

    fun generateImage(
        encodedPrompt: FloatArray,
        encodedPromptShape : LongArray,
        numSteps: Int,
        progressCallback: (Int, Int) -> Unit = { _, _ -> },
        randomSeed: Boolean,
    ): Bitmap? {
        var latentTensor: Tensor
       // var latentTensor = LatentUtil.loadLatentsFromFile(context, "diffusion/latents.bin")
        // Generate noise as a 4D array
        if (randomSeed){
            val noiseArray = generateGaussianNoise(1, 4, 64, 64)

            // Convert the 4D array to a flat FloatArray
            val flattenedNoise = noiseArray.flatten().toFloatArray()

            // Create a Tensor from the flattened array with proper shape
            latentTensor = Tensor.fromBlob(
                flattenedNoise,
                longArrayOf(1, 4, 64, 64)
            )
        } else {
            latentTensor = LatentUtil.loadLatentsFromFile(context, "diffusion/latents.bin")
        }


        //val latenTensorFloat = latentTensor.dataAsFloatArray

        scheduler.setTimesteps(numSteps)
        var timeSteps = scheduler.getTimeSteps()
        println("Timesteps: ${timeSteps?.joinToString()}")

        val unetPath = FileUtils.fetchUnetAssetPath()
        if (unetPath != null) {
            // Proceed to initialize UNET session with the file
            Log.d("UNETSession", "Initializing UNET session with file: ${unetPath}")
            // Example: Initialize your ONNX session here
            // unetSession.initialize(unetFile)
        } else {
            Log.e("UNETSession", "Failed to initialize UNET session: File not found")
        }
        val ortEnv = OrtEnvironment.getEnvironment()
        val ortSession = ortEnv.createSession(unetPath, OrtSession.SessionOptions())

        val noisePredList = mutableListOf<Array<Array<Array<FloatArray>>>>()
        val latentList = mutableListOf<Array<Array<Array<FloatArray>>>>()

// 1) Create the UNet session if you haven't already:
        val sessionOptions = OrtSession.SessionOptions()
        println("SessionOptions: $sessionOptions")
        //val unetSession = ortEnv.createSession(unetPath, sessionOptions)
        //To DO:  This path is  wrong - i need to get the path of the onnx model


// Assuming scaledLatentsTensor is your initial Tensor
        val shape = latentTensor.shape()
        val newBatchSize = shape[0] * 2 // Double the batch size

// Create a new shape with the doubled batch size
        val newShape = LongArray(shape.size) { index ->
            if (index == 0) newBatchSize.toLong() else shape[index]
        }

        // Get the data as a FloatArray
        var data = latentTensor.dataAsFloatArray



        val guidanceScale = 7.5f
        if (timeSteps != null) {
            var counter = 1
            for(timestep in timeSteps){
                println("current timestep ${timestep}")
                progressCallback(counter, timeSteps.size)
                counter = counter + 1
                // Create a new FloatArray to hold the concatenated data
                val combinedData = FloatArray(data.size * 2)

                // Copy the original data twice into the new array
                System.arraycopy(data, 0, combinedData, 0, data.size)
                System.arraycopy(data, 0, combinedData, data.size, data.size)

                val latentModelInput = Tensor.fromBlob(combinedData, newShape)
                println("Latent Model Input Shape: ${latentModelInput.shape().contentToString()}")
                val latentModelInputFloat = latentModelInput.dataAsFloatArray

                val latentBuffer = FloatBuffer.wrap(latentModelInputFloat)
                val combinedBuffer = FloatBuffer.wrap(encodedPrompt)

                // 2. Create input map for ONNX
                val inputName = ortSession.inputNames
                val latentOnnxTensor = OnnxTensor.createTensor(ortEnv, latentBuffer, newShape)
                val tensorShape = latentOnnxTensor.info.shape
                println("Latent Onnx Tensor Shape: ${tensorShape.contentToString()}")
                // Create a scalar OnnxTensor for timestep
                val timestepScalarValue = timestep.toLong()
                val timestepBuffer = LongBuffer.allocate(1) // Allocate a buffer for one Long
                timestepBuffer.put(timestepScalarValue)     // Put the scalar value into the buffer
                timestepBuffer.rewind()                     // Reset buffer position to the beginning

                // Create an OnnxTensor with an empty shape (scalar)
                val timestepOnnxTensor = OnnxTensor.createTensor(
                    ortEnv,
                    timestepBuffer,
                    longArrayOf() // Empty shape for scalar
                )

                println("Timestep Onnx Tensor Shape: ${timestepOnnxTensor.info.shape.contentToString()}")


                val combinedOnnxTensor = OnnxTensor.createTensor(ortEnv, combinedBuffer, encodedPromptShape)
                println("combined Onnx Tensor Shape: ${combinedOnnxTensor.info.shape.contentToString()}")

                println("Input: $inputName")
                val inputs = mapOf(
                    "sample" to latentOnnxTensor,
                    "timestep" to timestepOnnxTensor,
                    "encoder_hidden_states" to combinedOnnxTensor
                )
                println("Running inferencing...")

                val output = ortSession.run(inputs)
                println("Completed inferencing")

                // 4. Process the output
                val noisePredTensor  = output?.get(0) as OnnxTensor
                val noisePredShape = noisePredTensor.info.shape
                println("Noise Prediction Shape: ${noisePredShape.contentToString()}")

                // Access the FloatBuffer while preserving shape information
                val noisePredBuffer = noisePredTensor.floatBuffer

                // No need to create a new FloatArray here

                // 5. Split noise prediction for guidance (using shape information)
                val batchSize = noisePredShape[0].toInt()
                val channelSize = noisePredShape[1].toInt()
                val height = noisePredShape[2].toInt()
                val width = noisePredShape[3].toInt()
                val channelDataSize = channelSize * height * width // Size of data for one batch
                val splitSize = batchSize / 2 // Assuming batch size is always even

                // Create buffers for the unconditional and conditional noise predictions
                val noisePredUncondBuffer = FloatBuffer.allocate(splitSize * channelDataSize)
                val noisePredTextBuffer = FloatBuffer.allocate(splitSize * channelDataSize)

                // Manually split the data based on the shape
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

                // Convert the split buffers to FloatArrays for guidance calculation
                val noisePredUncond = FloatArray(noisePredUncondBuffer.remaining())
                noisePredUncondBuffer.get(noisePredUncond)
                val noisePredText = FloatArray(noisePredTextBuffer.remaining())
                noisePredTextBuffer.get(noisePredText)

                // 6. Perform guidance
                val guidedNoisePred = noisePredUncond.mapIndexed { index, uncond ->
                    uncond + guidanceScale * (noisePredText[index] - uncond)
                }.toFloatArray()

                // 7. Compute the previous noisy sample x_t -> x_t-1 using the scheduler
                // Create a new tensor for guidedNoisePred with the correct shape
                val guidedNoisePredTensor = Tensor.fromBlob(
                    guidedNoisePred,
                    longArrayOf(splitSize.toLong(), channelSize.toLong(), height.toLong(), width.toLong())
                )
                // 8. Compute the previous noisy sample x_t -> x_t-1 using the scheduler
                latentTensor = scheduler.step(
                    guidedNoisePredTensor,
                    timestep,
                    latentTensor
                )

                println("Scaled Latents Tensor Shape: ${latentTensor.shape().contentToString()}")
                data = latentTensor.dataAsFloatArray
            }
        }
        ortSession.close()
        println("First 10 values: ${data.take(10).joinToString()}")
        println("Last 10 values: ${data.takeLast(10).joinToString()}")
        progressCallback(-1, -1)
        // --- VAE Decoding ---

        // 1. Scale the latents
        val scalingFactor = 1 / 0.18215f
        val scaledVAEData = data.map { it * scalingFactor }.toFloatArray()

        println("Scaled VAE Data (first 10 values): ${scaledVAEData.take(10).joinToString()}")
        println("Scaled VAE Data (last 10 values): ${scaledVAEData.takeLast(10).joinToString()}")

        val vaePath = FileUtils.getFilePath(context, "diffusion/vae_onnx.onnx")
        val vaeOrtSession = ortEnv.createSession(vaePath, OrtSession.SessionOptions())

        // 2. Create an ONNX tensor for the scaled latents
        val vaeLatentTensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(scaledVAEData),
            latentTensor.shape() // Use the shape from before denoising
        )

        // 3. Prepare the input for the VAE
        val vaeInputs = mapOf(
            vaeOrtSession.inputNames.first() to vaeLatentTensor
        )

        println("running vae inference")
        val vaeOutput = vaeOrtSession.run(vaeInputs)
        println("completed vae inference")

        val vaeOutputTensor = vaeOutput[0] as? OnnxTensor
        // Get and print the shape of the VAE output tensor
        val vaeOutputShape = vaeOutputTensor?.info?.shape
        println("VAE Output Shape: ${vaeOutputShape.contentToString()}")

        println("First 10 values: ${vaeOutputTensor?.floatBuffer?.array()?.take(10)?.joinToString()}")
        println("Last 10 values: ${vaeOutputTensor?.floatBuffer?.array()?.takeLast(10)?.joinToString()}")


        val vaeOutputData = vaeOutputTensor?.floatBuffer
        vaeOrtSession.close()
        ortEnv.close()
        // Convert VAE output to an image
        if(vaeOutputData != null && vaeOutputShape != null) {
            val image = convertToImage(vaeOutputData, vaeOutputShape)
            return image
        }

        return null
    }
    private fun convertToImage(data: FloatBuffer, shape: LongArray): Bitmap {
        // Assuming the output shape is [batchSize, channels, height, width]
        val batchSize = shape[0].toInt()
        val channels = shape[1].toInt()
        val height = shape[2].toInt()
        val width = shape[3].toInt()

        require(batchSize == 1) { "Only batch size 1 is supported" }

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Convert FloatBuffer to FloatArray for easier access
        val dataArray = FloatArray(data.remaining())
        data.get(dataArray)
        data.rewind() // Reset buffer position if needed elsewhere

        val numPixels = height * width

        for (h in 0 until height) {
            for (w in 0 until width) {
                val pixelIndex = h * width + w

                when (channels) {
                    3 -> {
                        // Calculate indices for each channel in NCHW layout
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
                        // Grayscale conversion
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