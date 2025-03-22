package com.example.edgediffusionv14.diffusion.models

import android.content.Context
import java.io.BufferedReader
import java.io.InputStreamReader
import org.json.JSONObject
import java.lang.StringBuilder
import java.text.Normalizer

/**
 * A simple wrapper class for token encoding results following
 * HuggingFace's BatchEncoding interface, simplified for mobile use.
 */
class BatchEncoding(private val encoding: Map<String, Any>) {
    /**
     * Retrieves a value by key from the encoding map
     */
    operator fun get(key: String): Any? = encoding[key]

    /**
     * Returns all keys in the encoding map
     */
    fun keys(): Set<String> = encoding.keys
}

/**
 * A minimal implementation of CLIP tokenizer for text encoding in mobile environments.
 * This tokenizer applies Byte-Pair Encoding (BPE) for text-to-image diffusion models.
 *
 * @param context The Android application context
 * @param vocabFileName Path to the vocabulary file in assets
 * @param mergesFileName Path to the BPE merges file in assets
 */
class MinimalCLIPTokenizer(
    val context: Context,
    vocabFileName: String,
    mergesFileName: String
) {
    // Token-ID mappings
    private val encoder: MutableMap<String, Int> = mutableMapOf()
    private val decoder: MutableMap<Int, String> = mutableMapOf()

    // BPE merge rankings used for tokenization algorithm
    private val bpeRanks: MutableMap<Pair<String, String>, Int> = mutableMapOf()

    // Byte-level encoders for UTF-8 handling
    private val byteEncoder: Map<Int, String>
    private val byteDecoder: Map<String, Int>

    // Cache for previously processed tokens to improve performance
    private val cache: MutableMap<String, String> = mutableMapOf(
        "<|startoftext|>" to "<|startoftext|>",
        "<|endoftext|>" to "<|endoftext|>"
    )

    // Special tokens configuration
    private val bosToken = "<|startoftext|>"  // Beginning of sequence token
    private val eosToken = "<|endoftext|>"    // End of sequence token
    private val padToken = "<|endoftext|>"    // Padding token (same as EOS)

    // Token IDs for special tokens
    private val bosTokenId: Int
    private val eosTokenId: Int
    private val padTokenId: Int

    // Regex pattern for initial tokenization of text
    private val pat = Regex(
        """<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
        RegexOption.IGNORE_CASE
    )

    init {
        // Load and parse vocabulary file
        val vocabInputStream = context.assets.open(vocabFileName)
        val vocabReader = BufferedReader(InputStreamReader(vocabInputStream, "UTF-8"))
        val vocabJson = JSONObject(vocabReader.readText())

        // Populate encoder map from JSON
        for (key in vocabJson.keys()) {
            encoder[key] = vocabJson.getInt(key)
        }
        vocabReader.close()
        vocabInputStream.close()

        // Create decoder map (inverse of encoder for lookup by ID)
        decoder.putAll(encoder.entries.associate { (k, v) -> v to k })

        // Load BPE merges file
        val mergesInputStream = context.assets.open(mergesFileName)
        val mergesReader = BufferedReader(InputStreamReader(mergesInputStream, "UTF-8"))

        // Skip header line and load merges (up to vocabulary size limit)
        val bpeMerges = mergesReader.readLines().drop(1).take(49152 - 256 - 2 + 1)
        mergesReader.close()
        mergesInputStream.close()

        // Build BPE ranks map (index determines merge priority)
        bpeMerges.forEachIndexed { index, merge ->
            val parts = merge.split(" ")
            bpeRanks[Pair(parts[0], parts[1])] = index
        }

        // Initialize byte encoding maps for UTF-8 handling
        byteEncoder = bytesToUnicode()
        byteDecoder = byteEncoder.entries.associate { (k, v) -> v to k }

        // Get IDs for special tokens from vocabulary
        bosTokenId = encoder[bosToken]!!
        eosTokenId = encoder[eosToken]!!
        padTokenId = encoder[padToken]!!
    }

    /**
     * Creates a mapping from bytes to unicode characters.
     * This ensures all UTF-8 bytes can be represented as single characters,
     * avoiding tokenization issues with certain byte values.
     *
     * @return Map from byte values to unicode character strings
     */
    private fun bytesToUnicode(): Map<Int, String> {
        // Start with printable ASCII and Latin-1 Supplement characters
        val bs = mutableListOf<Int>().apply {
            // ASCII printable characters (33-126)
            addAll(('!'.code..'~'.code).toList())
            // Latin-1 Supplement characters
            addAll(('¡'.code..'¬'.code).toList())
            addAll(('®'.code..'ÿ'.code).toList())
        }

        val cs = bs.toMutableList()
        var n = 0

        // Map remaining bytes (0-255) to Unicode characters beyond ASCII
        for (b in 0 until 256) {
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n += 1
            }
        }

        val chars = cs.map { it.toChar().toString() }
        return bs.zip(chars).toMap()
    }

    /**
     * Extracts all adjacent pairs of tokens from a word.
     * Used as a helper for the BPE algorithm.
     *
     * @param word List of tokens to extract pairs from
     * @return Set of all adjacent token pairs
     */
    fun getPairs(word: List<String>): Set<Pair<String, String>> {
        val pairs = mutableSetOf<Pair<String, String>>()
        var prevChar = word[0]

        for (i in 1 until word.size) {
            pairs.add(Pair(prevChar, word[i]))
            prevChar = word[i]
        }

        return pairs
    }

    /**
     * Applies Byte-Pair Encoding to a token.
     * This is the core algorithm that recursively merges character pairs
     * according to the learned BPE ranks until no more merges are possible.
     *
     * @param token The string token to encode
     * @return BPE-encoded string
     */
    fun bpe(token: String): String {
        // Return cached result if available for performance
        if (cache.containsKey(token)) {
            return cache[token]!!
        }

        // Convert token to a list of characters and mark end of word
        var word = token.dropLast(1).map { it.toString() } + (token.last().toString() + "</w>")
        var pairs = getPairs(word)

        // If no pairs found, just return with end marker
        if (pairs.isEmpty()) {
            return token + "</w>"
        }

        // Main BPE algorithm loop
        while (true) {
            // Find the highest ranked (lowest value) bigram
            val bigram = pairs.minByOrNull { bpeRanks.getOrDefault(it, Int.MAX_VALUE) } ?: break

            // If bigram not in vocabulary, we're done
            if (!bpeRanks.containsKey(bigram)) {
                break
            }

            // Extract the pair to be merged
            val (first, second) = bigram
            val newWord = mutableListOf<String>()
            var i = 0

            // Merge selected bigram throughout the word
            while (i < word.size) {
                // Find next occurrence of first part of bigram
                val j = word.subList(i, word.size).indexOf(first).takeIf { it >= 0 }?.plus(i) ?: word.size

                // Add any skipped tokens
                if (j > i) {
                    newWord.addAll(word.subList(i, j))
                }
                i = j

                // If we found the bigram, merge it
                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else if (i < word.size) {
                    // Otherwise add current token and move on
                    newWord.add(word[i])
                    i += 1
                }
            }

            // Update our working word and find new pairs
            word = newWord
            pairs = getPairs(word)

            // If word reduced to a single token, we're done
            if (word.size == 1) {
                break
            }
        }

        // Cache and return result
        val result = word.joinToString(" ")
        cache[token] = result
        return result
    }

    /**
     * Normalizes and cleans text input for tokenization.
     * Handles whitespace and special characters uniformly.
     *
     * @param text The raw input text
     * @return Cleaned text ready for tokenization
     */
    private fun cleanText(text: String): String {
        val output = StringBuilder()

        for (char in text) {
            val code = char.code
            // Skip null bytes and replacement characters
            if (code == 0 || code == 0xfffd) {
                continue
            }
            // Normalize all whitespace to plain spaces
            if (char.isWhitespace()) {
                output.append(' ')
            } else {
                output.append(char)
            }
        }

        return output.toString()
    }

    /**
     * Converts text into BPE tokens.
     * This breaks text into initial tokens and then applies BPE.
     *
     * @param text The input text to tokenize
     * @return List of BPE tokens
     */
    private fun tokenize(text: String): List<String> {
        val bpeTokens = mutableListOf<String>()

        // Clean, normalize, and prepare the text
        val cleanedText = cleanText(text)
        val normalizedText = Normalizer.normalize(cleanedText, Normalizer.Form.NFC)
        val trimmedText = normalizedText.replace("\\s+".toRegex(), " ").trim().lowercase()

        // Split text into initial tokens using regex pattern
        for (token in pat.findAll(trimmedText).map { it.value }) {
            // Convert each token to bytes and then to Unicode characters
            val encodedToken = token.toByteArray(Charsets.UTF_8).joinToString("") {
                byteEncoder[it.toInt() and 0xff]!!
            }

            // Apply BPE to the token and add resulting subtokens
            bpeTokens.addAll(bpe(encodedToken).split(" "))
        }

        return bpeTokens
    }

    /**
     * Encodes text to token IDs.
     * Converts text into a sequence of vocabulary indices with special tokens.
     *
     * @param text The input text to encode
     * @return List of token IDs
     */
    private fun encode(text: String): List<Int> {
        // Handle empty string case
        if (text.isEmpty()) {
            return listOf(bosTokenId, eosTokenId)
        }

        // Tokenize text
        val bpeTokens = tokenize(text)

        // Convert tokens to IDs and add special tokens
        val ids = mutableListOf(bosTokenId)
        ids.addAll(bpeTokens.map { encoder.getOrDefault(it, eosTokenId) })
        ids.add(eosTokenId)

        return ids
    }

    /**
     * Main method to encode batches of text with padding options.
     * This is the public API that processes multiple texts at once.
     *
     * @param text List of input texts to encode
     * @param padding Padding strategy ("max_length" or "none")
     * @param maxLength Maximum sequence length
     * @param truncation Whether to truncate sequences exceeding maxLength
     * @return BatchEncoding containing input_ids and attention_mask
     */
    operator fun invoke(
        text: List<String>,
        padding: String = "max_length",
        maxLength: Int = 77,
        truncation: Boolean = true
    ): BatchEncoding {
        val batchInputIds = mutableListOf<List<Int>>()
        val batchAttentionMask = mutableListOf<List<Int>>()

        for (singleText in text) {
            // Encode the text
            var inputIds = encode(singleText)

            // Truncate if needed and requested
            if (truncation && inputIds.size > maxLength) {
                inputIds = inputIds.take(maxLength - 1) + listOf(eosTokenId)
            }

            // Create attention mask (1 for tokens, 0 for padding)
            val attentionMask = MutableList(inputIds.size) { 1 }

            // Apply padding if requested
            if (padding == "max_length") {
                val paddingLength = maxLength - inputIds.size
                if (paddingLength > 0) {
                    inputIds = inputIds + List(paddingLength) { padTokenId }
                    attentionMask.addAll(List(paddingLength) { 0 })
                }
            }

            batchInputIds.add(inputIds)
            batchAttentionMask.add(attentionMask)
        }

        // Return as BatchEncoding object
        return BatchEncoding(
            mapOf(
                "input_ids" to batchInputIds,
                "attention_mask" to batchAttentionMask
            )
        )
    }

    /**
     * Clears the BPE cache to help with memory management.
     * Useful for long-running applications to prevent memory leaks.
     */
    fun clearCache() {
        cache.clear()
        // Preserve cache entries for special tokens
        cache["<|startoftext|>"] = "<|startoftext|>"
        cache["<|endoftext|>"] = "<|endoftext|>"
    }
}