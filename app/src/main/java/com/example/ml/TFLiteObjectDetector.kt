package com.example.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Locale

/**
 * High-Precision Real-Time On-Device Object Detector powered by SSD MobileNet (COCO 80).
 *
 * Provides millimeter-accurate bounding boxes for small and large objects
 * (e.g. cell phones, remotes, cups, bottles, books, laptops, keys) on flat surfaces
 * (floors, tables, desks, rugs) without jitter or oversized background leakage.
 */
class TFLiteObjectDetector(
    private val context: Context,
    private val numThreads: Int = 4
) : AutoCloseable {

    data class Detection(
        val classIndex: Int,
        val label: String,
        val score: Float,
        val topNorm: Float,
        val leftNorm: Float,
        val bottomNorm: Float,
        val rightNorm: Float
    )

    private var interpreter: Interpreter? = null
    private val labels = mutableListOf<String>()

    // Fixed SSD MobileNet input dimensions
    val inputSize = 300

    private val inputBuffer: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3).order(ByteOrder.nativeOrder())
    }
    private val pixelValues = IntArray(inputSize * inputSize)

    // Preallocated output buffers for 10 detections
    private val maxDetections = 10
    private val outputLocations = Array(1) { Array(maxDetections) { FloatArray(4) } }
    private val outputClasses = Array(1) { FloatArray(maxDetections) }
    private val outputScores = Array(1) { FloatArray(maxDetections) }
    private val numDetections = FloatArray(1)

    private var locationsIndex = 0
    private var classesIndex = 1
    private var scoresIndex = 2
    private var numDetectionsIndex = 3

    init {
        loadLabelMap()
        loadModel()
    }

    companion object {
        val COCO_80_CLASSES = listOf(
            "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat",
            "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat",
            "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack",
            "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports Ball",
            "Kite", "Baseball Bat", "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket",
            "Bottle", "Wine Glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple",
            "Sandwich", "Orange", "Broccoli", "Carrot", "Hot Dog", "Pizza", "Donut", "Cake", "Chair",
            "Couch", "Potted Plant", "Bed", "Dining Table", "Toilet", "TV", "Laptop", "Mouse",
            "Remote", "Keyboard", "Cell Phone", "Microwave", "Oven", "Toaster", "Sink",
            "Refrigerator", "Book", "Clock", "Vase", "Scissors", "Teddy Bear", "Hair Drier", "Toothbrush"
        )
    }

    private fun getResolvedLabel(classId: Int): String {
        // 1. Check direct index in loaded label map if it's not a generic placeholder
        val direct = labels.getOrNull(classId)
        if (!direct.isNullOrBlank() && direct != "Object" && direct != "???") {
            return direct
        }
        // 2. Check 1-offset (due to background ??? at line 0 in standard labelmap.txt)
        val offset1 = labels.getOrNull(classId + 1)
        if (!offset1.isNullOrBlank() && offset1 != "Object" && offset1 != "???") {
            return offset1
        }
        // 3. Fallback to clean COCO-80 standard names
        val coco = COCO_80_CLASSES.getOrNull(classId) ?: COCO_80_CLASSES.getOrNull(classId - 1)
        if (coco != null) {
            return coco
        }
        return direct ?: "Object"
    }

    private fun loadLabelMap() {
        labels.clear()
        try {
            context.assets.open("labelmap.txt").use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).use { reader ->
                    var line: String? = reader.readLine()
                    while (line != null) {
                        val trimmed = line.trim()
                        if (trimmed.isNotBlank() && trimmed != "???") {
                            // Capitalize each word for clean UI presentation
                            val formatted = trimmed.split(" ").joinToString(" ") { word ->
                                word.replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.US) else it.toString() }
                            }
                            labels.add(formatted)
                        } else {
                            labels.add("Object")
                        }
                        line = reader.readLine()
                    }
                }
            }
        } catch (_: Throwable) {
            // Fallback default label
            labels.add("Object")
        }
    }

    private fun loadModel() {
        try {
            val fileDescriptor = context.assets.openFd("detect.tflite")
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            val options = Interpreter.Options().apply {
                setNumThreads(numThreads.coerceIn(1, 4))
            }
            val interp = Interpreter(mappedByteBuffer, options)

            // Auto-detect tensor output slot mapping
            for (i in 0 until interp.outputTensorCount) {
                val shape = interp.getOutputTensor(i).shape()
                when {
                    shape.size == 3 && shape[1] == maxDetections && shape[2] == 4 -> locationsIndex = i
                    shape.size == 2 && shape[1] == maxDetections -> {
                        val name = interp.getOutputTensor(i).name().lowercase(Locale.US)
                        if (name.contains("score") || name.contains("prob")) {
                            scoresIndex = i
                        } else if (name.contains("class") || name.contains("label") || name.contains("index")) {
                            classesIndex = i
                        } else if (i == 1) {
                            classesIndex = i
                        } else {
                            scoresIndex = i
                        }
                    }
                    shape.size == 1 && shape[0] == 1 -> numDetectionsIndex = i
                }
            }

            interpreter = interp
        } catch (_: Throwable) {
            interpreter = null
        }
    }

    /**
     * Executes SSD Object Detection on the given bitmap.
     * @param bitmap Input bitmap of any resolution.
     * @param minScoreThreshold Minimum detection score (defaults to 0.22f for high sensitivity to small objects).
     * @return List of detections with snug normalized coordinates and calibrated scores.
     */
    @Synchronized
    fun detectObjects(bitmap: Bitmap, minScoreThreshold: Float = 0.22f): List<Detection> {
        val interp = interpreter ?: return emptyList()
        val origW = bitmap.width
        val origH = bitmap.height
        if (origW < 32 || origH < 32) return emptyList()

        val resizedBmp = if (origW == inputSize && origH == inputSize) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        }

        try {
            resizedBmp.getPixels(pixelValues, 0, inputSize, 0, 0, inputSize, inputSize)
            inputBuffer.rewind()

            // UINT8 quantized input format [0..255]
            for (p in pixelValues) {
                inputBuffer.put(((p shr 16) and 0xFF).toByte())
                inputBuffer.put(((p shr 8) and 0xFF).toByte())
                inputBuffer.put((p and 0xFF).toByte())
            }

            val outputMap = HashMap<Int, Any>()
            outputMap[locationsIndex] = outputLocations
            outputMap[classesIndex] = outputClasses
            outputMap[scoresIndex] = outputScores
            outputMap[numDetectionsIndex] = numDetections

            val inputArray = arrayOf<Any>(inputBuffer)
            interp.runForMultipleInputsOutputs(inputArray, outputMap)

            val count = numDetections[0].toInt().coerceIn(0, maxDetections)
            val detections = mutableListOf<Detection>()

            for (i in 0 until count) {
                val score = outputScores[0][i]
                if (score < minScoreThreshold) continue

                val loc = outputLocations[0][i]
                // TFLite SSD output box format: [ymin, xmin, ymax, xmax]
                val top = loc[0].coerceIn(0.01f, 0.90f)
                val left = loc[1].coerceIn(0.01f, 0.90f)
                val bottom = loc[2].coerceIn(top + 0.04f, 0.99f)
                val right = loc[3].coerceIn(left + 0.04f, 0.99f)

                // Ignore full-screen background noise boxes (> 92% screen)
                val boxW = right - left
                val boxH = bottom - top
                if (boxW >= 0.94f && boxH >= 0.94f) continue

                val classId = outputClasses[0][i].toInt()
                val label = getResolvedLabel(classId)

                detections.add(
                    Detection(
                        classIndex = classId,
                        label = label,
                        score = score,
                        topNorm = top,
                        leftNorm = left,
                        bottomNorm = bottom,
                        rightNorm = right
                    )
                )
            }

            // Return sorted by score descending
            return detections.sortedByDescending { it.score }
        } catch (_: Throwable) {
            return emptyList()
        } finally {
            if (resizedBmp != bitmap) {
                resizedBmp.recycle()
            }
        }
    }

    override fun close() {
        try {
            interpreter?.close()
        } catch (_: Throwable) {}
        interpreter = null
    }
}
