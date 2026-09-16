package com.example.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Locale
import kotlin.math.exp
import kotlin.math.roundToInt

/**
 * Ultra-Fast High-Accuracy Real-Time On-Device Object Detector powered by YOLOX-Nano (COCO 80).
 *
 * Employs an anchor-free decoupled head architecture with letterbox scaling (416x416),
 * multi-scale feature pyramid grids (strides 8, 16, 32 -> 3549 anchors), and Fast NMS.
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

    private data class Anchor(
        val gx: Float,
        val gy: Float,
        val stride: Float
    )

    private var interpreter: Interpreter? = null
    private val labels = mutableListOf<String>()

    val inputSize = 416

    // Preallocated Float32 input buffer [1, 416, 416, 3] in BGR format
    private val inputBuffer: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4).order(ByteOrder.nativeOrder())
    }
    private val pixelValues = IntArray(inputSize * inputSize)

    // Precomputed anchor grids for YOLOX-Nano (total 3549 anchors)
    // Stride 8: 52x52 = 2704, Stride 16: 26x26 = 676, Stride 32: 13x13 = 169 -> 3549
    private val anchors = ArrayList<Anchor>(3549)

    // Preallocated output buffers for both tensor layouts:
    // Layout A (PyTorch / LiteRT): [1, 3549, 85]
    private val outputBufferCL = Array(1) { Array(3549) { FloatArray(85) } }
    // Layout B (ONNX channels-first): [1, 85, 3549]
    private val outputBufferCF = Array(1) { Array(85) { FloatArray(3549) } }
    private var isChannelsFirst = false

    private val letterboxPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        initAnchors()
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

    private fun initAnchors() {
        anchors.clear()
        val strides = intArrayOf(8, 16, 32)
        for (s in strides) {
            val n = inputSize / s
            val sFloat = s.toFloat()
            for (gy in 0 until n) {
                val gyFloat = gy.toFloat()
                for (gx in 0 until n) {
                    anchors.add(Anchor(gx.toFloat(), gyFloat, sFloat))
                }
            }
        }
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
            labels.addAll(COCO_80_CLASSES)
        }
    }

    private fun getResolvedLabel(classId: Int): String {
        return COCO_80_CLASSES.getOrNull(classId)
            ?: labels.getOrNull(classId)
            ?: "Object"
    }

    private fun loadModel() {
        // Preferred candidate models in order of device compatibility:
        val modelCandidates = listOf(
            "yolox_nano_fp32.tflite",
            "yolox_nano.tflite"
        )

        for (modelFile in modelCandidates) {
            try {
                val assetList = context.assets.list("") ?: emptyArray()
                if (!assetList.contains(modelFile)) continue

                val fileDescriptor = context.assets.openFd(modelFile)
                val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
                val fileChannel = inputStream.channel
                val startOffset = fileDescriptor.startOffset
                val declaredLength = fileDescriptor.declaredLength
                val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

                // Initialize TFLite Interpreter with multi-threading
                val options = Interpreter.Options().apply {
                    setNumThreads(numThreads.coerceIn(2, 4))
                }

                val interp = try {
                    Interpreter(mappedByteBuffer, options)
                } catch (_: Throwable) {
                    // Fallback without XNNPACK delegate if hardware requires
                    val fallbackOptions = Interpreter.Options().apply {
                        setNumThreads(numThreads.coerceIn(1, 2))
                        setUseXNNPACK(false)
                    }
                    Interpreter(mappedByteBuffer, fallbackOptions)
                }

                // Inspect output tensor shape to determine layout:
                if (interp.outputTensorCount > 0) {
                    val outShape = interp.getOutputTensor(0).shape()
                    isChannelsFirst = (outShape.size == 3 && outShape[1] == 85 && outShape[2] == 3549)
                }

                interpreter = interp
                break // Successfully loaded
            } catch (_: Throwable) {
                // Try next model candidate
            }
        }
    }

    /**
     * Executes YOLOX-Nano Object Detection on the given bitmap.
     * @param bitmap Input bitmap of any resolution.
     * @param minScoreThreshold Minimum detection score (defaults to 0.25f for crisp real-time detection).
     * @return List of detections with precise normalized coordinates [0.0..1.0] and COCO 80 labels.
     */
    @Synchronized
    fun detectObjects(bitmap: Bitmap, minScoreThreshold: Float = 0.25f): List<Detection> {
        val interp = interpreter ?: return emptyList()
        val origW = bitmap.width
        val origH = bitmap.height
        if (origW < 32 || origH < 32) return emptyList()

        // 1. Calculate YOLOX letterbox scale factor (pad with gray 114)
        val scale = minOf(inputSize.toFloat() / origW, inputSize.toFloat() / origH)
        val scaledW = (origW * scale).roundToInt().coerceIn(1, inputSize)
        val scaledH = (origH * scale).roundToInt().coerceIn(1, inputSize)

        // 2. Render letterboxed image onto 416x416 canvas
        val letterboxBmp = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(letterboxBmp)
        canvas.drawColor(Color.rgb(114, 114, 114))

        val scaledBmp = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)
        canvas.drawBitmap(scaledBmp, 0f, 0f, letterboxPaint)
        if (scaledBmp != bitmap) {
            scaledBmp.recycle()
        }

        try {
            letterboxBmp.getPixels(pixelValues, 0, inputSize, 0, 0, inputSize, inputSize)
            inputBuffer.rewind()

            // 3. YOLOX requires BGR format, float32, range [0.0f..255.0f]
            for (p in pixelValues) {
                val r = ((p shr 16) and 0xFF).toFloat()
                val g = ((p shr 8) and 0xFF).toFloat()
                val b = (p and 0xFF).toFloat()
                inputBuffer.putFloat(b)
                inputBuffer.putFloat(g)
                inputBuffer.putFloat(r)
            }

            // 4. Run model inference
            val inputArray = arrayOf<Any>(inputBuffer)
            val outputMap = HashMap<Int, Any>()
            if (isChannelsFirst) {
                outputMap[0] = outputBufferCF
            } else {
                outputMap[0] = outputBufferCL
            }
            interp.runForMultipleInputsOutputs(inputArray, outputMap)

            // 5. Decode anchor-free decoupled head outputs
            val numAnchors = minOf(3549, anchors.size)
            val candidates = mutableListOf<Detection>()

            for (i in 0 until numAnchors) {
                val rawCx: Float
                val rawCy: Float
                val rawW: Float
                val rawH: Float
                val obj: Float

                if (isChannelsFirst) {
                    obj = outputBufferCF[0][4][i]
                    if (obj < 0.20f) continue
                    rawCx = outputBufferCF[0][0][i]
                    rawCy = outputBufferCF[0][1][i]
                    rawW = outputBufferCF[0][2][i]
                    rawH = outputBufferCF[0][3][i]
                } else {
                    obj = outputBufferCL[0][i][4]
                    if (obj < 0.20f) continue
                    rawCx = outputBufferCL[0][i][0]
                    rawCy = outputBufferCL[0][i][1]
                    rawW = outputBufferCL[0][i][2]
                    rawH = outputBufferCL[0][i][3]
                }

                // Find top-scoring class among 80 COCO classes
                var maxClassScore = 0f
                var bestClassIdx = 0
                for (c in 0 until 80) {
                    val score = if (isChannelsFirst) {
                        outputBufferCF[0][5 + c][i]
                    } else {
                        outputBufferCL[0][i][5 + c]
                    }
                    if (score > maxClassScore) {
                        maxClassScore = score
                        bestClassIdx = c
                    }
                }

                val finalScore = obj * maxClassScore
                if (finalScore < minScoreThreshold) continue

                // Anchor grid decoding:
                // cx = (raw_cx + gx) * stride
                // cy = (raw_cy + gy) * stride
                // w = exp(raw_w) * stride
                // h = exp(raw_h) * stride
                val anchor = anchors[i]
                val cx = (rawCx + anchor.gx) * anchor.stride
                val cy = (rawCy + anchor.gy) * anchor.stride
                val w = exp(rawW.toDouble().coerceIn(-10.0, 10.0)).toFloat() * anchor.stride
                val h = exp(rawH.toDouble().coerceIn(-10.0, 10.0)).toFloat() * anchor.stride

                // 416-space box coordinates:
                val x1 = cx - w / 2f
                val y1 = cy - h / 2f
                val x2 = cx + w / 2f
                val y2 = cy + h / 2f

                // Map back to original image space and normalize:
                val origX1 = x1 / scale
                val origY1 = y1 / scale
                val origX2 = x2 / scale
                val origY2 = y2 / scale

                val leftNorm = (origX1 / origW).coerceIn(0.005f, 0.98f)
                val topNorm = (origY1 / origH).coerceIn(0.005f, 0.98f)
                val rightNorm = (origX2 / origW).coerceIn(leftNorm + 0.015f, 0.995f)
                val bottomNorm = (origY2 / origH).coerceIn(topNorm + 0.015f, 0.995f)

                // Ignore spurious full-screen boxes (>95% frame coverage)
                if ((rightNorm - leftNorm) >= 0.95f && (bottomNorm - topNorm) >= 0.95f) continue

                val label = getResolvedLabel(bestClassIdx)
                candidates.add(
                    Detection(
                        classIndex = bestClassIdx,
                        label = label,
                        score = finalScore,
                        topNorm = topNorm,
                        leftNorm = leftNorm,
                        bottomNorm = bottomNorm,
                        rightNorm = rightNorm
                    )
                )
            }

            // 6. Fast Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
            return applyNMS(candidates, iouThreshold = 0.45f)
        } catch (_: Throwable) {
            return emptyList()
        } finally {
            letterboxBmp.recycle()
        }
    }

    private fun calculateIoU(a: Detection, b: Detection): Float {
        val interLeft = maxOf(a.leftNorm, b.leftNorm)
        val interTop = maxOf(a.topNorm, b.topNorm)
        val interRight = minOf(a.rightNorm, b.rightNorm)
        val interBottom = minOf(a.bottomNorm, b.bottomNorm)

        val interWidth = maxOf(0f, interRight - interLeft)
        val interHeight = maxOf(0f, interBottom - interTop)
        val interArea = interWidth * interHeight

        val areaA = (a.rightNorm - a.leftNorm) * (a.bottomNorm - a.topNorm)
        val areaB = (b.rightNorm - b.leftNorm) * (b.bottomNorm - b.topNorm)
        val unionArea = areaA + areaB - interArea

        return if (unionArea > 0f) interArea / unionArea else 0f
    }

    private fun applyNMS(candidates: List<Detection>, iouThreshold: Float = 0.45f): List<Detection> {
        if (candidates.isEmpty()) return emptyList()
        val sorted = candidates.sortedByDescending { it.score }
        val selected = mutableListOf<Detection>()

        for (candidate in sorted) {
            var suppress = false
            for (prev in selected) {
                if (candidate.classIndex == prev.classIndex && calculateIoU(candidate, prev) > iouThreshold) {
                    suppress = true
                    break
                }
            }
            if (!suppress) {
                selected.add(candidate)
                if (selected.size >= 10) break
            }
        }
        return selected
    }

    override fun close() {
        try {
            interpreter?.close()
        } catch (_: Throwable) {}
        interpreter = null
    }
}
