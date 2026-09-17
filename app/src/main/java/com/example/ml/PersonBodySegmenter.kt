package com.example.ml

import android.graphics.Bitmap
import android.graphics.Point
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.SegmentationMask
import com.google.mlkit.vision.segmentation.Segmenter
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * High-Precision Neural Human Body Silhouette & Pose Contour Segmenter
 * Powered by Google ML Kit Selfie Segmentation.
 *
 * Accurately traces real human posture and silhouettes including:
 * - Raised arms, outstretched hands, and gesturing fingers
 * - Caps, hair, ears, head, neck, and facial perimeters
 * - Torso, coats, clothes, and visible body stature
 */
class PersonBodySegmenter(isStreamMode: Boolean = false) : AutoCloseable {

    private val options: SelfieSegmenterOptions = SelfieSegmenterOptions.Builder()
        .setDetectorMode(
            if (isStreamMode) SelfieSegmenterOptions.STREAM_MODE
            else SelfieSegmenterOptions.SINGLE_IMAGE_MODE
        )
        .build()

    private val segmenter: Segmenter = Segmentation.getClient(options)

    /**
     * Synchronously extracts high-definition body silhouette contour points from a bitmap.
     * Can be called from any background thread.
     */
    fun extractBodyContourSync(
        bitmap: Bitmap,
        targetBox: FaceBoundingBox? = null,
        confidenceThreshold: Float = 0.35f
    ): Pair<List<BiometricPoint>, String> {
        if (bitmap.isRecycled || bitmap.width < 16 || bitmap.height < 16) {
            return Pair(emptyList(), "Image unavailable")
        }

        return try {
            val inputImage = InputImage.fromBitmap(bitmap, 0)
            val mask: SegmentationMask = Tasks.await(segmenter.process(inputImage))

            val maskW = mask.width
            val maskH = mask.height
            val buffer = mask.buffer
            buffer.rewind()
            val floatBuf = buffer.asFloatBuffer()

            val totalPixels = maskW * maskH
            val rawBinary = BooleanArray(totalPixels)
            var foregroundPixelCount = 0

            for (i in 0 until totalPixels) {
                val conf = floatBuf.get(i)
                if (conf >= confidenceThreshold) {
                    rawBinary[i] = true
                    foregroundPixelCount++
                }
            }

            if (foregroundPixelCount < 40) {
                return Pair(emptyList(), "No silhouette detected")
            }

            // Morphological closing (dilation followed by erosion) to seal thin wrist/finger gaps
            val closedBinary = morphologicalClose(rawBinary, maskW, maskH)

            // Extract isolated component connected to targetBox or largest person blob (with 8-connected flood fill)
            val selectedMask = isolateTargetComponent(
                binaryMask = closedBinary,
                maskW = maskW,
                maskH = maskH,
                targetBox = targetBox
            )

            // Calculate min/max bounds of selected component
            var minX = maskW
            var maxX = 0
            var minY = maskH
            var maxY = 0
            var selectedPixels = 0

            for (y in 0 until maskH) {
                val row = y * maskW
                for (x in 0 until maskW) {
                    if (selectedMask[row + x]) {
                        selectedPixels++
                        if (x < minX) minX = x
                        if (x > maxX) maxX = x
                        if (y < minY) minY = y
                        if (y > maxY) maxY = y
                    }
                }
            }

            if (selectedPixels < 30) {
                return Pair(emptyList(), "No person segment isolated")
            }

            // Trace outer boundary using Moore-Neighbor 8-connected border following
            val rawContour = traceOuterContour(selectedMask, maskW, maskH)
            if (rawContour.size < 6) {
                return Pair(emptyList(), "Contour too small")
            }

            // Smooth & optimize points count (80 points for fluid 60fps rendering)
            val normalizedContour = optimizeAndSmoothContour(rawContour, maskW, maskH, targetCount = 80)

            val widthNorm = ((maxX - minX).toFloat() / maskW).coerceIn(0.05f, 1f)
            val heightNorm = ((maxY - minY).toFloat() / maskH).coerceIn(0.05f, 1f)
            val ratio = heightNorm / widthNorm

            // Determine if a raised hand or arm gesture is present
            val isRaisedArm = if (targetBox != null) {
                val headTopPx = (targetBox.topNorm * maskH).toInt()
                minY < (headTopPx + 5) || (minX < (targetBox.leftNorm * maskW - 15)) || (maxX > (targetBox.rightNorm * maskW + 15))
            } else {
                ratio < 1.15f
            }

            val posture = when {
                isRaisedArm -> "Raised Hand / Gesture Pose"
                ratio > 1.8f -> "Full Stature (Standing)"
                ratio > 1.2f -> "Upper Torso & Posture"
                else -> "Wide Stance / Gesturing Pose"
            }
            val diag = "$posture (${(heightNorm * 100).toInt()}%)"

            Pair(normalizedContour, diag)
        } catch (e: Throwable) {
            Pair(emptyList(), "Segmentation fallback: ${e.message ?: "error"}")
        }
    }

    /**
     * Suspend wrapper for coroutine callers.
     */
    suspend fun extractBodyContour(
        bitmap: Bitmap,
        targetBox: FaceBoundingBox? = null,
        confidenceThreshold: Float = 0.35f
    ): Pair<List<BiometricPoint>, String> = withContext(Dispatchers.Default) {
        extractBodyContourSync(bitmap, targetBox, confidenceThreshold)
    }

    /**
     * Morphological closing (dilation followed by erosion) bridges 1-2 pixel gaps at wrists,
     * watches, or sleeve cuffs while preserving full silhouette outline.
     */
    private fun morphologicalClose(mask: BooleanArray, w: Int, h: Int): BooleanArray {
        val dilated = BooleanArray(w * h)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                if (mask[row + x]) {
                    for (dy in -1..1) {
                        val ny = y + dy
                        if (ny in 0 until h) {
                            val nRow = ny * w
                            for (dx in -1..1) {
                                val nx = x + dx
                                if (nx in 0 until w) {
                                    dilated[nRow + nx] = true
                                }
                            }
                        }
                    }
                }
            }
        }

        val closed = BooleanArray(w * h)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                if (dilated[row + x]) {
                    var allNeighborsSet = true
                    for (dy in -1..1) {
                        val ny = y + dy
                        if (ny !in 0 until h) { allNeighborsSet = false; break }
                        val nRow = ny * w
                        for (dx in -1..1) {
                            val nx = x + dx
                            if (nx !in 0 until w || !dilated[nRow + nx]) {
                                allNeighborsSet = false
                                break
                            }
                        }
                        if (!allNeighborsSet) break
                    }
                    if (allNeighborsSet) {
                        closed[row + x] = true
                    }
                }
            }
        }
        return closed
    }

    /**
     * Isolates the connected component corresponding to the specified person or the primary person in the scene.
     * Uses 8-connected flood fill to naturally follow arms, raised hands, and outstretched gestures.
     */
    private fun isolateTargetComponent(
        binaryMask: BooleanArray,
        maskW: Int,
        maskH: Int,
        targetBox: FaceBoundingBox?
    ): BooleanArray {
        val total = maskW * maskH
        val visited = BooleanArray(total)
        val targetMask = BooleanArray(total)

        val targetCx = if (targetBox != null) ((targetBox.leftNorm + targetBox.rightNorm) * 0.5f * maskW).toInt() else maskW / 2
        val targetCy = if (targetBox != null) ((targetBox.topNorm + targetBox.bottomNorm) * 0.5f * maskH).toInt() else maskH / 2

        val boxL = if (targetBox != null) (targetBox.leftNorm * maskW).toInt() else 0
        val boxR = if (targetBox != null) (targetBox.rightNorm * maskW).toInt() else maskW
        val boxT = if (targetBox != null) (targetBox.topNorm * maskH).toInt() else 0
        val boxB = if (targetBox != null) (targetBox.bottomNorm * maskH).toInt() else maskH

        val components = mutableListOf<List<Int>>()

        for (y in 0 until maskH) {
            for (x in 0 until maskW) {
                val idx = y * maskW + x
                if (binaryMask[idx] && !visited[idx]) {
                    val comp = mutableListOf<Int>()
                    val queue = ArrayDeque<Int>()
                    queue.add(idx)
                    visited[idx] = true

                    while (queue.isNotEmpty()) {
                        val curr = queue.poll() ?: break
                        comp.add(curr)
                        val cx = curr % maskW
                        val cy = curr / maskW

                        // 8-connected neighbors so diagonal arm/hand connections are never severed
                        val neighbors = intArrayOf(
                            if (cx > 0) curr - 1 else -1,
                            if (cx < maskW - 1) curr + 1 else -1,
                            if (cy > 0) curr - maskW else -1,
                            if (cy < maskH - 1) curr + maskW else -1,
                            if (cx > 0 && cy > 0) curr - maskW - 1 else -1,
                            if (cx < maskW - 1 && cy > 0) curr - maskW + 1 else -1,
                            if (cx > 0 && cy < maskH - 1) curr + maskW - 1 else -1,
                            if (cx < maskW - 1 && cy < maskH - 1) curr + maskW + 1 else -1
                        )

                        for (nIdx in neighbors) {
                            if (nIdx in 0 until total && binaryMask[nIdx] && !visited[nIdx]) {
                                visited[nIdx] = true
                                queue.add(nIdx)
                            }
                        }
                    }

                    if (comp.size > 30) {
                        components.add(comp)
                    }
                }
            }
        }

        if (components.isEmpty()) {
            System.arraycopy(binaryMask, 0, targetMask, 0, total)
            return targetMask
        }

        // Score each component based on overlap with target box and size
        var bestComponent: List<Int>? = null
        var bestScore = -1f

        for (comp in components) {
            var overlapCount = 0
            var compMidX = 0f
            var compMidY = 0f

            for (c in comp) {
                val cx = c % maskW
                val cy = c / maskW
                compMidX += cx
                compMidY += cy
                if (cx in boxL..boxR && cy in boxT..boxB) {
                    overlapCount++
                }
            }
            compMidX /= comp.size
            compMidY /= comp.size

            val dist = sqrt((compMidX - targetCx) * (compMidX - targetCx) + (compMidY - targetCy) * (compMidY - targetCy))
            val score = if (targetBox != null) {
                overlapCount * 10f + (comp.size / (1f + dist * 0.05f))
            } else {
                comp.size.toFloat() / (1f + dist * 0.02f)
            }

            if (score > bestScore) {
                bestScore = score
                bestComponent = comp
            }
        }

        if (bestComponent != null) {
            for (idx in bestComponent) {
                targetMask[idx] = true
            }

            // Also merge any secondary component that is closely connected (e.g. raised hand nearby)
            val mainCompSet = bestComponent.toHashSet()
            for (comp in components) {
                if (comp === bestComponent) continue
                var minDistanceSq = Float.MAX_VALUE
                val step = max(1, comp.size / 20)
                for (i in comp.indices step step) {
                    val p = comp[i]
                    val px = p % maskW
                    val py = p / maskW
                    val mainStep = max(1, bestComponent.size / 30)
                    for (j in bestComponent.indices step mainStep) {
                        val m = bestComponent[j]
                        val mx = m % maskW
                        val my = m / maskW
                        val dSq = (px - mx) * (px - mx) + (py - my) * (py - my).toFloat()
                        if (dSq < minDistanceSq) minDistanceSq = dSq
                    }
                }
                // If secondary component is within ~8 pixels of main body (e.g. raised hand/mic), merge it!
                if (minDistanceSq < 64f) {
                    for (idx in comp) {
                        targetMask[idx] = true
                    }
                }
            }
        } else {
            System.arraycopy(binaryMask, 0, targetMask, 0, total)
        }

        return targetMask
    }

    /**
     * Traces the outer contour of a binary mask using Moore-Neighbor 8-connected boundary following.
     */
    private fun traceOuterContour(mask: BooleanArray, w: Int, h: Int): List<Point> {
        val padW = w + 2
        val padH = h + 2
        val padded = BooleanArray(padW * padH)

        for (y in 0 until h) {
            val srcRow = y * w
            val dstRow = (y + 1) * padW + 1
            for (x in 0 until w) {
                if (mask[srcRow + x]) {
                    padded[dstRow + x] = true
                }
            }
        }

        // Find starting pixel (first foreground pixel scanning from top to bottom)
        var startX = -1
        var startY = -1
        for (y in 1..h) {
            val row = y * padW
            for (x in 1..w) {
                if (padded[row + x]) {
                    startX = x
                    startY = y
                    break
                }
            }
            if (startX != -1) break
        }

        if (startX == -1) return emptyList()

        // Clockwise 8-neighborhood direction offsets:
        // 0: Left (-1, 0)
        // 1: Top-Left (-1, -1)
        // 2: Top (0, -1)
        // 3: Top-Right (1, -1)
        // 4: Right (1, 0)
        // 5: Bottom-Right (1, 1)
        // 6: Bottom (0, 1)
        // 7: Bottom-Left (-1, 1)
        val dx = intArrayOf(-1, -1, 0, 1, 1, 1, 0, -1)
        val dy = intArrayOf(0, -1, -1, -1, 0, 1, 1, 1)

        val contour = mutableListOf<Point>()
        var currX = startX
        var currY = startY
        var backtrackDir = 0 // Started from left neighbor

        val maxSteps = 4000
        var steps = 0

        contour.add(Point(currX - 1, currY - 1))

        while (steps < maxSteps) {
            steps++
            var foundNext = false
            for (i in 0 until 8) {
                val checkDir = (backtrackDir + 1 + i) % 8
                val nx = currX + dx[checkDir]
                val ny = currY + dy[checkDir]

                if (nx in 0 until padW && ny in 0 until padH && padded[ny * padW + nx]) {
                    currX = nx
                    currY = ny
                    backtrackDir = (checkDir + 4) % 8
                    contour.add(Point(currX - 1, currY - 1))
                    foundNext = true
                    break
                }
            }

            if (!foundNext) break
            if (currX == startX && currY == startY && contour.size > 3) {
                break
            }
        }

        return contour
    }

    /**
     * Resamples contour to target point count and applies Gaussian/moving average smoothing.
     */
    private fun optimizeAndSmoothContour(
        points: List<Point>,
        maskW: Int,
        maskH: Int,
        targetCount: Int = 80
    ): List<BiometricPoint> {
        if (points.isEmpty()) return emptyList()
        if (points.size < 4) {
            return points.map { BiometricPoint(it.x.toFloat() / maskW, it.y.toFloat() / maskH) }
        }

        val resampled = mutableListOf<BiometricPoint>()
        val totalPts = points.size
        val step = max(1f, totalPts.toFloat() / targetCount)

        var currIndexFloat = 0f
        while (currIndexFloat < totalPts && resampled.size < targetCount) {
            val idx = currIndexFloat.toInt().coerceIn(0, totalPts - 1)
            val pt = points[idx]
            val normX = (pt.x.toFloat() / maskW).coerceIn(0f, 1f)
            val normY = (pt.y.toFloat() / maskH).coerceIn(0f, 1f)
            resampled.add(BiometricPoint(normX, normY))
            currIndexFloat += step
        }

        val n = resampled.size
        if (n < 4) return resampled

        val smoothed = mutableListOf<BiometricPoint>()
        for (i in 0 until n) {
            val prev = resampled[(i - 1 + n) % n]
            val curr = resampled[i]
            val next = resampled[(i + 1) % n]
            val smX = prev.x * 0.25f + curr.x * 0.50f + next.x * 0.25f
            val smY = prev.y * 0.25f + curr.y * 0.50f + next.y * 0.25f
            smoothed.add(BiometricPoint(smX, smY))
        }

        return smoothed
    }

    override fun close() {
        try {
            segmenter.close()
        } catch (_: Throwable) {}
    }
}
