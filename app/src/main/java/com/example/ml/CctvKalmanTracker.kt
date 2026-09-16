package com.example.ml

import androidx.compose.ui.graphics.Color
import com.example.ui.components.LiveDetectedBox
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * 1D One Euro Filter for real-time bounding box jitter suppression.
 * Provides adaptive low-pass filtering:
 * - At low speed (stationary camera/object): applies heavy smoothing + deadband to freeze the box completely.
 * - At high speed (panning/moving): dynamically increases cutoff frequency to eliminate lag and overshoot.
 */
class OneEuroFilter(
    private val minCutoff: Float = 1.2f,  // Base cutoff frequency in Hz when stationary (clean jitter suppression)
    private val beta: Float = 0.025f,     // Speed coefficient for direct, responsive motion
    private val dCutoff: Float = 1.0f     // Derivative cutoff frequency in Hz
) {
    private var xPrev = 0f
    private var dxPrev = 0f
    private var tPrev = -1L
    private var initialized = false

    private fun computeAlpha(cutoff: Float, dt: Float): Float {
        val tau = 1.0f / (2.0f * Math.PI.toFloat() * cutoff)
        return 1.0f / (1.0f + tau / dt)
    }

    fun filter(x: Float, timestampMs: Long): Float {
        if (!initialized || tPrev < 0L) {
            initialized = true
            xPrev = x
            dxPrev = 0f
            tPrev = timestampMs
            return x
        }

        val dt = ((timestampMs - tPrev) / 1000f).coerceIn(0.005f, 0.25f)
        tPrev = timestampMs

        // Estimate velocity with derivative low-pass filter
        val dx = (x - xPrev) / dt
        val aD = computeAlpha(dCutoff, dt)
        val dxHat = aD * dx + (1f - aD) * dxPrev
        dxPrev = dxHat

        // Adaptive cutoff: low when still, higher during fast camera motion
        val cutoff = minCutoff + beta * abs(dxHat)
        val a = computeAlpha(cutoff, dt)
        val xHat = a * x + (1f - a) * xPrev
        xPrev = xHat

        return xHat
    }

    fun getValue(): Float = xPrev

    fun setValue(x: Float) {
        xPrev = x
        dxPrev = 0f
        initialized = true
    }

    fun reset(initialValue: Float = 0f) {
        initialized = false
        tPrev = -1L
        xPrev = initialValue
        dxPrev = 0f
    }
}

/**
 * Stable Bounding Box Track utilizing One Euro Filters and deadband stabilization.
 * Prevents rapid twitching, jumping, and position drift.
 */
class KalmanBoxTrack(
    var id: Int,
    initialBox: LiveDetectedBox,
    var timestampMs: Long = System.currentTimeMillis()
) {
    // 4 Independent One Euro Filters for normalized box coordinates
    private val filterL = OneEuroFilter().apply { setValue(initialBox.leftNorm) }
    private val filterT = OneEuroFilter().apply { setValue(initialBox.topNorm) }
    private val filterR = OneEuroFilter().apply { setValue(initialBox.rightNorm) }
    private val filterB = OneEuroFilter().apply { setValue(initialBox.bottomNorm) }

    var leftNorm: Float = initialBox.leftNorm
    var topNorm: Float = initialBox.topNorm
    var rightNorm: Float = initialBox.rightNorm
    var bottomNorm: Float = initialBox.bottomNorm

    // Velocity estimates for smooth coasting across missed frames
    private var vx: Float = 0f
    private var vy: Float = 0f

    // Metadata
    var label: String = initialBox.label
    var confidence: Float = initialBox.confidence
    var color: Color = initialBox.color
    var hits: Int = 1
    var timeSinceUpdate: Int = 0

    /**
     * Prediction / Coasting phase across temporary frame drops.
     */
    fun predict(currentTimestampMs: Long) {
        timeSinceUpdate++
        val dt = ((currentTimestampMs - timestampMs) / 1000f).coerceIn(0.01f, 0.15f)
        timestampMs = currentTimestampMs

        // Extrapolate smoothly along estimated velocity vector if temporarily missed
        if (timeSinceUpdate in 1..4 && (abs(vx) > 0.01f || abs(vy) > 0.01f)) {
            vx *= 0.85f
            vy *= 0.85f
            leftNorm = (leftNorm + vx * dt).coerceIn(0.005f, 0.95f)
            rightNorm = (rightNorm + vx * dt).coerceIn(0.02f, 0.995f)
            topNorm = (topNorm + vy * dt).coerceIn(0.005f, 0.95f)
            bottomNorm = (bottomNorm + vy * dt).coerceIn(0.02f, 0.995f)
        }
    }

    /**
     * Fluid measurement update without deadband freezing.
     * Continuously adapts filter cutoff based on instantaneous motion velocity.
     */
    fun update(measurement: LiveDetectedBox, currentTimestampMs: Long) {
        val dt = ((currentTimestampMs - timestampMs) / 1000f).coerceIn(0.01f, 0.25f)
        timestampMs = currentTimestampMs
        hits++
        timeSinceUpdate = 0

        val oldCx = (leftNorm + rightNorm) / 2f
        val oldCy = (topNorm + bottomNorm) / 2f

        // Continuous One Euro filtering: removes jitter while stationary, moves seamlessly during motion
        leftNorm = filterL.filter(measurement.leftNorm, currentTimestampMs)
        topNorm = filterT.filter(measurement.topNorm, currentTimestampMs)
        rightNorm = filterR.filter(measurement.rightNorm, currentTimestampMs)
        bottomNorm = filterB.filter(measurement.bottomNorm, currentTimestampMs)

        val newCx = (leftNorm + rightNorm) / 2f
        val newCy = (topNorm + bottomNorm) / 2f

        val instVx = (newCx - oldCx) / dt
        val instVy = (newCy - oldCy) / dt
        vx = vx * 0.4f + instVx * 0.6f
        vy = vy * 0.4f + instVy * 0.6f

        // Label stabilization with strong hysteresis
        if (measurement.label == label || measurement.confidence >= (confidence + 0.10f)) {
            label = measurement.label
            confidence = measurement.confidence
        } else {
            confidence = (confidence * 0.80f) + (measurement.confidence * 0.20f)
        }
        color = measurement.color
    }

    /**
     * Converts current state into a LiveDetectedBox with guaranteed crash-proof bounds.
     */
    fun toLiveDetectedBox(): LiveDetectedBox {
        val safeL = leftNorm.coerceIn(0.005f, 0.95f)
        val safeT = topNorm.coerceIn(0.005f, 0.95f)
        val minR = (safeL + 0.02f).coerceAtMost(0.995f)
        val minB = (safeT + 0.02f).coerceAtMost(0.995f)
        val safeR = rightNorm.coerceIn(minR, 0.995f)
        val safeB = bottomNorm.coerceIn(minB, 0.995f)

        return LiveDetectedBox(
            label = label,
            confidence = confidence,
            leftNorm = safeL,
            topNorm = safeT,
            rightNorm = safeR,
            bottomNorm = safeB,
            color = color,
            trackId = id
        )
    }
}

/**
 * Multi-Object Vision Tracker with One Euro Filter stabilization,
 * Hungarian/Greedy IoU association, and dropout coasting.
 */
class CctvKalmanTracker(
    private val maxCoastFrames: Int = 6, // Coast for ~200ms across detection dropouts
    private val minHitsToConfirm: Int = 1
) {
    private val tracks = mutableListOf<KalmanBoxTrack>()
    private var nextTrackId = 1

    /**
     * Processes new camera detections and returns rock-solid smoothed bounding boxes.
     */
    fun processFrame(
        detections: List<LiveDetectedBox>,
        timestampMs: Long = System.currentTimeMillis(),
        isSingleMode: Boolean = false
    ): List<LiveDetectedBox> {
        // Single Object Mode: Strictly maintain at most ONE track
        if (isSingleMode) {
            if (detections.isEmpty()) {
                if (tracks.isNotEmpty()) {
                    tracks[0].predict(timestampMs)
                    if (tracks[0].timeSinceUpdate > 3) {
                        tracks.clear()
                    }
                }
                return tracks.map { it.toLiveDetectedBox() }
            }

            val singleDet = detections.maxByOrNull { it.confidence } ?: detections.first()
            if (tracks.isEmpty()) {
                tracks.add(KalmanBoxTrack(id = 1, initialBox = singleDet, timestampMs = timestampMs))
            } else {
                // Keep only the single track
                while (tracks.size > 1) {
                    tracks.removeAt(tracks.lastIndex)
                }
                tracks[0].update(singleDet, timestampMs)
            }
            return listOf(tracks[0].toLiveDetectedBox())
        }

        // 1. Prediction Phase for all active tracks
        for (track in tracks) {
            track.predict(timestampMs)
        }

        // 2. Association Phase (Match Detections to Tracks)
        val matchedTrackIndices = mutableSetOf<Int>()
        val matchedDetectionIndices = mutableSetOf<Int>()

        // Greedy Best-Match Pairing
        for ((detIdx, det) in detections.withIndex()) {
            var bestTrackIdx = -1
            var bestScore = -1f

            val detCx = (det.leftNorm + det.rightNorm) / 2f
            val detCy = (det.topNorm + det.bottomNorm) / 2f
            val detW = max(0.02f, det.rightNorm - det.leftNorm)
            val detH = max(0.02f, det.bottomNorm - det.topNorm)

            for ((tIdx, track) in tracks.withIndex()) {
                if (tIdx in matchedTrackIndices) continue

                val trCx = (track.leftNorm + track.rightNorm) / 2f
                val trCy = (track.topNorm + track.bottomNorm) / 2f
                val trW = max(0.02f, track.rightNorm - track.leftNorm)
                val trH = max(0.02f, track.bottomNorm - track.topNorm)

                // Compute IoU
                val interL = max(det.leftNorm, track.leftNorm)
                val interT = max(det.topNorm, track.topNorm)
                val interR = min(det.rightNorm, track.rightNorm)
                val interB = min(det.bottomNorm, track.bottomNorm)
                val interArea = if (interR > interL && interB > interT) (interR - interL) * (interB - interT) else 0f
                val area1 = detW * detH
                val area2 = trW * trH
                val unionArea = area1 + area2 - interArea
                val iou = if (unionArea > 0f) interArea / unionArea else 0f

                // Center proximity
                val centerDist = hypot(detCx - trCx, detCy - trCy)
                val proximityScore = max(0f, 1.0f - centerDist.coerceIn(0f, 1f))

                val labelBonus = if (det.label == track.label) 0.35f else 0.0f
                val totalScore = (iou * 0.5f) + (proximityScore * 0.3f) + labelBonus

                if (totalScore > bestScore && (iou > 0.12f || centerDist < 0.28f || det.label == track.label)) {
                    bestScore = totalScore
                    bestTrackIdx = tIdx
                }
            }

            if (bestTrackIdx != -1) {
                matchedTrackIndices.add(bestTrackIdx)
                matchedDetectionIndices.add(detIdx)
                tracks[bestTrackIdx].update(det, timestampMs)
            }
        }

        // 3. Create new tracks for unmatched detections
        for ((detIdx, det) in detections.withIndex()) {
            if (detIdx !in matchedDetectionIndices) {
                tracks.add(KalmanBoxTrack(id = nextTrackId++, initialBox = det, timestampMs = timestampMs))
            }
        }

        // 4. Coasting & Pruning: Remove dead tracks that haven't been detected for maxCoastFrames
        tracks.removeAll { it.timeSinceUpdate > maxCoastFrames }

        // 5. Output Confirmed Tracks with safe bounds
        return tracks
            .filter { it.hits >= minHitsToConfirm }
            .map { it.toLiveDetectedBox() }
    }

    /**
     * Resets all active tracks (e.g. when camera mode or lens switches).
     */
    fun reset() {
        tracks.clear()
        nextTrackId = 1
    }
}
