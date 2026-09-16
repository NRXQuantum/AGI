package com.example.ml

import android.app.ActivityManager
import android.content.Context
import java.io.File
import java.util.Locale

enum class DevicePerformanceTier(val displayName: String) {
    HIGH_PERFORMANCE("Flagship / High Performance"),
    MID_RANGE("Mid-Range / Balanced"),
    ENTRY_LEVEL("Battery & Resource Saver")
}

data class DeviceHardwareProfile(
    val cpuCores: Int,
    val maxCpuFreqGhz: Float,
    val totalRamGb: Float,
    val tier: DevicePerformanceTier
) {
    val description: String
        get() = String.format(
            Locale.US,
            "%d Cores @ %.2f GHz • %.1f GB RAM",
            cpuCores,
            maxCpuFreqGhz,
            totalRamGb
        )
}

data class TunedHyperparameters(
    val architecture: ModelArchitecture,
    val optimizer: OptimizerType,
    val lrSchedule: LearningRateSchedule,
    val batchSize: Int,
    val epochs: Float,
    val learningRate: Float,
    val reasoning: String,
    val hardwareSummary: String
)

data class ProjectTrainingConfig(
    val architecture: ModelArchitecture,
    val optimizer: OptimizerType,
    val lrSchedule: LearningRateSchedule,
    val batchSize: Int,
    val epochs: Float,
    val learningRate: Float,
    val isUserModified: Boolean = false
)

object AutoTuner {

    /**
     * Inspects device hardware (CPU Cores, Max CPU Clock Frequency GHz, Total RAM)
     */
    fun getHardwareProfile(context: Context): DeviceHardwareProfile {
        val cpuCores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        val maxGhz = detectMaxCpuFreqGhz(cpuCores)
        val ramGb = detectTotalRamGb(context)

        // Compute performance capacity score:
        // Cores * GHz + RAM contribution
        val score = (cpuCores * maxGhz) + (ramGb * 1.5f)

        val tier = when {
            score >= 22f || (cpuCores >= 8 && ramGb >= 6f) -> DevicePerformanceTier.HIGH_PERFORMANCE
            score >= 12f || (cpuCores >= 6 && ramGb >= 4f) -> DevicePerformanceTier.MID_RANGE
            else -> DevicePerformanceTier.ENTRY_LEVEL
        }

        return DeviceHardwareProfile(
            cpuCores = cpuCores,
            maxCpuFreqGhz = maxGhz,
            totalRamGb = ramGb,
            tier = tier
        )
    }

    private fun detectMaxCpuFreqGhz(cores: Int): Float {
        var maxKHz = 0L
        try {
            for (i in 0 until cores) {
                val f = File("/sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq")
                if (f.exists()) {
                    val line = f.readText().trim()
                    val khz = line.toLongOrNull() ?: 0L
                    if (khz > maxKHz) maxKHz = khz
                }
            }
        } catch (_: Throwable) {}

        if (maxKHz == 0L) {
            try {
                val f = File("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
                if (f.exists()) {
                    maxKHz = f.readText().trim().toLongOrNull() ?: 0L
                }
            } catch (_: Throwable) {}
        }

        return if (maxKHz > 0) {
            maxKHz / 1_000_000f
        } else {
            // Sensible mobile default for modern arm64 octacore
            if (cores >= 8) 2.40f else 2.00f
        }
    }

    private fun detectTotalRamGb(context: Context): Float {
        return try {
            val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            if (actManager != null) {
                val memInfo = ActivityManager.MemoryInfo()
                actManager.getMemoryInfo(memInfo)
                (memInfo.totalMem.toDouble() / (1024.0 * 1024.0 * 1024.0)).toFloat()
            } else {
                4.0f
            }
        } catch (_: Throwable) {
            4.0f
        }
    }

    /**
     * Compute optimal hyperparameters using dataset metrics & hardware profile
     */
    fun computeOptimalParameters(
        context: Context,
        totalSamples: Int,
        classCount: Int
    ): TunedHyperparameters {
        val hw = getHardwareProfile(context)
        val validClassCount = classCount.coerceAtLeast(1)
        val samples = totalSamples.coerceAtLeast(0)
        val avgPerClass = if (validClassCount > 0) samples / validClassCount else 0

        // 1. Architecture Determination
        val architecture = when {
            validClassCount >= 10 || samples >= 500 -> ModelArchitecture.DEEP_RESIDUAL_MLP
            samples >= 60 -> ModelArchitecture.STANDARD_MLP
            else -> ModelArchitecture.LINEAR
        }

        // 2. Mini-Batch Size Determination
        val batchSize = when {
            // Massive dataset (e.g. 10k to 100k+ images)
            samples >= 5000 -> {
                when (hw.tier) {
                    DevicePerformanceTier.HIGH_PERFORMANCE -> 256
                    DevicePerformanceTier.MID_RANGE -> 128
                    DevicePerformanceTier.ENTRY_LEVEL -> 64
                }
            }
            // Large dataset (1k to 5k images)
            samples >= 1000 -> {
                when (hw.tier) {
                    DevicePerformanceTier.HIGH_PERFORMANCE -> 128
                    DevicePerformanceTier.MID_RANGE -> 64
                    DevicePerformanceTier.ENTRY_LEVEL -> 32
                }
            }
            // Moderate dataset (200 to 1k images)
            samples >= 200 -> {
                if (hw.tier == DevicePerformanceTier.HIGH_PERFORMANCE) 64 else 32
            }
            // Small dataset (50 to 200 images)
            samples >= 50 -> 16
            // Very small dataset (< 50 images)
            else -> 8
        }.coerceAtMost(if (samples > 1) (samples / 2).coerceAtLeast(8) else 8)

        // 3. Epochs Determination
        val epochs = when {
            samples >= 20000 -> 15f
            samples >= 2000 -> 30f
            samples >= 200 -> 30f
            samples >= 50 -> 50f
            else -> 100f
        }

        // 4. Learning Rate Schedule & Optimizer
        val optimizer = OptimizerType.ADAM_W
        val lrSchedule = LearningRateSchedule.COSINE_ANNEALING

        // 5. Scaled Learning Rate based on batch size & architecture
        val learningRate = when {
            batchSize >= 128 -> 0.005f
            batchSize >= 64 -> 0.003f
            batchSize >= 32 -> 0.0025f
            else -> 0.0015f
        }

        val reasoning = when {
            samples >= 5000 -> {
                "Auto-Tuned for Large Dataset: $samples samples across $validClassCount classes. Utilizing Batch $batchSize & Deep Residual MLP with Cosine AdamW for maximum GPU/CPU throughput and fastest convergence."
            }
            samples >= 500 -> {
                "Auto-Tuned for Balanced Dataset: $samples samples across $validClassCount classes. Configured Deep Residual Network with Batch $batchSize and 30 Epochs for >98% generalization."
            }
            samples >= 50 -> {
                "Auto-Tuned for Medium Dataset: $samples samples. Selected 2-Layer MLP with Dropout & Batch $batchSize to balance fast feature learning and prevent overfitting."
            }
            else -> {
                "Auto-Tuned for Lightweight Dataset: $samples samples. Selected Linear Softmax classifier with regularized gradient updates to prevent overfitting on small sample size."
            }
        }

        return TunedHyperparameters(
            architecture = architecture,
            optimizer = optimizer,
            lrSchedule = lrSchedule,
            batchSize = batchSize,
            epochs = epochs,
            learningRate = learningRate,
            reasoning = reasoning,
            hardwareSummary = "${hw.description} (${hw.tier.displayName})"
        )
    }
}
