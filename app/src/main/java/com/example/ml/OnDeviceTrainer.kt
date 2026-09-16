package com.example.ml

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.Random
import kotlin.math.*

data class TrainingSample(
    val features: FloatArray,
    val classIndex: Int,
    val classLabel: String
)

enum class TrainingPhase(val title: String) {
    IDLE("Idle"),
    EXTRACTING_FEATURES("Extracting Image Features"),
    STANDARDIZING_FEATURES("Standardizing Feature Vectors"),
    TRAINING_NEURAL_NET("Training Deep Neural Network"),
    FINALIZING_MODEL("Saving Model Weights & Scaler"),
    COMPLETED("Training Complete"),
    CANCELLED("Training Cancelled"),
    ERROR("Error")
}

data class TrainingProgress(
    val currentEpoch: Int = 0,
    val totalEpochs: Int = 0,
    val loss: Float = 0f,
    val accuracy: Float = 0f,
    val statusMessage: String = "",
    val overallPercentage: Float = 0f, // 0.00% to 100.00%
    val phase: TrainingPhase = TrainingPhase.IDLE,
    val currentStep: Int = 0,
    val totalSteps: Int = 0,
    val elapsedSeconds: Long = 0L,
    val estimatedRemainingSeconds: Long = 0L,
    val speedText: String = ""
)

data class DetectedObjectRegion(
    val classIndex: Int,
    val classLabel: String,
    val confidence: Float,
    val boxLeftNorm: Float,    // 0.0f - 1.0f relative to image width
    val boxTopNorm: Float,     // 0.0f - 1.0f relative to image height
    val boxRightNorm: Float,   // 0.0f - 1.0f relative to image width
    val boxBottomNorm: Float,  // 0.0f - 1.0f relative to image height
    val regionTitle: String
)

data class PredictionResult(
    val classIndex: Int,
    val classLabel: String,
    val confidence: Float,
    val allProbabilities: List<ClassConfidence>,
    val inferenceTimeMs: Long,
    val detectedObjects: List<DetectedObjectRegion> = emptyList()
)

data class ClassConfidence(
    val classIndex: Int,
    val classLabel: String,
    val probability: Float
)

enum class ModelArchitecture(val displayName: String, val subtitle: String) {
    DEEP_RESIDUAL_MLP("Deep Neural Network (3 Layers + Residual)", "Multi-layer depth with LayerNorm, GELU & Skip Connections (Recommended for 200 classes)"),
    STANDARD_MLP("2-Layer MLP Network", "Hidden layer with ReLU and Dropout for fast non-linear convergence"),
    LINEAR("Linear Softmax Classifier", "Single-layer classifier for small datasets")
}

enum class OptimizerType(val displayName: String) {
    ADAM_W("AdamW (Adaptive Moment + Decoupled Weight Decay)"),
    MOMENTUM_SGD("SGD with Nesterov Momentum (0.9)"),
    VANILLA_SGD("Standard SGD")
}

enum class LearningRateSchedule(val displayName: String) {
    COSINE_ANNEALING("Cosine Annealing with Warmup"),
    STEP_DECAY("Step Decay (0.5x every 10 epochs)"),
    CONSTANT("Constant Learning Rate")
}

/**
 * Standard Scaler for feature normalization (Z-score standardization: (x - mean) / std).
 * Projects features onto a well-conditioned space with unit variance and zero mean,
 * followed by L2 unit-sphere projection for optimal classification margins.
 */
data class FeatureScaler(
    val means: FloatArray,
    val stds: FloatArray
) {
    fun transformInPlace(features: FloatArray) {
        var sumSq = 0f
        for (j in features.indices) {
            val mean = if (j < means.size) means[j] else 0f
            val std = if (j < stds.size) stds[j] else 1f
            val z = (features[j] - mean) / std.coerceAtLeast(1e-5f)
            features[j] = z
            sumSq += z * z
        }
        val norm = sqrt(sumSq).coerceAtLeast(1e-6f)
        for (j in features.indices) {
            features[j] /= norm
        }
    }

    fun transform(features: FloatArray): FloatArray {
        val scaled = FloatArray(features.size)
        var sumSq = 0f
        for (j in features.indices) {
            val mean = if (j < means.size) means[j] else 0f
            val std = if (j < stds.size) stds[j] else 1f
            val z = (features[j] - mean) / std.coerceAtLeast(1e-5f)
            scaled[j] = z
            sumSq += z * z
        }
        val norm = sqrt(sumSq).coerceAtLeast(1e-6f)
        for (j in scaled.indices) {
            scaled[j] /= norm
        }
        return scaled
    }

    companion object {
        fun fitFromTrainingSamples(samples: List<TrainingSample>, featureDim: Int = 128): FeatureScaler {
            if (samples.isEmpty()) {
                return FeatureScaler(FloatArray(featureDim) { 0f }, FloatArray(featureDim) { 1f })
            }
            val means = FloatArray(featureDim)
            val stds = FloatArray(featureDim)
            val n = samples.size.toFloat()

            for (sample in samples) {
                val f = sample.features
                for (j in 0 until featureDim) {
                    if (j < f.size) {
                        means[j] += f[j]
                    }
                }
            }
            for (j in 0 until featureDim) {
                means[j] /= n
            }

            for (sample in samples) {
                val f = sample.features
                for (j in 0 until featureDim) {
                    if (j < f.size) {
                        val diff = f[j] - means[j]
                        stds[j] += diff * diff
                    }
                }
            }
            for (j in 0 until featureDim) {
                stds[j] = sqrt(stds[j] / n).coerceAtLeast(1e-4f)
            }

            return FeatureScaler(means, stds)
        }

        fun fit(samples: List<FloatArray>, featureDim: Int = 128): FeatureScaler {
            if (samples.isEmpty()) {
                return FeatureScaler(FloatArray(featureDim) { 0f }, FloatArray(featureDim) { 1f })
            }
            val means = FloatArray(featureDim)
            val stds = FloatArray(featureDim)
            val n = samples.size.toFloat()

            for (sample in samples) {
                for (j in 0 until featureDim) {
                    if (j < sample.size) {
                        means[j] += sample[j]
                    }
                }
            }
            for (j in 0 until featureDim) {
                means[j] /= n
            }

            for (sample in samples) {
                for (j in 0 until featureDim) {
                    if (j < sample.size) {
                        val diff = sample[j] - means[j]
                        stds[j] += diff * diff
                    }
                }
            }
            for (j in 0 until featureDim) {
                stds[j] = sqrt(stds[j] / n).coerceAtLeast(1e-4f)
            }

            return FeatureScaler(means, stds)
        }
    }
}

/**
 * High-Performance On-Device Deep Neural Network Classifier & Trainer.
 * Supports:
 * - Deep Multi-Layer Perceptron (3 Layers) with Layer Normalization, GELU activations, and Residual Skip Connections.
 * - AdamW Optimizer with Decoupled Weight Decay.
 * - Cosine Annealing Learning Rate Schedule with Warmup.
 * - Label Smoothing Cross-Entropy Loss for high-class datasets (e.g. 200 classes).
 * - Mini-Batch Vectorized Backpropagation with Gradient Clipping.
 */
class OnDeviceTrainer(
    val numClasses: Int,
    val featureDim: Int = 128,
    val classLabels: List<String>,
    var architecture: ModelArchitecture = ModelArchitecture.DEEP_RESIDUAL_MLP
) {
    // Hidden Layer Dimensions
    val h1Dim: Int = if (numClasses > 50) 256 else 128
    val h2Dim: Int = if (numClasses > 50) 128 else 64

    // Deep Neural Network Weights & Biases
    // Layer 1: [h1Dim][featureDim]
    var w1: Array<FloatArray> = Array(h1Dim) { FloatArray(featureDim) }
    var b1: FloatArray = FloatArray(h1Dim)
    var gamma1: FloatArray = FloatArray(h1Dim) { 1.0f }
    var beta1: FloatArray = FloatArray(h1Dim) { 0.0f }

    // Layer 2: [h2Dim][h1Dim]
    var w2: Array<FloatArray> = Array(h2Dim) { FloatArray(h1Dim) }
    var b2: FloatArray = FloatArray(h2Dim)
    var gamma2: FloatArray = FloatArray(h2Dim) { 1.0f }
    var beta2: FloatArray = FloatArray(h2Dim) { 0.0f }

    // Residual Skip projection: [h2Dim][h1Dim]
    var wSkip: Array<FloatArray> = Array(h2Dim) { FloatArray(h1Dim) }

    // Layer 3 (Classification Head): [numClasses][h2Dim]
    var w3: Array<FloatArray> = Array(numClasses) { FloatArray(h2Dim) }
    var b3: FloatArray = FloatArray(numClasses)

    // Legacy / Linear classifier weights: [numClasses][featureDim]
    var weights: Array<FloatArray> = Array(numClasses) { FloatArray(featureDim) }
    var biases: FloatArray = FloatArray(numClasses)

    // Feature Scaler
    var scaler: FeatureScaler? = null
    var isTrained: Boolean = false

    private val random = Random(42)

    init {
        initializeWeights()
    }

    private fun initializeWeights() {
        // He (Kaiming) normal initialization for Layer 1
        val std1 = sqrt(2.0 / featureDim).toFloat()
        for (i in 0 until h1Dim) {
            for (j in 0 until featureDim) {
                w1[i][j] = (random.nextGaussian() * std1).toFloat()
            }
            b1[i] = 0f
            gamma1[i] = 1f
            beta1[i] = 0f
        }

        // He initialization for Layer 2
        val std2 = sqrt(2.0 / h1Dim).toFloat()
        for (i in 0 until h2Dim) {
            for (j in 0 until h1Dim) {
                w2[i][j] = (random.nextGaussian() * std2).toFloat()
                // Skip projection initialized with mild variance (0.2x)
                wSkip[i][j] = (random.nextGaussian() * (std2 * 0.2f)).toFloat()
            }
            b2[i] = 0f
            gamma2[i] = 1f
            beta2[i] = 0f
        }

        // Xavier initialization for Layer 3 (Output Head)
        val std3 = sqrt(2.0 / (h2Dim + numClasses)).toFloat()
        for (k in 0 until numClasses) {
            for (j in 0 until h2Dim) {
                w3[k][j] = (random.nextGaussian() * std3).toFloat()
            }
            b3[k] = 0f
        }

        // Linear fallback initialization
        val stdLinear = sqrt(2.0 / (featureDim + numClasses)).toFloat()
        for (k in 0 until numClasses) {
            for (j in 0 until featureDim) {
                weights[k][j] = (random.nextGaussian() * stdLinear).toFloat()
            }
            biases[k] = 0f
        }
    }

    // GELU activation function with fast bounds checking and NaN protection
    private fun gelu(x: Float): Float {
        if (x.isNaN()) return 0f
        if (x < -3.0f) return 0f
        if (x > 3.0f) return x
        val s = 0.79788456f // sqrt(2/pi)
        val inner = (s * (x + 0.044715f * x * x * x)).coerceIn(-10f, 10f)
        val tanhVal = tanh(inner.toDouble()).toFloat()
        return 0.5f * x * (1.0f + tanhVal)
    }

    private fun geluDerivative(x: Float): Float {
        if (x.isNaN()) return 0f
        if (x < -3.0f) return 0f
        if (x > 3.0f) return 1f
        val s = 0.79788456f
        val inner = (s * (x + 0.044715f * x * x * x)).coerceIn(-10f, 10f)
        val t = tanh(inner.toDouble()).toFloat()
        val dt = (1.0f - t * t).coerceIn(0f, 1f)
        val dInner = s * (1.0f + 3.0f * 0.044715f * x * x)
        return (0.5f * (1.0f + t) + 0.5f * x * dt * dInner).coerceIn(0f, 1.5f)
    }

    // Zero-allocation in-place softmax with NaN safety
    private fun softmaxInPlace(logits: FloatArray, outProbs: FloatArray, count: Int) {
        var maxLogit = logits[0]
        for (i in 1 until count) {
            val v = logits[i]
            if (!v.isNaN() && v > maxLogit) maxLogit = v
        }
        if (maxLogit.isNaN()) maxLogit = 0f

        var sumExp = 0f
        for (i in 0 until count) {
            val v = if (logits[i].isNaN()) 0f else logits[i]
            val expVal = exp((v - maxLogit).toDouble().coerceIn(-40.0, 40.0)).toFloat()
            outProbs[i] = expVal
            sumExp += expVal
        }
        val invSum = 1.0f / sumExp.coerceAtLeast(1e-7f)
        for (i in 0 until count) {
            outProbs[i] *= invSum
        }
    }

    suspend fun train(
        samples: List<TrainingSample>,
        epochs: Int = 30,
        learningRate: Float = 0.003f,
        batchSize: Int = 16,
        architecture: ModelArchitecture = this.architecture,
        optimizerType: OptimizerType = OptimizerType.ADAM_W,
        lrSchedule: LearningRateSchedule = LearningRateSchedule.COSINE_ANNEALING,
        deviceProtectionEnabled: Boolean = true,
        overallStartMs: Long = System.currentTimeMillis(),
        onProgress: suspend (TrainingProgress) -> Unit
    ) = withContext(Dispatchers.Default) {
        if (samples.isEmpty() || numClasses < 2) return@withContext
        this@OnDeviceTrainer.architecture = architecture

        val elapsedMsStart = System.currentTimeMillis() - overallStartMs
        val elapsedSecStart = elapsedMsStart / 1000L

        // Initial realistic ETA estimate based on total operations
        val estTrainingTimeRemainingSec = ((epochs.toLong() * samples.size) / 2500L).coerceAtLeast(20L)

        // Step 1: Feature Standardization
        onProgress(
            TrainingProgress(
                currentEpoch = 0,
                totalEpochs = epochs,
                loss = 0f,
                accuracy = 0f,
                statusMessage = "Standardizing extracted neural features with Z-score & L2 norm...",
                overallPercentage = 65.0f,
                phase = TrainingPhase.STANDARDIZING_FEATURES,
                currentStep = 0,
                totalSteps = epochs * samples.size,
                elapsedSeconds = elapsedSecStart,
                estimatedRemainingSeconds = estTrainingTimeRemainingSec,
                speedText = "Standardizing"
            )
        )
        yield()

        scaler = FeatureScaler.fitFromTrainingSamples(samples, featureDim)

        for (sample in samples) {
            scaler!!.transformInPlace(sample.features)
        }

        val numSamples = samples.size

        // Optimizer state for AdamW
        var stepCount = 0
        val beta1Adam = 0.9f
        val beta2Adam = 0.999f
        val epsAdam = 1e-7f
        val weightDecay = 0.001f

        // Moments for W1, b1
        val mW1 = Array(h1Dim) { FloatArray(featureDim) }
        val vW1 = Array(h1Dim) { FloatArray(featureDim) }
        val mb1 = FloatArray(h1Dim)
        val vb1 = FloatArray(h1Dim)

        // Moments for W2, b2, W_skip
        val mW2 = Array(h2Dim) { FloatArray(h1Dim) }
        val vW2 = Array(h2Dim) { FloatArray(h1Dim) }
        val mb2 = FloatArray(h2Dim)
        val vb2 = FloatArray(h2Dim)
        val mWSkip = Array(h2Dim) { FloatArray(h1Dim) }
        val vWSkip = Array(h2Dim) { FloatArray(h1Dim) }

        // Moments for W3, b3 (or Standard MLP Layer 2)
        val mW3 = Array(numClasses) { FloatArray(if (architecture == ModelArchitecture.STANDARD_MLP) h1Dim else h2Dim) }
        val vW3 = Array(numClasses) { FloatArray(if (architecture == ModelArchitecture.STANDARD_MLP) h1Dim else h2Dim) }
        val mb3 = FloatArray(numClasses)
        val vb3 = FloatArray(numClasses)

        // Moments for Linear
        val mLinearW = Array(numClasses) { FloatArray(featureDim) }
        val vLinearW = Array(numClasses) { FloatArray(featureDim) }
        val mLinearB = FloatArray(numClasses)
        val vLinearB = FloatArray(numClasses)

        // Reusable gradient accumulators (allocated ONCE outside loop)
        val gradW1 = Array(h1Dim) { FloatArray(featureDim) }
        val gradB1 = FloatArray(h1Dim)
        val gradW2 = Array(h2Dim) { FloatArray(h1Dim) }
        val gradB2 = FloatArray(h2Dim)
        val gradWSkip = Array(h2Dim) { FloatArray(h1Dim) }
        val gradW3 = Array(numClasses) { FloatArray(if (architecture == ModelArchitecture.STANDARD_MLP) h1Dim else h2Dim) }
        val gradB3 = FloatArray(numClasses)
        val gradLinearW = Array(numClasses) { FloatArray(featureDim) }
        val gradLinearB = FloatArray(numClasses)

        // Reusable activation buffers
        val u1 = FloatArray(h1Dim)
        val u1Hat = FloatArray(h1Dim)
        val a1 = FloatArray(h1Dim)
        val dHat1 = FloatArray(h1Dim)

        val u2 = FloatArray(h2Dim)
        val u2Hat = FloatArray(h2Dim)
        val a2 = FloatArray(h2Dim)
        val dHat2 = FloatArray(h2Dim)

        val logits = FloatArray(numClasses)
        val probs = FloatArray(numClasses)
        val dLogits = FloatArray(numClasses)
        val da2 = FloatArray(h2Dim)
        val da1 = FloatArray(h1Dim)

        val sampleIndices = IntArray(numSamples) { it }

        // Label smoothing parameter
        val labelSmoothingEps = 0.05f

        val warmupEpochs = (epochs * 0.1f).coerceAtLeast(1f)
        val trainingStartMs = System.currentTimeMillis()
        var lastProgressEmitMs = 0L

        // Immediately transition UI to Training Neural Network phase
        onProgress(
            TrainingProgress(
                currentEpoch = 1,
                totalEpochs = epochs,
                loss = 0f,
                accuracy = 0f,
                statusMessage = "Starting Epoch 1/$epochs for ${architecture.displayName}...",
                overallPercentage = 66.0f,
                phase = TrainingPhase.TRAINING_NEURAL_NET,
                currentStep = 0,
                totalSteps = epochs * numSamples,
                elapsedSeconds = (trainingStartMs - overallStartMs) / 1000L,
                estimatedRemainingSeconds = estTrainingTimeRemainingSec,
                speedText = "Starting"
            )
        )
        yield()

        for (epoch in 1..epochs) {
            // Compute current epoch learning rate
            val currentLr = when (lrSchedule) {
                LearningRateSchedule.COSINE_ANNEALING -> {
                    if (epoch <= warmupEpochs) {
                        learningRate * (epoch / warmupEpochs)
                    } else {
                        val progress = (epoch - warmupEpochs) / (epochs - warmupEpochs).coerceAtLeast(1f)
                        val minLr = learningRate * 0.05f
                        minLr + 0.5f * (learningRate - minLr) * (1.0f + cos(Math.PI * progress).toFloat())
                    }
                }
                LearningRateSchedule.STEP_DECAY -> {
                    val factor = 0.5f.pow((epoch / 10).toFloat())
                    learningRate * factor
                }
                LearningRateSchedule.CONSTANT -> learningRate
            }

            // In-place Fisher-Yates shuffle
            for (i in numSamples - 1 downTo 1) {
                val j = random.nextInt(i + 1)
                val temp = sampleIndices[i]
                sampleIndices[i] = sampleIndices[j]
                sampleIndices[j] = temp
            }

            var totalLoss = 0f
            var correctPredictions = 0

            val effectiveBatchSize = batchSize.coerceIn(1, numSamples.coerceAtLeast(1))
            var batchIndex = 0

            while (batchIndex < numSamples) {
                val batchEnd = (batchIndex + effectiveBatchSize).coerceAtMost(numSamples)
                val currentBatchSize = batchEnd - batchIndex

                // Fast zeroing of reusable mini-batch gradient accumulators
                when (architecture) {
                    ModelArchitecture.DEEP_RESIDUAL_MLP -> {
                        for (i in 0 until h1Dim) {
                            java.util.Arrays.fill(gradW1[i], 0f)
                            gradB1[i] = 0f
                        }
                        for (i in 0 until h2Dim) {
                            java.util.Arrays.fill(gradW2[i], 0f)
                            gradB2[i] = 0f
                            java.util.Arrays.fill(gradWSkip[i], 0f)
                        }
                        for (k in 0 until numClasses) {
                            java.util.Arrays.fill(gradW3[k], 0f)
                            gradB3[k] = 0f
                        }
                    }
                    ModelArchitecture.STANDARD_MLP -> {
                        for (i in 0 until h1Dim) {
                            java.util.Arrays.fill(gradW1[i], 0f)
                            gradB1[i] = 0f
                        }
                        for (k in 0 until numClasses) {
                            java.util.Arrays.fill(gradW3[k], 0f)
                            gradB3[k] = 0f
                        }
                    }
                    ModelArchitecture.LINEAR -> {
                        for (k in 0 until numClasses) {
                            java.util.Arrays.fill(gradLinearW[k], 0f)
                            gradLinearB[k] = 0f
                        }
                    }
                }

                for (idx in batchIndex until batchEnd) {
                    val sample = samples[sampleIndices[idx]]
                    val x = sample.features
                    val target = sample.classIndex

                    if (architecture == ModelArchitecture.DEEP_RESIDUAL_MLP) {
                        // === FORWARD PASS: Deep Residual MLP ===
                        // Layer 1
                        for (i in 0 until h1Dim) {
                            var sum = b1[i]
                            val wRow = w1[i]
                            for (j in 0 until featureDim) {
                                sum += wRow[j] * x[j]
                            }
                            u1[i] = sum
                        }

                        // LayerNorm 1
                        var mean1 = 0f
                        for (i in 0 until h1Dim) mean1 += u1[i]
                        mean1 /= h1Dim
                        var var1 = 0f
                        for (i in 0 until h1Dim) {
                            val diff = u1[i] - mean1
                            var1 += diff * diff
                        }
                        var1 /= h1Dim
                        val std1 = sqrt(var1 + 1e-5f)
                        val invStd1 = 1.0f / std1
                        for (i in 0 until h1Dim) {
                            val u1H = (u1[i] - mean1) * invStd1
                            u1Hat[i] = u1H
                            val z1 = u1H * gamma1[i] + beta1[i]
                            a1[i] = gelu(z1)
                        }

                        // Layer 2 with Residual Skip
                        for (i in 0 until h2Dim) {
                            var sum = b2[i]
                            val wRow = w2[i]
                            for (j in 0 until h1Dim) {
                                sum += wRow[j] * a1[j]
                            }
                            u2[i] = sum
                        }

                        // LayerNorm 2
                        var mean2 = 0f
                        for (i in 0 until h2Dim) mean2 += u2[i]
                        mean2 /= h2Dim
                        var var2 = 0f
                        for (i in 0 until h2Dim) {
                            val diff = u2[i] - mean2
                            var2 += diff * diff
                        }
                        var2 /= h2Dim
                        val std2 = sqrt(var2 + 1e-5f)
                        val invStd2 = 1.0f / std2
                        for (i in 0 until h2Dim) {
                            val u2H = (u2[i] - mean2) * invStd2
                            u2Hat[i] = u2H
                            val z2 = u2H * gamma2[i] + beta2[i]
                            val gelu2 = gelu(z2)

                            var skipVal = 0f
                            val sRow = wSkip[i]
                            for (j in 0 until h1Dim) {
                                skipVal += sRow[j] * a1[j]
                            }
                            a2[i] = gelu2 + 0.2f * skipVal
                        }

                        // Layer 3 (Classification Head)
                        for (k in 0 until numClasses) {
                            var sum = b3[k]
                            val wRow = w3[k]
                            for (j in 0 until h2Dim) {
                                sum += wRow[j] * a2[j]
                            }
                            logits[k] = sum
                        }

                        softmaxInPlace(logits, probs, numClasses)

                        // Smoothed Target Cross-Entropy Loss
                        val smoothTargetProb = 1.0f - labelSmoothingEps
                        val uniformProb = labelSmoothingEps / numClasses
                        var sampleLoss = 0f
                        for (k in 0 until numClasses) {
                            val yK = if (k == target) (smoothTargetProb + uniformProb) else uniformProb
                            sampleLoss -= yK * ln(probs[k].coerceAtLeast(1e-7f))
                        }
                        totalLoss += sampleLoss

                        // Check Accuracy
                        var bestK = 0
                        var bestP = probs[0]
                        for (k in 1 until numClasses) {
                            if (probs[k] > bestP) {
                                bestP = probs[k]
                                bestK = k
                            }
                        }
                        if (bestK == target) correctPredictions++

                        // === BACKWARD PASS: Deep Residual MLP ===
                        // Output Layer
                        for (k in 0 until numClasses) {
                            val yK = if (k == target) (smoothTargetProb + uniformProb) else uniformProb
                            val dL = probs[k] - yK
                            dLogits[k] = dL
                            gradB3[k] += dL
                            val gradWRow = gradW3[k]
                            for (j in 0 until h2Dim) {
                                gradWRow[j] += dL * a2[j]
                            }
                        }

                        // Backprop into a2
                        for (j in 0 until h2Dim) {
                            var sum = 0f
                            for (k in 0 until numClasses) {
                                sum += dLogits[k] * w3[k][j]
                            }
                            da2[j] = sum
                        }

                        // Backprop Layer 2 & Skip
                        java.util.Arrays.fill(da1, 0f)
                        var sumDHat2 = 0f
                        var sumDHatU2 = 0f

                        for (i in 0 until h2Dim) {
                            val da2_i = da2[i]
                            // Skip gradient (0.2x)
                            val dSkip = 0.2f * da2_i
                            val sRow = gradWSkip[i]
                            val wSRow = wSkip[i]
                            for (j in 0 until h1Dim) {
                                sRow[j] += dSkip * a1[j]
                                da1[j] += dSkip * wSRow[j]
                            }

                            // GELU 2 gradient
                            val z2 = u2Hat[i] * gamma2[i] + beta2[i]
                            val dGelu = da2_i * geluDerivative(z2)
                            val dHat = dGelu * gamma2[i]
                            dHat2[i] = dHat
                            sumDHat2 += dHat
                            sumDHatU2 += dHat * u2Hat[i]
                        }

                        // Exact LayerNorm 2 Backward Pass
                        val invH2 = 1.0f / h2Dim
                        for (i in 0 until h2Dim) {
                            val du2_i = invStd2 * (dHat2[i] - sumDHat2 * invH2 - u2Hat[i] * sumDHatU2 * invH2)
                            gradB2[i] += du2_i

                            val gradW2Row = gradW2[i]
                            val w2Row = w2[i]
                            for (j in 0 until h1Dim) {
                                gradW2Row[j] += du2_i * a1[j]
                                da1[j] += du2_i * w2Row[j]
                            }
                        }

                        // Backprop Layer 1
                        var sumDHat1 = 0f
                        var sumDHatU1 = 0f
                        for (i in 0 until h1Dim) {
                            val z1 = u1Hat[i] * gamma1[i] + beta1[i]
                            val dGelu1 = da1[i] * geluDerivative(z1)
                            val dHat = dGelu1 * gamma1[i]
                            dHat1[i] = dHat
                            sumDHat1 += dHat
                            sumDHatU1 += dHat * u1Hat[i]
                        }

                        // Exact LayerNorm 1 Backward Pass
                        val invH1 = 1.0f / h1Dim
                        for (i in 0 until h1Dim) {
                            val du1_i = invStd1 * (dHat1[i] - sumDHat1 * invH1 - u1Hat[i] * sumDHatU1 * invH1)
                            gradB1[i] += du1_i

                            val gradW1Row = gradW1[i]
                            for (j in 0 until featureDim) {
                                gradW1Row[j] += du1_i * x[j]
                            }
                        }
                    } else if (architecture == ModelArchitecture.STANDARD_MLP) {
                        // === FORWARD PASS: 2-Layer MLP ===
                        // Layer 1
                        for (i in 0 until h1Dim) {
                            var sum = b1[i]
                            val wRow = w1[i]
                            for (j in 0 until featureDim) {
                                sum += wRow[j] * x[j]
                            }
                            u1[i] = sum
                            a1[i] = gelu(sum)
                        }

                        // Layer 2 Head
                        for (k in 0 until numClasses) {
                            var sum = b3[k]
                            val wRow = w3[k]
                            for (j in 0 until h1Dim) {
                                sum += wRow[j] * a1[j]
                            }
                            logits[k] = sum
                        }

                        softmaxInPlace(logits, probs, numClasses)

                        val smoothTargetProb = 1.0f - labelSmoothingEps
                        val uniformProb = labelSmoothingEps / numClasses
                        var sampleLoss = 0f
                        for (k in 0 until numClasses) {
                            val yK = if (k == target) (smoothTargetProb + uniformProb) else uniformProb
                            sampleLoss -= yK * ln(probs[k].coerceAtLeast(1e-7f))
                        }
                        totalLoss += sampleLoss

                        var bestK = 0
                        var bestP = probs[0]
                        for (k in 1 until numClasses) {
                            if (probs[k] > bestP) {
                                bestP = probs[k]
                                bestK = k
                            }
                        }
                        if (bestK == target) correctPredictions++

                        // === BACKWARD PASS: 2-Layer MLP ===
                        for (k in 0 until numClasses) {
                            val yK = if (k == target) (smoothTargetProb + uniformProb) else uniformProb
                            val dL = probs[k] - yK
                            dLogits[k] = dL
                            gradB3[k] += dL
                            val gradWRow = gradW3[k]
                            for (j in 0 until h1Dim) {
                                gradWRow[j] += dL * a1[j]
                            }
                        }

                        java.util.Arrays.fill(da1, 0f)
                        for (j in 0 until h1Dim) {
                            var sum = 0f
                            for (k in 0 until numClasses) {
                                sum += dLogits[k] * w3[k][j]
                            }
                            da1[j] = sum
                        }

                        for (i in 0 until h1Dim) {
                            val du1_i = da1[i] * geluDerivative(u1[i])
                            gradB1[i] += du1_i
                            val gradW1Row = gradW1[i]
                            for (j in 0 until featureDim) {
                                gradW1Row[j] += du1_i * x[j]
                            }
                        }
                    } else {
                        // === FORWARD & BACKWARD PASS: Linear Softmax Classifier ===
                        for (k in 0 until numClasses) {
                            var sum = biases[k]
                            val wRow = weights[k]
                            for (j in 0 until featureDim) {
                                sum += wRow[j] * x[j]
                            }
                            logits[k] = sum
                        }

                        softmaxInPlace(logits, probs, numClasses)
                        val pTarget = probs[target].coerceAtLeast(1e-7f)
                        totalLoss += -ln(pTarget)

                        var maxIdx = 0
                        var maxProb = probs[0]
                        for (k in 1 until numClasses) {
                            if (probs[k] > maxProb) {
                                maxProb = probs[k]
                                maxIdx = k
                            }
                        }
                        if (maxIdx == target) correctPredictions++

                        for (k in 0 until numClasses) {
                            val yK = if (k == target) 1.0f else 0.0f
                            val dK = probs[k] - yK
                            gradLinearB[k] += dK
                            val wRow = gradLinearW[k]
                            for (j in 0 until featureDim) {
                                wRow[j] += dK * x[j]
                            }
                        }
                    }
                }

                // Apply Mini-Batch Optimizer Updates
                stepCount++
                val invBatch = 1.0f / currentBatchSize

                if (architecture == ModelArchitecture.DEEP_RESIDUAL_MLP) {
                    val biasCorrection1 = (1.0 - beta1Adam.toDouble().pow(stepCount.toDouble())).toFloat().coerceAtLeast(1e-4f)
                    val biasCorrection2 = (1.0 - beta2Adam.toDouble().pow(stepCount.toDouble())).toFloat().coerceAtLeast(1e-4f)

                    // Layer 3 (Head) AdamW
                    for (k in 0 until numClasses) {
                        val gB = (gradB3[k] * invBatch).coerceIn(-3f, 3f)
                        mb3[k] = beta1Adam * mb3[k] + (1 - beta1Adam) * gB
                        vb3[k] = beta2Adam * vb3[k] + (1 - beta2Adam) * gB * gB
                        val mHatB = mb3[k] / biasCorrection1
                        val vHatB = vb3[k] / biasCorrection2
                        b3[k] -= currentLr * (mHatB / (sqrt(vHatB) + epsAdam))

                        val wRow = w3[k]
                        val gRow = gradW3[k]
                        val mRow = mW3[k]
                        val vRow = vW3[k]
                        for (j in 0 until h2Dim) {
                            val gW = (gRow[j] * invBatch).coerceIn(-3f, 3f)
                            mRow[j] = beta1Adam * mRow[j] + (1 - beta1Adam) * gW
                            vRow[j] = beta2Adam * vRow[j] + (1 - beta2Adam) * gW * gW
                            val mHat = mRow[j] / biasCorrection1
                            val vHat = vRow[j] / biasCorrection2
                            val update = mHat / (sqrt(vHat) + epsAdam) + weightDecay * wRow[j]
                            wRow[j] -= currentLr * update
                        }
                    }

                    // Layer 2 AdamW
                    for (i in 0 until h2Dim) {
                        val gB = (gradB2[i] * invBatch).coerceIn(-3f, 3f)
                        mb2[i] = beta1Adam * mb2[i] + (1 - beta1Adam) * gB
                        vb2[i] = beta2Adam * vb2[i] + (1 - beta2Adam) * gB * gB
                        b2[i] -= currentLr * ((mb2[i] / biasCorrection1) / (sqrt(vb2[i] / biasCorrection2) + epsAdam))

                        val wRow = w2[i]
                        val gRow = gradW2[i]
                        val mRow = mW2[i]
                        val vRow = vW2[i]
                        for (j in 0 until h1Dim) {
                            val gW = (gRow[j] * invBatch).coerceIn(-3f, 3f)
                            mRow[j] = beta1Adam * mRow[j] + (1 - beta1Adam) * gW
                            vRow[j] = beta2Adam * vRow[j] + (1 - beta2Adam) * gW * gW
                            val update = (mRow[j] / biasCorrection1) / (sqrt(vRow[j] / biasCorrection2) + epsAdam) + weightDecay * wRow[j]
                            wRow[j] -= currentLr * update
                        }

                        // Skip weights
                        val sRow = wSkip[i]
                        val gSRow = gradWSkip[i]
                        val mSRow = mWSkip[i]
                        val vSRow = vWSkip[i]
                        for (j in 0 until h1Dim) {
                            val gS = (gSRow[j] * invBatch).coerceIn(-3f, 3f)
                            mSRow[j] = beta1Adam * mSRow[j] + (1 - beta1Adam) * gS
                            vSRow[j] = beta2Adam * vSRow[j] + (1 - beta2Adam) * gS * gS
                            val update = (mSRow[j] / biasCorrection1) / (sqrt(vSRow[j] / biasCorrection2) + epsAdam) + weightDecay * sRow[j]
                            sRow[j] -= currentLr * update
                        }
                    }

                    // Layer 1 AdamW
                    for (i in 0 until h1Dim) {
                        val gB = (gradB1[i] * invBatch).coerceIn(-3f, 3f)
                        mb1[i] = beta1Adam * mb1[i] + (1 - beta1Adam) * gB
                        vb1[i] = beta2Adam * vb1[i] + (1 - beta2Adam) * gB * gB
                        b1[i] -= currentLr * ((mb1[i] / biasCorrection1) / (sqrt(vb1[i] / biasCorrection2) + epsAdam))

                        val wRow = w1[i]
                        val gRow = gradW1[i]
                        val mRow = mW1[i]
                        val vRow = vW1[i]
                        for (j in 0 until featureDim) {
                            val gW = (gRow[j] * invBatch).coerceIn(-3f, 3f)
                            mRow[j] = beta1Adam * mRow[j] + (1 - beta1Adam) * gW
                            vRow[j] = beta2Adam * vRow[j] + (1 - beta2Adam) * gW * gW
                            val update = (mRow[j] / biasCorrection1) / (sqrt(vRow[j] / biasCorrection2) + epsAdam) + weightDecay * wRow[j]
                            wRow[j] -= currentLr * update
                        }
                    }
                } else if (architecture == ModelArchitecture.STANDARD_MLP) {
                    val biasCorrection1 = (1.0 - beta1Adam.toDouble().pow(stepCount.toDouble())).toFloat().coerceAtLeast(1e-4f)
                    val biasCorrection2 = (1.0 - beta2Adam.toDouble().pow(stepCount.toDouble())).toFloat().coerceAtLeast(1e-4f)

                    // Head AdamW
                    for (k in 0 until numClasses) {
                        val gB = (gradB3[k] * invBatch).coerceIn(-3f, 3f)
                        mb3[k] = beta1Adam * mb3[k] + (1 - beta1Adam) * gB
                        vb3[k] = beta2Adam * vb3[k] + (1 - beta2Adam) * gB * gB
                        b3[k] -= currentLr * ((mb3[k] / biasCorrection1) / (sqrt(vb3[k] / biasCorrection2) + epsAdam))

                        val wRow = w3[k]
                        val gRow = gradW3[k]
                        val mRow = mW3[k]
                        val vRow = vW3[k]
                        for (j in 0 until h1Dim) {
                            val gW = (gRow[j] * invBatch).coerceIn(-3f, 3f)
                            mRow[j] = beta1Adam * mRow[j] + (1 - beta1Adam) * gW
                            vRow[j] = beta2Adam * vRow[j] + (1 - beta2Adam) * gW * gW
                            val update = (mRow[j] / biasCorrection1) / (sqrt(vRow[j] / biasCorrection2) + epsAdam) + weightDecay * wRow[j]
                            wRow[j] -= currentLr * update
                        }
                    }

                    // Layer 1 AdamW
                    for (i in 0 until h1Dim) {
                        val gB = (gradB1[i] * invBatch).coerceIn(-3f, 3f)
                        mb1[i] = beta1Adam * mb1[i] + (1 - beta1Adam) * gB
                        vb1[i] = beta2Adam * vb1[i] + (1 - beta2Adam) * gB * gB
                        b1[i] -= currentLr * ((mb1[i] / biasCorrection1) / (sqrt(vb1[i] / biasCorrection2) + epsAdam))

                        val wRow = w1[i]
                        val gRow = gradW1[i]
                        val mRow = mW1[i]
                        val vRow = vW1[i]
                        for (j in 0 until featureDim) {
                            val gW = (gRow[j] * invBatch).coerceIn(-3f, 3f)
                            mRow[j] = beta1Adam * mRow[j] + (1 - beta1Adam) * gW
                            vRow[j] = beta2Adam * vRow[j] + (1 - beta2Adam) * gW * gW
                            val update = (mRow[j] / biasCorrection1) / (sqrt(vRow[j] / biasCorrection2) + epsAdam) + weightDecay * wRow[j]
                            wRow[j] -= currentLr * update
                        }
                    }
                } else {
                    // Linear AdamW
                    val biasCorrection1 = (1.0 - beta1Adam.toDouble().pow(stepCount.toDouble())).toFloat().coerceAtLeast(1e-4f)
                    val biasCorrection2 = (1.0 - beta2Adam.toDouble().pow(stepCount.toDouble())).toFloat().coerceAtLeast(1e-4f)

                    for (k in 0 until numClasses) {
                        val gB = (gradLinearB[k] * invBatch).coerceIn(-3f, 3f)
                        mLinearB[k] = beta1Adam * mLinearB[k] + (1 - beta1Adam) * gB
                        vLinearB[k] = beta2Adam * vLinearB[k] + (1 - beta2Adam) * gB * gB
                        biases[k] -= currentLr * ((mLinearB[k] / biasCorrection1) / (sqrt(vLinearB[k] / biasCorrection2) + epsAdam))

                        val wRow = weights[k]
                        val gRow = gradLinearW[k]
                        val mRow = mLinearW[k]
                        val vRow = vLinearW[k]
                        for (j in 0 until featureDim) {
                            val gW = (gRow[j] * invBatch).coerceIn(-3f, 3f)
                            mRow[j] = beta1Adam * mRow[j] + (1 - beta1Adam) * gW
                            vRow[j] = beta2Adam * vRow[j] + (1 - beta2Adam) * gW * gW
                            val update = (mRow[j] / biasCorrection1) / (sqrt(vRow[j] / biasCorrection2) + epsAdam) + 0.001f * wRow[j]
                            wRow[j] -= currentLr * update
                        }
                    }
                }

                batchIndex += effectiveBatchSize

                // Live continuous progress emission every ~350ms during the epoch
                val now = System.currentTimeMillis()
                val isEpochEnd = batchIndex >= numSamples
                if (now - lastProgressEmitMs >= 350L || isEpochEnd) {
                    lastProgressEmitMs = now
                    val processedInEpoch = batchIndex.coerceAtMost(numSamples)
                    val totalTrainedSamplesSoFar = (epoch - 1L) * numSamples + processedInEpoch
                    val totalTrainingSamplesAllEpochs = epochs.toLong() * numSamples

                    val elapsedMs = now - overallStartMs
                    val elapsedSec = elapsedMs / 1000L
                    val trainingElapsedSec = (now - trainingStartMs) / 1000L

                    val epochFraction = (epoch - 1).toFloat() / epochs + (processedInEpoch.toFloat() / numSamples) / epochs
                    val currentOverallPct = (66.0f + epochFraction * 33.5f).coerceIn(66.0f, 99.5f)

                    val samplesPerSec = if (trainingElapsedSec > 0) {
                        totalTrainedSamplesSoFar.toFloat() / trainingElapsedSec.coerceAtLeast(1L)
                    } else 0f

                    val remainingSamples = (totalTrainingSamplesAllEpochs - totalTrainedSamplesSoFar).coerceAtLeast(0L)
                    val estRemainingSec = if (samplesPerSec > 1f) (remainingSamples / samplesPerSec).toLong() else 0L

                    val runningLoss = if (processedInEpoch > 0) totalLoss / processedInEpoch else 0f
                    val runningAcc = if (processedInEpoch > 0) correctPredictions.toFloat() / processedInEpoch else 0f

                    val speedText = if (samplesPerSec > 0) {
                        String.format(Locale.US, "%.0f smp/s", samplesPerSec)
                    } else "Training"

                    val message = "Epoch $epoch/$epochs [${(processedInEpoch * 100) / numSamples}%] • ${processedInEpoch}/$numSamples smp • Loss: ${String.format(Locale.US, "%.4f", runningLoss)} • Acc: ${String.format(Locale.US, "%.1f%%", runningAcc * 100)}"

                    onProgress(
                        TrainingProgress(
                            currentEpoch = epoch,
                            totalEpochs = epochs,
                            loss = runningLoss,
                            accuracy = runningAcc,
                            statusMessage = message,
                            overallPercentage = currentOverallPct,
                            phase = TrainingPhase.TRAINING_NEURAL_NET,
                            currentStep = totalTrainedSamplesSoFar.toInt(),
                            totalSteps = (epochs * numSamples),
                            elapsedSeconds = elapsedSec,
                            estimatedRemainingSeconds = estRemainingSec,
                            speedText = speedText
                        )
                    )
                }

                yield()
            }

            if (deviceProtectionEnabled) {
                delay(6)
            } else {
                yield()
            }
        }
        isTrained = true
    }

    fun predict(features: FloatArray): PredictionResult {
        val startTime = System.currentTimeMillis()
        val scaledFeatures = scaler?.transform(features) ?: features

        val logits = FloatArray(numClasses)

        when (architecture) {
            ModelArchitecture.DEEP_RESIDUAL_MLP -> {
                // Forward Layer 1
                val u1 = FloatArray(h1Dim)
                for (i in 0 until h1Dim) {
                    var sum = b1[i]
                    val wRow = w1[i]
                    for (j in 0 until featureDim) {
                        sum += wRow[j] * scaledFeatures[j]
                    }
                    u1[i] = sum
                }

                // LayerNorm 1
                var mean1 = 0f
                for (v in u1) mean1 += v
                mean1 /= h1Dim
                var var1 = 0f
                for (v in u1) {
                    val diff = v - mean1
                    var1 += diff * diff
                }
                var1 /= h1Dim
                val std1 = sqrt(var1 + 1e-5f)
                val a1 = FloatArray(h1Dim)
                for (i in 0 until h1Dim) {
                    val uHat = (u1[i] - mean1) / std1
                    a1[i] = gelu(uHat * gamma1[i] + beta1[i])
                }

                // Forward Layer 2 + Skip
                val u2 = FloatArray(h2Dim)
                for (i in 0 until h2Dim) {
                    var sum = b2[i]
                    val wRow = w2[i]
                    for (j in 0 until h1Dim) {
                        sum += wRow[j] * a1[j]
                    }
                    u2[i] = sum
                }

                // LayerNorm 2
                var mean2 = 0f
                for (v in u2) mean2 += v
                mean2 /= h2Dim
                var var2 = 0f
                for (v in u2) {
                    val diff = v - mean2
                    var2 += diff * diff
                }
                var2 /= h2Dim
                val std2 = sqrt(var2 + 1e-5f)
                val a2 = FloatArray(h2Dim)
                for (i in 0 until h2Dim) {
                    val uHat = (u2[i] - mean2) / std2
                    val gelu2 = gelu(uHat * gamma2[i] + beta2[i])

                    var skipVal = 0f
                    val sRow = wSkip[i]
                    for (j in 0 until h1Dim) {
                        skipVal += sRow[j] * a1[j]
                    }
                    a2[i] = gelu2 + 0.2f * skipVal
                }

                // Forward Layer 3 (Head)
                for (k in 0 until numClasses) {
                    var sum = b3[k]
                    val wRow = w3[k]
                    for (j in 0 until h2Dim) {
                        sum += wRow[j] * a2[j]
                    }
                    logits[k] = sum
                }
            }
            ModelArchitecture.STANDARD_MLP -> {
                // Forward Layer 1
                val a1 = FloatArray(h1Dim)
                for (i in 0 until h1Dim) {
                    var sum = b1[i]
                    val wRow = w1[i]
                    for (j in 0 until featureDim) {
                        sum += wRow[j] * scaledFeatures[j]
                    }
                    a1[i] = gelu(sum)
                }

                // Forward Layer 2 (Head)
                for (k in 0 until numClasses) {
                    var sum = b3[k]
                    val wRow = w3[k]
                    for (j in 0 until h1Dim) {
                        sum += wRow[j] * a1[j]
                    }
                    logits[k] = sum
                }
            }
            ModelArchitecture.LINEAR -> {
                // Linear forward pass
                for (k in 0 until numClasses) {
                    var sum = biases[k]
                    val wRow = weights[k]
                    for (j in 0 until featureDim) {
                        sum += wRow[j] * scaledFeatures[j]
                    }
                    logits[k] = sum
                }
            }
        }

        val probs = softmax(logits)

        var bestClass = 0
        var highestProb = probs[0]
        val classConfidences = mutableListOf<ClassConfidence>()

        for (k in 0 until numClasses) {
            val label = classLabels.getOrElse(k) { "Class $k" }
            classConfidences.add(ClassConfidence(k, label, probs[k]))

            if (probs[k] > highestProb) {
                highestProb = probs[k]
                bestClass = k
            }
        }

        val endTime = System.currentTimeMillis()
        val bestLabel = classLabels.getOrElse(bestClass) { "Class $bestClass" }

        return PredictionResult(
            classIndex = bestClass,
            classLabel = bestLabel,
            confidence = highestProb,
            allProbabilities = classConfidences.sortedByDescending { it.probability },
            inferenceTimeMs = (endTime - startTime).coerceAtLeast(1)
        )
    }

    private fun softmax(logits: FloatArray): FloatArray {
        var maxLogit = logits[0]
        for (v in logits) {
            if (!v.isNaN() && v > maxLogit) maxLogit = v
        }
        if (maxLogit.isNaN()) maxLogit = 0f

        val expVals = FloatArray(logits.size)
        var sumExp = 0f
        for (i in logits.indices) {
            val v = if (logits[i].isNaN()) 0f else logits[i]
            val expVal = exp((v - maxLogit).toDouble().coerceIn(-40.0, 40.0)).toFloat()
            expVals[i] = expVal
            sumExp += expVal
        }

        val probs = FloatArray(logits.size)
        val denom = sumExp.coerceAtLeast(1e-6f)
        for (i in logits.indices) {
            probs[i] = expVals[i] / denom
        }
        return probs
    }

    fun exportWeightsJson(): String {
        val sb = StringBuilder(1024 * 512)
        sb.append("{\"architecture\":\"").append(architecture.name).append("\",")
        sb.append("\"numClasses\":").append(numClasses).append(",")
        sb.append("\"featureDim\":").append(featureDim).append(",")
        sb.append("\"h1Dim\":").append(h1Dim).append(",")
        sb.append("\"h2Dim\":").append(h2Dim).append(",")

        when (architecture) {
            ModelArchitecture.DEEP_RESIDUAL_MLP -> {
                // w1
                sb.append("\"w1\":[")
                for (i in 0 until h1Dim) {
                    if (i > 0) sb.append(",")
                    sb.append("[")
                    val row = w1[i]
                    for (j in 0 until featureDim) {
                        if (j > 0) sb.append(",")
                        sb.append(row[j])
                    }
                    sb.append("]")
                }
                sb.append("],")

                // b1
                sb.append("\"b1\":[")
                for (i in 0 until h1Dim) {
                    if (i > 0) sb.append(",")
                    sb.append(b1[i])
                }
                sb.append("],")

                // gamma1
                sb.append("\"gamma1\":[")
                for (i in 0 until h1Dim) {
                    if (i > 0) sb.append(",")
                    sb.append(gamma1[i])
                }
                sb.append("],")

                // beta1
                sb.append("\"beta1\":[")
                for (i in 0 until h1Dim) {
                    if (i > 0) sb.append(",")
                    sb.append(beta1[i])
                }
                sb.append("],")

                // w2
                sb.append("\"w2\":[")
                for (i in 0 until h2Dim) {
                    if (i > 0) sb.append(",")
                    sb.append("[")
                    val row = w2[i]
                    for (j in 0 until h1Dim) {
                        if (j > 0) sb.append(",")
                        sb.append(row[j])
                    }
                    sb.append("]")
                }
                sb.append("],")

                // b2
                sb.append("\"b2\":[")
                for (i in 0 until h2Dim) {
                    if (i > 0) sb.append(",")
                    sb.append(b2[i])
                }
                sb.append("],")

                // gamma2
                sb.append("\"gamma2\":[")
                for (i in 0 until h2Dim) {
                    if (i > 0) sb.append(",")
                    sb.append(gamma2[i])
                }
                sb.append("],")

                // beta2
                sb.append("\"beta2\":[")
                for (i in 0 until h2Dim) {
                    if (i > 0) sb.append(",")
                    sb.append(beta2[i])
                }
                sb.append("],")

                // wSkip
                sb.append("\"wSkip\":[")
                for (i in 0 until h2Dim) {
                    if (i > 0) sb.append(",")
                    sb.append("[")
                    val row = wSkip[i]
                    for (j in 0 until h1Dim) {
                        if (j > 0) sb.append(",")
                        sb.append(row[j])
                    }
                    sb.append("]")
                }
                sb.append("],")

                // w3
                sb.append("\"w3\":[")
                for (k in 0 until numClasses) {
                    if (k > 0) sb.append(",")
                    sb.append("[")
                    val row = w3[k]
                    for (j in 0 until h2Dim) {
                        if (j > 0) sb.append(",")
                        sb.append(row[j])
                    }
                    sb.append("]")
                }
                sb.append("],")

                // b3
                sb.append("\"b3\":[")
                for (k in 0 until numClasses) {
                    if (k > 0) sb.append(",")
                    sb.append(b3[k])
                }
                sb.append("]")
            }
            ModelArchitecture.STANDARD_MLP -> {
                // w1
                sb.append("\"w1\":[")
                for (i in 0 until h1Dim) {
                    if (i > 0) sb.append(",")
                    sb.append("[")
                    val row = w1[i]
                    for (j in 0 until featureDim) {
                        if (j > 0) sb.append(",")
                        sb.append(row[j])
                    }
                    sb.append("]")
                }
                sb.append("],")

                // b1
                sb.append("\"b1\":[")
                for (i in 0 until h1Dim) {
                    if (i > 0) sb.append(",")
                    sb.append(b1[i])
                }
                sb.append("],")

                // w3
                sb.append("\"w3\":[")
                for (k in 0 until numClasses) {
                    if (k > 0) sb.append(",")
                    sb.append("[")
                    val row = w3[k]
                    for (j in 0 until h1Dim) {
                        if (j > 0) sb.append(",")
                        sb.append(row[j])
                    }
                    sb.append("]")
                }
                sb.append("],")

                // b3
                sb.append("\"b3\":[")
                for (k in 0 until numClasses) {
                    if (k > 0) sb.append(",")
                    sb.append(b3[k])
                }
                sb.append("]")
            }
            ModelArchitecture.LINEAR -> {
                sb.append("\"weights\":[")
                for (k in 0 until numClasses) {
                    if (k > 0) sb.append(",")
                    sb.append("[")
                    val row = weights[k]
                    for (j in 0 until featureDim) {
                        if (j > 0) sb.append(",")
                        sb.append(row[j])
                    }
                    sb.append("]")
                }
                sb.append("],")

                sb.append("\"biases\":[")
                for (k in 0 until numClasses) {
                    if (k > 0) sb.append(",")
                    sb.append(biases[k])
                }
                sb.append("]")
            }
        }
        sb.append("}")
        return sb.toString()
    }

    fun exportBiasesJson(): String {
        val biasArray = JSONArray()
        if (architecture == ModelArchitecture.DEEP_RESIDUAL_MLP || architecture == ModelArchitecture.STANDARD_MLP) {
            for (b in b3) biasArray.put(b.toDouble())
        } else {
            for (b in biases) biasArray.put(b.toDouble())
        }
        return biasArray.toString()
    }

    fun exportLabelsJson(): String {
        val labelsArray = JSONArray()
        for (l in classLabels) {
            labelsArray.put(l)
        }
        return labelsArray.toString()
    }

    fun exportScaleMeansJson(): String {
        val array = JSONArray()
        scaler?.means?.forEach { array.put(it.toDouble()) }
        return array.toString()
    }

    fun exportScaleStdsJson(): String {
        val array = JSONArray()
        scaler?.stds?.forEach { array.put(it.toDouble()) }
        return array.toString()
    }

    companion object {
        fun resolveJsonFromFileIfNeeded(raw: String?): String {
            if (raw.isNullOrBlank()) return "{}"
            val trimmed = raw.trim()
            if (trimmed.startsWith("file:")) {
                return try {
                    val file = File(trimmed.removePrefix("file:"))
                    if (file.exists()) file.readText() else "{}"
                } catch (e: Exception) {
                    "{}"
                }
            } else if (trimmed.startsWith("/") && File(trimmed).exists()) {
                return try {
                    File(trimmed).readText()
                } catch (e: Exception) {
                    "{}"
                }
            }
            return trimmed
        }

        fun loadFromModel(
            weightsJson: String,
            biasesJson: String,
            labelsJson: String,
            numClasses: Int,
            featureDim: Int,
            scaleMeansJson: String? = null,
            scaleStdsJson: String? = null
        ): OnDeviceTrainer {
            val resolvedWeights = resolveJsonFromFileIfNeeded(weightsJson)
            val resolvedBiases = resolveJsonFromFileIfNeeded(biasesJson)
            val resolvedLabels = resolveJsonFromFileIfNeeded(labelsJson)
            val resolvedMeans = if (scaleMeansJson != null) resolveJsonFromFileIfNeeded(scaleMeansJson) else null
            val resolvedStds = if (scaleStdsJson != null) resolveJsonFromFileIfNeeded(scaleStdsJson) else null

            val labelsList = mutableListOf<String>()
            val labelsArray = JSONArray(resolvedLabels)
            for (i in 0 until labelsArray.length()) {
                labelsList.add(labelsArray.getString(i))
            }

            val trainer = OnDeviceTrainer(numClasses, featureDim, labelsList)

            try {
                if (resolvedWeights.trim().startsWith("{")) {
                    val root = JSONObject(resolvedWeights)
                    val archStr = root.optString("architecture", ModelArchitecture.DEEP_RESIDUAL_MLP.name)
                    trainer.architecture = try {
                        ModelArchitecture.valueOf(archStr)
                    } catch (e: Exception) {
                        ModelArchitecture.DEEP_RESIDUAL_MLP
                    }

                    if (trainer.architecture == ModelArchitecture.DEEP_RESIDUAL_MLP) {
                        // Load Layer 1
                        if (root.has("w1")) {
                            val w1Arr = root.getJSONArray("w1")
                            for (i in 0 until minOf(trainer.h1Dim, w1Arr.length())) {
                                val row = w1Arr.getJSONArray(i)
                                for (j in 0 until minOf(trainer.featureDim, row.length())) {
                                    trainer.w1[i][j] = row.getDouble(j).toFloat()
                                }
                            }
                        }
                        if (root.has("b1")) {
                            val b1Arr = root.getJSONArray("b1")
                            for (i in 0 until minOf(trainer.h1Dim, b1Arr.length())) {
                                trainer.b1[i] = b1Arr.getDouble(i).toFloat()
                            }
                        }
                        if (root.has("gamma1")) {
                            val gArr = root.getJSONArray("gamma1")
                            for (i in 0 until minOf(trainer.h1Dim, gArr.length())) {
                                trainer.gamma1[i] = gArr.getDouble(i).toFloat()
                            }
                        }
                        if (root.has("beta1")) {
                            val bArr = root.getJSONArray("beta1")
                            for (i in 0 until minOf(trainer.h1Dim, bArr.length())) {
                                trainer.beta1[i] = bArr.getDouble(i).toFloat()
                            }
                        }

                        // Load Layer 2
                        if (root.has("w2")) {
                            val w2Arr = root.getJSONArray("w2")
                            for (i in 0 until minOf(trainer.h2Dim, w2Arr.length())) {
                                val row = w2Arr.getJSONArray(i)
                                for (j in 0 until minOf(trainer.h1Dim, row.length())) {
                                    trainer.w2[i][j] = row.getDouble(j).toFloat()
                                }
                            }
                        }
                        if (root.has("b2")) {
                            val b2Arr = root.getJSONArray("b2")
                            for (i in 0 until minOf(trainer.h2Dim, b2Arr.length())) {
                                trainer.b2[i] = b2Arr.getDouble(i).toFloat()
                            }
                        }
                        if (root.has("gamma2")) {
                            val gArr = root.getJSONArray("gamma2")
                            for (i in 0 until minOf(trainer.h2Dim, gArr.length())) {
                                trainer.gamma2[i] = gArr.getDouble(i).toFloat()
                            }
                        }
                        if (root.has("beta2")) {
                            val bArr = root.getJSONArray("beta2")
                            for (i in 0 until minOf(trainer.h2Dim, bArr.length())) {
                                trainer.beta2[i] = bArr.getDouble(i).toFloat()
                            }
                        }
                        if (root.has("wSkip")) {
                            val sArr = root.getJSONArray("wSkip")
                            for (i in 0 until minOf(trainer.h2Dim, sArr.length())) {
                                val row = sArr.getJSONArray(i)
                                for (j in 0 until minOf(trainer.h1Dim, row.length())) {
                                    trainer.wSkip[i][j] = row.getDouble(j).toFloat()
                                }
                            }
                        }

                        // Load Layer 3
                        if (root.has("w3")) {
                            val w3Arr = root.getJSONArray("w3")
                            for (k in 0 until minOf(numClasses, w3Arr.length())) {
                                val row = w3Arr.getJSONArray(k)
                                for (j in 0 until minOf(trainer.h2Dim, row.length())) {
                                    trainer.w3[k][j] = row.getDouble(j).toFloat()
                                }
                            }
                        }
                        if (root.has("b3")) {
                            val b3Arr = root.getJSONArray("b3")
                            for (k in 0 until minOf(numClasses, b3Arr.length())) {
                                trainer.b3[k] = b3Arr.getDouble(k).toFloat()
                            }
                        }
                    } else if (trainer.architecture == ModelArchitecture.STANDARD_MLP) {
                        // Load Standard MLP
                        if (root.has("w1")) {
                            val w1Arr = root.getJSONArray("w1")
                            for (i in 0 until minOf(trainer.h1Dim, w1Arr.length())) {
                                val row = w1Arr.getJSONArray(i)
                                for (j in 0 until minOf(trainer.featureDim, row.length())) {
                                    trainer.w1[i][j] = row.getDouble(j).toFloat()
                                }
                            }
                        }
                        if (root.has("b1")) {
                            val b1Arr = root.getJSONArray("b1")
                            for (i in 0 until minOf(trainer.h1Dim, b1Arr.length())) {
                                trainer.b1[i] = b1Arr.getDouble(i).toFloat()
                            }
                        }
                        if (root.has("w3")) {
                            val w3Arr = root.getJSONArray("w3")
                            for (k in 0 until minOf(numClasses, w3Arr.length())) {
                                val row = w3Arr.getJSONArray(k)
                                for (j in 0 until minOf(trainer.h1Dim, row.length())) {
                                    trainer.w3[k][j] = row.getDouble(j).toFloat()
                                }
                            }
                        }
                        if (root.has("b3")) {
                            val b3Arr = root.getJSONArray("b3")
                            for (k in 0 until minOf(numClasses, b3Arr.length())) {
                                trainer.b3[k] = b3Arr.getDouble(k).toFloat()
                            }
                        }
                    } else if (root.has("weights")) {
                        val weightsRoot = root.getJSONArray("weights")
                        for (k in 0 until numClasses) {
                            if (k < weightsRoot.length()) {
                                val classWeights = weightsRoot.getJSONArray(k)
                                for (j in 0 until featureDim) {
                                    if (j < classWeights.length()) {
                                        trainer.weights[k][j] = classWeights.getDouble(j).toFloat()
                                    }
                                }
                            }
                        }
                        if (root.has("biases")) {
                            val bArr = root.getJSONArray("biases")
                            for (b in 0 until minOf(numClasses, bArr.length())) {
                                trainer.biases[b] = bArr.getDouble(b).toFloat()
                            }
                        }
                    }
                } else {
                    // Legacy linear model weights
                    trainer.architecture = ModelArchitecture.LINEAR
                    val weightsRoot = JSONArray(resolvedWeights)
                    for (k in 0 until numClasses) {
                        if (k < weightsRoot.length()) {
                            val classWeights = weightsRoot.getJSONArray(k)
                            for (j in 0 until featureDim) {
                                if (j < classWeights.length()) {
                                    trainer.weights[k][j] = classWeights.getDouble(j).toFloat()
                                }
                            }
                        }
                    }
                    val biasArray = JSONArray(resolvedBiases)
                    for (b in 0 until numClasses) {
                        if (b < biasArray.length()) {
                            trainer.biases[b] = biasArray.getDouble(b).toFloat()
                        }
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }

            // Restore feature scaler if present
            if (!resolvedMeans.isNullOrBlank() && !resolvedStds.isNullOrBlank()) {
                try {
                    val meansArray = JSONArray(resolvedMeans)
                    val stdsArray = JSONArray(resolvedStds)
                    if (meansArray.length() == featureDim && stdsArray.length() == featureDim) {
                        val means = FloatArray(featureDim) { meansArray.getDouble(it).toFloat() }
                        val stds = FloatArray(featureDim) { stdsArray.getDouble(it).toFloat() }
                        trainer.scaler = FeatureScaler(means, stds)
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }

            trainer.isTrained = true
            return trainer
        }
    }
}
