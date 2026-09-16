package com.example.service

import android.content.Context
import android.content.Intent
import androidx.core.content.ContextCompat
import com.example.ml.LearningRateSchedule
import com.example.ml.ModelArchitecture
import com.example.ml.OptimizerType
import com.example.ml.TrainingPhase
import com.example.ml.TrainingProgress
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

object TrainingManager {

    private val _trainingProgress = MutableStateFlow<TrainingProgress?>(null)
    val trainingProgress: StateFlow<TrainingProgress?> = _trainingProgress.asStateFlow()

    private val _isTraining = MutableStateFlow(false)
    val isTraining: StateFlow<Boolean> = _isTraining.asStateFlow()

    private val _activeTrainingProjectId = MutableStateFlow<Long?>(null)
    val activeTrainingProjectId: StateFlow<Long?> = _activeTrainingProjectId.asStateFlow()

    private val _activeProjectName = MutableStateFlow<String>("")
    val activeProjectName: StateFlow<String> = _activeProjectName.asStateFlow()

    fun startTraining(
        context: Context,
        projectId: Long,
        projectName: String,
        epochs: Int = 30,
        learningRate: Float = 0.003f,
        batchSize: Int = 16,
        architecture: ModelArchitecture = ModelArchitecture.DEEP_RESIDUAL_MLP,
        optimizerType: OptimizerType = OptimizerType.ADAM_W,
        lrSchedule: LearningRateSchedule = LearningRateSchedule.COSINE_ANNEALING,
        deviceProtectionEnabled: Boolean = true
    ) {
        _activeTrainingProjectId.value = projectId
        _activeProjectName.value = projectName
        _isTraining.value = true
        _trainingProgress.value = TrainingProgress(
            currentEpoch = 0,
            totalEpochs = epochs,
            loss = 0f,
            accuracy = 0f,
            statusMessage = "Starting deep neural network training service...",
            overallPercentage = 0f,
            phase = TrainingPhase.EXTRACTING_FEATURES
        )

        val intent = Intent(context, TrainingForegroundService::class.java).apply {
            action = TrainingForegroundService.ACTION_START_TRAINING
            putExtra(TrainingForegroundService.EXTRA_PROJECT_ID, projectId)
            putExtra(TrainingForegroundService.EXTRA_PROJECT_NAME, projectName)
            putExtra(TrainingForegroundService.EXTRA_EPOCHS, epochs)
            putExtra(TrainingForegroundService.EXTRA_LEARNING_RATE, learningRate)
            putExtra(TrainingForegroundService.EXTRA_BATCH_SIZE, batchSize)
            putExtra(TrainingForegroundService.EXTRA_ARCHITECTURE, architecture.name)
            putExtra(TrainingForegroundService.EXTRA_OPTIMIZER, optimizerType.name)
            putExtra(TrainingForegroundService.EXTRA_LR_SCHEDULE, lrSchedule.name)
            putExtra(TrainingForegroundService.EXTRA_DEVICE_PROTECTION, deviceProtectionEnabled)
        }

        try {
            ContextCompat.startForegroundService(context.applicationContext, intent)
        } catch (e: Exception) {
            // Fallback for context start
            try {
                context.startService(intent)
            } catch (ignored: Exception) {}
        }
    }

    fun stopTraining(context: Context) {
        val intent = Intent(context, TrainingForegroundService::class.java).apply {
            action = TrainingForegroundService.ACTION_STOP_TRAINING
        }
        try {
            context.startService(intent)
        } catch (ignored: Exception) {}

        _isTraining.value = false
        _trainingProgress.value = _trainingProgress.value?.copy(
            phase = TrainingPhase.CANCELLED,
            statusMessage = "Training stopped by user."
        )
    }

    fun updateProgress(progress: TrainingProgress) {
        _trainingProgress.value = progress
        if (progress.phase == TrainingPhase.COMPLETED ||
            progress.phase == TrainingPhase.CANCELLED ||
            progress.phase == TrainingPhase.ERROR) {
            _isTraining.value = false
        }
    }

    fun setTrainingFinished(finalProgress: TrainingProgress?) {
        _isTraining.value = false
        if (finalProgress != null) {
            _trainingProgress.value = finalProgress
        }
    }
}
