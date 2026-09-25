package com.example.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.example.MainActivity
import com.example.data.db.AppDatabase
import com.example.data.repository.ProjectRepository
import com.example.ml.LearningRateSchedule
import com.example.ml.ModelArchitecture
import com.example.ml.OptimizerType
import com.example.ml.TrainingPhase
import com.example.ml.TrainingProgress
import com.example.util.AppLogger
import kotlinx.coroutines.*
import java.util.Locale

class TrainingForegroundService : Service() {

    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(serviceJob + Dispatchers.Default)

    private var trainingCoroutineJob: Job? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var notificationManager: NotificationManager? = null

    private var currentProjectName: String = "Model"
    private var lastNotificationUpdateTime = 0L
    private var lastNotifiedEpoch = -1

    override fun onCreate() {
        super.onCreate()
        notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        createNotificationChannel()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action ?: return START_NOT_STICKY

        when (action) {
            ACTION_START_TRAINING -> {
                val projectId = intent.getLongExtra(EXTRA_PROJECT_ID, -1L)
                currentProjectName = intent.getStringExtra(EXTRA_PROJECT_NAME) ?: "Model"
                val epochs = intent.getIntExtra(EXTRA_EPOCHS, 30)
                val learningRate = intent.getFloatExtra(EXTRA_LEARNING_RATE, 0.003f)
                val batchSize = intent.getIntExtra(EXTRA_BATCH_SIZE, 16)
                val deviceProtection = intent.getBooleanExtra(EXTRA_DEVICE_PROTECTION, true)

                val archStr = intent.getStringExtra(EXTRA_ARCHITECTURE)
                val architecture = try {
                    ModelArchitecture.valueOf(archStr ?: "")
                } catch (e: Exception) {
                    ModelArchitecture.DEEP_RESIDUAL_MLP
                }

                val optStr = intent.getStringExtra(EXTRA_OPTIMIZER)
                val optimizerType = try {
                    OptimizerType.valueOf(optStr ?: "")
                } catch (e: Exception) {
                    OptimizerType.ADAM_W
                }

                val lrsStr = intent.getStringExtra(EXTRA_LR_SCHEDULE)
                val lrSchedule = try {
                    LearningRateSchedule.valueOf(lrsStr ?: "")
                } catch (e: Exception) {
                    LearningRateSchedule.COSINE_ANNEALING
                }

                val perfStr = intent.getStringExtra(EXTRA_PERFORMANCE_PROFILE)
                val performanceProfile = try {
                    com.example.util.HardwareResourceMonitor.PerformanceProfile.valueOf(perfStr ?: "")
                } catch (e: Exception) {
                    com.example.util.HardwareResourceMonitor.PerformanceProfile.SMART_ADAPTIVE
                }

                if (projectId != -1L) {
                    startTrainingJob(
                        projectId = projectId,
                        epochs = epochs,
                        learningRate = learningRate,
                        batchSize = batchSize,
                        architecture = architecture,
                        optimizerType = optimizerType,
                        lrSchedule = lrSchedule,
                        deviceProtection = deviceProtection,
                        performanceProfile = performanceProfile
                    )
                } else {
                    stopSelf()
                }
            }
            ACTION_STOP_TRAINING -> {
                stopTrainingJob()
                stopSelf()
            }
        }

        return START_NOT_STICKY
    }

    private fun startTrainingJob(
        projectId: Long,
        epochs: Int,
        learningRate: Float,
        batchSize: Int,
        architecture: ModelArchitecture,
        optimizerType: OptimizerType,
        lrSchedule: LearningRateSchedule,
        deviceProtection: Boolean,
        performanceProfile: com.example.util.HardwareResourceMonitor.PerformanceProfile = com.example.util.HardwareResourceMonitor.PerformanceProfile.SMART_ADAPTIVE
    ) {
        acquireWakeLock()

        val initialNotification = buildProgressNotification(
            title = "Training: $currentProjectName",
            statusText = "Starting ${architecture.displayName} background training...",
            progressPercent = 0f,
            isIndeterminate = true
        )

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                ServiceCompat.startForeground(
                    this,
                    NOTIFICATION_ID,
                    initialNotification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
                )
            } else {
                startForeground(NOTIFICATION_ID, initialNotification)
            }
        } catch (e: Exception) {
            try {
                startForeground(NOTIFICATION_ID, initialNotification)
            } catch (ignored: Exception) {}
        }

        trainingCoroutineJob?.cancel()
        trainingCoroutineJob = serviceScope.launch {
            val database = AppDatabase.getDatabase(applicationContext)
            val repository = ProjectRepository(applicationContext, database.projectDao())

            AppLogger.i("TrainingService", "Beginning training job for Project #$projectId ('$currentProjectName'). Epochs: $epochs, Batch: $batchSize, Arch: ${architecture.displayName}")
            try {
                repository.trainModelOnDevice(
                    projectId = projectId,
                    epochs = epochs,
                    learningRate = learningRate,
                    batchSize = batchSize,
                    architecture = architecture,
                    optimizerType = optimizerType,
                    lrSchedule = lrSchedule,
                    deviceProtectionEnabled = deviceProtection,
                    performanceProfile = performanceProfile,
                    onProgress = { progress ->
                        TrainingManager.updateProgress(progress)
                        updateNotificationThrottled(progress)
                        AppLogger.d("TrainingService", "Progress: ${progress.phase.title} | Epoch ${progress.currentEpoch}/$epochs | Loss: ${String.format(Locale.US, "%.4f", progress.loss)} | Acc: ${String.format(Locale.US, "%.1f%%", progress.accuracy * 100f)} | Status: ${progress.statusMessage}")
                    }
                )
                AppLogger.i("TrainingService", "Training job for Project #$projectId completed successfully!")
            } catch (e: Exception) {
                if (e !is CancellationException) {
                    AppLogger.e("TrainingService", "Training job failed with exception: ${e.message}", e)
                    val errorProgress = TrainingProgress(
                        statusMessage = "Training error: ${e.localizedMessage}",
                        phase = TrainingPhase.ERROR
                    )
                    TrainingManager.updateProgress(errorProgress)
                    showFinalNotification(
                        title = "Training Failed",
                        message = e.localizedMessage ?: "Unknown error occurred",
                        isSuccess = false
                    )
                } else {
                    AppLogger.w("TrainingService", "Training job was cancelled.")
                }
            } finally {
                withContext(NonCancellable) {
                    releaseWakeLock()
                    try {
                        ServiceCompat.stopForeground(this@TrainingForegroundService, ServiceCompat.STOP_FOREGROUND_REMOVE)
                        notificationManager?.cancel(NOTIFICATION_ID)
                    } catch (ignored: Exception) {}
                    stopSelf()
                }
            }
        }
    }

    private fun updateNotificationThrottled(progress: TrainingProgress) {
        val now = System.currentTimeMillis()
        val isFinal = progress.phase == TrainingPhase.COMPLETED || progress.phase == TrainingPhase.ERROR || progress.phase == TrainingPhase.CANCELLED
        val isEpochChange = progress.currentEpoch > 0 && progress.currentEpoch != lastNotifiedEpoch

        if (isFinal || isEpochChange || now - lastNotificationUpdateTime >= 500L) {
            lastNotificationUpdateTime = now
            if (progress.currentEpoch > 0) {
                lastNotifiedEpoch = progress.currentEpoch
            }

            if (progress.phase == TrainingPhase.COMPLETED) {
                showFinalNotification(
                    title = "Training Complete: $currentProjectName",
                    message = "Accuracy: ${String.format(Locale.US, "%.1f%%", progress.accuracy * 100)} • Model ready for inference",
                    isSuccess = true
                )
            } else if (progress.phase != TrainingPhase.CANCELLED && progress.phase != TrainingPhase.ERROR) {
                val etaText = if (progress.estimatedRemainingSeconds > 0) " • ETA: ${progress.estimatedRemainingSeconds}s" else ""
                val statusText = "${progress.phase.title} (${String.format(Locale.US, "%.1f%%", progress.overallPercentage)})$etaText"

                val notification = buildProgressNotification(
                    title = "Training: $currentProjectName",
                    statusText = statusText,
                    progressPercent = progress.overallPercentage,
                    isIndeterminate = progress.overallPercentage <= 0f
                )
                notificationManager?.notify(NOTIFICATION_ID, notification)
            }
        }
    }

    private fun showFinalNotification(title: String, message: String, isSuccess: Boolean) {
        // Cancel the ongoing progress notification
        try {
            notificationManager?.cancel(NOTIFICATION_ID)
        } catch (ignored: Exception) {}

        val openAppIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val openAppPendingIntent = PendingIntent.getActivity(
            this,
            0,
            openAppIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val notification = NotificationCompat.Builder(this, COMPLETION_CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(message)
            .setSmallIcon(if (isSuccess) android.R.drawable.stat_sys_upload_done else android.R.drawable.stat_notify_error)
            .setContentIntent(openAppPendingIntent)
            .setAutoCancel(true)
            .setOngoing(false)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setDefaults(NotificationCompat.DEFAULT_ALL)
            .build()

        try {
            notificationManager?.notify(NOTIFICATION_COMPLETION_ID, notification)
        } catch (ignored: Exception) {}
    }

    private fun buildProgressNotification(
        title: String,
        statusText: String,
        progressPercent: Float,
        isIndeterminate: Boolean
    ): Notification {
        val openAppIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val openAppPendingIntent = PendingIntent.getActivity(
            this,
            0,
            openAppIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val stopIntent = Intent(this, TrainingForegroundService::class.java).apply {
            action = ACTION_STOP_TRAINING
        }
        val stopPendingIntent = PendingIntent.getService(
            this,
            1,
            stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(statusText)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentIntent(openAppPendingIntent)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .addAction(android.R.drawable.ic_menu_close_clear_cancel, "Stop", stopPendingIntent)

        if (isIndeterminate) {
            builder.setProgress(100, 0, true)
        } else {
            builder.setProgress(100, progressPercent.toInt().coerceIn(0, 100), false)
        }

        return builder.build()
    }

    private fun acquireWakeLock() {
        try {
            if (wakeLock == null) {
                val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
                wakeLock = powerManager.newWakeLock(
                    PowerManager.PARTIAL_WAKE_LOCK,
                    "ModelTrainer:TrainingWakeLock"
                ).apply {
                    setReferenceCounted(false)
                }
            }
            wakeLock?.acquire(45 * 60 * 1000L) // Safe 45-minute timeout
        } catch (ignored: Exception) {}
    }

    private fun releaseWakeLock() {
        try {
            if (wakeLock?.isHeld == true) {
                wakeLock?.release()
            }
        } catch (ignored: Exception) {}
    }

    private fun stopTrainingJob() {
        trainingCoroutineJob?.cancel()
        TrainingManager.setTrainingFinished(
            TrainingProgress(
                phase = TrainingPhase.CANCELLED,
                statusMessage = "Training stopped by user."
            )
        )
        releaseWakeLock()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val progressChannel = NotificationChannel(
                CHANNEL_ID,
                "On-Device Model Training",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Shows real-time on-device training progress and status in background"
                setShowBadge(false)
            }
            val completionChannel = NotificationChannel(
                COMPLETION_CHANNEL_ID,
                "Model Training Completion",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Notifies when on-device model training has completed"
                setShowBadge(true)
                enableVibration(true)
            }
            notificationManager?.createNotificationChannel(progressChannel)
            notificationManager?.createNotificationChannel(completionChannel)
        }
    }

    override fun onDestroy() {
        stopTrainingJob()
        serviceJob.cancel()
        try {
            notificationManager?.cancel(NOTIFICATION_ID)
        } catch (ignored: Exception) {}
        super.onDestroy()
    }

    companion object {
        const val CHANNEL_ID = "on_device_training_channel"
        const val COMPLETION_CHANNEL_ID = "on_device_training_completion_channel"
        const val NOTIFICATION_ID = 1001
        const val NOTIFICATION_COMPLETION_ID = 1002

        const val ACTION_START_TRAINING = "com.example.service.ACTION_START_TRAINING"
        const val ACTION_STOP_TRAINING = "com.example.service.ACTION_STOP_TRAINING"

        const val EXTRA_PROJECT_ID = "extra_project_id"
        const val EXTRA_PROJECT_NAME = "extra_project_name"
        const val EXTRA_EPOCHS = "extra_epochs"
        const val EXTRA_LEARNING_RATE = "extra_learning_rate"
        const val EXTRA_BATCH_SIZE = "extra_batch_size"
        const val EXTRA_ARCHITECTURE = "extra_architecture"
        const val EXTRA_OPTIMIZER = "extra_optimizer"
        const val EXTRA_LR_SCHEDULE = "extra_lr_schedule"
        const val EXTRA_DEVICE_PROTECTION = "extra_device_protection"
        const val EXTRA_PERFORMANCE_PROFILE = "extra_performance_profile"
    }
}
