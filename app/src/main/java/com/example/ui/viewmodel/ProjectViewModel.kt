package com.example.ui.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.data.db.*
import com.example.data.repository.ProjectRepository
import com.example.ml.ExportedModelResult
import com.example.ml.LearningRateSchedule
import com.example.ml.LoadedExportedModel
import com.example.ml.ModelArchitecture
import com.example.ml.OptimizerType
import com.example.ml.PredictionResult
import com.example.ml.ProjectTrainingConfig
import com.example.ml.TrainingPhase
import com.example.ml.TrainingProgress
import com.example.service.TrainingManager
import com.example.util.AppLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

enum class AppMode {
    IMAGE_CLASSIFICATION,
    FACE_RECOGNITION
}

class ProjectViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: ProjectRepository

    fun getRepository(): ProjectRepository = repository

    private val appModePrefs = application.getSharedPreferences("app_mode_prefs", android.content.Context.MODE_PRIVATE)
    private val _currentAppMode = MutableStateFlow(
        try {
            AppMode.valueOf(appModePrefs.getString("current_app_mode", AppMode.IMAGE_CLASSIFICATION.name) ?: AppMode.IMAGE_CLASSIFICATION.name)
        } catch (_: Exception) {
            AppMode.IMAGE_CLASSIFICATION
        }
    )
    val currentAppMode: StateFlow<AppMode> = _currentAppMode.asStateFlow()

    fun setAppMode(mode: AppMode) {
        _currentAppMode.value = mode
        appModePrefs.edit().putString("current_app_mode", mode.name).apply()
    }

    val projects: StateFlow<List<ProjectEntity>>

    private val _selectedProjectId = MutableStateFlow<Long?>(null)
    val selectedProjectId: StateFlow<Long?> = _selectedProjectId.asStateFlow()

    @OptIn(ExperimentalCoroutinesApi::class)
    val currentProject: StateFlow<ProjectEntity?> = _selectedProjectId.flatMapLatest { id ->
        if (id != null) repository.getProject(id) else flowOf(null)
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), null)

    @OptIn(ExperimentalCoroutinesApi::class)
    val projectClasses: StateFlow<List<ClassificationClassEntity>> = _selectedProjectId.flatMapLatest { id ->
        if (id != null) repository.getClassesForProject(id) else flowOf(emptyList())
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    @OptIn(ExperimentalCoroutinesApi::class)
    val projectTotalSamples: StateFlow<Int> = _selectedProjectId.flatMapLatest { id ->
        if (id != null) repository.getTotalSampleCountForProject(id) else flowOf(0)
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), 0)

    @OptIn(ExperimentalCoroutinesApi::class)
    val latestModel: StateFlow<TrainedModelEntity?> = _selectedProjectId.flatMapLatest { id ->
        if (id != null) repository.getLatestTrainedModel(id) else flowOf(null)
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), null)

    val trainingProgress: StateFlow<TrainingProgress?> = TrainingManager.trainingProgress

    val isTraining: StateFlow<Boolean> = TrainingManager.isTraining

    private val _inferenceResult = MutableStateFlow<PredictionResult?>(null)
    val inferenceResult: StateFlow<PredictionResult?> = _inferenceResult.asStateFlow()

    private val _isInferenceRunning = MutableStateFlow(false)
    val isInferenceRunning: StateFlow<Boolean> = _isInferenceRunning.asStateFlow()

    private val _exportedModels = MutableStateFlow<List<ExportedModelResult>>(emptyList())
    val exportedModels: StateFlow<List<ExportedModelResult>> = _exportedModels.asStateFlow()

    private val _pythonScript = MutableStateFlow("")
    val pythonScript: StateFlow<String> = _pythonScript.asStateFlow()

    // Device Thermal & Battery Protection Mode
    private val _deviceProtectionEnabled = MutableStateFlow(true)
    val deviceProtectionEnabled: StateFlow<Boolean> = _deviceProtectionEnabled.asStateFlow()

    // Inference Engine Selection & Loaded Model Testing
    private val _selectedEngineMode = MutableStateFlow(InferenceEngineMode.ACTIVE_TRAINED_MODEL)
    val selectedEngineMode: StateFlow<InferenceEngineMode> = _selectedEngineMode.asStateFlow()

    private val _loadedExportedModel = MutableStateFlow<LoadedExportedModel?>(null)
    val loadedExportedModel: StateFlow<LoadedExportedModel?> = _loadedExportedModel.asStateFlow()

    private val _modelLoadMessage = MutableStateFlow<String?>(null)
    val modelLoadMessage: StateFlow<String?> = _modelLoadMessage.asStateFlow()

    // Active Learning & Continuous Feedback Tracking
    private val _feedbackSamplesCount = MutableStateFlow(0)
    val feedbackSamplesCount: StateFlow<Int> = _feedbackSamplesCount.asStateFlow()

    // Smart Auto-Tune Global State & Project Config Cache
    private val _isAutoTuneGlobalEnabled = MutableStateFlow(true)
    val isAutoTuneGlobalEnabled: StateFlow<Boolean> = _isAutoTuneGlobalEnabled.asStateFlow()

    private val _projectConfigs = mutableMapOf<Long, ProjectTrainingConfig>()
    private val _configVersion = MutableStateFlow(0)
    val configVersion: StateFlow<Int> = _configVersion.asStateFlow()

    fun setAutoTuneGlobalEnabled(enabled: Boolean) {
        _isAutoTuneGlobalEnabled.value = enabled
        if (enabled) {
            // When user re-enables Auto-Tune in Settings, reset customized configs so projects re-tune
            _projectConfigs.clear()
            _configVersion.value += 1
        }
    }

    fun resetAllProjectsToAutoTune() {
        _projectConfigs.clear()
        _configVersion.value += 1
    }

    fun getTrainingConfig(
        projectId: Long,
        context: android.content.Context,
        totalSamples: Int,
        classCount: Int
    ): ProjectTrainingConfig {
        val existing = _projectConfigs[projectId]
        if (existing != null) {
            if (existing.isUserModified) {
                return existing
            }
            if (_isAutoTuneGlobalEnabled.value && totalSamples > 0) {
                val tuned = com.example.ml.AutoTuner.computeOptimalParameters(context, totalSamples, classCount)
                val updatedConfig = ProjectTrainingConfig(
                    architecture = tuned.architecture,
                    optimizer = tuned.optimizer,
                    lrSchedule = tuned.lrSchedule,
                    batchSize = tuned.batchSize,
                    epochs = tuned.epochs,
                    learningRate = tuned.learningRate,
                    isUserModified = false
                )
                _projectConfigs[projectId] = updatedConfig
                return updatedConfig
            }
            return existing
        }

        val currentProj = currentProject.value
        if (currentProj != null && currentProj.id == projectId && currentProj.isTrained) {
            val dbConfig = ProjectTrainingConfig(
                architecture = ModelArchitecture.DEEP_RESIDUAL_MLP,
                optimizer = OptimizerType.ADAM_W,
                lrSchedule = LearningRateSchedule.COSINE_ANNEALING,
                batchSize = currentProj.batchSize.coerceIn(8, 256),
                epochs = currentProj.trainingEpochs.toFloat().coerceIn(10f, 100f),
                learningRate = currentProj.learningRate.coerceIn(0.0001f, 0.1f),
                isUserModified = true
            )
            _projectConfigs[projectId] = dbConfig
            return dbConfig
        }

        if (_isAutoTuneGlobalEnabled.value) {
            val tuned = com.example.ml.AutoTuner.computeOptimalParameters(context, totalSamples, classCount)
            val config = ProjectTrainingConfig(
                architecture = tuned.architecture,
                optimizer = tuned.optimizer,
                lrSchedule = tuned.lrSchedule,
                batchSize = tuned.batchSize,
                epochs = tuned.epochs,
                learningRate = tuned.learningRate,
                isUserModified = false
            )
            _projectConfigs[projectId] = config
            return config
        }
        val fallback = ProjectTrainingConfig(
            architecture = ModelArchitecture.DEEP_RESIDUAL_MLP,
            optimizer = OptimizerType.ADAM_W,
            lrSchedule = LearningRateSchedule.COSINE_ANNEALING,
            batchSize = 32,
            epochs = 30f,
            learningRate = 0.003f,
            isUserModified = false
        )
        _projectConfigs[projectId] = fallback
        return fallback
    }

    fun updateTrainingConfig(projectId: Long, config: ProjectTrainingConfig) {
        _projectConfigs[projectId] = config.copy(isUserModified = true)
        _configVersion.value += 1
        viewModelScope.launch {
            repository.updateProjectHyperparameters(
                projectId = projectId,
                epochs = config.epochs.toInt(),
                batchSize = config.batchSize,
                learningRate = config.learningRate
            )
        }
    }

    // Detection Mode: Single Object vs Multi-Object
    private val _detectionMode = MutableStateFlow(ObjectDetectionMode.SINGLE_OBJECT)
    val detectionMode: StateFlow<ObjectDetectionMode> = _detectionMode.asStateFlow()

    fun setDetectionMode(mode: ObjectDetectionMode) {
        _detectionMode.value = mode
    }

    // Biometric Face Recognition Match Sensitivity / Threshold
    private val _faceMatchThreshold = MutableStateFlow(0.60f)
    val faceMatchThreshold: StateFlow<Float> = _faceMatchThreshold.asStateFlow()

    fun setFaceMatchThreshold(threshold: Float) {
        val clamped = threshold.coerceIn(0.20f, 0.95f)
        _faceMatchThreshold.value = clamped
        repository.setFaceMatchThreshold(clamped)
    }

    // Dataset Class Balancing & Leveling (rebalances weights when sample counts differ e.g. 300 vs 20)
    private val _classBalancingEnabled = MutableStateFlow(true)
    val classBalancingEnabled: StateFlow<Boolean> = _classBalancingEnabled.asStateFlow()

    fun toggleClassBalancing(enabled: Boolean) {
        _classBalancingEnabled.value = enabled
    }

    init {
        val database = AppDatabase.getDatabase(application)
        repository = ProjectRepository(application, database.projectDao())
        projects = repository.allProjects.stateIn(
            viewModelScope,
            SharingStarted.WhileSubscribed(5000),
            emptyList()
        )
    }

    fun selectProject(projectId: Long) {
        if (_selectedProjectId.value != projectId) {
            _selectedProjectId.value = projectId
            _inferenceResult.value = null
            if (!isTraining.value) {
                TrainingManager.setTrainingFinished(null)
            }
        }
    }

    fun cancelTraining() {
        TrainingManager.stopTraining(getApplication())
    }

    fun createProject(
        name: String,
        description: String,
        projectType: String = if (_currentAppMode.value == AppMode.FACE_RECOGNITION) "FACE_RECOGNITION" else "IMAGE_CLASSIFICATION",
        onCreated: (Long) -> Unit
    ) {
        viewModelScope.launch {
            val id = repository.createProject(name, description, projectType)
            _selectedProjectId.value = id
            onCreated(id)
        }
    }

    fun deleteProject(projectId: Long) {
        viewModelScope.launch {
            repository.deleteProject(projectId)
            if (_selectedProjectId.value == projectId) {
                _selectedProjectId.value = null
            }
        }
    }

    fun addClass(className: String, colorHex: String) {
        val projectId = _selectedProjectId.value ?: return
        viewModelScope.launch {
            repository.addClass(projectId, className, colorHex)
        }
    }

    fun updateClass(classId: Long, newClassName: String, newColorHex: String) {
        val projectId = _selectedProjectId.value ?: return
        viewModelScope.launch {
            repository.updateClass(projectId, classId, newClassName, newColorHex)
        }
    }

    fun importDatasetFromZip(zipUri: Uri, onResult: (String) -> Unit) {
        val projectId = _selectedProjectId.value ?: return
        viewModelScope.launch {
            val result = repository.importDatasetFromZip(projectId, zipUri)
            onResult(result)
        }
    }

    fun deleteClass(classId: Long) {
        viewModelScope.launch {
            repository.deleteClass(classId)
        }
    }

    fun addSample(classId: Long, bitmap: Bitmap) {
        val projectId = _selectedProjectId.value ?: return
        viewModelScope.launch {
            repository.addImageSample(classId, projectId, bitmap)
        }
    }

    fun submitInferenceFeedback(
        classId: Long,
        bitmap: Bitmap,
        isCorrection: Boolean,
        onSuccess: (String) -> Unit
    ) {
        val projectId = _selectedProjectId.value ?: return
        viewModelScope.launch {
            repository.addImageSample(classId, projectId, bitmap)
            _feedbackSamplesCount.value += 1
            if (isCorrection) {
                onSuccess("Saved image as corrected sample for re-training!")
            } else {
                onSuccess("Confirmed! Saved image to strengthen model accuracy.")
            }
        }
    }

    fun submitRegionFeedback(
        classId: Long,
        originalBitmap: Bitmap,
        region: com.example.ml.DetectedObjectRegion,
        onSuccess: (String) -> Unit
    ) {
        val projectId = _selectedProjectId.value ?: return
        viewModelScope.launch {
            val w = originalBitmap.width
            val h = originalBitmap.height
            val x = (region.boxLeftNorm * w).toInt().coerceIn(0, w - 1)
            val y = (region.boxTopNorm * h).toInt().coerceIn(0, h - 1)
            val cropW = ((region.boxRightNorm - region.boxLeftNorm) * w).toInt().coerceIn(10, w - x)
            val cropH = ((region.boxBottomNorm - region.boxTopNorm) * h).toInt().coerceIn(10, h - y)
            val cropped = Bitmap.createBitmap(originalBitmap, x, y, cropW, cropH)
            repository.addImageSample(classId, projectId, cropped)
            _feedbackSamplesCount.value += 1
            onSuccess("Region cropped & saved to category for re-training!")
        }
    }

    fun submitMultipleRegionsFeedback(
        regionsWithClasses: List<Pair<com.example.ml.DetectedObjectRegion, Long>>,
        originalBitmap: Bitmap,
        onSuccess: (Int) -> Unit
    ) {
        val projectId = _selectedProjectId.value ?: return
        viewModelScope.launch {
            var count = 0
            val w = originalBitmap.width
            val h = originalBitmap.height
            regionsWithClasses.forEach { (region, classId) ->
                val x = (region.boxLeftNorm * w).toInt().coerceIn(0, w - 1)
                val y = (region.boxTopNorm * h).toInt().coerceIn(0, h - 1)
                val cropW = ((region.boxRightNorm - region.boxLeftNorm) * w).toInt().coerceIn(10, w - x)
                val cropH = ((region.boxBottomNorm - region.boxTopNorm) * h).toInt().coerceIn(10, h - y)
                if (cropW > 5 && cropH > 5) {
                    val cropped = Bitmap.createBitmap(originalBitmap, x, y, cropW, cropH)
                    repository.addImageSample(classId, projectId, cropped)
                    count++
                }
            }
            if (count > 0) {
                _feedbackSamplesCount.value += count
            }
            onSuccess(count)
        }
    }

    fun resetFeedbackCount() {
        _feedbackSamplesCount.value = 0
    }

    fun addSampleFromUri(classId: Long, uri: Uri) {
        val projectId = _selectedProjectId.value ?: return
        viewModelScope.launch {
            repository.addImageSampleFromUri(classId, projectId, uri)
        }
    }

    fun deleteSample(sampleId: Long) {
        viewModelScope.launch {
            repository.deleteSample(sampleId)
        }
    }

    fun getSamplesForClass(classId: Long): Flow<List<ImageSampleEntity>> {
        return repository.getSamplesForClass(classId)
    }

    fun getSampleCountForClass(classId: Long): Flow<Int> {
        return repository.getSampleCountForClass(classId)
    }

    fun toggleDeviceProtection(enabled: Boolean) {
        _deviceProtectionEnabled.value = enabled
    }

    fun setEngineMode(mode: InferenceEngineMode) {
        _selectedEngineMode.value = mode
        _inferenceResult.value = null
        if (mode == InferenceEngineMode.EXPORTED_TFLITE) {
            loadExportedTfLiteForCurrentProject()
        } else if (mode == InferenceEngineMode.ACTIVE_TRAINED_MODEL) {
            _loadedExportedModel.value = null
            _modelLoadMessage.value = "Using Active In-Memory Trained Weights & Scaler"
        }
    }

    fun loadExportedTfLiteForCurrentProject() {
        val project = currentProject.value ?: return
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val file = repository.getExportedTfLiteFileForProject(project.name)
                if (file != null && file.exists()) {
                    val loaded = repository.parseExportedModelFile(file)
                    if (loaded != null) {
                        _loadedExportedModel.value = loaded
                        _modelLoadMessage.value = "Loaded ${file.name} (${loaded.formatName}) • ${loaded.numClasses} classes"
                    } else {
                        _modelLoadMessage.value = "Failed to parse exported .tflite container"
                    }
                } else {
                    _modelLoadMessage.value = "No exported .tflite found yet. Please export models in the Export tab first!"
                }
            } catch (e: Throwable) {
                _modelLoadMessage.value = "Error loading model: ${e.message ?: "Unknown error"}"
            }
        }
    }

    fun loadCustomModelFromUri(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val loaded = repository.parseExportedModelUri(uri)
                if (loaded != null) {
                    _loadedExportedModel.value = loaded
                    _selectedEngineMode.value = InferenceEngineMode.CUSTOM_IMPORTED_FILE
                    _modelLoadMessage.value = "Successfully loaded: ${loaded.fileName} (${loaded.formatName}) • ${loaded.numClasses} classes"
                } else {
                    _modelLoadMessage.value = "Error: Could not parse model file. Please select a valid .tflite, .onnx, .mlmodel, or .json file."
                }
            } catch (e: Throwable) {
                _modelLoadMessage.value = "Error loading custom model: ${e.message ?: "Unknown error"}"
            }
        }
    }

    fun startOnDeviceTraining(
        epochs: Int = 30,
        learningRate: Float = 0.003f,
        batchSize: Int = 16,
        architecture: ModelArchitecture = ModelArchitecture.DEEP_RESIDUAL_MLP,
        optimizerType: OptimizerType = OptimizerType.ADAM_W,
        lrSchedule: LearningRateSchedule = LearningRateSchedule.COSINE_ANNEALING
    ) {
        val projectId = _selectedProjectId.value ?: return
        val projectName = currentProject.value?.name ?: "Model"
        TrainingManager.startTraining(
            context = getApplication(),
            projectId = projectId,
            projectName = projectName,
            epochs = epochs,
            learningRate = learningRate,
            batchSize = batchSize,
            architecture = architecture,
            optimizerType = optimizerType,
            lrSchedule = lrSchedule,
            deviceProtectionEnabled = _deviceProtectionEnabled.value
        )
    }

    fun testImageInference(bitmap: Bitmap, mode: ObjectDetectionMode = _detectionMode.value) {
        val projectId = _selectedProjectId.value ?: return
        _isInferenceRunning.value = true
        val isMulti = (mode == ObjectDetectionMode.MULTI_OBJECT)
        repository.resetPredictionStabilizer()
        AppLogger.i("Inference", "Starting testImageInference. Project: $projectId, Mode: $mode, Bitmap size: ${bitmap.width}x${bitmap.height}, Engine: ${_selectedEngineMode.value}")
        viewModelScope.launch {
            try {
                val result = if (_selectedEngineMode.value != InferenceEngineMode.ACTIVE_TRAINED_MODEL && _loadedExportedModel.value != null) {
                    AppLogger.d("Inference", "Running inference with loaded exported model...")
                    repository.runInferenceWithLoadedModel(_loadedExportedModel.value!!, bitmap, isMulti)
                } else {
                    AppLogger.d("Inference", "Running inference with database active model...")
                    repository.runInference(projectId, bitmap, isMulti)
                }
                _inferenceResult.value = result
                AppLogger.i("Inference", "Inference completed successfully. Predicted class: ${result?.classLabel} (${String.format(java.util.Locale.US, "%.1f%%", (result?.confidence ?: 0f) * 100)}) in ${result?.inferenceTimeMs}ms")
            } catch (e: Throwable) {
                AppLogger.e("Inference", "Error or crash during testImageInference: ${e.message}", e)
            } finally {
                _isInferenceRunning.value = false
            }
        }
    }

    suspend fun runLiveFrameInference(bitmap: Bitmap, isMulti: Boolean): PredictionResult? {
        val projectId = _selectedProjectId.value ?: return null
        return try {
            if (_selectedEngineMode.value != InferenceEngineMode.ACTIVE_TRAINED_MODEL && _loadedExportedModel.value != null) {
                repository.runInferenceWithLoadedModel(_loadedExportedModel.value!!, bitmap, isMulti)
            } else {
                repository.runInference(projectId, bitmap, isMulti)
            }
        } catch (e: Throwable) {
            null
        }
    }

    private val _isExporting = MutableStateFlow(false)
    val isExporting: StateFlow<Boolean> = _isExporting.asStateFlow()

    fun exportModelFiles() {
        val projectId = _selectedProjectId.value ?: return
        if (_isExporting.value) return
        viewModelScope.launch {
            _isExporting.value = true
            try {
                val results = repository.exportModelFormats(projectId)
                _exportedModels.value = results
                val script = repository.getPythonScript(projectId)
                _pythonScript.value = script
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                _isExporting.value = false
            }
        }
    }

    fun clearInferenceResult() {
        _inferenceResult.value = null
    }
}

enum class ObjectDetectionMode(val title: String, val titleBn: String, val subtitle: String) {
    SINGLE_OBJECT("Single Object", "একক অবজেক্ট", "Identify dominant single object"),
    MULTI_OBJECT("Multi Object", "মাল্টি অবজেক্ট", "Scan and identify multiple objects with bounding boxes")
}

enum class InferenceEngineMode(val displayName: String, val subtitle: String) {
    ACTIVE_TRAINED_MODEL("Active Database Model", "Trained model weights with fitted feature standardization"),
    EXPORTED_TFLITE("Exported .tflite Model", "Loads exported TensorFlow Lite model container directly from storage"),
    CUSTOM_IMPORTED_FILE("Custom Exported Model File", "Load any exported .tflite, .onnx, .mlmodel, or .pb file from device storage")
}
