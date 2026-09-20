package com.example.ml

import android.app.ActivityManager
import android.content.ContentUris
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.PowerManager
import android.os.StatFs
import android.provider.MediaStore
import androidx.documentfile.provider.DocumentFile
import com.example.data.repository.ProjectRepository
import com.example.util.AppLogger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.Locale

/**
 * Strategy for handling image files when sorting into category folders.
 * MOVE avoids 2x duplicate storage on large 10,000+ photo collections.
 */
enum class FileSortAction(val displayName: String, val description: String) {
    MOVE(
        displayName = "Move / Cut (Save Storage)",
        description = "Moves photos directly into categories without duplicate storage usage (recommended for 10,000+ files)"
    ),
    COPY(
        displayName = "Copy (Keep Originals)",
        description = "Keeps original photos in the source folder (requires 2x storage space)"
    )
}

/**
 * Pacing & hardware thermal modes to prevent high-load crashes or thermal throttling.
 */
enum class SorterPacingMode(val displayName: String, val description: String) {
    AUTO(
        displayName = "Auto-Adaptive",
        description = "Self-tunes speed & RAM protection based on your device hardware specifications"
    ),
    COOL_ECO(
        displayName = "Eco / Cool (Battery & Heat Saver)",
        description = "Minimizes CPU temperature & battery drain for long 10,000+ image sessions"
    ),
    TURBO(
        displayName = "Turbo (Maximum Speed)",
        description = "Highest throughput for high-performance phones"
    )
}

data class BatchSortState(
    val isRunning: Boolean = false,
    val isPaused: Boolean = false,
    val isCompleted: Boolean = false,
    val currentImageIndex: Int = 0,
    val totalImages: Int = 0,
    val currentImageName: String = "",
    val detectedLabel: String = "",
    val confidence: Float = 0f,
    val sortedSummary: Map<String, Int> = emptyMap(),
    val skippedCount: Int = 0,
    val skipIfNoFaceDetected: Boolean = false,
    val errorMessage: String? = null,
    val sourceDisplayName: String = "",
    val destinationDisplayName: String = "",
    val activeModelName: String = "",
    val fileAction: FileSortAction = FileSortAction.MOVE,
    val pacingMode: SorterPacingMode = SorterPacingMode.AUTO,
    val availableRamMb: Long = 0L,
    val freeStorageGb: Float = 0f,
    val hardwareTierName: String = "",
    val logs: List<String> = emptyList()
)

/**
 * Lightweight representation of an image item to prevent holding heavy DocumentFile
 * object graphs in memory when processing 10,000+ items.
 */
private sealed class QueuedImageItem {
    abstract val name: String
    abstract fun openStream(context: Context): InputStream?
    abstract fun deleteSource(context: Context): Boolean

    class FileSource(val file: File) : QueuedImageItem() {
        override val name: String = file.name
        override fun openStream(context: Context): InputStream? = if (file.exists()) file.inputStream() else null
        override fun deleteSource(context: Context): Boolean = try {
            file.delete()
        } catch (_: Exception) {
            false
        }
    }

    class DocSource(val uri: Uri, override val name: String) : QueuedImageItem() {
        override fun openStream(context: Context): InputStream? =
            context.contentResolver.openInputStream(uri)
        override fun deleteSource(context: Context): Boolean = try {
            val doc = DocumentFile.fromSingleUri(context, uri)
            doc?.delete() ?: false
        } catch (_: Exception) {
            false
        }
    }

    class MediaStoreSource(val uri: Uri, override val name: String) : QueuedImageItem() {
        override fun openStream(context: Context): InputStream? = try {
            context.contentResolver.openInputStream(uri)
        } catch (_: Exception) {
            null
        }

        override fun deleteSource(context: Context): Boolean = try {
            context.contentResolver.delete(uri, null, null) > 0
        } catch (_: Exception) {
            false
        }
    }
}

class BatchFolderSorter private constructor(private val appContext: Context) {

    private val sorterScope = CoroutineScope(Dispatchers.IO + Job())
    private var sortingJob: Job? = null
    private var wakeLock: PowerManager.WakeLock? = null

    private val _state = MutableStateFlow(BatchSortState())
    val state: StateFlow<BatchSortState> = _state.asStateFlow()

    private val activityManager = appContext.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager

    fun startSorting(
        projectId: Long,
        projectName: String,
        sourceUriOrPath: String,
        sourceDisplayName: String,
        destinationUriOrPath: String,
        destinationDisplayName: String,
        repository: ProjectRepository,
        fileAction: FileSortAction = FileSortAction.MOVE,
        pacingMode: SorterPacingMode = SorterPacingMode.AUTO,
        skipIfNoFaceDetected: Boolean = false
    ) {
        startSortingInternal(
            projectId = projectId,
            projectName = projectName,
            customModel = null,
            sourceUriOrPath = sourceUriOrPath,
            sourceDisplayName = sourceDisplayName,
            destinationUriOrPath = destinationUriOrPath,
            destinationDisplayName = destinationDisplayName,
            repository = repository,
            fileAction = fileAction,
            pacingMode = pacingMode,
            skipIfNoFaceDetected = skipIfNoFaceDetected
        )
    }

    fun startSortingWithCustomModel(
        customModel: LoadedExportedModel,
        sourceUriOrPath: String,
        sourceDisplayName: String,
        destinationUriOrPath: String,
        destinationDisplayName: String,
        fileAction: FileSortAction = FileSortAction.MOVE,
        pacingMode: SorterPacingMode = SorterPacingMode.AUTO,
        skipIfNoFaceDetected: Boolean = false
    ) {
        startSortingInternal(
            projectId = null,
            projectName = customModel.fileName,
            customModel = customModel,
            sourceUriOrPath = sourceUriOrPath,
            sourceDisplayName = sourceDisplayName,
            destinationUriOrPath = destinationUriOrPath,
            destinationDisplayName = destinationDisplayName,
            repository = null,
            fileAction = fileAction,
            pacingMode = pacingMode,
            skipIfNoFaceDetected = skipIfNoFaceDetected
        )
    }

    private fun startSortingInternal(
        projectId: Long?,
        projectName: String,
        customModel: LoadedExportedModel?,
        sourceUriOrPath: String,
        sourceDisplayName: String,
        destinationUriOrPath: String,
        destinationDisplayName: String,
        repository: ProjectRepository?,
        fileAction: FileSortAction = FileSortAction.MOVE,
        pacingMode: SorterPacingMode = SorterPacingMode.AUTO,
        skipIfNoFaceDetected: Boolean = false
    ) {
        if (_state.value.isRunning) {
            stopSorting()
        }

        val cleanSource = sourceUriOrPath.trim()
        val cleanDest = destinationUriOrPath.trim()

        val hardwareProfile = AutoTuner.getHardwareProfile(appContext)
        val initialRamMb = getAvailableRamMb()
        val initialStorageGb = getAvailableStorageGb(cleanDest)

        val activeModelLabel = customModel?.let { "${it.fileName} (${it.formatName})" } ?: projectName
        val classesInfo = customModel?.let { "${it.numClasses} categories" } ?: ""
        val faceFilterLog = if (skipIfNoFaceDetected) " | 👤 Face Filter: Skip Non-Faces" else ""

        _state.value = BatchSortState(
            isRunning = true,
            isPaused = false,
            isCompleted = false,
            sourceDisplayName = sourceDisplayName,
            destinationDisplayName = destinationDisplayName,
            activeModelName = activeModelLabel,
            fileAction = fileAction,
            pacingMode = pacingMode,
            skipIfNoFaceDetected = skipIfNoFaceDetected,
            availableRamMb = initialRamMb,
            freeStorageGb = initialStorageGb,
            hardwareTierName = hardwareProfile.tier.displayName,
            logs = listOf(
                "🚀 Initiating Batch Auto-Sorter for model: '$activeModelLabel' $classesInfo".trim(),
                "📱 Device Spec: ${hardwareProfile.description} (${hardwareProfile.tier.displayName})",
                "⚙️ Config: Mode = ${pacingMode.displayName} | Storage Action = ${fileAction.displayName}$faceFilterLog",
                "📂 Source: $sourceDisplayName",
                "📂 Target: $destinationDisplayName"
            )
        )

        acquireWakeLock()

        sortingJob = sorterScope.launch {
            var featureExtractor: FeatureExtractor? = null
            var faceEngine: FaceRecognitionEngine? = null
            try {
                // 1. Verify and instantiate neural model
                val trainer: OnDeviceTrainer = if (customModel != null) {
                    addLog("📦 Using imported custom model: ${customModel.fileName} (${customModel.formatName}) with ${customModel.numClasses} classes")
                    customModel.trainer
                } else {
                    if (projectId == null || repository == null) {
                        val msg = "No valid model or project specified."
                        addLog("❌ $msg", isError = true)
                        _state.value = _state.value.copy(isRunning = false, errorMessage = msg)
                        return@launch
                    }
                    val model = repository.getDirectTrainedModel(projectId)
                    if (model == null) {
                        val msg = "Model '$projectName' is not trained yet! Please train it first."
                        addLog("❌ $msg", isError = true)
                        _state.value = _state.value.copy(
                            isRunning = false,
                            errorMessage = msg
                        )
                        return@launch
                    }
                    OnDeviceTrainer.loadFromModel(
                        weightsJson = model.weightsJson,
                        biasesJson = model.biasJson,
                        labelsJson = model.classLabelsJson,
                        numClasses = model.numClasses,
                        featureDim = model.featureDim,
                        scaleMeansJson = model.featureScaleMeansJson,
                        scaleStdsJson = model.featureScaleStdsJson
                    )
                }

                // 2. Scan images from source (Memory-optimized queue)
                val imageItems = mutableListOf<QueuedImageItem>()
                val supportedExtensions = setOf("jpg", "jpeg", "png", "webp", "bmp")

                if (cleanSource == "all_device_photos" || cleanSource.startsWith("mediastore")) {
                    addLog("📱 Scanning all photos from phone storage (Images only)...")
                    val projection = arrayOf(
                        MediaStore.Images.Media._ID,
                        MediaStore.Images.Media.DISPLAY_NAME,
                        MediaStore.Images.Media.MIME_TYPE,
                        MediaStore.Images.Media.SIZE
                    )
                    val selection = "${MediaStore.Images.Media.MIME_TYPE} LIKE 'image/%'"
                    val sortOrder = "${MediaStore.Images.Media.DATE_MODIFIED} DESC"
                    try {
                        appContext.contentResolver.query(
                            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                            projection,
                            selection,
                            null,
                            sortOrder
                        )?.use { cursor ->
                            val idCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
                            val nameCol = cursor.getColumnIndex(MediaStore.Images.Media.DISPLAY_NAME)
                            val mimeCol = cursor.getColumnIndex(MediaStore.Images.Media.MIME_TYPE)
                            val sizeCol = cursor.getColumnIndex(MediaStore.Images.Media.SIZE)

                            while (cursor.moveToNext()) {
                                val id = cursor.getLong(idCol)
                                val size = if (sizeCol >= 0) cursor.getLong(sizeCol) else 1L
                                if (size <= 0) continue

                                val name = if (nameCol >= 0) cursor.getString(nameCol) else null
                                val mime = if (mimeCol >= 0) cursor.getString(mimeCol) else null
                                val displayName = name ?: "device_photo_$id.jpg"
                                val ext = displayName.substringAfterLast('.', "").lowercase(Locale.ROOT)

                                if (mime?.startsWith("image/") == true || ext in supportedExtensions) {
                                    val contentUri = ContentUris.withAppendedId(
                                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                                        id
                                    )
                                    imageItems.add(QueuedImageItem.MediaStoreSource(contentUri, displayName))
                                }
                            }
                        }
                        addLog("📱 Discovered ${imageItems.size} photos across phone gallery & storage.")
                    } catch (e: Exception) {
                        val msg = "Could not access device photos: ${e.message}"
                        addLog("❌ $msg", isError = true)
                        _state.value = _state.value.copy(isRunning = false, errorMessage = msg)
                        return@launch
                    }
                } else if (cleanSource.startsWith("content://")) {
                    val sourceUri = Uri.parse(cleanSource)
                    val treeDoc = DocumentFile.fromTreeUri(appContext, sourceUri)
                    if (treeDoc == null || !treeDoc.isDirectory) {
                        val msg = "Cannot access selected source folder. Please re-select and grant permission."
                        addLog("❌ $msg", isError = true)
                        _state.value = _state.value.copy(isRunning = false, errorMessage = msg)
                        return@launch
                    }

                    val files = treeDoc.listFiles()
                    for (f in files) {
                        if (f.isFile) {
                            val fileName = f.name ?: "image_${imageItems.size + 1}.jpg"
                            val ext = fileName.substringAfterLast('.', "").lowercase(Locale.ROOT)
                            val mime = f.type ?: ""
                            if (ext in supportedExtensions || mime.startsWith("image/")) {
                                imageItems.add(QueuedImageItem.DocSource(f.uri, fileName))
                            }
                        }
                    }
                } else {
                    val srcDir = File(cleanSource)
                    if (!srcDir.exists() || !srcDir.isDirectory) {
                        val msg = "Source folder does not exist: $cleanSource"
                        addLog("❌ $msg", isError = true)
                        _state.value = _state.value.copy(isRunning = false, errorMessage = msg)
                        return@launch
                    }

                    val files = srcDir.listFiles { file ->
                        file.isFile && file.extension.lowercase(Locale.ROOT) in supportedExtensions
                    }?.sortedBy { it.name } ?: emptyList()

                    files.forEach { imageItems.add(QueuedImageItem.FileSource(it)) }
                }

                if (imageItems.isEmpty()) {
                    val msg = "No image files found in the source folder. Please verify the folder contains JPEG/PNG photos."
                    addLog("⚠️ $msg", isError = true)
                    _state.value = _state.value.copy(isRunning = false, errorMessage = msg)
                    return@launch
                }

                val totalCount = imageItems.size
                _state.value = _state.value.copy(
                    totalImages = totalCount,
                    currentImageIndex = 0
                )
                addLog("🔍 Found $totalCount photos queued. Memory-safe batch sorting running...")

                // 3. Prepare Destination Target and Category Folder Cache
                val isDestContentUri = cleanDest.startsWith("content://")
                val destTreeDoc = if (isDestContentUri) {
                    val doc = DocumentFile.fromTreeUri(appContext, Uri.parse(cleanDest))
                    if (doc == null || !doc.isDirectory) {
                        val msg = "Cannot access destination folder. Please re-select and grant permission."
                        addLog("❌ $msg", isError = true)
                        _state.value = _state.value.copy(isRunning = false, errorMessage = msg)
                        return@launch
                    }
                    doc
                } else null

                val destFileDir = if (!isDestContentUri) {
                    val dir = File(cleanDest)
                    if (!dir.exists()) dir.mkdirs()
                    dir
                } else null

                // Performance cache to prevent thousands of slow DocumentFile / File searches
                val categoryDocCache = mutableMapOf<String, DocumentFile>()
                val categoryDirCache = mutableMapOf<String, File>()

                // 4. Hardware-adaptive thread allocation & pacing
                val tfliteThreads = when (pacingMode) {
                    SorterPacingMode.COOL_ECO -> 2
                    SorterPacingMode.TURBO -> 4.coerceAtMost(hardwareProfile.cpuCores)
                    SorterPacingMode.AUTO -> when (hardwareProfile.tier) {
                        DevicePerformanceTier.ENTRY_LEVEL -> 2
                        DevicePerformanceTier.MID_RANGE -> 3.coerceAtMost(hardwareProfile.cpuCores)
                        DevicePerformanceTier.HIGH_PERFORMANCE -> 4.coerceAtMost(hardwareProfile.cpuCores)
                    }
                }

                val pacingDelayMs: Long = when (pacingMode) {
                    SorterPacingMode.COOL_ECO -> 35L
                    SorterPacingMode.TURBO -> 0L
                    SorterPacingMode.AUTO -> when (hardwareProfile.tier) {
                        DevicePerformanceTier.ENTRY_LEVEL -> 25L
                        DevicePerformanceTier.MID_RANGE -> 8L
                        DevicePerformanceTier.HIGH_PERFORMANCE -> 2L
                    }
                }

                val directProject = if (projectId != null) repository?.getDirectProject(projectId) else null
                val isFaceModel = projectName.startsWith("[Face ID]") ||
                        projectName.contains("Face", ignoreCase = true) ||
                        (directProject?.projectType == "FACE_RECOGNITION" || directProject?.name?.contains("Face", ignoreCase = true) == true) ||
                        (customModel?.fileName?.contains("Face", ignoreCase = true) == true) ||
                        skipIfNoFaceDetected

                if (isFaceModel || skipIfNoFaceDetected) {
                    faceEngine = FaceRecognitionEngine(appContext)
                }

                if (!isFaceModel) {
                    featureExtractor = FeatureExtractor(appContext, numThreads = tfliteThreads)
                }

                val enrolledPersons: List<EnrolledPerson> = if (isFaceModel && faceEngine != null) {
                    addLog("👤 Face Biometric Model active: using facial crop detection and cosine centroid matching.")
                    trainer.classLabels.mapIndexed { cIdx, label ->
                        val centroid = FloatArray(trainer.featureDim) { f ->
                            trainer.weights[cIdx][f] / 8.0f
                        }
                        EnrolledPerson(
                            id = cIdx.toLong(),
                            name = label,
                            faceSamplePaths = emptyList(),
                            centroidEmbedding = centroid
                        )
                    }
                } else {
                    emptyList()
                }

                var skippedCount = 0

                val summary = mutableMapOf<String, Int>()
                val streamBuffer = ByteArray(65536) // 64KB fast buffered stream
                var lastUiUpdateTime = 0L

                // 5. Process each photo with strict memory & storage management
                for ((index, item) in imageItems.withIndex()) {
                    // Check Pause state
                    while (_state.value.isPaused && _state.value.isRunning && isActive) {
                        delay(300)
                    }

                    if (!isActive || !_state.value.isRunning) {
                        addLog("🛑 Auto-sorting was stopped.")
                        break
                    }

                    // Periodic Storage Check (prevent filling disk on large batches)
                    if (index % 50 == 0) {
                        val currentStorageGb = getAvailableStorageGb(cleanDest)
                        _state.value = _state.value.copy(
                            availableRamMb = getAvailableRamMb(),
                            freeStorageGb = currentStorageGb
                        )

                        if (currentStorageGb < 0.25f && fileAction == FileSortAction.COPY) {
                            addLog("⚠️ Low device storage (${String.format(Locale.US, "%.2f", currentStorageGb)} GB left). Pausing to prevent disk full crash.", isError = true)
                            _state.value = _state.value.copy(
                                isPaused = true,
                                errorMessage = "Device storage critically low (<250MB). Please clear storage or select 'Move' mode."
                            )
                            while (_state.value.isPaused && _state.value.isRunning && isActive) {
                                delay(500)
                            }
                        }
                    }

                    // Memory Health Guard (prevent OOM on 10,000+ items)
                    if (index % 25 == 0) {
                        val memInfo = ActivityManager.MemoryInfo()
                        activityManager.getMemoryInfo(memInfo)
                        val lowMemory = memInfo.lowMemory || (memInfo.availMem.toFloat() / memInfo.totalMem.toFloat()) < 0.12f
                        if (lowMemory) {
                            addLog("⚠️ High memory pressure detected (${memInfo.availMem / (1024 * 1024)}MB free). Pausing 350ms to reclaim RAM...")
                            System.gc()
                            delay(350)
                        } else if (index % 50 == 0) {
                            yield() // Cooperative coroutine dispatch
                        }
                    }

                    // Adaptive Thermal & CPU Pacing
                    if (pacingDelayMs > 0L) {
                        delay(pacingDelayMs)
                    }

                    // Decode with gentle downsampling to preserve facial features while saving RAM
                    val decodeBoundsOpts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    item.openStream(appContext)?.use {
                        BitmapFactory.decodeStream(it, null, decodeBoundsOpts)
                    }

                    val maxDim = maxOf(decodeBoundsOpts.outWidth, decodeBoundsOpts.outHeight)
                    var sampleSize = 1
                    while (maxDim / (sampleSize * 2) >= 640) {
                        sampleSize *= 2
                    }

                    val actualOpts = BitmapFactory.Options().apply {
                        inSampleSize = sampleSize
                        inPreferredConfig = Bitmap.Config.ARGB_8888 // Full 32-bit true color for 100% feature extraction accuracy matching training
                    }

                    val bitmap = item.openStream(appContext)?.use {
                        BitmapFactory.decodeStream(it, null, actualOpts)
                    }

                    if (bitmap == null) {
                        skippedCount++
                        addLog("⚠️ Could not decode '${item.name}', skipping.")
                        _state.value = _state.value.copy(
                            currentImageIndex = index + 1,
                            currentImageName = item.name,
                            skippedCount = skippedCount
                        )
                        continue
                    }

                    var rawLabel = ""
                    var predictionConfidence = 0f

                    if (isFaceModel && faceEngine != null && enrolledPersons.isNotEmpty()) {
                        // Accurate Face Recognition Pipeline: Detect face, crop, extract 128D embedding, match against enrolled centroids
                        val matchThreshold = repository?.getFaceMatchThreshold() ?: 0.45f
                        val identified = faceEngine.identifyHumansInScene(
                            sceneBitmap = bitmap,
                            enrolledPersons = enrolledPersons,
                            matchThreshold = matchThreshold,
                            maxPersons = 1
                        )

                        if (identified.isEmpty()) {
                            if (skipIfNoFaceDetected) {
                                bitmap.recycle()
                                skippedCount++
                                addLog("⏭️ [Skipped] No human or face detected in '${item.name}', moving to next photo...")
                                _state.value = _state.value.copy(
                                    currentImageIndex = index + 1,
                                    currentImageName = item.name,
                                    detectedLabel = "Skipped (No Face Detected)",
                                    confidence = 0f,
                                    skippedCount = skippedCount
                                )
                                continue
                            } else {
                                rawLabel = "No Face Detected"
                                predictionConfidence = 0f
                            }
                        } else {
                            val topPerson = identified[0]
                            if (topPerson.personName == "Unknown Person") {
                                if (enrolledPersons.size == 1) {
                                    rawLabel = enrolledPersons[0].name
                                    predictionConfidence = topPerson.confidence
                                } else {
                                    rawLabel = "Unknown Person"
                                    predictionConfidence = topPerson.confidence
                                }
                            } else {
                                rawLabel = topPerson.personName
                                predictionConfidence = topPerson.confidence
                            }
                        }
                    } else {
                        // Standard Image Classification check for face / human filter if enabled
                        if (skipIfNoFaceDetected && faceEngine != null) {
                            val detectedFaces = faceEngine.detectFaces(bitmap, maxFaces = 1)
                            val detectedBodies = if (detectedFaces.isEmpty()) faceEngine.detectHumanBodies(bitmap, maxBodies = 1) else emptyList()
                            if (detectedFaces.isEmpty() && detectedBodies.isEmpty()) {
                                bitmap.recycle()
                                skippedCount++
                                addLog("⏭️ [Skipped] No human or face detected in '${item.name}', moving to next photo...")
                                _state.value = _state.value.copy(
                                    currentImageIndex = index + 1,
                                    currentImageName = item.name,
                                    detectedLabel = "Skipped (No Face Detected)",
                                    confidence = 0f,
                                    skippedCount = skippedCount
                                )
                                continue
                            }
                        }

                        val extractor = featureExtractor ?: FeatureExtractor(appContext, numThreads = tfliteThreads).also { featureExtractor = it }
                        val features = extractor.extractFeatures(bitmap)
                        val prediction = trainer.predict(features)
                        rawLabel = prediction.classLabel
                        predictionConfidence = prediction.confidence
                    }

                    bitmap.recycle() // Immediately recycle native bitmap memory

                    val safeCategoryName = rawLabel.trim().replace("/", "_").replace("\\", "_").ifBlank { "Unclassified" }

                    var sortedSuccess = false

                    if (destTreeDoc != null) {
                        // SAF Document Target with category folder caching
                        var categoryDoc = categoryDocCache[safeCategoryName]
                        if (categoryDoc == null || !categoryDoc.isDirectory) {
                            categoryDoc = destTreeDoc.findFile(safeCategoryName)
                            if (categoryDoc == null || !categoryDoc.isDirectory) {
                                categoryDoc = destTreeDoc.createDirectory(safeCategoryName)
                                if (categoryDoc != null) {
                                    addLog("📁 Created new category folder: '/$safeCategoryName'")
                                }
                            }
                            if (categoryDoc != null) {
                                categoryDocCache[safeCategoryName] = categoryDoc
                            }
                        }

                        if (categoryDoc != null) {
                            val targetFile = categoryDoc.createFile("image/jpeg", item.name)
                            if (targetFile != null) {
                                item.openStream(appContext)?.use { inStream ->
                                    appContext.contentResolver.openOutputStream(targetFile.uri)?.use { outStream ->
                                        inStream.copyTo(outStream, bufferSize = 65536)
                                        sortedSuccess = true
                                    }
                                }
                                // If MOVE action: delete source to save duplicate storage
                                if (sortedSuccess && fileAction == FileSortAction.MOVE) {
                                    item.deleteSource(appContext)
                                }
                            }
                        }
                    } else if (destFileDir != null) {
                        // Standard filesystem target with directory caching
                        var categoryFolder = categoryDirCache[safeCategoryName]
                        if (categoryFolder == null || !categoryFolder.exists()) {
                            categoryFolder = File(destFileDir, safeCategoryName)
                            if (!categoryFolder.exists()) {
                                val created = categoryFolder.mkdirs()
                                if (created) {
                                    addLog("📁 Created new category folder: '/$safeCategoryName'")
                                }
                            }
                            categoryDirCache[safeCategoryName] = categoryFolder
                        }

                        var destFile = File(categoryFolder, item.name)
                        if (destFile.exists()) {
                            val baseName = item.name.substringBeforeLast('.')
                            val ext = item.name.substringAfterLast('.', "jpg")
                            var counter = 1
                            while (destFile.exists()) {
                                destFile = File(categoryFolder, "${baseName}_$counter.$ext")
                                counter++
                            }
                        }

                        if (fileAction == FileSortAction.MOVE && item is QueuedImageItem.FileSource) {
                            // Instant atomic file move (0 extra disk space, 0 memory used!)
                            val renamed = item.file.renameTo(destFile)
                            if (renamed) {
                                sortedSuccess = true
                            } else {
                                // Fallback copy + delete across mount points
                                item.openStream(appContext)?.use { inStream ->
                                    FileOutputStream(destFile).use { outStream ->
                                        inStream.copyTo(outStream, bufferSize = 65536)
                                        sortedSuccess = true
                                    }
                                }
                                if (sortedSuccess) {
                                    item.deleteSource(appContext)
                                }
                            }
                        } else {
                            // Standard copy
                            item.openStream(appContext)?.use { inStream ->
                                FileOutputStream(destFile).use { outStream ->
                                    inStream.copyTo(outStream, bufferSize = 65536)
                                    sortedSuccess = true
                                }
                            }
                            if (sortedSuccess && fileAction == FileSortAction.MOVE) {
                                item.deleteSource(appContext)
                            }
                        }
                    }

                    if (sortedSuccess) {
                        summary[safeCategoryName] = (summary[safeCategoryName] ?: 0) + 1
                    }

                    // UI Throttling: Update StateFlow at most once per 200ms or on completion
                    val now = System.currentTimeMillis()
                    val isFirstOrLast = index == 0 || index == totalCount - 1
                    if (isFirstOrLast || now - lastUiUpdateTime >= 200L) {
                        lastUiUpdateTime = now
                        _state.value = _state.value.copy(
                            currentImageIndex = index + 1,
                            currentImageName = item.name,
                            detectedLabel = safeCategoryName,
                            confidence = predictionConfidence,
                            sortedSummary = summary.toMap(),
                            availableRamMb = getAvailableRamMb()
                        )
                    }

                    // Milestone logging (log first 5, then every 25 images to avoid log array bloat)
                    if (index < 5 || (index + 1) % 25 == 0 || index == totalCount - 1) {
                        val pct = String.format(Locale.US, "%.1f%%", predictionConfidence * 100)
                        if (isFaceModel) {
                            if (rawLabel == "Unknown Person") {
                                addLog("👤 [${index + 1}/$totalCount] ${item.name} ➔ Unrecognized Face ($pct) ➔ /$safeCategoryName")
                            } else if (rawLabel == "No Face Detected") {
                                addLog("📁 [${index + 1}/$totalCount] ${item.name} ➔ No Face Detected ➔ /$safeCategoryName")
                            } else {
                                addLog("✅ [${index + 1}/$totalCount] ${item.name} ➔ '$safeCategoryName' ($pct)")
                            }
                        } else {
                            addLog("✅ [${index + 1}/$totalCount] ${item.name} ➔ '$safeCategoryName' ($pct)")
                        }
                    }
                }

                if (isActive && _state.value.isRunning && !_state.value.isPaused) {
                    val skippedMsg = if (skippedCount > 0) ", $skippedCount photos skipped (no face detected)" else ""
                    addLog("🎉 Sorting complete! Processed ${_state.value.currentImageIndex} photos. Sorted ${summary.values.sum()} photos into ${summary.size} category folders$skippedMsg.")
                    _state.value = _state.value.copy(
                        currentImageIndex = totalCount,
                        isRunning = false,
                        isPaused = false,
                        isCompleted = true,
                        sortedSummary = summary.toMap(),
                        skippedCount = skippedCount
                    )
                }

            } catch (e: Exception) {
                AppLogger.e("BatchFolderSorter", "Error during sorting: ${e.message}", e)
                addLog("❌ Error: ${e.message}", isError = true)
                _state.value = _state.value.copy(
                    isRunning = false,
                    isPaused = false,
                    errorMessage = e.message
                )
            } finally {
                featureExtractor?.close()
                faceEngine?.close()
                releaseWakeLock()
            }
        }
    }

    fun pauseSorting() {
        if (_state.value.isRunning && !_state.value.isPaused) {
            _state.value = _state.value.copy(isPaused = true)
            releaseWakeLock()
            addLog("⏸️ Auto-sorting paused. Background CPU and device resources released.")
        }
    }

    fun resumeSorting() {
        if (_state.value.isRunning && _state.value.isPaused) {
            acquireWakeLock()
            _state.value = _state.value.copy(isPaused = false)
            addLog("▶️ Resuming batch auto-sorting...")
        }
    }

    fun stopSorting() {
        sortingJob?.cancel()
        sortingJob = null
        releaseWakeLock()
        _state.value = _state.value.copy(isRunning = false, isPaused = false)
        addLog("🛑 Auto-sorting stopped by user.")
    }

    fun resetState() {
        _state.value = BatchSortState()
    }

    private fun acquireWakeLock() {
        try {
            if (wakeLock == null) {
                val powerManager = appContext.getSystemService(Context.POWER_SERVICE) as PowerManager
                wakeLock = powerManager.newWakeLock(
                    PowerManager.PARTIAL_WAKE_LOCK,
                    "BatchFolderSorter:WakeLock"
                ).apply {
                    setReferenceCounted(false)
                }
            }
            if (wakeLock?.isHeld != true) {
                wakeLock?.acquire(6 * 60 * 60 * 1000L) // 6-hour safe timeout
            }
        } catch (_: Exception) {}
    }

    private fun releaseWakeLock() {
        try {
            if (wakeLock?.isHeld == true) {
                wakeLock?.release()
            }
        } catch (_: Exception) {}
    }

    private fun getAvailableRamMb(): Long {
        return try {
            val memInfo = ActivityManager.MemoryInfo()
            activityManager.getMemoryInfo(memInfo)
            memInfo.availMem / (1024 * 1024)
        } catch (_: Exception) {
            0L
        }
    }

    private fun getAvailableStorageGb(pathOrUri: String): Float {
        return try {
            val targetDir = if (pathOrUri.startsWith("content://")) {
                appContext.filesDir
            } else {
                File(pathOrUri)
            }
            val stat = StatFs(targetDir.absolutePath)
            val availableBytes = stat.availableBlocksLong * stat.blockSizeLong
            (availableBytes.toDouble() / (1024.0 * 1024.0 * 1024.0)).toFloat()
        } catch (_: Exception) {
            0f
        }
    }

    suspend fun generateDemoImages(
        targetDir: File,
        projectId: Long,
        repository: ProjectRepository
    ): Int = withContext(Dispatchers.IO) {
        if (!targetDir.exists()) {
            targetDir.mkdirs()
        }

        var createdCount = 0

        val projectSamples = repository.getAllDirectSamples(projectId)
        if (projectSamples.isNotEmpty()) {
            for ((index, sample) in projectSamples.take(12).withIndex()) {
                val srcFile = File(sample.imagePath)
                if (srcFile.exists()) {
                    val destFile = File(targetDir, "test_sample_${index + 1}.${srcFile.extension}")
                    srcFile.copyTo(destFile, overwrite = true)
                    createdCount++
                }
            }
        }

        if (createdCount < 6) {
            val classes = repository.getDirectClasses(projectId)
            val colors = listOf(Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.CYAN, Color.MAGENTA)
            val needMore = 8 - createdCount

            for (i in 0 until needMore) {
                val label = classes.getOrNull(i % maxOf(1, classes.size))?.className ?: "Class_${i + 1}"
                val color = colors[i % colors.size]

                val bmp = Bitmap.createBitmap(300, 300, Bitmap.Config.RGB_565)
                val canvas = Canvas(bmp)
                canvas.drawColor(color)

                val paint = Paint().apply {
                    this.color = Color.WHITE
                    textSize = 32f
                    isAntiAlias = true
                    textAlign = Paint.Align.CENTER
                }
                canvas.drawText(label, 150f, 150f, paint)

                val outFile = File(targetDir, "generated_test_${createdCount + 1}.jpg")
                FileOutputStream(outFile).use { out ->
                    bmp.compress(Bitmap.CompressFormat.JPEG, 85, out)
                }
                bmp.recycle()
                createdCount++
            }
        }

        addLog("✨ Generated $createdCount demo test images in: ${targetDir.name}")
        createdCount
    }

    private fun addLog(message: String, isError: Boolean = false) {
        val formatted = if (isError) "⚠️ $message" else message
        _state.value = _state.value.copy(
            logs = (_state.value.logs + formatted).takeLast(40)
        )
    }

    /**
     * Returns the total count of images currently available in the device MediaStore.
     */
    fun getDevicePhotosCount(): Int {
        val projection = arrayOf(MediaStore.Images.Media._ID)
        val selection = "${MediaStore.Images.Media.MIME_TYPE} LIKE 'image/%'"
        return try {
            appContext.contentResolver.query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                projection,
                selection,
                null,
                null
            )?.use { it.count } ?: 0
        } catch (_: Exception) {
            0
        }
    }

    companion object {
        @Volatile
        private var instance: BatchFolderSorter? = null

        fun getInstance(context: Context): BatchFolderSorter {
            return instance ?: synchronized(this) {
                instance ?: BatchFolderSorter(context.applicationContext).also { instance = it }
            }
        }
    }
}
