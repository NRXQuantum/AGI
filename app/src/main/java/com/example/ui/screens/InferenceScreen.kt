package com.example.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File
import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.example.ml.DetectedObjectRegion
import com.example.ui.components.LiveCameraViewfinder
import com.example.ui.components.CameraTestMode
import com.example.ui.components.LiveSinglePrediction
import com.example.ui.components.LiveDetectedBox
import com.example.ui.viewmodel.InferenceEngineMode
import com.example.ui.viewmodel.ObjectDetectionMode
import com.example.ui.viewmodel.ProjectViewModel
import com.example.util.AppLogger
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun InferenceScreen(
    viewModel: ProjectViewModel,
    onNavigateToExport: () -> Unit,
    onNavigateToTrain: () -> Unit = {},
    onNavigateBack: (() -> Unit)? = null
) {
    val context = LocalContext.current
    val project by viewModel.currentProject.collectAsState()
    val latestModel by viewModel.latestModel.collectAsState()
    val classes by viewModel.projectClasses.collectAsState()
    val inferenceResult by viewModel.inferenceResult.collectAsState()
    val isInferenceRunning by viewModel.isInferenceRunning.collectAsState()
    val engineMode by viewModel.selectedEngineMode.collectAsState()
    val detectionMode by viewModel.detectionMode.collectAsState()
    val loadedExportedModel by viewModel.loadedExportedModel.collectAsState()
    val modelLoadMessage by viewModel.modelLoadMessage.collectAsState()
    val feedbackSamplesCount by viewModel.feedbackSamplesCount.collectAsState()
    val isTraining by viewModel.isTraining.collectAsState()
    val trainingProgress by viewModel.trainingProgress.collectAsState()
    val faceMatchThreshold by viewModel.faceMatchThreshold.collectAsState()
    val isFaceMode = project?.projectType == "FACE_RECOGNITION"

    var testBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var feedbackSubmittedForCurrentPhoto by remember { mutableStateOf<String?>(null) }
    var feedbackStatusMessage by remember { mutableStateOf<String?>(null) }
    var showCorrectionDialog by remember { mutableStateOf(false) }
    var showAddClassInFeedbackDialog by remember { mutableStateOf(false) }
    var newFeedbackClassName by remember { mutableStateOf("") }
    var selectedRegionForCorrection by remember { mutableStateOf<DetectedObjectRegion?>(null) }
    var selectedHighlightIndex by remember { mutableStateOf<Int?>(null) }
    val regionFeedbackSaved = remember { mutableStateMapOf<String, String>() }

    val objectColors = remember {
        listOf(
            Color(0xFF38BDF8), // Sky Blue (Object 1)
            Color(0xFF10B981), // Emerald (Object 2)
            Color(0xFFF59E0B), // Amber (Object 3)
            Color(0xFFA855F7), // Purple (Object 4)
            Color(0xFFF43F5E), // Rose (Object 5)
            Color(0xFF06B6D4)  // Cyan (Object 6)
        )
    }

    // Pick single test photo from gallery
    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            try {
                AppLogger.i("InferenceScreen", "Selected test image from gallery: $uri")
                val inputStream = context.contentResolver.openInputStream(uri)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()
                if (bitmap != null) {
                    testBitmap = bitmap
                    selectedHighlightIndex = null
                    feedbackSubmittedForCurrentPhoto = null
                    feedbackStatusMessage = null
                    regionFeedbackSaved.clear()
                    AppLogger.i("InferenceScreen", "Decoded bitmap successfully (${bitmap.width}x${bitmap.height}, ${bitmap.byteCount / 1024} KB). Triggering testImageInference.")
                    viewModel.testImageInference(bitmap, detectionMode)
                } else {
                    AppLogger.e("InferenceScreen", "Failed to decode bitmap from URI: $uri")
                }
            } catch (e: Throwable) {
                AppLogger.e("InferenceScreen", "Exception during gallery image processing: ${e.message}", e)
            }
        }
    }

    // Camera capture photo
    var currentCameraUri by remember { mutableStateOf<android.net.Uri?>(null) }

    val takePictureLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        if (success && currentCameraUri != null) {
            try {
                val inputStream = context.contentResolver.openInputStream(currentCameraUri!!)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()
                if (bitmap != null) {
                    testBitmap = bitmap
                    selectedHighlightIndex = null
                    feedbackSubmittedForCurrentPhoto = null
                    feedbackStatusMessage = null
                    regionFeedbackSaved.clear()
                    AppLogger.i("InferenceScreen", "Captured full camera photo (${bitmap.width}x${bitmap.height}). Running inference.")
                    viewModel.testImageInference(bitmap, detectionMode)
                }
            } catch (e: Throwable) {
                AppLogger.e("InferenceScreen", "Failed to load captured photo: ${e.message}", e)
                Toast.makeText(context, "Failed to load photo: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    val cameraPreviewLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        if (bitmap != null) {
            try {
                testBitmap = bitmap
                selectedHighlightIndex = null
                feedbackSubmittedForCurrentPhoto = null
                feedbackStatusMessage = null
                regionFeedbackSaved.clear()
                AppLogger.i("InferenceScreen", "Captured camera preview bitmap (${bitmap.width}x${bitmap.height}). Triggering testImageInference.")
                viewModel.testImageInference(bitmap, detectionMode)
            } catch (e: Throwable) {
                AppLogger.e("InferenceScreen", "Exception during camera capture processing: ${e.message}", e)
            }
        }
    }

    fun launchCameraInternal() {
        try {
            val photoFile = File(context.cacheDir, "inference_photo_${System.currentTimeMillis()}.jpg")
            val uri = FileProvider.getUriForFile(
                context,
                "${context.packageName}.fileprovider",
                photoFile
            )
            currentCameraUri = uri
            takePictureLauncher.launch(uri)
        } catch (e: Exception) {
            AppLogger.e("InferenceScreen", "FileProvider capture failed, trying preview: ${e.message}", e)
            try {
                cameraPreviewLauncher.launch(null)
            } catch (ex: Exception) {
                AppLogger.e("InferenceScreen", "Camera launch failed: ${ex.message}", ex)
                Toast.makeText(context, "Unable to launch camera app: ${ex.localizedMessage ?: "Device camera not accessible"}", Toast.LENGTH_LONG).show()
            }
        }
    }

    var showCameraModeDialog by remember { mutableStateOf(false) }
    var showLiveCameraViewfinder by remember { mutableStateOf(false) }
    var selectedCameraMode by remember { mutableStateOf(CameraTestMode.LIVE_SINGLE) }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            showCameraModeDialog = true
        } else {
            Toast.makeText(
                context,
                "Camera permission is required for AI camera detection",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    fun handleCameraClick() {
        val permissionCheck = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.CAMERA
        )
        if (permissionCheck == PackageManager.PERMISSION_GRANTED) {
            showCameraModeDialog = true
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    // File Picker to load ANY exported model (.tflite, .onnx, .mlmodel, .pb)
    val modelFilePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            viewModel.loadCustomModelFromUri(uri)
        }
    }

    // Interactive Testing Overlay Visibility States (Instant toggle without re-scanning)
    var showBoundingBoxOverlay by remember { mutableStateOf(true) }
    var showBodyContourOverlay by remember { mutableStateOf(true) }
    var showFacialMeshOverlay by remember { mutableStateOf(true) }
    var showEnlargedPhotoViewer by remember { mutableStateOf(false) }

    Box(modifier = Modifier.fillMaxSize()) {
    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        topBar = {
            TopAppBar(
                navigationIcon = {
                    if (onNavigateBack != null) {
                        IconButton(
                            onClick = onNavigateBack,
                            modifier = Modifier.testTag("inference_back_btn")
                        ) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                contentDescription = "Back"
                            )
                        }
                    }
                },
                title = {
                    Column(verticalArrangement = Arrangement.Center) {
                        Text(
                            text = if (isFaceMode) "Person & Human Identification" else "Model Testing & Detection",
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                        )
                        Text(
                            text = project?.name ?: (if (isFaceMode) "Multi-Modal Human Re-ID" else "On-Device Inference"),
                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                actions = {
                    IconButton(
                        onClick = onNavigateToExport,
                        modifier = Modifier.testTag("nav_to_export_icon_btn")
                    ) {
                        Icon(Icons.Default.FileDownload, contentDescription = "Export")
                    }
                }
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 14.dp, vertical = 10.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {

            // ==========================================
            // 1. INFERENCE ENGINE & SOURCE SELECTOR
            // ==========================================
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(14.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = if (isFaceMode) Icons.Default.Face else Icons.Default.Memory,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = if (isFaceMode) "Person Identification Source" else "Inference Source",
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                            )
                        }

                        val activeEngineTitle = when (engineMode) {
                            InferenceEngineMode.ACTIVE_TRAINED_MODEL -> {
                                if (latestModel != null || (isFaceMode && project?.isTrained == true)) {
                                    if (isFaceMode) "Human DB (${classes.size} Persons)" else "Active (${String.format(Locale.US, "%.0f%%", latestModel!!.accuracy * 100)})"
                                } else {
                                    if (isFaceMode) "Not Calibrated" else "Not Trained"
                                }
                            }
                            InferenceEngineMode.EXPORTED_TFLITE -> if (loadedExportedModel != null) (if (isFaceMode) "Person .tflite Ready" else ".tflite Ready") else "No .tflite"
                            InferenceEngineMode.CUSTOM_IMPORTED_FILE -> if (loadedExportedModel != null) (if (isFaceMode) "Custom Biometrics" else "Custom Model") else "No File"
                        }
                        Surface(
                            color = MaterialTheme.colorScheme.primaryContainer,
                            shape = RoundedCornerShape(6.dp)
                        ) {
                            Text(
                                text = activeEngineTitle,
                                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                color = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    // 3-Tab Segmented Selector
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(
                                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                                RoundedCornerShape(10.dp)
                            )
                            .padding(3.dp),
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        EnginePillTab(
                            modifier = Modifier.weight(1f),
                            title = if (isFaceMode) "Face DB" else "Active DB",
                            icon = if (isFaceMode) Icons.Default.Face else Icons.Default.Psychology,
                            isSelected = engineMode == InferenceEngineMode.ACTIVE_TRAINED_MODEL,
                            onClick = {
                                viewModel.setEngineMode(InferenceEngineMode.ACTIVE_TRAINED_MODEL)
                                if (testBitmap != null) {
                                    viewModel.testImageInference(testBitmap!!, detectionMode)
                                }
                            },
                            testTag = "engine_active_chip"
                        )

                        EnginePillTab(
                            modifier = Modifier.weight(1f),
                            title = if (isFaceMode) "Exported Face" else "Exported",
                            icon = Icons.Default.FileDownload,
                            isSelected = engineMode == InferenceEngineMode.EXPORTED_TFLITE,
                            onClick = {
                                viewModel.setEngineMode(InferenceEngineMode.EXPORTED_TFLITE)
                                if (testBitmap != null) {
                                    viewModel.testImageInference(testBitmap!!, detectionMode)
                                }
                            },
                            testTag = "engine_tflite_chip"
                        )

                        EnginePillTab(
                            modifier = Modifier.weight(1f),
                            title = if (isFaceMode) "Custom Face" else "Custom File",
                            icon = Icons.Default.FolderOpen,
                            isSelected = engineMode == InferenceEngineMode.CUSTOM_IMPORTED_FILE,
                            onClick = {
                                viewModel.setEngineMode(InferenceEngineMode.CUSTOM_IMPORTED_FILE)
                                if (loadedExportedModel == null) {
                                    modelFilePickerLauncher.launch(arrayOf("*/*"))
                                } else if (testBitmap != null) {
                                    viewModel.testImageInference(testBitmap!!, detectionMode)
                                }
                            },
                            testTag = "engine_load_file_chip"
                        )
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    // Status description for engine
                    Surface(
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(8.dp)) {
                            when (engineMode) {
                                InferenceEngineMode.ACTIVE_TRAINED_MODEL -> {
                                    if (latestModel != null || (isFaceMode && project?.isTrained == true)) {
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Row(verticalAlignment = Alignment.CenterVertically) {
                                                Icon(
                                                    Icons.Default.CheckCircle,
                                                    contentDescription = null,
                                                    tint = Color(0xFF10B981),
                                                    modifier = Modifier.size(16.dp)
                                                )
                                                Spacer(modifier = Modifier.width(6.dp))
                                                Text(
                                                    text = if (isFaceMode) "Active Face ID Biometrics" else "Trained Model Active",
                                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                                                    color = MaterialTheme.colorScheme.onSurface
                                                )
                                            }
                                            Text(
                                                text = if (isFaceMode) {
                                                    "${classes.size} Enrolled • Centroids Active"
                                                } else {
                                                    "${classes.size} categories • ${String.format(Locale.US, "%.1f%%", latestModel!!.accuracy * 100)} acc"
                                                },
                                                style = MaterialTheme.typography.labelSmall,
                                                color = MaterialTheme.colorScheme.primary
                                            )
                                        }
                                    }
                                }
                                InferenceEngineMode.EXPORTED_TFLITE -> {
                                    val currentLoaded = loadedExportedModel
                                    if (currentLoaded != null) {
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Column(modifier = Modifier.weight(1f)) {
                                                Text(
                                                    text = currentLoaded.fileName,
                                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                                                    color = MaterialTheme.colorScheme.primary,
                                                    maxLines = 1
                                                )
                                                Text(
                                                    text = "${currentLoaded.formatName} • ${currentLoaded.classLabels.size} classes",
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                            }
                                            OutlinedButton(
                                                onClick = onNavigateToExport,
                                                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp),
                                                shape = RoundedCornerShape(6.dp),
                                                modifier = Modifier.height(30.dp)
                                            ) {
                                                Text("Re-Export", style = MaterialTheme.typography.labelSmall)
                                            }
                                        }
                                    } else {
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Text(
                                                text = modelLoadMessage ?: "Exported .tflite not found.",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.error,
                                                modifier = Modifier.weight(1f)
                                            )
                                            Button(
                                                onClick = { viewModel.loadExportedTfLiteForCurrentProject() },
                                                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp),
                                                shape = RoundedCornerShape(6.dp),
                                                modifier = Modifier.height(30.dp)
                                            ) {
                                                Text("Load", style = MaterialTheme.typography.labelSmall)
                                            }
                                        }
                                    }
                                }
                                InferenceEngineMode.CUSTOM_IMPORTED_FILE -> {
                                    val currentLoaded = loadedExportedModel
                                    if (currentLoaded != null) {
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Column(modifier = Modifier.weight(1f)) {
                                                Text(
                                                    text = currentLoaded.fileName,
                                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                                                    color = MaterialTheme.colorScheme.primary,
                                                    maxLines = 1
                                                )
                                                Text(
                                                    text = "${currentLoaded.formatName} • ${currentLoaded.classLabels.size} classes",
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                            }
                                            OutlinedButton(
                                                onClick = { modelFilePickerLauncher.launch(arrayOf("*/*")) },
                                                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp),
                                                shape = RoundedCornerShape(6.dp),
                                                modifier = Modifier.height(30.dp)
                                            ) {
                                                Text("Change", style = MaterialTheme.typography.labelSmall)
                                            }
                                        }
                                    } else {
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Text(
                                                text = modelLoadMessage ?: "No model loaded from storage.",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                modifier = Modifier.weight(1f)
                                            )
                                            Button(
                                                onClick = { modelFilePickerLauncher.launch(arrayOf("*/*")) },
                                                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp),
                                                shape = RoundedCornerShape(6.dp),
                                                modifier = Modifier.height(30.dp)
                                            ) {
                                                Text("Choose File", style = MaterialTheme.typography.labelSmall)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ==========================================
            // 2. DETECTION MODE SELECTOR (SINGLE OBJECT VS MULTI OBJECT / SINGLE FACE VS MULTI FACE)
            // ==========================================
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("card_detection_mode_selector"),
                shape = RoundedCornerShape(14.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = if (isFaceMode) Icons.Default.Face else Icons.Default.CenterFocusStrong,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = if (isFaceMode) "Face Detection Mode (ফেস শনাক্তকরণ মোড)" else "Detection Mode (শনাক্তকরণ মোড)",
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                            )
                        }

                        Surface(
                            color = if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT)
                                MaterialTheme.colorScheme.secondaryContainer
                            else
                                Color(0xFF0284C7).copy(alpha = 0.15f),
                            shape = RoundedCornerShape(6.dp)
                        ) {
                            Text(
                                text = if (isFaceMode) {
                                    if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT) "Single Face" else "Multi-Face Group"
                                } else {
                                    if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT) "Single Mode" else "Multi Mode"
                                },
                                style = MaterialTheme.typography.labelSmall.copy(
                                    fontWeight = FontWeight.Bold,
                                    color = if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT)
                                        MaterialTheme.colorScheme.onSecondaryContainer
                                    else
                                        Color(0xFF0284C7)
                                ),
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.5.dp)
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    // Two distinct tabs: Single vs Multi
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(
                                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                                RoundedCornerShape(10.dp)
                            )
                            .padding(4.dp),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        // Single Mode Tab
                        Surface(
                            modifier = Modifier
                                .weight(1f)
                                .clip(RoundedCornerShape(8.dp))
                                .clickable {
                                    if (detectionMode != ObjectDetectionMode.SINGLE_OBJECT) {
                                        viewModel.setDetectionMode(ObjectDetectionMode.SINGLE_OBJECT)
                                        if (testBitmap != null) {
                                            viewModel.testImageInference(testBitmap!!, ObjectDetectionMode.SINGLE_OBJECT)
                                        }
                                    }
                                }
                                .testTag("tab_single_object_mode"),
                            color = if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT)
                                MaterialTheme.colorScheme.surface
                            else
                                Color.Transparent,
                            shape = RoundedCornerShape(8.dp),
                            shadowElevation = if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT) 2.dp else 0.dp
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 9.dp, horizontal = 6.dp),
                                horizontalArrangement = Arrangement.Center,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    imageVector = if (isFaceMode) Icons.Default.Person else Icons.Default.FilterCenterFocus,
                                    contentDescription = null,
                                    tint = if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT)
                                        MaterialTheme.colorScheme.primary
                                    else
                                        MaterialTheme.colorScheme.onSurfaceVariant,
                                    modifier = Modifier.size(17.dp)
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text(
                                        text = if (isFaceMode) "Single Face" else "Single Object",
                                        style = MaterialTheme.typography.labelMedium.copy(
                                            fontWeight = if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT) FontWeight.Bold else FontWeight.Medium
                                        ),
                                        color = if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT)
                                            MaterialTheme.colorScheme.primary
                                        else
                                            MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                    Text(
                                        text = if (isFaceMode) "একক ব্যক্তি / ফেস" else "একক বস্তু",
                                        style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp),
                                        color = if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT)
                                            MaterialTheme.colorScheme.primary.copy(alpha = 0.8f)
                                        else
                                            MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                                    )
                                }
                            }
                        }

                        // Multi Mode Tab
                        Surface(
                            modifier = Modifier
                                .weight(1f)
                                .clip(RoundedCornerShape(8.dp))
                                .clickable {
                                    if (detectionMode != ObjectDetectionMode.MULTI_OBJECT) {
                                        viewModel.setDetectionMode(ObjectDetectionMode.MULTI_OBJECT)
                                        if (testBitmap != null) {
                                            viewModel.testImageInference(testBitmap!!, ObjectDetectionMode.MULTI_OBJECT)
                                        }
                                    }
                                }
                                .testTag("tab_multi_object_mode"),
                            color = if (detectionMode == ObjectDetectionMode.MULTI_OBJECT)
                                MaterialTheme.colorScheme.surface
                            else
                                Color.Transparent,
                            shape = RoundedCornerShape(8.dp),
                            shadowElevation = if (detectionMode == ObjectDetectionMode.MULTI_OBJECT) 2.dp else 0.dp
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 9.dp, horizontal = 6.dp),
                                horizontalArrangement = Arrangement.Center,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    imageVector = if (isFaceMode) Icons.Default.Groups else Icons.Default.Grid4x4,
                                    contentDescription = null,
                                    tint = if (detectionMode == ObjectDetectionMode.MULTI_OBJECT)
                                        Color(0xFF0284C7)
                                    else
                                        MaterialTheme.colorScheme.onSurfaceVariant,
                                    modifier = Modifier.size(17.dp)
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text(
                                        text = if (isFaceMode) "Multi Face" else "Multi Object",
                                        style = MaterialTheme.typography.labelMedium.copy(
                                            fontWeight = if (detectionMode == ObjectDetectionMode.MULTI_OBJECT) FontWeight.Bold else FontWeight.Medium
                                        ),
                                        color = if (detectionMode == ObjectDetectionMode.MULTI_OBJECT)
                                            Color(0xFF0284C7)
                                        else
                                            MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                    Text(
                                        text = if (isFaceMode) "একাধিক ব্যক্তি (গ্রুপ)" else "একাধিক বস্তু",
                                        style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp),
                                        color = if (detectionMode == ObjectDetectionMode.MULTI_OBJECT)
                                            Color(0xFF0284C7).copy(alpha = 0.8f)
                                        else
                                            MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                                    )
                                }
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    // Mode description banner
                    Surface(
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                imageVector = if (isFaceMode) {
                                    if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT) Icons.Default.Face else Icons.Default.Groups
                                } else {
                                    if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT) Icons.Default.Info else Icons.Default.CropFree
                                },
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(15.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = if (isFaceMode) {
                                    if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT)
                                        "Single Face: ক্যামেরায় বা ছবিতে থাকা মূল ব্যক্তির মুখ স্ক্যান করে তার নাম ও পরিচয় চিহ্নিত করবে।"
                                    else
                                        "Multi Face: একই ফ্রেমে বা গ্রুপ ছবিতে থাকা একাধিক ব্যক্তির মুখ আলাদাভাবে স্ক্যান করে পরিচয় দেখাবে।"
                                } else {
                                    if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT)
                                        "Single Object: পুরো ছবির মূল প্রধান বস্তুটি এবং তার ক্যাটাগরি শনাক্ত করবে।"
                                    else
                                        "Multi-Object: ছবির প্রতিটি অংশ স্ক্যান করে একাধিক ভিন্ন ভিন্ন বস্তু শনাক্ত করবে।"
                                },
                                style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.5.sp),
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }

            // ==========================================
            // 2.5 BIOMETRIC MATCH THRESHOLD & SENSITIVITY (FACE MODE ONLY)
            // ==========================================
            if (isFaceMode) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("card_face_threshold_tuning"),
                    shape = RoundedCornerShape(14.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                    border = BorderStroke(1.dp, Color(0xFF6366F1).copy(alpha = 0.4f))
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    imageVector = Icons.Default.Fingerprint,
                                    contentDescription = null,
                                    tint = Color(0xFF6366F1),
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = "Biometric Match Sensitivity",
                                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                                )
                            }

                            Surface(
                                color = Color(0xFF6366F1).copy(alpha = 0.15f),
                                shape = RoundedCornerShape(6.dp)
                            ) {
                                val label = when {
                                    faceMatchThreshold >= 0.72f -> "Strict (${(faceMatchThreshold * 100).toInt()}%)"
                                    faceMatchThreshold >= 0.58f -> "Balanced (${(faceMatchThreshold * 100).toInt()}%)"
                                    else -> "Lenient (${(faceMatchThreshold * 100).toInt()}%)"
                                }
                                Text(
                                    text = label,
                                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                    color = Color(0xFF6366F1),
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.5.dp)
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(8.dp))

                        // Quick Presets
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            val presets = listOf(
                                Triple("Lenient (50%)", 0.50f, "Dim light / angles"),
                                Triple("Balanced (60%)", 0.60f, "Recommended"),
                                Triple("Strict (75%)", 0.75f, "High security")
                            )
                            presets.forEach { (title, valFloat, sub) ->
                                val isSelected = Math.abs(faceMatchThreshold - valFloat) < 0.04f
                                Surface(
                                    modifier = Modifier
                                        .weight(1f)
                                        .clip(RoundedCornerShape(8.dp))
                                        .clickable {
                                            viewModel.setFaceMatchThreshold(valFloat)
                                            if (testBitmap != null) {
                                                viewModel.testImageInference(testBitmap!!, detectionMode)
                                            }
                                        },
                                    shape = RoundedCornerShape(8.dp),
                                    color = if (isSelected) Color(0xFF6366F1).copy(alpha = 0.18f) else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f),
                                    border = BorderStroke(
                                        1.dp,
                                        if (isSelected) Color(0xFF6366F1) else MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f)
                                    )
                                ) {
                                    Column(
                                        modifier = Modifier.padding(vertical = 6.dp, horizontal = 4.dp),
                                        horizontalAlignment = Alignment.CenterHorizontally
                                    ) {
                                        Text(
                                            text = title,
                                            style = MaterialTheme.typography.labelSmall.copy(
                                                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                                                fontSize = 11.sp
                                            ),
                                            color = if (isSelected) Color(0xFF6366F1) else MaterialTheme.colorScheme.onSurface,
                                            maxLines = 1
                                        )
                                        Text(
                                            text = sub,
                                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 9.sp),
                                            color = if (isSelected) Color(0xFF6366F1) else MaterialTheme.colorScheme.onSurfaceVariant,
                                            maxLines = 1
                                        )
                                    }
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(8.dp))

                        // Continuous Slider
                        Slider(
                            value = faceMatchThreshold,
                            onValueChange = {
                                viewModel.setFaceMatchThreshold(it)
                            },
                            onValueChangeFinished = {
                                if (testBitmap != null) {
                                    viewModel.testImageInference(testBitmap!!, detectionMode)
                                }
                            },
                            valueRange = 0.35f..0.85f,
                            steps = 9,
                            modifier = Modifier.fillMaxWidth()
                        )

                        Text(
                            text = "ম্যাচ থ্রেশহোল্ড ${(faceMatchThreshold * 100).toInt()}%: এর চেয়ে কম মিল পাওয়া গেলে ব্যক্তিকে 'Unknown / অপরিচিত' হিসেবে চিহ্নিত করা হবে।",
                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            // ==========================================
            // 3. INPUT IMAGE SECTION (VIEWFINDER & ACTIONS)
            // ==========================================
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(14.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.Image,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = "Test Photo Preview",
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                            )
                        }

                        if (testBitmap != null) {
                            val detectedCount = inferenceResult?.detectedObjects?.size ?: 0
                            if (detectionMode == ObjectDetectionMode.MULTI_OBJECT && detectedCount > 0) {
                                Surface(
                                    color = Color(0xFF0F172A),
                                    border = BorderStroke(1.dp, Color(0xFF38BDF8)),
                                    shape = RoundedCornerShape(6.dp)
                                ) {
                                    Text(
                                        text = "⚡ $detectedCount Objects Detected & Pointed",
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontWeight = FontWeight.Bold,
                                            color = Color(0xFF38BDF8)
                                        ),
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                                    )
                                }
                            } else if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT && detectedCount > 0) {
                                Surface(
                                    color = Color(0xFF0F172A),
                                    border = BorderStroke(1.dp, Color(0xFF10B981)),
                                    shape = RoundedCornerShape(6.dp)
                                ) {
                                    Text(
                                        text = "🎯 Object Located & Pointed",
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontWeight = FontWeight.Bold,
                                            color = Color(0xFF34D399)
                                        ),
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                                    )
                                }
                            } else {
                                Surface(
                                    color = MaterialTheme.colorScheme.secondaryContainer,
                                    shape = RoundedCornerShape(6.dp)
                                ) {
                                    Text(
                                        text = if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT) "🎯 Single Mode" else "Scanning Ready",
                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                        color = MaterialTheme.colorScheme.onSecondaryContainer,
                                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                    )
                                }
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    if (testBitmap != null) {
                        // Quick 1-Click Interactive Overlay Controls (Instantly toggle elements without re-scanning)
                        val allOverlaysHidden = if (isFaceMode) {
                            !showBoundingBoxOverlay && !showBodyContourOverlay && !showFacialMeshOverlay
                        } else {
                            !showBoundingBoxOverlay
                        }
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(bottom = 6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            // Master 1-Click Hide/Show All Toggle Button
                            Surface(
                                onClick = {
                                    val nextState = allOverlaysHidden
                                    showBoundingBoxOverlay = nextState
                                    if (isFaceMode) {
                                        showBodyContourOverlay = nextState
                                        showFacialMeshOverlay = nextState
                                    }
                                },
                                color = if (allOverlaysHidden) Color(0xFFEF4444).copy(alpha = 0.18f) else MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.85f),
                                shape = RoundedCornerShape(8.dp),
                                border = BorderStroke(
                                    1.dp,
                                    if (allOverlaysHidden) Color(0xFFEF4444).copy(alpha = 0.7f) else MaterialTheme.colorScheme.primary.copy(alpha = 0.6f)
                                )
                            ) {
                                Row(
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Icon(
                                        imageVector = if (allOverlaysHidden) Icons.Default.VisibilityOff else Icons.Default.Visibility,
                                        contentDescription = null,
                                        modifier = Modifier.size(13.dp),
                                        tint = if (allOverlaysHidden) Color(0xFFF87171) else MaterialTheme.colorScheme.onPrimaryContainer
                                    )
                                    Text(
                                        text = if (allOverlaysHidden) "Show All" else "Hide All",
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontWeight = FontWeight.Bold,
                                            fontSize = 11.sp
                                        ),
                                        color = if (allOverlaysHidden) Color(0xFFF87171) else MaterialTheme.colorScheme.onPrimaryContainer
                                    )
                                }
                            }

                            // Layer Specific Toggles
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(4.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                // 1. Bounding Box
                                FilterChip(
                                    selected = showBoundingBoxOverlay,
                                    onClick = { showBoundingBoxOverlay = !showBoundingBoxOverlay },
                                    label = { Text(if (isFaceMode) "▣ Face / Body" else "▣ Detection Box", fontSize = 10.sp) },
                                    modifier = Modifier.height(26.dp)
                                )
                                if (isFaceMode) {
                                    // 2. Body Contour
                                    FilterChip(
                                        selected = showBodyContourOverlay,
                                        onClick = { showBodyContourOverlay = !showBodyContourOverlay },
                                        label = { Text("📐 Contour", fontSize = 10.sp) },
                                        modifier = Modifier.height(26.dp)
                                    )
                                    // 3. Face Mesh
                                    FilterChip(
                                        selected = showFacialMeshOverlay,
                                        onClick = { showFacialMeshOverlay = !showFacialMeshOverlay },
                                        label = { Text("🕸️ Mesh", fontSize = 10.sp) },
                                        modifier = Modifier.height(26.dp)
                                    )
                                }
                            }
                        }

                        // Framed Photo Viewfinder (Tappable to enlarge full-screen)
                        Surface(
                            onClick = { showEnlargedPhotoViewer = true },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(220.dp)
                                .clip(RoundedCornerShape(10.dp))
                                .testTag("framed_test_photo_viewfinder"),
                            color = Color(0xFF0F172A),
                            border = BorderStroke(1.5.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.5f))
                        ) {
                            BoxWithConstraints(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .padding(6.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                val containerW = maxWidth
                                val containerH = maxHeight

                                val imgW = testBitmap!!.width.toFloat().coerceAtLeast(1f)
                                val imgH = testBitmap!!.height.toFloat().coerceAtLeast(1f)
                                val imgAspect = imgW / imgH
                                val containerAspect = (containerW / containerH).coerceAtLeast(0.01f)

                                val (renderedW, renderedH) = if (imgAspect > containerAspect) {
                                    containerW to (containerW / imgAspect)
                                } else {
                                    (containerH * imgAspect) to containerH
                                }

                                Box(
                                    modifier = Modifier.size(width = renderedW, height = renderedH)
                                ) {
                                    Image(
                                        bitmap = testBitmap!!.asImageBitmap(),
                                        contentDescription = "Test Image Preview",
                                        contentScale = ContentScale.Fit,
                                        modifier = Modifier.fillMaxSize()
                                    )

                                    // Display full biometric wireframe mesh, body contour, and precision target boxes
                                    if (inferenceResult != null) {
                                        val detected = inferenceResult!!.detectedObjects

                                        // 1. Full-Resolution Biometric & Structural Contour Layer (Face Recognition Only)
                                        if (isFaceMode && (showBodyContourOverlay || showFacialMeshOverlay)) {
                                            Canvas(modifier = Modifier.fillMaxSize()) {
                                                val wPx = size.width
                                                val hPx = size.height

                                                for ((idx, obj) in detected.withIndex()) {
                                                    // A. Structural Body / Visible Silhouette Contour (Matching User Sketch)
                                                    if (showBodyContourOverlay && obj.bodyContourPoints.isNotEmpty()) {
                                                        val contourPath = androidx.compose.ui.graphics.Path()
                                                        obj.bodyContourPoints.forEachIndexed { cIdx, pt ->
                                                            val px = pt.x * wPx
                                                            val py = pt.y * hPx
                                                            if (cIdx == 0) contourPath.moveTo(px, py) else contourPath.lineTo(px, py)
                                                        }
                                                        contourPath.close()

                                                        // Outer Glowing Halo in Orange
                                                        drawPath(
                                                            path = contourPath,
                                                            color = Color(0xFFFF6D00).copy(alpha = 0.35f),
                                                            style = androidx.compose.ui.graphics.drawscope.Stroke(
                                                                width = 6.dp.toPx(),
                                                                cap = androidx.compose.ui.graphics.StrokeCap.Round,
                                                                join = androidx.compose.ui.graphics.StrokeJoin.Round
                                                            )
                                                        )

                                                        // Solid High-Definition Vibrant Body Contour Line
                                                        drawPath(
                                                            path = contourPath,
                                                            color = Color(0xFFFF8800),
                                                            style = androidx.compose.ui.graphics.drawscope.Stroke(
                                                                width = 2.5.dp.toPx(),
                                                                cap = androidx.compose.ui.graphics.StrokeCap.Round,
                                                                join = androidx.compose.ui.graphics.StrokeJoin.Round
                                                            )
                                                        )
                                                    }

                                                    // B. 3D Geodesic Facial Topology Mesh (Wireframe + Micro-Nodes)
                                                    if (showFacialMeshOverlay && obj.facialLandmarks.isNotEmpty()) {
                                                        val landmarkPx = obj.facialLandmarks.map { pt ->
                                                            Offset(pt.x * wPx, pt.y * hPx)
                                                        }

                                                        // Wireframe dual-glow edges
                                                        for (edge in obj.facialMeshEdges) {
                                                            if (edge.first < landmarkPx.size && edge.second < landmarkPx.size) {
                                                                val p1 = landmarkPx[edge.first]
                                                                val p2 = landmarkPx[edge.second]
                                                                // Soft Cyan Glow Halo
                                                                drawLine(
                                                                    color = Color(0xFF06B6D4).copy(alpha = 0.50f),
                                                                    start = p1,
                                                                    end = p2,
                                                                    strokeWidth = 2.8.dp.toPx()
                                                                )
                                                                // Core Brilliant White-Cyan Line
                                                                drawLine(
                                                                    color = Color(0xFFF0F9FF).copy(alpha = 0.95f),
                                                                    start = p1,
                                                                    end = p2,
                                                                    strokeWidth = 1.3.dp.toPx()
                                                                )
                                                            }
                                                        }

                                                        // Glowing biometric node vertices
                                                        for (pt in landmarkPx) {
                                                            drawCircle(
                                                                color = Color(0xFF06B6D4).copy(alpha = 0.55f),
                                                                radius = 4.5.dp.toPx(),
                                                                center = pt
                                                            )
                                                            drawCircle(
                                                                color = Color(0xFF38BDF8),
                                                                radius = 2.4.dp.toPx(),
                                                                center = pt
                                                            )
                                                            drawCircle(
                                                                color = Color.White,
                                                                radius = 1.2.dp.toPx(),
                                                                center = pt
                                                            )
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        if (showBoundingBoxOverlay) {
                                            detected.forEachIndexed { idx, obj ->
                                                val isSelected = selectedHighlightIndex == idx
                                                val baseColor = if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT) {
                                                    Color(0xFF10B981) // High-contrast Emerald for Single Object
                                                } else {
                                                    objectColors[idx % objectColors.size]
                                                }
                                                val activeColor = if (isSelected) Color(0xFFFBBF24) else baseColor

                                                val leftDp = renderedW * obj.boxLeftNorm
                                                val topDp = renderedH * obj.boxTopNorm
                                                val itemWidth = (renderedW * (obj.boxRightNorm - obj.boxLeftNorm)).coerceAtLeast(36.dp)
                                                val itemHeight = (renderedH * (obj.boxBottomNorm - obj.boxTopNorm)).coerceAtLeast(36.dp)

                                                val isNearTop = topDp < 22.dp

                                                Box(
                                                    modifier = Modifier
                                                        .absoluteOffset(x = leftDp, y = topDp)
                                                        .size(width = itemWidth, height = itemHeight)
                                                        .clickable {
                                                            selectedHighlightIndex = if (selectedHighlightIndex == idx) null else idx
                                                        }
                                                        .border(
                                                            width = if (isSelected) 2.5.dp else 1.5.dp,
                                                            color = activeColor.copy(alpha = if (isSelected) 0.95f else 0.75f),
                                                            shape = RoundedCornerShape(4.dp)
                                                        )
                                                        .background(activeColor.copy(alpha = if (isSelected) 0.22f else 0.12f), RoundedCornerShape(4.dp))
                                                ) {
                                                    // High-Tech Reticle Canvas: 4 Corner Brackets + Center Target Reticle & Pinpoint
                                                    Canvas(modifier = Modifier.fillMaxSize()) {
                                                        val strokeW = if (isSelected) 3.dp.toPx() else 2.dp.toPx()
                                                        val bracketLen = (8.dp.toPx()).coerceAtMost(size.minDimension / 4f)

                                                        // 1. Top-Left Bracket
                                                        drawLine(color = activeColor, start = Offset(0f, 0f), end = Offset(bracketLen, 0f), strokeWidth = strokeW)
                                                        drawLine(color = activeColor, start = Offset(0f, 0f), end = Offset(0f, bracketLen), strokeWidth = strokeW)

                                                        // 2. Top-Right Bracket
                                                        drawLine(color = activeColor, start = Offset(size.width, 0f), end = Offset(size.width - bracketLen, 0f), strokeWidth = strokeW)
                                                        drawLine(color = activeColor, start = Offset(size.width, 0f), end = Offset(size.width, bracketLen), strokeWidth = strokeW)

                                                        // 3. Bottom-Left Bracket
                                                        drawLine(color = activeColor, start = Offset(0f, size.height), end = Offset(bracketLen, size.height), strokeWidth = strokeW)
                                                        drawLine(color = activeColor, start = Offset(0f, size.height), end = Offset(0f, size.height - bracketLen), strokeWidth = strokeW)

                                                        // 4. Bottom-Right Bracket
                                                        drawLine(color = activeColor, start = Offset(size.width, size.height), end = Offset(size.width - bracketLen, size.height), strokeWidth = strokeW)
                                                        drawLine(color = activeColor, start = Offset(size.width, size.height), end = Offset(size.width, size.height - bracketLen), strokeWidth = strokeW)

                                                        // Center Target Crosshair & Pinpoint Reticle
                                                        val cx = size.width / 2f
                                                        val cy = size.height / 2f
                                                        val ringRad = 7.dp.toPx().coerceAtMost(size.minDimension / 5f)
                                                        val tickLen = 4.dp.toPx()

                                                        // Outer target circle
                                                        drawCircle(
                                                            color = activeColor.copy(alpha = 0.9f),
                                                            radius = ringRad,
                                                            center = Offset(cx, cy),
                                                            style = androidx.compose.ui.graphics.drawscope.Stroke(width = 1.5.dp.toPx())
                                                        )

                                                        // 4 Crosshair ticks
                                                        drawLine(color = activeColor, start = Offset(cx - ringRad - tickLen, cy), end = Offset(cx - ringRad + 1f, cy), strokeWidth = 1.5.dp.toPx())
                                                        drawLine(color = activeColor, start = Offset(cx + ringRad - 1f, cy), end = Offset(cx + ringRad + tickLen, cy), strokeWidth = 1.5.dp.toPx())
                                                        drawLine(color = activeColor, start = Offset(cx, cy - ringRad - tickLen), end = Offset(cx, cy - ringRad + 1f), strokeWidth = 1.5.dp.toPx())
                                                        drawLine(color = activeColor, start = Offset(cx, cy + ringRad - 1f), end = Offset(cx, cy + ringRad + tickLen), strokeWidth = 1.5.dp.toPx())

                                                        // Center pinpoint
                                                        drawCircle(color = Color.White, radius = 2.5.dp.toPx(), center = Offset(cx, cy))
                                                        drawCircle(color = activeColor, radius = 1.2.dp.toPx(), center = Offset(cx, cy))
                                                    }

                                                    // Top Label Pill Badge
                                                    Surface(
                                                        modifier = Modifier
                                                            .align(if (isNearTop) Alignment.BottomStart else Alignment.TopStart)
                                                            .padding(3.dp),
                                                        color = activeColor,
                                                        shape = RoundedCornerShape(4.dp),
                                                        shadowElevation = 3.dp
                                                    ) {
                                                        Text(
                                                            text = if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT) {
                                                                "🎯 ${obj.classLabel} (${String.format(Locale.US, "%.0f%%", obj.confidence * 100)})"
                                                            } else {
                                                                "#${idx + 1}: ${obj.classLabel} (${String.format(Locale.US, "%.0f%%", obj.confidence * 100)})"
                                                            },
                                                            style = MaterialTheme.typography.labelSmall.copy(
                                                                fontSize = 9.5.sp,
                                                                fontWeight = FontWeight.Bold
                                                            ),
                                                            color = Color.White,
                                                            maxLines = 1,
                                                            modifier = Modifier.padding(horizontal = 5.dp, vertical = 2.dp)
                                                        )
                                                    }

                                                    // Stature & Physical Form Diagnostics Badge
                                                    if (obj.statureDiagnostics.isNotBlank()) {
                                                        Surface(
                                                            modifier = Modifier
                                                                .align(if (isNearTop) Alignment.TopStart else Alignment.BottomStart)
                                                                .padding(3.dp),
                                                            color = Color(0xFF0F172A).copy(alpha = 0.90f),
                                                            shape = RoundedCornerShape(3.dp),
                                                            border = androidx.compose.foundation.BorderStroke(0.8.dp, Color(0xFFF59E0B).copy(alpha = 0.7f))
                                                        ) {
                                                            Text(
                                                                text = "📐 ${obj.statureDiagnostics}",
                                                                style = MaterialTheme.typography.labelSmall.copy(
                                                                    fontSize = 8.5.sp,
                                                                    fontWeight = FontWeight.Medium
                                                                ),
                                                                color = Color(0xFFFDE68A),
                                                                maxLines = 1,
                                                                modifier = Modifier.padding(horizontal = 4.dp, vertical = 1.5.dp)
                                                            )
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(8.dp))

                        // Clear, Change, and Enlarge Action Row
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            OutlinedButton(
                                onClick = {
                                    galleryLauncher.launch(
                                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                    )
                                },
                                modifier = Modifier
                                    .weight(1f)
                                    .height(38.dp)
                                    .testTag("change_photo_btn"),
                                shape = RoundedCornerShape(8.dp),
                                contentPadding = PaddingValues(horizontal = 8.dp)
                            ) {
                                Icon(Icons.Default.PhotoLibrary, contentDescription = null, modifier = Modifier.size(15.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("New Photo", style = MaterialTheme.typography.labelMedium)
                            }

                            OutlinedButton(
                                onClick = { handleCameraClick() },
                                modifier = Modifier
                                    .weight(1f)
                                    .height(38.dp)
                                    .testTag("retake_camera_btn"),
                                shape = RoundedCornerShape(8.dp),
                                contentPadding = PaddingValues(horizontal = 8.dp)
                            ) {
                                Icon(Icons.Default.CameraAlt, contentDescription = null, modifier = Modifier.size(15.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Camera", style = MaterialTheme.typography.labelMedium)
                            }

                            Button(
                                onClick = {
                                    testBitmap = null
                                    selectedHighlightIndex = null
                                    feedbackSubmittedForCurrentPhoto = null
                                    feedbackStatusMessage = null
                                    regionFeedbackSaved.clear()
                                    viewModel.clearInferenceResult()
                                },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = MaterialTheme.colorScheme.errorContainer,
                                    contentColor = MaterialTheme.colorScheme.onErrorContainer
                                ),
                                shape = RoundedCornerShape(8.dp),
                                modifier = Modifier
                                    .weight(1f)
                                    .height(38.dp)
                                    .testTag("clear_test_photo_btn"),
                                contentPadding = PaddingValues(horizontal = 8.dp)
                            ) {
                                Icon(Icons.Default.Close, contentDescription = null, modifier = Modifier.size(15.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Clear", style = MaterialTheme.typography.labelMedium)
                            }
                        }
                    } else {
                        // Empty State Upload Card
                        Surface(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(130.dp)
                                .clip(RoundedCornerShape(10.dp)),
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)
                        ) {
                            Column(
                                modifier = Modifier.fillMaxSize(),
                                verticalArrangement = Arrangement.Center,
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                Icon(
                                    imageVector = Icons.Outlined.AddPhotoAlternate,
                                    contentDescription = null,
                                    modifier = Modifier.size(36.dp),
                                    tint = MaterialTheme.colorScheme.primary
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = if (isFaceMode) "Select or Capture Face to Test" else "Select or Capture a Photo to Test",
                                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                                )
                                Spacer(modifier = Modifier.height(8.dp))

                                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Button(
                                        onClick = {
                                            galleryLauncher.launch(
                                                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                            )
                                        },
                                        modifier = Modifier
                                            .height(34.dp)
                                            .testTag("test_gallery_btn"),
                                        shape = RoundedCornerShape(8.dp),
                                        contentPadding = PaddingValues(horizontal = 12.dp)
                                    ) {
                                        Icon(if (isFaceMode) Icons.Default.Face else Icons.Default.PhotoLibrary, contentDescription = null, modifier = Modifier.size(15.dp))
                                        Spacer(modifier = Modifier.width(4.dp))
                                        Text(if (isFaceMode) "Pick Face Photo" else "Pick Photo", style = MaterialTheme.typography.labelMedium)
                                    }

                                    OutlinedButton(
                                        onClick = { handleCameraClick() },
                                        modifier = Modifier
                                            .height(34.dp)
                                            .testTag("test_camera_btn"),
                                        shape = RoundedCornerShape(8.dp),
                                        contentPadding = PaddingValues(horizontal = 12.dp)
                                    ) {
                                        Icon(Icons.Default.CameraAlt, contentDescription = null, modifier = Modifier.size(15.dp))
                                        Spacer(modifier = Modifier.width(4.dp))
                                        Text(if (isFaceMode) "Scan Face" else "Camera", style = MaterialTheme.typography.labelMedium)
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ==========================================
            // 4. INFERENCE PREDICTION RESULT SECTION
            // ==========================================
            if (isInferenceRunning) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(14.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(18.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center
                    ) {
                        CircularProgressIndicator(modifier = Modifier.size(24.dp))
                        Spacer(modifier = Modifier.width(12.dp))
                        Text(
                            text = if (isFaceMode) {
                                if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT)
                                    "Recognizing facial biometrics..."
                                else
                                    "Scanning multi-face spatial regions for enrolled persons..."
                            } else {
                                if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT)
                                    "Classifying primary object..."
                                else
                                    "Scanning spatial regions for multiple objects..."
                            },
                            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                        )
                    }
                }
            } else if (inferenceResult != null) {
                val res = inferenceResult!!

                // A) MULTI-OBJECT MODE RESULTS CARD
                if (detectionMode == ObjectDetectionMode.MULTI_OBJECT) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(14.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
                        border = BorderStroke(1.5.dp, Color(0xFF38BDF8))
                    ) {
                        Column(modifier = Modifier.padding(14.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        imageVector = Icons.Default.AutoAwesome,
                                        contentDescription = null,
                                        tint = Color(0xFF38BDF8),
                                        modifier = Modifier.size(20.dp)
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = if (isFaceMode) {
                                            "Multi-Face Recognition (${res.detectedObjects.size} Faces Found)"
                                        } else {
                                            "Multi-Object Detections (${res.detectedObjects.size} Found)"
                                        },
                                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                                        color = Color.White
                                    )
                                }

                                Surface(
                                    color = Color(0xFF38BDF8).copy(alpha = 0.2f),
                                    shape = RoundedCornerShape(6.dp)
                                ) {
                                    Text(
                                        text = "⚡ ${res.inferenceTimeMs} ms",
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontWeight = FontWeight.Bold,
                                            color = Color(0xFF38BDF8)
                                        ),
                                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(6.dp))

                            if (res.detectedObjects.isEmpty()) {
                                Surface(
                                    color = Color(0xFF1E293B),
                                    shape = RoundedCornerShape(8.dp),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Column(modifier = Modifier.padding(12.dp)) {
                                        Text(
                                            text = "Single dominant object identified: \"${res.classLabel}\" (${String.format(Locale.US, "%.1f%%", res.confidence * 100)})",
                                            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
                                            color = Color.White
                                        )
                                        Text(
                                            text = "No secondary colliding objects detected in other spatial sections.",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = Color(0xFF94A3B8)
                                        )
                                    }
                                }
                            } else {
                                Column(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    res.detectedObjects.forEachIndexed { idx, obj ->
                                        val color = objectColors[idx % objectColors.size]
                                        val isTaught = regionFeedbackSaved.containsKey(obj.regionTitle)
                                        val taughtCategory = regionFeedbackSaved[obj.regionTitle]
                                        val isSelected = selectedHighlightIndex == idx

                                        Surface(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .clickable {
                                                    selectedHighlightIndex = if (selectedHighlightIndex == idx) null else idx
                                                },
                                            color = Color(0xFF1E293B),
                                            shape = RoundedCornerShape(10.dp),
                                            border = BorderStroke(
                                                width = if (isSelected) 2.dp else 1.dp,
                                                color = if (isSelected) Color(0xFFFBBF24) else color.copy(alpha = 0.6f)
                                            )
                                        ) {
                                            Row(
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .padding(10.dp),
                                                horizontalArrangement = Arrangement.SpaceBetween,
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                Column(modifier = Modifier.weight(1f)) {
                                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                                        Surface(
                                                            modifier = Modifier.size(9.dp),
                                                            shape = CircleShape,
                                                            color = color
                                                        ) {}
                                                        Spacer(modifier = Modifier.width(6.dp))
                                                        Text(
                                                            text = if (isFaceMode) "Person / Face #${idx + 1} (${obj.regionTitle})" else "Object #${idx + 1} (${obj.regionTitle})",
                                                            style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp),
                                                            color = Color(0xFF94A3B8)
                                                        )
                                                    }
                                                    Spacer(modifier = Modifier.height(2.dp))
                                                    Text(
                                                        text = obj.classLabel,
                                                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                                                        color = color
                                                    )
                                                    Text(
                                                        text = "${String.format(Locale.US, "%.1f", obj.confidence * 100)}% Match",
                                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                                        color = Color(0xFF4ADE80)
                                                    )
                                                }

                                                if (isTaught) {
                                                    Surface(
                                                        color = Color(0xFF10B981).copy(alpha = 0.2f),
                                                        shape = RoundedCornerShape(6.dp),
                                                        border = BorderStroke(1.dp, Color(0xFF10B981))
                                                    ) {
                                                        Text(
                                                            text = "✓ Taught \"$taughtCategory\"",
                                                            style = MaterialTheme.typography.labelSmall.copy(
                                                                fontWeight = FontWeight.Bold,
                                                                color = Color(0xFF34D399)
                                                            ),
                                                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                                        )
                                                    }
                                                } else {
                                                    OutlinedButton(
                                                        onClick = {
                                                            selectedRegionForCorrection = obj
                                                        },
                                                        modifier = Modifier.height(34.dp),
                                                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 2.dp),
                                                        shape = RoundedCornerShape(6.dp),
                                                        border = BorderStroke(1.dp, color)
                                                    ) {
                                                        Icon(
                                                            imageVector = Icons.Default.Edit,
                                                            contentDescription = null,
                                                            tint = color,
                                                            modifier = Modifier.size(13.dp)
                                                        )
                                                        Spacer(modifier = Modifier.width(4.dp))
                                                        Text(
                                                            text = "Teach / ভুল শুধরান",
                                                            style = MaterialTheme.typography.labelSmall.copy(
                                                                fontWeight = FontWeight.Bold,
                                                                fontSize = 11.sp
                                                            ),
                                                            color = Color.White
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // B) PRIMARY PREDICTION BANNER (Clean & Bold)
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.45f)
                    ),
                    shape = RoundedCornerShape(14.dp),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.4f))
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column {
                                Text(
                                    text = if (isFaceMode) {
                                        if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT)
                                            "Identified Individual / ব্যক্তি"
                                        else
                                            "Dominant Recognized Face"
                                    } else {
                                        if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT)
                                            "Primary Detected Category"
                                        else
                                            "Dominant Overall Category"
                                    },
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
                                )
                                Text(
                                    text = res.classLabel,
                                    style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold),
                                    color = MaterialTheme.colorScheme.onPrimaryContainer
                                )
                            }

                            Surface(
                                color = MaterialTheme.colorScheme.primary,
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Text(
                                    text = "${String.format(Locale.US, "%.1f", res.confidence * 100)}%",
                                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                                    color = Color.White,
                                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp)
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(8.dp))
                        HorizontalDivider(color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.2f))
                        Spacer(modifier = Modifier.height(8.dp))

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "Source: ${loadedExportedModel?.formatName ?: "Active DB Model"}",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
                            )
                            Surface(
                                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.6f),
                                shape = RoundedCornerShape(6.dp)
                            ) {
                                Text(
                                    text = "⚡ ${res.inferenceTimeMs} ms on-device",
                                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                    color = MaterialTheme.colorScheme.onSurface,
                                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                )
                            }
                        }
                    }
                }

                // C) SINGLE OBJECT LOCALIZATION & PINPOINT CARD
                if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT && res.detectedObjects.isNotEmpty()) {
                    val pointedObj = res.detectedObjects.first()
                    val isTaught = regionFeedbackSaved.containsKey(pointedObj.regionTitle)
                    val taughtCategory = regionFeedbackSaved[pointedObj.regionTitle]

                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(14.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A)),
                        border = BorderStroke(1.dp, Color(0xFF10B981))
                    ) {
                        Column(modifier = Modifier.padding(14.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        imageVector = Icons.Default.GpsFixed,
                                        contentDescription = null,
                                        tint = Color(0xFF34D399),
                                        modifier = Modifier.size(20.dp)
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = "Object Localized & Pointed",
                                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                                        color = Color.White
                                    )
                                }

                                Surface(
                                    color = Color(0xFF10B981).copy(alpha = 0.2f),
                                    shape = RoundedCornerShape(6.dp)
                                ) {
                                    Text(
                                        text = "🎯 Active Reticle",
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontWeight = FontWeight.Bold,
                                            color = Color(0xFF34D399)
                                        ),
                                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(8.dp))

                            Surface(
                                modifier = Modifier.fillMaxWidth(),
                                color = Color(0xFF1E293B),
                                shape = RoundedCornerShape(8.dp),
                                border = BorderStroke(1.dp, Color(0xFF334155))
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(10.dp),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Column(modifier = Modifier.weight(1f)) {
                                        Text(
                                            text = "Position Coordinates:",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = Color(0xFF94A3B8)
                                        )
                                        Text(
                                            text = "X: [${(pointedObj.boxLeftNorm * 100).toInt()}% - ${(pointedObj.boxRightNorm * 100).toInt()}%]  •  Y: [${(pointedObj.boxTopNorm * 100).toInt()}% - ${(pointedObj.boxBottomNorm * 100).toInt()}%]",
                                            style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Medium),
                                            color = Color.White
                                        )
                                    }

                                    if (isTaught) {
                                        Surface(
                                            color = Color(0xFF10B981).copy(alpha = 0.2f),
                                            shape = RoundedCornerShape(6.dp),
                                            border = BorderStroke(1.dp, Color(0xFF10B981))
                                        ) {
                                            Text(
                                                text = "✓ Taught \"$taughtCategory\"",
                                                style = MaterialTheme.typography.labelSmall.copy(
                                                    fontWeight = FontWeight.Bold,
                                                    color = Color(0xFF34D399)
                                                ),
                                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                            )
                                        }
                                    } else {
                                        OutlinedButton(
                                            onClick = {
                                                selectedRegionForCorrection = pointedObj
                                            },
                                            modifier = Modifier.height(34.dp),
                                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 2.dp),
                                            shape = RoundedCornerShape(6.dp),
                                            border = BorderStroke(1.dp, Color(0xFF10B981))
                                        ) {
                                            Icon(
                                                imageVector = Icons.Default.Edit,
                                                contentDescription = null,
                                                tint = Color(0xFF34D399),
                                                modifier = Modifier.size(13.dp)
                                            )
                                            Spacer(modifier = Modifier.width(4.dp))
                                            Text(
                                                text = "Teach / ভুল শুধরান",
                                                style = MaterialTheme.typography.labelSmall.copy(
                                                    fontWeight = FontWeight.Bold,
                                                    fontSize = 11.sp
                                                ),
                                                color = Color.White
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // D) CONFIDENCE BREAKDOWN CARD (Visible in Single Object Mode)
                if (detectionMode == ObjectDetectionMode.SINGLE_OBJECT) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(14.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)
                    ) {
                        Column(modifier = Modifier.padding(14.dp)) {
                            Text(
                                text = if (isFaceMode) "Facial Identity Biometric Probabilities" else "Class Confidence Breakdown",
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                            )

                            Spacer(modifier = Modifier.height(10.dp))

                            val isNoSubject = res.confidence == 0f || res.allProbabilities.isEmpty() || res.classLabel.startsWith("No ")
                            if (isNoSubject) {
                                Surface(
                                    color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                                    shape = RoundedCornerShape(8.dp),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Row(
                                        modifier = Modifier.padding(12.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Info,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.primary,
                                            modifier = Modifier.size(20.dp)
                                        )
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(
                                            text = if (isFaceMode) "কোনো ব্যক্তির মুখ বা বায়োমেট্রিক ফ্রেম পাওয়া যায়নি (No Face Subject Detected)।"
                                            else "কোনো বিষয়বস্তু বা বস্তু শনাক্ত হয়নি (No Object Detected)।",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
                            } else {
                                Column(
                                    verticalArrangement = Arrangement.spacedBy(8.dp),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    res.allProbabilities.forEach { prob ->
                                        Column(modifier = Modifier.fillMaxWidth()) {
                                            Row(
                                                modifier = Modifier.fillMaxWidth(),
                                                horizontalArrangement = Arrangement.SpaceBetween
                                            ) {
                                                Text(
                                                    text = prob.classLabel,
                                                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                                                )
                                                Text(
                                                    text = "${String.format(Locale.US, "%.1f", prob.probability * 100)}%",
                                                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Bold),
                                                    color = if (prob.classIndex == res.classIndex) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
                                                )
                                            }
                                            Spacer(modifier = Modifier.height(3.dp))
                                            LinearProgressIndicator(
                                                progress = { prob.probability },
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .height(6.dp)
                                                    .clip(RoundedCornerShape(3.dp)),
                                                color = if (prob.classIndex == res.classIndex) MaterialTheme.colorScheme.primary
                                                else MaterialTheme.colorScheme.outlineVariant
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // ==========================================
                // 5. ACTIVE LEARNING & FEEDBACK LOOP (সঠিক ও ভুল অপশন)
                // ==========================================
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("card_active_feedback"),
                    shape = RoundedCornerShape(14.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    imageVector = Icons.Default.FactCheck,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(20.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    text = if (isFaceMode) "Face ID Recognition Quality Feedback" else "Prediction Quality Feedback",
                                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                                )
                            }

                            if (feedbackSubmittedForCurrentPhoto != null) {
                                Surface(
                                    color = Color(0xFF10B981).copy(alpha = 0.15f),
                                    shape = RoundedCornerShape(6.dp)
                                ) {
                                    Text(
                                        text = "✓ Feedback Saved",
                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                        color = Color(0xFF059669),
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                                    )
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            text = if (isFaceMode)
                                "Was this Face ID recognition correct? Your feedback updates the enrolled facial biometric vectors to avoid false positives/negatives."
                            else
                                "Was this classification correct? Your feedback teaches the model to fix mistakes and improve continuous on-device training.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )

                        Spacer(modifier = Modifier.height(12.dp))

                        if (feedbackSubmittedForCurrentPhoto != null) {
                            Surface(
                                modifier = Modifier.fillMaxWidth(),
                                color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.45f),
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Row(
                                    modifier = Modifier.padding(10.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.TaskAlt,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(20.dp)
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = feedbackStatusMessage ?: "Sample added to dataset for continuous learning.",
                                        style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Medium),
                                        color = MaterialTheme.colorScheme.onSecondaryContainer
                                    )
                                }
                            }
                        } else {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                // CORRECT / সঠিক
                                Button(
                                    onClick = {
                                        val targetClass = classes.find { it.className.equals(res.classLabel, ignoreCase = true) }
                                            ?: classes.getOrNull(res.classIndex)
                                            ?: classes.firstOrNull()

                                        if (targetClass != null && testBitmap != null) {
                                            viewModel.submitInferenceFeedback(
                                                classId = targetClass.id,
                                                bitmap = testBitmap!!,
                                                isCorrection = false
                                            ) { msg ->
                                                feedbackSubmittedForCurrentPhoto = "correct"
                                                feedbackStatusMessage = "Confirmed! Added photo to \"${targetClass.className}\" to reinforce accuracy."
                                            }
                                        }
                                    },
                                    modifier = Modifier
                                        .weight(1f)
                                        .height(46.dp)
                                        .testTag("btn_feedback_correct"),
                                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp),
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = Color(0xFF10B981),
                                        contentColor = Color.White
                                    ),
                                    shape = RoundedCornerShape(10.dp)
                                ) {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.Center
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.CheckCircle,
                                            contentDescription = null,
                                            modifier = Modifier.size(18.dp)
                                        )
                                        Spacer(modifier = Modifier.width(6.dp))
                                        Text(
                                            text = "Correct (সঠিক)",
                                            style = MaterialTheme.typography.labelLarge.copy(
                                                fontWeight = FontWeight.Bold,
                                                fontSize = 13.sp
                                            ),
                                            maxLines = 1
                                        )
                                    }
                                }

                                // WRONG / ভুল
                                Button(
                                    onClick = {
                                        showCorrectionDialog = true
                                    },
                                    modifier = Modifier
                                        .weight(1f)
                                        .height(46.dp)
                                        .testTag("btn_feedback_incorrect"),
                                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp),
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = MaterialTheme.colorScheme.error,
                                        contentColor = MaterialTheme.colorScheme.onError
                                    ),
                                    shape = RoundedCornerShape(10.dp)
                                ) {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.Center
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Cancel,
                                            contentDescription = null,
                                            modifier = Modifier.size(18.dp)
                                        )
                                        Spacer(modifier = Modifier.width(6.dp))
                                        Text(
                                            text = "Wrong (ভুল)",
                                            style = MaterialTheme.typography.labelLarge.copy(
                                                fontWeight = FontWeight.Bold,
                                                fontSize = 13.sp
                                            ),
                                            maxLines = 1
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ==========================================
            // 6. CONTINUOUS LEARNING / RE-TRAINING CALLOUT BANNER
            // ==========================================
            if (feedbackSamplesCount > 0) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("card_continuous_retrain_banner"),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.85f)
                    ),
                    shape = RoundedCornerShape(14.dp),
                    border = BorderStroke(1.5.dp, MaterialTheme.colorScheme.tertiary)
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                imageVector = Icons.Default.AutoAwesome,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.onTertiaryContainer,
                                modifier = Modifier.size(24.dp)
                            )
                            Spacer(modifier = Modifier.width(10.dp))
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = "Continuous Learning ($feedbackSamplesCount New Feedback Samples)",
                                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                                    color = MaterialTheme.colorScheme.onTertiaryContainer
                                )
                                Text(
                                    text = "New verified samples saved. Train the model now to improve accuracy.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onTertiaryContainer.copy(alpha = 0.9f)
                                )
                            }
                        }

                        if (isTraining && trainingProgress != null) {
                            val prog = trainingProgress!!
                            val fraction = (prog.overallPercentage / 100f).coerceIn(0f, 1f)
                            Spacer(modifier = Modifier.height(8.dp))
                            LinearProgressIndicator(
                                progress = { fraction },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(6.dp)
                                    .clip(RoundedCornerShape(3.dp)),
                                color = MaterialTheme.colorScheme.tertiary
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "Training: ${prog.phase.title} (${prog.overallPercentage.toInt()}%)",
                                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                color = MaterialTheme.colorScheme.onTertiaryContainer
                            )
                        }

                        Spacer(modifier = Modifier.height(10.dp))

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Button(
                                onClick = {
                                    viewModel.startOnDeviceTraining()
                                },
                                enabled = !isTraining,
                                modifier = Modifier
                                    .weight(1.2f)
                                    .height(42.dp)
                                    .testTag("btn_retrain_now_direct"),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = MaterialTheme.colorScheme.tertiary,
                                    contentColor = MaterialTheme.colorScheme.onTertiary
                                ),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                if (isTraining) {
                                    CircularProgressIndicator(
                                        modifier = Modifier.size(16.dp),
                                        color = MaterialTheme.colorScheme.onTertiary,
                                        strokeWidth = 2.dp
                                    )
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text("Training...", style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold))
                                } else {
                                    Icon(
                                        imageVector = Icons.Default.PlayArrow,
                                        contentDescription = null,
                                        modifier = Modifier.size(18.dp)
                                    )
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text(
                                        text = "Train Now (ট্রেনিং শুরু)",
                                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                                    )
                                }
                            }

                            OutlinedButton(
                                onClick = onNavigateToTrain,
                                modifier = Modifier
                                    .weight(0.8f)
                                    .height(42.dp)
                                    .testTag("btn_open_train_tab"),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Text(
                                    text = "Train Tab",
                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Icon(
                                    imageVector = Icons.AutoMirrored.Filled.ArrowForward,
                                    contentDescription = null,
                                    modifier = Modifier.size(14.dp)
                                )
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))
        }
    }

    // ==========================================
    // 7. CORRECTION MODAL DIALOG (ভুল প্রেডিকশনের সঠিক ক্যাটাগরি নির্বাচন)
    // ==========================================
    if (showCorrectionDialog && testBitmap != null) {
        AlertDialog(
            onDismissRequest = { showCorrectionDialog = false },
            title = {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.EditNote,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Teach Correct Category",
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                    )
                }
            },
            text = {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f), RoundedCornerShape(8.dp))
                            .padding(8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Image(
                            bitmap = testBitmap!!.asImageBitmap(),
                            contentDescription = "Sample thumbnail",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier
                                .size(48.dp)
                                .clip(RoundedCornerShape(6.dp))
                        )
                        Spacer(modifier = Modifier.width(10.dp))
                        Column {
                            Text(
                                text = "Model predicted: \"${inferenceResult?.classLabel ?: "Unknown"}\"",
                                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                                color = MaterialTheme.colorScheme.error
                            )
                            Text(
                                text = "Select the true category below:",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    classes.forEach { cls ->
                        Surface(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp))
                                .clickable {
                                    viewModel.submitInferenceFeedback(
                                        classId = cls.id,
                                        bitmap = testBitmap!!,
                                        isCorrection = true
                                    ) { msg ->
                                        feedbackSubmittedForCurrentPhoto = "corrected"
                                        feedbackStatusMessage = "Corrected! Image added to \"${cls.className}\" for continuous re-training."
                                        showCorrectionDialog = false
                                    }
                                },
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f),
                            shape = RoundedCornerShape(8.dp),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 12.dp, vertical = 10.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Surface(
                                        modifier = Modifier.size(12.dp),
                                        shape = RoundedCornerShape(3.dp),
                                        color = try {
                                            Color(android.graphics.Color.parseColor(cls.colorHex))
                                        } catch (e: Exception) {
                                            MaterialTheme.colorScheme.primary
                                        }
                                    ) {}
                                    Spacer(modifier = Modifier.width(10.dp))
                                    Text(
                                        text = cls.className,
                                        style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                                    )
                                }

                                Icon(
                                    imageVector = Icons.Default.AddCircleOutline,
                                    contentDescription = "Add sample to ${cls.className}",
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(20.dp)
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    OutlinedButton(
                        onClick = {
                            newFeedbackClassName = ""
                            showAddClassInFeedbackDialog = true
                        },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Create New Category", style = MaterialTheme.typography.labelMedium)
                    }
                }
            },
            confirmButton = {},
            dismissButton = {
                TextButton(onClick = { showCorrectionDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }

    // REGION CORRECTION DIALOG (ভুল বস্তু শুধরে শেখানো)
    if (selectedRegionForCorrection != null) {
        val reg = selectedRegionForCorrection!!
        AlertDialog(
            onDismissRequest = { selectedRegionForCorrection = null },
            title = {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Tune,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Teach Region: ${reg.regionTitle}",
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                    )
                }
            },
            text = {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Surface(
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(10.dp)) {
                            Text(
                                text = "Model Detected: \"${reg.classLabel}\" (${String.format(Locale.US, "%.1f%%", reg.confidence * 100)})",
                                style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold),
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Text(
                                text = "Select what this cropped region actually is so the neural network learns this specific object:",
                                style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.8f)
                            )
                        }
                    }

                    Text(
                        text = "Assign to Category:",
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                    )

                    classes.forEach { cls ->
                        Surface(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    val currentBmp = testBitmap
                                    if (currentBmp != null) {
                                        viewModel.submitRegionFeedback(
                                            classId = cls.id,
                                            originalBitmap = currentBmp,
                                            region = reg
                                        ) { msg ->
                                            regionFeedbackSaved[reg.regionTitle] = cls.className
                                            feedbackStatusMessage = "Region \"${reg.regionTitle}\" saved as \"${cls.className}\" for continuous re-training."
                                            selectedRegionForCorrection = null
                                        }
                                    }
                                },
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f),
                            shape = RoundedCornerShape(8.dp),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 12.dp, vertical = 10.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Surface(
                                        modifier = Modifier.size(12.dp),
                                        shape = RoundedCornerShape(3.dp),
                                        color = try {
                                            Color(android.graphics.Color.parseColor(cls.colorHex))
                                        } catch (e: Exception) {
                                            MaterialTheme.colorScheme.primary
                                        }
                                    ) {}
                                    Spacer(modifier = Modifier.width(10.dp))
                                    Text(
                                        text = cls.className,
                                        style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                                    )
                                }

                                Icon(
                                    imageVector = Icons.Default.AddCircleOutline,
                                    contentDescription = "Assign to ${cls.className}",
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(20.dp)
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    OutlinedButton(
                        onClick = {
                            newFeedbackClassName = ""
                            showAddClassInFeedbackDialog = true
                        },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Create New Category", style = MaterialTheme.typography.labelMedium)
                    }
                }
            },
            confirmButton = {},
            dismissButton = {
                TextButton(onClick = { selectedRegionForCorrection = null }) {
                    Text("Cancel")
                }
            }
        )
    }

    if (showAddClassInFeedbackDialog) {
        AlertDialog(
            onDismissRequest = { showAddClassInFeedbackDialog = false },
            title = { Text("Add New Category") },
            text = {
                Column {
                    Text(
                        text = "Enter a name for the new category to categorize this image:",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedTextField(
                        value = newFeedbackClassName,
                        onValueChange = { newFeedbackClassName = it },
                        label = { Text("Category Name") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        if (newFeedbackClassName.isNotBlank() && testBitmap != null) {
                            val colors = listOf("#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6")
                            val color = colors[(classes.size) % colors.size]
                            viewModel.addClass(newFeedbackClassName.trim(), color)
                            feedbackSubmittedForCurrentPhoto = "corrected"
                            feedbackStatusMessage = "Created category \"${newFeedbackClassName.trim()}\" and saved sample for re-training."
                            showAddClassInFeedbackDialog = false
                            showCorrectionDialog = false
                        }
                    },
                    enabled = newFeedbackClassName.isNotBlank()
                ) {
                    Text("Save & Assign")
                }
            },
            dismissButton = {
                TextButton(onClick = { showAddClassInFeedbackDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }

    // Camera Mode Selector Dialog (3 Modes)
    if (showCameraModeDialog) {
        AlertDialog(
            onDismissRequest = { showCameraModeDialog = false },
            icon = {
                Icon(
                    imageVector = Icons.Default.CameraAlt,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(32.dp)
                )
            },
            title = {
                Text(
                    text = "AI Camera Test Modes",
                    style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold)
                )
            },
            text = {
                Column(
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = "Choose your camera detection experience:",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    // Mode 1: Live Single Detection
                    Surface(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .clickable {
                                showCameraModeDialog = false
                                selectedCameraMode = CameraTestMode.LIVE_SINGLE
                                showLiveCameraViewfinder = true
                            },
                        color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.5f),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.35f))
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Surface(
                                shape = CircleShape,
                                color = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(40.dp)
                            ) {
                                Box(contentAlignment = Alignment.Center) {
                                    Icon(
                                        imageVector = Icons.Default.Bolt,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.onPrimary,
                                        modifier = Modifier.size(22.dp)
                                    )
                                }
                            }
                            Spacer(modifier = Modifier.width(12.dp))
                            Column {
                                Text(
                                    text = "⚡ Live Real-Time Single",
                                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                                )
                                Text(
                                    text = "Instant continuous prediction as camera moves • No photo needed",
                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }

                    // Mode 2: Live Multi Detection
                    Surface(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .clickable {
                                showCameraModeDialog = false
                                selectedCameraMode = CameraTestMode.LIVE_MULTI
                                showLiveCameraViewfinder = true
                            },
                        color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.5f),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.secondary.copy(alpha = 0.35f))
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Surface(
                                shape = CircleShape,
                                color = MaterialTheme.colorScheme.secondary,
                                modifier = Modifier.size(40.dp)
                            ) {
                                Box(contentAlignment = Alignment.Center) {
                                    Icon(
                                        imageVector = Icons.Default.FilterCenterFocus,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.onSecondary,
                                        modifier = Modifier.size(22.dp)
                                    )
                                }
                            }
                            Spacer(modifier = Modifier.width(12.dp))
                            Column {
                                Text(
                                    text = "🎯 Live Multi-Detect (Bounding Boxes)",
                                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                                )
                                Text(
                                    text = "Real-time colored bounding boxes over multiple objects & faces",
                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }

                    // Mode 3: Capture & Deep Test
                    Surface(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .clickable {
                                showCameraModeDialog = false
                                selectedCameraMode = CameraTestMode.CAPTURE
                                showLiveCameraViewfinder = true
                            },
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.25f))
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Surface(
                                shape = CircleShape,
                                color = MaterialTheme.colorScheme.surfaceVariant,
                                modifier = Modifier.size(40.dp)
                            ) {
                                Box(contentAlignment = Alignment.Center) {
                                    Icon(
                                        imageVector = Icons.Default.CameraAlt,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                        modifier = Modifier.size(22.dp)
                                    )
                                }
                            }
                            Spacer(modifier = Modifier.width(12.dp))
                            Column {
                                Text(
                                    text = "📸 Capture Photo & Deep Test",
                                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                                )
                                Text(
                                    text = "Snap high-res photo for full saliency & re-training feedback",
                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }

                    // System Camera Fallback
                    OutlinedButton(
                        onClick = {
                            showCameraModeDialog = false
                            launchCameraInternal()
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(imageVector = Icons.Default.Launch, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Use External System Camera App", style = MaterialTheme.typography.labelMedium)
                    }
                }
            },
            confirmButton = {},
            dismissButton = {
                TextButton(onClick = { showCameraModeDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }

    // Live Fullscreen Camera Viewfinder
    if (showLiveCameraViewfinder) {
        LiveCameraViewfinder(
            initialMode = selectedCameraMode,
            onDismiss = { showLiveCameraViewfinder = false },
            onPhotoCaptured = { capturedBmp ->
                testBitmap = capturedBmp
                showLiveCameraViewfinder = false
                selectedHighlightIndex = null
                feedbackSubmittedForCurrentPhoto = null
                feedbackStatusMessage = null
                regionFeedbackSaved.clear()
                viewModel.testImageInference(capturedBmp, detectionMode)
            },
            onAnalyzeFrame = { frameBmp, isMulti ->
                val startTime = System.currentTimeMillis()
                val pred = viewModel.runLiveFrameInference(frameBmp, isMulti)
                val latency = System.currentTimeMillis() - startTime
                val single = if (pred != null) {
                    LiveSinglePrediction(
                        label = pred.classLabel,
                        confidence = pred.confidence,
                        latencyMs = latency
                    )
                } else null
                val multiBoxes = pred?.detectedObjects?.map { obj ->
                    LiveDetectedBox(
                        label = obj.classLabel,
                        confidence = obj.confidence,
                        leftNorm = obj.boxLeftNorm,
                        topNorm = obj.boxTopNorm,
                        rightNorm = obj.boxRightNorm,
                        bottomNorm = obj.boxBottomNorm,
                        facialLandmarks = obj.facialLandmarks,
                        facialMeshEdges = obj.facialMeshEdges,
                        bodyContourPoints = obj.bodyContourPoints,
                        statureDiagnostics = obj.statureDiagnostics,
                        statureRatio = obj.statureRatio
                    )
                } ?: emptyList()
                Pair(single, multiBoxes)
            }
        )
    }

    // Enlarged Fullscreen High-Resolution Image Viewer
    if (showEnlargedPhotoViewer && testBitmap != null) {
        EnlargedPhotoViewerDialog(
            bitmap = testBitmap!!,
            inferenceResult = inferenceResult,
            isFaceMode = isFaceMode,
            detectionMode = detectionMode,
            showBoundingBox = showBoundingBoxOverlay,
            showBodyContour = showBodyContourOverlay,
            showFacialMesh = showFacialMeshOverlay,
            onToggleBoundingBox = { showBoundingBoxOverlay = !showBoundingBoxOverlay },
            onToggleBodyContour = { showBodyContourOverlay = !showBodyContourOverlay },
            onToggleFacialMesh = { showFacialMeshOverlay = !showFacialMeshOverlay },
            onDismiss = { showEnlargedPhotoViewer = false }
        )
    }
    } // End of outer Box
}

@Composable
fun EnginePillTab(
    modifier: Modifier = Modifier,
    title: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    isSelected: Boolean,
    onClick: () -> Unit,
    testTag: String
) {
    Surface(
        modifier = modifier
            .clip(RoundedCornerShape(8.dp))
            .clickable(onClick = onClick)
            .testTag(testTag),
        color = if (isSelected) MaterialTheme.colorScheme.surface else Color.Transparent,
        shape = RoundedCornerShape(8.dp),
        shadowElevation = if (isSelected) 2.dp else 0.dp
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp, horizontal = 4.dp),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(15.dp)
            )
            Spacer(modifier = Modifier.width(4.dp))
            Text(
                text = title,
                style = MaterialTheme.typography.labelMedium.copy(
                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium
                ),
                color = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1
            )
        }
    }
}

@Composable
fun EnlargedPhotoViewerDialog(
    bitmap: Bitmap,
    inferenceResult: com.example.ml.PredictionResult?,
    isFaceMode: Boolean,
    detectionMode: ObjectDetectionMode,
    showBoundingBox: Boolean,
    showBodyContour: Boolean,
    showFacialMesh: Boolean,
    onToggleBoundingBox: () -> Unit,
    onToggleBodyContour: () -> Unit,
    onToggleFacialMesh: () -> Unit,
    onDismiss: () -> Unit
) {
    var scale by remember { mutableFloatStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }

    val objectColors = remember {
        listOf(
            Color(0xFF38BDF8),
            Color(0xFF10B981),
            Color(0xFFF59E0B),
            Color(0xFFA855F7),
            Color(0xFFF43F5E),
            Color(0xFF06B6D4)
        )
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(
            usePlatformDefaultWidth = false,
            dismissOnBackPress = true,
            dismissOnClickOutside = false
        )
    ) {
        Surface(
            modifier = Modifier
                .fillMaxSize()
                .testTag("enlarged_photo_viewer_dialog"),
            color = Color(0xFF020617)
        ) {
            Box(modifier = Modifier.fillMaxSize()) {
                // Interactive Zoomable / Pannable Photo and Biometric Overlays
                BoxWithConstraints(
                    modifier = Modifier
                        .fillMaxSize()
                        .pointerInput(Unit) {
                            detectTransformGestures { _, pan, zoom, _ ->
                                scale = (scale * zoom).coerceIn(1f, 6f)
                                if (scale > 1f) {
                                    offset = Offset(
                                        x = offset.x + pan.x,
                                        y = offset.y + pan.y
                                    )
                                } else {
                                    offset = Offset.Zero
                                }
                            }
                        }
                        .pointerInput(Unit) {
                            detectTapGestures(
                                onDoubleTap = {
                                    if (scale > 1.5f) {
                                        scale = 1f
                                        offset = Offset.Zero
                                    } else {
                                        scale = 2.5f
                                    }
                                }
                            )
                        },
                    contentAlignment = Alignment.Center
                ) {
                    val containerW = maxWidth
                    val containerH = maxHeight

                    val imgW = bitmap.width.toFloat().coerceAtLeast(1f)
                    val imgH = bitmap.height.toFloat().coerceAtLeast(1f)
                    val imgAspect = imgW / imgH
                    val containerAspect = (containerW / containerH).coerceAtLeast(0.01f)

                    val (renderedW, renderedH) = if (imgAspect > containerAspect) {
                        containerW to (containerW / imgAspect)
                    } else {
                        (containerH * imgAspect) to containerH
                    }

                    Box(
                        modifier = Modifier
                            .size(width = renderedW, height = renderedH)
                            .graphicsLayer {
                                scaleX = scale
                                scaleY = scale
                                translationX = offset.x
                                translationY = offset.y
                            }
                    ) {
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = "Enlarged Test Image",
                            contentScale = ContentScale.Fit,
                            modifier = Modifier.fillMaxSize()
                        )

                        if (inferenceResult != null) {
                            val detected = inferenceResult.detectedObjects

                            // 1. Biometric & Structural Contour Layer (Face ID Mode Only)
                            if (isFaceMode && (showBodyContour || showFacialMesh)) {
                                Canvas(modifier = Modifier.fillMaxSize()) {
                                    val wPx = size.width
                                    val hPx = size.height

                                    for ((_, obj) in detected.withIndex()) {
                                        // A. Body Contour
                                        if (showBodyContour && obj.bodyContourPoints.isNotEmpty()) {
                                            val contourPath = androidx.compose.ui.graphics.Path()
                                            obj.bodyContourPoints.forEachIndexed { cIdx, pt ->
                                                val px = pt.x * wPx
                                                val py = pt.y * hPx
                                                if (cIdx == 0) contourPath.moveTo(px, py) else contourPath.lineTo(px, py)
                                            }
                                            contourPath.close()

                                            drawPath(
                                                path = contourPath,
                                                color = Color(0xFFFF6D00).copy(alpha = 0.35f),
                                                style = androidx.compose.ui.graphics.drawscope.Stroke(
                                                    width = 6.dp.toPx(),
                                                    cap = androidx.compose.ui.graphics.StrokeCap.Round,
                                                    join = androidx.compose.ui.graphics.StrokeJoin.Round
                                                )
                                            )
                                            drawPath(
                                                path = contourPath,
                                                color = Color(0xFFFF8800),
                                                style = androidx.compose.ui.graphics.drawscope.Stroke(
                                                    width = 2.5.dp.toPx(),
                                                    cap = androidx.compose.ui.graphics.StrokeCap.Round,
                                                    join = androidx.compose.ui.graphics.StrokeJoin.Round
                                                )
                                            )
                                        }

                                        // B. Facial Mesh
                                        if (showFacialMesh && obj.facialLandmarks.isNotEmpty()) {
                                            val landmarkPx = obj.facialLandmarks.map { pt ->
                                                Offset(pt.x * wPx, pt.y * hPx)
                                            }

                                            for (edge in obj.facialMeshEdges) {
                                                if (edge.first < landmarkPx.size && edge.second < landmarkPx.size) {
                                                    val p1 = landmarkPx[edge.first]
                                                    val p2 = landmarkPx[edge.second]
                                                    drawLine(
                                                        color = Color(0xFF06B6D4).copy(alpha = 0.50f),
                                                        start = p1,
                                                        end = p2,
                                                        strokeWidth = 2.8.dp.toPx()
                                                    )
                                                    drawLine(
                                                        color = Color(0xFFE0F2FE),
                                                        start = p1,
                                                        end = p2,
                                                        strokeWidth = 1.2.dp.toPx()
                                                    )
                                                }
                                            }

                                            for (pt in landmarkPx) {
                                                drawCircle(
                                                    color = Color(0xFF0284C7),
                                                    radius = 2.2.dp.toPx(),
                                                    center = pt
                                                )
                                                drawCircle(
                                                    color = Color.White,
                                                    radius = 1.1.dp.toPx(),
                                                    center = pt
                                                )
                                            }
                                        }
                                    }
                                }
                            }

                            // 2. High-Definition Target Bounding Box Overlays
                            if (showBoundingBox && detected.isNotEmpty()) {
                                detected.forEachIndexed { idx, obj ->
                                    val activeColor = objectColors[idx % objectColors.size]
                                    val left = renderedW * obj.boxLeftNorm.coerceIn(0f, 1f)
                                    val top = renderedH * obj.boxTopNorm.coerceIn(0f, 1f)
                                    val width = (renderedW * (obj.boxRightNorm - obj.boxLeftNorm)).coerceAtLeast(36.dp)
                                    val height = (renderedH * (obj.boxBottomNorm - obj.boxTopNorm)).coerceAtLeast(36.dp)
                                    val isNearTop = top < 22.dp

                                    Box(
                                        modifier = Modifier
                                            .absoluteOffset(x = left, y = top)
                                            .size(width = width, height = height)
                                    ) {
                                        Canvas(modifier = Modifier.fillMaxSize()) {
                                            val strokeW = 2.5.dp.toPx()
                                            val bracketLen = (8.dp.toPx()).coerceAtMost(size.minDimension / 4f)

                                            drawRect(
                                                color = activeColor.copy(alpha = 0.15f),
                                                size = size
                                            )

                                            // 4 Corner brackets
                                            drawLine(color = activeColor, start = Offset(0f, 0f), end = Offset(bracketLen, 0f), strokeWidth = strokeW)
                                            drawLine(color = activeColor, start = Offset(0f, 0f), end = Offset(0f, bracketLen), strokeWidth = strokeW)
                                            drawLine(color = activeColor, start = Offset(size.width, 0f), end = Offset(size.width - bracketLen, 0f), strokeWidth = strokeW)
                                            drawLine(color = activeColor, start = Offset(size.width, 0f), end = Offset(size.width, bracketLen), strokeWidth = strokeW)
                                            drawLine(color = activeColor, start = Offset(0f, size.height), end = Offset(bracketLen, size.height), strokeWidth = strokeW)
                                            drawLine(color = activeColor, start = Offset(0f, size.height), end = Offset(0f, size.height - bracketLen), strokeWidth = strokeW)
                                            drawLine(color = activeColor, start = Offset(size.width, size.height), end = Offset(size.width - bracketLen, size.height), strokeWidth = strokeW)
                                            drawLine(color = activeColor, start = Offset(size.width, size.height), end = Offset(size.width, size.height - bracketLen), strokeWidth = strokeW)

                                            // Center crosshair & pinpoint
                                            val cx = size.width / 2f
                                            val cy = size.height / 2f
                                            val ringRad = 7.dp.toPx().coerceAtMost(size.minDimension / 5f)
                                            val tickLen = 4.dp.toPx()

                                            drawCircle(
                                                color = activeColor.copy(alpha = 0.9f),
                                                radius = ringRad,
                                                center = Offset(cx, cy),
                                                style = androidx.compose.ui.graphics.drawscope.Stroke(width = 1.5.dp.toPx())
                                            )
                                            drawLine(color = activeColor, start = Offset(cx - ringRad - tickLen, cy), end = Offset(cx - ringRad + 1f, cy), strokeWidth = 1.5.dp.toPx())
                                            drawLine(color = activeColor, start = Offset(cx + ringRad - 1f, cy), end = Offset(cx + ringRad + tickLen, cy), strokeWidth = 1.5.dp.toPx())
                                            drawLine(color = activeColor, start = Offset(cx, cy - ringRad - tickLen), end = Offset(cx, cy - ringRad + 1f), strokeWidth = 1.5.dp.toPx())
                                            drawLine(color = activeColor, start = Offset(cx, cy + ringRad - 1f), end = Offset(cx, cy + ringRad + tickLen), strokeWidth = 1.5.dp.toPx())
                                            drawCircle(color = Color.White, radius = 2.5.dp.toPx(), center = Offset(cx, cy))
                                            drawCircle(color = activeColor, radius = 1.2.dp.toPx(), center = Offset(cx, cy))
                                        }

                                        // Label Badge
                                        Surface(
                                            modifier = Modifier
                                                .align(if (isNearTop) Alignment.BottomStart else Alignment.TopStart)
                                                .padding(3.dp),
                                            color = activeColor,
                                            shape = RoundedCornerShape(4.dp),
                                            shadowElevation = 4.dp
                                        ) {
                                            Text(
                                                text = "${obj.classLabel} (${String.format(Locale.US, "%.0f%%", obj.confidence * 100)})",
                                                style = MaterialTheme.typography.labelSmall.copy(
                                                    fontSize = 11.sp,
                                                    fontWeight = FontWeight.Bold
                                                ),
                                                color = Color.White,
                                                maxLines = 1,
                                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Top Control Header Bar
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .align(Alignment.TopCenter),
                    color = Color.Black.copy(alpha = 0.85f),
                    border = BorderStroke(0.dp, Color.Transparent)
                ) {
                    Column(modifier = Modifier.statusBarsPadding().padding(horizontal = 12.dp, vertical = 8.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                IconButton(
                                    onClick = onDismiss,
                                    modifier = Modifier.testTag("enlarged_photo_close_btn")
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Close,
                                        contentDescription = "Close",
                                        tint = Color.White
                                    )
                                }
                                Spacer(modifier = Modifier.width(4.dp))
                                Column {
                                    Text(
                                        text = if (isFaceMode) "Face Biometric Inspector" else "Image Inspection & Zoom",
                                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                                        color = Color.White
                                    )
                                    Text(
                                        text = "${bitmap.width} × ${bitmap.height} px • Zoom ${String.format(Locale.US, "%.1fx", scale)}",
                                        style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                        color = Color(0xFF94A3B8)
                                    )
                                }
                            }

                            // Quick Zoom Buttons (+ / - / Reset)
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(4.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                IconButton(
                                    onClick = {
                                        scale = (scale - 0.5f).coerceAtLeast(1f)
                                        if (scale == 1f) offset = Offset.Zero
                                    },
                                    modifier = Modifier.size(34.dp)
                                ) {
                                    Icon(Icons.Default.Remove, contentDescription = "Zoom Out", tint = Color.White, modifier = Modifier.size(18.dp))
                                }

                                Surface(
                                    onClick = {
                                        scale = if (scale > 1.2f) 1f else 2.5f
                                        if (scale == 1f) offset = Offset.Zero
                                    },
                                    color = Color(0xFF1E293B),
                                    shape = RoundedCornerShape(6.dp),
                                    border = BorderStroke(1.dp, Color(0xFF475569))
                                ) {
                                    Text(
                                        text = if (scale > 1.2f) "Reset" else "2x",
                                        fontSize = 11.sp,
                                        fontWeight = FontWeight.Bold,
                                        color = Color.White,
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                    )
                                }

                                IconButton(
                                    onClick = { scale = (scale + 0.5f).coerceAtMost(6f) },
                                    modifier = Modifier.size(34.dp)
                                ) {
                                    Icon(Icons.Default.Add, contentDescription = "Zoom In", tint = Color.White, modifier = Modifier.size(18.dp))
                                }
                            }
                        }

                        // Layer Toggles Row
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 4.dp),
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            FilterChip(
                                selected = showBoundingBox,
                                onClick = onToggleBoundingBox,
                                label = { Text(if (isFaceMode) "▣ Face / Body" else "▣ Detection Box", fontSize = 11.sp) },
                                modifier = Modifier.height(28.dp)
                            )
                            if (isFaceMode) {
                                FilterChip(
                                    selected = showBodyContour,
                                    onClick = onToggleBodyContour,
                                    label = { Text("📐 Contour", fontSize = 11.sp) },
                                    modifier = Modifier.height(28.dp)
                                )
                                FilterChip(
                                    selected = showFacialMesh,
                                    onClick = onToggleFacialMesh,
                                    label = { Text("🕸️ Mesh", fontSize = 11.sp) },
                                    modifier = Modifier.height(28.dp)
                                )
                            }
                        }
                    }
                }

                // Bottom Floating Biometric Summary Pill
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .align(Alignment.BottomCenter)
                        .navigationBarsPadding()
                        .padding(12.dp),
                    color = Color(0xFF0F172A).copy(alpha = 0.92f),
                    shape = RoundedCornerShape(14.dp),
                    border = BorderStroke(1.dp, Color(0xFF38BDF8).copy(alpha = 0.4f))
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Box(
                                    modifier = Modifier
                                        .size(10.dp)
                                        .clip(CircleShape)
                                        .background(
                                            if (inferenceResult != null && inferenceResult.confidence > 0f) Color(0xFF22C55E)
                                            else Color(0xFF94A3B8)
                                        )
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    text = inferenceResult?.classLabel ?: "No Prediction",
                                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                                    color = Color.White
                                )
                            }

                            if (inferenceResult != null && inferenceResult.confidence > 0f) {
                                Surface(
                                    color = Color(0xFF38BDF8).copy(alpha = 0.2f),
                                    shape = RoundedCornerShape(6.dp)
                                ) {
                                    Text(
                                        text = "${String.format(Locale.US, "%.1f%%", inferenceResult.confidence * 100)} (${inferenceResult.inferenceTimeMs}ms)",
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontWeight = FontWeight.Bold,
                                            color = Color(0xFF38BDF8)
                                        ),
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                                    )
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "💡 দুই আঙুল দিয়ে টেনে বড় (Pinch to zoom) এবং সরিয়ে দেখতে পারেন। Double-tap করে রিসেট করুন।",
                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                            color = Color(0xFF94A3B8)
                        )
                    }
                }
            }
        }
    }
}
