package com.example.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.example.util.AppLogger
import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.data.db.ClassificationClassEntity
import com.example.data.db.ImageSampleEntity
import com.example.ui.components.FaceEnrollmentDialog
import com.example.ui.viewmodel.ProjectViewModel
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ClassManagementScreen(
    viewModel: ProjectViewModel,
    onNavigateToTrain: () -> Unit,
    onNavigateBack: (() -> Unit)? = null
) {
    val project by viewModel.currentProject.collectAsState()
    val classes by viewModel.projectClasses.collectAsState()
    val context = LocalContext.current

    if (project?.projectType == "TEXT_CLASSIFICATION") {
        com.example.ui.components.TextDatasetStudio(
            viewModel = viewModel,
            project = project!!,
            classes = classes,
            onNavigateToTrain = onNavigateToTrain,
            onNavigateBack = onNavigateBack
        )
        return
    }

    var selectedClassId by remember(classes) {
        mutableStateOf(classes.firstOrNull()?.id)
    }

    var showAddClassDialog by remember { mutableStateOf(false) }
    var classToEdit by remember { mutableStateOf<ClassificationClassEntity?>(null) }
    var isImportingZip by remember { mutableStateOf(false) }

    // Interactive Data Labeling & Batch Management States
    var isSelectMode by remember { mutableStateOf(false) }
    var selectedSampleIds by remember { mutableStateOf(setOf<Long>()) }
    var previewSample by remember { mutableStateOf<ImageSampleEntity?>(null) }

    // Clear selection when switching classes
    LaunchedEffect(selectedClassId) {
        isSelectMode = false
        selectedSampleIds = emptySet()
    }

    // Pick photos from Gallery
    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickMultipleVisualMedia()
    ) { uris ->
        selectedClassId?.let { classId ->
            uris.forEach { uri ->
                viewModel.addSampleFromUri(classId, uri)
            }
        }
    }

    // Pick photos from File Manager / Documents Section
    val documentPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenMultipleDocuments()
    ) { uris ->
        selectedClassId?.let { classId ->
            uris.forEach { uri ->
                viewModel.addSampleFromUri(classId, uri)
            }
        }
    }

    // Temporary URI for full-resolution camera captures
    var currentCameraUri by remember { mutableStateOf<Uri?>(null) }

    val takePictureLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        if (success && currentCameraUri != null && selectedClassId != null) {
            viewModel.addSampleFromUri(selectedClassId!!, currentCameraUri!!)
            Toast.makeText(context, "Photo added to dataset", Toast.LENGTH_SHORT).show()
        }
    }

    val cameraPreviewLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        if (bitmap != null && selectedClassId != null) {
            viewModel.addSample(selectedClassId!!, bitmap)
            Toast.makeText(context, "Photo added to dataset", Toast.LENGTH_SHORT).show()
        }
    }

    fun launchCameraInternal() {
        try {
            val photoFile = File(context.cacheDir, "camera_sample_${System.currentTimeMillis()}.jpg")
            val uri = FileProvider.getUriForFile(
                context,
                "${context.packageName}.fileprovider",
                photoFile
            )
            currentCameraUri = uri
            takePictureLauncher.launch(uri)
        } catch (e: Exception) {
            AppLogger.e("ClassManagementScreen", "FileProvider capture failed, trying preview: ${e.message}", e)
            try {
                cameraPreviewLauncher.launch(null)
            } catch (ex: Exception) {
                AppLogger.e("ClassManagementScreen", "Camera launch failed: ${ex.message}", ex)
                Toast.makeText(context, "Unable to launch camera app: ${ex.localizedMessage ?: "Device camera not accessible"}", Toast.LENGTH_LONG).show()
            }
        }
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            launchCameraInternal()
        } else {
            Toast.makeText(
                context,
                "Camera permission is required to capture photos",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    fun handleCameraClick() {
        if (selectedClassId == null) {
            Toast.makeText(context, "Please select or create a class first", Toast.LENGTH_SHORT).show()
            return
        }
        val permissionCheck = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.CAMERA
        )
        if (permissionCheck == PackageManager.PERMISSION_GRANTED) {
            launchCameraInternal()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    // ZIP Dataset Import Launcher
    val zipPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            isImportingZip = true
            viewModel.importDatasetFromZip(uri) { message ->
                isImportingZip = false
                Toast.makeText(context, message, Toast.LENGTH_LONG).show()
            }
        }
    }

    val selectedClass = classes.find { it.id == selectedClassId }
    val isFaceRecognition = project?.projectType == "FACE_RECOGNITION"
    var showFaceEnrollmentDialog by remember { mutableStateOf(false) }

    BackHandler {
        if (isSelectMode) {
            isSelectMode = false
            selectedSampleIds = emptySet()
        } else {
            onNavigateBack?.invoke()
        }
    }

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        topBar = {
            TopAppBar(
                navigationIcon = {
                    if (onNavigateBack != null) {
                        IconButton(
                            onClick = onNavigateBack,
                            modifier = Modifier.testTag("dataset_back_btn")
                        ) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                contentDescription = "Back to Project List"
                            )
                        }
                    }
                },
                title = {
                    Column {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                text = project?.name ?: (if (isFaceRecognition) "Person & Human ID Enrollment" else "Dataset Collection"),
                                style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold)
                            )
                            if (isFaceRecognition) {
                                Spacer(modifier = Modifier.width(8.dp))
                                Surface(
                                    color = Color(0xFF10B981).copy(alpha = 0.15f),
                                    shape = RoundedCornerShape(6.dp)
                                ) {
                                    Text(
                                        text = "HYBRID RE-ID",
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontSize = 9.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = Color(0xFF10B981)
                                        ),
                                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                    )
                                }
                            }
                        }
                        Text(
                            text = if (isFaceRecognition)
                                "Enroll individuals with face, full body, or mixed photos (Auto-Balanced)"
                            else
                                "Add categories, rename classes, or import ZIP dataset",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                actions = {
                    OutlinedButton(
                        onClick = {
                            zipPickerLauncher.launch(arrayOf("application/zip", "application/x-zip-compressed", "*/*"))
                        },
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier
                            .padding(end = 8.dp)
                            .testTag("import_zip_btn")
                    ) {
                        Icon(Icons.Default.FolderZip, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("ZIP Dataset")
                    }

                    if (classes.size >= 2) {
                        Button(
                            onClick = onNavigateToTrain,
                            shape = RoundedCornerShape(12.dp),
                            modifier = Modifier
                                .padding(end = 8.dp)
                                .testTag("go_to_train_btn")
                        ) {
                            Text(if (isFaceRecognition) "Train Human ID" else "Train Model")
                            Spacer(modifier = Modifier.width(4.dp))
                            Icon(Icons.Default.ArrowForward, contentDescription = null, modifier = Modifier.size(16.dp))
                        }
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
        ) {
            if (isImportingZip) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }

            // Class Tabs Selector Bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                LazyRow(
                    modifier = Modifier.weight(1f),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(classes, key = { it.id }) { item ->
                        val isSelected = item.id == selectedClassId
                        val count by viewModel.getSampleCountForClass(item.id).collectAsState(initial = 0)

                        FilterChip(
                            selected = isSelected,
                            onClick = { selectedClassId = item.id },
                            label = {
                                Text("${item.className} ($count)")
                            },
                            leadingIcon = {
                                if (isFaceRecognition) {
                                    Icon(
                                        Icons.Default.Face,
                                        contentDescription = null,
                                        modifier = Modifier.size(14.dp),
                                        tint = try {
                                            Color(android.graphics.Color.parseColor(item.colorHex))
                                        } catch (e: Exception) {
                                            MaterialTheme.colorScheme.primary
                                        }
                                    )
                                } else {
                                    Box(
                                        modifier = Modifier
                                            .size(12.dp)
                                            .clip(CircleShape)
                                            .background(
                                                try {
                                                    Color(android.graphics.Color.parseColor(item.colorHex))
                                                } catch (e: Exception) {
                                                    MaterialTheme.colorScheme.primary
                                                }
                                            )
                                    )
                                }
                            },
                            modifier = Modifier.testTag("class_chip_${item.id}")
                        )
                    }
                }

                IconButton(
                    onClick = { showAddClassDialog = true },
                    modifier = Modifier
                        .padding(start = 8.dp)
                        .testTag("add_class_chip_btn")
                ) {
                    Icon(
                        imageVector = Icons.Default.AddCircle,
                        contentDescription = if (isFaceRecognition) "Add Person" else "Add Class",
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(32.dp)
                    )
                }
            }

            Divider()

            if (selectedClass != null) {
                val samples by viewModel.getSamplesForClass(selectedClass.id).collectAsState(initial = emptyList())

                var currentPage by remember(selectedClass.id) { mutableIntStateOf(0) }
                var pageSize by remember { mutableIntStateOf(24) } // 24, 48, 96, 0 (All)

                val totalCount = samples.size
                val effectivePageSize = if (pageSize <= 0) totalCount else pageSize
                val totalPages = if (effectivePageSize > 0) ((totalCount + effectivePageSize - 1) / effectivePageSize).coerceAtLeast(1) else 1
                val safeCurrentPage = currentPage.coerceIn(0, totalPages - 1)

                val paginatedSamples = remember(samples, safeCurrentPage, pageSize) {
                    if (pageSize > 0 && totalCount > pageSize) {
                        val start = safeCurrentPage * pageSize
                        val end = minOf(start + pageSize, totalCount)
                        samples.subList(start, end)
                    } else {
                        samples
                    }
                }

                val startItemIndex = if (totalCount == 0) 0 else (safeCurrentPage * effectivePageSize + 1)
                val endItemIndex = minOf((safeCurrentPage * effectivePageSize) + paginatedSamples.size, totalCount)

                if (samples.isEmpty()) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .verticalScroll(rememberScrollState())
                            .padding(16.dp)
                    ) {
                        ClassProgressHeader(
                            classificationClass = selectedClass,
                            sampleCount = 0,
                            isFaceRecognition = isFaceRecognition,
                            onEditClass = { classToEdit = selectedClass },
                            onAddGallery = {
                                galleryLauncher.launch(
                                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                )
                            },
                            onAddFiles = {
                                documentPickerLauncher.launch(arrayOf("image/*"))
                            },
                            onAddCamera = {
                                handleCameraClick()
                            },
                            onEnrollFace = {
                                showFaceEnrollmentDialog = true
                            },
                            onDeleteClass = {
                                viewModel.deleteClass(selectedClass.id)
                                selectedClassId = classes.firstOrNull { it.id != selectedClass.id }?.id
                            },
                            initiallyExpanded = true
                        )

                        Spacer(modifier = Modifier.height(16.dp))

                        EmptySamplesView(
                            className = selectedClass.className,
                            isFaceRecognition = isFaceRecognition,
                            onPickGallery = {
                                galleryLauncher.launch(
                                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                )
                            },
                            onPickFiles = {
                                documentPickerLauncher.launch(arrayOf("image/*"))
                            },
                            onTakePhoto = {
                                handleCameraClick()
                            },
                            onEnrollFace = {
                                showFaceEnrollmentDialog = true
                            }
                        )
                    }
                } else {
                    LazyVerticalGrid(
                        columns = GridCells.Adaptive(minSize = 100.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 10.dp)
                    ) {
                        // 1. Sleek, scrollable/collapsible header (doesn't block the screen)
                        item(span = { GridItemSpan(maxLineSpan) }) {
                            ClassProgressHeader(
                                classificationClass = selectedClass,
                                sampleCount = samples.size,
                                isFaceRecognition = isFaceRecognition,
                                onEditClass = { classToEdit = selectedClass },
                                onAddGallery = {
                                    galleryLauncher.launch(
                                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                    )
                                },
                                onAddFiles = {
                                    documentPickerLauncher.launch(arrayOf("image/*"))
                                },
                                onAddCamera = {
                                    handleCameraClick()
                                },
                                onEnrollFace = {
                                    showFaceEnrollmentDialog = true
                                },
                                onDeleteClass = {
                                    viewModel.deleteClass(selectedClass.id)
                                    selectedClassId = classes.firstOrNull { it.id != selectedClass.id }?.id
                                },
                                initiallyExpanded = false
                            )
                        }

                        // 2. Data Labeling & Batch Management Toolbar
                        item(span = { GridItemSpan(maxLineSpan) }) {
                            Surface(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 4.dp),
                                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
                                shape = RoundedCornerShape(12.dp)
                            ) {
                                Column(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(horizontal = 12.dp, vertical = 8.dp)
                                ) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        if (isSelectMode) {
                                            val isCurrentPageAllSelected = paginatedSamples.isNotEmpty() && paginatedSamples.all { selectedSampleIds.contains(it.id) }
                                            Row(verticalAlignment = Alignment.CenterVertically) {
                                                Checkbox(
                                                    checked = isCurrentPageAllSelected,
                                                    onCheckedChange = { checked ->
                                                        val pageIds = paginatedSamples.map { it.id }.toSet()
                                                        selectedSampleIds = if (checked) {
                                                            selectedSampleIds + pageIds
                                                        } else {
                                                            selectedSampleIds - pageIds
                                                        }
                                                    }
                                                )
                                                Text(
                                                    text = "${selectedSampleIds.size} selected",
                                                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Bold)
                                                )
                                            }

                                            Row(
                                                horizontalArrangement = Arrangement.spacedBy(6.dp),
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                if (totalCount > paginatedSamples.size) {
                                                    TextButton(
                                                        onClick = {
                                                            selectedSampleIds = if (selectedSampleIds.size == totalCount) emptySet() else samples.map { it.id }.toSet()
                                                        },
                                                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp)
                                                    ) {
                                                        Text(
                                                            if (selectedSampleIds.size == totalCount) "Clear All" else "Select All ($totalCount)",
                                                            fontSize = 11.sp
                                                        )
                                                    }
                                                }

                                                if (selectedSampleIds.isNotEmpty()) {
                                                    OutlinedButton(
                                                        onClick = {
                                                            viewModel.deleteSamplesBatch(selectedSampleIds.toList())
                                                            selectedSampleIds = emptySet()
                                                            isSelectMode = false
                                                            Toast.makeText(context, "Deleted selected samples", Toast.LENGTH_SHORT).show()
                                                        },
                                                        colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error),
                                                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp)
                                                    ) {
                                                        Icon(Icons.Default.Delete, contentDescription = null, modifier = Modifier.size(15.dp))
                                                        Spacer(modifier = Modifier.width(4.dp))
                                                        Text("Delete (${selectedSampleIds.size})", fontSize = 12.sp)
                                                    }
                                                }

                                                TextButton(
                                                    onClick = {
                                                        isSelectMode = false
                                                        selectedSampleIds = emptySet()
                                                    },
                                                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp)
                                                ) {
                                                    Text("Cancel", fontSize = 12.sp)
                                                }
                                            }
                                        } else {
                                            Row(verticalAlignment = Alignment.CenterVertically) {
                                                Icon(
                                                    Icons.Default.PhotoLibrary,
                                                    contentDescription = null,
                                                    tint = MaterialTheme.colorScheme.primary,
                                                    modifier = Modifier.size(18.dp)
                                                )
                                                Spacer(modifier = Modifier.width(6.dp))
                                                Text(
                                                    text = "Dataset ($totalCount items)",
                                                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                                                )
                                            }

                                            OutlinedButton(
                                                onClick = { isSelectMode = true },
                                                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                                            ) {
                                                Icon(Icons.Default.Checklist, contentDescription = null, modifier = Modifier.size(16.dp))
                                                Spacer(modifier = Modifier.width(4.dp))
                                                Text("Select Batch", fontSize = 12.sp)
                                            }
                                        }
                                    }

                                    // Pagination Info & Page Size Options
                                    if (totalCount > 12) {
                                        Spacer(modifier = Modifier.height(6.dp))
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Text(
                                                text = if (pageSize > 0 && totalCount > pageSize) {
                                                    "Showing $startItemIndex–$endItemIndex of $totalCount items"
                                                } else {
                                                    "Showing all $totalCount items"
                                                },
                                                style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                                color = MaterialTheme.colorScheme.onSurfaceVariant
                                            )

                                            Row(
                                                horizontalArrangement = Arrangement.spacedBy(4.dp),
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                Text(
                                                    text = "Per page:",
                                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                                listOf(24, 48, 96, 0).forEach { sizeOption ->
                                                    val isSelected = pageSize == sizeOption
                                                    Surface(
                                                        onClick = {
                                                            pageSize = sizeOption
                                                            currentPage = 0
                                                        },
                                                        shape = RoundedCornerShape(6.dp),
                                                        color = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surface,
                                                        border = BorderStroke(
                                                            1.dp,
                                                            if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant
                                                        )
                                                    ) {
                                                        Text(
                                                            text = if (sizeOption == 0) "All" else "$sizeOption",
                                                            fontSize = 11.sp,
                                                            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                                            color = if (isSelected) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurface,
                                                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // 3. Grid of Images (paginated and fast)
                        items(paginatedSamples, key = { it.id }) { sample ->
                            ImageSampleGridTile(
                                sample = sample,
                                isSelectMode = isSelectMode,
                                isSelected = selectedSampleIds.contains(sample.id),
                                onToggleSelect = {
                                    selectedSampleIds = if (selectedSampleIds.contains(sample.id)) {
                                        selectedSampleIds - sample.id
                                    } else {
                                        selectedSampleIds + sample.id
                                    }
                                },
                                onClickPreview = { previewSample = sample },
                                onDelete = { viewModel.deleteSample(sample.id) }
                            )
                        }

                        // 4. Bottom Pagination Controls (when dataset is split into multiple pages)
                        if (totalPages > 1) {
                            item(span = { GridItemSpan(maxLineSpan) }) {
                                Surface(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(top = 4.dp, bottom = 12.dp),
                                    color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                                    shape = RoundedCornerShape(12.dp)
                                ) {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(horizontal = 8.dp, vertical = 4.dp),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Row(
                                            horizontalArrangement = Arrangement.spacedBy(4.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            IconButton(
                                                onClick = { currentPage = 0 },
                                                enabled = safeCurrentPage > 0,
                                                modifier = Modifier.size(32.dp)
                                            ) {
                                                Icon(
                                                    Icons.Default.FirstPage,
                                                    contentDescription = "First Page",
                                                    modifier = Modifier.size(18.dp)
                                                )
                                            }
                                            OutlinedButton(
                                                onClick = { currentPage = (safeCurrentPage - 1).coerceAtLeast(0) },
                                                enabled = safeCurrentPage > 0,
                                                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                                                modifier = Modifier.height(32.dp)
                                            ) {
                                                Icon(Icons.Default.ChevronLeft, contentDescription = null, modifier = Modifier.size(16.dp))
                                                Spacer(modifier = Modifier.width(2.dp))
                                                Text("Prev", fontSize = 11.sp)
                                            }
                                        }

                                        Surface(
                                            color = MaterialTheme.colorScheme.primaryContainer,
                                            shape = RoundedCornerShape(8.dp)
                                        ) {
                                            Text(
                                                text = "Page ${safeCurrentPage + 1} of $totalPages",
                                                style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Bold),
                                                color = MaterialTheme.colorScheme.onPrimaryContainer,
                                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp)
                                            )
                                        }

                                        Row(
                                            horizontalArrangement = Arrangement.spacedBy(4.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            OutlinedButton(
                                                onClick = { currentPage = (safeCurrentPage + 1).coerceAtMost(totalPages - 1) },
                                                enabled = safeCurrentPage < totalPages - 1,
                                                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                                                modifier = Modifier.height(32.dp)
                                            ) {
                                                Text("Next", fontSize = 11.sp)
                                                Spacer(modifier = Modifier.width(2.dp))
                                                Icon(Icons.Default.ChevronRight, contentDescription = null, modifier = Modifier.size(16.dp))
                                            }
                                            IconButton(
                                                onClick = { currentPage = totalPages - 1 },
                                                enabled = safeCurrentPage < totalPages - 1,
                                                modifier = Modifier.size(32.dp)
                                            ) {
                                                Icon(
                                                    Icons.Default.LastPage,
                                                    contentDescription = "Last Page",
                                                    modifier = Modifier.size(18.dp)
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(if (isFaceRecognition) "No Individuals Enrolled Yet" else "No Classes Created Yet")
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            Button(onClick = { showAddClassDialog = true }) {
                                Text(if (isFaceRecognition) "Enroll Person" else "Create Class Category")
                            }
                            OutlinedButton(onClick = {
                                zipPickerLauncher.launch(arrayOf("application/zip", "application/x-zip-compressed", "*/*"))
                            }) {
                                Icon(Icons.Default.FolderZip, contentDescription = null)
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Import ZIP Dataset")
                            }
                        }
                    }
                }
            }
        }
    }

    if (showAddClassDialog) {
        AddClassDialog(
            isFaceRecognition = isFaceRecognition,
            onDismiss = { showAddClassDialog = false },
            onAdd = { className, colorHex ->
                viewModel.addClass(className, colorHex)
                showAddClassDialog = false
            }
        )
    }

    if (classToEdit != null) {
        EditClassDialog(
            classificationClass = classToEdit!!,
            isFaceRecognition = isFaceRecognition,
            onDismiss = { classToEdit = null },
            onSave = { newName, newColor ->
                viewModel.updateClass(classToEdit!!.id, newName, newColor)
                classToEdit = null
            }
        )
    }

    if (showFaceEnrollmentDialog && selectedClass != null) {
        FaceEnrollmentDialog(
            personName = selectedClass.className,
            onPhotosEnrolled = { photos ->
                photos.forEach { bmp ->
                    viewModel.addSample(selectedClass.id, bmp)
                }
                Toast.makeText(context, "${photos.size}টি বায়োমেট্রিক ফেস অ্যাঙ্গেল সফলভাবে যুক্ত হয়েছে!", Toast.LENGTH_LONG).show()
                showFaceEnrollmentDialog = false
            },
            onDismiss = {
                showFaceEnrollmentDialog = false
            }
        )
    }



    // High-Resolution Sample Inspection Preview Dialog
    if (previewSample != null) {
        val file = remember(previewSample!!.imagePath) { File(previewSample!!.imagePath) }

        AlertDialog(
            onDismissRequest = { previewSample = null },
            title = {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("Sample Inspection", fontWeight = FontWeight.Bold)
                    IconButton(onClick = { previewSample = null }, modifier = Modifier.size(24.dp)) {
                        Icon(Icons.Default.Close, contentDescription = "Close")
                    }
                }
            },
            text = {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    if (file.exists()) {
                        AsyncImage(
                            model = ImageRequest.Builder(context)
                                .data(file)
                                .crossfade(true)
                                .build(),
                            contentDescription = "Full Sample View",
                            contentScale = ContentScale.Fit,
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(max = 300.dp)
                                .clip(RoundedCornerShape(12.dp))
                        )
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(
                            text = file.name,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    } else {
                        Text("Unable to load image file.")
                    }
                }
            },
            confirmButton = {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(
                        onClick = {
                            val id = previewSample!!.id
                            previewSample = null
                            viewModel.deleteSample(id)
                            Toast.makeText(context, "Sample deleted", Toast.LENGTH_SHORT).show()
                        },
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error)
                    ) {
                        Icon(Icons.Default.Delete, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Delete")
                    }
                    Button(onClick = { previewSample = null }) {
                        Text("Close")
                    }
                }
            },
            dismissButton = {}
        )
    }
}

@Composable
fun ClassProgressHeader(
    classificationClass: ClassificationClassEntity,
    sampleCount: Int,
    isFaceRecognition: Boolean = false,
    onEditClass: () -> Unit,
    onAddGallery: () -> Unit,
    onAddFiles: () -> Unit,
    onAddCamera: () -> Unit,
    onEnrollFace: (() -> Unit)? = null,
    onDeleteClass: () -> Unit,
    initiallyExpanded: Boolean = false
) {
    var isExpanded by remember(classificationClass.id) { mutableStateOf(initiallyExpanded) }
    val recommendedMin = if (isFaceRecognition) 3 else 20
    val progress = (sampleCount.toFloat() / recommendedMin).coerceIn(0f, 1f)

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
        ),
        shape = RoundedCornerShape(14.dp)
    ) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
            // Header Row: Class info + Quick Action Buttons or Expand Toggle
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .weight(1f, fill = false)
                        .clickable { onEditClass() }
                        .padding(vertical = 4.dp)
                ) {
                    if (isFaceRecognition) {
                        Icon(
                            Icons.Default.Face,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp),
                            tint = try {
                                Color(android.graphics.Color.parseColor(classificationClass.colorHex))
                            } catch (e: Exception) {
                                MaterialTheme.colorScheme.primary
                            }
                        )
                    } else {
                        Box(
                            modifier = Modifier
                                .size(14.dp)
                                .clip(CircleShape)
                                .background(
                                    try {
                                        Color(android.graphics.Color.parseColor(classificationClass.colorHex))
                                    } catch (e: Exception) {
                                        MaterialTheme.colorScheme.primary
                                    }
                                )
                        )
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = classificationClass.className,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                        maxLines = 1
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Surface(
                        color = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
                        shape = RoundedCornerShape(6.dp)
                    ) {
                        Text(
                            text = "$sampleCount",
                            style = MaterialTheme.typography.labelSmall.copy(
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.primary
                            ),
                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                        )
                    }
                    Spacer(modifier = Modifier.width(4.dp))
                    Icon(
                        imageVector = Icons.Default.Edit,
                        contentDescription = "Edit Name",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                        modifier = Modifier.size(16.dp)
                    )
                }

                // Quick Action buttons row when in compact mode
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(2.dp)
                ) {
                    if (isFaceRecognition && onEnrollFace != null) {
                        IconButton(
                            onClick = onEnrollFace,
                            modifier = Modifier
                                .size(34.dp)
                                .testTag("quick_face_scan_btn")
                        ) {
                            Icon(
                                Icons.Default.Face,
                                contentDescription = "Face Lock Scan",
                                tint = Color(0xFF0284C7),
                                modifier = Modifier.size(18.dp)
                            )
                        }
                    }
                    IconButton(
                        onClick = onAddCamera,
                        modifier = Modifier
                            .size(34.dp)
                            .testTag("quick_camera_btn")
                    ) {
                        Icon(
                            Icons.Default.CameraAlt,
                            contentDescription = "Camera",
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                    IconButton(
                        onClick = onAddGallery,
                        modifier = Modifier
                            .size(34.dp)
                            .testTag("quick_gallery_btn")
                    ) {
                        Icon(
                            Icons.Default.PhotoLibrary,
                            contentDescription = "Gallery",
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                    IconButton(
                        onClick = onAddFiles,
                        modifier = Modifier
                            .size(34.dp)
                            .testTag("quick_files_btn")
                    ) {
                        Icon(
                            Icons.Default.Folder,
                            contentDescription = "Files",
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                    IconButton(
                        onClick = { isExpanded = !isExpanded },
                        modifier = Modifier.size(34.dp)
                    ) {
                        Icon(
                            imageVector = if (isExpanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                            contentDescription = if (isExpanded) "Collapse details" else "Expand details",
                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }

            // Expanded Details Section
            AnimatedVisibility(visible = isExpanded) {
                Column(modifier = Modifier.padding(top = 8.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = if (sampleCount >= recommendedMin)
                                (if (isFaceRecognition) "Recommended face samples met! ($sampleCount)" else "Recommended sample size met! ($sampleCount)")
                            else
                                (if (isFaceRecognition) "Recommended: 1-3 face photos (have $sampleCount)" else "Recommended: 20+ images (have $sampleCount)"),
                            style = MaterialTheme.typography.bodySmall,
                            color = if (sampleCount >= recommendedMin) MaterialTheme.colorScheme.secondary else MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        IconButton(onClick = onDeleteClass, modifier = Modifier.size(28.dp)) {
                            Icon(
                                Icons.Outlined.Delete,
                                contentDescription = "Delete Class",
                                tint = MaterialTheme.colorScheme.error,
                                modifier = Modifier.size(18.dp)
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(6.dp))

                    LinearProgressIndicator(
                        progress = { progress },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(6.dp)
                            .clip(RoundedCornerShape(3.dp)),
                        color = if (sampleCount >= recommendedMin) MaterialTheme.colorScheme.secondary else MaterialTheme.colorScheme.primary
                    )

                    if (isFaceRecognition && onEnrollFace != null) {
                        Spacer(modifier = Modifier.height(10.dp))
                        Button(
                            onClick = onEnrollFace,
                            modifier = Modifier
                                .fillMaxWidth()
                                .defaultMinSize(minHeight = 42.dp)
                                .testTag("face_enroll_header_btn"),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFF0284C7),
                                contentColor = Color.White
                            ),
                            shape = RoundedCornerShape(10.dp),
                            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                        ) {
                            Icon(Icons.Default.Face, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Face Lock Scan (হেড টার্ন বায়োমেট্রিক)",
                                fontWeight = FontWeight.Bold,
                                fontSize = 12.5.sp,
                                textAlign = TextAlign.Center
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Button(
                            onClick = onAddGallery,
                            modifier = Modifier
                                .weight(1f)
                                .defaultMinSize(minHeight = 36.dp)
                                .testTag("pick_gallery_btn"),
                            contentPadding = PaddingValues(horizontal = 6.dp, vertical = 4.dp),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Icon(Icons.Default.PhotoLibrary, contentDescription = null, modifier = Modifier.size(15.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Gallery", fontSize = 11.5.sp, maxLines = 1)
                        }

                        OutlinedButton(
                            onClick = onAddFiles,
                            modifier = Modifier
                                .weight(1f)
                                .defaultMinSize(minHeight = 36.dp)
                                .testTag("pick_files_btn"),
                            contentPadding = PaddingValues(horizontal = 6.dp, vertical = 4.dp),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Icon(Icons.Default.Folder, contentDescription = null, modifier = Modifier.size(15.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Files", fontSize = 11.5.sp, maxLines = 1)
                        }

                        OutlinedButton(
                            onClick = onAddCamera,
                            modifier = Modifier
                                .weight(1f)
                                .defaultMinSize(minHeight = 36.dp)
                                .testTag("take_photo_btn"),
                            contentPadding = PaddingValues(horizontal = 6.dp, vertical = 4.dp),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Icon(Icons.Default.CameraAlt, contentDescription = null, modifier = Modifier.size(15.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Camera", fontSize = 11.5.sp, maxLines = 1)
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun ImageSampleGridTile(
    sample: ImageSampleEntity,
    isSelectMode: Boolean = false,
    isSelected: Boolean = false,
    onToggleSelect: () -> Unit = {},
    onClickPreview: () -> Unit = {},
    onDelete: () -> Unit
) {
    val file = remember(sample.imagePath) { File(sample.imagePath) }
    val context = LocalContext.current

    Box(
        modifier = Modifier
            .aspectRatio(1f)
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .border(
                width = if (isSelected) 2.5.dp else 0.5.dp,
                color = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant,
                shape = RoundedCornerShape(12.dp)
            )
            .clickable {
                if (isSelectMode) {
                    onToggleSelect()
                } else {
                    onClickPreview()
                }
            }
    ) {
        if (file.exists()) {
            AsyncImage(
                model = ImageRequest.Builder(context)
                    .data(file)
                    .crossfade(true)
                    .size(240, 240)
                    .build(),
                contentDescription = "Sample Image",
                contentScale = ContentScale.Crop,
                modifier = Modifier.fillMaxSize()
            )
        } else {
            Icon(
                imageVector = Icons.Default.BrokenImage,
                contentDescription = null,
                modifier = Modifier
                    .align(Alignment.Center)
                    .size(32.dp),
                tint = MaterialTheme.colorScheme.outline
            )
        }

        if (isSelectMode) {
            // Selection Checkbox Overlay
            Box(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(4.dp)
                    .size(26.dp)
                    .background(Color.Black.copy(alpha = 0.5f), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Checkbox(
                    checked = isSelected,
                    onCheckedChange = { onToggleSelect() },
                    modifier = Modifier.size(20.dp)
                )
            }
        } else {
            // Quick Action Buttons
            Row(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(4.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                IconButton(
                    onClick = onDelete,
                    modifier = Modifier
                        .size(26.dp)
                        .background(Color.Black.copy(alpha = 0.65f), CircleShape)
                ) {
                    Icon(
                        Icons.Default.Close,
                        contentDescription = "Remove sample",
                        tint = Color.White,
                        modifier = Modifier.size(14.dp)
                    )
                }
            }
        }
    }
}

@Composable
fun EmptySamplesView(
    className: String,
    isFaceRecognition: Boolean = false,
    onPickGallery: () -> Unit,
    onPickFiles: () -> Unit,
    onTakePhoto: () -> Unit,
    onEnrollFace: (() -> Unit)? = null
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 28.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            imageVector = if (isFaceRecognition) Icons.Default.Face else Icons.Outlined.AddAPhoto,
            contentDescription = null,
            modifier = Modifier.size(64.dp),
            tint = if (isFaceRecognition) Color(0xFF0284C7).copy(alpha = 0.7f) else MaterialTheme.colorScheme.primary.copy(alpha = 0.6f)
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = if (isFaceRecognition) "No Face Photos for '$className'" else "No Images in '$className'",
            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
        )
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = if (isFaceRecognition)
                "Enroll $className by taking or picking clear photos showing their face, or use the 5-angle Face Lock scanner for maximum recognition accuracy."
            else
                "Add photos from gallery, pick files from storage sections, or capture with camera.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 32.dp)
        )

        if (isFaceRecognition && onEnrollFace != null) {
            Spacer(modifier = Modifier.height(16.dp))
            Button(
                onClick = onEnrollFace,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF0284C7),
                    contentColor = Color.White
                ),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.testTag("empty_face_enroll_btn")
            ) {
                Icon(Icons.Default.Face, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text("Face Lock Scan (৫-অ্যাঙ্গেল স্ক্যান)", fontWeight = FontWeight.Bold)
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = onPickGallery) {
                Icon(Icons.Default.PhotoLibrary, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("Gallery")
            }
            OutlinedButton(onClick = onPickFiles) {
                Icon(Icons.Default.Folder, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("Files")
            }
            OutlinedButton(onClick = onTakePhoto) {
                Icon(Icons.Default.CameraAlt, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("Camera")
            }
        }
    }
}

@Composable
fun AddClassDialog(
    isFaceRecognition: Boolean = false,
    onDismiss: () -> Unit,
    onAdd: (String, String) -> Unit
) {
    var className by remember { mutableStateOf("") }
    val colors = listOf("#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6")
    var selectedColor by remember { mutableStateOf(colors[0]) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(if (isFaceRecognition) "Enroll New Person" else "Add Class Category") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                OutlinedTextField(
                    value = className,
                    onValueChange = { className = it },
                    label = { Text(if (isFaceRecognition) "Person Name" else "Class Name") },
                    placeholder = { Text(if (isFaceRecognition) "e.g., John Doe, Alice" else "e.g., Cat, Dog, Apple") },
                    singleLine = true,
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("class_name_input")
                )

                Text(if (isFaceRecognition) "Person Badge Color:" else "Class Badge Color:", style = MaterialTheme.typography.bodyMedium)

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    colors.forEach { hex ->
                        val color = Color(android.graphics.Color.parseColor(hex))
                        Box(
                            modifier = Modifier
                                .size(32.dp)
                                .clip(CircleShape)
                                .background(color)
                                .border(
                                    width = if (hex == selectedColor) 3.dp else 0.dp,
                                    color = if (hex == selectedColor) MaterialTheme.colorScheme.onSurface else Color.Transparent,
                                    shape = CircleShape
                                )
                                .clickable { selectedColor = hex }
                        )
                    }
                }
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    if (className.isNotBlank()) {
                        onAdd(className.trim(), selectedColor)
                    }
                },
                enabled = className.isNotBlank(),
                modifier = Modifier.testTag("confirm_add_class_btn")
            ) {
                Text(if (isFaceRecognition) "Enroll Person" else "Add Class")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}

@Composable
fun EditClassDialog(
    classificationClass: ClassificationClassEntity,
    isFaceRecognition: Boolean = false,
    onDismiss: () -> Unit,
    onSave: (String, String) -> Unit
) {
    var className by remember { mutableStateOf(classificationClass.className) }
    val colors = listOf("#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6")
    var selectedColor by remember { mutableStateOf(classificationClass.colorHex) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(if (isFaceRecognition) "Rename Person" else "Rename Class Category") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                OutlinedTextField(
                    value = className,
                    onValueChange = { className = it },
                    label = { Text(if (isFaceRecognition) "Person Name" else "Class Name") },
                    singleLine = true,
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("edit_class_name_input")
                )

                Text(if (isFaceRecognition) "Person Badge Color:" else "Class Badge Color:", style = MaterialTheme.typography.bodyMedium)

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    colors.forEach { hex ->
                        val color = Color(android.graphics.Color.parseColor(hex))
                        Box(
                            modifier = Modifier
                                .size(32.dp)
                                .clip(CircleShape)
                                .background(color)
                                .border(
                                    width = if (hex == selectedColor) 3.dp else 0.dp,
                                    color = if (hex == selectedColor) MaterialTheme.colorScheme.onSurface else Color.Transparent,
                                    shape = CircleShape
                                )
                                .clickable { selectedColor = hex }
                        )
                    }
                }
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    if (className.isNotBlank()) {
                        onSave(className.trim(), selectedColor)
                    }
                },
                enabled = className.isNotBlank(),
                modifier = Modifier.testTag("confirm_edit_class_btn")
            ) {
                Text("Save")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}
