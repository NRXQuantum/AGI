package com.example.ui.screens

import android.content.Context
import android.content.Intent
import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.documentfile.provider.DocumentFile
import android.net.Uri
import android.os.Environment
import com.example.data.db.ProjectEntity
import com.example.ml.AutoTuner
import com.example.ml.BatchFolderSorter
import com.example.ml.FileSortAction
import com.example.ml.LoadedExportedModel
import com.example.ml.ModelExporter
import com.example.ml.SorterPacingMode
import com.example.service.TelegramBotService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.example.ui.viewmodel.AppMode
import com.example.ui.viewmodel.ProjectViewModel
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

enum class SettingsSubPage {
    MAIN,
    TELEGRAM_BOT,
    BATCH_SORTER
}

enum class SorterModelSource {
    APP_PROJECT,
    CUSTOM_FILE
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrainerSettingsScreen(
    viewModel: ProjectViewModel,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val allProjects by viewModel.projects.collectAsState()
    val currentAppMode by viewModel.currentAppMode.collectAsState()

    var currentSubPage by remember { mutableStateOf(SettingsSubPage.MAIN) }

    // Intercept hardware Back button / gesture:
    // If inside a sub-page (Telegram Bot / Batch Sorter), return to MAIN settings.
    // If on MAIN settings, onBack() returns smoothly to the Project List Screen.
    BackHandler {
        if (currentSubPage != SettingsSubPage.MAIN) {
            currentSubPage = SettingsSubPage.MAIN
        } else {
            onBack()
        }
    }

    // Hardware & Auto-Tune state
    val isAutoTuneEnabled by viewModel.isAutoTuneGlobalEnabled.collectAsState()
    val hardwareProfile = remember { AutoTuner.getHardwareProfile(context) }

    // Telegram Bot Service State
    val botService = remember { TelegramBotService.getInstance(context) }
    val isBotRunning by botService.isRunning.collectAsState()
    val botUsername by botService.botUsername.collectAsState()
    val activeProjectName by botService.activeProjectName.collectAsState()
    val processedCount by botService.processedCount.collectAsState()
    val botLogs by botService.logs.collectAsState()

    val tgPrefs = remember { context.getSharedPreferences("tg_bot_prefs", Context.MODE_PRIVATE) }
    var botToken by remember { mutableStateOf(tgPrefs.getString("saved_bot_token", "") ?: "") }
    var selectedProjectForBot by remember(allProjects) {
        mutableStateOf(allProjects.firstOrNull { it.isTrained } ?: allProjects.firstOrNull())
    }
    var showBotProjectDropdown by remember { mutableStateOf(false) }

    // Batch Sorter Service State
    val batchSorter = remember { BatchFolderSorter.getInstance(context) }
    val batchSortState by batchSorter.state.collectAsState()

    val defaultSourceDir = remember {
        File(context.getExternalFilesDir(null), "Unsorted_Images").absolutePath
    }
    val defaultDestDir = remember {
        File(context.getExternalFilesDir(null), "Classified_Results").absolutePath
    }

    val sorterPrefs = remember { context.getSharedPreferences("batch_sorter_prefs", Context.MODE_PRIVATE) }
    var sourceUriOrPath by remember {
        mutableStateOf(sorterPrefs.getString("saved_source_uri", defaultSourceDir) ?: defaultSourceDir)
    }
    var sourceDisplayName by remember {
        mutableStateOf(sorterPrefs.getString("saved_source_name", "App Folder (Unsorted_Images)") ?: "App Folder (Unsorted_Images)")
    }
    var destUriOrPath by remember {
        mutableStateOf(sorterPrefs.getString("saved_dest_uri", defaultDestDir) ?: defaultDestDir)
    }
    var destDisplayName by remember {
        mutableStateOf(sorterPrefs.getString("saved_dest_name", "App Results (Classified_Results)") ?: "App Results (Classified_Results)")
    }

    var selectedFileAction by remember {
        val savedAction = sorterPrefs.getString("saved_file_action", FileSortAction.MOVE.name)
        mutableStateOf(try { FileSortAction.valueOf(savedAction ?: "MOVE") } catch (_: Exception) { FileSortAction.MOVE })
    }
    var selectedPacingMode by remember {
        val savedMode = sorterPrefs.getString("saved_pacing_mode", SorterPacingMode.AUTO.name)
        mutableStateOf(try { SorterPacingMode.valueOf(savedMode ?: "AUTO") } catch (_: Exception) { SorterPacingMode.AUTO })
    }
    var skipIfNoFaceDetected by remember {
        mutableStateOf(sorterPrefs.getBoolean("saved_skip_no_face", false))
    }

    val sourceFolderPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { uri ->
        if (uri != null) {
            try {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                )
            } catch (_: Exception) {}
            val doc = DocumentFile.fromTreeUri(context, uri)
            val name = doc?.name ?: "Selected Folder"
            sourceUriOrPath = uri.toString()
            sourceDisplayName = name
            sorterPrefs.edit()
                .putString("saved_source_uri", uri.toString())
                .putString("saved_source_name", name)
                .apply()
            Toast.makeText(context, "Source folder allowed: $name", Toast.LENGTH_SHORT).show()
        }
    }

    val destFolderPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { uri ->
        if (uri != null) {
            try {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                )
            } catch (_: Exception) {}
            val doc = DocumentFile.fromTreeUri(context, uri)
            val name = doc?.name ?: "Selected Folder"
            destUriOrPath = uri.toString()
            destDisplayName = name
            sorterPrefs.edit()
                .putString("saved_dest_uri", uri.toString())
                .putString("saved_dest_name", name)
                .apply()
            Toast.makeText(context, "Destination folder allowed: $name", Toast.LENGTH_SHORT).show()
        }
    }

    var selectedProjectForSorter by remember(allProjects) {
        mutableStateOf(allProjects.firstOrNull { it.isTrained } ?: allProjects.firstOrNull())
    }
    var showSorterProjectDropdown by remember { mutableStateOf(false) }

    LaunchedEffect(selectedProjectForSorter) {
        val isFaceProj = selectedProjectForSorter?.projectType == "FACE_RECOGNITION" ||
                selectedProjectForSorter?.name?.contains("Face", ignoreCase = true) == true
        if (isFaceProj) {
            // Automatically enable face filter for Face Recognition project if not explicitly turned off
            if (!sorterPrefs.contains("saved_skip_no_face")) {
                skipIfNoFaceDetected = true
                sorterPrefs.edit().putBoolean("saved_skip_no_face", true).apply()
            }
        } else {
            // For Normal Image Classification projects, NEVER enable face filtering
            skipIfNoFaceDetected = false
        }
    }

    var sorterModelSource by remember { mutableStateOf(SorterModelSource.APP_PROJECT) }
    var loadedCustomModel by remember { mutableStateOf<LoadedExportedModel?>(null) }
    var customModelLoadError by remember { mutableStateOf<String?>(null) }

    val customModelPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            coroutineScope.launch(Dispatchers.IO) {
                try {
                    context.contentResolver.takePersistableUriPermission(
                        uri,
                        Intent.FLAG_GRANT_READ_URI_PERMISSION
                    )
                } catch (_: Exception) {}
                val exporter = ModelExporter(context)
                val parsed = exporter.parseExportedModelUri(uri)
                withContext(Dispatchers.Main) {
                    if (parsed != null) {
                        loadedCustomModel = parsed
                        customModelLoadError = null
                        Toast.makeText(
                            context,
                            "Loaded model: ${parsed.fileName} (${parsed.numClasses} classes)",
                            Toast.LENGTH_SHORT
                        ).show()
                    } else {
                        customModelLoadError = "Could not parse model file. Please select a valid .tflite, .onnx, .mlmodel, or .json file."
                        Toast.makeText(context, "Invalid model format", Toast.LENGTH_LONG).show()
                    }
                }
            }
        }
    }

    val availableExportedModels by produceState<List<File>>(initialValue = emptyList(), key1 = currentSubPage) {
        val files = mutableListOf<File>()
        try {
            val appExportDir = File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), "CustomMLModels")
            if (appExportDir.exists() && appExportDir.isDirectory) {
                val list = appExportDir.listFiles { f ->
                    f.isFile && (f.name.endsWith(".tflite") || f.name.endsWith(".onnx") || f.name.endsWith(".mlmodel") || f.name.endsWith(".json"))
                }
                if (list != null) files.addAll(list)
            }
            val publicDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "CustomMLModels")
            if (publicDir.exists() && publicDir.isDirectory) {
                val list = publicDir.listFiles { f ->
                    f.isFile && (f.name.endsWith(".tflite") || f.name.endsWith(".onnx") || f.name.endsWith(".mlmodel") || f.name.endsWith(".json")) && !files.any { it.name == f.name }
                }
                if (list != null) files.addAll(list)
            }
        } catch (_: Exception) {}
        value = files.sortedByDescending { it.lastModified() }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = when (currentSubPage) {
                                SettingsSubPage.MAIN -> "Trainer Settings"
                                SettingsSubPage.TELEGRAM_BOT -> "Telegram Bot Integration"
                                SettingsSubPage.BATCH_SORTER -> "Batch Folder Auto-Sorter"
                            },
                            style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        Text(
                            text = when (currentSubPage) {
                                SettingsSubPage.MAIN -> "On-Device Automation & Hardware Tuning"
                                SettingsSubPage.TELEGRAM_BOT -> if (isBotRunning) "🟢 Online ${botUsername?.let { "(@$it)" } ?: ""}" else "On-device AI Telegram Bot"
                                SettingsSubPage.BATCH_SORTER -> if (batchSortState.isRunning) "🟢 Sorting images..." else "Auto-sort unorganized images by AI class"
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = if (currentSubPage == SettingsSubPage.TELEGRAM_BOT && isBotRunning || currentSubPage == SettingsSubPage.BATCH_SORTER && batchSortState.isRunning)
                                Color(0xFF2E7D32)
                            else
                                MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                },
                navigationIcon = {
                    IconButton(
                        onClick = {
                            if (currentSubPage != SettingsSubPage.MAIN) {
                                currentSubPage = SettingsSubPage.MAIN
                            } else {
                                onBack()
                            }
                        },
                        modifier = Modifier.testTag("settings_back_btn")
                    ) {
                        Icon(
                            Icons.Default.ArrowBack,
                            contentDescription = if (currentSubPage != SettingsSubPage.MAIN) "Back to Settings" else "Back to Projects"
                        )
                    }
                },
                actions = {
                    if (currentSubPage != SettingsSubPage.MAIN) {
                        IconButton(onClick = onBack) {
                            Icon(Icons.Default.Close, contentDescription = "Close Settings")
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { innerPadding ->
        AnimatedContent(
            targetState = currentSubPage,
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
            transitionSpec = {
                if (targetState != SettingsSubPage.MAIN) {
                    slideInHorizontally { it } + fadeIn() togetherWith slideOutHorizontally { -it / 3 } + fadeOut()
                } else {
                    slideInHorizontally { -it / 3 } + fadeIn() togetherWith slideOutHorizontally { it } + fadeOut()
                }
            },
            label = "settings_fullscreen_subpage_anim"
        ) { page ->
            when (page) {
                SettingsSubPage.MAIN -> {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(horizontal = 16.dp)
                            .verticalScroll(rememberScrollState())
                    ) {
                        Spacer(modifier = Modifier.height(12.dp))

                        // 0. Core Workspace Mode Switcher
                        Text(
                            text = "CORE WORKSPACE MODE (কাজের প্রধান মোড)",
                            style = MaterialTheme.typography.labelSmall.copy(
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp
                            ),
                            color = MaterialTheme.colorScheme.primary
                        )
                        Spacer(modifier = Modifier.height(8.dp))

                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(16.dp)),
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                            ),
                            shape = RoundedCornerShape(16.dp),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)
                        ) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Box(
                                        modifier = Modifier
                                            .size(42.dp)
                                            .clip(CircleShape)
                                            .background(
                                                if (currentAppMode == AppMode.FACE_RECOGNITION)
                                                    Color(0xFF0284C7).copy(alpha = 0.15f)
                                                else
                                                    MaterialTheme.colorScheme.primary.copy(alpha = 0.15f)
                                            ),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Icon(
                                            imageVector = if (currentAppMode == AppMode.FACE_RECOGNITION)
                                                Icons.Default.Face
                                            else
                                                Icons.Default.Image,
                                            contentDescription = null,
                                            tint = if (currentAppMode == AppMode.FACE_RECOGNITION)
                                                Color(0xFF0284C7)
                                            else
                                                MaterialTheme.colorScheme.primary,
                                            modifier = Modifier.size(22.dp)
                                        )
                                    }
                                    Spacer(modifier = Modifier.width(12.dp))
                                    Column(modifier = Modifier.weight(1f)) {
                                        Text(
                                            text = "Application Workflow Mode",
                                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                                        )
                                        Text(
                                            text = if (currentAppMode == AppMode.FACE_RECOGNITION)
                                                "Active: Face / Person ID Mode (ব্যক্তি চেনার মোড)"
                                            else
                                                "Active: Image Classification Mode (ছবি শনাক্তকরণ)",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = if (currentAppMode == AppMode.FACE_RECOGNITION)
                                                Color(0xFF0284C7)
                                            else
                                                MaterialTheme.colorScheme.primary,
                                            fontWeight = FontWeight.SemiBold
                                        )
                                    }
                                }

                                Spacer(modifier = Modifier.height(14.dp))

                                SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                                    SegmentedButton(
                                        selected = currentAppMode == AppMode.IMAGE_CLASSIFICATION,
                                        onClick = { viewModel.setAppMode(AppMode.IMAGE_CLASSIFICATION) },
                                        shape = SegmentedButtonDefaults.itemShape(index = 0, count = 2),
                                        icon = {
                                            Icon(Icons.Default.Image, contentDescription = null, modifier = Modifier.size(16.dp))
                                        }
                                    ) {
                                        Text("Normal Images", maxLines = 1)
                                    }
                                    SegmentedButton(
                                        selected = currentAppMode == AppMode.FACE_RECOGNITION,
                                        onClick = { viewModel.setAppMode(AppMode.FACE_RECOGNITION) },
                                        shape = SegmentedButtonDefaults.itemShape(index = 1, count = 2),
                                        icon = {
                                            Icon(Icons.Default.Face, contentDescription = null, modifier = Modifier.size(16.dp))
                                        }
                                    ) {
                                        Text("Face / Person ID", maxLines = 1)
                                    }
                                }

                                Spacer(modifier = Modifier.height(10.dp))

                                Text(
                                    text = if (currentAppMode == AppMode.FACE_RECOGNITION)
                                        "Face / Person ID Mode: Projects create 'Person A' and 'Person B' by default. Uses 1-3 face photos per person with instant 100% on-device calibration and multi-face camera recognition."
                                    else
                                        "Normal Image Mode: Standard visual classification for general objects, categories, plants, and animals.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(20.dp))

                        Text(
                            text = "INTELLIGENT ON-DEVICE TOOLS",
                            style = MaterialTheme.typography.labelSmall.copy(
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp
                            ),
                            color = MaterialTheme.colorScheme.primary
                        )
                        Spacer(modifier = Modifier.height(10.dp))

                        // 1. Telegram Bot Integration Menu Item
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(16.dp))
                                .clickable { currentSubPage = SettingsSubPage.TELEGRAM_BOT }
                                .testTag("telegram_bot_settings_item"),
                            colors = CardDefaults.cardColors(
                                containerColor = if (isBotRunning)
                                    MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.35f)
                                else
                                    MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                            ),
                            shape = RoundedCornerShape(16.dp),
                            border = BorderStroke(
                                1.dp,
                                if (isBotRunning) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant
                            )
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(46.dp)
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.15f)),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        Icons.Default.Send,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(24.dp)
                                    )
                                }
                                Spacer(modifier = Modifier.width(14.dp))
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = "Telegram Bot Integration",
                                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                                    )
                                    Spacer(modifier = Modifier.height(2.dp))
                                    Text(
                                        text = if (isBotRunning)
                                            "🟢 Online ${botUsername?.let { "(@$it)" } ?: ""} • $processedCount analyzed"
                                        else
                                            "On-device AI bot • Zero-retention privacy",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = if (isBotRunning) Color(0xFF2E7D32) else MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                                Spacer(modifier = Modifier.width(8.dp))
                                Surface(
                                    color = if (isBotRunning) Color(0xFF1B5E20) else MaterialTheme.colorScheme.surfaceContainerHigh,
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Text(
                                        text = if (isBotRunning) "ONLINE" else "EDGE BOT",
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontWeight = FontWeight.Bold,
                                            color = if (isBotRunning) Color.White else MaterialTheme.colorScheme.onSurfaceVariant
                                        ),
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                                    )
                                }
                                Spacer(modifier = Modifier.width(6.dp))
                                Icon(
                                    Icons.Default.ChevronRight,
                                    contentDescription = "Open Telegram Bot Settings",
                                    tint = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(12.dp))

                        // 2. Batch Folder Auto-Sorter Menu Item
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(16.dp))
                                .clickable { currentSubPage = SettingsSubPage.BATCH_SORTER }
                                .testTag("batch_sorter_settings_item"),
                            colors = CardDefaults.cardColors(
                                containerColor = if (batchSortState.isRunning)
                                    MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.35f)
                                else
                                    MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                            ),
                            shape = RoundedCornerShape(16.dp),
                            border = BorderStroke(
                                1.dp,
                                if (batchSortState.isRunning) MaterialTheme.colorScheme.secondary else MaterialTheme.colorScheme.outlineVariant
                            )
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(46.dp)
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(MaterialTheme.colorScheme.secondary.copy(alpha = 0.15f)),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        Icons.Default.DriveFolderUpload,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.secondary,
                                        modifier = Modifier.size(24.dp)
                                    )
                                }
                                Spacer(modifier = Modifier.width(14.dp))
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = "Batch Folder Auto-Sorter",
                                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                                    )
                                    Spacer(modifier = Modifier.height(2.dp))
                                    Text(
                                        text = when {
                                            batchSortState.isRunning -> "🟢 Sorting ${batchSortState.currentImageIndex}/${batchSortState.totalImages} photos..."
                                            batchSortState.isCompleted -> "🎉 Sorted ${batchSortState.sortedSummary.values.sum()} photos into folders"
                                            else -> "Auto-sort unorganized images by AI class"
                                        },
                                        style = MaterialTheme.typography.bodySmall,
                                        color = if (batchSortState.isRunning) Color(0xFF2E7D32) else MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                                Spacer(modifier = Modifier.width(8.dp))
                                Surface(
                                    color = when {
                                        batchSortState.isRunning -> Color(0xFF1B5E20)
                                        batchSortState.isCompleted -> MaterialTheme.colorScheme.primaryContainer
                                        else -> MaterialTheme.colorScheme.surfaceContainerHigh
                                    },
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Text(
                                        text = when {
                                            batchSortState.isRunning -> "RUNNING"
                                            batchSortState.isCompleted -> "DONE"
                                            else -> "AI SORTER"
                                        },
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontWeight = FontWeight.Bold,
                                            color = when {
                                                batchSortState.isRunning -> Color.White
                                                batchSortState.isCompleted -> MaterialTheme.colorScheme.onPrimaryContainer
                                                else -> MaterialTheme.colorScheme.onSurfaceVariant
                                            }
                                        ),
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                                    )
                                }
                                Spacer(modifier = Modifier.width(6.dp))
                                Icon(
                                    Icons.Default.ChevronRight,
                                    contentDescription = "Open Batch Folder Auto-Sorter",
                                    tint = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(24.dp))

                        // Device Hardware & Engine Tuning Section
                        Text(
                            text = "HARDWARE & PERFORMANCE",
                            style = MaterialTheme.typography.labelSmall.copy(
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp
                            ),
                            color = MaterialTheme.colorScheme.primary
                        )
                        Spacer(modifier = Modifier.height(10.dp))

                        // Hardware Intelligence Card
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                            ),
                            shape = RoundedCornerShape(14.dp)
                        ) {
                            Column(modifier = Modifier.padding(14.dp)) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Icon(
                                            Icons.Default.Memory,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.primary,
                                            modifier = Modifier.size(20.dp)
                                        )
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(
                                            text = "Device Hardware Profile",
                                            style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                                        )
                                    }
                                    Surface(
                                        color = MaterialTheme.colorScheme.primaryContainer,
                                        shape = RoundedCornerShape(6.dp)
                                    ) {
                                        Text(
                                            text = hardwareProfile.tier.displayName,
                                            style = MaterialTheme.typography.labelSmall.copy(
                                                fontWeight = FontWeight.Bold,
                                                color = MaterialTheme.colorScheme.onPrimaryContainer
                                            ),
                                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                                        )
                                    }
                                }

                                Spacer(modifier = Modifier.height(10.dp))

                                Text(
                                    text = "• CPU Capacity: ${hardwareProfile.cpuCores} Cores @ ${String.format(Locale.US, "%.2f", hardwareProfile.maxCpuFreqGhz)} GHz max clock\n• System Memory: ${String.format(Locale.US, "%.1f", hardwareProfile.totalRamGb)} GB RAM",
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(14.dp))

                        // Auto-Tune Settings Card
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(
                                containerColor = if (isAutoTuneEnabled)
                                    MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.35f)
                                else
                                    MaterialTheme.colorScheme.surface
                            ),
                            shape = RoundedCornerShape(14.dp),
                            border = BorderStroke(
                                1.dp,
                                if (isAutoTuneEnabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant
                            )
                        ) {
                            Column(modifier = Modifier.padding(14.dp)) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Row(
                                        modifier = Modifier.weight(1f),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Icon(
                                            Icons.Default.AutoAwesome,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.primary,
                                            modifier = Modifier.size(20.dp)
                                        )
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Column {
                                            Text(
                                                text = "Smart Auto-Tune",
                                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                                            )
                                            Text(
                                                text = "Automatically determines optimal batch size, epochs & architecture based on device CPU/RAM and dataset size.",
                                                style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.5.sp),
                                                color = MaterialTheme.colorScheme.onSurfaceVariant
                                            )
                                        }
                                    }
                                    Spacer(modifier = Modifier.width(12.dp))
                                    Switch(
                                        checked = isAutoTuneEnabled,
                                        onCheckedChange = { viewModel.setAutoTuneGlobalEnabled(it) },
                                        modifier = Modifier.testTag("auto_tune_settings_switch")
                                    )
                                }

                                Spacer(modifier = Modifier.height(10.dp))

                                Text(
                                    text = "Note: Any manual adjustments you make in the Training screen are remembered per project. Turning Auto-Tune off and on again will re-calculate optimal defaults.",
                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )

                                if (isAutoTuneEnabled) {
                                    Spacer(modifier = Modifier.height(12.dp))
                                    OutlinedButton(
                                        onClick = { viewModel.resetAllProjectsToAutoTune() },
                                        modifier = Modifier.fillMaxWidth(),
                                        shape = RoundedCornerShape(8.dp)
                                    ) {
                                        Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
                                        Spacer(modifier = Modifier.width(6.dp))
                                        Text("Re-Apply Auto-Tune to All Projects")
                                    }
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(32.dp))
                    }
                }

                SettingsSubPage.TELEGRAM_BOT -> {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(horizontal = 16.dp)
                            .verticalScroll(rememberScrollState())
                    ) {
                        Spacer(modifier = Modifier.height(12.dp))

                        Text(
                            text = "Connect any trained neural model directly to Telegram. When anyone sends a photo to your bot, it classifies objects using this phone's on-device AI.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )

                        Spacer(modifier = Modifier.height(12.dp))

                        // Zero-Retention Privacy Banner
                        Surface(
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f),
                            shape = RoundedCornerShape(10.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Row(
                                modifier = Modifier.padding(12.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    Icons.Default.CleaningServices,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(20.dp)
                                )
                                Spacer(modifier = Modifier.width(10.dp))
                                Text(
                                    text = "Zero-Retention Privacy: Photos are auto-deleted immediately after classification. 0 bytes retained.",
                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.5.sp),
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(16.dp))

                        // Select Trained Model
                        Text(
                            text = "Select Trained Model:",
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold)
                        )
                        Spacer(modifier = Modifier.height(4.dp))

                        Box {
                            OutlinedCard(
                                onClick = { if (!isBotRunning) showBotProjectDropdown = true },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(12.dp),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Icon(
                                            Icons.Default.ModelTraining,
                                            contentDescription = null,
                                            modifier = Modifier.size(18.dp),
                                            tint = MaterialTheme.colorScheme.primary
                                        )
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(
                                            text = selectedProjectForBot?.let {
                                                "${it.name} (${if (it.isTrained) "Trained" else "Untrained"})"
                                            } ?: "No project available",
                                            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Medium)
                                        )
                                    }
                                    Icon(Icons.Default.ArrowDropDown, contentDescription = null)
                                }
                            }

                            DropdownMenu(
                                expanded = showBotProjectDropdown,
                                onDismissRequest = { showBotProjectDropdown = false }
                            ) {
                                if (allProjects.isEmpty()) {
                                    DropdownMenuItem(
                                        text = { Text("No projects found") },
                                        onClick = { showBotProjectDropdown = false }
                                    )
                                } else {
                                    allProjects.forEach { proj ->
                                        DropdownMenuItem(
                                            text = {
                                                Row(
                                                    modifier = Modifier.fillMaxWidth(),
                                                    horizontalArrangement = Arrangement.SpaceBetween,
                                                    verticalAlignment = Alignment.CenterVertically
                                                ) {
                                                    Text(proj.name)
                                                    Spacer(modifier = Modifier.width(8.dp))
                                                    if (proj.isTrained) {
                                                        Surface(
                                                            color = MaterialTheme.colorScheme.primaryContainer,
                                                            shape = RoundedCornerShape(4.dp)
                                                        ) {
                                                            Text(
                                                                "Trained",
                                                                style = MaterialTheme.typography.labelSmall,
                                                                modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp)
                                                            )
                                                        }
                                                    } else {
                                                        Text(
                                                            "Untrained",
                                                            style = MaterialTheme.typography.labelSmall,
                                                            color = MaterialTheme.colorScheme.outline
                                                        )
                                                    }
                                                }
                                            },
                                            onClick = {
                                                selectedProjectForBot = proj
                                                showBotProjectDropdown = false
                                            }
                                        )
                                    }
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(14.dp))

                        // Telegram Bot Token
                        Text(
                            text = "Telegram Bot Token (from @BotFather):",
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold)
                        )
                        Spacer(modifier = Modifier.height(4.dp))

                        OutlinedTextField(
                            value = botToken,
                            onValueChange = { botToken = it },
                            placeholder = { Text("e.g. 123456789:ABCdefGhIJKlmNo...") },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true,
                            enabled = !isBotRunning,
                            shape = RoundedCornerShape(8.dp)
                        )

                        Spacer(modifier = Modifier.height(16.dp))

                        // Start / Stop Button
                        if (isBotRunning) {
                            Surface(
                                color = Color(0xFF2E7D32).copy(alpha = 0.15f),
                                shape = RoundedCornerShape(8.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(12.dp),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Column {
                                        Text(
                                            text = "🟢 Bot is Running Live",
                                            style = MaterialTheme.typography.bodyMedium.copy(
                                                fontWeight = FontWeight.Bold,
                                                color = Color(0xFF2E7D32)
                                            )
                                        )
                                        Text(
                                            text = "Model: $activeProjectName • Photos analyzed: $processedCount",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                    Button(
                                        onClick = { botService.stopBot() },
                                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
                                        shape = RoundedCornerShape(8.dp)
                                    ) {
                                        Icon(Icons.Default.Stop, contentDescription = null, modifier = Modifier.size(16.dp))
                                        Spacer(modifier = Modifier.width(4.dp))
                                        Text("Stop Bot")
                                    }
                                }
                            }
                        } else {
                            Button(
                                onClick = {
                                    if (botToken.isBlank()) {
                                        Toast.makeText(context, "Please enter your Telegram Bot Token!", Toast.LENGTH_SHORT).show()
                                        return@Button
                                    }
                                    val proj = selectedProjectForBot
                                    if (proj == null) {
                                        Toast.makeText(context, "Please select a model first!", Toast.LENGTH_SHORT).show()
                                        return@Button
                                    }
                                    tgPrefs.edit().putString("saved_bot_token", botToken).apply()
                                    botService.startBot(
                                        botToken = botToken,
                                        project = proj,
                                        repository = viewModel.getRepository()
                                    )
                                },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Icon(Icons.Default.PlayArrow, contentDescription = null)
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Start Telegram Bot")
                            }
                        }

                        // Activity logs
                        if (botLogs.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(16.dp))
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = "Live Activity & Purge Logs:",
                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                                )
                                TextButton(onClick = { botService.clearLogs() }) {
                                    Text("Clear", style = MaterialTheme.typography.labelSmall)
                                }
                            }
                            Surface(
                                color = MaterialTheme.colorScheme.surfaceContainerHighest,
                                shape = RoundedCornerShape(8.dp),
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(max = 180.dp)
                            ) {
                                LazyColumn(
                                    modifier = Modifier.padding(8.dp),
                                    reverseLayout = true
                                ) {
                                    items(botLogs.reversed()) { log ->
                                        val timeStr = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(log.timestamp))
                                        Text(
                                            text = "[$timeStr] ${log.message}",
                                            style = MaterialTheme.typography.bodySmall.copy(
                                                fontSize = 11.sp,
                                                fontFamily = FontFamily.Monospace,
                                                color = if (log.isError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface
                                            ),
                                            modifier = Modifier.padding(vertical = 2.dp)
                                        )
                                    }
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(32.dp))
                    }
                }

                SettingsSubPage.BATCH_SORTER -> {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(horizontal = 16.dp)
                            .verticalScroll(rememberScrollState())
                    ) {
                        Spacer(modifier = Modifier.height(12.dp))

                        Text(
                            text = "Scan a folder of unorganized photos, classify them using your on-device AI model, and auto-sort them into category folders.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )

                        Spacer(modifier = Modifier.height(14.dp))

                        // 1. Select AI Model (প্রজেক্ট তালিকা অথবা সরাসরি এক্সপোর্ট করা মডেল ফাইল)
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "1. Select AI Model (মডেল নির্বাচন বা ফাইল ইনপুট):",
                                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold)
                            )
                            val isReady = if (sorterModelSource == SorterModelSource.CUSTOM_FILE) {
                                loadedCustomModel != null
                            } else {
                                selectedProjectForSorter?.isTrained == true
                            }
                            Surface(
                                color = if (isReady) Color(0xFF2E7D32).copy(alpha = 0.15f) else MaterialTheme.colorScheme.surfaceVariant,
                                shape = RoundedCornerShape(12.dp)
                            ) {
                                Text(
                                    text = if (isReady) "✓ Ready" else "Pending",
                                    color = if (isReady) Color(0xFF4CAF50) else MaterialTheme.colorScheme.onSurfaceVariant,
                                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                                )
                            }
                        }
                        Spacer(modifier = Modifier.height(6.dp))

                        // Model Source Switcher: [ 📱 App Projects ] [ 📦 Direct Model File ]
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f), RoundedCornerShape(10.dp))
                                .padding(4.dp),
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            FilterChip(
                                selected = sorterModelSource == SorterModelSource.APP_PROJECT,
                                onClick = { if (!batchSortState.isRunning) sorterModelSource = SorterModelSource.APP_PROJECT },
                                leadingIcon = {
                                    Icon(Icons.Default.ModelTraining, contentDescription = null, modifier = Modifier.size(16.dp))
                                },
                                label = { Text("App Projects", style = MaterialTheme.typography.labelMedium) },
                                modifier = Modifier.weight(1f),
                                enabled = !batchSortState.isRunning
                            )

                            FilterChip(
                                selected = sorterModelSource == SorterModelSource.CUSTOM_FILE,
                                onClick = { if (!batchSortState.isRunning) sorterModelSource = SorterModelSource.CUSTOM_FILE },
                                leadingIcon = {
                                    Icon(Icons.Default.FileUpload, contentDescription = null, modifier = Modifier.size(16.dp))
                                },
                                label = { Text("Direct File (.tflite)", style = MaterialTheme.typography.labelMedium) },
                                modifier = Modifier.weight(1f),
                                enabled = !batchSortState.isRunning
                            )
                        }

                        Spacer(modifier = Modifier.height(8.dp))

                        if (sorterModelSource == SorterModelSource.APP_PROJECT) {
                            // Option A: Choose from App Projects
                            Box {
                                OutlinedCard(
                                    onClick = { if (!batchSortState.isRunning) showSorterProjectDropdown = true },
                                    modifier = Modifier.fillMaxWidth(),
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(12.dp),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Icon(
                                                Icons.Default.ModelTraining,
                                                contentDescription = null,
                                                modifier = Modifier.size(18.dp),
                                                tint = MaterialTheme.colorScheme.secondary
                                            )
                                            Spacer(modifier = Modifier.width(8.dp))
                                            Text(
                                                text = selectedProjectForSorter?.let {
                                                    "${it.name} (${if (it.isTrained) "Trained" else "Untrained"})"
                                                } ?: "No project available",
                                                style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Medium)
                                            )
                                        }
                                        Icon(Icons.Default.ArrowDropDown, contentDescription = null)
                                    }
                                }

                                DropdownMenu(
                                    expanded = showSorterProjectDropdown,
                                    onDismissRequest = { showSorterProjectDropdown = false }
                                ) {
                                    if (allProjects.isEmpty()) {
                                        DropdownMenuItem(
                                            text = { Text("No projects found") },
                                            onClick = { showSorterProjectDropdown = false }
                                        )
                                    } else {
                                        allProjects.forEach { proj ->
                                            DropdownMenuItem(
                                                text = {
                                                    Row(
                                                        modifier = Modifier.fillMaxWidth(),
                                                        horizontalArrangement = Arrangement.SpaceBetween,
                                                        verticalAlignment = Alignment.CenterVertically
                                                    ) {
                                                        Text(proj.name)
                                                        Spacer(modifier = Modifier.width(8.dp))
                                                        if (proj.isTrained) {
                                                            Surface(
                                                                color = MaterialTheme.colorScheme.primaryContainer,
                                                                shape = RoundedCornerShape(4.dp)
                                                            ) {
                                                                Text(
                                                                    "Trained",
                                                                    style = MaterialTheme.typography.labelSmall,
                                                                    modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp)
                                                                )
                                                            }
                                                        } else {
                                                            Text(
                                                                "Untrained",
                                                                style = MaterialTheme.typography.labelSmall,
                                                                color = MaterialTheme.colorScheme.outline
                                                            )
                                                        }
                                                    }
                                                },
                                                onClick = {
                                                    selectedProjectForSorter = proj
                                                    showSorterProjectDropdown = false
                                                }
                                            )
                                        }
                                    }
                                }
                            }
                            if (allProjects.none { it.isTrained }) {
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = "💡 Tip: No trained model found in app projects. Switch to 'Direct File' above to use an exported .tflite, .onnx, or .json model directly!",
                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.5.sp),
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                        } else {
                            // Option B: Direct Exported Model File Input
                            val currentCustom = loadedCustomModel
                            if (currentCustom != null) {
                                Card(
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = CardDefaults.cardColors(
                                        containerColor = Color(0xFF2E7D32).copy(alpha = 0.08f)
                                    ),
                                    border = BorderStroke(1.dp, Color(0xFF4CAF50).copy(alpha = 0.6f)),
                                    shape = RoundedCornerShape(10.dp)
                                ) {
                                    Column(modifier = Modifier.padding(12.dp)) {
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.weight(1f)) {
                                                Box(
                                                    modifier = Modifier
                                                        .size(36.dp)
                                                        .clip(RoundedCornerShape(8.dp))
                                                        .background(Color(0xFF4CAF50).copy(alpha = 0.2f)),
                                                    contentAlignment = Alignment.Center
                                                ) {
                                                    Icon(
                                                        Icons.Default.FilePresent,
                                                        contentDescription = null,
                                                        tint = Color(0xFF2E7D32),
                                                        modifier = Modifier.size(20.dp)
                                                    )
                                                }
                                                Spacer(modifier = Modifier.width(10.dp))
                                                Column {
                                                    Text(
                                                        text = currentCustom.fileName,
                                                        style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Bold),
                                                        maxLines = 1,
                                                        overflow = TextOverflow.Ellipsis
                                                    )
                                                    Text(
                                                        text = currentCustom.formatName,
                                                        style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                                    )
                                                }
                                            }

                                            OutlinedButton(
                                                onClick = {
                                                    if (!batchSortState.isRunning) {
                                                        customModelPickerLauncher.launch(arrayOf("*/*"))
                                                    }
                                                },
                                                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                                                shape = RoundedCornerShape(8.dp),
                                                enabled = !batchSortState.isRunning
                                            ) {
                                                Text("Change", style = MaterialTheme.typography.labelSmall)
                                            }
                                        }

                                        Spacer(modifier = Modifier.height(8.dp))

                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                                        ) {
                                            Surface(
                                                color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.6f),
                                                shape = RoundedCornerShape(6.dp)
                                            ) {
                                                Text(
                                                    text = "🏷️ ${currentCustom.numClasses} Categories",
                                                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                                )
                                            }
                                            if (currentCustom.hasFeatureScaling) {
                                                Surface(
                                                    color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.6f),
                                                    shape = RoundedCornerShape(6.dp)
                                                ) {
                                                    Text(
                                                        text = "⚡ Scaler Active",
                                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                                    )
                                                }
                                            }
                                        }

                                        if (currentCustom.classLabels.isNotEmpty()) {
                                            Spacer(modifier = Modifier.height(4.dp))
                                            Text(
                                                text = "Categories: ${currentCustom.classLabels.joinToString(", ")}",
                                                style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.5.sp),
                                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                maxLines = 2,
                                                overflow = TextOverflow.Ellipsis
                                            )
                                        }
                                    }
                                }
                            } else {
                                Card(
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f)
                                    ),
                                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
                                    shape = RoundedCornerShape(10.dp)
                                ) {
                                    Column(
                                        modifier = Modifier.padding(14.dp),
                                        horizontalAlignment = Alignment.CenterHorizontally
                                    ) {
                                        Icon(
                                            Icons.Default.FileUpload,
                                            contentDescription = null,
                                            modifier = Modifier.size(32.dp),
                                            tint = MaterialTheme.colorScheme.primary
                                        )
                                        Spacer(modifier = Modifier.height(6.dp))
                                        Text(
                                            text = "Select Exported Model File",
                                            style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                                        )
                                        Text(
                                            text = "Load any exported .tflite, .onnx, .mlmodel, or .json model directly from phone storage without needing a trained project.",
                                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.5.sp),
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            modifier = Modifier.padding(top = 2.dp, bottom = 10.dp)
                                        )

                                        Button(
                                            onClick = {
                                                if (!batchSortState.isRunning) {
                                                    customModelPickerLauncher.launch(arrayOf("*/*"))
                                                }
                                            },
                                            modifier = Modifier.fillMaxWidth(),
                                            shape = RoundedCornerShape(8.dp),
                                            enabled = !batchSortState.isRunning
                                        ) {
                                            Icon(Icons.Default.FolderOpen, contentDescription = null, modifier = Modifier.size(18.dp))
                                            Spacer(modifier = Modifier.width(8.dp))
                                            Text("📂 Choose Model File (.tflite / .onnx / .json)")
                                        }

                                        if (customModelLoadError != null) {
                                            Spacer(modifier = Modifier.height(6.dp))
                                            Text(
                                                text = customModelLoadError!!,
                                                style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                                color = MaterialTheme.colorScheme.error
                                            )
                                        }
                                    }
                                }
                            }

                            // Quick pick from recently exported models on device
                            if (availableExportedModels.isNotEmpty()) {
                                Spacer(modifier = Modifier.height(10.dp))
                                Text(
                                    text = "Recently Exported Models (ডিভাইসে সংরক্ষিত মডেলসমূহ):",
                                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                    availableExportedModels.take(4).forEach { file ->
                                        Surface(
                                            onClick = {
                                                if (!batchSortState.isRunning) {
                                                    coroutineScope.launch(Dispatchers.IO) {
                                                        val exporter = ModelExporter(context)
                                                        val parsed = exporter.parseExportedModelFile(file)
                                                        withContext(Dispatchers.Main) {
                                                            if (parsed != null) {
                                                                loadedCustomModel = parsed
                                                                customModelLoadError = null
                                                                Toast.makeText(context, "Loaded: ${parsed.fileName}", Toast.LENGTH_SHORT).show()
                                                            } else {
                                                                Toast.makeText(context, "Failed to load ${file.name}", Toast.LENGTH_SHORT).show()
                                                            }
                                                        }
                                                    }
                                                }
                                            },
                                            modifier = Modifier.fillMaxWidth(),
                                            shape = RoundedCornerShape(8.dp),
                                            color = MaterialTheme.colorScheme.surface,
                                            border = BorderStroke(0.8.dp, MaterialTheme.colorScheme.outlineVariant)
                                        ) {
                                            Row(
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .padding(horizontal = 10.dp, vertical = 6.dp),
                                                horizontalArrangement = Arrangement.SpaceBetween,
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.weight(1f)) {
                                                    Icon(
                                                        Icons.Default.FilePresent,
                                                        contentDescription = null,
                                                        modifier = Modifier.size(16.dp),
                                                        tint = MaterialTheme.colorScheme.primary
                                                    )
                                                    Spacer(modifier = Modifier.width(6.dp))
                                                    Text(
                                                        text = file.name,
                                                        style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Medium),
                                                        maxLines = 1,
                                                        overflow = TextOverflow.Ellipsis
                                                    )
                                                }
                                                Text(
                                                    text = "Select",
                                                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                                    color = MaterialTheme.colorScheme.primary
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(14.dp))

                        // 2. Source Folder (Input Photos)
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "2. Source Folder (ইনপুট ফোল্ডার):",
                                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold)
                            )
                            if (sourceUriOrPath.startsWith("content://")) {
                                Surface(
                                    color = Color(0xFF2E7D32).copy(alpha = 0.15f),
                                    shape = RoundedCornerShape(12.dp)
                                ) {
                                    Text(
                                        text = "✓ Access Allowed",
                                        color = Color(0xFF4CAF50),
                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                                    )
                                }
                            }
                        }
                        Spacer(modifier = Modifier.height(6.dp))

                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(
                                containerColor = if (sourceUriOrPath.startsWith("content://"))
                                    MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.25f)
                                else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f)
                            ),
                            border = BorderStroke(
                                1.dp,
                                if (sourceUriOrPath.startsWith("content://"))
                                    MaterialTheme.colorScheme.primary.copy(alpha = 0.5f)
                                else MaterialTheme.colorScheme.outlineVariant
                            ),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Column(modifier = Modifier.padding(12.dp)) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Box(
                                        modifier = Modifier
                                            .size(38.dp)
                                            .clip(RoundedCornerShape(8.dp))
                                            .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.15f)),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Icon(
                                            Icons.Default.FolderOpen,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.primary,
                                            modifier = Modifier.size(22.dp)
                                        )
                                    }
                                    Spacer(modifier = Modifier.width(10.dp))
                                    Column(modifier = Modifier.weight(1f)) {
                                        Text(
                                            text = sourceDisplayName,
                                            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Bold),
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis
                                        )
                                        Text(
                                            text = if (sourceUriOrPath.startsWith("content://"))
                                                "Storage Access Framework (SAF Allowed)"
                                            else "App internal folder",
                                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                            color = if (sourceUriOrPath.startsWith("content://")) Color(0xFF4CAF50) else MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }

                                Spacer(modifier = Modifier.height(10.dp))

                                Button(
                                    onClick = { sourceFolderPickerLauncher.launch(null) },
                                    modifier = Modifier.fillMaxWidth(),
                                    enabled = !batchSortState.isRunning,
                                    shape = RoundedCornerShape(8.dp),
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = MaterialTheme.colorScheme.primary
                                    )
                                ) {
                                    Icon(Icons.Default.FolderOpen, contentDescription = null, modifier = Modifier.size(18.dp))
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(if (sourceUriOrPath.startsWith("content://")) "Change Source Folder (Allow SAF)" else "📂 Select Source Folder (Allow Access)")
                                }

                                Spacer(modifier = Modifier.height(6.dp))

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    SuggestionChip(
                                        onClick = {
                                            sourceUriOrPath = defaultSourceDir
                                            sourceDisplayName = "App Folder (Unsorted_Images)"
                                            sorterPrefs.edit()
                                                .putString("saved_source_uri", defaultSourceDir)
                                                .putString("saved_source_name", sourceDisplayName)
                                                .apply()
                                        },
                                        label = { Text("App Folder", style = MaterialTheme.typography.labelSmall) },
                                        enabled = !batchSortState.isRunning
                                    )
                                    Spacer(modifier = Modifier.weight(1f))
                                    OutlinedButton(
                                        onClick = {
                                            coroutineScope.launch {
                                                val proj = selectedProjectForSorter
                                                if (proj == null) {
                                                    Toast.makeText(context, "Select a model first!", Toast.LENGTH_SHORT).show()
                                                    return@launch
                                                }
                                                val srcDir = if (!sourceUriOrPath.startsWith("content://")) {
                                                    File(sourceUriOrPath)
                                                } else {
                                                    File(context.getExternalFilesDir(null), "Unsorted_Images").also {
                                                        sourceUriOrPath = it.absolutePath
                                                        sourceDisplayName = "App Folder (${it.name})"
                                                    }
                                                }
                                                val generated = batchSorter.generateDemoImages(srcDir, proj.id, viewModel.getRepository())
                                                Toast.makeText(context, "Generated $generated test images in '${srcDir.name}'!", Toast.LENGTH_LONG).show()
                                            }
                                        },
                                        enabled = !batchSortState.isRunning,
                                        shape = RoundedCornerShape(8.dp),
                                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp)
                                    ) {
                                        Icon(Icons.Default.AutoFixHigh, contentDescription = null, modifier = Modifier.size(14.dp))
                                        Spacer(modifier = Modifier.width(4.dp))
                                        Text("Add Demo Photos", style = MaterialTheme.typography.labelSmall)
                                    }
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(14.dp))

                        // 3. Destination Folder (Auto-Created Subfolders)
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "3. Destination Folder (আউটপুট ফোল্ডার):",
                                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold)
                            )
                            if (destUriOrPath.startsWith("content://")) {
                                Surface(
                                    color = Color(0xFF2E7D32).copy(alpha = 0.15f),
                                    shape = RoundedCornerShape(12.dp)
                                ) {
                                    Text(
                                        text = "✓ Access Allowed",
                                        color = Color(0xFF4CAF50),
                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                                    )
                                }
                            }
                        }
                        Spacer(modifier = Modifier.height(6.dp))

                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(
                                containerColor = if (destUriOrPath.startsWith("content://"))
                                    MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.25f)
                                else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f)
                            ),
                            border = BorderStroke(
                                1.dp,
                                if (destUriOrPath.startsWith("content://"))
                                    MaterialTheme.colorScheme.secondary.copy(alpha = 0.5f)
                                else MaterialTheme.colorScheme.outlineVariant
                            ),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Column(modifier = Modifier.padding(12.dp)) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Box(
                                        modifier = Modifier
                                            .size(38.dp)
                                            .clip(RoundedCornerShape(8.dp))
                                            .background(MaterialTheme.colorScheme.secondary.copy(alpha = 0.15f)),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Icon(
                                            Icons.Default.CreateNewFolder,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.secondary,
                                            modifier = Modifier.size(22.dp)
                                        )
                                    }
                                    Spacer(modifier = Modifier.width(10.dp))
                                    Column(modifier = Modifier.weight(1f)) {
                                        Text(
                                            text = destDisplayName,
                                            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Bold),
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis
                                        )
                                        Text(
                                            text = if (destUriOrPath.startsWith("content://"))
                                                "Storage Access Framework (Auto Subfolders Allowed)"
                                            else "App internal results folder",
                                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                            color = if (destUriOrPath.startsWith("content://")) Color(0xFF4CAF50) else MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }

                                Spacer(modifier = Modifier.height(10.dp))

                                Button(
                                    onClick = { destFolderPickerLauncher.launch(null) },
                                    modifier = Modifier.fillMaxWidth(),
                                    enabled = !batchSortState.isRunning,
                                    shape = RoundedCornerShape(8.dp),
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = MaterialTheme.colorScheme.secondary
                                    )
                                ) {
                                    Icon(Icons.Default.CreateNewFolder, contentDescription = null, modifier = Modifier.size(18.dp))
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(if (destUriOrPath.startsWith("content://")) "Change Destination Folder (Allow SAF)" else "📁 Select Destination Folder (Allow Access)")
                                }

                                Spacer(modifier = Modifier.height(6.dp))

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    SuggestionChip(
                                        onClick = {
                                            destUriOrPath = defaultDestDir
                                            destDisplayName = "App Results (Classified_Results)"
                                            sorterPrefs.edit()
                                                .putString("saved_dest_uri", defaultDestDir)
                                                .putString("saved_dest_name", destDisplayName)
                                                .apply()
                                        },
                                        label = { Text("App Results Folder", style = MaterialTheme.typography.labelSmall) },
                                        enabled = !batchSortState.isRunning
                                    )
                                }
                            }
                        }

                        // Smart Folder Creation Banner
                        Surface(
                            color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.4f),
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp)
                        ) {
                            Row(
                                modifier = Modifier.padding(10.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    Icons.Default.CheckCircle,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    text = "Smart Folder Handling: Existing category folders are reused without recreating; new category folders are created automatically.",
                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.5.sp),
                                    color = MaterialTheme.colorScheme.onPrimaryContainer
                                )
                            }
                        }

                        val isFaceProj = selectedProjectForSorter?.projectType == "FACE_RECOGNITION" ||
                                selectedProjectForSorter?.name?.contains("Face", ignoreCase = true) == true

                        if (isFaceProj) {
                            Spacer(modifier = Modifier.height(10.dp))

                            // 4. Detection & Face Filtering (শুধুমাত্র ফেস আইডি প্রজেক্টের জন্য প্রযোজ্য)
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = "4. Face Filter (ফেস ফিল্টারিং ও স্কিপ অপশন):",
                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold)
                                )
                                Surface(
                                    color = Color(0xFF0284C7).copy(alpha = 0.15f),
                                    shape = RoundedCornerShape(12.dp)
                                ) {
                                    Text(
                                        text = "👤 Face ID Project Only",
                                        color = Color(0xFF0284C7),
                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                                    )
                                }
                            }
                            Spacer(modifier = Modifier.height(6.dp))

                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(
                                    containerColor = if (skipIfNoFaceDetected)
                                        MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.25f)
                                    else
                                        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f)
                                ),
                                border = BorderStroke(
                                    1.dp,
                                    if (skipIfNoFaceDetected) MaterialTheme.colorScheme.primary.copy(alpha = 0.5f)
                                    else MaterialTheme.colorScheme.outlineVariant
                                ),
                                shape = RoundedCornerShape(12.dp)
                            ) {
                                Column(modifier = Modifier.padding(14.dp)) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Row(
                                            verticalAlignment = Alignment.CenterVertically,
                                            modifier = Modifier.weight(1f)
                                        ) {
                                            Icon(
                                                Icons.Default.Face,
                                                contentDescription = null,
                                                tint = if (skipIfNoFaceDetected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                                                modifier = Modifier.size(20.dp)
                                            )
                                            Spacer(modifier = Modifier.width(8.dp))
                                            Column {
                                                Text(
                                                    text = "Skip if No Face Detected",
                                                    style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                                                )
                                                Text(
                                                    text = "কোনো মুখ/ফেস ডিটেক্ট না হলে ছবি স্কিপ করুন",
                                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                                    color = MaterialTheme.colorScheme.primary
                                                )
                                            }
                                        }
                                        Switch(
                                            checked = skipIfNoFaceDetected,
                                            onCheckedChange = { checked ->
                                                if (!batchSortState.isRunning) {
                                                    skipIfNoFaceDetected = checked
                                                    sorterPrefs.edit().putBoolean("saved_skip_no_face", checked).apply()
                                                }
                                            },
                                            enabled = !batchSortState.isRunning
                                        )
                                    }

                                    Spacer(modifier = Modifier.height(6.dp))

                                    Text(
                                        text = "শুধুমাত্র ফেস আইডি প্রজেক্টের জন্য: যদি ছবিতে মানুষের মুখ ডিটেক্ট না হয় (যেমন ফুল, ফল, অবজেক্ট, সিনারি), তবে ছবিটি স্কিপ হবে। সাধারণ ইমেজ ট্রেনিং মডেলের ক্ষেত্রে কোনো ছবি স্কিপ হয় না।",
                                        style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(10.dp))

                        // 5. Large Dataset & Resource Optimization (10,000+ images RAM & Storage management)
                        Text(
                            text = "5. Resource & Scale Optimization (১০,০০০+ ছবি ও রেম কন্ট্রোল):",
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold)
                        )
                        Spacer(modifier = Modifier.height(6.dp))

                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f)
                            ),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Column(modifier = Modifier.padding(14.dp)) {
                                // Storage Mode Header
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        Icons.Default.Storage,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(18.dp)
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = "Storage Management Mode",
                                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                                    )
                                }
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = "Choose how photos are moved to category folders to protect mobile storage:",
                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.5.sp),
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )

                                Spacer(modifier = Modifier.height(10.dp))

                                // MOVE Option Card
                                Card(
                                    onClick = {
                                        if (!batchSortState.isRunning) {
                                            selectedFileAction = FileSortAction.MOVE
                                            sorterPrefs.edit().putString("saved_file_action", FileSortAction.MOVE.name).apply()
                                        }
                                    },
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = CardDefaults.cardColors(
                                        containerColor = if (selectedFileAction == FileSortAction.MOVE)
                                            MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.4f)
                                        else MaterialTheme.colorScheme.surface
                                    ),
                                    border = BorderStroke(
                                        1.dp,
                                        if (selectedFileAction == FileSortAction.MOVE) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant
                                    ),
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Row(
                                        modifier = Modifier.padding(10.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        RadioButton(
                                            selected = selectedFileAction == FileSortAction.MOVE,
                                            onClick = {
                                                if (!batchSortState.isRunning) {
                                                    selectedFileAction = FileSortAction.MOVE
                                                    sorterPrefs.edit().putString("saved_file_action", FileSortAction.MOVE.name).apply()
                                                }
                                            },
                                            enabled = !batchSortState.isRunning
                                        )
                                        Spacer(modifier = Modifier.width(6.dp))
                                        Column {
                                            Row(verticalAlignment = Alignment.CenterVertically) {
                                                Text(
                                                    text = "Move / Cut (Saves Storage)",
                                                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Bold)
                                                )
                                                Spacer(modifier = Modifier.width(6.dp))
                                                Surface(
                                                    color = Color(0xFF2E7D32).copy(alpha = 0.18f),
                                                    shape = RoundedCornerShape(4.dp)
                                                ) {
                                                    Text(
                                                        text = "RECOMMENDED FOR 10K+",
                                                        style = MaterialTheme.typography.labelSmall.copy(
                                                            color = Color(0xFF2E7D32),
                                                            fontSize = 9.sp,
                                                            fontWeight = FontWeight.Bold
                                                        ),
                                                        modifier = Modifier.padding(horizontal = 5.dp, vertical = 1.dp)
                                                    )
                                                }
                                            }
                                            Text(
                                                text = "Moves images into category folders. Uses 0 MB duplicate storage and saves phone disk space & battery.",
                                                style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                                color = MaterialTheme.colorScheme.onSurfaceVariant
                                            )
                                        }
                                    }
                                }

                                Spacer(modifier = Modifier.height(8.dp))

                                // COPY Option Card
                                Card(
                                    onClick = {
                                        if (!batchSortState.isRunning) {
                                            selectedFileAction = FileSortAction.COPY
                                            sorterPrefs.edit().putString("saved_file_action", FileSortAction.COPY.name).apply()
                                        }
                                    },
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = CardDefaults.cardColors(
                                        containerColor = if (selectedFileAction == FileSortAction.COPY)
                                            MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.4f)
                                        else MaterialTheme.colorScheme.surface
                                    ),
                                    border = BorderStroke(
                                        1.dp,
                                        if (selectedFileAction == FileSortAction.COPY) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant
                                    ),
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Row(
                                        modifier = Modifier.padding(10.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        RadioButton(
                                            selected = selectedFileAction == FileSortAction.COPY,
                                            onClick = {
                                                if (!batchSortState.isRunning) {
                                                    selectedFileAction = FileSortAction.COPY
                                                    sorterPrefs.edit().putString("saved_file_action", FileSortAction.COPY.name).apply()
                                                }
                                            },
                                            enabled = !batchSortState.isRunning
                                        )
                                        Spacer(modifier = Modifier.width(6.dp))
                                        Column {
                                            Text(
                                                text = "Copy (Keep Original Files)",
                                                style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Bold)
                                            )
                                            Text(
                                                text = "Original images remain in the source folder (requires 2x storage space).",
                                                style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                                color = MaterialTheme.colorScheme.onSurfaceVariant
                                            )
                                        }
                                    }
                                }

                                Spacer(modifier = Modifier.height(14.dp))

                                // Hardware & Thermal Pacing
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        Icons.Default.Speed,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(18.dp)
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = "Hardware Pacing & Thermal Control",
                                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                                    )
                                }
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = "Device Profile: ${hardwareProfile.description} (${hardwareProfile.tier.displayName})",
                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                    color = MaterialTheme.colorScheme.primary
                                )

                                Spacer(modifier = Modifier.height(8.dp))

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                                ) {
                                    SorterPacingMode.values().forEach { mode ->
                                        FilterChip(
                                            selected = selectedPacingMode == mode,
                                            onClick = {
                                                if (!batchSortState.isRunning) {
                                                    selectedPacingMode = mode
                                                    sorterPrefs.edit().putString("saved_pacing_mode", mode.name).apply()
                                                }
                                            },
                                            label = {
                                                Text(
                                                    text = when (mode) {
                                                        SorterPacingMode.AUTO -> "🟢 Auto-Adaptive"
                                                        SorterPacingMode.COOL_ECO -> "🍃 Eco / Cool"
                                                        SorterPacingMode.TURBO -> "⚡ Turbo"
                                                    },
                                                    style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.5.sp)
                                                )
                                            },
                                            enabled = !batchSortState.isRunning,
                                            modifier = Modifier.weight(1f)
                                        )
                                    }
                                }

                                Spacer(modifier = Modifier.height(6.dp))

                                Text(
                                    text = when (selectedPacingMode) {
                                        SorterPacingMode.AUTO -> "• Self-tunes thread count and pauses according to mobile CPU cores and RAM to prevent freezing."
                                        SorterPacingMode.COOL_ECO -> "• Inserts cool-down delays and aggressive RAM recycling. Ideal for 10,000+ files to keep battery cold."
                                        SorterPacingMode.TURBO -> "• Maximum multi-threaded throughput for flagship devices."
                                    },
                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 10.5.sp),
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(14.dp))

                        // 5. Action Controls (Start / Pause / Resume / Stop)
                        if (batchSortState.isRunning) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                // Pause / Resume Button
                                Button(
                                    onClick = {
                                        if (batchSortState.isPaused) {
                                            batchSorter.resumeSorting()
                                        } else {
                                            batchSorter.pauseSorting()
                                        }
                                    },
                                    modifier = Modifier.weight(1f),
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = if (batchSortState.isPaused)
                                            MaterialTheme.colorScheme.primary
                                        else
                                            Color(0xFFE65100)
                                    ),
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Icon(
                                        if (batchSortState.isPaused) Icons.Default.PlayArrow else Icons.Default.Pause,
                                        contentDescription = null,
                                        modifier = Modifier.size(18.dp)
                                    )
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text(if (batchSortState.isPaused) "Resume" else "Pause")
                                }

                                // Stop Button
                                Button(
                                    onClick = { batchSorter.stopSorting() },
                                    modifier = Modifier.weight(1f),
                                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Icon(Icons.Default.Stop, contentDescription = null, modifier = Modifier.size(18.dp))
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text("Stop")
                                }
                            }
                        } else {
                            Button(
                                onClick = {
                                    if (sorterModelSource == SorterModelSource.APP_PROJECT) {
                                        val proj = selectedProjectForSorter
                                        if (proj == null) {
                                            Toast.makeText(context, "Please select an AI model first!", Toast.LENGTH_SHORT).show()
                                            return@Button
                                        }
                                        if (!proj.isTrained) {
                                            Toast.makeText(context, "Selected model '${proj.name}' is not trained yet! Please train it first.", Toast.LENGTH_LONG).show()
                                            return@Button
                                        }
                                        batchSorter.startSorting(
                                            projectId = proj.id,
                                            projectName = proj.name,
                                            sourceUriOrPath = sourceUriOrPath,
                                            sourceDisplayName = sourceDisplayName,
                                            destinationUriOrPath = destUriOrPath,
                                            destinationDisplayName = destDisplayName,
                                            repository = viewModel.getRepository(),
                                            fileAction = selectedFileAction,
                                            pacingMode = selectedPacingMode,
                                            skipIfNoFaceDetected = skipIfNoFaceDetected
                                        )
                                    } else {
                                        val customModel = loadedCustomModel
                                        if (customModel == null) {
                                            Toast.makeText(context, "Please select or import a model file first!", Toast.LENGTH_SHORT).show()
                                            return@Button
                                        }
                                        batchSorter.startSortingWithCustomModel(
                                            customModel = customModel,
                                            sourceUriOrPath = sourceUriOrPath,
                                            sourceDisplayName = sourceDisplayName,
                                            destinationUriOrPath = destUriOrPath,
                                            destinationDisplayName = destDisplayName,
                                            fileAction = selectedFileAction,
                                            pacingMode = selectedPacingMode,
                                            skipIfNoFaceDetected = skipIfNoFaceDetected
                                        )
                                    }
                                },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Icon(Icons.Default.PlayArrow, contentDescription = null)
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Start Batch Auto-Sorting")
                            }
                        }

                        // 6. Progress & Live Resource Monitor Info
                        if (batchSortState.isRunning || batchSortState.totalImages > 0) {
                            Spacer(modifier = Modifier.height(14.dp))
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(
                                    containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f)
                                ),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Column(modifier = Modifier.padding(12.dp)) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Surface(
                                                color = when {
                                                    batchSortState.isPaused -> Color(0xFFE65100).copy(alpha = 0.18f)
                                                    batchSortState.isRunning -> Color(0xFF2E7D32).copy(alpha = 0.18f)
                                                    batchSortState.isCompleted -> MaterialTheme.colorScheme.primaryContainer
                                                    else -> MaterialTheme.colorScheme.surfaceContainerHighest
                                                },
                                                shape = RoundedCornerShape(6.dp)
                                            ) {
                                                Text(
                                                    text = when {
                                                        batchSortState.isPaused -> "⏸️ PAUSED"
                                                        batchSortState.isRunning -> "🟢 RUNNING"
                                                        batchSortState.isCompleted -> "🎉 COMPLETED"
                                                        else -> "IDLE"
                                                    },
                                                    style = MaterialTheme.typography.labelSmall.copy(
                                                        fontWeight = FontWeight.Bold,
                                                        color = when {
                                                            batchSortState.isPaused -> Color(0xFFE65100)
                                                            batchSortState.isRunning -> Color(0xFF2E7D32)
                                                            else -> MaterialTheme.colorScheme.primary
                                                        }
                                                    ),
                                                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                                )
                                            }
                                        }
                                        Text(
                                            text = "${batchSortState.currentImageIndex} / ${batchSortState.totalImages}",
                                            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Bold)
                                        )
                                    }

                                    Spacer(modifier = Modifier.height(8.dp))

                                    LinearProgressIndicator(
                                        progress = {
                                            if (batchSortState.totalImages > 0)
                                                batchSortState.currentImageIndex.toFloat() / batchSortState.totalImages.toFloat()
                                            else 0f
                                        },
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .height(8.dp)
                                            .clip(RoundedCornerShape(4.dp))
                                    )

                                    // Real-Time Resource Health Badges (RAM & Storage)
                                    Spacer(modifier = Modifier.height(10.dp))
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                                    ) {
                                        Surface(
                                            color = MaterialTheme.colorScheme.surface,
                                            shape = RoundedCornerShape(6.dp),
                                            border = BorderStroke(0.5.dp, MaterialTheme.colorScheme.outlineVariant),
                                            modifier = Modifier.weight(1f)
                                        ) {
                                            Row(
                                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                Icon(
                                                    Icons.Default.Memory,
                                                    contentDescription = null,
                                                    tint = MaterialTheme.colorScheme.primary,
                                                    modifier = Modifier.size(14.dp)
                                                )
                                                Spacer(modifier = Modifier.width(4.dp))
                                                Text(
                                                    text = "RAM: ${batchSortState.availableRamMb} MB",
                                                    style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp)
                                                )
                                            }
                                        }

                                        Surface(
                                            color = MaterialTheme.colorScheme.surface,
                                            shape = RoundedCornerShape(6.dp),
                                            border = BorderStroke(0.5.dp, MaterialTheme.colorScheme.outlineVariant),
                                            modifier = Modifier.weight(1f)
                                        ) {
                                            Row(
                                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                Icon(
                                                    Icons.Default.Storage,
                                                    contentDescription = null,
                                                    tint = MaterialTheme.colorScheme.secondary,
                                                    modifier = Modifier.size(14.dp)
                                                )
                                                Spacer(modifier = Modifier.width(4.dp))
                                                Text(
                                                    text = "Free: ${String.format(Locale.US, "%.1f", batchSortState.freeStorageGb)} GB",
                                                    style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp)
                                                )
                                            }
                                        }

                                        Surface(
                                            color = MaterialTheme.colorScheme.surface,
                                            shape = RoundedCornerShape(6.dp),
                                            border = BorderStroke(0.5.dp, MaterialTheme.colorScheme.outlineVariant),
                                            modifier = Modifier.weight(1f)
                                        ) {
                                            Row(
                                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                Icon(
                                                    Icons.Default.Tune,
                                                    contentDescription = null,
                                                    tint = MaterialTheme.colorScheme.tertiary,
                                                    modifier = Modifier.size(14.dp)
                                                )
                                                Spacer(modifier = Modifier.width(4.dp))
                                                Text(
                                                    text = batchSortState.pacingMode.displayName.take(8),
                                                    style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp),
                                                    maxLines = 1
                                                )
                                            }
                                        }
                                    }

                                    if (batchSortState.isPaused) {
                                        Spacer(modifier = Modifier.height(6.dp))
                                        Text(
                                            text = "⏸️ Paused: Device CPU and memory resources are released. Tap 'Resume' to continue.",
                                            style = MaterialTheme.typography.bodySmall.copy(
                                                fontSize = 11.sp,
                                                color = Color(0xFFE65100),
                                                fontWeight = FontWeight.Medium
                                            )
                                        )
                                    }

                                    if (batchSortState.skippedCount > 0) {
                                        Spacer(modifier = Modifier.height(6.dp))
                                        Surface(
                                            color = Color(0xFFE65100).copy(alpha = 0.12f),
                                            shape = RoundedCornerShape(6.dp)
                                        ) {
                                            Row(
                                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                Icon(
                                                    Icons.Default.Face,
                                                    contentDescription = null,
                                                    tint = Color(0xFFE65100),
                                                    modifier = Modifier.size(13.dp)
                                                )
                                                Spacer(modifier = Modifier.width(4.dp))
                                                Text(
                                                    text = "Skipped (No Face Detected): ${batchSortState.skippedCount} photos",
                                                    style = MaterialTheme.typography.labelSmall.copy(
                                                        fontSize = 10.5.sp,
                                                        color = Color(0xFFE65100),
                                                        fontWeight = FontWeight.SemiBold
                                                    )
                                                )
                                            }
                                        }
                                    }

                                    if (batchSortState.currentImageName.isNotBlank() && !batchSortState.isPaused) {
                                        Spacer(modifier = Modifier.height(6.dp))
                                        Text(
                                            text = "Scanning: ${batchSortState.currentImageName}",
                                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis
                                        )
                                    }

                                    if (batchSortState.detectedLabel.isNotBlank()) {
                                        Text(
                                            text = "➔ Identified: ${batchSortState.detectedLabel} (${(batchSortState.confidence * 100).toInt()}%)",
                                            style = MaterialTheme.typography.bodySmall.copy(
                                                fontSize = 11.sp,
                                                fontWeight = FontWeight.SemiBold
                                            ),
                                            color = MaterialTheme.colorScheme.primary
                                        )
                                    }
                                }
                            }
                        }

                        // 6. Sorted Summary
                        if (batchSortState.sortedSummary.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(14.dp))
                            Text(
                                text = "Sorted Output Categories (${batchSortState.sortedSummary.size} Folders Created):",
                                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                            )
                            Spacer(modifier = Modifier.height(6.dp))

                            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                batchSortState.sortedSummary.forEach { (category, count) ->
                                    Surface(
                                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                                        shape = RoundedCornerShape(8.dp),
                                        modifier = Modifier.fillMaxWidth()
                                    ) {
                                        Row(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .padding(horizontal = 12.dp, vertical = 8.dp),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Row(verticalAlignment = Alignment.CenterVertically) {
                                                Icon(
                                                    Icons.Default.Folder,
                                                    contentDescription = null,
                                                    tint = MaterialTheme.colorScheme.primary,
                                                    modifier = Modifier.size(18.dp)
                                                )
                                                Spacer(modifier = Modifier.width(8.dp))
                                                Text(
                                                    text = category,
                                                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                                                )
                                            }
                                            Surface(
                                                color = MaterialTheme.colorScheme.primary,
                                                shape = RoundedCornerShape(12.dp)
                                            ) {
                                                Text(
                                                    text = "$count ${if (count == 1) "photo" else "photos"}",
                                                    style = MaterialTheme.typography.labelSmall.copy(
                                                        color = Color.White,
                                                        fontWeight = FontWeight.Bold
                                                    ),
                                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                                                )
                                            }
                                        }
                                    }
                                }

                                if (batchSortState.skippedCount > 0) {
                                    Surface(
                                        color = Color(0xFFE65100).copy(alpha = 0.08f),
                                        shape = RoundedCornerShape(8.dp),
                                        border = BorderStroke(0.5.dp, Color(0xFFE65100).copy(alpha = 0.25f)),
                                        modifier = Modifier.fillMaxWidth()
                                    ) {
                                        Row(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .padding(horizontal = 12.dp, vertical = 8.dp),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Row(verticalAlignment = Alignment.CenterVertically) {
                                                Icon(
                                                    Icons.Default.Face,
                                                    contentDescription = null,
                                                    tint = Color(0xFFE65100),
                                                    modifier = Modifier.size(18.dp)
                                                )
                                                Spacer(modifier = Modifier.width(8.dp))
                                                Column {
                                                    Text(
                                                        text = "Skipped (No Face Detected)",
                                                        style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                                                    )
                                                    Text(
                                                        text = "কোনো মুখ ডিটেক্ট না হওয়ায় স্কিপ করা হয়েছে",
                                                        style = MaterialTheme.typography.bodySmall.copy(fontSize = 10.5.sp),
                                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                                    )
                                                }
                                            }
                                            Surface(
                                                color = Color(0xFFE65100),
                                                shape = RoundedCornerShape(12.dp)
                                            ) {
                                                Text(
                                                    text = "${batchSortState.skippedCount} skipped",
                                                    style = MaterialTheme.typography.labelSmall.copy(
                                                        color = Color.White,
                                                        fontWeight = FontWeight.Bold
                                                    ),
                                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // 7. Activity Logs
                        if (batchSortState.logs.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(14.dp))
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = "Sorting Logs:",
                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                                )
                                TextButton(onClick = { batchSorter.resetState() }) {
                                    Text("Clear", style = MaterialTheme.typography.labelSmall)
                                }
                            }
                            Surface(
                                color = MaterialTheme.colorScheme.surfaceContainerHighest,
                                shape = RoundedCornerShape(8.dp),
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(max = 140.dp)
                            ) {
                                LazyColumn(
                                    modifier = Modifier.padding(8.dp),
                                    reverseLayout = true
                                ) {
                                    items(batchSortState.logs.reversed()) { log ->
                                        Text(
                                            text = log,
                                            style = MaterialTheme.typography.bodySmall.copy(
                                                fontSize = 11.sp,
                                                fontFamily = FontFamily.Monospace
                                            ),
                                            modifier = Modifier.padding(vertical = 2.dp)
                                        )
                                    }
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(32.dp))
                    }
                }
            }
        }
    }
}
