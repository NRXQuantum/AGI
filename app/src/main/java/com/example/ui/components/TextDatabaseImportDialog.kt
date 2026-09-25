package com.example.ui.components

import android.net.Uri
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.example.ml.TextDatasetParser
import com.example.ml.TextModelEngine

/**
 * Comprehensive Database & File Import Dialog for Text NLP Projects.
 *
 * Supports:
 * 1. Uploading input.txt, CSV, TSV, JSON files via Android System Document/File Picker
 * 2. Pasting or editing raw file content
 * 3. 1-Tap Preloading Shakespeare Coriolanus dialogue sample (user's input.txt structure)
 * 4. Auto-detecting file format (Shakespeare dialogue, CSV, JSON, section headers)
 * 5. Interactive class selection and sample preview before inserting into the local Room database
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TextDatabaseImportDialog(
    onDismiss: () -> Unit,
    onImportConfirmed: (parseResult: TextDatasetParser.ParseResult, replaceExisting: Boolean) -> Unit
) {
    val context = LocalContext.current

    var selectedTab by remember { mutableStateOf(0) } // 0: File Upload, 1: Text Paste/Edit, 2: Shakespeare Sample
    var rawText by remember { mutableStateOf("") }
    var loadedFileName by remember { mutableStateOf<String?>(null) }
    var loadedFileSize by remember { mutableStateOf<String?>(null) }

    var replaceExisting by remember { mutableStateOf(false) }
    var minSampleThreshold by remember { mutableStateOf(1) }
    var excludedClasses by remember { mutableStateOf(setOf<String>()) }
    var isImporting by remember { mutableStateOf(false) }

    // Parse result derived from rawText
    val baseParseResult = remember(rawText) {
        if (rawText.isBlank()) null else TextDatasetParser.parse(rawText)
    }

    // Filtered parse result based on UI selections
    val filteredParseResult = remember(baseParseResult, minSampleThreshold, excludedClasses) {
        baseParseResult
            ?.filterByMinSamples(minSampleThreshold)
            ?.filterBySelectedClasses(baseParseResult.classCounts.keys - excludedClasses)
    }

    // File picker launcher for .txt, .csv, .tsv, .json or any text file
    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            try {
                val contentResolver = context.contentResolver
                // Retrieve display name
                var displayName = "uploaded_file.txt"
                contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                    val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                    val sizeIndex = cursor.getColumnIndex(android.provider.OpenableColumns.SIZE)
                    if (cursor.moveToFirst()) {
                        if (nameIndex >= 0) displayName = cursor.getString(nameIndex) ?: displayName
                        if (sizeIndex >= 0) {
                            val sizeBytes = cursor.getLong(sizeIndex)
                            loadedFileSize = if (sizeBytes > 1024) "${sizeBytes / 1024} KB" else "$sizeBytes bytes"
                        }
                    }
                }

                val text = contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() } ?: ""
                if (text.isNotBlank()) {
                    rawText = text
                    loadedFileName = displayName
                    selectedTab = 0 // Keep on file view
                    Toast.makeText(context, "Loaded $displayName (${text.lines().size} lines)", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(context, "Selected file is empty.", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Toast.makeText(context, "Failed to read file: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
            }
        }
    }

    Dialog(
        onDismissRequest = { if (!isImporting) onDismiss() },
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Surface(
            modifier = Modifier
                .fillMaxSize()
                .padding(12.dp),
            shape = RoundedCornerShape(20.dp),
            color = MaterialTheme.colorScheme.surface,
            tonalElevation = 6.dp
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp)
            ) {
                // Top Header
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Surface(
                            color = MaterialTheme.colorScheme.primaryContainer,
                            shape = CircleShape,
                            modifier = Modifier.size(40.dp)
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Icon(
                                    Icons.Default.CloudUpload,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.onPrimaryContainer,
                                    modifier = Modifier.size(22.dp)
                                )
                            }
                        }
                        Spacer(modifier = Modifier.width(12.dp))
                        Column {
                            Text(
                                text = "Database & File Import Studio",
                                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                            )
                            Text(
                                text = "Auto-parse input.txt, dialogues, CSV, JSON",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }

                    IconButton(
                        onClick = onDismiss,
                        enabled = !isImporting
                    ) {
                        Icon(Icons.Default.Close, contentDescription = "Close")
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                // Source Tabs
                PrimaryTabRow(
                    selectedTabIndex = selectedTab,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Tab(
                        selected = selectedTab == 0,
                        onClick = { selectedTab = 0 },
                        text = {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Default.UploadFile, contentDescription = null, modifier = Modifier.size(16.dp))
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Upload File", fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                            }
                        }
                    )
                    Tab(
                        selected = selectedTab == 1,
                        onClick = { selectedTab = 1 },
                        text = {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Default.EditNote, contentDescription = null, modifier = Modifier.size(16.dp))
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Paste Text", fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                            }
                        }
                    )
                    Tab(
                        selected = selectedTab == 2,
                        onClick = {
                            selectedTab = 2
                            rawText = TextDatasetParser.SHAKESPEARE_CORIOLANUS_SAMPLE
                            loadedFileName = "input.txt (Shakespeare Sample)"
                            loadedFileSize = "3.2 KB"
                        },
                        text = {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text("🎭", fontSize = 14.sp)
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Shakespeare input.txt", fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                            }
                        }
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                // Content Area based on Tab
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                ) {
                    when (selectedTab) {
                        0 -> {
                            // File Upload View
                            Column(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .verticalScroll(rememberScrollState()),
                                verticalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                // Upload Trigger Box
                                Card(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable { filePickerLauncher.launch("*/*") },
                                    shape = RoundedCornerShape(16.dp),
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.35f)
                                    ),
                                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.5f))
                                ) {
                                    Column(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(20.dp),
                                        horizontalAlignment = Alignment.CenterHorizontally
                                    ) {
                                        Icon(
                                            Icons.Default.DriveFolderUpload,
                                            contentDescription = null,
                                            modifier = Modifier.size(48.dp),
                                            tint = MaterialTheme.colorScheme.primary
                                        )
                                        Spacer(modifier = Modifier.height(8.dp))
                                        Text(
                                            text = if (loadedFileName != null) "Selected: $loadedFileName" else "Tap to Choose Dataset File",
                                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                                            color = MaterialTheme.colorScheme.primary
                                        )
                                        if (loadedFileSize != null) {
                                            Text(
                                                text = "Size: $loadedFileSize • ${rawText.lines().size} lines",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant
                                            )
                                        }
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Text(
                                            text = "Supports: input.txt, .txt, .csv, .tsv, .json",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                        Spacer(modifier = Modifier.height(12.dp))
                                        Button(
                                            onClick = { filePickerLauncher.launch("*/*") },
                                            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp)
                                        ) {
                                            Icon(Icons.Default.FolderOpen, contentDescription = null, modifier = Modifier.size(16.dp))
                                            Spacer(modifier = Modifier.width(6.dp))
                                            Text(if (loadedFileName != null) "Choose Another File" else "Browse Files")
                                        }
                                    }
                                }

                                // Quick 1-tap Shakespeare button
                                Card(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable {
                                            rawText = TextDatasetParser.SHAKESPEARE_CORIOLANUS_SAMPLE
                                            loadedFileName = "input.txt (Shakespeare Sample)"
                                            loadedFileSize = "3.2 KB"
                                        },
                                    shape = RoundedCornerShape(12.dp),
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                                    )
                                ) {
                                    Row(
                                        modifier = Modifier.padding(14.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text("🎭", fontSize = 28.sp)
                                        Spacer(modifier = Modifier.width(12.dp))
                                        Column(modifier = Modifier.weight(1f)) {
                                            Text(
                                                text = "Load Your 'input.txt' Shakespeare Sample",
                                                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                                            )
                                            Text(
                                                text = "Includes First Citizen, All, Second Citizen & MENENIUS dialogues",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant
                                            )
                                        }
                                        FilledTonalButton(
                                            onClick = {
                                                rawText = TextDatasetParser.SHAKESPEARE_CORIOLANUS_SAMPLE
                                                loadedFileName = "input.txt (Shakespeare Sample)"
                                                loadedFileSize = "3.2 KB"
                                            },
                                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                                            modifier = Modifier.height(34.dp)
                                        ) {
                                            Text("Load Sample", fontSize = 11.sp)
                                        }
                                    }
                                }

                                // Parser Results Section
                                if (filteredParseResult != null) {
                                    ParserResultsInspector(
                                        parseResult = filteredParseResult,
                                        baseClassCounts = baseParseResult?.classCounts ?: emptyMap(),
                                        excludedClasses = excludedClasses,
                                        onToggleClass = { cls ->
                                            excludedClasses = if (cls in excludedClasses) excludedClasses - cls else excludedClasses + cls
                                        },
                                        minThreshold = minSampleThreshold,
                                        onThresholdChange = { minSampleThreshold = it },
                                        replaceExisting = replaceExisting,
                                        onReplaceToggle = { replaceExisting = it }
                                    )
                                }
                            }
                        }

                        1 -> {
                            // Raw Paste View
                            Column(modifier = Modifier.fillMaxSize()) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = "Paste entire file content (input.txt, CSV, etc.):",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                    if (rawText.isNotBlank()) {
                                        TextButton(onClick = { rawText = ""; loadedFileName = null }) {
                                            Text("Clear", color = MaterialTheme.colorScheme.error, fontSize = 12.sp)
                                        }
                                    }
                                }

                                OutlinedTextField(
                                    value = rawText,
                                    onValueChange = {
                                        rawText = it
                                        loadedFileName = "Pasted Text"
                                    },
                                    placeholder = {
                                        Text(
                                            "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nMENENIUS:\nWhat work's, my countrymen...",
                                            fontFamily = FontFamily.Monospace,
                                            fontSize = 12.sp
                                        )
                                    },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .weight(1f),
                                    textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace)
                                )

                                Spacer(modifier = Modifier.height(8.dp))

                                if (filteredParseResult != null) {
                                    Surface(
                                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                                        shape = RoundedCornerShape(10.dp),
                                        modifier = Modifier.fillMaxWidth()
                                    ) {
                                        Row(
                                            modifier = Modifier.padding(10.dp),
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.SpaceBetween
                                        ) {
                                            Text(
                                                text = "Format: ${filteredParseResult.formatName}",
                                                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                                color = MaterialTheme.colorScheme.primary
                                            )
                                            Text(
                                                text = "${filteredParseResult.samples.size} samples across ${filteredParseResult.classCounts.size} classes",
                                                style = MaterialTheme.typography.labelSmall,
                                                fontWeight = FontWeight.SemiBold
                                            )
                                        }
                                    }
                                }
                            }
                        }

                        2 -> {
                            // Shakespeare Preload Preview
                            Column(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .verticalScroll(rememberScrollState()),
                                verticalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                Surface(
                                    color = Color(0xFF10B981).copy(alpha = 0.12f),
                                    shape = RoundedCornerShape(12.dp),
                                    border = BorderStroke(1.dp, Color(0xFF10B981).copy(alpha = 0.4f))
                                ) {
                                    Row(modifier = Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                                        Text("🎭", fontSize = 24.sp)
                                        Spacer(modifier = Modifier.width(10.dp))
                                        Column {
                                            Text(
                                                text = "Shakespeare: Coriolanus (Act I, Scene I)",
                                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold, color = Color(0xFF047857))
                                            )
                                            Text(
                                                text = "Matches your provided input.txt format with 4 speaker roles and real dialogues.",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant
                                            )
                                        }
                                    }
                                }

                                if (filteredParseResult != null) {
                                    ParserResultsInspector(
                                        parseResult = filteredParseResult,
                                        baseClassCounts = baseParseResult?.classCounts ?: emptyMap(),
                                        excludedClasses = excludedClasses,
                                        onToggleClass = { cls ->
                                            excludedClasses = if (cls in excludedClasses) excludedClasses - cls else excludedClasses + cls
                                        },
                                        minThreshold = minSampleThreshold,
                                        onThresholdChange = { minSampleThreshold = it },
                                        replaceExisting = replaceExisting,
                                        onReplaceToggle = { replaceExisting = it }
                                    )
                                }
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                // Bottom Action Bar
                HorizontalDivider()
                Spacer(modifier = Modifier.height(10.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    TextButton(
                        onClick = onDismiss,
                        enabled = !isImporting
                    ) {
                        Text("Cancel")
                    }

                    val validSampleCount = filteredParseResult?.samples?.size ?: 0
                    val validClassCount = filteredParseResult?.classCounts?.size ?: 0

                    Button(
                        onClick = {
                            if (filteredParseResult != null && validSampleCount > 0) {
                                isImporting = true
                                onImportConfirmed(filteredParseResult, replaceExisting)
                            }
                        },
                        enabled = validSampleCount > 0 && validClassCount >= 2 && !isImporting,
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        if (isImporting) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(16.dp),
                                strokeWidth = 2.dp,
                                color = MaterialTheme.colorScheme.onPrimary
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Importing...")
                        } else {
                            Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                if (validSampleCount > 0) "Import $validSampleCount Samples ($validClassCount Classes)"
                                else "Select Valid Dataset"
                            )
                        }
                    }
                }
            }
        }
    }
}

/**
 * Inspector sub-component for parsed database preview
 */
@Composable
private fun ParserResultsInspector(
    parseResult: TextDatasetParser.ParseResult,
    baseClassCounts: Map<String, Int>,
    excludedClasses: Set<String>,
    onToggleClass: (String) -> Unit,
    minThreshold: Int,
    onThresholdChange: (Int) -> Unit,
    replaceExisting: Boolean,
    onReplaceToggle: (Boolean) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f)
        )
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            // Format & Metrics Badges
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Surface(
                    color = MaterialTheme.colorScheme.primaryContainer,
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(
                        text = "Format: ${parseResult.formatName}",
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }

                Surface(
                    color = Color(0xFF10B981).copy(alpha = 0.15f),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(
                        text = "⚡ ${parseResult.samples.size} Samples • ${parseResult.classCounts.size} Classes",
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold, color = Color(0xFF047857)),
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }

            // Detected Classes Chip Row
            Text(
                text = "Detected Classes / Roles (Tap to include/exclude):",
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
            )

            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(baseClassCounts.entries.toList()) { (className, count) ->
                    val isIncluded = className !in excludedClasses
                    FilterChip(
                        selected = isIncluded,
                        onClick = { onToggleClass(className) },
                        label = {
                            Text("$className ($count)")
                        },
                        leadingIcon = {
                            if (isIncluded) {
                                Icon(Icons.Default.Check, contentDescription = null, modifier = Modifier.size(14.dp))
                            }
                        }
                    )
                }
            }

            // Samples Preview (First 5 samples)
            Text(
                text = "Sample Preview (First ${minOf(parseResult.samples.size, 5)}):",
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                parseResult.samples.take(5).forEach { sample ->
                    Surface(
                        color = MaterialTheme.colorScheme.surface,
                        shape = RoundedCornerShape(8.dp),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.4f)),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(8.dp)) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Surface(
                                    color = MaterialTheme.colorScheme.secondaryContainer,
                                    shape = RoundedCornerShape(4.dp)
                                ) {
                                    Text(
                                        text = sample.className,
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontWeight = FontWeight.Bold,
                                            fontSize = 10.sp
                                        ),
                                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                    )
                                }
                                Spacer(modifier = Modifier.width(8.dp))
                                val tokenEstimate = sample.text.split("\\s+".toRegex()).size
                                Text(
                                    text = "~$tokenEstimate words",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    fontSize = 10.sp
                                )
                            }
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = sample.text,
                                style = MaterialTheme.typography.bodySmall,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    }
                }
            }

            // Import Destination Choice
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onReplaceToggle(!replaceExisting) }
                    .padding(vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Checkbox(
                    checked = replaceExisting,
                    onCheckedChange = { onReplaceToggle(it) }
                )
                Spacer(modifier = Modifier.width(6.dp))
                Column {
                    Text(
                        text = "Replace Existing Dataset",
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                    )
                    Text(
                        text = if (replaceExisting) "Old classes & samples will be wiped before importing" else "New classes & samples will be merged with current dataset",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}
