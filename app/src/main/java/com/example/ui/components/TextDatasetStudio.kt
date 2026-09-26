package com.example.ui.components

import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
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
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.data.db.ClassificationClassEntity
import com.example.data.db.ProjectEntity
import com.example.data.db.TextSampleEntity
import com.example.ml.TextModelEngine
import com.example.ui.viewmodel.ProjectViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TextDatasetStudio(
    viewModel: ProjectViewModel,
    project: ProjectEntity,
    classes: List<ClassificationClassEntity>,
    onNavigateToTrain: () -> Unit,
    onNavigateBack: (() -> Unit)? = null
) {
    val context = LocalContext.current
    val clipboardManager = LocalClipboardManager.current

    var selectedClassId by remember(classes) {
        mutableStateOf(classes.firstOrNull()?.id)
    }

    val selectedClass = classes.find { it.id == selectedClassId }

    val textSamplesFlow = remember(selectedClassId) {
        if (selectedClassId != null) {
            viewModel.getTextSamplesForClass(selectedClassId!!)
        } else {
            kotlinx.coroutines.flow.flowOf(emptyList())
        }
    }
    val samples by textSamplesFlow.collectAsState(initial = emptyList())
    val totalSamplesCount by viewModel.projectTotalTextSamples.collectAsState()

    var showAddSampleDialog by remember { mutableStateOf(false) }
    var showMultiLineDialog by remember { mutableStateOf(false) }
    var showBenchmarkDialog by remember { mutableStateOf(false) }
    var showDatabaseImportDialog by remember { mutableStateOf(false) }
    var showAddClassDialog by remember { mutableStateOf(false) }
    var classToEdit by remember { mutableStateOf<ClassificationClassEntity?>(null) }
    var sampleToEdit by remember { mutableStateOf<TextSampleEntity?>(null) }

    var isSelectionMode by remember { mutableStateOf(false) }
    val selectedSampleIds = remember { mutableStateListOf<Long>() }

    BackHandler {
        if (isSelectionMode) {
            isSelectionMode = false
            selectedSampleIds.clear()
        } else if (onNavigateBack != null) {
            onNavigateBack()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                text = project.name,
                                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Surface(
                                color = Color(0xFF10B981).copy(alpha = 0.15f),
                                shape = RoundedCornerShape(6.dp)
                            ) {
                                Text(
                                    text = "TEXT NLP",
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        fontSize = 9.sp,
                                        fontWeight = FontWeight.Bold,
                                        color = Color(0xFF059669)
                                    ),
                                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                )
                            }
                        }
                        Text(
                            text = "${classes.size} Classes • $totalSamplesCount Total Samples",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                navigationIcon = {
                    if (onNavigateBack != null) {
                        IconButton(onClick = onNavigateBack) {
                            Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                        }
                    }
                },
                actions = {
                    FilledTonalButton(
                        onClick = { showDatabaseImportDialog = true },
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                        modifier = Modifier.height(34.dp)
                    ) {
                        Icon(Icons.Default.CloudUpload, contentDescription = null, modifier = Modifier.size(15.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Upload File / DB", fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                    }
                    Spacer(modifier = Modifier.width(6.dp))
                    FilledTonalButton(
                        onClick = { showBenchmarkDialog = true },
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                        modifier = Modifier.height(34.dp)
                    ) {
                        Icon(Icons.Default.AutoAwesome, contentDescription = null, modifier = Modifier.size(15.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Preload", fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                    }
                }
            )
        },
        bottomBar = {
            Surface(
                tonalElevation = 3.dp,
                shadowElevation = 8.dp,
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 12.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = if (classes.size < 2) "Needs 2+ classes" else if (totalSamplesCount == 0) "Add samples to train" else "Dataset Ready",
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                            color = if (classes.size >= 2 && totalSamplesCount >= 2) Color(0xFF059669) else MaterialTheme.colorScheme.error
                        )
                        Text(
                            text = "$totalSamplesCount text samples in project",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    Button(
                        onClick = onNavigateToTrain,
                        enabled = classes.size >= 2 && totalSamplesCount >= 2,
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier.testTag("nav_to_train_btn")
                    ) {
                        Icon(Icons.Default.Psychology, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Train Model")
                    }
                }
            }
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            // 1. Classes Horizontal Selector
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                LazyRow(
                    modifier = Modifier.weight(1f),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    items(classes) { cls ->
                        val isSelected = cls.id == selectedClassId
                        val parsedColor = try {
                            Color(android.graphics.Color.parseColor(cls.colorHex))
                        } catch (_: Exception) {
                            MaterialTheme.colorScheme.primary
                        }

                        FilterChip(
                            selected = isSelected,
                            onClick = { selectedClassId = cls.id },
                            label = {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Box(
                                        modifier = Modifier
                                            .size(10.dp)
                                            .clip(CircleShape)
                                            .background(parsedColor)
                                    )
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text(cls.className, fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal)
                                }
                            },
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = parsedColor.copy(alpha = 0.2f),
                                selectedLabelColor = MaterialTheme.colorScheme.onSurface
                            ),
                            border = FilterChipDefaults.filterChipBorder(
                                enabled = true,
                                selected = isSelected,
                                selectedBorderColor = parsedColor
                            )
                        )
                    }
                }

                Spacer(modifier = Modifier.width(8.dp))
                IconButton(
                    onClick = { showAddClassDialog = true },
                    modifier = Modifier
                        .size(36.dp)
                        .background(MaterialTheme.colorScheme.primaryContainer, CircleShape)
                ) {
                    Icon(
                        Icons.Default.Add,
                        contentDescription = "Add Class",
                        tint = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier.size(20.dp)
                    )
                }
            }

            // Quick Dataset Upload Banner
            Surface(
                color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.28f),
                shape = RoundedCornerShape(10.dp),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.25f)),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 4.dp)
                    .clickable { showDatabaseImportDialog = true }
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        Icons.Default.CloudUpload,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Import Database / File (input.txt, CSV, JSON)",
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                            color = MaterialTheme.colorScheme.primary
                        )
                        Text(
                            text = "Auto-parse Shakespeare plays, drama scripts, transcripts or datasets",
                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    FilledTonalButton(
                        onClick = { showDatabaseImportDialog = true },
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                        modifier = Modifier.height(30.dp)
                    ) {
                        Text("Upload", fontSize = 11.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }

            if (classes.size > 20) {
                Surface(
                    color = Color(0xFFF59E0B).copy(alpha = 0.15f),
                    shape = RoundedCornerShape(10.dp),
                    border = BorderStroke(1.dp, Color(0xFFF59E0B).copy(alpha = 0.5f)),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 4.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Default.AutoFixHigh,
                            contentDescription = null,
                            tint = Color(0xFFD97706),
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(10.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "High Class Count (${classes.size} Classes)",
                                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                                color = Color(0xFFB45309)
                            )
                            Text(
                                text = "Auto-cluster into 10 Topic Categories for 100x faster training & accuracy",
                                style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Button(
                            onClick = {
                                viewModel.autoClusterProjectClasses { msg ->
                                    Toast.makeText(context, msg, Toast.LENGTH_LONG).show()
                                }
                            },
                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                            modifier = Modifier.height(30.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFD97706))
                        ) {
                            Text("Auto-Cluster", fontSize = 11.sp, fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }

            // 2. Class Action Bar
            if (selectedClass != null) {
                Surface(
                    color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                text = "${selectedClass.className} (${samples.size})",
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                            )
                            IconButton(
                                onClick = { classToEdit = selectedClass },
                                modifier = Modifier.size(28.dp)
                            ) {
                                Icon(Icons.Default.Edit, contentDescription = "Edit Class", modifier = Modifier.size(15.dp))
                            }
                        }

                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            FilledTonalButton(
                                onClick = { showAddSampleDialog = true },
                                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                                modifier = Modifier.height(32.dp)
                            ) {
                                Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(15.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Add Text", fontSize = 12.sp)
                            }

                            OutlinedButton(
                                onClick = { showMultiLineDialog = true },
                                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                                modifier = Modifier.height(32.dp)
                            ) {
                                Icon(Icons.Default.FormatListBulleted, contentDescription = null, modifier = Modifier.size(14.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Paste Lines", fontSize = 12.sp)
                            }
                        }
                    }
                }
            }

            // 3. Samples List
            if (selectedClass == null) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("No class selected or created yet.")
                }
            } else if (samples.isEmpty()) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(20.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.verticalScroll(rememberScrollState())
                    ) {
                        Icon(
                            Icons.Outlined.TextFields,
                            contentDescription = null,
                            modifier = Modifier.size(52.dp),
                            tint = MaterialTheme.colorScheme.primary.copy(alpha = 0.6f)
                        )
                        Spacer(modifier = Modifier.height(10.dp))
                        Text(
                            text = "No samples in '${selectedClass.className}'",
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            text = "Have an input.txt file, drama script, or dataset? Upload it directly or add samples manually.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = androidx.compose.ui.text.style.TextAlign.Center
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Button(
                            onClick = { showDatabaseImportDialog = true },
                            shape = RoundedCornerShape(12.dp),
                            modifier = Modifier.fillMaxWidth(0.9f)
                        ) {
                            Icon(Icons.Default.CloudUpload, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Upload Database File (input.txt, CSV, JSON)")
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        FilledTonalButton(
                            onClick = { showDatabaseImportDialog = true },
                            shape = RoundedCornerShape(12.dp),
                            modifier = Modifier.fillMaxWidth(0.9f)
                        ) {
                            Text("🎭", fontSize = 16.sp)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Load Shakespeare Coriolanus Sample")
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.fillMaxWidth(0.9f)
                        ) {
                            OutlinedButton(
                                onClick = { showAddSampleDialog = true },
                                modifier = Modifier.weight(1f)
                            ) {
                                Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(16.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Add Text", fontSize = 12.sp)
                            }
                            OutlinedButton(
                                onClick = { showBenchmarkDialog = true },
                                modifier = Modifier.weight(1f)
                            ) {
                                Icon(Icons.Default.AutoAwesome, contentDescription = null, modifier = Modifier.size(16.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Preload", fontSize = 12.sp)
                            }
                        }
                    }
                }
            } else {
                LazyColumn(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(samples, key = { it.id }) { sample ->
                        TextSampleCard(
                            sample = sample,
                            onCopy = {
                                clipboardManager.setText(AnnotatedString(sample.textContent))
                                Toast.makeText(context, "Copied to clipboard", Toast.LENGTH_SHORT).show()
                            },
                            onEdit = { sampleToEdit = sample },
                            onDelete = { viewModel.deleteTextSample(sample.id) }
                        )
                    }
                }
            }
        }
    }

    // Dialogs
    if (showAddSampleDialog && selectedClassId != null) {
        AddEditSampleDialog(
            title = "Add Text Sample to '${selectedClass?.className}'",
            initialText = "",
            onDismiss = { showAddSampleDialog = false },
            onSave = { text ->
                viewModel.addTextSample(selectedClassId!!, text)
                showAddSampleDialog = false
            }
        )
    }

    if (sampleToEdit != null) {
        AddEditSampleDialog(
            title = "Edit Text Sample",
            initialText = sampleToEdit!!.textContent,
            onDismiss = { sampleToEdit = null },
            onSave = { newText ->
                viewModel.updateTextSample(sampleToEdit!!.id, sampleToEdit!!.classId, newText)
                sampleToEdit = null
            }
        )
    }

    if (showMultiLineDialog && selectedClassId != null) {
        MultiLinePasteDialog(
            className = selectedClass?.className ?: "",
            onDismiss = { showMultiLineDialog = false },
            onImport = { lines ->
                viewModel.addTextSamplesBatch(selectedClassId!!, lines)
                showMultiLineDialog = false
                Toast.makeText(context, "Added ${lines.size} samples", Toast.LENGTH_SHORT).show()
            }
        )
    }

    if (showDatabaseImportDialog) {
        TextDatabaseImportDialog(
            onDismiss = { showDatabaseImportDialog = false },
            onImportConfirmed = { parseResult, replaceExisting ->
                viewModel.importParsedTextDataset(parseResult, replaceExisting) { message ->
                    showDatabaseImportDialog = false
                    Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                }
            }
        )
    }

    if (showBenchmarkDialog) {
        PreloadedDatasetDialog(
            onDismiss = { showBenchmarkDialog = false },
            onSelect = { dataset ->
                viewModel.loadBenchmarkDataset(dataset)
                showBenchmarkDialog = false
                Toast.makeText(context, "Loaded dataset '${dataset.name}'!", Toast.LENGTH_SHORT).show()
            }
        )
    }

    if (showAddClassDialog) {
        ClassDialog(
            title = "Add New Class",
            initialName = "",
            initialColor = "#10B981",
            onDismiss = { showAddClassDialog = false },
            onSave = { name, color ->
                viewModel.addClass(name, color)
                showAddClassDialog = false
            }
        )
    }

    if (classToEdit != null) {
        ClassDialog(
            title = "Edit Class",
            initialName = classToEdit!!.className,
            initialColor = classToEdit!!.colorHex,
            isEdit = true,
            onDismiss = { classToEdit = null },
            onSave = { name, color ->
                viewModel.updateClass(classToEdit!!.id, name, color)
                classToEdit = null
            },
            onDelete = {
                viewModel.deleteClass(classToEdit!!.id)
                classToEdit = null
            }
        )
    }
}

@Composable
fun TextSampleCard(
    sample: TextSampleEntity,
    onCopy: () -> Unit,
    onEdit: () -> Unit,
    onDelete: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text = sample.textContent,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )

            Spacer(modifier = Modifier.height(10.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Token budget indicator badge
                val tokenCount = sample.tokenCount
                val tokenBadgeColor = when {
                    tokenCount > 200 -> Color(0xFFEF4444)
                    tokenCount > 100 -> Color(0xFFF59E0B)
                    else -> Color(0xFF10B981)
                }

                Surface(
                    color = tokenBadgeColor.copy(alpha = 0.12f),
                    shape = RoundedCornerShape(6.dp)
                ) {
                    Text(
                        text = "⚡ $tokenCount Tokens",
                        style = MaterialTheme.typography.labelSmall.copy(
                            fontWeight = FontWeight.Bold,
                            color = tokenBadgeColor,
                            fontSize = 11.sp
                        ),
                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                    )
                }

                Row {
                    IconButton(onClick = onCopy, modifier = Modifier.size(30.dp)) {
                        Icon(Icons.Outlined.ContentCopy, contentDescription = "Copy", modifier = Modifier.size(16.dp))
                    }
                    IconButton(onClick = onEdit, modifier = Modifier.size(30.dp)) {
                        Icon(Icons.Outlined.Edit, contentDescription = "Edit", modifier = Modifier.size(16.dp))
                    }
                    IconButton(onClick = onDelete, modifier = Modifier.size(30.dp)) {
                        Icon(Icons.Outlined.Delete, contentDescription = "Delete", tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(16.dp))
                    }
                }
            }
        }
    }
}

@Composable
fun AddEditSampleDialog(
    title: String,
    initialText: String,
    onDismiss: () -> Unit,
    onSave: (String) -> Unit
) {
    var text by remember { mutableStateOf(initialText) }
    val tokenRes = remember(text) { TextModelEngine.tokenize(text) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title, style = MaterialTheme.typography.titleMedium) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(
                    value = text,
                    onValueChange = { text = it },
                    label = { Text("Enter sample text") },
                    placeholder = { Text("Type sentences, reviews, questions...") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(140.dp),
                    maxLines = 6
                )

                // Live Token Budget Visualizer
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Tokens: ${tokenRes.tokenCount} / ${tokenRes.tokenLimit}",
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold),
                        color = if (tokenRes.isTruncated) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    if (tokenRes.isTruncated) {
                        Text(
                            text = "⚠️ Exceeds limit (Auto-budgeted)",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                }

                LinearProgressIndicator(
                    progress = { (tokenRes.tokenCount.toFloat() / tokenRes.tokenLimit.toFloat()).coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth().height(4.dp),
                    color = if (tokenRes.isTruncated) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
                )
            }
        },
        confirmButton = {
            Button(
                onClick = { onSave(text.trim()) },
                enabled = text.isNotBlank()
            ) {
                Text("Save Sample")
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
fun MultiLinePasteDialog(
    className: String,
    onDismiss: () -> Unit,
    onImport: (List<String>) -> Unit
) {
    var rawText by remember { mutableStateOf("") }
    val lines = remember(rawText) {
        rawText.lines().map { it.trim() }.filter { it.isNotBlank() }
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Paste Multiple Samples ($className)", style = MaterialTheme.typography.titleMedium) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                    text = "Paste text where each line represents one training sample.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                OutlinedTextField(
                    value = rawText,
                    onValueChange = { rawText = it },
                    label = { Text("Paste lines here") },
                    placeholder = { Text("Line 1 sample\nLine 2 sample\nLine 3 sample...") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(180.dp),
                    maxLines = 10
                )

                Text(
                    text = "${lines.size} samples detected",
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.primary
                )
            }
        },
        confirmButton = {
            Button(
                onClick = { onImport(lines) },
                enabled = lines.isNotEmpty()
            ) {
                Text("Import ${lines.size} Samples")
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
fun PreloadedDatasetDialog(
    onDismiss: () -> Unit,
    onSelect: (TextModelEngine.PreloadedDataset) -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Preload Benchmark Dataset", style = MaterialTheme.typography.titleMedium) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(
                    text = "Select a ready-to-train high-accuracy NLP dataset to seed classes & samples in 1 tap:",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                LazyColumn(
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth().heightIn(max = 350.dp)
                ) {
                    items(TextModelEngine.BENCHMARK_DATASETS) { dataset ->
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable { onSelect(dataset) },
                            shape = RoundedCornerShape(10.dp),
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                            )
                        ) {
                            Column(modifier = Modifier.padding(12.dp)) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Text(dataset.icon, fontSize = 20.sp)
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = dataset.name,
                                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                                    )
                                }
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = dataset.description,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Spacer(modifier = Modifier.height(6.dp))
                                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                    dataset.classes.forEach { c ->
                                        Surface(
                                            color = MaterialTheme.colorScheme.surface,
                                            shape = RoundedCornerShape(4.dp)
                                        ) {
                                            Text(
                                                text = "${c.name} (${c.samples.size})",
                                                style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp),
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
        },
        confirmButton = {},
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}

@Composable
fun ClassDialog(
    title: String,
    initialName: String,
    initialColor: String,
    isEdit: Boolean = false,
    onDismiss: () -> Unit,
    onSave: (String, String) -> Unit,
    onDelete: (() -> Unit)? = null
) {
    var name by remember { mutableStateOf(initialName) }
    var selectedColor by remember { mutableStateOf(initialColor) }

    val colors = listOf(
        "#10B981", "#3B82F6", "#EF4444", "#F59E0B",
        "#8B5CF6", "#EC4899", "#06B6D4", "#64748B"
    )

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = name,
                    onValueChange = { name = it },
                    label = { Text("Class Name") },
                    placeholder = { Text("e.g. Positive, Urgent, Bug") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                Text("Label Color", style = MaterialTheme.typography.labelMedium)
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    colors.forEach { hex ->
                        val parsed = Color(android.graphics.Color.parseColor(hex))
                        Box(
                            modifier = Modifier
                                .size(32.dp)
                                .clip(CircleShape)
                                .background(parsed)
                                .clickable { selectedColor = hex }
                                .padding(4.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            if (selectedColor == hex) {
                                Icon(
                                    Icons.Default.Check,
                                    contentDescription = null,
                                    tint = Color.White,
                                    modifier = Modifier.size(16.dp)
                                )
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            Button(
                onClick = { onSave(name.trim(), selectedColor) },
                enabled = name.isNotBlank()
            ) {
                Text("Save")
            }
        },
        dismissButton = {
            Row {
                if (isEdit && onDelete != null) {
                    TextButton(onClick = onDelete) {
                        Text("Delete", color = MaterialTheme.colorScheme.error)
                    }
                }
                TextButton(onClick = onDismiss) {
                    Text("Cancel")
                }
            }
        }
    )
}
