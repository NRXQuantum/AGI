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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Universal, Memory-Safe Database & File Import Dialog for Text NLP Projects.
 *
 * Supports:
 * 1. ZIP Archives (.zip directly unpacked & streamed in memory without OOM).
 * 2. Uploading input.txt, CSV, TSV, JSON, JSONL files via Android Document Picker (500MB+ Safe).
 * 3. 28+ Major Benchmark Datasets catalog (Tiny Shakespeare, TinyStories, Alpaca, Dolly 15k,
 *    SQuAD 2.0, TriviaQA, TyDi QA, WizardLM, UltraChat, Bangla DailyDialog, Bangla SQuAD,
 *    Bangla Alpaca, Bangla2B, PubMed Cancer NLP, SMS Spam, etc.).
 * 4. Multi-Format Strategies (QA pairs, Dolly/Alpaca instruction tuning, Shakespeare dialogues, CSV/TSV, Section headers).
 * 5. Streaming line-by-line parser with zero OOM risk and live progress reporting.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TextDatabaseImportDialog(
    onDismiss: () -> Unit,
    onImportConfirmed: (parseResult: TextDatasetParser.ParseResult, replaceExisting: Boolean) -> Unit
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    var selectedTab by remember { mutableStateOf(0) } // 0: File / ZIP Upload, 1: Paste Text, 2: 28+ Benchmark Catalog
    var selectedStrategy by remember { mutableStateOf(TextDatasetParser.DatasetFormatStrategy.AUTO_DETECT) }

    var rawText by remember { mutableStateOf("") }
    var loadedFileName by remember { mutableStateOf<String?>(null) }
    var loadedFileSize by remember { mutableStateOf<String?>(null) }
    var isZipArchive by remember { mutableStateOf(false) }

    var isStreamingParsing by remember { mutableStateOf(false) }
    var streamProgressLines by remember { mutableLongStateOf(0L) }
    var streamProgressSamples by remember { mutableIntStateOf(0) }

    var streamingParseResult by remember { mutableStateOf<TextDatasetParser.ParseResult?>(null) }

    var replaceExisting by remember { mutableStateOf(false) }
    var minSampleThreshold by remember { mutableStateOf(1) }
    var excludedClasses by remember { mutableStateOf(setOf<String>()) }
    var isImporting by remember { mutableStateOf(false) }

    // Benchmark Catalog Filter State
    var catalogCategoryFilter by remember { mutableStateOf("All") }
    var catalogSearchQuery by remember { mutableStateOf("") }

    // Active parse result (either from streaming file/ZIP or from text area)
    val effectiveBaseResult = remember(streamingParseResult, rawText, selectedStrategy) {
        if (streamingParseResult != null) {
            streamingParseResult
        } else if (rawText.isNotBlank()) {
            TextDatasetParser.parse(rawText, selectedStrategy)
        } else {
            null
        }
    }

    // Filtered parse result based on UI selections
    val filteredParseResult = remember(effectiveBaseResult, minSampleThreshold, excludedClasses) {
        effectiveBaseResult
            ?.filterByMinSamples(minSampleThreshold)
            ?.filterBySelectedClasses(effectiveBaseResult.classCounts.keys - excludedClasses)
    }

    // Memory-safe streaming file picker launcher (.zip, .txt, .csv, .tsv, .json, .jsonl, etc.)
    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            coroutineScope.launch {
                isStreamingParsing = true
                streamProgressLines = 0L
                streamProgressSamples = 0
                streamingParseResult = null
                rawText = ""

                try {
                    val contentResolver = context.contentResolver
                    var displayName = "uploaded_file.txt"
                    var sizeBytes = 0L

                    contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                        val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                        val sizeIndex = cursor.getColumnIndex(android.provider.OpenableColumns.SIZE)
                        if (cursor.moveToFirst()) {
                            if (nameIndex >= 0) displayName = cursor.getString(nameIndex) ?: displayName
                            if (sizeIndex >= 0) sizeBytes = cursor.getLong(sizeIndex)
                        }
                    }

                    loadedFileName = displayName
                    isZipArchive = displayName.endsWith(".zip", ignoreCase = true)
                    loadedFileSize = when {
                        sizeBytes > 1024 * 1024 -> String.format(java.util.Locale.US, "%.1f MB", sizeBytes / (1024.0 * 1024.0))
                        sizeBytes > 1024 -> "${sizeBytes / 1024} KB"
                        else -> "$sizeBytes bytes"
                    }

                    // Parse stream in background without allocating full string in RAM
                    val result = withContext(Dispatchers.IO) {
                        contentResolver.openInputStream(uri)?.use { stream ->
                            if (isZipArchive) {
                                TextDatasetParser.parseZipStream(
                                    inputStream = stream,
                                    strategy = selectedStrategy,
                                    maxSampleCap = 40_000,
                                    onProgress = { lines, samples ->
                                        streamProgressLines = lines
                                        streamProgressSamples = samples
                                    }
                                )
                            } else {
                                TextDatasetParser.parseStream(
                                    inputStream = stream,
                                    strategy = selectedStrategy,
                                    maxSampleCap = 40_000,
                                    onProgress = { lines, samples ->
                                        streamProgressLines = lines
                                        streamProgressSamples = samples
                                    }
                                )
                            }
                        }
                    }

                    if (result != null && result.samples.isNotEmpty()) {
                        streamingParseResult = result
                        selectedTab = 0
                        Toast.makeText(
                            context,
                            "Parsed ${result.samples.size} samples across ${result.classCounts.size} classes!",
                            Toast.LENGTH_SHORT
                        ).show()
                    } else {
                        Toast.makeText(context, "No valid samples identified in file.", Toast.LENGTH_LONG).show()
                    }
                } catch (e: Throwable) {
                    Toast.makeText(context, "Error reading file: ${e.message}", Toast.LENGTH_LONG).show()
                } finally {
                    isStreamingParsing = false
                }
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
                .padding(10.dp),
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
                                text = "ZIP / 500MB Stream • 28+ Benchmarks • Multi-Format",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.primary
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

                Spacer(modifier = Modifier.height(10.dp))

                // Memory-Safe RAM & ZIP Protection Banner
                Surface(
                    color = Color(0xFF10B981).copy(alpha = 0.12f),
                    shape = RoundedCornerShape(8.dp),
                    border = BorderStroke(1.dp, Color(0xFF10B981).copy(alpha = 0.3f)),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Default.Memory,
                            contentDescription = null,
                            tint = Color(0xFF059669),
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "⚡ Out-of-Core Stream & ZIP Engine: 500MB+ safe • Zero RAM exhaustion",
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                            color = Color(0xFF065F46)
                        )
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Format Strategy Selector Chips
                Text(
                    text = "Parsing Format Strategy (ফরম্যাট সনাক্তকরণ পদ্ধতি):",
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.height(4.dp))
                LazyRow(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    items(TextDatasetParser.DatasetFormatStrategy.entries) { strat ->
                        FilterChip(
                            selected = selectedStrategy == strat,
                            onClick = {
                                selectedStrategy = strat
                                if (rawText.isNotBlank()) {
                                    streamingParseResult = null
                                }
                            },
                            label = {
                                Text(
                                    strat.title,
                                    fontSize = 11.sp,
                                    fontWeight = if (selectedStrategy == strat) FontWeight.Bold else FontWeight.Normal
                                )
                            },
                            leadingIcon = {
                                if (selectedStrategy == strat) {
                                    Icon(Icons.Default.Check, contentDescription = null, modifier = Modifier.size(12.dp))
                                }
                            }
                        )
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                // Primary Tabs: 0: File/ZIP Upload, 1: Text Paste/Edit, 2: 28+ Benchmark Catalog
                PrimaryTabRow(
                    selectedTabIndex = selectedTab,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Tab(
                        selected = selectedTab == 0,
                        onClick = { selectedTab = 0 },
                        text = { Text("File / ZIP (500MB)", maxLines = 1, fontSize = 12.sp) },
                        icon = { Icon(Icons.Default.FolderZip, contentDescription = null, modifier = Modifier.size(16.dp)) }
                    )
                    Tab(
                        selected = selectedTab == 1,
                        onClick = { selectedTab = 1 },
                        text = { Text("Paste Text", maxLines = 1, fontSize = 12.sp) },
                        icon = { Icon(Icons.Default.EditNote, contentDescription = null, modifier = Modifier.size(16.dp)) }
                    )
                    Tab(
                        selected = selectedTab == 2,
                        onClick = { selectedTab = 2 },
                        text = { Text("28+ Datasets (${TextDatasetParser.BENCHMARK_CATALOG.size})", maxLines = 1, fontSize = 12.sp) },
                        icon = { Icon(Icons.Default.Dataset, contentDescription = null, modifier = Modifier.size(16.dp)) }
                    )
                }

                Spacer(modifier = Modifier.height(10.dp))

                // Tab Viewport (Weight 1f)
                Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
                    when (selectedTab) {
                        0 -> {
                            // File / ZIP Upload Tab
                            Column(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .verticalScroll(rememberScrollState()),
                                verticalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                Surface(
                                    shape = RoundedCornerShape(14.dp),
                                    color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f),
                                    border = BorderStroke(1.5.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.6f)),
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable(enabled = !isStreamingParsing) {
                                            filePickerLauncher.launch("*/*")
                                        }
                                ) {
                                    Column(
                                        modifier = Modifier.padding(20.dp),
                                        horizontalAlignment = Alignment.CenterHorizontally
                                    ) {
                                        Icon(
                                            if (isZipArchive) Icons.Default.FolderZip else Icons.Default.FolderOpen,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.primary,
                                            modifier = Modifier.size(42.dp)
                                        )
                                        Spacer(modifier = Modifier.height(8.dp))
                                        Text(
                                            text = if (loadedFileName != null) "File: $loadedFileName" else "Select Database File or ZIP Archive",
                                            style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                                            color = MaterialTheme.colorScheme.onSurface
                                        )
                                        Text(
                                            text = if (loadedFileSize != null) "Size: $loadedFileSize • Tap to change file" else "Supports .zip, .txt, .csv, .tsv, .json, .jsonl (up to 500MB+)",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }

                                if (isStreamingParsing) {
                                    Surface(
                                        color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f),
                                        shape = RoundedCornerShape(10.dp),
                                        modifier = Modifier.fillMaxWidth()
                                    ) {
                                        Row(
                                            modifier = Modifier.padding(12.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.5.dp)
                                            Spacer(modifier = Modifier.width(12.dp))
                                            Column {
                                                Text(
                                                    if (isZipArchive) "Unpacking & Streaming ZIP Dataset..." else "Streaming & Parsing Database...",
                                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                                                )
                                                Text(
                                                    "$streamProgressLines lines scanned • $streamProgressSamples samples extracted",
                                                    style = MaterialTheme.typography.labelSmall,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                            }
                                        }
                                    }
                                }

                                // Results Preview
                                if (effectiveBaseResult != null) {
                                    DatasetPreviewCard(
                                        result = effectiveBaseResult,
                                        filteredResult = filteredParseResult,
                                        minSampleThreshold = minSampleThreshold,
                                        onMinSampleChange = { minSampleThreshold = it },
                                        excludedClasses = excludedClasses,
                                        onToggleClass = { className ->
                                            excludedClasses = if (excludedClasses.contains(className)) {
                                                excludedClasses - className
                                            } else {
                                                excludedClasses + className
                                            }
                                        }
                                    )
                                }
                            }
                        }

                        1 -> {
                            // Text Paste Tab
                            Column(modifier = Modifier.fillMaxSize()) {
                                OutlinedTextField(
                                    value = rawText,
                                    onValueChange = {
                                        rawText = it
                                        streamingParseResult = null
                                    },
                                    placeholder = { Text("Paste raw dataset here (e.g. CSV lines, Dolly JSONL, Shakespeare dialogue)...") },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .weight(1f),
                                    textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace)
                                )

                                if (effectiveBaseResult != null) {
                                    Spacer(modifier = Modifier.height(8.dp))
                                    DatasetPreviewCard(
                                        result = effectiveBaseResult,
                                        filteredResult = filteredParseResult,
                                        minSampleThreshold = minSampleThreshold,
                                        onMinSampleChange = { minSampleThreshold = it },
                                        excludedClasses = excludedClasses,
                                        onToggleClass = { className ->
                                            excludedClasses = if (excludedClasses.contains(className)) {
                                                excludedClasses - className
                                            } else {
                                                excludedClasses + className
                                            }
                                        }
                                    )
                                }
                            }
                        }

                        2 -> {
                            // 28+ Major NLP Benchmark Catalog Tab
                            Column(
                                modifier = Modifier.fillMaxSize(),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                // Search Bar
                                OutlinedTextField(
                                    value = catalogSearchQuery,
                                    onValueChange = { catalogSearchQuery = it },
                                    placeholder = { Text("Search 28+ datasets (e.g. Alpaca, Shakespeare, Bangla SQuAD)...", fontSize = 12.sp) },
                                    modifier = Modifier.fillMaxWidth(),
                                    leadingIcon = { Icon(Icons.Default.Search, contentDescription = null, modifier = Modifier.size(18.dp)) },
                                    trailingIcon = {
                                        if (catalogSearchQuery.isNotEmpty()) {
                                            IconButton(onClick = { catalogSearchQuery = "" }) {
                                                Icon(Icons.Default.Clear, contentDescription = "Clear", modifier = Modifier.size(16.dp))
                                            }
                                        }
                                    },
                                    maxLines = 1,
                                    shape = RoundedCornerShape(12.dp)
                                )

                                // Category Filter Chips
                                val categories = listOf("All", "Bangla NLP", "Instruction & LLM", "QA & Knowledge", "Literature & Drama", "Dialogue & Chat", "Specialized")
                                LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                    items(categories) { cat ->
                                        FilterChip(
                                            selected = catalogCategoryFilter == cat,
                                            onClick = { catalogCategoryFilter = cat },
                                            label = { Text(cat, fontSize = 11.sp, fontWeight = if (catalogCategoryFilter == cat) FontWeight.Bold else FontWeight.Normal) }
                                        )
                                    }
                                }

                                val filteredCatalog = remember(catalogCategoryFilter, catalogSearchQuery) {
                                    TextDatasetParser.BENCHMARK_CATALOG.filter { item ->
                                        val matchesCat = (catalogCategoryFilter == "All" || item.category == catalogCategoryFilter)
                                        val matchesQuery = catalogSearchQuery.isBlank() ||
                                                item.name.contains(catalogSearchQuery, ignoreCase = true) ||
                                                item.nameBn.contains(catalogSearchQuery, ignoreCase = true) ||
                                                item.description.contains(catalogSearchQuery, ignoreCase = true)
                                        matchesCat && matchesQuery
                                    }
                                }

                                LazyColumn(
                                    modifier = Modifier.fillMaxSize(),
                                    verticalArrangement = Arrangement.spacedBy(8.dp),
                                    contentPadding = PaddingValues(vertical = 4.dp)
                                ) {
                                    items(filteredCatalog, key = { it.id }) { item ->
                                        BenchmarkCatalogItemCard(
                                            item = item,
                                            onLoad = {
                                                rawText = item.sampleSnippet
                                                selectedStrategy = item.defaultStrategy
                                                streamingParseResult = null
                                                selectedTab = 1
                                                Toast.makeText(context, "Loaded ${item.name} sample!", Toast.LENGTH_SHORT).show()
                                            }
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                // Replace Existing Toggle
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Checkbox(
                        checked = replaceExisting,
                        onCheckedChange = { replaceExisting = it }
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "Replace existing project classes & samples with this database",
                        style = MaterialTheme.typography.bodySmall
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Bottom Actions
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedButton(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f),
                        enabled = !isImporting
                    ) {
                        Text("Cancel")
                    }

                    val canImport = filteredParseResult != null &&
                            filteredParseResult.samples.isNotEmpty() &&
                            filteredParseResult.classCounts.size >= 2 &&
                            !isStreamingParsing

                    Button(
                        onClick = {
                            filteredParseResult?.let { res ->
                                isImporting = true
                                onImportConfirmed(res, replaceExisting)
                            }
                        },
                        enabled = canImport && !isImporting,
                        modifier = Modifier.weight(1.5f)
                    ) {
                        if (isImporting) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(16.dp),
                                strokeWidth = 2.dp,
                                color = MaterialTheme.colorScheme.onPrimary
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Importing Database...")
                        } else {
                            Icon(Icons.Default.DownloadDone, contentDescription = null, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Import ${filteredParseResult?.samples?.size ?: 0} Samples")
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun BenchmarkCatalogItemCard(
    item: TextDatasetParser.BenchmarkDatasetInfo,
    onLoad: () -> Unit
) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.7f)),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = item.name,
                            style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                            color = MaterialTheme.colorScheme.onSurface
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Surface(
                            shape = RoundedCornerShape(4.dp),
                            color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.6f)
                        ) {
                            Text(
                                text = item.category,
                                style = MaterialTheme.typography.labelSmall.copy(fontSize = 9.sp, fontWeight = FontWeight.Bold),
                                color = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.padding(horizontal = 4.dp, vertical = 1.dp)
                            )
                        }
                    }
                    Text(
                        text = item.nameBn,
                        style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp, fontWeight = FontWeight.Medium),
                        color = MaterialTheme.colorScheme.primary
                    )
                }

                Button(
                    onClick = onLoad,
                    contentPadding = PaddingValues(horizontal = 10.dp, vertical = 2.dp),
                    modifier = Modifier.height(32.dp),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text("Load Sample", fontSize = 11.sp)
                }
            }

            Spacer(modifier = Modifier.height(4.dp))

            Text(
                text = item.description,
                style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                lineHeight = 15.sp
            )

            Spacer(modifier = Modifier.height(6.dp))

            Surface(
                color = MaterialTheme.colorScheme.surface,
                shape = RoundedCornerShape(6.dp),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.4f)),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = item.sampleSnippet,
                    style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace, fontSize = 9.5.sp),
                    modifier = Modifier.padding(6.dp),
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
fun DatasetPreviewCard(
    result: TextDatasetParser.ParseResult,
    filteredResult: TextDatasetParser.ParseResult?,
    minSampleThreshold: Int,
    onMinSampleChange: (Int) -> Unit,
    excludedClasses: Set<String>,
    onToggleClass: (String) -> Unit
) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.3f)),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = result.formatName,
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    color = MaterialTheme.colorScheme.primary
                )

                Surface(
                    color = MaterialTheme.colorScheme.primary,
                    shape = RoundedCornerShape(6.dp)
                ) {
                    Text(
                        text = "${filteredResult?.samples?.size ?: 0} Samples • ${filteredResult?.classCounts?.size ?: 0} Classes",
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onPrimary,
                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Discovered Classes chips
            Text(
                "Discovered Classes (Tap to toggle):",
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(4.dp))
            LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                items(result.classCounts.entries.toList()) { (className, count) ->
                    val isIncluded = !excludedClasses.contains(className)
                    FilterChip(
                        selected = isIncluded,
                        onClick = { onToggleClass(className) },
                        label = {
                            Text(
                                text = "$className ($count)",
                                fontSize = 11.sp,
                                fontWeight = if (isIncluded) FontWeight.Bold else FontWeight.Normal
                            )
                        }
                    )
                }
            }
        }
    }
}
