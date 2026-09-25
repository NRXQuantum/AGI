package com.example.ui.components

import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
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
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.data.db.ClassificationClassEntity
import com.example.data.db.ProjectEntity
import com.example.data.db.TrainedModelEntity
import com.example.ml.TextModelEngine
import com.example.ui.viewmodel.ProjectViewModel
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TextInferenceStudio(
    viewModel: ProjectViewModel,
    project: ProjectEntity,
    latestModel: TrainedModelEntity?,
    classes: List<ClassificationClassEntity>,
    onNavigateToTrain: () -> Unit,
    onNavigateToExport: () -> Unit,
    onNavigateBack: (() -> Unit)? = null
) {
    val context = LocalContext.current
    var inputText by remember { mutableStateOf("") }
    var livePredictEnabled by remember { mutableStateOf(true) }

    val prediction by viewModel.textInferenceResult.collectAsState()
    val isRunning by viewModel.isInferenceRunning.collectAsState()

    var showFeedbackDialog by remember { mutableStateOf(false) }

    BackHandler {
        if (onNavigateBack != null) onNavigateBack()
    }

    // Live predict as user types
    LaunchedEffect(inputText, livePredictEnabled, latestModel) {
        if (livePredictEnabled && inputText.isNotBlank() && latestModel != null) {
            viewModel.predictText(inputText)
        } else if (inputText.isBlank()) {
            viewModel.clearTextInference()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = "${project.name} • Test",
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                        )
                        Text(
                            text = "Live On-Device NLP Inference Studio",
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
                    IconButton(onClick = onNavigateToExport) {
                        Icon(Icons.Outlined.FileDownload, contentDescription = "Export")
                    }
                }
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // 1. Model Status Banner
            if (latestModel == null) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(14.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Default.Warning, contentDescription = null, tint = MaterialTheme.colorScheme.error)
                        Spacer(modifier = Modifier.width(10.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                "Model Not Trained Yet",
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                            Text(
                                "Train this project on your dataset to activate live inference.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                        }
                        Button(onClick = onNavigateToTrain) {
                            Text("Train Now")
                        }
                    }
                }
            } else {
                Surface(
                    color = Color(0xFF10B981).copy(alpha = 0.12f),
                    shape = RoundedCornerShape(12.dp),
                    border = BorderStroke(1.dp, Color(0xFF10B981).copy(alpha = 0.3f)),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.CheckCircle, contentDescription = null, tint = Color(0xFF059669), modifier = Modifier.size(20.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Column {
                                Text(
                                    "Active Neural Model (${String.format(Locale.US, "%.1f%%", latestModel.accuracy * 100f)} Accuracy)",
                                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                                    color = Color(0xFF065F46)
                                )
                                Text(
                                    "3-Expert MoE • 128-Dim Embeddings • On-Device CPU",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color(0xFF047857)
                                )
                            }
                        }

                        TextButton(onClick = onNavigateToTrain) {
                            Text("Retrain")
                        }
                    }
                }
            }

            // 2. Input Text Area
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(14.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f))
            ) {
                Column(modifier = Modifier.padding(14.dp)) {
                    Text(
                        "Test Text Input",
                        style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                    )
                    Spacer(modifier = Modifier.height(8.dp))

                    OutlinedTextField(
                        value = inputText,
                        onValueChange = { inputText = it },
                        placeholder = { Text("Type or paste any text sentence, review, or message to classify in real-time...") },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(130.dp)
                            .testTag("text_inference_input"),
                        maxLines = 5
                    )

                    Spacer(modifier = Modifier.height(10.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Switch(
                                checked = livePredictEnabled,
                                onCheckedChange = { livePredictEnabled = it },
                                modifier = Modifier.graphicsLayer { scaleX = 0.85f; scaleY = 0.85f }
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Live Predict", style = MaterialTheme.typography.labelSmall)
                        }

                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            if (inputText.isNotBlank()) {
                                TextButton(onClick = { inputText = "" }) {
                                    Text("Clear")
                                }
                            }
                            Button(
                                onClick = { viewModel.predictText(inputText) },
                                enabled = inputText.isNotBlank() && latestModel != null && !isRunning
                            ) {
                                Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(16.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Classify")
                            }
                        }
                    }

                    // Quick Sample Chips
                    Spacer(modifier = Modifier.height(10.dp))
                    Text("Quick Test Samples:", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Spacer(modifier = Modifier.height(6.dp))
                    LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        val samplePrompts = listOf(
                            "This product is absolutely wonderful, works like a charm!",
                            "Worst app update ever, keeps crashing and losing data.",
                            "Meeting confirmed for 3 PM tomorrow in main room.",
                            "URGENT: Click here to verify your account or get banned!",
                            "I need to change my credit card payment method for subscription."
                        )
                        items(samplePrompts) { prompt ->
                            SuggestionChip(
                                onClick = { inputText = prompt },
                                label = {
                                    Text(
                                        text = prompt,
                                        maxLines = 1,
                                        fontSize = 11.sp
                                    )
                                }
                            )
                        }
                    }
                }
            }

            // 3. Inference Results
            if (isRunning) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(24.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(modifier = Modifier.size(32.dp), strokeWidth = 3.dp)
                        Spacer(modifier = Modifier.height(8.dp))
                        Text("Extracting MoE features & running neural pass...", style = MaterialTheme.typography.labelMedium)
                    }
                }
            } else {
                val currentPred = prediction
                if (currentPred != null) {
                    val pred = currentPred
                    val winningClass = classes.find { it.className == pred.classLabel }
                val parsedColor = try {
                    Color(android.graphics.Color.parseColor(winningClass?.colorHex ?: "#10B981"))
                } catch (_: Exception) {
                    Color(0xFF10B981)
                }

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                    border = BorderStroke(1.5.dp, parsedColor.copy(alpha = 0.6f))
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                "Predicted Class",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )

                            Surface(
                                color = parsedColor.copy(alpha = 0.15f),
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Row(
                                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Box(modifier = Modifier.size(8.dp).clip(CircleShape).background(parsedColor))
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text(
                                        pred.classLabel,
                                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                                        color = parsedColor
                                    )
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(8.dp))

                        // Large Confidence Meter
                        Row(
                            verticalAlignment = Alignment.Bottom,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = "${String.format(Locale.US, "%.1f", pred.confidence * 100f)}%",
                                style = MaterialTheme.typography.displaySmall.copy(fontWeight = FontWeight.Bold),
                                color = parsedColor
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Confidence Score",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.padding(bottom = 6.dp)
                            )
                        }

                        Spacer(modifier = Modifier.height(12.dp))

                        // Efficiency & Telemetry Badges
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Surface(
                                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                                shape = RoundedCornerShape(8.dp),
                                modifier = Modifier.weight(1f)
                            ) {
                                Column(modifier = Modifier.padding(8.dp)) {
                                    Text("⚡ Latency", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text("${pred.inferenceTimeMs} ms", style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold))
                                }
                            }

                            Surface(
                                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                                shape = RoundedCornerShape(8.dp),
                                modifier = Modifier.weight(1f)
                            ) {
                                Column(modifier = Modifier.padding(8.dp)) {
                                    Text("🔋 Energy", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text("< 0.001 mWh", style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold))
                                }
                            }

                            Surface(
                                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                                shape = RoundedCornerShape(8.dp),
                                modifier = Modifier.weight(1f)
                            ) {
                                Column(modifier = Modifier.padding(8.dp)) {
                                    Text("🪙 Tokens", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text("${pred.tokenCount} / ${pred.tokenLimit}", style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold))
                                }
                            }
                        }

                        // Explainable AI (Salient Keywords)
                        if (pred.salientTokens.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(14.dp))
                            Text(
                                "Key Influencing Words (Explainable AI):",
                                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Spacer(modifier = Modifier.height(6.dp))
                            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                for (tokenPair in pred.salientTokens) {
                                    val word = tokenPair.first
                                    Surface(
                                        color = parsedColor.copy(alpha = 0.15f),
                                        shape = RoundedCornerShape(6.dp)
                                    ) {
                                        Text(
                                            text = "“$word”",
                                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold, color = parsedColor),
                                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                        )
                                    }
                                }
                            }
                        }

                        // Full Probabilities Breakdown
                        Spacer(modifier = Modifier.height(14.dp))
                        Text(
                            "Class Probability Distribution:",
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Spacer(modifier = Modifier.height(8.dp))

                        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            for (classProb in pred.allProbabilities) {
                                val targetClass = classes.find { it.className == classProb.classLabel }
                                val cColor = try {
                                    Color(android.graphics.Color.parseColor(targetClass?.colorHex ?: "#3B82F6"))
                                } catch (_: Exception) {
                                    Color(0xFF3B82F6)
                                }

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        classProb.classLabel,
                                        style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Medium),
                                        modifier = Modifier.width(100.dp),
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                    LinearProgressIndicator(
                                        progress = { classProb.probability.coerceIn(0f, 1f) },
                                        modifier = Modifier
                                            .weight(1f)
                                            .height(8.dp)
                                            .clip(RoundedCornerShape(4.dp)),
                                        color = cColor,
                                        trackColor = MaterialTheme.colorScheme.surfaceVariant
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        "${String.format(Locale.US, "%.1f", classProb.probability * 100f)}%",
                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                        modifier = Modifier.width(42.dp)
                                    )
                                }
                            }
                        }

                        // Add to Dataset Button
                        Spacer(modifier = Modifier.height(14.dp))
                        OutlinedButton(
                            onClick = { showFeedbackDialog = true },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Default.AddComment, contentDescription = null, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Add this text to a class to improve dataset")
                        }
                    }
                }
            }
        }
    }
    }

    // Feedback Dialog: Add tested text directly to a class
    if (showFeedbackDialog) {
        var selectedFeedbackClassId by remember { mutableStateOf(classes.firstOrNull()?.id) }
        AlertDialog(
            onDismissRequest = { showFeedbackDialog = false },
            title = { Text("Save to Dataset") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(
                        "Add this test sample to reinforce your model:",
                        style = MaterialTheme.typography.bodySmall
                    )
                    Surface(
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text(
                            text = inputText,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.padding(10.dp),
                            maxLines = 3,
                            overflow = TextOverflow.Ellipsis
                        )
                    }

                    Text("Assign to Class:", style = MaterialTheme.typography.labelMedium)
                    classes.forEach { cls ->
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp))
                                .clickable { selectedFeedbackClassId = cls.id }
                                .padding(vertical = 4.dp)
                        ) {
                            RadioButton(
                                selected = selectedFeedbackClassId == cls.id,
                                onClick = { selectedFeedbackClassId = cls.id }
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(cls.className, fontWeight = FontWeight.Medium)
                        }
                    }
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        selectedFeedbackClassId?.let { classId ->
                            viewModel.addTextSample(classId, inputText)
                            Toast.makeText(context, "Added to dataset!", Toast.LENGTH_SHORT).show()
                            showFeedbackDialog = false
                        }
                    }
                ) {
                    Text("Save")
                }
            },
            dismissButton = {
                TextButton(onClick = { showFeedbackDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }
}
