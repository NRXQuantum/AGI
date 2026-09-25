package com.example.ui.components

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
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
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Send
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
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

enum class TextStudioMode(val title: String, val titleBn: String, val icon: androidx.compose.ui.graphics.vector.ImageVector) {
    SENTENCE_ANALYSIS("Sentence Classifier", "একক বাক্য বিশ্লেষণ", Icons.Default.TextFields),
    INTERACTIVE_CHAT("Dialogue & Model Chat", "ডায়লগ ও মডেল চ্যাট", Icons.Default.Chat)
}

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
    val coroutineScope = rememberCoroutineScope()
    var selectedStudioMode by remember { mutableStateOf(TextStudioMode.INTERACTIVE_CHAT) }

    // Tab 1 state: Single sentence classification
    var inputText by remember { mutableStateOf("") }
    var livePredictEnabled by remember { mutableStateOf(true) }
    val prediction by viewModel.textInferenceResult.collectAsState()
    val isRunning by viewModel.isInferenceRunning.collectAsState()
    var showFeedbackDialog by remember { mutableStateOf(false) }
    var feedbackTextContent by remember { mutableStateOf("") }

    // Tab 2 state: Interactive dialogue chat
    val chatMessages by viewModel.chatMessages.collectAsState()
    val isChatGenerating by viewModel.isChatGenerating.collectAsState()
    var chatInputText by remember { mutableStateOf("") }
    val chatListState = rememberLazyListState()

    BackHandler {
        if (onNavigateBack != null) onNavigateBack()
    }

    // Auto scroll chat to bottom on new message
    LaunchedEffect(chatMessages.size, isChatGenerating) {
        if (chatMessages.isNotEmpty()) {
            chatListState.animateScrollToItem(chatMessages.size - 1)
        }
    }

    // Live predict as user types in Single sentence tab
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
                            text = if (selectedStudioMode == TextStudioMode.INTERACTIVE_CHAT) "Interactive Dialogue & Chat Test" else "Sentence Classifier & Explainable AI",
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
        ) {
            // 1. Model Status Banner & Retrain
            Box(modifier = Modifier.padding(horizontal = 16.dp, vertical = 6.dp)) {
                if (latestModel == null) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
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
                                    "Train this text dataset to activate live inference & dialogue chat.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onErrorContainer
                                )
                            }
                            Button(
                                onClick = onNavigateToTrain,
                                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                            ) {
                                Text("Train")
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
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    Icons.Default.CheckCircle,
                                    contentDescription = null,
                                    tint = Color(0xFF059669),
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Column {
                                    Text(
                                        "Trained Model Active (${String.format(Locale.US, "%.1f%%", latestModel.accuracy * 100f)} Accuracy)",
                                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                                        color = Color(0xFF065F46)
                                    )
                                    Text(
                                        "${classes.size} Text Classes • On-Device Neural Engine",
                                        style = MaterialTheme.typography.labelSmall,
                                        color = Color(0xFF047857)
                                    )
                                }
                            }

                            TextButton(
                                onClick = onNavigateToTrain,
                                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp)
                            ) {
                                Text("Retrain", style = MaterialTheme.typography.labelMedium)
                            }
                        }
                    }
                }
            }

            // 2. Mode Switcher (Sentence Classifier vs Dialogue Chat)
            SingleChoiceSegmentedButtonRow(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 4.dp)
            ) {
                SegmentedButton(
                    selected = selectedStudioMode == TextStudioMode.INTERACTIVE_CHAT,
                    onClick = { selectedStudioMode = TextStudioMode.INTERACTIVE_CHAT },
                    shape = SegmentedButtonDefaults.itemShape(index = 0, count = 2),
                    icon = { Icon(Icons.Default.Chat, contentDescription = null, modifier = Modifier.size(16.dp)) }
                ) {
                    Text("Dialogue Chat", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                }

                SegmentedButton(
                    selected = selectedStudioMode == TextStudioMode.SENTENCE_ANALYSIS,
                    onClick = { selectedStudioMode = TextStudioMode.SENTENCE_ANALYSIS },
                    shape = SegmentedButtonDefaults.itemShape(index = 1, count = 2),
                    icon = { Icon(Icons.Default.TextFields, contentDescription = null, modifier = Modifier.size(16.dp)) }
                ) {
                    Text("Single Classifier", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                }
            }

            // 3. Tab Content
            when (selectedStudioMode) {
                TextStudioMode.INTERACTIVE_CHAT -> {
                    InteractiveDialogueChatTab(
                        viewModel = viewModel,
                        classes = classes,
                        latestModel = latestModel,
                        chatMessages = chatMessages,
                        isChatGenerating = isChatGenerating,
                        chatInputText = chatInputText,
                        onChatInputChange = { chatInputText = it },
                        chatListState = chatListState,
                        onSendMessage = { prompt ->
                            if (latestModel == null) {
                                Toast.makeText(context, "Please train the model first!", Toast.LENGTH_SHORT).show()
                            } else {
                                viewModel.sendChatMessage(prompt)
                                chatInputText = ""
                            }
                        },
                        onClearChat = { viewModel.clearChatMessages() },
                        onSaveFeedback = { text ->
                            feedbackTextContent = text
                            showFeedbackDialog = true
                        }
                    )
                }

                TextStudioMode.SENTENCE_ANALYSIS -> {
                    SentenceClassifierTab(
                        viewModel = viewModel,
                        classes = classes,
                        latestModel = latestModel,
                        inputText = inputText,
                        onInputTextChange = { inputText = it },
                        livePredictEnabled = livePredictEnabled,
                        onLivePredictChange = { livePredictEnabled = it },
                        prediction = prediction,
                        isRunning = isRunning,
                        onSaveToDataset = {
                            feedbackTextContent = inputText
                            showFeedbackDialog = true
                        }
                    )
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
                        "Add this dialogue line or sentence to reinforce your model's knowledge:",
                        style = MaterialTheme.typography.bodySmall
                    )
                    Surface(
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text(
                            text = feedbackTextContent,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.padding(10.dp),
                            maxLines = 4,
                            overflow = TextOverflow.Ellipsis
                        )
                    }

                    Text("Assign to Character / Category:", style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold))
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 200.dp)
                            .verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        classes.forEach { cls ->
                            val cColor = try {
                                Color(android.graphics.Color.parseColor(cls.colorHex))
                            } catch (_: Exception) {
                                MaterialTheme.colorScheme.primary
                            }
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(8.dp))
                                    .clickable { selectedFeedbackClassId = cls.id }
                                    .padding(vertical = 4.dp, horizontal = 6.dp)
                            ) {
                                RadioButton(
                                    selected = selectedFeedbackClassId == cls.id,
                                    onClick = { selectedFeedbackClassId = cls.id }
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Box(modifier = Modifier.size(10.dp).clip(CircleShape).background(cColor))
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(cls.className, fontWeight = FontWeight.Medium)
                            }
                        }
                    }
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        selectedFeedbackClassId?.let { classId ->
                            viewModel.addTextSample(classId, feedbackTextContent)
                            Toast.makeText(context, "Saved to category dataset!", Toast.LENGTH_SHORT).show()
                            showFeedbackDialog = false
                        }
                    }
                ) {
                    Text("Save Sample")
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

/**
 * Tab 1: Interactive Dialogue & Model Chat View
 */
@Composable
fun InteractiveDialogueChatTab(
    viewModel: ProjectViewModel,
    classes: List<ClassificationClassEntity>,
    latestModel: TrainedModelEntity?,
    chatMessages: List<TextModelEngine.TextChatMessage>,
    isChatGenerating: Boolean,
    chatInputText: String,
    onChatInputChange: (String) -> Unit,
    chatListState: androidx.compose.foundation.lazy.LazyListState,
    onSendMessage: (String) -> Unit,
    onClearChat: () -> Unit,
    onSaveFeedback: (String) -> Unit
) {
    val context = LocalContext.current

    Column(
        modifier = Modifier
            .fillMaxSize()
            .imePadding()
    ) {
        // Chat Header / Quick Actions
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 4.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "💬 Model Dialogue Chat (${chatMessages.size} turns)",
                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            if (chatMessages.isNotEmpty()) {
                TextButton(
                    onClick = onClearChat,
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp)
                ) {
                    Icon(Icons.Default.DeleteOutline, contentDescription = null, modifier = Modifier.size(14.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Clear Chat", style = MaterialTheme.typography.labelSmall)
                }
            }
        }

        // Messages List or Empty Welcome State
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
        ) {
            if (chatMessages.isEmpty()) {
                // Empty state with quick starter prompt chips
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(20.dp)
                        .verticalScroll(rememberScrollState()),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Surface(
                        color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.5f),
                        shape = CircleShape,
                        modifier = Modifier.size(56.dp)
                    ) {
                        Box(contentAlignment = Alignment.Center) {
                            Icon(
                                Icons.Default.Forum,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(30.dp)
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    Text(
                        text = "Real-Time Dialogue & Chat Studio",
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.onSurface
                    )

                    Spacer(modifier = Modifier.height(4.dp))

                    Text(
                        text = "আপনার আপলোড করা বড় টেক্সট ডাটাবেজ (যেমন input.txt / ডায়লগ স্ক্রিপ্ট) এর প্রতিটি চরিত্র বা ক্যাটাগরির সাথে সরাসরি চ্যাট করুন। মডেলটি রিয়েল-টাইমে প্রেডিক্ট করে ডায়লগ রেসপন্স দেবে।",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(horizontal = 16.dp),
                        textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                        lineHeight = 18.sp
                    )

                    Spacer(modifier = Modifier.height(18.dp))

                    Text(
                        text = "Try a sample dialogue prompt:",
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                        color = MaterialTheme.colorScheme.primary
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    val samplePrompts = remember(classes) {
                        val list = mutableListOf<String>()
                        if (classes.any { it.className.contains("Citizen", ignoreCase = true) || it.className.contains("Menenius", ignoreCase = true) }) {
                            list.add("What work's, my countrymen, in hand? where go you with bats and clubs?")
                            list.add("You are all resolved rather to die than to famish?")
                            list.add("First, you know Caius Marcius is chief enemy to the people.")
                            list.add("Let us kill him, and we'll have corn at our own price.")
                            list.add("Worthy Menenius Agrippa; one that hath always loved the people.")
                        } else {
                            list.add("This is an outstanding product, works completely as expected!")
                            list.add("My account was charged twice, how can I request a refund?")
                            list.add("App keeps freezing on launch screen and crashing.")
                            list.add("Can you add custom theme support in the next update?")
                            list.add("URGENT: Verify your account within 24 hours or get locked!")
                        }
                        list
                    }

                    Column(
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        samplePrompts.forEach { prompt ->
                            Surface(
                                shape = RoundedCornerShape(10.dp),
                                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f)),
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clickable {
                                        onSendMessage(prompt)
                                    }
                            ) {
                                Row(
                                    modifier = Modifier.padding(10.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Icon(
                                        Icons.Default.ChatBubbleOutline,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(16.dp)
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = prompt,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurface,
                                        maxLines = 2,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                }
                            }
                        }
                    }
                }
            } else {
                // Chat conversation messages
                LazyColumn(
                    state = chatListState,
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(horizontal = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                    contentPadding = PaddingValues(vertical = 10.dp)
                ) {
                    items(chatMessages, key = { it.id }) { msg ->
                        ChatMessageItem(
                            message = msg,
                            classes = classes,
                            onSaveFeedback = { onSaveFeedback(msg.text) }
                        )
                    }

                    if (isChatGenerating) {
                        item {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 4.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Surface(
                                    shape = RoundedCornerShape(12.dp),
                                    color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f)
                                ) {
                                    Row(
                                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        CircularProgressIndicator(
                                            modifier = Modifier.size(14.dp),
                                            strokeWidth = 2.dp,
                                            color = MaterialTheme.colorScheme.primary
                                        )
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(
                                            text = "Model predicting intent & dialogue reply...",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Quick Suggestion Chips above input bar
        val quickChips = remember(classes) {
            val list = mutableListOf<String>()
            if (classes.any { it.className.contains("Citizen", ignoreCase = true) }) {
                list.add("First Citizen")
                list.add("Second Citizen")
                list.add("MENENIUS")
                list.add("All")
            } else {
                classes.take(4).forEach { list.add(it.className) }
            }
            list
        }

        if (quickChips.isNotEmpty()) {
            LazyRow(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 2.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                items(quickChips) { chipText ->
                    AssistChip(
                        onClick = {
                            val newText = if (chatInputText.isBlank()) "$chipText: " else "$chatInputText $chipText"
                            onChatInputChange(newText)
                        },
                        label = { Text(chipText, fontSize = 11.sp) },
                        leadingIcon = {
                            Icon(Icons.Default.Tag, contentDescription = null, modifier = Modifier.size(12.dp))
                        }
                    )
                }
            }
        }

        // Bottom Input Row
        Surface(
            tonalElevation = 3.dp,
            shadowElevation = 6.dp,
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 10.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = chatInputText,
                    onValueChange = onChatInputChange,
                    placeholder = { Text("Type dialogue or message to chat...", fontSize = 13.sp) },
                    modifier = Modifier
                        .weight(1f)
                        .testTag("chat_input_field"),
                    maxLines = 3,
                    shape = RoundedCornerShape(20.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedContainerColor = MaterialTheme.colorScheme.surface,
                        unfocusedContainerColor = MaterialTheme.colorScheme.surface
                    )
                )

                Spacer(modifier = Modifier.width(6.dp))

                IconButton(
                    onClick = {
                        if (chatInputText.isNotBlank()) {
                            onSendMessage(chatInputText)
                        }
                    },
                    enabled = chatInputText.isNotBlank() && !isChatGenerating,
                    modifier = Modifier
                        .size(44.dp)
                        .clip(CircleShape)
                        .background(
                            if (chatInputText.isNotBlank() && !isChatGenerating) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.surfaceVariant
                        )
                        .testTag("send_chat_btn")
                ) {
                    Icon(
                        Icons.AutoMirrored.Filled.Send,
                        contentDescription = "Send",
                        tint = if (chatInputText.isNotBlank() && !isChatGenerating) MaterialTheme.colorScheme.onPrimary
                        else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
                        modifier = Modifier.size(20.dp)
                    )
                }
            }
        }
    }
}

/**
 * Individual Chat Bubble
 */
@Composable
fun ChatMessageItem(
    message: TextModelEngine.TextChatMessage,
    classes: List<ClassificationClassEntity>,
    onSaveFeedback: () -> Unit
) {
    val context = LocalContext.current
    val timeFormat = remember { SimpleDateFormat("h:mm a", Locale.getDefault()) }
    val timeStr = remember(message.timestampMs) { timeFormat.format(Date(message.timestampMs)) }

    val matchedClass = classes.find { it.className.equals(message.predictedClass, ignoreCase = true) }
    val classColor = try {
        Color(android.graphics.Color.parseColor(matchedClass?.colorHex ?: "#10B981"))
    } catch (_: Exception) {
        Color(0xFF10B981)
    }

    if (message.isUser) {
        // User Message (Right Aligned)
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.End
        ) {
            Row(
                verticalAlignment = Alignment.Bottom,
                horizontalArrangement = Arrangement.End,
                modifier = Modifier.fillMaxWidth(0.88f)
            ) {
                Surface(
                    shape = RoundedCornerShape(16.dp, 16.dp, 4.dp, 16.dp),
                    color = MaterialTheme.colorScheme.primary,
                    tonalElevation = 2.dp
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(
                            text = message.text,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onPrimary
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = timeStr,
                            style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp),
                            color = MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.7f),
                            modifier = Modifier.align(Alignment.End)
                        )
                    }
                }
            }
        }
    } else {
        // Model / Character Reply (Left Aligned)
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.Start
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(0.92f),
                horizontalArrangement = Arrangement.Start,
                verticalAlignment = Alignment.Top
            ) {
                // Avatar
                Surface(
                    shape = CircleShape,
                    color = classColor.copy(alpha = 0.2f),
                    border = BorderStroke(1.5.dp, classColor),
                    modifier = Modifier.size(36.dp)
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Text(
                            text = message.senderName.take(1).uppercase(Locale.ROOT),
                            style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                            color = classColor
                        )
                    }
                }

                Spacer(modifier = Modifier.width(8.dp))

                Column {
                    // Header Badge
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Text(
                            text = message.senderName,
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                            color = classColor
                        )

                        if (message.confidence > 0f) {
                            Surface(
                                shape = RoundedCornerShape(4.dp),
                                color = classColor.copy(alpha = 0.15f)
                            ) {
                                Text(
                                    text = "${String.format(Locale.US, "%.1f", message.confidence * 100f)}% match",
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 10.sp
                                    ),
                                    color = classColor,
                                    modifier = Modifier.padding(horizontal = 5.dp, vertical = 1.dp)
                                )
                            }
                        }

                        if (message.latencyMs > 0L) {
                            Text(
                                text = "• ${message.latencyMs}ms",
                                style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp),
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    // Message Bubble
                    Surface(
                        shape = RoundedCornerShape(4.dp, 16.dp, 16.dp, 16.dp),
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
                    ) {
                        Column(modifier = Modifier.padding(12.dp)) {
                            Text(
                                text = message.text,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                lineHeight = 20.sp
                            )

                            // Salient Keywords
                            if (message.salientKeywords.isNotEmpty()) {
                                Spacer(modifier = Modifier.height(8.dp))
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Text(
                                        text = "Salient:",
                                        style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp),
                                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                                    )
                                    message.salientKeywords.take(3).forEach { kw ->
                                        Surface(
                                            color = classColor.copy(alpha = 0.12f),
                                            shape = RoundedCornerShape(4.dp)
                                        ) {
                                            Text(
                                                text = kw,
                                                style = MaterialTheme.typography.labelSmall.copy(
                                                    fontSize = 9.sp,
                                                    fontWeight = FontWeight.Bold
                                                ),
                                                color = classColor,
                                                modifier = Modifier.padding(horizontal = 4.dp, vertical = 1.dp)
                                            )
                                        }
                                    }
                                }
                            }

                            Spacer(modifier = Modifier.height(6.dp))

                            // Bottom actions inside bubble
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = timeStr,
                                    style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp),
                                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                                )

                                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                                    IconButton(
                                        onClick = {
                                            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                                            val clip = ClipData.newPlainText("Dialogue", message.text)
                                            clipboard.setPrimaryClip(clip)
                                            Toast.makeText(context, "Copied dialogue to clipboard", Toast.LENGTH_SHORT).show()
                                        },
                                        modifier = Modifier.size(24.dp)
                                    ) {
                                        Icon(
                                            Icons.Outlined.ContentCopy,
                                            contentDescription = "Copy",
                                            modifier = Modifier.size(13.dp),
                                            tint = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }

                                    IconButton(
                                        onClick = onSaveFeedback,
                                        modifier = Modifier.size(24.dp)
                                    ) {
                                        Icon(
                                            Icons.Outlined.BookmarkAdd,
                                            contentDescription = "Save to Dataset",
                                            modifier = Modifier.size(14.dp),
                                            tint = classColor
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

/**
 * Tab 2: Single Sentence Classifier & Explainable AI Tab
 */
@Composable
fun SentenceClassifierTab(
    viewModel: ProjectViewModel,
    classes: List<ClassificationClassEntity>,
    latestModel: TrainedModelEntity?,
    inputText: String,
    onInputTextChange: (String) -> Unit,
    livePredictEnabled: Boolean,
    onLivePredictChange: (Boolean) -> Unit,
    prediction: TextModelEngine.TextPrediction?,
    isRunning: Boolean,
    onSaveToDataset: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        // Input Text Area
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
                    onValueChange = onInputTextChange,
                    placeholder = { Text("Type or paste any text sentence, dialogue, review, or message to classify in real-time...") },
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
                            onCheckedChange = onLivePredictChange,
                            modifier = Modifier.graphicsLayer { scaleX = 0.85f; scaleY = 0.85f }
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Live Predict", style = MaterialTheme.typography.labelSmall)
                    }

                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        if (inputText.isNotBlank()) {
                            TextButton(onClick = { onInputTextChange("") }) {
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
                Text(
                    "Quick Test Samples:",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.height(6.dp))
                LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    val samplePrompts = listOf(
                        "You are all resolved rather to die than to famish?",
                        "Let us kill him, and we'll have corn at our own price.",
                        "What work's, my countrymen, in hand? where go you With bats and clubs?",
                        "This product is absolutely wonderful, works like a charm!",
                        "Worst update ever, keeps crashing and losing data."
                    )
                    items(samplePrompts) { prompt ->
                        SuggestionChip(
                            onClick = { onInputTextChange(prompt) },
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

        // Inference Results
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
                    Text("Extracting semantic features & running neural pass...", style = MaterialTheme.typography.labelMedium)
                }
            }
        } else {
            val currentPred = prediction
            if (currentPred != null) {
                val winningClass = classes.find { it.className.equals(currentPred.classLabel, ignoreCase = true) }
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
                                        currentPred.classLabel,
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
                                text = "${String.format(Locale.US, "%.1f", currentPred.confidence * 100f)}%",
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
                                    Text("${currentPred.inferenceTimeMs} ms", style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold))
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
                                    Text("${currentPred.tokenCount} / ${currentPred.tokenLimit}", style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold))
                                }
                            }
                        }

                        // Explainable AI (Salient Keywords)
                        if (currentPred.salientTokens.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(14.dp))
                            Text(
                                "Key Influencing Words (Explainable AI):",
                                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Spacer(modifier = Modifier.height(6.dp))
                            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                for (tokenPair in currentPred.salientTokens) {
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
                            for (classProb in currentPred.allProbabilities) {
                                val targetClass = classes.find { it.className.equals(classProb.classLabel, ignoreCase = true) }
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
                            onClick = onSaveToDataset,
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
