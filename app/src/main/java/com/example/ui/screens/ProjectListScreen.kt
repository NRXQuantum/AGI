package com.example.ui.screens

import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.data.db.ProjectEntity
import com.example.ui.viewmodel.AppMode
import com.example.ui.viewmodel.ProjectViewModel
import java.text.SimpleDateFormat
import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ProjectListScreen(
    viewModel: ProjectViewModel,
    onProjectSelected: (Long) -> Unit,
    onTestProject: (Long) -> Unit = { id ->
        viewModel.selectProject(id)
        onProjectSelected(id)
    }
) {
    val projects by viewModel.projects.collectAsState()
    val currentMode by viewModel.currentAppMode.collectAsState()
    var showAllModes by remember { mutableStateOf(false) }
    var showCreateDialog by remember { mutableStateOf(false) }
    var showSettingsScreen by remember { mutableStateOf(false) }

    val filteredProjects = remember(projects, currentMode, showAllModes) {
        if (showAllModes) {
            projects
        } else if (currentMode == AppMode.FACE_RECOGNITION) {
            projects.filter { it.projectType == "FACE_RECOGNITION" }
        } else {
            projects.filter { it.projectType != "FACE_RECOGNITION" }
        }
    }

    AnimatedContent(
        targetState = showSettingsScreen,
        transitionSpec = {
            if (targetState) {
                slideInHorizontally { it } + fadeIn() togetherWith slideOutHorizontally { -it / 3 } + fadeOut()
            } else {
                slideInHorizontally { -it / 3 } + fadeIn() togetherWith slideOutHorizontally { it } + fadeOut()
            }
        },
        label = "project_list_or_settings_anim"
    ) { inSettings ->
        if (inSettings) {
            TrainerSettingsScreen(
                viewModel = viewModel,
                onBack = { showSettingsScreen = false }
            )
        } else {
            Scaffold(
                topBar = {
                    TopAppBar(
                        title = {
                            Column {
                                Text(
                                    text = "Model Exporter",
                                    style = MaterialTheme.typography.titleLarge.copy(
                                        fontWeight = FontWeight.Bold
                                    )
                                )
                                Text(
                                    text = "On-Device Custom Image Classification",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        },
                        actions = {
                            IconButton(
                                onClick = { showSettingsScreen = true },
                                modifier = Modifier.testTag("settings_button")
                            ) {
                                Icon(
                                    Icons.Default.Settings,
                                    contentDescription = "Settings",
                                    tint = MaterialTheme.colorScheme.onSurface
                                )
                            }
                        },
                        colors = TopAppBarDefaults.topAppBarColors(
                            containerColor = MaterialTheme.colorScheme.surface
                        )
                    )
                },
                floatingActionButton = {
                    ExtendedFloatingActionButton(
                        onClick = { showCreateDialog = true },
                        icon = { Icon(Icons.Default.Add, contentDescription = null) },
                        text = { Text("New Project") },
                        modifier = Modifier.testTag("create_project_fab")
                    )
                }
            ) { innerPadding ->
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(innerPadding)
                        .padding(horizontal = 16.dp)
                ) {
                    // 1. Core Mode Switcher Segmented Control
                    SingleChoiceSegmentedButtonRow(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 8.dp, bottom = 6.dp)
                    ) {
                        SegmentedButton(
                            selected = currentMode == AppMode.IMAGE_CLASSIFICATION,
                            onClick = { viewModel.setAppMode(AppMode.IMAGE_CLASSIFICATION) },
                            shape = SegmentedButtonDefaults.itemShape(index = 0, count = 2),
                            icon = {
                                Icon(Icons.Default.Image, contentDescription = null, modifier = Modifier.size(16.dp))
                            }
                        ) {
                            Text("Normal Images", maxLines = 1, fontSize = 13.sp)
                        }
                        SegmentedButton(
                            selected = currentMode == AppMode.FACE_RECOGNITION,
                            onClick = { viewModel.setAppMode(AppMode.FACE_RECOGNITION) },
                            shape = SegmentedButtonDefaults.itemShape(index = 1, count = 2),
                            icon = {
                                Icon(Icons.Default.Face, contentDescription = null, modifier = Modifier.size(16.dp))
                            }
                        ) {
                            Text("Person / Face ID", maxLines = 1, fontSize = 13.sp)
                        }
                    }

                    // 2. Mode-Aware Banner
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 8.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = if (currentMode == AppMode.FACE_RECOGNITION)
                                Color(0xFF0284C7).copy(alpha = 0.14f)
                            else
                                MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.8f)
                        ),
                        shape = RoundedCornerShape(16.dp)
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(14.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(46.dp)
                                    .clip(CircleShape)
                                    .background(
                                        if (currentMode == AppMode.FACE_RECOGNITION)
                                            Color(0xFF0284C7)
                                        else
                                            MaterialTheme.colorScheme.primary
                                    ),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = if (currentMode == AppMode.FACE_RECOGNITION)
                                        Icons.Default.Face
                                    else
                                        Icons.Default.Psychology,
                                    contentDescription = null,
                                    tint = Color.White,
                                    modifier = Modifier.size(26.dp)
                                )
                            }
                            Spacer(modifier = Modifier.width(14.dp))
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = if (currentMode == AppMode.FACE_RECOGNITION)
                                        "100% On-Device Face Recognition"
                                    else
                                        "100% On-Device Neural Training",
                                    style = MaterialTheme.typography.titleMedium.copy(
                                        fontWeight = FontWeight.Bold
                                    ),
                                    color = if (currentMode == AppMode.FACE_RECOGNITION)
                                        Color(0xFF0369A1)
                                    else
                                        MaterialTheme.colorScheme.onPrimaryContainer
                                )
                                Text(
                                    text = if (currentMode == AppMode.FACE_RECOGNITION)
                                        "Recognize specific individuals (Person A, Person B). Train biometrics from 1-3 photos per person with instant calibration."
                                    else
                                        "Train custom vision models locally & export to TFLite, ONNX, CoreML, and SavedModel.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = if (currentMode == AppMode.FACE_RECOGNITION)
                                        Color(0xFF0369A1).copy(alpha = 0.85f)
                                    else
                                        MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    if (filteredProjects.isEmpty()) {
                        EmptyProjectsView(
                            currentMode = currentMode,
                            onCreateClick = { showCreateDialog = true }
                        )
                    } else {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 6.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = if (currentMode == AppMode.FACE_RECOGNITION)
                                    "Face ID Projects (${filteredProjects.size})"
                                else
                                    "Image Classification Projects (${filteredProjects.size})",
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold),
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )

                            if (projects.size > filteredProjects.size || showAllModes) {
                                TextButton(
                                    onClick = { showAllModes = !showAllModes },
                                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp)
                                ) {
                                    Text(
                                        text = if (showAllModes) "Filter by Mode" else "Show All (${projects.size})",
                                        style = MaterialTheme.typography.labelSmall
                                    )
                                }
                            }
                        }

                        LazyColumn(
                            verticalArrangement = Arrangement.spacedBy(12.dp),
                            contentPadding = PaddingValues(bottom = 80.dp)
                        ) {
                            items(filteredProjects, key = { it.id }) { project ->
                                ProjectItemCard(
                                    project = project,
                                    onClick = {
                                        viewModel.selectProject(project.id)
                                        onProjectSelected(project.id)
                                    },
                                    onTestClick = {
                                        viewModel.selectProject(project.id)
                                        onTestProject(project.id)
                                    },
                                    onDelete = { viewModel.deleteProject(project.id) }
                                )
                            }
                        }
                    }
                }
            }

            if (showCreateDialog) {
                CreateProjectDialog(
                    initialMode = currentMode,
                    onDismiss = { showCreateDialog = false },
                    onCreate = { name, desc, pType ->
                        viewModel.createProject(name, desc, pType) { newId ->
                            onProjectSelected(newId)
                        }
                        showCreateDialog = false
                    }
                )
            }
        }
    }
}

@Composable
fun ProjectItemCard(
    project: ProjectEntity,
    onClick: () -> Unit,
    onTestClick: () -> Unit,
    onDelete: () -> Unit
) {
    var showDeleteConfirm by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() }
            .testTag("project_card_${project.id}"),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f)
        ),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = project.name,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        if (project.projectType == "FACE_RECOGNITION") {
                            Surface(
                                color = Color(0xFF0284C7).copy(alpha = 0.15f),
                                shape = RoundedCornerShape(6.dp)
                            ) {
                                Row(
                                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Icon(
                                        Icons.Default.Face,
                                        contentDescription = null,
                                        tint = Color(0xFF0284C7),
                                        modifier = Modifier.size(12.dp)
                                    )
                                    Spacer(modifier = Modifier.width(3.dp))
                                    Text(
                                        text = "FACE ID",
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontSize = 9.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = Color(0xFF0284C7)
                                        )
                                    )
                                }
                            }
                        } else {
                            Surface(
                                color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.45f),
                                shape = RoundedCornerShape(6.dp)
                            ) {
                                Row(
                                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Icon(
                                        Icons.Default.Image,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(12.dp)
                                    )
                                    Spacer(modifier = Modifier.width(3.dp))
                                    Text(
                                        text = "VISION",
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontSize = 9.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = MaterialTheme.colorScheme.primary
                                        )
                                    )
                                }
                            }
                        }
                    }
                }

                Row(verticalAlignment = Alignment.CenterVertically) {
                    StatusBadge(isTrained = project.isTrained, accuracy = project.trainingAccuracy)
                    IconButton(
                        onClick = { showDeleteConfirm = true },
                        modifier = Modifier.size(36.dp)
                    ) {
                        Icon(
                            Icons.Outlined.Delete,
                            contentDescription = "Delete Project",
                            tint = MaterialTheme.colorScheme.error
                        )
                    }
                }
            }

            if (project.description.isNotBlank()) {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = project.description,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            if (project.isTrained) {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    color = Color(0xFF10B981).copy(alpha = 0.12f),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.CheckCircle,
                            contentDescription = null,
                            tint = Color(0xFF059669),
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "Training 100% Completed • Model Accuracy: ${String.format(Locale.US, "%.0f%%", project.trainingAccuracy * 100)}",
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold),
                            color = Color(0xFF065F46)
                        )
                    }
                }
                Spacer(modifier = Modifier.height(10.dp))
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                val dateFormat = SimpleDateFormat("MMM dd, yyyy", Locale.getDefault())
                val dateStr = dateFormat.format(Date(project.createdAt))

                Text(
                    text = "Created: $dateStr",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                )

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    if (project.isTrained) {
                        FilledTonalButton(
                            onClick = onTestClick,
                            shape = RoundedCornerShape(8.dp),
                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 2.dp),
                            modifier = Modifier.height(34.dp).testTag("card_test_btn_${project.id}")
                        ) {
                            Icon(
                                imageVector = Icons.Default.CheckCircle,
                                contentDescription = null,
                                modifier = Modifier.size(15.dp)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = "Test Model",
                                style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                    }

                    Button(
                        onClick = onClick,
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 2.dp),
                        modifier = Modifier.height(34.dp).testTag("card_open_btn_${project.id}")
                    ) {
                        Text(
                            text = "Workspace",
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Icon(
                            imageVector = Icons.Default.ArrowForward,
                            contentDescription = null,
                            modifier = Modifier.size(14.dp)
                        )
                    }
                }
            }
        }
    }

    if (showDeleteConfirm) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = false },
            title = { Text("Delete Project?") },
            text = { Text("Are you sure you want to delete '${project.name}' and all associated training images?") },
            confirmButton = {
                TextButton(
                    onClick = {
                        onDelete()
                        showDeleteConfirm = false
                    }
                ) {
                    Text("Delete", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = false }) {
                    Text("Cancel")
                }
            }
        )
    }
}

@Composable
fun StatusBadge(isTrained: Boolean, accuracy: Float) {
    Surface(
        color = if (isTrained) Color(0xFF10B981).copy(alpha = 0.18f) else MaterialTheme.colorScheme.surface,
        shape = RoundedCornerShape(12.dp)
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .clip(CircleShape)
                    .background(
                        if (isTrained) Color(0xFF10B981) else MaterialTheme.colorScheme.outline
                    )
            )
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = if (isTrained) "Trained (${String.format(Locale.US, "%.0f%%", accuracy * 100)} Acc)" else "Untrained",
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                color = if (isTrained) Color(0xFF059669) else MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun EmptyProjectsView(
    currentMode: AppMode = AppMode.IMAGE_CLASSIFICATION,
    onCreateClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 40.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            imageVector = if (currentMode == AppMode.FACE_RECOGNITION) Icons.Default.Face else Icons.Outlined.FolderOpen,
            contentDescription = null,
            modifier = Modifier.size(72.dp),
            tint = if (currentMode == AppMode.FACE_RECOGNITION)
                Color(0xFF0284C7).copy(alpha = 0.7f)
            else
                MaterialTheme.colorScheme.primary.copy(alpha = 0.6f)
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = if (currentMode == AppMode.FACE_RECOGNITION)
                "No Face ID Projects Yet"
            else
                "No Classification Projects",
            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = if (currentMode == AppMode.FACE_RECOGNITION)
                "Create your first biometric project to enroll individuals (Person A, Person B, etc.) with 1-3 photos and recognize them in real time."
            else
                "Create your first project to collect training images, train custom neural models on-device, and export them.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(horizontal = 24.dp)
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(
            onClick = onCreateClick,
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.testTag("empty_state_create_btn")
        ) {
            Icon(Icons.Default.Add, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                if (currentMode == AppMode.FACE_RECOGNITION)
                    "Create Face ID Project"
                else
                    "Create Classification Project"
            )
        }
    }
}

@Composable
fun CreateProjectDialog(
    initialMode: AppMode = AppMode.IMAGE_CLASSIFICATION,
    onDismiss: () -> Unit,
    onCreate: (String, String, String) -> Unit
) {
    var name by remember { mutableStateOf("") }
    var description by remember { mutableStateOf("") }
    var selectedMode by remember { mutableStateOf(initialMode) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(if (selectedMode == AppMode.FACE_RECOGNITION) "New Face / Person Project" else "New Vision Project")
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                    SegmentedButton(
                        selected = selectedMode == AppMode.IMAGE_CLASSIFICATION,
                        onClick = { selectedMode = AppMode.IMAGE_CLASSIFICATION },
                        shape = SegmentedButtonDefaults.itemShape(index = 0, count = 2),
                        icon = {
                            Icon(Icons.Default.Image, contentDescription = null, modifier = Modifier.size(14.dp))
                        }
                    ) {
                        Text("Images", fontSize = 12.sp)
                    }
                    SegmentedButton(
                        selected = selectedMode == AppMode.FACE_RECOGNITION,
                        onClick = { selectedMode = AppMode.FACE_RECOGNITION },
                        shape = SegmentedButtonDefaults.itemShape(index = 1, count = 2),
                        icon = {
                            Icon(Icons.Default.Face, contentDescription = null, modifier = Modifier.size(14.dp))
                        }
                    ) {
                        Text("Face ID", fontSize = 12.sp)
                    }
                }

                OutlinedTextField(
                    value = name,
                    onValueChange = { name = it },
                    label = { Text("Project Name") },
                    placeholder = {
                        Text(if (selectedMode == AppMode.FACE_RECOGNITION) "e.g., Office Team, My Family" else "e.g., Plant Disease Classifier")
                    },
                    singleLine = true,
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("project_name_input")
                )
                OutlinedTextField(
                    value = description,
                    onValueChange = { description = it },
                    label = { Text("Description (Optional)") },
                    placeholder = {
                        Text(if (selectedMode == AppMode.FACE_RECOGNITION) "e.g., Identify friends and family members" else "e.g., Detect leaf diseases on tomato plants")
                    },
                    modifier = Modifier.fillMaxWidth()
                )

                Surface(
                    color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(
                        text = if (selectedMode == AppMode.FACE_RECOGNITION)
                            "💡 Creates 'Person A' and 'Person B' by default. Needs 1-3 face photos per person for instant on-device recognition."
                        else
                            "💡 Creates 'Class A' and 'Class B' by default for multi-class image classification.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(10.dp)
                    )
                }
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    if (name.isNotBlank()) {
                        onCreate(
                            name.trim(),
                            description.trim(),
                            if (selectedMode == AppMode.FACE_RECOGNITION) "FACE_RECOGNITION" else "IMAGE_CLASSIFICATION"
                        )
                    }
                },
                enabled = name.isNotBlank(),
                modifier = Modifier.testTag("confirm_create_project_btn")
            ) {
                Text(if (selectedMode == AppMode.FACE_RECOGNITION) "Create Face Project" else "Create Project")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}
