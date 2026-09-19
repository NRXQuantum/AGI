package com.example.ui.screens

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.example.ml.EnrolledPerson
import com.example.ml.FaceRecognitionEngine
import com.example.ui.components.CameraTestMode
import com.example.ui.components.FaceEnrollmentDialog
import com.example.ui.components.LiveCameraViewfinder
import com.example.ui.components.LiveDetectedBox
import com.example.ui.components.LiveSinglePrediction
import com.example.ui.viewmodel.ProjectViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID

data class PersonEnrollmentEntry(
    val id: String = UUID.randomUUID().toString(),
    var name: String,
    val photos: MutableList<Bitmap> = mutableListOf()
)

@Composable
fun FaceRecognitionStudio(
    viewModel: ProjectViewModel,
    onOpenBatchSorterWithProject: (Long) -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val faceEngine = remember { FaceRecognitionEngine(context) }

    // Enrolled persons list (in-memory studio)
    val enrolledList = remember {
        mutableStateListOf(
            PersonEnrollmentEntry(name = "Person 1"),
            PersonEnrollmentEntry(name = "Person 2")
        )
    }

    var activePersonForPhotoPicker by remember { mutableStateOf<PersonEnrollmentEntry?>(null) }
    var activePersonForEnrollment by remember { mutableStateOf<PersonEnrollmentEntry?>(null) }
    var showFaceEnrollmentDialog by remember { mutableStateOf(false) }
    var showAddPersonDialog by remember { mutableStateOf(false) }
    var newPersonNameInput by remember { mutableStateOf("") }
    var isBuildingModel by remember { mutableStateOf(false) }
    var createdProjectId by remember { mutableStateOf<Long?>(null) }
    var buildStatusMessage by remember { mutableStateOf<String?>(null) }

    // Live Camera State
    var showLiveCamera by remember { mutableStateOf(false) }
    var liveCameraMode by remember { mutableStateOf(CameraTestMode.LIVE_MULTI) }

    // Pre-calculated enrolled embeddings for live camera inference
    var cachedEnrolledPersons by remember { mutableStateOf<List<EnrolledPerson>>(emptyList()) }

    fun refreshEnrolledBiometrics() {
        coroutineScope.launch(Dispatchers.Default) {
            val list = mutableListOf<EnrolledPerson>()
            for ((idx, person) in enrolledList.withIndex()) {
                val faceEmbeddings = mutableListOf<FloatArray>()
                val bodyEmbeddings = mutableListOf<FloatArray>()
                val patchEmbeddings = mutableListOf<FloatArray>()

                for (photo in person.photos) {
                    // 1. Detect Face
                    val faces = faceEngine.detectFaces(photo, maxFaces = 1)
                    if (faces.isNotEmpty()) {
                        val fBox = faces[0]
                        faceEmbeddings.add(faceEngine.extractFaceEmbedding(photo, fBox))
                        patchEmbeddings.add(faceEngine.extractMultiPatchEmbedding(photo, fBox))
                    }

                    // 2. Detect Body
                    val bodies = faceEngine.detectHumanBodies(photo, maxBodies = 1)
                    if (bodies.isNotEmpty()) {
                        val bBox = bodies[0]
                        bodyEmbeddings.add(faceEngine.extractBodyAppearanceEmbedding(photo, bBox))
                        patchEmbeddings.add(faceEngine.extractMultiPatchEmbedding(photo, bBox))
                    }

                    // 3. Fallback for cropped/partial photos
                    if (faces.isEmpty() && bodies.isEmpty()) {
                        val defaultBox = com.example.ml.FaceBoundingBox(0.05f, 0.05f, 0.95f, 0.95f)
                        patchEmbeddings.add(faceEngine.extractMultiPatchEmbedding(photo, defaultBox))
                        bodyEmbeddings.add(faceEngine.extractBodyAppearanceEmbedding(photo, defaultBox))
                    }
                }

                fun computeNormCentroid(embs: List<FloatArray>): FloatArray? {
                    if (embs.isEmpty()) return null
                    val dim = embs[0].size
                    val centroid = FloatArray(dim)
                    for (e in embs) {
                        for (i in 0 until dim) centroid[i] += e[i]
                    }
                    val count = embs.size.toFloat()
                    for (i in 0 until dim) centroid[i] /= count
                    var sumSq = 0f
                    for (v in centroid) sumSq += v * v
                    val norm = kotlin.math.sqrt(sumSq).coerceAtLeast(1e-7f)
                    for (i in 0 until dim) centroid[i] /= norm
                    return centroid
                }

                val faceCentroid = computeNormCentroid(faceEmbeddings)
                val bodyCentroid = computeNormCentroid(bodyEmbeddings)
                val patchCentroid = computeNormCentroid(patchEmbeddings)

                if (faceCentroid != null || bodyCentroid != null || patchCentroid != null) {
                    list.add(
                        EnrolledPerson(
                            id = idx.toLong(),
                            name = person.name,
                            faceSamplePaths = emptyList(),
                            centroidEmbedding = faceCentroid ?: bodyCentroid ?: patchCentroid,
                            bodyCentroidEmbedding = bodyCentroid,
                            patchCentroidEmbedding = patchCentroid,
                            sampleEmbeddings = faceEmbeddings,
                            totalFacePhotos = faceEmbeddings.size,
                            totalBodyPhotos = bodyEmbeddings.size
                        )
                    )
                }
            }
            cachedEnrolledPersons = list
        }
    }

    // Photo picker for adding face photos
    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null && activePersonForPhotoPicker != null) {
            coroutineScope.launch(Dispatchers.IO) {
                try {
                    val bmp = loadBitmapFromUri(context, uri)
                    if (bmp != null) {
                        withContext(Dispatchers.Main) {
                            activePersonForPhotoPicker?.photos?.add(bmp)
                            refreshEnrolledBiometrics()
                            Toast.makeText(context, "Added photo for ${activePersonForPhotoPicker?.name}", Toast.LENGTH_SHORT).show()
                        }
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }
    }

    // Camera permission for live testing
    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            refreshEnrolledBiometrics()
            showLiveCamera = true
        } else {
            Toast.makeText(context, "Camera permission needed for live face recognition", Toast.LENGTH_SHORT).show()
        }
    }

    Box(modifier = modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Spacer(modifier = Modifier.height(4.dp))

            // 1. Hero Card
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.4f)
                ),
                shape = RoundedCornerShape(16.dp),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.35f)),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Surface(
                            shape = CircleShape,
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(42.dp)
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Icon(
                                    imageVector = Icons.Default.Face,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.onPrimary,
                                    modifier = Modifier.size(24.dp)
                                )
                            }
                        }
                        Spacer(modifier = Modifier.width(12.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "Person & Human Identification Studio",
                                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                            )
                            Text(
                                text = "ব্যক্তি ও হিউম্যান রিকগনিশন (Face + Full Body + Patches)",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.primary
                            )
                        }
                        Surface(
                            color = Color(0xFF10B981).copy(alpha = 0.2f),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Text(
                                text = "HYBRID RE-ID",
                                style = MaterialTheme.typography.labelSmall.copy(
                                    fontWeight = FontWeight.Bold,
                                    color = Color(0xFF10B981)
                                ),
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))
                    Text(
                        text = "Train models to recognize people across Face, Full Body, and Partial Upper Torso. Built-in Smart Data Balancing & Mixed-Patch Handler automatically handles uneven photo counts (e.g. 300 vs 20) and cropped photos without bias!",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        lineHeight = 18.sp
                    )

                    Spacer(modifier = Modifier.height(8.dp))
                    Surface(
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f),
                        shape = RoundedCornerShape(8.dp),
                        border = BorderStroke(0.5.dp, MaterialTheme.colorScheme.outlineVariant)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 5.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                imageVector = Icons.Default.AutoFixHigh,
                                contentDescription = null,
                                tint = Color(0xFF10B981),
                                modifier = Modifier.size(14.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = "Smart Data Leveling: Face-only, Full-body, or Cropped photos supported seamlessly",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }

            // 2. Action Buttons Top Bar
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Enrolled Persons (${enrolledList.size})",
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )

                Button(
                    onClick = {
                        newPersonNameInput = "Person ${enrolledList.size + 1}"
                        showAddPersonDialog = true
                    },
                    shape = RoundedCornerShape(10.dp),
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp),
                    modifier = Modifier.testTag("add_person_btn")
                ) {
                    Icon(imageVector = Icons.Default.PersonAdd, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Add Person", style = MaterialTheme.typography.labelMedium)
                }
            }

            // 3. Enrolled Persons List
            enrolledList.forEachIndexed { index, person ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(14.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f)),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                // Avatar circle
                                if (person.photos.isNotEmpty()) {
                                    Image(
                                        bitmap = person.photos.first().asImageBitmap(),
                                        contentDescription = person.name,
                                        modifier = Modifier
                                            .size(44.dp)
                                            .clip(CircleShape)
                                            .border(1.5.dp, MaterialTheme.colorScheme.primary, CircleShape)
                                    )
                                } else {
                                    Surface(
                                        shape = CircleShape,
                                        color = MaterialTheme.colorScheme.surfaceContainerHighest,
                                        modifier = Modifier.size(44.dp)
                                    ) {
                                        Box(contentAlignment = Alignment.Center) {
                                            Icon(
                                                imageVector = Icons.Default.AccountCircle,
                                                contentDescription = null,
                                                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                                modifier = Modifier.size(28.dp)
                                            )
                                        }
                                    }
                                }

                                Spacer(modifier = Modifier.width(12.dp))
                                Column {
                                    Text(
                                        text = person.name,
                                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                                    )
                                    Text(
                                        text = if (person.photos.isNotEmpty())
                                            "${person.photos.size} face sample(s) enrolled"
                                        else
                                            "No face photo added yet",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = if (person.photos.isNotEmpty()) Color(0xFF10B981) else MaterialTheme.colorScheme.error
                                    )
                                }
                            }

                            // Remove person button
                            IconButton(
                                onClick = {
                                    enrolledList.removeAt(index)
                                    refreshEnrolledBiometrics()
                                }
                            ) {
                                Icon(
                                    imageVector = Icons.Default.DeleteOutline,
                                    contentDescription = "Delete Person",
                                    tint = MaterialTheme.colorScheme.error
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(10.dp))

                        // Photos thumbnail row
                        if (person.photos.isNotEmpty()) {
                            LazyRow(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                items(person.photos) { photo ->
                                    Image(
                                        bitmap = photo.asImageBitmap(),
                                        contentDescription = null,
                                        modifier = Modifier
                                            .size(56.dp)
                                            .clip(RoundedCornerShape(8.dp))
                                            .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(8.dp))
                                    )
                                }

                                item {
                                    // Add more photo tile
                                    Surface(
                                        shape = RoundedCornerShape(8.dp),
                                        color = MaterialTheme.colorScheme.surfaceContainerHigh,
                                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
                                        modifier = Modifier
                                            .size(56.dp)
                                            .clip(RoundedCornerShape(8.dp))
                                            .clickable {
                                                activePersonForPhotoPicker = person
                                                photoPickerLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                                            }
                                    ) {
                                        Box(contentAlignment = Alignment.Center) {
                                            Icon(imageVector = Icons.Default.AddPhotoAlternate, contentDescription = "Add Photo", tint = MaterialTheme.colorScheme.primary)
                                        }
                                    }
                                }

                                item {
                                    // Face Lock Scan tile
                                    Surface(
                                        shape = RoundedCornerShape(8.dp),
                                        color = Color(0xFF0284C7).copy(alpha = 0.15f),
                                        border = BorderStroke(1.dp, Color(0xFF0284C7)),
                                        modifier = Modifier
                                            .size(56.dp)
                                            .clip(RoundedCornerShape(8.dp))
                                            .clickable {
                                                activePersonForEnrollment = person
                                                showFaceEnrollmentDialog = true
                                            }
                                    ) {
                                        Box(contentAlignment = Alignment.Center) {
                                            Icon(imageVector = Icons.Default.Face, contentDescription = "Face Lock Scan", tint = Color(0xFF0284C7))
                                        }
                                    }
                                }
                            }
                        } else {
                            // Prompt to add photo or scan head angles
                            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Button(
                                    onClick = {
                                        activePersonForEnrollment = person
                                        showFaceEnrollmentDialog = true
                                    },
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = Color(0xFF0284C7),
                                        contentColor = Color.White
                                    ),
                                    shape = RoundedCornerShape(10.dp),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Icon(imageVector = Icons.Default.Face, contentDescription = null, modifier = Modifier.size(18.dp))
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text("Face Lock Scan (হেড টার্ন বায়োমেট্রিক)")
                                }

                                OutlinedButton(
                                    onClick = {
                                        activePersonForPhotoPicker = person
                                        photoPickerLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                                    },
                                    shape = RoundedCornerShape(10.dp),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Icon(imageVector = Icons.Default.AddPhotoAlternate, contentDescription = null, modifier = Modifier.size(18.dp))
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text("Pick Photo (Gallery / Files)")
                                }
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            // 4. Primary Actions (Build Model & Test Live Camera)
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                shape = RoundedCornerShape(16.dp),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(
                        text = "Model Generation & Live Testing",
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                    )

                    // Button 1: Build & Calibrate Model
                    Button(
                        onClick = {
                            val validPersons = enrolledList.filter { it.photos.isNotEmpty() }
                            if (validPersons.size < 2) {
                                Toast.makeText(context, "Please add at least 1 photo for at least 2 people to train the model.", Toast.LENGTH_LONG).show()
                                return@Button
                            }

                            isBuildingModel = true
                            buildStatusMessage = null
                            coroutineScope.launch {
                                try {
                                    val pairs = validPersons.map { it.name to it.photos.toList() }
                                    val newId = faceEngine.buildAndSaveFaceRecognitionProject("Family & Persons", pairs)
                                    createdProjectId = newId
                                    buildStatusMessage = "✅ Successfully calibrated and saved Face Recognition Model with ${validPersons.size} persons!"
                                    refreshEnrolledBiometrics()
                                    viewModel.selectProject(newId)
                                    Toast.makeText(context, "Face Recognition Model Built!", Toast.LENGTH_SHORT).show()
                                } catch (e: Exception) {
                                    buildStatusMessage = "❌ Error: ${e.localizedMessage}"
                                } finally {
                                    isBuildingModel = false
                                }
                            }
                        },
                        enabled = !isBuildingModel && enrolledList.any { it.photos.isNotEmpty() },
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier
                            .fillMaxWidth()
                            .defaultMinSize(minHeight = 48.dp)
                            .testTag("build_face_model_btn")
                    ) {
                        if (isBuildingModel) {
                            CircularProgressIndicator(modifier = Modifier.size(20.dp), color = MaterialTheme.colorScheme.onPrimary, strokeWidth = 2.dp)
                            Spacer(modifier = Modifier.width(10.dp))
                            Text("Calibrating Face Embeddings...")
                        } else {
                            Icon(imageVector = Icons.Default.Memory, contentDescription = null, modifier = Modifier.size(20.dp))
                            Spacer(modifier = Modifier.width(10.dp))
                            Text("Build Face Recognition Model", style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold))
                        }
                    }

                    // Button 2: Test with Live Real-Time Camera
                    FilledTonalButton(
                        onClick = {
                            val permissionCheck = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                            if (permissionCheck == PackageManager.PERMISSION_GRANTED) {
                                refreshEnrolledBiometrics()
                                showLiveCamera = true
                            } else {
                                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                            }
                        },
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier
                            .fillMaxWidth()
                            .defaultMinSize(minHeight = 48.dp)
                            .testTag("test_face_live_camera_btn")
                    ) {
                        Icon(imageVector = Icons.Default.Videocam, contentDescription = null, modifier = Modifier.size(20.dp))
                        Spacer(modifier = Modifier.width(10.dp))
                        Text("Open 3-Mode Live Camera", style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold))
                    }

                    // Button 3: Use in Batch Folder Auto-Sorter
                    createdProjectId?.let { projId ->
                        OutlinedButton(
                            onClick = { onOpenBatchSorterWithProject(projId) },
                            shape = RoundedCornerShape(12.dp),
                            modifier = Modifier
                                .fillMaxWidth()
                                .defaultMinSize(minHeight = 46.dp)
                        ) {
                            Icon(imageVector = Icons.Default.FolderSpecial, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Use in Batch Folder Auto-Sorter", style = MaterialTheme.typography.labelLarge)
                        }
                    }

                    buildStatusMessage?.let { msg ->
                        Surface(
                            color = if (msg.startsWith("✅")) Color(0xFF10B981).copy(alpha = 0.15f) else MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.5f),
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = msg,
                                style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Medium),
                                color = if (msg.startsWith("✅")) Color(0xFF10B981) else MaterialTheme.colorScheme.error,
                                modifier = Modifier.padding(10.dp)
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(30.dp))
        }

        // Add Person Dialog
        if (showAddPersonDialog) {
            AlertDialog(
                onDismissRequest = { showAddPersonDialog = false },
                title = { Text("Add Person to Identify") },
                text = {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            text = "Enter person's name or label (e.g. Sajim, John, Mom, Friend 1):",
                            style = MaterialTheme.typography.bodySmall
                        )
                        OutlinedTextField(
                            value = newPersonNameInput,
                            onValueChange = { newPersonNameInput = it },
                            label = { Text("Person Name") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                },
                confirmButton = {
                    Button(
                        onClick = {
                            if (newPersonNameInput.isNotBlank()) {
                                val newEntry = PersonEnrollmentEntry(name = newPersonNameInput.trim())
                                enrolledList.add(newEntry)
                                showAddPersonDialog = false
                                activePersonForPhotoPicker = newEntry
                                photoPickerLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                            }
                        },
                        enabled = newPersonNameInput.isNotBlank()
                    ) {
                        Text("Next: Add Face Photo")
                    }
                },
                dismissButton = {
                    TextButton(onClick = { showAddPersonDialog = false }) {
                        Text("Cancel")
                    }
                }
            )
        }

        if (showFaceEnrollmentDialog && activePersonForEnrollment != null) {
            FaceEnrollmentDialog(
                personName = activePersonForEnrollment!!.name,
                onPhotosEnrolled = { photos ->
                    activePersonForEnrollment!!.photos.addAll(photos)
                    refreshEnrolledBiometrics()
                    Toast.makeText(context, "${photos.size}টি বায়োমেট্রিক ফেস স্ক্যান সফলভাবে সংরক্ষিত হয়েছে!", Toast.LENGTH_LONG).show()
                    showFaceEnrollmentDialog = false
                    activePersonForEnrollment = null
                },
                onDismiss = {
                    showFaceEnrollmentDialog = false
                    activePersonForEnrollment = null
                }
            )
        }

        // Live Camera Viewfinder Overlay for Face Recognition!
        if (showLiveCamera) {
            LiveCameraViewfinder(
                initialMode = liveCameraMode,
                onDismiss = { showLiveCamera = false },
                onPhotoCaptured = { capturedBmp ->
                    showLiveCamera = false
                    Toast.makeText(context, "Captured photo!", Toast.LENGTH_SHORT).show()
                },
                onAnalyzeFrame = { frameBmp, isMulti ->
                    val startTime = System.currentTimeMillis()
                    val identifiedFaces = faceEngine.identifyFacesInScene(
                        sceneBitmap = frameBmp,
                        enrolledPersons = cachedEnrolledPersons,
                        matchThreshold = 0.60f
                    )
                    val latency = System.currentTimeMillis() - startTime

                    val single = identifiedFaces.firstOrNull()?.let { face ->
                        LiveSinglePrediction(
                            label = if (face.personName != "Unknown Person") "${face.personName} (${face.matchType})" else "Unknown Person",
                            confidence = face.confidence,
                            latencyMs = latency
                        )
                    }

                    val boxes = identifiedFaces.map { face ->
                        val boxColor = when {
                            face.personName == "Unknown Person" -> Color(0xFFEF4444)
                            face.matchType == "Face + Body Match" -> Color(0xFF10B981) // Emerald
                            face.matchType == "Face Match" -> Color(0xFF0284C7) // Sky blue
                            face.matchType == "Body / Torso Lock" -> Color(0xFF6366F1) // Indigo
                            else -> Color(0xFFF59E0B) // Amber
                        }
                        val displayLabel = if (face.personName != "Unknown Person") {
                            "${face.personName} • ${face.matchType}"
                        } else {
                            "Unknown Person"
                        }
                        LiveDetectedBox(
                            label = displayLabel,
                            confidence = face.confidence,
                            leftNorm = face.boundingBox.leftNorm,
                            topNorm = face.boundingBox.topNorm,
                            rightNorm = face.boundingBox.rightNorm,
                            bottomNorm = face.boundingBox.bottomNorm,
                            color = boxColor,
                            facialLandmarks = face.facialLandmarks,
                            facialMeshEdges = face.facialMeshEdges,
                            bodyContourPoints = face.bodyContour,
                            statureDiagnostics = face.statureDiagnostics,
                            statureRatio = if (face.boundingBox.rightNorm - face.boundingBox.leftNorm > 0.01f) {
                                (face.boundingBox.bottomNorm - face.boundingBox.topNorm) / (face.boundingBox.rightNorm - face.boundingBox.leftNorm)
                            } else 1.5f
                        )
                    }

                    Pair(single, boxes)
                }
            )
        }
    }
}

private fun loadBitmapFromUri(context: Context, uri: Uri): Bitmap? {
    return try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(context.contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.isMutableRequired = true
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
        }
    } catch (e: Exception) {
        null
    }
}
