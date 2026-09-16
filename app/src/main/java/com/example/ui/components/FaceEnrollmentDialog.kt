package com.example.ui.components

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import java.util.Locale
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.ml.DetectedHeadPose
import com.example.ml.FaceBoundingBox
import com.example.ml.FaceRecognitionEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

data class EnrollmentStep(
    val id: Int,
    val titleBn: String,
    val titleEn: String,
    val instructionBn: String,
    val instructionEn: String,
    val iconEmoji: String,
    val hintDirection: String
)

val ENROLLMENT_STEPS = listOf(
    EnrollmentStep(
        id = 1,
        titleBn = "সোজা ফ্রন্ট",
        titleEn = "Look Straight",
        instructionBn = "সোজা ক্যামেরার দিকে তাকান এবং স্থির থাকুন",
        instructionEn = "Look directly at the front camera",
        iconEmoji = "👤",
        hintDirection = "সরাসরি সোজা"
    ),
    EnrollmentStep(
        id = 2,
        titleBn = "বামে ঘোরান",
        titleEn = "Turn Left",
        instructionBn = "মাথাটি সামান্য বাম দিকে ঘোরান",
        instructionEn = "Turn your head slightly to the left",
        iconEmoji = "👈",
        hintDirection = "বাম কোণ"
    ),
    EnrollmentStep(
        id = 3,
        titleBn = "ডানে ঘোরান",
        titleEn = "Turn Right",
        instructionBn = "মাথাটি সামান্য ডান দিকে ঘোরান",
        instructionEn = "Turn your head slightly to the right",
        iconEmoji = "👉",
        hintDirection = "ডান কোণ"
    ),
    EnrollmentStep(
        id = 4,
        titleBn = "উপরে তুলুন",
        titleEn = "Tilt Up",
        instructionBn = "থুতনি সামান্য উপরের দিকে তুলুন",
        instructionEn = "Tilt your chin slightly upwards",
        iconEmoji = "👆",
        hintDirection = "উপরের কোণ"
    ),
    EnrollmentStep(
        id = 5,
        titleBn = "স্বাভাবিক হাসি",
        titleEn = "Smile / Expression",
        instructionBn = "স্বাভাবিক ভঙ্গিতে মৃদু হাসিমুখ করুন",
        instructionEn = "Give a natural smile or expression",
        iconEmoji = "😊",
        hintDirection = "হাসিমুখ"
    )
)

@Composable
fun FaceEnrollmentDialog(
    personName: String,
    onPhotosEnrolled: (List<Bitmap>) -> Unit,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val faceEngine = remember { FaceRecognitionEngine(context) }

    // Auto-capture & step state
    var currentStepIndex by remember { mutableIntStateOf(0) }
    val capturedAnglePhotos = remember { mutableStateListOf<Bitmap>() }
    var isFaceDetectedInFrame by remember { mutableStateOf(false) }
    var liveFaceBox by remember { mutableStateOf<FaceBoundingBox?>(null) }
    var isFinished by remember { mutableStateOf(false) }
    var isAutoCaptureEnabled by remember { mutableStateOf(true) }
    var isCapturing by remember { mutableStateOf(false) }
    var isFlashVisible by remember { mutableStateOf(false) }

    // Live pose analysis feedback
    var liveFeedbackBn by remember { mutableStateOf("ক্যামেরার সামনে মুখ রাখুন") }
    var liveFeedbackEn by remember { mutableStateOf("Position your face in front of camera") }
    var liveMatchScore by remember { mutableFloatStateOf(0f) }
    var autoProgressFraction by remember { mutableFloatStateOf(0f) }
    var goodFrameCounter by remember { mutableIntStateOf(0) }
    var lastStepChangeTime by remember { mutableLongStateOf(System.currentTimeMillis() + 800L) }

    // Live Timing & Progress Calculations
    val scanStartTime = remember { System.currentTimeMillis() }
    var elapsedSeconds by remember { mutableLongStateOf(0L) }

    LaunchedEffect(isFinished) {
        while (!isFinished) {
            kotlinx.coroutines.delay(500)
            elapsedSeconds = (System.currentTimeMillis() - scanStartTime) / 1000L
        }
    }

    // Dynamic total progress computation (0% to 100%)
    val totalProgressPct by remember(currentStepIndex, autoProgressFraction, isFinished) {
        derivedStateOf {
            if (isFinished) 100f
            else {
                val stepContribution = currentStepIndex.toFloat() / ENROLLMENT_STEPS.size.toFloat()
                val currentStepFraction = (autoProgressFraction.coerceIn(0f, 1f) / ENROLLMENT_STEPS.size.toFloat())
                ((stepContribution + currentStepFraction) * 100f).coerceIn(0f, 99f)
            }
        }
    }

    // Dynamic Estimated Time Remaining (seconds)
    val estimatedRemainingSeconds by remember(currentStepIndex, autoProgressFraction, isFinished) {
        derivedStateOf {
            if (isFinished) 0L
            else {
                val remainingFraction = (ENROLLMENT_STEPS.size - currentStepIndex - autoProgressFraction).coerceAtLeast(0.1f)
                (remainingFraction * 2.2f).toLong().coerceAtLeast(1L)
            }
        }
    }

    var previewViewRef by remember { mutableStateOf<PreviewView?>(null) }
    var cameraProviderRef by remember { mutableStateOf<ProcessCameraProvider?>(null) }

    // Audio & Haptic feedback utilities
    val toneGenerator = remember {
        try {
            ToneGenerator(AudioManager.STREAM_NOTIFICATION, 80)
        } catch (_: Exception) {
            null
        }
    }

    val vibrator = remember {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val vm = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
                vm?.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
            }
        } catch (_: Exception) {
            null
        }
    }

    // Pulsing animation for biometric focus
    val infiniteTransition = rememberInfiniteTransition(label = "biometric_pulse")
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1.0f,
        targetValue = 1.03f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse_scale"
    )

    // Directional nudge animation
    val arrowOffset by infiniteTransition.animateFloat(
        initialValue = -6f,
        targetValue = 6f,
        animationSpec = infiniteRepeatable(
            animation = tween(700, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "arrow_offset"
    )

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
            faceEngine.close()
            try {
                toneGenerator?.release()
            } catch (_: Exception) {}
        }
    }

    val currentStep = ENROLLMENT_STEPS.getOrElse(currentStepIndex) { ENROLLMENT_STEPS.last() }

    fun playCaptureFeedback() {
        try {
            toneGenerator?.startTone(ToneGenerator.TONE_PROP_BEEP, 120)
        } catch (_: Exception) {}

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator?.vibrate(VibrationEffect.createOneShot(60, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                @Suppress("DEPRECATION")
                vibrator?.vibrate(60)
            }
        } catch (_: Exception) {}
    }

    fun processAndSaveCroppedFace(sourceBitmap: Bitmap) {
        if (isCapturing || isFinished) return
        isCapturing = true
        isFlashVisible = true
        playCaptureFeedback()

        coroutineScope.launch(Dispatchers.Default) {
            try {
                val faces = faceEngine.detectFaces(sourceBitmap, maxFaces = 1)
                val croppedSample = if (faces.isNotEmpty()) {
                    val box = faces.first()
                    val w = sourceBitmap.width
                    val h = sourceBitmap.height
                    val x = (box.leftNorm * w).toInt().coerceIn(0, w - 1)
                    val y = (box.topNorm * h).toInt().coerceIn(0, h - 1)
                    val cw = ((box.rightNorm - box.leftNorm) * w).toInt().coerceIn(32, w - x)
                    val ch = ((box.bottomNorm - box.topNorm) * h).toInt().coerceIn(32, h - y)
                    Bitmap.createBitmap(sourceBitmap, x, y, cw, ch)
                } else {
                    val cw = (sourceBitmap.width * 0.70f).toInt()
                    val ch = (sourceBitmap.height * 0.70f).toInt()
                    val cx = (sourceBitmap.width - cw) / 2
                    val cy = (sourceBitmap.height - ch) / 2
                    Bitmap.createBitmap(sourceBitmap, cx, cy, cw, ch)
                }

                val scaled = Bitmap.createScaledBitmap(croppedSample, 480, 480, true)

                withContext(Dispatchers.Main) {
                    capturedAnglePhotos.add(scaled)
                    goodFrameCounter = 0
                    autoProgressFraction = 0f
                    lastStepChangeTime = System.currentTimeMillis() + 1100L

                    if (currentStepIndex + 1 < ENROLLMENT_STEPS.size) {
                        currentStepIndex += 1
                    } else {
                        isFinished = true
                    }
                }

                delay(400)
                withContext(Dispatchers.Main) {
                    isFlashVisible = false
                    isCapturing = false
                }
            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    isFlashVisible = false
                    isCapturing = false
                }
            }
        }
    }

    fun captureCurrentAngleManually() {
        val pv = previewViewRef ?: return
        val currentBitmap = pv.bitmap ?: return
        processAndSaveCroppedFace(currentBitmap)
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Surface(
            modifier = Modifier
                .fillMaxSize()
                .background(Color(0xFF0B1120))
                .testTag("face_enrollment_dialog"),
            color = Color(0xFF0B1120)
        ) {
            if (isFinished) {
                // Completion Confirmation Screen
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Box(
                        modifier = Modifier
                            .size(88.dp)
                            .clip(CircleShape)
                            .background(Color(0xFF10B981).copy(alpha = 0.2f))
                            .border(2.5.dp, Color(0xFF10B981), CircleShape),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Default.CheckCircle,
                            contentDescription = null,
                            tint = Color(0xFF10B981),
                            modifier = Modifier.size(54.dp)
                        )
                    }

                    Spacer(modifier = Modifier.height(20.dp))

                    Text(
                        text = "স্বয়ংক্রিয় বায়োমেট্রিক স্ক্যান সম্পন্ন!",
                        style = MaterialTheme.typography.headlineSmall.copy(
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        ),
                        textAlign = TextAlign.Center
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Text(
                        text = "${capturedAnglePhotos.size}টি ফেস অ্যাঙ্গেল সফলভাবে ধারণ করা হয়েছে। এই ডেটাসেট দিয়ে মডেল ট্রেনিং করলে $personName-কে যেকোনো কোণ ও আলোতে নির্ভুলভাবে শনাক্ত করা যাবে।",
                        style = MaterialTheme.typography.bodyMedium.copy(
                            color = Color.White.copy(alpha = 0.85f),
                            lineHeight = 22.sp
                        ),
                        textAlign = TextAlign.Center
                    )

                    Spacer(modifier = Modifier.height(24.dp))

                    // Thumbnails Row
                    LazyRow(
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        modifier = Modifier.fillMaxWidth(),
                        contentPadding = PaddingValues(horizontal = 8.dp)
                    ) {
                        itemsIndexed(capturedAnglePhotos) { idx, photo ->
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Box {
                                    Image(
                                        bitmap = photo.asImageBitmap(),
                                        contentDescription = "Angle ${idx + 1}",
                                        contentScale = ContentScale.Crop,
                                        modifier = Modifier
                                            .size(68.dp)
                                            .clip(RoundedCornerShape(14.dp))
                                            .border(2.dp, Color(0xFF38BDF8), RoundedCornerShape(14.dp))
                                    )
                                    Surface(
                                        shape = CircleShape,
                                        color = Color(0xFF10B981),
                                        modifier = Modifier
                                            .size(20.dp)
                                            .align(Alignment.TopEnd)
                                    ) {
                                        Box(contentAlignment = Alignment.Center) {
                                            Icon(
                                                imageVector = Icons.Default.Check,
                                                contentDescription = null,
                                                tint = Color.Black,
                                                modifier = Modifier.size(14.dp)
                                            )
                                        }
                                    }
                                }
                                Spacer(modifier = Modifier.height(6.dp))
                                Text(
                                    text = ENROLLMENT_STEPS.getOrNull(idx)?.titleBn ?: "#${idx + 1}",
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        color = Color.White.copy(alpha = 0.75f),
                                        fontSize = 11.sp,
                                        fontWeight = FontWeight.Medium
                                    )
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(32.dp))

                    Button(
                        onClick = {
                            onPhotosEnrolled(capturedAnglePhotos.toList())
                            onDismiss()
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(52.dp)
                            .testTag("save_enrolled_faces_btn"),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF10B981),
                            contentColor = Color.White
                        ),
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Icon(imageVector = Icons.Default.Save, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "মডেলের জন্য সংরক্ষণ করুন (Save & Calibrate)",
                            fontWeight = FontWeight.Bold,
                            fontSize = 15.sp
                        )
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    OutlinedButton(
                        onClick = {
                            capturedAnglePhotos.clear()
                            currentStepIndex = 0
                            goodFrameCounter = 0
                            autoProgressFraction = 0f
                            isFinished = false
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(48.dp),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = Color.White.copy(alpha = 0.85f)
                        ),
                        border = BorderStroke(1.dp, Color.White.copy(alpha = 0.3f)),
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Icon(imageVector = Icons.Default.Refresh, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("পুনরায় নতুন করে স্ক্যান করুন (Retake All)")
                    }
                }
            } else {
                // Active Biometric Scanning Viewfinder with Auto-Guide
                Box(modifier = Modifier.fillMaxSize()) {
                    // 1. CameraX PreviewView (Front Camera)
                    AndroidView(
                        factory = { ctx ->
                            val pv = PreviewView(ctx).apply {
                                layoutParams = FrameLayout.LayoutParams(
                                    ViewGroup.LayoutParams.MATCH_PARENT,
                                    ViewGroup.LayoutParams.MATCH_PARENT
                                )
                                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                                scaleType = PreviewView.ScaleType.FILL_CENTER
                            }
                            previewViewRef = pv

                            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                            cameraProviderFuture.addListener({
                                val cameraProvider = cameraProviderFuture.get()
                                cameraProviderRef = cameraProvider

                                try {
                                    cameraProvider.unbindAll()

                                    val preview = Preview.Builder().build().also {
                                        it.setSurfaceProvider(pv.surfaceProvider)
                                    }

                                    val imageAnalysis = ImageAnalysis.Builder()
                                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                                        .setTargetResolution(android.util.Size(480, 480))
                                        .build()

                                    imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                                        try {
                                            val bmp = imageProxy.toBitmap()
                                            val rot = imageProxy.imageInfo.rotationDegrees
                                            val matrix = Matrix()
                                            if (rot != 0) matrix.postRotate(rot.toFloat())
                                            matrix.postScale(-1f, 1f) // Mirror front camera
                                            val mirrored = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)

                                            val analysis = faceEngine.analyzeFacePose(mirrored, currentStep.id)

                                            // Update UI state on Main Thread
                                            coroutineScope.launch(Dispatchers.Main) {
                                                isFaceDetectedInFrame = analysis.hasFace
                                                liveFaceBox = analysis.boundingBox
                                                liveFeedbackBn = analysis.feedbackMessageBn
                                                liveFeedbackEn = analysis.feedbackMessageEn
                                                liveMatchScore = analysis.poseMatchScore

                                                val now = System.currentTimeMillis()
                                                val isTransitioning = now < lastStepChangeTime

                                                if (isTransitioning) {
                                                    goodFrameCounter = 0
                                                    autoProgressFraction = 0f
                                                    val remainingSec = ((lastStepChangeTime - now) / 1000f) + 0.1f
                                                    liveFeedbackBn = "পরবর্তী কোণের জন্য প্রস্তুত হোন..."
                                                    liveFeedbackEn = "Get ready for the next angle (${String.format(Locale.US, "%.1f", remainingSec)}s)"
                                                } else {
                                                    val isStrictFaceValid = analysis.hasFace &&
                                                            analysis.boundingBox != null &&
                                                            analysis.eyeDistance >= 14f &&
                                                            analysis.poseMatchScore >= 0.80f

                                                    if (isAutoCaptureEnabled && isStrictFaceValid && !isCapturing && !isFinished) {
                                                        goodFrameCounter += 1
                                                        autoProgressFraction = (goodFrameCounter / 18f).coerceIn(0f, 1f)

                                                        if (goodFrameCounter >= 18) {
                                                            processAndSaveCroppedFace(mirrored)
                                                        }
                                                    } else {
                                                        if (!analysis.hasFace) {
                                                            goodFrameCounter = 0
                                                            autoProgressFraction = 0f
                                                        } else if (goodFrameCounter > 0) {
                                                            goodFrameCounter = (goodFrameCounter - 1).coerceAtLeast(0)
                                                            autoProgressFraction = (goodFrameCounter / 18f).coerceIn(0f, 1f)
                                                        }
                                                    }
                                                }
                                            }
                                        } catch (_: Throwable) {
                                        } finally {
                                            imageProxy.close()
                                        }
                                    }

                                    val cameraSelector = CameraSelector.Builder()
                                        .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                                        .build()

                                    cameraProvider.bindToLifecycle(
                                        lifecycleOwner,
                                        cameraSelector,
                                        preview,
                                        imageAnalysis
                                    )
                                } catch (e: Exception) {
                                    e.printStackTrace()
                                }
                            }, ContextCompat.getMainExecutor(ctx))

                            pv
                        },
                        modifier = Modifier.fillMaxSize()
                    )

                    // 2. Camera Shutter Flash Effect
                    AnimatedVisibility(
                        visible = isFlashVisible,
                        enter = fadeIn(animationSpec = tween(50)),
                        exit = fadeOut(animationSpec = tween(300)),
                        modifier = Modifier.fillMaxSize()
                    ) {
                        Box(modifier = Modifier.fillMaxSize().background(Color.White))
                    }

                    // 3. Dynamic AR Free-Motion Viewfinder & Face Tracker
                    BoxWithConstraints(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(top = 80.dp, bottom = 170.dp)
                    ) {
                        val containerWidth = maxWidth
                        val containerHeight = maxHeight
                        val isAligned = liveMatchScore >= 0.70f
                        val activeColor = if (isAligned) Color(0xFF10B981) else if (isFaceDetectedInFrame) Color(0xFF38BDF8) else Color.White.copy(alpha = 0.45f)

                        // 3A. Ambient Cyber Viewfinder Corners (Broad screen coverage, no restrictive oval)
                        Canvas(modifier = Modifier.fillMaxSize().padding(24.dp)) {
                            val stroke = 3.dp.toPx()
                            val cornerLen = 36.dp.toPx()
                            val w = size.width
                            val h = size.height
                            val c = activeColor.copy(alpha = 0.5f)

                            // Top-Left Corner
                            drawLine(c, Offset(0f, 0f), Offset(cornerLen, 0f), stroke)
                            drawLine(c, Offset(0f, 0f), Offset(0f, cornerLen), stroke)
                            // Top-Right Corner
                            drawLine(c, Offset(w, 0f), Offset(w - cornerLen, 0f), stroke)
                            drawLine(c, Offset(w, 0f), Offset(w, cornerLen), stroke)
                            // Bottom-Left Corner
                            drawLine(c, Offset(0f, h), Offset(cornerLen, h), stroke)
                            drawLine(c, Offset(0f, h), Offset(0f, h - cornerLen), stroke)
                            // Bottom-Right Corner
                            drawLine(c, Offset(w, h), Offset(w - cornerLen, h), stroke)
                            drawLine(c, Offset(w, h), Offset(w, h - cornerLen), stroke)
                        }

                        // 3B. Dynamic Live Face Tracking Box (Moves smoothly wherever the user's face is)
                        val box = liveFaceBox
                        val targetLeft = if (box != null) (box.leftNorm * containerWidth.value).dp else (containerWidth * 0.18f)
                        val targetTop = if (box != null) (box.topNorm * containerHeight.value).dp else (containerHeight * 0.16f)
                        val targetWidth = if (box != null) ((box.rightNorm - box.leftNorm) * containerWidth.value).dp.coerceAtLeast(140.dp) else (containerWidth * 0.64f)
                        val targetHeight = if (box != null) ((box.bottomNorm - box.topNorm) * containerHeight.value).dp.coerceAtLeast(160.dp) else (containerHeight * 0.55f)

                        val animLeft by animateDpAsState(targetValue = targetLeft, animationSpec = spring(stiffness = Spring.StiffnessMediumLow), label = "face_x")
                        val animTop by animateDpAsState(targetValue = targetTop, animationSpec = spring(stiffness = Spring.StiffnessMediumLow), label = "face_y")
                        val animW by animateDpAsState(targetValue = targetWidth, animationSpec = spring(stiffness = Spring.StiffnessMediumLow), label = "face_w")
                        val animH by animateDpAsState(targetValue = targetHeight, animationSpec = spring(stiffness = Spring.StiffnessMediumLow), label = "face_h")

                        Box(
                            modifier = Modifier
                                .offset(x = animLeft, y = animTop)
                                .size(width = animW, height = animH)
                                .scale(if (isAligned) pulseScale else 1.0f)
                        ) {
                            // High-Tech AR Target Frame around detected face
                            Canvas(modifier = Modifier.fillMaxSize()) {
                                val stroke = if (isAligned) 3.5.dp.toPx() else 2.5.dp.toPx()
                                val bracketLen = (size.minDimension * 0.22f).coerceIn(16f, 42f)
                                val w = size.width
                                val h = size.height

                                // 4 Dynamic target brackets around the face
                                drawLine(activeColor, Offset(0f, 0f), Offset(bracketLen, 0f), stroke)
                                drawLine(activeColor, Offset(0f, 0f), Offset(0f, bracketLen), stroke)

                                drawLine(activeColor, Offset(w, 0f), Offset(w - bracketLen, 0f), stroke)
                                drawLine(activeColor, Offset(w, 0f), Offset(w, bracketLen), stroke)

                                drawLine(activeColor, Offset(0f, h), Offset(bracketLen, h), stroke)
                                drawLine(activeColor, Offset(0f, h), Offset(0f, h - bracketLen), stroke)

                                drawLine(activeColor, Offset(w, h), Offset(w - bracketLen, h), stroke)
                                drawLine(activeColor, Offset(w, h), Offset(w, h - bracketLen), stroke)

                                // Auto capture progress border
                                if (isAutoCaptureEnabled && autoProgressFraction > 0.05f) {
                                    drawArc(
                                        brush = Brush.sweepGradient(
                                            listOf(Color(0xFF10B981), Color(0xFF34D399), Color(0xFF06B6D4))
                                        ),
                                        startAngle = -90f,
                                        sweepAngle = 360f * autoProgressFraction,
                                        useCenter = false,
                                        style = Stroke(width = 5.dp.toPx(), cap = StrokeCap.Round),
                                        size = Size(w, h)
                                    )
                                }
                            }

                            // Dynamic Live Match / Alignment Indicator Chip
                            if (isFaceDetectedInFrame) {
                                Surface(
                                    shape = RoundedCornerShape(12.dp),
                                    color = if (isAligned) Color(0xFF10B981).copy(alpha = 0.9f) else Color.Black.copy(alpha = 0.7f),
                                    border = BorderStroke(1.dp, activeColor),
                                    modifier = Modifier
                                        .align(Alignment.TopCenter)
                                        .offset(y = (-16).dp)
                                ) {
                                    Row(
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Icon(
                                            imageVector = if (isAligned) Icons.Default.CheckCircle else Icons.Default.Face,
                                            contentDescription = null,
                                            tint = Color.White,
                                            modifier = Modifier.size(13.dp)
                                        )
                                        Spacer(modifier = Modifier.width(4.dp))
                                        Text(
                                            text = if (isAligned) "ম্যাচ হয়েছে (Ready)" else "মুখ ট্র্যাকিং হচ্ছে",
                                            style = MaterialTheme.typography.labelSmall.copy(
                                                fontSize = 10.sp,
                                                fontWeight = FontWeight.Bold,
                                                color = Color.White
                                            )
                                        )
                                    }
                                }
                            }

                            // Directional Visual Pointer / Turn Arrow
                            when (currentStep.id) {
                                2 -> { // Left
                                    Surface(
                                        shape = CircleShape,
                                        color = Color(0xFF0284C7).copy(alpha = 0.9f),
                                        border = BorderStroke(1.5.dp, Color.White),
                                        modifier = Modifier
                                            .align(Alignment.CenterStart)
                                            .offset(x = (-22).dp + arrowOffset.dp)
                                            .size(46.dp)
                                    ) {
                                        Box(contentAlignment = Alignment.Center) {
                                            Icon(
                                                imageVector = Icons.Default.ArrowBack,
                                                contentDescription = "Turn Left",
                                                tint = Color.White,
                                                modifier = Modifier.size(26.dp)
                                            )
                                        }
                                    }
                                }
                                3 -> { // Right
                                    Surface(
                                        shape = CircleShape,
                                        color = Color(0xFF0284C7).copy(alpha = 0.9f),
                                        border = BorderStroke(1.5.dp, Color.White),
                                        modifier = Modifier
                                            .align(Alignment.CenterEnd)
                                            .offset(x = (22).dp - arrowOffset.dp)
                                            .size(46.dp)
                                    ) {
                                        Box(contentAlignment = Alignment.Center) {
                                            Icon(
                                                imageVector = Icons.AutoMirrored.Filled.ArrowForward,
                                                contentDescription = "Turn Right",
                                                tint = Color.White,
                                                modifier = Modifier.size(26.dp)
                                            )
                                        }
                                    }
                                }
                                4 -> { // Up
                                    Surface(
                                        shape = CircleShape,
                                        color = Color(0xFF0284C7).copy(alpha = 0.9f),
                                        border = BorderStroke(1.5.dp, Color.White),
                                        modifier = Modifier
                                            .align(Alignment.TopCenter)
                                            .offset(y = (-24).dp + arrowOffset.dp)
                                            .size(46.dp)
                                    ) {
                                        Box(contentAlignment = Alignment.Center) {
                                            Icon(
                                                imageVector = Icons.Default.ArrowUpward,
                                                contentDescription = "Tilt Up",
                                                tint = Color.White,
                                                modifier = Modifier.size(26.dp)
                                            )
                                        }
                                    }
                                }
                            }
                        }

                        // Target Angle Emoji Badge at bottom center of tracking area
                        Surface(
                            shape = CircleShape,
                            color = Color(0xFF0F172A).copy(alpha = 0.85f),
                            border = BorderStroke(2.dp, activeColor),
                            modifier = Modifier
                                .size(54.dp)
                                .align(Alignment.BottomCenter)
                                .offset(y = 24.dp)
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Text(
                                    text = currentStep.iconEmoji,
                                    fontSize = 26.sp
                                )
                            }
                        }
                    }

                    // 4. Top Header with Step Dots & Auto-Capture Toggle
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .statusBarsPadding()
                            .padding(horizontal = 16.dp, vertical = 12.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            IconButton(
                                onClick = onDismiss,
                                modifier = Modifier
                                    .size(38.dp)
                                    .clip(CircleShape)
                                    .background(Color.Black.copy(alpha = 0.6f))
                                    .testTag("close_face_enrollment_btn")
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Close,
                                    contentDescription = "Cancel",
                                    tint = Color.White
                                )
                            }

                            // Auto-Capture Mode Switch Chip
                            Surface(
                                shape = RoundedCornerShape(20.dp),
                                color = if (isAutoCaptureEnabled) Color(0xFF10B981).copy(alpha = 0.25f) else Color.Black.copy(alpha = 0.6f),
                                border = BorderStroke(1.dp, if (isAutoCaptureEnabled) Color(0xFF10B981) else Color.White.copy(alpha = 0.3f)),
                                modifier = Modifier.clickable {
                                    isAutoCaptureEnabled = !isAutoCaptureEnabled
                                    goodFrameCounter = 0
                                    autoProgressFraction = 0f
                                }
                            ) {
                                Row(
                                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Icon(
                                        imageVector = if (isAutoCaptureEnabled) Icons.Default.AutoAwesome else Icons.Default.TouchApp,
                                        contentDescription = null,
                                        tint = if (isAutoCaptureEnabled) Color(0xFF10B981) else Color.White,
                                        modifier = Modifier.size(16.dp)
                                    )
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text(
                                        text = if (isAutoCaptureEnabled) "⚡ স্বয়ংক্রিয় ক্যাপচার (Auto: ON)" else "👆 ম্যানুয়াল ট্যাপ (Manual)",
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontWeight = FontWeight.Bold,
                                            color = if (isAutoCaptureEnabled) Color(0xFF10B981) else Color.White
                                        )
                                    )
                                }
                            }

                            // Step Progress Count & Live Percentage
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                Surface(
                                    shape = RoundedCornerShape(16.dp),
                                    color = Color(0xFF10B981).copy(alpha = 0.25f),
                                    border = BorderStroke(1.dp, Color(0xFF10B981))
                                ) {
                                    Text(
                                        text = "${totalProgressPct.toInt()}%",
                                        style = MaterialTheme.typography.labelMedium.copy(
                                            fontWeight = FontWeight.ExtraBold,
                                            color = Color(0xFF34D399)
                                        ),
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                    )
                                }

                                Surface(
                                    shape = RoundedCornerShape(16.dp),
                                    color = Color.Black.copy(alpha = 0.6f),
                                    border = BorderStroke(1.dp, Color.White.copy(alpha = 0.2f))
                                ) {
                                    Text(
                                        text = "${currentStepIndex + 1}/${ENROLLMENT_STEPS.size}",
                                        style = MaterialTheme.typography.labelMedium.copy(
                                            fontWeight = FontWeight.Bold,
                                            color = Color.White
                                        ),
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                    )
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(6.dp))

                        // Live Status Bar: ETA remaining & Elapsed time
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 8.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    imageVector = Icons.Default.HourglassBottom,
                                    contentDescription = null,
                                    tint = Color(0xFF38BDF8),
                                    modifier = Modifier.size(13.dp)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text(
                                    text = "বাকি সময়: ~${estimatedRemainingSeconds} সেকেন্ড",
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        fontSize = 10.5.sp,
                                        fontWeight = FontWeight.Bold,
                                        color = Color(0xFF38BDF8)
                                    )
                                )
                            }

                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    imageVector = Icons.Default.AccessTime,
                                    contentDescription = null,
                                    tint = Color.White.copy(alpha = 0.6f),
                                    modifier = Modifier.size(13.dp)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text(
                                    text = "সময়: ${elapsedSeconds}s",
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        fontSize = 10.5.sp,
                                        color = Color.White.copy(alpha = 0.7f)
                                    )
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(6.dp))

                        // Step Progression Bar
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 8.dp),
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            ENROLLMENT_STEPS.forEachIndexed { idx, step ->
                                val isDone = idx < currentStepIndex
                                val isCurrent = idx == currentStepIndex
                                Box(
                                    modifier = Modifier
                                        .weight(1f)
                                        .height(5.dp)
                                        .clip(RoundedCornerShape(2.5.dp))
                                        .background(
                                            when {
                                                isDone -> Color(0xFF10B981)
                                                isCurrent -> Color(0xFF38BDF8)
                                                else -> Color.White.copy(alpha = 0.25f)
                                            }
                                        )
                                )
                            }
                        }
                    }

                    // 5. Bottom Instruction & Action Card
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .align(Alignment.BottomCenter)
                            .navigationBarsPadding()
                            .padding(horizontal = 16.dp, vertical = 12.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        // Live Smart Speech-Bubble Guidance Card
                        Surface(
                            shape = RoundedCornerShape(22.dp),
                            color = Color(0xFF0F172A).copy(alpha = 0.88f),
                            border = BorderStroke(
                                1.5.dp,
                                if (liveMatchScore >= 0.75f) Color(0xFF10B981) else Color(0xFF38BDF8).copy(alpha = 0.5f)
                            ),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column(
                                modifier = Modifier.padding(14.dp),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.Center
                                ) {
                                    Text(
                                        text = currentStep.iconEmoji,
                                        fontSize = 20.sp
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = "${currentStep.titleBn} (${currentStep.hintDirection})",
                                        style = MaterialTheme.typography.titleMedium.copy(
                                            fontWeight = FontWeight.Bold,
                                            color = if (liveMatchScore >= 0.75f) Color(0xFF10B981) else Color.White
                                        )
                                    )
                                }

                                Spacer(modifier = Modifier.height(4.dp))

                                Text(
                                    text = if (isAutoCaptureEnabled && autoProgressFraction > 0.1f) {
                                        "📸 সুন্দর! স্থির থাকুন... (${(autoProgressFraction * 100).toInt()}%)"
                                    } else {
                                        liveFeedbackBn
                                    },
                                    style = MaterialTheme.typography.bodyMedium.copy(
                                        fontWeight = FontWeight.SemiBold,
                                        color = if (liveMatchScore >= 0.75f) Color(0xFF34D399) else Color.White.copy(alpha = 0.85f)
                                    ),
                                    textAlign = TextAlign.Center
                                )

                                Spacer(modifier = Modifier.height(4.dp))

                                // Live Mini ETA & Percentage status inside HUD
                                Surface(
                                    shape = RoundedCornerShape(12.dp),
                                    color = Color.White.copy(alpha = 0.08f),
                                    modifier = Modifier.padding(vertical = 2.dp)
                                ) {
                                    Text(
                                        text = "মোট অগ্রগতি: ${totalProgressPct.toInt()}% • আনুমানিক আর ~${estimatedRemainingSeconds} সে. বাকি",
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontSize = 11.sp,
                                            fontWeight = FontWeight.Medium,
                                            color = Color(0xFF38BDF8)
                                        ),
                                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 3.dp)
                                    )
                                }

                                Spacer(modifier = Modifier.height(2.dp))

                                Text(
                                    text = currentStep.instructionEn,
                                    style = MaterialTheme.typography.bodySmall.copy(
                                        color = Color.White.copy(alpha = 0.55f),
                                        fontSize = 10.5.sp
                                    ),
                                    textAlign = TextAlign.Center
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(12.dp))

                        // Controls Row
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceEvenly,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            // Skip Step Button
                            TextButton(
                                onClick = {
                                    goodFrameCounter = 0
                                    autoProgressFraction = 0f
                                    if (currentStepIndex + 1 < ENROLLMENT_STEPS.size) {
                                        currentStepIndex += 1
                                    } else {
                                        isFinished = true
                                    }
                                },
                                colors = ButtonDefaults.textButtonColors(
                                    contentColor = Color.White.copy(alpha = 0.75f)
                                )
                            ) {
                                Text("স্কিপ (Skip)")
                            }

                            // Center Shutter Button (Manual tap or visual progress)
                            Box(
                                modifier = Modifier
                                    .size(72.dp)
                                    .clip(CircleShape)
                                    .border(3.5.dp, Color.White, CircleShape)
                                    .clickable { captureCurrentAngleManually() }
                                    .testTag("face_enroll_capture_btn"),
                                contentAlignment = Alignment.Center
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(56.dp)
                                        .clip(CircleShape)
                                        .background(
                                            if (liveMatchScore >= 0.75f) Color(0xFF10B981) else Color(0xFF38BDF8)
                                        ),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        imageVector = if (isAutoCaptureEnabled) Icons.Default.AutoAwesome else Icons.Default.CameraAlt,
                                        contentDescription = "Capture Angle",
                                        tint = Color.Black,
                                        modifier = Modifier.size(28.dp)
                                    )
                                }
                            }

                            // Finish early if at least 1 photo captured
                            TextButton(
                                onClick = {
                                    if (capturedAnglePhotos.isNotEmpty()) {
                                        isFinished = true
                                    }
                                },
                                enabled = capturedAnglePhotos.isNotEmpty(),
                                colors = ButtonDefaults.textButtonColors(
                                    contentColor = if (capturedAnglePhotos.isNotEmpty()) Color(0xFF10B981) else Color.Gray
                                )
                            ) {
                                Text("সমাপ্ত (Done)")
                            }
                        }
                    }
                }
            }
        }
    }
}
