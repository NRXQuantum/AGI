package com.example.service

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import com.example.data.db.ProjectEntity
import com.example.data.repository.ProjectRepository
import com.example.ml.PredictionResult
import com.example.util.AppLogger
import com.example.util.ImageUtils
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale

data class TelegramBotLog(
    val id: Long = System.currentTimeMillis(),
    val timestamp: Long = System.currentTimeMillis(),
    val message: String,
    val isError: Boolean = false,
    val senderName: String? = null
)

class TelegramBotService private constructor(private val appContext: Context) {

    private val serviceScope = CoroutineScope(Dispatchers.IO + Job())
    private var pollingJob: Job? = null

    private val _isRunning = MutableStateFlow(false)
    val isRunning: StateFlow<Boolean> = _isRunning.asStateFlow()

    private val _botUsername = MutableStateFlow<String?>(null)
    val botUsername: StateFlow<String?> = _botUsername.asStateFlow()

    private val _activeProjectName = MutableStateFlow<String?>(null)
    val activeProjectName: StateFlow<String?> = _activeProjectName.asStateFlow()

    private val _logs = MutableStateFlow<List<TelegramBotLog>>(emptyList())
    val logs: StateFlow<List<TelegramBotLog>> = _logs.asStateFlow()

    private val _processedCount = MutableStateFlow(0)
    val processedCount: StateFlow<Int> = _processedCount.asStateFlow()

    private var currentBotToken: String = ""
    private var currentProjectId: Long = -1L
    private var lastUpdateId: Long = 0L

    fun startBot(
        botToken: String,
        project: ProjectEntity,
        repository: ProjectRepository
    ) {
        if (_isRunning.value) {
            stopBot()
        }

        currentBotToken = botToken.trim()
        currentProjectId = project.id
        _activeProjectName.value = project.name
        _isRunning.value = true
        addLog("🚀 Starting Telegram Bot with AI Model: '${project.name}'")

        pollingJob = serviceScope.launch {
            // Verify Bot Token
            val botInfo = getMe(currentBotToken)
            if (botInfo == null) {
                addLog("❌ Invalid Bot Token! Please verify token from @BotFather", isError = true)
                _isRunning.value = false
                return@launch
            }

            _botUsername.value = botInfo
            addLog("✅ Connected as @$botInfo | Ready to classify photos!")

            while (isActive && _isRunning.value) {
                try {
                    val updates = getUpdates(currentBotToken, lastUpdateId + 1)
                    if (updates != null && updates.length() > 0) {
                        for (i in 0 until updates.length()) {
                            val update = updates.getJSONObject(i)
                            val updateId = update.optLong("update_id", 0)
                            if (updateId > lastUpdateId) {
                                lastUpdateId = updateId
                            }

                            if (update.has("message")) {
                                val message = update.getJSONObject("message")
                                val chatId = message.getJSONObject("chat").getLong("id")
                                val fromUser = message.optJSONObject("from")
                                val senderName = fromUser?.optString("first_name", "User") ?: "User"

                                // Handle text commands
                                if (message.has("text")) {
                                    val text = message.getString("text")
                                    if (text.startsWith("/start") || text.startsWith("/help")) {
                                        val welcomeMsg = """
                                            🤖 <b>Welcome to ${project.name} AI Classifier Bot!</b>
                                            
                                            Send or upload any <b>photo/image</b> to this chat, and I will instantly classify it using the on-device neural network!
                                            
                                            🎯 <b>Active Model:</b> <code>${project.name}</code>
                                            📊 <b>Trained Accuracy:</b> ${String.format(Locale.US, "%.1f%%", project.trainingAccuracy * 100)}
                                            ⚡ <b>Zero-Retention:</b> Images are auto-deleted immediately after analysis.
                                        """.trimIndent()
                                        sendMessage(currentBotToken, chatId, welcomeMsg)
                                        addLog("💬 Sent welcome message to $senderName")
                                    } else {
                                        sendMessage(currentBotToken, chatId, "📸 Please send an image to detect and classify with <b>${project.name}</b>!")
                                    }
                                }

                                // Handle photo or image document
                                val documentObj = message.optJSONObject("document")
                                val isImageDoc = documentObj != null && documentObj.optString("mime_type", "").startsWith("image/")

                                if (message.has("photo") || isImageDoc) {
                                    val fileId = if (message.has("photo")) {
                                        val photos = message.getJSONArray("photo")
                                        photos.getJSONObject(photos.length() - 1).getString("file_id")
                                    } else {
                                        documentObj!!.getString("file_id")
                                    }

                                    addLog("📸 Received image from $senderName! Analyzing...", senderName = senderName)
                                    sendMessage(currentBotToken, chatId, "🔍 Analyzing image with <b>${project.name}</b> AI model...\n<i>(Storage auto-purged immediately)</i>")

                                    val (tempFile, bitmap) = downloadTelegramPhotoWithFile(currentBotToken, fileId)
                                    try {
                                        if (bitmap != null) {
                                            val result: PredictionResult? = repository.runInference(project.id, bitmap, false)

                                            if (result != null) {
                                                _processedCount.value += 1
                                                addLog("🎯 Detected: '${result.classLabel}' (${String.format(Locale.US, "%.1f%%", result.confidence * 100)}) for $senderName")

                                                val replyBuilder = StringBuilder()
                                                replyBuilder.append("✨ <b>Detection Result:</b>\n\n")
                                                replyBuilder.append("🏷️ <b>Primary Object:</b> <b>${result.classLabel}</b>\n")
                                                replyBuilder.append("🎯 <b>Confidence:</b> <b>${String.format(Locale.US, "%.1f%%", result.confidence * 100)}</b>\n")
                                                replyBuilder.append("⚡ <b>Latency:</b> ${result.inferenceTimeMs} ms\n\n")

                                                if (result.allProbabilities.isNotEmpty()) {
                                                    replyBuilder.append("📊 <b>Top Predictions:</b>\n")
                                                    result.allProbabilities.take(5).forEachIndexed { index, conf ->
                                                        val medal = when (index) {
                                                            0 -> "🥇"
                                                            1 -> "🥈"
                                                            2 -> "🥉"
                                                            else -> "🔹"
                                                        }
                                                        replyBuilder.append("$medal <b>${conf.classLabel}</b>: <code>${String.format(Locale.US, "%.1f%%", conf.probability * 100)}</code>\n")
                                                    }
                                                }

                                                replyBuilder.append("\n🤖 <i>Powered by ${project.name} On-Device AI</i>")
                                                sendMessage(currentBotToken, chatId, replyBuilder.toString())
                                            } else {
                                                sendMessage(currentBotToken, chatId, "⚠️ Error: Active model could not run inference.")
                                                addLog("❌ Model inference returned null for project ${project.name}", isError = true)
                                            }
                                        } else {
                                            sendMessage(currentBotToken, chatId, "❌ Failed to download photo. Please try again.")
                                            addLog("❌ Download failed for file $fileId", isError = true)
                                        }
                                    } finally {
                                        // IMMEDIATE AUTO-CLEANUP: Recycle bitmap and delete temp file
                                        try {
                                            bitmap?.let {
                                                if (!it.isRecycled) it.recycle()
                                            }
                                            tempFile?.let {
                                                if (it.exists()) it.delete()
                                            }
                                            addLog("🧹 Auto-Cleaned: Temporary file and memory freed (0 bytes retained).")
                                        } catch (e: Exception) {
                                            AppLogger.w("TelegramBotService", "Cleanup exception: ${e.message}")
                                        }
                                    }
                                }
                            }
                        }
                    }
                    delay(1200)
                } catch (e: Exception) {
                    if (isActive && _isRunning.value) {
                        AppLogger.e("TelegramBotService", "Polling error: ${e.message}", e)
                        addLog("⚠️ Polling issue: ${e.message ?: "Network timeout"}", isError = true)
                        delay(3500)
                    }
                }
            }
        }
    }

    fun stopBot() {
        pollingJob?.cancel()
        pollingJob = null
        _isRunning.value = false
        addLog("🛑 Telegram Bot stopped.")
    }

    fun clearLogs() {
        _logs.value = emptyList()
    }

    private fun addLog(message: String, isError: Boolean = false, senderName: String? = null) {
        val newLog = TelegramBotLog(
            message = message,
            isError = isError,
            senderName = senderName
        )
        _logs.value = (_logs.value + newLog).takeLast(60)
    }

    private suspend fun getMe(token: String): String? = withContext(Dispatchers.IO) {
        try {
            val url = URL("https://api.telegram.org/bot$token/getMe")
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "GET"
            conn.connectTimeout = 8000
            conn.readTimeout = 8000

            if (conn.responseCode == 200) {
                val reader = BufferedReader(InputStreamReader(conn.inputStream))
                val response = reader.readText()
                reader.close()
                val json = JSONObject(response)
                if (json.optBoolean("ok")) {
                    return@withContext json.getJSONObject("result").optString("username", "ClassifierBot")
                }
            }
        } catch (e: Exception) {
            AppLogger.e("TelegramBotService", "getMe failed", e)
        }
        null
    }

    private suspend fun getUpdates(token: String, offset: Long): org.json.JSONArray? = withContext(Dispatchers.IO) {
        try {
            val url = URL("https://api.telegram.org/bot$token/getUpdates?offset=$offset&timeout=10")
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "GET"
            conn.connectTimeout = 15000
            conn.readTimeout = 15000

            if (conn.responseCode == 200) {
                val reader = BufferedReader(InputStreamReader(conn.inputStream))
                val response = reader.readText()
                reader.close()
                val json = JSONObject(response)
                if (json.optBoolean("ok")) {
                    return@withContext json.getJSONArray("result")
                }
            }
        } catch (e: Exception) {
            // timeout or polling wait
        }
        null
    }

    private suspend fun sendMessage(token: String, chatId: Long, text: String) = withContext(Dispatchers.IO) {
        try {
            val url = URL("https://api.telegram.org/bot$token/sendMessage")
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.setRequestProperty("Content-Type", "application/json; utf-8")
            conn.doOutput = true
            conn.connectTimeout = 10000
            conn.readTimeout = 10000

            val jsonBody = JSONObject().apply {
                put("chat_id", chatId)
                put("parse_mode", "HTML")
                put("text", text)
            }

            val writer = OutputStreamWriter(conn.outputStream)
            writer.write(jsonBody.toString())
            writer.flush()
            writer.close()

            conn.responseCode
        } catch (e: Exception) {
            AppLogger.e("TelegramBotService", "sendMessage failed", e)
        }
    }

    private suspend fun downloadTelegramPhotoWithFile(token: String, fileId: String): Pair<File?, Bitmap?> = withContext(Dispatchers.IO) {
        var tempFile: File? = null
        try {
            val getFileUrl = URL("https://api.telegram.org/bot$token/getFile?file_id=$fileId")
            val conn = getFileUrl.openConnection() as HttpURLConnection
            conn.requestMethod = "GET"
            conn.connectTimeout = 10000

            var filePath: String? = null
            if (conn.responseCode == 200) {
                val reader = BufferedReader(InputStreamReader(conn.inputStream))
                val resp = reader.readText()
                reader.close()
                val json = JSONObject(resp)
                if (json.optBoolean("ok")) {
                    filePath = json.getJSONObject("result").optString("file_path")
                }
            }

            if (filePath != null) {
                val downloadUrl = URL("https://api.telegram.org/file/bot$token/$filePath")
                val downloadConn = downloadUrl.openConnection() as HttpURLConnection
                downloadConn.connectTimeout = 15000
                downloadConn.readTimeout = 15000

                if (downloadConn.responseCode == 200) {
                    val cacheDir = appContext.cacheDir
                    tempFile = File(cacheDir, "tg_bot_${System.currentTimeMillis()}_${fileId.takeLast(6)}.tmp")

                    downloadConn.inputStream.use { input ->
                        tempFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }

                    val bitmap = ImageUtils.decodeOrientedBitmap(tempFile, maxDim = 1280)
                    return@withContext Pair(tempFile, bitmap)
                }
            }
        } catch (e: Exception) {
            AppLogger.e("TelegramBotService", "downloadTelegramPhotoWithFile failed", e)
            tempFile?.let { if (it.exists()) it.delete() }
        }
        Pair(null, null)
    }

    companion object {
        @Volatile
        private var instance: TelegramBotService? = null

        fun getInstance(context: Context): TelegramBotService {
            return instance ?: synchronized(this) {
                instance ?: TelegramBotService(context.applicationContext).also { instance = it }
            }
        }
    }
}
