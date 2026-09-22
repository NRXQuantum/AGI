package com.example.util

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

enum class LogLevel(val tag: String) {
    DEBUG("DEBUG"),
    INFO("INFO"),
    WARN("WARN"),
    ERROR("ERROR"),
    CRITICAL("CRASH/FATAL")
}

data class AppLogEntry(
    val id: Long = System.nanoTime(),
    val timestamp: Long = System.currentTimeMillis(),
    val level: LogLevel,
    val tag: String,
    val message: String,
    val throwableStackTrace: String? = null
) {
    fun formattedTime(): String {
        val sdf = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)
        return sdf.format(Date(timestamp))
    }

    fun toFormattedString(): String {
        val base = "[${formattedTime()}] [${level.name}] [$tag] $message"
        return if (throwableStackTrace != null) {
            "$base\n$throwableStackTrace"
        } else {
            base
        }
    }
}

object AppLogger {
    private const val MAX_LOGS = 2000
    private val _logs = MutableStateFlow<List<AppLogEntry>>(emptyList())
    val logs: StateFlow<List<AppLogEntry>> = _logs.asStateFlow()

    private var defaultExceptionHandler: Thread.UncaughtExceptionHandler? = null
    private var isInitialized = false

    fun init() {
        if (isInitialized) return
        isInitialized = true

        i("AppLogger", "=== App session started. Logger initialized ===")
        logSystemDiagnostics()

        defaultExceptionHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            val sw = StringWriter()
            throwable.printStackTrace(PrintWriter(sw))
            val stackTrace = sw.toString()
            
            log(
                level = LogLevel.CRITICAL,
                tag = "CRASH",
                message = "FATAL UNCAUGHT EXCEPTION in thread '${thread.name}': ${throwable.message}",
                throwable = throwable
            )
            
            // Allow system or default handler to proceed
            defaultExceptionHandler?.uncaughtException(thread, throwable)
        }
    }

    private fun logSystemDiagnostics() {
        val runtime = Runtime.getRuntime()
        val maxMemMb = runtime.maxMemory() / (1024 * 1024)
        val totalMemMb = runtime.totalMemory() / (1024 * 1024)
        val freeMemMb = runtime.freeMemory() / (1024 * 1024)
        val availCores = runtime.availableProcessors()
        i(
            "SystemDiagnostics",
            "Hardware: $availCores cores | Max Heap: ${maxMemMb}MB | Total Alloc: ${totalMemMb}MB | Free: ${freeMemMb}MB"
        )
    }

    @Synchronized
    fun log(level: LogLevel, tag: String, message: String, throwable: Throwable? = null) {
        val stackTrace = throwable?.let {
            val sw = StringWriter()
            it.printStackTrace(PrintWriter(sw))
            sw.toString()
        }

        val entry = AppLogEntry(
            level = level,
            tag = tag,
            message = message,
            throwableStackTrace = stackTrace
        )

        // Native Android Logcat logging (safely caught in local JVM unit tests)
        try {
            when (level) {
                LogLevel.DEBUG -> Log.d(tag, message, throwable)
                LogLevel.INFO -> Log.i(tag, message, throwable)
                LogLevel.WARN -> Log.w(tag, message, throwable)
                LogLevel.ERROR -> Log.e(tag, message, throwable)
                LogLevel.CRITICAL -> Log.e(tag, "FATAL: $message", throwable)
            }
        } catch (_: Throwable) {
            println("[$tag] [${level.name}] $message")
        }

        // State update
        val currentList = _logs.value
        val newList = if (currentList.size >= MAX_LOGS) {
            currentList.drop(currentList.size - MAX_LOGS + 1) + entry
        } else {
            currentList + entry
        }
        _logs.value = newList
    }

    fun d(tag: String, message: String) = log(LogLevel.DEBUG, tag, message)
    fun i(tag: String, message: String) = log(LogLevel.INFO, tag, message)
    fun w(tag: String, message: String, throwable: Throwable? = null) = log(LogLevel.WARN, tag, message, throwable)
    fun e(tag: String, message: String, throwable: Throwable? = null) = log(LogLevel.ERROR, tag, message, throwable)
    fun crash(tag: String, message: String, throwable: Throwable) = log(LogLevel.CRITICAL, tag, message, throwable)

    fun clearLogs() {
        _logs.value = emptyList()
        i("AppLogger", "Logs cleared by user")
    }

    fun getFullLogText(): String {
        val sb = StringBuilder()
        sb.append("=== VISION MODEL STUDIO - COMPLETE DIAGNOSTIC LOGS ===\n")
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
        sb.append("Exported At: ${sdf.format(Date())}\n\n")
        _logs.value.forEach { entry ->
            sb.append(entry.toFormattedString()).append("\n")
        }
        return sb.toString()
    }

    fun copyToClipboard(context: Context): Boolean {
        return try {
            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("App Diagnostic Logs", getFullLogText())
            clipboard.setPrimaryClip(clip)
            true
        } catch (e: Exception) {
            e("AppLogger", "Failed to copy logs to clipboard", e)
            false
        }
    }
}
