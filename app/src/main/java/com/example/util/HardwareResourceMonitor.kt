package com.example.util

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import java.util.Locale

/**
 * Real-Time Adaptive Hardware Resource, Thermal & Battery Monitor.
 *
 * Monitors:
 * 1. Battery Temperature (°C) & Charging Status.
 * 2. System Thermal Pressure (PowerManager ThermalStatus on Android 10+).
 * 3. RAM Memory Available, Used & Thresholds (ActivityManager).
 * 4. Dynamic Throttle Regulator: Adjusts computational throughput dynamically
 *    to prevent overheating, protect battery longevity, and maintain maximum speed.
 */
object HardwareResourceMonitor {

    enum class PerformanceProfile(
        val displayName: String,
        val displayNameBn: String,
        val description: String,
        val icon: String
    ) {
        TURBO(
            displayName = "Turbo Speed (Max Throughput)",
            displayNameBn = "টার্বো স্পিড (সর্বোচ্চ গতি)",
            description = "Zero artificial pauses, maximum vector acceleration. Completes training in seconds.",
            icon = "⚡"
        ),
        SMART_ADAPTIVE(
            displayName = "Smart Adaptive (Thermal & RAM Guard)",
            displayNameBn = "স্মার্ট অ্যাডাপ্টিভ (থার্মাল ও র‍্যাম ব্যালান্সড)",
            description = "Runs at full speed when phone is cool (<39°C). Intelligently balances micro-pauses if device temperature rises.",
            icon = "🛡️"
        ),
        ECO_BATTERY(
            displayName = "Eco Battery Saver",
            displayNameBn = "ইকো ব্যাটারি সাশ্রয়ী",
            description = "Limits peak CPU drain to conserve battery life and minimize thermal footprint.",
            icon = "🔋"
        )
    }

    enum class ThermalState(val label: String, val labelBn: String, val colorHex: String) {
        COOL("Cool & Safe", "স্বাভাবিক ও শীতল", "#10B981"),
        OPTIMAL("Optimal Temp", "অনুকূল তাপমাত্রা", "#3B82F6"),
        WARM("Warm (Monitored)", "হালকা গরম", "#F59E0B"),
        HOT("High Temp (Cooling)", "অতিরিক্ত গরম (সুরক্ষিত)", "#EF4444")
    }

    data class HardwareSnapshot(
        val temperatureCelsius: Float,
        val thermalState: ThermalState,
        val batteryPercent: Int,
        val isCharging: Boolean,
        val totalRamMb: Long,
        val availableRamMb: Long,
        val usedRamMb: Long,
        val ramUsagePercent: Int,
        val cpuCores: Int,
        val activeProfile: PerformanceProfile,
        val throttleDelayMs: Long = 0L,
        val statusSummary: String = ""
    )

    fun captureSnapshot(context: Context, profile: PerformanceProfile = PerformanceProfile.SMART_ADAPTIVE): HardwareSnapshot {
        // 1. Battery temperature & level
        var tempC = 32.5f
        var batteryPct = 85
        var isCharging = false

        try {
            val batteryStatus: Intent? = context.registerReceiver(
                null,
                IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            )
            batteryStatus?.let { intent ->
                val rawTemp = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)
                if (rawTemp > 0) {
                    tempC = rawTemp / 10.0f
                }
                val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
                if (level >= 0 && scale > 0) {
                    batteryPct = ((level.toFloat() / scale.toFloat()) * 100).toInt()
                }
                val status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
                isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING || status == BatteryManager.BATTERY_STATUS_FULL
            }
        } catch (_: Throwable) {}

        // 2. RAM Memory Info
        var totalMb = 4096L
        var availMb = 2048L
        try {
            val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            val memInfo = ActivityManager.MemoryInfo()
            actManager?.getMemoryInfo(memInfo)
            totalMb = memInfo.totalMem / (1024L * 1024L)
            availMb = memInfo.availMem / (1024L * 1024L)
        } catch (_: Throwable) {}

        val usedMb = (totalMb - availMb).coerceAtLeast(0L)
        val ramPct = if (totalMb > 0) ((usedMb.toFloat() / totalMb.toFloat()) * 100).toInt().coerceIn(0, 100) else 50

        // 3. System Thermal Status (API 29+)
        var isSystemThrottled = false
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            try {
                val powerManager = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
                val thermalStatus = powerManager?.currentThermalStatus ?: PowerManager.THERMAL_STATUS_NONE
                if (thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE) {
                    isSystemThrottled = true
                }
            } catch (_: Throwable) {}
        }

        // 4. Determine Thermal State
        val thermalState = when {
            tempC >= 43.0f || isSystemThrottled -> ThermalState.HOT
            tempC >= 39.5f -> ThermalState.WARM
            tempC >= 35.0f -> ThermalState.OPTIMAL
            else -> ThermalState.COOL
        }

        // 5. Dynamic throttle pacing computation
        val throttleDelayMs = when (profile) {
            PerformanceProfile.TURBO -> 0L // Absolute 0ms - zero artificial sleep
            PerformanceProfile.SMART_ADAPTIVE -> {
                when {
                    thermalState == ThermalState.HOT -> 6L // Tiny micro-yield if phone is hot
                    thermalState == ThermalState.WARM -> 2L // Negligible micro-yield
                    else -> 0L // Cool & Optimal: full speed, zero pause
                }
            }
            PerformanceProfile.ECO_BATTERY -> 8L
        }

        val cpuCores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)

        val summary = String.format(
            Locale.US,
            "%.1f°C (%s) • RAM %d%% (%d MB free) • %s",
            tempC,
            thermalState.label,
            ramPct,
            availMb,
            profile.displayName
        )

        return HardwareSnapshot(
            temperatureCelsius = tempC,
            thermalState = thermalState,
            batteryPercent = batteryPct,
            isCharging = isCharging,
            totalRamMb = totalMb,
            availableRamMb = availMb,
            usedRamMb = usedMb,
            ramUsagePercent = ramPct,
            cpuCores = cpuCores,
            activeProfile = profile,
            throttleDelayMs = throttleDelayMs,
            statusSummary = summary
        )
    }
}
