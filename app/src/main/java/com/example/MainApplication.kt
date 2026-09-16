package com.example

import android.app.Application
import com.example.util.AppLogger

class MainApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        AppLogger.init()
        AppLogger.i("MainApplication", "Application onCreate() executed successfully.")
    }
}
