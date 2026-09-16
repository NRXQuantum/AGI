package com.example

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.example.ml.FeatureScaler
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class ExampleRobolectricTest {

  @Test
  fun `read string from context`() {
    val context = ApplicationProvider.getApplicationContext<Context>()
    val appName = context.getString(R.string.app_name)
    assertEquals("Model Trainer", appName)
  }

  @Test
  fun `feature scaler normalizes data correctly`() {
    val sampleFeatures = listOf(
        floatArrayOf(1.0f, 10.0f, 100.0f),
        floatArrayOf(2.0f, 20.0f, 200.0f),
        floatArrayOf(3.0f, 30.0f, 300.0f)
    )
    val scaler = FeatureScaler.fit(sampleFeatures, 3)
    assertEquals(3, scaler.means.size)
    assertEquals(3, scaler.stds.size)

    val transformed = scaler.transform(floatArrayOf(2.0f, 20.0f, 200.0f))
    assertEquals(3, transformed.size)
    // transformed should be centered near 0
    assertTrue(Math.abs(transformed[0]) < 0.01f)
  }
}
