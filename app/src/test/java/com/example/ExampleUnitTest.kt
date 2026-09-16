package com.example

import com.example.ml.OnDeviceTrainer
import com.example.ml.TrainingSample
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class ExampleUnitTest {
  @Test
  fun testOnDeviceTrainingAndInference() = runBlocking {
    val labels = listOf("Cat", "Dog")
    val trainer = OnDeviceTrainer(numClasses = 2, featureDim = 10, classLabels = labels)

    val sampleCat = FloatArray(10) { 0.8f }
    val sampleDog = FloatArray(10) { -0.8f }

    val samples = listOf(
      TrainingSample(sampleCat, 0, "Cat"),
      TrainingSample(sampleDog, 1, "Dog")
    )

    trainer.train(samples, epochs = 20, learningRate = 0.1f, batchSize = 2) { }

    val predCat = trainer.predict(sampleCat)
    assertEquals("Cat", predCat.classLabel)

    val predDog = trainer.predict(sampleDog)
    assertEquals("Dog", predDog.classLabel)
  }
}
