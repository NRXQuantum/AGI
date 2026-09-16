package com.example.data.db

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(tableName = "projects")
data class ProjectEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val name: String,
    val description: String,
    val createdAt: Long = System.currentTimeMillis(),
    val isTrained: Boolean = false,
    val trainedAt: Long? = null,
    val trainingAccuracy: Float = 0f,
    val trainingEpochs: Int = 30,
    val learningRate: Float = 0.01f,
    val batchSize: Int = 8,
    val projectType: String = "IMAGE_CLASSIFICATION"
)

@Entity(
    tableName = "classification_classes",
    foreignKeys = [
        ForeignKey(
            entity = ProjectEntity::class,
            parentColumns = ["id"],
            childColumns = ["projectId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("projectId")]
)
data class ClassificationClassEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val projectId: Long,
    val className: String,
    val colorHex: String,
    val createdAt: Long = System.currentTimeMillis()
)

@Entity(
    tableName = "image_samples",
    foreignKeys = [
        ForeignKey(
            entity = ClassificationClassEntity::class,
            parentColumns = ["id"],
            childColumns = ["classId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("classId"), Index("projectId")]
)
data class ImageSampleEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val classId: Long,
    val projectId: Long,
    val imagePath: String,
    val createdAt: Long = System.currentTimeMillis()
)

@Entity(
    tableName = "trained_models",
    foreignKeys = [
        ForeignKey(
            entity = ProjectEntity::class,
            parentColumns = ["id"],
            childColumns = ["projectId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("projectId")]
)
data class TrainedModelEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val projectId: Long,
    val numClasses: Int,
    val featureDim: Int,
    val weightsJson: String, // Flattened weight matrix as JSON array
    val biasJson: String,    // Bias array as JSON
    val classLabelsJson: String, // List of class names as JSON
    val accuracy: Float,
    val trainedAt: Long = System.currentTimeMillis(),
    val featureScaleMeansJson: String = "[]",
    val featureScaleStdsJson: String = "[]"
)
