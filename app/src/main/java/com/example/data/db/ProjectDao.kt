package com.example.data.db

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface ProjectDao {
    @Query("SELECT * FROM projects ORDER BY createdAt DESC")
    fun getAllProjects(): Flow<List<ProjectEntity>>

    @Query("SELECT * FROM projects WHERE id = :projectId")
    fun getProjectById(projectId: Long): Flow<ProjectEntity?>

    @Query("SELECT * FROM projects WHERE id = :projectId")
    suspend fun getProjectByIdDirect(projectId: Long): ProjectEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertProject(project: ProjectEntity): Long

    @Update
    suspend fun updateProject(project: ProjectEntity)

    @Query("DELETE FROM projects WHERE id = :projectId")
    suspend fun deleteProject(projectId: Long)

    // Classes
    @Query("SELECT * FROM classification_classes WHERE projectId = :projectId ORDER BY createdAt ASC")
    fun getClassesForProject(projectId: Long): Flow<List<ClassificationClassEntity>>

    @Query("SELECT * FROM classification_classes WHERE projectId = :projectId ORDER BY createdAt ASC")
    suspend fun getClassesForProjectDirect(projectId: Long): List<ClassificationClassEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertClass(classificationClass: ClassificationClassEntity): Long

    @Update
    suspend fun updateClass(classificationClass: ClassificationClassEntity)

    @Delete
    suspend fun deleteClass(classificationClass: ClassificationClassEntity)

    @Query("DELETE FROM classification_classes WHERE id = :classId")
    suspend fun deleteClassById(classId: Long)

    // Image Samples
    @Query("SELECT * FROM image_samples WHERE classId = :classId ORDER BY createdAt DESC")
    fun getSamplesForClass(classId: Long): Flow<List<ImageSampleEntity>>

    @Query("SELECT * FROM image_samples WHERE projectId = :projectId")
    suspend fun getAllSamplesForProjectDirect(projectId: Long): List<ImageSampleEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertSample(sample: ImageSampleEntity): Long

    @Update
    suspend fun updateSample(sample: ImageSampleEntity)

    @Query("UPDATE image_samples SET classId = :newClassId WHERE id = :sampleId")
    suspend fun updateSampleClass(sampleId: Long, newClassId: Long)

    @Query("UPDATE image_samples SET classId = :newClassId WHERE id IN (:sampleIds)")
    suspend fun updateSamplesBatchClass(sampleIds: List<Long>, newClassId: Long)

    @Delete
    suspend fun deleteSample(sample: ImageSampleEntity)

    @Query("DELETE FROM image_samples WHERE id = :sampleId")
    suspend fun deleteSampleById(sampleId: Long)

    @Query("SELECT COUNT(*) FROM image_samples WHERE classId = :classId")
    fun getSampleCountForClass(classId: Long): Flow<Int>

    @Query("SELECT COUNT(*) FROM image_samples WHERE projectId = :projectId")
    fun getTotalSampleCountForProject(projectId: Long): Flow<Int>

    // Text Samples
    @Query("SELECT * FROM text_samples WHERE classId = :classId ORDER BY createdAt DESC")
    fun getTextSamplesForClass(classId: Long): Flow<List<TextSampleEntity>>

    @Query("SELECT * FROM text_samples WHERE projectId = :projectId")
    suspend fun getAllTextSamplesForProjectDirect(projectId: Long): List<TextSampleEntity>

    @Query("SELECT * FROM text_samples WHERE classId = :classId ORDER BY id DESC LIMIT :limit")
    suspend fun getTextSamplesForClassDirect(classId: Long, limit: Int = 200): List<TextSampleEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertTextSample(sample: TextSampleEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertTextSamplesBatch(samples: List<TextSampleEntity>)

    @Update
    suspend fun updateTextSample(sample: TextSampleEntity)

    @Delete
    suspend fun deleteTextSample(sample: TextSampleEntity)

    @Query("DELETE FROM text_samples WHERE id = :sampleId")
    suspend fun deleteTextSampleById(sampleId: Long)

    @Query("UPDATE text_samples SET classId = :newClassId WHERE id = :sampleId")
    suspend fun updateTextSampleClass(sampleId: Long, newClassId: Long)

    @Query("SELECT COUNT(*) FROM text_samples WHERE classId = :classId")
    fun getTextSampleCountForClass(classId: Long): Flow<Int>

    @Query("SELECT COUNT(*) FROM text_samples WHERE projectId = :projectId")
    fun getTotalTextSampleCountForProject(projectId: Long): Flow<Int>

    // Trained Models
    @Query("SELECT * FROM trained_models WHERE projectId = :projectId ORDER BY trainedAt DESC LIMIT 1")
    fun getLatestTrainedModel(projectId: Long): Flow<TrainedModelEntity?>

    @Query("SELECT * FROM trained_models WHERE projectId = :projectId ORDER BY trainedAt DESC LIMIT 1")
    suspend fun getLatestTrainedModelDirect(projectId: Long): TrainedModelEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertTrainedModel(model: TrainedModelEntity): Long
}
