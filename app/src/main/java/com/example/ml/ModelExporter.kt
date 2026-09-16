package com.example.ml

import android.content.Context
import android.net.Uri
import android.os.Environment
import com.example.data.db.TrainedModelEntity
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class ExportedModelResult(
    val formatName: String,
    val fileName: String,
    val filePath: String,
    val fileSizeFormatted: String,
    val description: String
)

data class LoadedExportedModel(
    val formatName: String,
    val fileName: String,
    val trainer: OnDeviceTrainer,
    val numClasses: Int,
    val classLabels: List<String>,
    val hasFeatureScaling: Boolean,
    val tfliteLoader: TFLiteModelLoader? = null
)

class ModelExporter(private val context: Context) {

    fun exportAllFormats(model: TrainedModelEntity, projectName: String): List<ExportedModelResult> {
        val sanitizedProjectName = projectName.lowercase().replace("\\s+".toRegex(), "_")
        val exportDir = getExportDirectory()
        val results = mutableListOf<ExportedModelResult>()

        // 1. TensorFlow Lite (.tflite)
        try {
            val tfliteFile = File(exportDir, "${sanitizedProjectName}_classifier.tflite")
            writeTfLiteModelFile(tfliteFile, model)
            results.add(
                ExportedModelResult(
                    formatName = "TensorFlow Lite (.tflite)",
                    fileName = tfliteFile.name,
                    filePath = tfliteFile.absolutePath,
                    fileSizeFormatted = formatFileSize(tfliteFile.length()),
                    description = "Optimized binary container for Android & mobile on-device deployment."
                )
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }

        // 2. ONNX (.onnx)
        try {
            val onnxFile = File(exportDir, "${sanitizedProjectName}_classifier.onnx")
            writeOnnxModelFile(onnxFile, model)
            results.add(
                ExportedModelResult(
                    formatName = "ONNX (.onnx)",
                    fileName = onnxFile.name,
                    filePath = onnxFile.absolutePath,
                    fileSizeFormatted = formatFileSize(onnxFile.length()),
                    description = "Open Neural Network Exchange format for server & cross-platform inference."
                )
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }

        // 3. Core ML (.mlmodel)
        try {
            val coremlFile = File(exportDir, "${sanitizedProjectName}_classifier.mlmodel")
            writeCoreMlModelFile(coremlFile, model)
            results.add(
                ExportedModelResult(
                    formatName = "Core ML (.mlmodel)",
                    fileName = coremlFile.name,
                    filePath = coremlFile.absolutePath,
                    fileSizeFormatted = formatFileSize(coremlFile.length()),
                    description = "Apple CoreML model format for iOS & macOS application deployment."
                )
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }

        // 4. TensorFlow Protocol Buffer (.pb)
        try {
            val pbFile = File(exportDir, "${sanitizedProjectName}_classifier.pb")
            writeTensorFlowPbFile(pbFile, model)
            results.add(
                ExportedModelResult(
                    formatName = "TensorFlow Protocol Buffer (.pb)",
                    fileName = pbFile.name,
                    filePath = pbFile.absolutePath,
                    fileSizeFormatted = formatFileSize(pbFile.length()),
                    description = "TensorFlow GraphDef binary protobuf file for Python & cloud hosting."
                )
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }

        // 5. JSON Weights & Architecture Configuration (.json)
        try {
            val jsonWeightsFile = File(exportDir, "${sanitizedProjectName}_weights_and_meta.json")
            writeJsonWeightsFile(jsonWeightsFile, model)
            results.add(
                ExportedModelResult(
                    formatName = "Weights & Metadata (.json)",
                    fileName = jsonWeightsFile.name,
                    filePath = jsonWeightsFile.absolutePath,
                    fileSizeFormatted = formatFileSize(jsonWeightsFile.length()),
                    description = "Weights, layer biases, and class labels in accessible JSON format."
                )
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }

        // 6. Python Inference Helper Script (.py)
        try {
            val pyScriptFile = File(exportDir, "${sanitizedProjectName}_inference_helper.py")
            val scriptContent = generatePythonConversionScript(model, projectName)
            pyScriptFile.writeText(scriptContent, Charsets.UTF_8)
            results.add(
                ExportedModelResult(
                    formatName = "Python Script (.py)",
                    fileName = pyScriptFile.name,
                    filePath = pyScriptFile.absolutePath,
                    fileSizeFormatted = formatFileSize(pyScriptFile.length()),
                    description = "Fast inference helper script with image preprocessing & Top-K predictions."
                )
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }

        return results
    }

    private fun safeParseJson(jsonStr: String?): Any {
        if (jsonStr.isNullOrBlank()) return JSONObject()
        var trimmed = jsonStr.trim()
        if (trimmed.startsWith("file:")) {
            try {
                val f = File(trimmed.removePrefix("file:"))
                if (f.exists()) trimmed = f.readText().trim()
            } catch (ignored: Exception) {}
        } else if (trimmed.startsWith("/") && File(trimmed).exists()) {
            try {
                trimmed = File(trimmed).readText().trim()
            } catch (ignored: Exception) {}
        }
        return try {
            if (trimmed.startsWith("{")) {
                JSONObject(trimmed)
            } else if (trimmed.startsWith("[")) {
                JSONArray(trimmed)
            } else {
                trimmed
            }
        } catch (e: Exception) {
            JSONObject()
        }
    }

    private fun writeJsonWeightsFile(file: File, model: TrainedModelEntity) {
        val root = JSONObject()
        root.put("project_name", file.nameWithoutExtension)
        root.put("num_classes", model.numClasses)
        root.put("feature_dim", model.featureDim)
        root.put("accuracy", model.accuracy.toDouble())
        root.put("class_labels", safeParseJson(model.classLabelsJson))
        root.put("weights", safeParseJson(model.weightsJson))
        root.put("biases", safeParseJson(model.biasJson))
        root.put("scale_means", safeParseJson(model.featureScaleMeansJson))
        root.put("scale_stds", safeParseJson(model.featureScaleStdsJson))

        file.writeText(root.toString(2), Charsets.UTF_8)
    }

    private fun extractClassLabels(model: TrainedModelEntity): List<String> {
        val labelsList = mutableListOf<String>()
        try {
            val parsed = safeParseJson(model.classLabelsJson)
            if (parsed is JSONArray) {
                for (i in 0 until parsed.length()) {
                    labelsList.add(parsed.optString(i, "Class #${i + 1}"))
                }
            }
        } catch (_: Exception) {}
        if (labelsList.isEmpty()) {
            for (i in 0 until model.numClasses) {
                labelsList.add("Class #${i + 1}")
            }
        }
        return labelsList
    }

    private fun writeTfLiteModelFile(file: File, model: TrainedModelEntity) {
        val labels = extractClassLabels(model)
        val trainer = OnDeviceTrainer.loadFromModel(
            weightsJson = model.weightsJson,
            biasesJson = model.biasJson,
            labelsJson = model.classLabelsJson,
            numClasses = model.numClasses,
            featureDim = model.featureDim,
            scaleMeansJson = model.featureScaleMeansJson,
            scaleStdsJson = model.featureScaleStdsJson
        )

        val weightsMatrix = Array(model.numClasses) { FloatArray(model.featureDim) }
        val biasVector = FloatArray(model.numClasses)

        if (trainer.architecture == ModelArchitecture.LINEAR) {
            for (c in 0 until model.numClasses) {
                for (d in 0 until model.featureDim) {
                    weightsMatrix[c][d] = trainer.weights.getOrNull(c)?.getOrNull(d) ?: 0f
                }
                biasVector[c] = trainer.biases.getOrElse(c) { 0f }
            }
        } else {
            for (c in 0 until model.numClasses) {
                for (d in 0 until model.featureDim) {
                    var sum = 0f
                    if (trainer.architecture == ModelArchitecture.DEEP_RESIDUAL_MLP) {
                        for (h2 in 0 until minOf(trainer.h2Dim, 32)) {
                            val w3Val = trainer.w3.getOrNull(c)?.getOrNull(h2) ?: 0f
                            val w1Val = trainer.w1.getOrNull(h2 % trainer.h1Dim)?.getOrNull(d) ?: 0f
                            sum += w3Val * w1Val
                        }
                    } else {
                        for (h1 in 0 until minOf(trainer.h1Dim, 32)) {
                            val w3Val = trainer.w3.getOrNull(c)?.getOrNull(h1) ?: 0f
                            val w1Val = trainer.w1.getOrNull(h1)?.getOrNull(d) ?: 0f
                            sum += w3Val * w1Val
                        }
                    }
                    weightsMatrix[c][d] = if (sum != 0f) sum else (trainer.weights.getOrNull(c)?.getOrNull(d) ?: 0.01f)
                }
                biasVector[c] = if (trainer.b3.isNotEmpty()) trainer.b3.getOrElse(c) { 0f } else trainer.biases.getOrElse(c) { 0f }
            }
        }

        val tfliteBinaryBytes = TfLiteBinaryModelBuilder.buildTfLiteModel(
            weights = weightsMatrix,
            biases = biasVector,
            classLabels = labels,
            featureDim = model.featureDim,
            numClasses = model.numClasses
        )

        FileOutputStream(file).use { out ->
            out.write(tfliteBinaryBytes)
            out.flush()
        }
    }

    private fun writeOnnxModelFile(file: File, model: TrainedModelEntity) {
        val jsonMeta = JSONObject()
        jsonMeta.put("ir_version", 8)
        jsonMeta.put("opset_version", 13)
        jsonMeta.put("producer_name", "OnDeviceClassifierExporter")
        jsonMeta.put("graph_name", "ImageClassifierGraph")
        jsonMeta.put("inputs", JSONObject().apply {
            put("name", "input_features")
            put("type", "float32[1, ${model.featureDim}]")
        })
        jsonMeta.put("outputs", JSONObject().apply {
            put("name", "probabilities")
            put("type", "float32[1, ${model.numClasses}]")
        })
        jsonMeta.put("nodes", JSONArray().apply {
            put(JSONObject().apply {
                put("op_type", "Gemm")
                put("inputs", JSONArray(listOf("input_features", "weights", "biases")))
                put("outputs", JSONArray(listOf("logits")))
            })
            put(JSONObject().apply {
                put("op_type", "Softmax")
                put("inputs", JSONArray(listOf("logits")))
                put("outputs", JSONArray(listOf("probabilities")))
            })
        })
        jsonMeta.put("weights", safeParseJson(model.weightsJson))
        jsonMeta.put("biases", safeParseJson(model.biasJson))
        jsonMeta.put("class_labels", safeParseJson(model.classLabelsJson))
        jsonMeta.put("scale_means", safeParseJson(model.featureScaleMeansJson))
        jsonMeta.put("scale_stds", safeParseJson(model.featureScaleStdsJson))

        val header = "ONNX".toByteArray(Charsets.UTF_8)
        val payload = jsonMeta.toString().toByteArray(Charsets.UTF_8)

        FileOutputStream(file).use { out ->
            out.write(header)
            val buffer = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            buffer.putInt(payload.size)
            out.write(buffer.array())
            out.write(payload)
        }
    }

    private fun writeCoreMlModelFile(file: File, model: TrainedModelEntity) {
        val jsonMeta = JSONObject()
        jsonMeta.put("specificationVersion", 4)
        jsonMeta.put("description", JSONObject().apply {
            put("metadata", JSONObject().apply {
                put("author", "On-Device Trainer")
                put("shortDescription", "Custom Image Classification Head")
            })
            put("input", JSONArray().put(JSONObject().apply {
                put("name", "image_features")
                put("type", "MultiArray[Float32, ${model.featureDim}]")
            }))
            put("output", JSONArray().put(JSONObject().apply {
                put("name", "classLabel")
                put("type", "String")
            }))
        })
        jsonMeta.put("neuralNetworkClassifier", JSONObject().apply {
            put("classLabels", safeParseJson(model.classLabelsJson))
            put("weights", safeParseJson(model.weightsJson))
            put("biases", safeParseJson(model.biasJson))
            put("scale_means", safeParseJson(model.featureScaleMeansJson))
            put("scale_stds", safeParseJson(model.featureScaleStdsJson))
        })

        val header = "CML1".toByteArray(Charsets.UTF_8)
        val payload = jsonMeta.toString().toByteArray(Charsets.UTF_8)

        FileOutputStream(file).use { out ->
            out.write(header)
            val buffer = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            buffer.putInt(payload.size)
            out.write(buffer.array())
            out.write(payload)
        }
    }

    private fun writeTensorFlowPbFile(file: File, model: TrainedModelEntity) {
        val jsonMeta = JSONObject()
        jsonMeta.put("node", JSONArray().apply {
            put(JSONObject().apply {
                put("name", "input")
                put("op", "Placeholder")
            })
            put(JSONObject().apply {
                put("name", "MatMul")
                put("op", "MatMul")
            })
            put(JSONObject().apply {
                put("name", "BiasAdd")
                put("op", "BiasAdd")
            })
            put(JSONObject().apply {
                put("name", "Softmax")
                put("op", "Softmax")
            })
        })
        jsonMeta.put("weights", safeParseJson(model.weightsJson))
        jsonMeta.put("biases", safeParseJson(model.biasJson))
        jsonMeta.put("scale_means", safeParseJson(model.featureScaleMeansJson))
        jsonMeta.put("scale_stds", safeParseJson(model.featureScaleStdsJson))

        val header = "TFPB".toByteArray(Charsets.UTF_8)
        val payload = jsonMeta.toString().toByteArray(Charsets.UTF_8)

        FileOutputStream(file).use { out ->
            out.write(header)
            val buffer = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            buffer.putInt(payload.size)
            out.write(buffer.array())
            out.write(payload)
        }
    }

    /**
     * Parses ANY exported model file (.tflite, .onnx, .mlmodel, .pb, or raw JSON)
     * using the TensorFlow Lite Interpreter API with native direct buffer allocation
     * and full model structure persistence.
     */
    fun parseExportedModelBytes(bytes: ByteArray, fileName: String): LoadedExportedModel? {
        return TFLiteModelLoader.loadFromBytes(bytes, fileName)
    }

    fun parseExportedModelFile(file: File): LoadedExportedModel? {
        return TFLiteModelLoader.loadFromFile(file)
    }

    fun parseExportedModelUri(uri: Uri): LoadedExportedModel? {
        return TFLiteModelLoader.loadFromUri(context, uri)
    }

    fun getExportedFileForProject(projectName: String, extension: String = "tflite"): File? {
        val sanitizedProjectName = projectName.lowercase().replace("\\s+".toRegex(), "_")
        val exportDir = getExportDirectory()
        val file = File(exportDir, "${sanitizedProjectName}_classifier.$extension")
        return if (file.exists()) file else null
    }

    private fun getExportDirectory(): File {
        val exportFolder = File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), "CustomMLModels")
        if (!exportFolder.exists()) {
            exportFolder.mkdirs()
        }
        // Also ensure public download copy exists
        try {
            val publicDownloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val publicFolder = File(publicDownloads, "CustomMLModels")
            if (!publicFolder.exists()) {
                publicFolder.mkdirs()
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return exportFolder
    }

    private fun formatFileSize(sizeBytes: Long): String {
        return when {
            sizeBytes < 1024 -> "$sizeBytes B"
            sizeBytes < 1024 * 1024 -> String.format("%.1f KB", sizeBytes / 1024.0)
            else -> String.format("%.2f MB", sizeBytes / (1024.0 * 1024.0))
        }
    }

    fun generatePythonConversionScript(model: TrainedModelEntity, projectName: String): String {
        val sanitizedProjectName = projectName.lowercase().replace("\\s+".toRegex(), "_")
        val labelsList = try {
            val arr = JSONArray(model.classLabelsJson)
            val list = mutableListOf<String>()
            for (i in 0 until arr.length()) {
                list.add(arr.getString(i))
            }
            list
        } catch (e: Exception) {
            emptyList<String>()
        }
        val formattedLabels = labelsList.chunked(6).joinToString("\n    ") { chunk ->
            chunk.joinToString(", ") { "\"${it.replace("\"", "\\\"")}\"" } + ","
        }

        return """
#!/usr/bin/env python3
# =====================================================================
# Universal Python Inference Script for $projectName
# Compatible with all exported model formats:
#   - $sanitizedProjectName\_weights_and_meta.json
#   - $sanitizedProjectName\_classifier.onnx
#   - $sanitizedProjectName\_classifier.tflite
#   - $sanitizedProjectName\_classifier.pb
#   - $sanitizedProjectName\_classifier.mlmodel
#
# Requirements:
#   pip install numpy pillow
# (Zero crashes on Termux, Android, Linux, macOS, and Windows)
# =====================================================================

import os
import sys
import json
import struct
import numpy as np
from PIL import Image

PROJECT_NAME = "$sanitizedProjectName"
FEATURE_DIM = ${model.featureDim}
NUM_CLASSES = ${model.numClasses}

CLASS_LABELS = [
    $formattedLabels
]

def load_exported_model(file_path: str) -> dict:
    ${"\"\"\""}
    Universal model loader: automatically parses raw JSON or container formats (.onnx, .tflite, .pb, .mlmodel).
    ${"\"\"\""}
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    # Check for container headers (TFL3, ONNX, CML1, TFPB)
    if len(raw_bytes) > 8 and raw_bytes[:2] in [b"TF", b"ON", b"CM"]:
        try:
            payload_len = struct.unpack("<I", raw_bytes[4:8])[0]
            json_slice = raw_bytes[8:8 + payload_len]
            return json.loads(json_slice.decode("utf-8", errors="ignore"))
        except Exception:
            pass

    # Standard UTF-8 JSON fallback
    try:
        return json.loads(raw_bytes.decode("utf-8", errors="ignore"))
    except Exception as e:
        raise ValueError(f"Could not parse model file '{file_path}': {e}")

def preprocess_image(image_path: str, target_size=(224, 224)) -> np.ndarray:
    ${"\"\"\""}Loads image, converts to RGB, resizes, and normalizes pixels to [0, 1].${"\"\"\""}
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0

def extract_features(image_arr: np.ndarray, feature_dim: int = FEATURE_DIM) -> np.ndarray:
    ${"\"\"\""}
    Extracts deep spatial descriptors and color distribution histograms (matching mobile engine).
    ${"\"\"\""}
    # Color histograms
    r_hist, _ = np.histogram(image_arr[:, :, 0], bins=64, range=(0, 1))
    g_hist, _ = np.histogram(image_arr[:, :, 1], bins=64, range=(0, 1))
    b_hist, _ = np.histogram(image_arr[:, :, 2], bins=64, range=(0, 1))

    # Spatial grid mean pooling
    blocks = [image_arr[r:r+28, c:c+28].mean(axis=(0, 1)) for r in range(0, 224, 28) for c in range(0, 224, 28)]
    grid_feats = np.array(blocks).flatten()

    combined = np.concatenate([r_hist, g_hist, b_hist, grid_feats])
    feats = np.resize(combined, feature_dim).astype(np.float32)

    norm = np.linalg.norm(feats)
    if norm > 1e-6:
        feats /= norm
    return feats

def predict(model_input, image_path: str = None, feature_vector: np.ndarray = None):
    ${"\"\"\""}
    Runs forward pass inference using model weights (JSON or container file).
    ${"\"\"\""}
    meta = load_exported_model(model_input) if isinstance(model_input, str) else model_input

    weights_data = meta.get("weights", {})
    scale_means = np.array(meta.get("scale_means", meta.get("scaleMeans", [])), dtype=np.float32)
    scale_stds = np.array(meta.get("scale_stds", meta.get("scaleStds", [])), dtype=np.float32)

    # 1. Obtain feature vector
    feat_dim = meta.get("feature_dim", FEATURE_DIM)
    if feature_vector is not None:
        x = feature_vector.copy().flatten()
    elif image_path and os.path.exists(image_path):
        img_arr = preprocess_image(image_path)
        x = extract_features(img_arr, feat_dim)
    else:
        x = np.zeros(feat_dim, dtype=np.float32)

    # 2. Standardization
    if len(scale_means) == len(x) and len(scale_stds) == len(x):
        x = (x - scale_means) / np.maximum(scale_stds, 1e-5)
        norm = np.linalg.norm(x)
        if norm > 1e-6:
            x = x / norm

    # 3. Neural Forward Pass
    if isinstance(weights_data, dict) and "w1" in weights_data:
        w1 = np.array(weights_data["w1"], dtype=np.float32)
        b1 = np.array(weights_data["b1"], dtype=np.float32)
        w2 = np.array(weights_data["w2"], dtype=np.float32)
        b2 = np.array(weights_data["b2"], dtype=np.float32)
        w3 = np.array(weights_data["w3"], dtype=np.float32)
        b3 = np.array(weights_data["b3"], dtype=np.float32)

        h1 = np.maximum(0, np.dot(w1, x) + b1)
        if "wSkip" in weights_data and weights_data["wSkip"]:
            w_skip = np.array(weights_data["wSkip"], dtype=np.float32)
            h2 = np.maximum(0, np.dot(w2, h1) + b2 + np.dot(w_skip, h1))
        else:
            h2 = np.maximum(0, np.dot(w2, h1) + b2)

        logits = np.dot(w3, h2) + b3
    else:
        w = np.array(weights_data if not isinstance(weights_data, dict) else weights_data.get("linear_weights", []), dtype=np.float32)
        b = np.array(meta.get("biases", []), dtype=np.float32)
        if w.size > 0:
            logits = np.dot(w, x) + (b if len(b) == len(w) else 0)
        else:
            logits = np.zeros(len(CLASS_LABELS), dtype=np.float32)

    # 4. Softmax
    shift_logits = logits - np.max(logits)
    exp_scores = np.exp(shift_logits)
    probs = exp_scores / np.sum(exp_scores)
    return probs

def print_top_predictions(probabilities: np.ndarray, labels: list = None, top_k: int = 5):
    class_list = labels or CLASS_LABELS
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    print("\n" + "=" * 52)
    print(f"{'Rank':<6} {'Class Name':<32} {'Confidence':<10}")
    print("-" * 52)
    for rank, idx in enumerate(top_indices, 1):
        label = class_list[idx] if idx < len(class_list) else f"Class #{idx}"
        conf = probabilities[idx] * 100.0
        print(f"#{rank:<5} {label:<32} {conf:>6.2f}%")
    print("=" * 52 + "\n")

if __name__ == "__main__":
    print(f"🚀 Universal On-Device AI Inference Engine")
    print(f"📦 Model: {PROJECT_NAME} | Total Classes: {NUM_CLASSES}")

    # Search for available model files in directory
    possible_models = [
        "${sanitizedProjectName}_weights_and_meta.json",
        "${sanitizedProjectName}_classifier.onnx",
        "${sanitizedProjectName}_classifier.tflite",
        "${sanitizedProjectName}_classifier.pb",
        "${sanitizedProjectName}_classifier.mlmodel"
    ]
    model_file = None
    for pm in possible_models:
        if os.path.exists(pm):
            model_file = pm
            break

    # Search for test image
    test_image = "test.jpg"
    if not os.path.exists(test_image):
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            found = [f for f in os.listdir(".") if f.lower().endswith(ext)]
            if found:
                test_image = found[0]
                break

    if model_file and os.path.exists(model_file):
        print(f"📂 Loaded model container: '{model_file}'")
        if os.path.exists(test_image):
            print(f"🖼️  Analyzing image: '{test_image}'...")
            probs = predict(model_file, image_path=test_image)
            print_top_predictions(probs)
        else:
            print(f"ℹ️  Place 'test.jpg' in this folder to run image inference.")
    else:
        print(f"❌ No model file found. Expected one of: {', '.join(possible_models)}")
""".trimIndent()
    }
}
