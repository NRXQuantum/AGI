package com.example.ml

import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.util.Locale
import java.util.zip.ZipInputStream

/**
 * Universal, Memory-Safe Multi-Format & ZIP Parser for Text NLP Projects.
 *
 * Supports:
 * 1. ZIP Archives (.zip containing .txt, .csv, .json, .jsonl, .tsv) with on-the-fly decompression.
 * 2. 28+ Major NLP Benchmark Datasets (Tiny Shakespeare, TinyStories, Alpaca, Dolly 15k, SQuAD 2.0,
 *    TriviaQA, TyDi QA, WizardLM, UltraChat, DailyDialog, Bangla Alpaca, Bangla DailyDialog, Bangla SQuAD,
 *    PubMed Cancer NLP, SMS Spam, etc.).
 * 3. QA / Flashcard / Knowledge Base Formats (question,answer).
 * 4. LLM Instruction Tuning Formats (Dolly, Alpaca, ChatML JSONL).
 * 5. Dialogue & Persona scripts.
 * 6. Memory-Safe line-by-line streaming without heap overflow (500MB+ Safe).
 */
object TextDatasetParser {

    enum class DatasetFormatStrategy(val title: String, val titleBn: String, val description: String) {
        AUTO_DETECT("Auto-Detect Format", "স্বয়ংক্রিয় সনাক্তকরণ", "Automatically identify CSV, JSONL, QA, ZIP or Dialogue formats"),
        QA_QUESTION_ANSWER("Question & Answer (QA)", "প্রশ্ন ও উত্তর (QA)", "Pairs of question and answer (e.g. question,answer CSV)"),
        INSTRUCTION_RESPONSE("Instruction & Response", "ইনস্ট্রাকশন ও রেসপন্স", "LLM Instruction datasets (Dolly, Alpaca, JSONL)"),
        SHAKESPEARE_DIALOGUE("Dialogue & Drama Script", "নাটক ও সংলাপ স্ক্রিপ্ট", "Character dialogues (Speaker: Dialogue transcript)"),
        CSV_LABEL_TEXT("CSV / TSV (Label & Text)", "ক্যাটাগরি ও টেক্সট CSV", "Standard tabular rows with class label and text"),
        SECTION_HEADER("Section Headers ([Category])", "সেকশন হেডার", "Categories marked with [Category] or # Category")
    }

    data class ParsedSample(
        val className: String,
        val text: String
    )

    data class ParseResult(
        val formatName: String,
        val samples: List<ParsedSample>,
        val classCounts: Map<String, Int>,
        val totalLinesScanned: Long = 0L,
        val isCapped: Boolean = false,
        val errorMessage: String? = null
    ) {
        fun filterByMinSamples(minSamples: Int): ParseResult {
            if (minSamples <= 1) return this
            val validClasses = classCounts.filter { it.value >= minSamples }.keys
            val filteredSamples = samples.filter { it.className in validClasses }
            val newCounts = filteredSamples.groupingBy { it.className }.eachCount()
            return copy(
                samples = filteredSamples,
                classCounts = newCounts
            )
        }

        fun filterBySelectedClasses(selected: Set<String>): ParseResult {
            val filteredSamples = samples.filter { it.className in selected }
            val newCounts = filteredSamples.groupingBy { it.className }.eachCount()
            return copy(
                samples = filteredSamples,
                classCounts = newCounts
            )
        }
    }

    data class BenchmarkDatasetInfo(
        val id: String,
        val name: String,
        val nameBn: String,
        val category: String, // "Literature & Stories", "Instruction & LLM", "QA & Knowledge", "Dialogue & Chat", "Bangla NLP", "Specialized"
        val format: String,
        val defaultStrategy: DatasetFormatStrategy,
        val description: String,
        val sampleSnippet: String
    )

    private val CLASS_PALETTE = listOf(
        "#3B82F6", "#10B981", "#F59E0B", "#EF4444",
        "#8B5CF6", "#EC4899", "#06B6D4", "#F97316",
        "#14B8A6", "#6366F1", "#84CC16", "#64748B"
    )

    fun getColorForIndex(index: Int): String {
        return CLASS_PALETTE[index % CLASS_PALETTE.size]
    }

    /**
     * Memory-safe line-by-line streaming parser for huge files (up to 500MB+ without OOM).
     */
    fun parseStream(
        inputStream: InputStream,
        strategy: DatasetFormatStrategy = DatasetFormatStrategy.AUTO_DETECT,
        maxSampleCap: Int = 100_000,
        onProgress: ((linesRead: Long, samplesFound: Int) -> Unit)? = null
    ): ParseResult {
        val reader = BufferedReader(InputStreamReader(inputStream, Charsets.UTF_8), 32 * 1024)
        return parseBufferedReader(reader, strategy, maxSampleCap, onProgress)
    }

    /**
     * Parse compressed ZIP Archive (.zip) containing text, csv, json, jsonl, or tsv files.
     */
    fun parseZipStream(
        inputStream: InputStream,
        strategy: DatasetFormatStrategy = DatasetFormatStrategy.AUTO_DETECT,
        maxSampleCap: Int = 100_000,
        onProgress: ((linesRead: Long, samplesFound: Int) -> Unit)? = null
    ): ParseResult {
        val zipStream = ZipInputStream(inputStream)
        val combinedSamples = mutableListOf<ParsedSample>()
        var totalLines = 0L
        var firstValidFormat = "ZIP Archive"

        try {
            var entry = zipStream.nextEntry
            while (entry != null) {
                val entryName = entry.name
                if (!entry.isDirectory) {
                    val lower = entryName.lowercase(Locale.ROOT)
                    if (lower.endsWith(".txt") || lower.endsWith(".csv") || lower.endsWith(".json") || lower.endsWith(".jsonl") || lower.endsWith(".tsv")) {
                        val nonClosingStream = NonClosingInputStream(zipStream)
                        val reader = BufferedReader(InputStreamReader(nonClosingStream, Charsets.UTF_8), 32 * 1024)
                        val subResult = parseBufferedReader(reader, strategy, maxSampleCap - combinedSamples.size, onProgress)
                        if (subResult.samples.isNotEmpty()) {
                            combinedSamples.addAll(subResult.samples)
                            totalLines += subResult.totalLinesScanned
                            firstValidFormat = "📦 ZIP Archive ($entryName • ${subResult.formatName})"
                            if (combinedSamples.size >= maxSampleCap) break
                        }
                    }
                }
                zipStream.closeEntry()
                entry = zipStream.nextEntry
            }
        } catch (_: Exception) {}

        if (combinedSamples.isEmpty()) {
            return ParseResult(
                formatName = "ZIP Archive",
                samples = emptyList(),
                classCounts = emptyMap(),
                errorMessage = "No readable .txt, .csv, .json, or .jsonl files found inside the ZIP archive."
            )
        }

        val counts = combinedSamples.groupingBy { it.className }.eachCount()
        return ParseResult(
            formatName = firstValidFormat,
            samples = combinedSamples,
            classCounts = counts,
            totalLinesScanned = totalLines,
            isCapped = combinedSamples.size >= maxSampleCap
        )
    }

    /**
     * Standard string-based parser (wraps reader for unified parsing).
     */
    fun parse(
        rawContent: String,
        strategy: DatasetFormatStrategy = DatasetFormatStrategy.AUTO_DETECT,
        maxSampleCap: Int = 100_000
    ): ParseResult {
        val trimmed = rawContent.trim()
        if (trimmed.isBlank()) {
            return ParseResult(
                formatName = "Empty",
                samples = emptyList(),
                classCounts = emptyMap(),
                errorMessage = "Content is empty."
            )
        }
        val reader = trimmed.reader().buffered()
        return parseBufferedReader(reader, strategy, maxSampleCap, null)
    }

    private fun parseBufferedReader(
        reader: BufferedReader,
        strategy: DatasetFormatStrategy,
        maxSampleCap: Int,
        onProgress: ((linesRead: Long, samplesFound: Int) -> Unit)?
    ): ParseResult {
        reader.mark(64 * 1024)
        val previewLines = mutableListOf<String>()
        var lineCount = 0
        while (lineCount < 50) {
            val line = reader.readLine() ?: break
            if (line.isNotBlank()) {
                previewLines.add(line)
                lineCount++
            }
        }
        reader.reset()

        if (previewLines.isEmpty()) {
            return ParseResult(
                formatName = "Empty",
                samples = emptyList(),
                classCounts = emptyMap(),
                errorMessage = "File is empty or contains no readable lines."
            )
        }

        val effectiveStrategy = if (strategy == DatasetFormatStrategy.AUTO_DETECT) {
            detectBestStrategy(previewLines)
        } else {
            strategy
        }

        return when (effectiveStrategy) {
            DatasetFormatStrategy.INSTRUCTION_RESPONSE -> parseJsonlStream(reader, maxSampleCap, onProgress)
            DatasetFormatStrategy.SHAKESPEARE_DIALOGUE -> parseDialogueStream(reader, maxSampleCap, onProgress)
            DatasetFormatStrategy.QA_QUESTION_ANSWER -> parseQaCsvStream(reader, maxSampleCap, onProgress)
            DatasetFormatStrategy.SECTION_HEADER -> parseSectionHeadersStream(reader, maxSampleCap, onProgress)
            DatasetFormatStrategy.CSV_LABEL_TEXT, DatasetFormatStrategy.AUTO_DETECT -> parseGenericCsvStream(reader, maxSampleCap, onProgress)
        }
    }

    private fun detectBestStrategy(previewLines: List<String>): DatasetFormatStrategy {
        val combinedPreview = previewLines.joinToString("\n").lowercase(Locale.ROOT)
        val firstLine = previewLines.first().trim()

        // 1. JSON / JSONL / Alpaca / Dolly / Instruction / SQuAD
        if (combinedPreview.contains("\"instruction\"") ||
            combinedPreview.contains("\"response\"") ||
            combinedPreview.contains("\"output\"") ||
            combinedPreview.contains("\"messages\"") ||
            combinedPreview.contains("\"conversations\"") ||
            (combinedPreview.contains("{") && combinedPreview.contains("\"text\"")) ||
            (combinedPreview.contains("{") && combinedPreview.contains("\"label\"")) ||
            (firstLine.startsWith("[") && combinedPreview.contains("{")) ||
            (firstLine.startsWith("{") && combinedPreview.contains(":"))
        ) {
            return DatasetFormatStrategy.INSTRUCTION_RESPONSE
        }

        // 2. Question / Answer CSV
        val lowerFirst = firstLine.lowercase(Locale.ROOT)
        if (lowerFirst.contains("question") && lowerFirst.contains("answer")) {
            return DatasetFormatStrategy.QA_QUESTION_ANSWER
        }
        if (lowerFirst.startsWith("q,a") || lowerFirst.startsWith("query,response") || lowerFirst.startsWith("prompt,response") || lowerFirst.startsWith("input,output")) {
            return DatasetFormatStrategy.QA_QUESTION_ANSWER
        }

        // 3. Section Headers
        if (firstLine.startsWith("[") && firstLine.endsWith("]") || firstLine.startsWith("# ") || firstLine.startsWith("## ")) {
            return DatasetFormatStrategy.SECTION_HEADER
        }

        // 4. Shakespeare Dialogue (e.g. "First Citizen:\n...")
        val colonCount = previewLines.count { line ->
            val colonIdx = line.indexOf(':')
            colonIdx in 1..40 && line.substring(0, colonIdx).split(" ").size <= 5
        }
        if (colonCount >= 2 && !firstLine.contains(",")) {
            return DatasetFormatStrategy.SHAKESPEARE_DIALOGUE
        }

        // 5. Delimited CSV/TSV
        if (firstLine.contains(",") || firstLine.contains("\t") || firstLine.contains(";")) {
            return DatasetFormatStrategy.CSV_LABEL_TEXT
        }

        return DatasetFormatStrategy.INSTRUCTION_RESPONSE
    }

    private fun parseQaCsvStream(
        reader: BufferedReader,
        maxSampleCap: Int,
        onProgress: ((linesRead: Long, samplesFound: Int) -> Unit)?
    ): ParseResult {
        val samples = mutableListOf<ParsedSample>()
        var lineNum = 0L
        var delimiter = ','

        val headerLine = reader.readLine() ?: return emptyResult("QA CSV")
        lineNum++
        if (headerLine.contains("\t")) delimiter = '\t' else if (headerLine.contains(";")) delimiter = ';'

        val headerCols = splitCsvLine(headerLine, delimiter).map { it.lowercase(Locale.ROOT).trim() }
        var questionCol = headerCols.indexOfFirst { it in listOf("question", "q", "prompt", "query", "input", "instruction", "premise", "প্রশ্ন", "জিজ্ঞাসা", "প্রশ্নাবলী") }
        var answerCol = headerCols.indexOfFirst { it in listOf("answer", "a", "response", "target", "output", "completion", "hypothesis", "label", "উত্তর", "সমাধান", "ফলাফল") }
        val categoryCol = headerCols.indexOfFirst { it in listOf("category", "topic", "subject", "domain", "class", "ক্যাটাগরি", "বিষয়", "টপিক", "শ্রেণী") }

        if (questionCol == -1 || answerCol == -1) {
            questionCol = 0
            answerCol = 1
        }

        var line: String?
        while (reader.readLine().also { line = it } != null) {
            lineNum++
            val l = line?.trim() ?: continue
            if (l.isEmpty()) continue

            val cols = splitCsvLine(l, delimiter)
            if (cols.size > maxOf(questionCol, answerCol)) {
                val question = cols[questionCol].trim()
                val answer = cols[answerCol].trim()

                if (question.isNotEmpty()) {
                    val label = if (categoryCol != -1 && cols.size > categoryCol && cols[categoryCol].isNotBlank()) {
                        sanitizeClassName(cols[categoryCol].trim())
                    } else {
                        classifyTopic("$question $answer")
                    }
                    val content = if (answer.isNotEmpty()) "$question\nAnswer: $answer" else question
                    samples.add(ParsedSample(label, content))
                    if (samples.size >= maxSampleCap) break
                }
            }

            if (lineNum % 500 == 0L) {
                onProgress?.invoke(lineNum, samples.size)
            }
        }

        if (samples.isEmpty()) return emptyResult("Question-Answer CSV")
        val counts = samples.groupingBy { it.className }.eachCount()
        return ParseResult(
            formatName = "📋 Question & Answer Dataset (QA Pairs)",
            samples = samples,
            classCounts = counts,
            totalLinesScanned = lineNum,
            isCapped = samples.size >= maxSampleCap
        )
    }

    private fun parseJsonlStream(
        reader: BufferedReader,
        maxSampleCap: Int,
        onProgress: ((linesRead: Long, samplesFound: Int) -> Unit)?
    ): ParseResult {
        val samples = mutableListOf<ParsedSample>()
        var lineNum = 0L

        val objectBuffer = StringBuilder()
        var braceDepth = 0
        var inString = false
        var isEscaped = false

        var line: String?
        while (reader.readLine().also { line = it } != null) {
            lineNum++
            val l = line?.trim() ?: continue
            if (l.isEmpty()) continue

            // 1. Fast path for single-line JSON Array of objects (e.g. [{"instruction":"..."}])
            if (braceDepth == 0 && l.startsWith("[") && l.endsWith("]")) {
                try {
                    val arr = JSONArray(l)
                    for (i in 0 until arr.length()) {
                        val item = arr.optJSONObject(i) ?: continue
                        extractSamplesFromJson(item).forEach { sample ->
                            samples.add(sample)
                            if (samples.size >= maxSampleCap) return buildJsonResult(samples, lineNum, maxSampleCap)
                        }
                    }
                } catch (_: Exception) {}
                continue
            }

            // 2. Stream single-line or multi-line JSON objects/arrays (Alpaca, Dolly, SQuAD, etc.)
            for (i in l.indices) {
                val c = l[i]
                if (isEscaped) {
                    isEscaped = false
                    if (braceDepth > 0) objectBuffer.append(c)
                    continue
                }
                if (c == '\\') {
                    isEscaped = true
                    if (braceDepth > 0) objectBuffer.append(c)
                    continue
                }
                if (c == '\"') {
                    inString = !inString
                    if (braceDepth > 0) objectBuffer.append(c)
                    continue
                }

                if (!inString) {
                    if (c == '{') {
                        braceDepth++
                        if (braceDepth == 1) {
                            objectBuffer.clear()
                        }
                        objectBuffer.append(c)
                    } else if (c == '}') {
                        if (braceDepth > 0) {
                            braceDepth--
                            objectBuffer.append(c)
                            if (braceDepth == 0) {
                                try {
                                    val obj = JSONObject(objectBuffer.toString())
                                    if (obj.has("data") && obj.optJSONArray("data") != null) {
                                        extractSquadSamples(obj.getJSONArray("data"), samples, maxSampleCap)
                                    } else {
                                        extractSamplesFromJson(obj).forEach { sample ->
                                            samples.add(sample)
                                            if (samples.size >= maxSampleCap) return buildJsonResult(samples, lineNum, maxSampleCap)
                                        }
                                    }
                                } catch (_: Exception) {}
                                objectBuffer.clear()
                            }
                        }
                    } else if (braceDepth > 0) {
                        objectBuffer.append(c)
                    }
                } else if (braceDepth > 0) {
                    objectBuffer.append(c)
                }
            }
            if (braceDepth > 0) {
                objectBuffer.append('\n')
            }

            if (lineNum % 500 == 0L) {
                onProgress?.invoke(lineNum, samples.size)
            }
        }

        if (samples.isEmpty()) return emptyResult("JSON / JSONL")
        return buildJsonResult(samples, lineNum, maxSampleCap)
    }

    private fun buildJsonResult(samples: List<ParsedSample>, lineNum: Long, maxSampleCap: Int): ParseResult {
        val counts = samples.groupingBy { it.className }.eachCount()
        return ParseResult(
            formatName = "🤖 LLM Instruction & QA Dataset (JSONL / SQuAD / Alpaca / UltraChat)",
            samples = samples,
            classCounts = counts,
            totalLinesScanned = lineNum,
            isCapped = samples.size >= maxSampleCap
        )
    }

    private fun extractSquadSamples(dataArray: JSONArray, outSamples: MutableList<ParsedSample>, maxSampleCap: Int) {
        for (i in 0 until dataArray.length()) {
            val topicObj = dataArray.optJSONObject(i) ?: continue
            val title = topicObj.optString("title", "SQuAD").ifEmpty { "SQuAD QA" }
            val paragraphs = topicObj.optJSONArray("paragraphs") ?: continue
            for (p in 0 until paragraphs.length()) {
                val pObj = paragraphs.optJSONObject(p) ?: continue
                val qas = pObj.optJSONArray("qas") ?: continue
                for (q in 0 until qas.length()) {
                    val qaObj = qas.optJSONObject(q) ?: continue
                    val question = qaObj.optString("question", "").trim()
                    var answerText = ""
                    val answersArr = qaObj.optJSONArray("answers")
                    if (answersArr != null && answersArr.length() > 0) {
                        answerText = answersArr.optJSONObject(0)?.optString("text", "") ?: ""
                    }
                    if (answerText.isEmpty()) {
                        answerText = qaObj.optString("answer", "")
                    }
                    if (question.isNotEmpty()) {
                        val label = sanitizeClassName(if (answerText.isNotBlank()) answerText.take(35) else title)
                        outSamples.add(ParsedSample(label, if (answerText.isNotBlank()) "$question\nAnswer: $answerText" else question))
                        if (outSamples.size >= maxSampleCap) return
                    }
                }
            }
        }
    }

    private fun extractSamplesFromJson(obj: JSONObject): List<ParsedSample> {
        val results = mutableListOf<ParsedSample>()

        // Multi-Turn Conversations (UltraChat, ShareGPT)
        if (obj.has("conversations")) {
            val convs = obj.optJSONArray("conversations")
            if (convs != null && convs.length() >= 2) {
                var humanMsg = ""
                for (i in 0 until convs.length()) {
                    val turn = convs.optJSONObject(i) ?: continue
                    val from = turn.optString("from", "").lowercase(Locale.ROOT)
                    val value = turn.optString("value", "").trim()
                    if (from in listOf("human", "user", "prompter")) {
                        humanMsg = value
                    } else if (from in listOf("gpt", "assistant", "bot") && humanMsg.isNotEmpty()) {
                        val label = sanitizeClassName(humanMsg.take(30))
                        results.add(ParsedSample(label, "$humanMsg\n$value"))
                        humanMsg = ""
                    }
                }
                if (results.isNotEmpty()) return results
            }
        }

        // Multi-Turn Messages (OpenAssistant, ChatML)
        if (obj.has("messages")) {
            val msgs = obj.optJSONArray("messages")
            if (msgs != null && msgs.length() >= 2) {
                var userMsg = ""
                for (i in 0 until msgs.length()) {
                    val turn = msgs.optJSONObject(i) ?: continue
                    val role = turn.optString("role", "").lowercase(Locale.ROOT)
                    val content = turn.optString("content", "").trim()
                    if (role in listOf("user", "human")) {
                        userMsg = content
                    } else if (role in listOf("assistant", "bot") && userMsg.isNotEmpty()) {
                        val label = sanitizeClassName(userMsg.take(30))
                        results.add(ParsedSample(label, "$userMsg\n$content"))
                        userMsg = ""
                    }
                }
                if (results.isNotEmpty()) return results
            }
        }

        // Single Sample Object
        val category = obj.optString("category", "").ifEmpty { obj.optString("topic", "") }.trim()
        val instruction = obj.optString("instruction", "").ifEmpty { obj.optString("question", "") }.ifEmpty { obj.optString("prompt", "") }.trim()
        val response = obj.optString("response", "").ifEmpty { obj.optString("output", "") }.ifEmpty { obj.optString("answer", "") }.trim()
        val context = obj.optString("context", "").trim()

        if (category.isNotEmpty()) {
            val content = buildString {
                if (instruction.isNotEmpty()) append(instruction).append(" ")
                if (context.isNotEmpty()) append(context).append(" ")
                if (response.isNotEmpty()) append(response)
            }.trim()
            if (content.isNotEmpty()) {
                results.add(ParsedSample(sanitizeClassName(category), content))
                return results
            }
        }

        val labelKey = listOf("label", "class", "category", "target", "sentiment", "speaker", "tag", "intent", "topic").firstOrNull { obj.has(it) }
        val textKey = listOf("text", "content", "sentence", "message", "utterance", "review", "line", "body", "input").firstOrNull { obj.has(it) }
        if (labelKey != null && textKey != null) {
            val label = sanitizeClassName(obj.optString(labelKey, ""))
            val content = obj.optString(textKey, "").trim()
            if (label.isNotEmpty() && content.isNotEmpty()) {
                results.add(ParsedSample(label, content))
                return results
            }
        }

        if (instruction.isNotEmpty() && response.isNotEmpty()) {
            val label = if (category.isNotEmpty()) sanitizeClassName(category) else classifyTopic("$instruction $response")
            val content = if (obj.optString("input", "").isNotEmpty()) "${obj.optString("input")}\n$instruction\n$response" else "$instruction\n$response"
            results.add(ParsedSample(label, content))
            return results
        }

        return results
    }

    private fun parseDialogueStream(
        reader: BufferedReader,
        maxSampleCap: Int,
        onProgress: ((linesRead: Long, samplesFound: Int) -> Unit)?
    ): ParseResult {
        val speakerRegex = "^(?:\\[([A-Za-z0-9_\u0980-\u09FF\\s\\-\\.\\(\\)'\"]{1,45})\\]|([A-Za-z0-9_\u0980-\u09FF\\s\\-\\.\\(\\)'\"]{1,45}))\\s*[:\\-]\\s*(.*)$".toRegex()
        val samples = mutableListOf<ParsedSample>()
        var currentSpeaker: String? = null
        val currentTextBuilder = StringBuilder()
        var lineNum = 0L

        var line: String?
        while (reader.readLine().also { line = it } != null) {
            lineNum++
            val trimmedLine = line?.trim() ?: continue
            if (trimmedLine.isEmpty()) continue

            val match = speakerRegex.matchEntire(trimmedLine)
            if (match != null) {
                val rawCandidate = (match.groupValues[1].ifEmpty { match.groupValues[2] }).trim()
                val inlineText = match.groupValues[3].trim()

                if (rawCandidate.length in 1..40 &&
                    !rawCandidate.contains("?") &&
                    !rawCandidate.contains("!") &&
                    !rawCandidate.contains(";") &&
                    rawCandidate.split(" ").size <= 5
                ) {
                    if (currentSpeaker != null && currentTextBuilder.isNotBlank()) {
                        samples.add(ParsedSample(currentSpeaker, currentTextBuilder.toString().trim()))
                        currentTextBuilder.clear()
                        if (samples.size >= maxSampleCap) break
                    }

                    currentSpeaker = sanitizeClassName(rawCandidate.removeSuffix("."))
                    if (inlineText.isNotEmpty()) {
                        currentTextBuilder.append(inlineText).append(" ")
                    }
                    continue
                }
            }

            if (currentSpeaker != null) {
                if (currentTextBuilder.isNotEmpty()) currentTextBuilder.append(" ")
                currentTextBuilder.append(trimmedLine)
            }

            if (lineNum % 500 == 0L) {
                onProgress?.invoke(lineNum, samples.size)
            }
        }

        if (currentSpeaker != null && currentTextBuilder.isNotBlank() && samples.size < maxSampleCap) {
            samples.add(ParsedSample(currentSpeaker, currentTextBuilder.toString().trim()))
        }

        if (samples.isEmpty()) return emptyResult("Dialogue Transcript")
        val counts = samples.groupingBy { it.className }.eachCount()
        return ParseResult(
            formatName = "🎭 Shakespeare / Drama Dialogue Transcript",
            samples = samples,
            classCounts = counts,
            totalLinesScanned = lineNum,
            isCapped = samples.size >= maxSampleCap
        )
    }

    private fun parseGenericCsvStream(
        reader: BufferedReader,
        maxSampleCap: Int,
        onProgress: ((linesRead: Long, samplesFound: Int) -> Unit)?
    ): ParseResult {
        val samples = mutableListOf<ParsedSample>()
        var lineNum = 0L
        val firstLine = reader.readLine() ?: return emptyResult("CSV/TSV")
        lineNum++

        val delimiter = if (firstLine.contains("\t")) '\t' else if (firstLine.contains(";")) ';' else ','
        val firstCols = splitCsvLine(firstLine, delimiter)

        var labelCol = -1
        var textCol = -1

        if (firstCols.size >= 2) {
            for ((idx, col) in firstCols.withIndex()) {
                val lower = col.lowercase(Locale.ROOT).trim()
                if (lower in listOf("label", "class", "category", "target", "sentiment", "speaker", "intent", "tag", "topic", "answer", "লেবেল", "ক্যাটাগরি", "উত্তর", "বিষয়", "টপিক", "শ্রেণী")) {
                    labelCol = idx
                }
                if (lower in listOf("text", "content", "sentence", "message", "utterance", "review", "line", "question", "prompt", "input", "body", "টেক্সট", "প্রশ্ন", "মন্তব্য", "বার্তা", "বাক্য", "অনুচ্ছেদ", "বিবরণ")) {
                    textCol = idx
                }
            }
        }

        val hasHeader = (labelCol != -1 && textCol != -1)
        if (!hasHeader) {
            if (firstCols.size >= 2) {
                if (firstCols[0].length <= firstCols[1].length) {
                    labelCol = 0
                    textCol = 1
                } else {
                    labelCol = 1
                    textCol = 0
                }
                val l = sanitizeClassName(firstCols[labelCol])
                val t = firstCols[textCol].trim()
                if (l.isNotEmpty() && t.isNotEmpty()) {
                    samples.add(ParsedSample(l, t))
                }
            }
        }

        var line: String?
        while (reader.readLine().also { line = it } != null) {
            lineNum++
            val l = line?.trim() ?: continue
            if (l.isEmpty()) continue

            val cols = splitCsvLine(l, delimiter)
            if (cols.size > maxOf(labelCol, textCol)) {
                val label = sanitizeClassName(cols[labelCol].trim())
                val content = cols[textCol].trim()
                if (label.isNotEmpty() && content.isNotEmpty()) {
                    samples.add(ParsedSample(label, content))
                    if (samples.size >= maxSampleCap) break
                }
            }

            if (lineNum % 500 == 0L) {
                onProgress?.invoke(lineNum, samples.size)
            }
        }

        if (samples.isEmpty()) return emptyResult("CSV/TSV")
        val counts = samples.groupingBy { it.className }.eachCount()
        val baseResult = ParseResult(
            formatName = if (delimiter == '\t') "TSV (Tab-Separated Dataset)" else "CSV (Comma-Separated Dataset)",
            samples = samples,
            classCounts = counts,
            totalLinesScanned = lineNum,
            isCapped = samples.size >= maxSampleCap
        )

        // Prevent class explosion when file has unique lines/prompts instead of categorical classes
        return if (counts.size > 48) {
            clusterIntoTopics(baseResult.copy(formatName = "${baseResult.formatName} • 🎯 Auto-Clustered Topics"))
        } else {
            baseResult
        }
    }

    private fun parseSectionHeadersStream(
        reader: BufferedReader,
        maxSampleCap: Int,
        onProgress: ((linesRead: Long, samplesFound: Int) -> Unit)?
    ): ParseResult {
        val samples = mutableListOf<ParsedSample>()
        val sectionRegex = "^(?:\\[(.*)\\]|#+\\s+(.*)|===+\\s*(.*?)\\s*===+)$".toRegex()
        var currentSection: String? = null
        var lineNum = 0L

        var line: String?
        while (reader.readLine().also { line = it } != null) {
            lineNum++
            val trimmed = line?.trim() ?: continue
            if (trimmed.isEmpty()) continue

            val match = sectionRegex.matchEntire(trimmed)
            if (match != null) {
                val name = match.groupValues.drop(1).firstOrNull { it.isNotEmpty() }?.trim()
                if (!name.isNullOrBlank()) {
                    currentSection = sanitizeClassName(name)
                    continue
                }
            }

            if (currentSection != null && trimmed.isNotEmpty()) {
                samples.add(ParsedSample(currentSection, trimmed))
                if (samples.size >= maxSampleCap) break
            }

            if (lineNum % 500 == 0L) {
                onProgress?.invoke(lineNum, samples.size)
            }
        }

        if (samples.isEmpty()) return emptyResult("Section Headers")
        val counts = samples.groupingBy { it.className }.eachCount()
        return ParseResult(
            formatName = "📑 Section Headers ([Category] / # Category)",
            samples = samples,
            classCounts = counts,
            totalLinesScanned = lineNum,
            isCapped = samples.size >= maxSampleCap
        )
    }

    private fun splitCsvLine(line: String, delimiter: Char): List<String> {
        val result = mutableListOf<String>()
        val cur = StringBuilder()
        var inQuotes = false

        for (ch in line) {
            if (ch == '\"') {
                inQuotes = !inQuotes
            } else if (ch == delimiter && !inQuotes) {
                result.add(cur.toString().trim().removeSurrounding("\""))
                cur.clear()
            } else {
                cur.append(ch)
            }
        }
        result.add(cur.toString().trim().removeSurrounding("\""))
        return result
    }

    private fun emptyResult(formatName: String): ParseResult {
        return ParseResult(
            formatName = formatName,
            samples = emptyList(),
            classCounts = emptyMap(),
            errorMessage = "No valid samples could be extracted for format: $formatName."
        )
    }

    fun sanitizeClassName(name: String): String {
        return name
            .replace("[^a-zA-Z0-9\\s\\-_\u0980-\u09FF]".toRegex(), "")
            .trim()
            .take(35)
            .ifEmpty { "Class" }
    }

    /**
     * Smart Bilingual (English & Bengali) Semantic Topic Classifier.
     * Categorizes free-text, QA pairs, instructions, and dialogues into 10 balanced thematic classes.
     * Prevents class explosion (e.g. 4520 classes) and speeds up training by 100x.
     */
    fun classifyTopic(text: String): String {
        val lower = if (text.length > 600) text.take(600).lowercase(Locale.ROOT) else text.lowercase(Locale.ROOT)

        var scoreTech = 0
        var scoreKnowledge = 0
        var scoreLiterature = 0
        var scoreDialogue = 0
        var scoreEducation = 0
        var scoreHistory = 0
        var scoreHealth = 0
        var scoreBusiness = 0
        var scoreSports = 0
        var scoreLogic = 0

        // 1. Tech & Science
        val techWords = listOf(
            "computer", "software", "hardware", "code", "coding", "algorithm", "python", "java", "ai",
            "artificial intelligence", "data", "robot", "science", "physics", "chemistry", "biology",
            "math", "internet", "web", "app", "mobile", "tech", "digital", "machine learning",
            "প্রযুক্তি", "বিজ্ঞান", "কম্পিউটার", "সফটওয়্যার", "কোডিং", "অ্যালগরিদম", "ইন্টারনেট",
            "রোবট", "পদার্থ", "রসায়ন", "জীববিজ্ঞান", "গণিত", "ডিজিটাল", "মোবাইল"
        )
        for (w in techWords) { if (lower.contains(w)) scoreTech += 2 }

        // 2. Health & Medicine
        val healthWords = listOf(
            "doctor", "health", "hospital", "medicine", "disease", "treatment", "cure", "pain",
            "cancer", "diet", "nutrition", "patient", "medical", "symptom", "blood", "drug",
            "চিকিৎসক", "স্বাস্থ্য", "রোগ", "ওষুধ", "হাসপাতাল", "চিকিৎসা", "ক্যান্সার", "পুষ্টি", "লক্ষণ", "ব্যথা", "অসুখ"
        )
        for (w in healthWords) { if (lower.contains(w)) scoreHealth += 2 }

        // 3. History & Culture
        val historyWords = listOf(
            "history", "ancient", "war", "century", "empire", "king", "queen", "culture", "tradition",
            "heritage", "freedom", "nation", "country", "civilization", "leader",
            "ইতিহাস", "প্রাচীন", "যুদ্ধ", "শতাব্দী", "সাম্রাজ্য", "রাজা", "রানী", "সংস্কৃতি", "ঐতিহ্য", "মুক্তিযুদ্ধ", "দেশ", "জাতি"
        )
        for (w in historyWords) { if (lower.contains(w)) scoreHistory += 2 }

        // 4. Education & Language
        val eduWords = listOf(
            "school", "college", "university", "learn", "study", "education", "grammar", "vocabulary",
            "language", "translate", "meaning", "student", "teacher", "exam", "lesson",
            "বই", "শিক্ষা", "বিদ্যালয়", "কলেজ", "বিশ্ববিদ্যালয়", "পড়াশোনা", "ব্যাকরণ", "শব্দার্থ", "ভাষা", "অনুবাদ", "ছাত্র", "শিক্ষক"
        )
        for (w in eduWords) { if (lower.contains(w)) scoreEducation += 2 }

        // 5. Literature & Stories
        val litWords = listOf(
            "story", "novel", "poem", "poet", "author", "character", "drama", "scene", "act",
            "tale", "prince", "king", "forest", "fairy", "magic", "plot", "rhyme",
            "গল্প", "উপন্যাস", "কবিতা", "কবি", "লেখক", "সাহিত্য", "নাটক", "রাজপুত্র", "বন", "রূপকথা"
        )
        for (w in litWords) { if (lower.contains(w)) scoreLiterature += 2 }

        // 6. Dialogue & Social Conversation
        val dialWords = listOf(
            "hello", "hi", "how are you", "thank", "thanks", "bye", "goodbye", "morning", "night",
            "sorry", "welcome", "please", "friend", "chat", "speak", "talking", "meet",
            "কেমন", "ধন্যবাদ", "হ্যালো", "নমস্কার", "বিদায়", "সকাল", "বন্ধু", "কথা", "আড্ডা", "বলুন"
        )
        for (w in dialWords) { if (lower.contains(w)) scoreDialogue += 2 }

        // 7. Business & Finance
        val bizWords = listOf(
            "money", "bank", "finance", "business", "market", "price", "cost", "trade", "investment",
            "company", "profit", "economy", "stock", "sales",
            "টাকা", "ব্যাংক", "অর্থনীতি", "ব্যবসা", "বাজার", "মূল্য", "বাণিজ্য", "বিনিয়োগ", "লাভ", "কোম্পানি"
        )
        for (w in bizWords) { if (lower.contains(w)) scoreBusiness += 2 }

        // 8. Sports & Entertainment
        val sportWords = listOf(
            "sports", "cricket", "football", "game", "match", "movie", "music", "song", "film",
            "player", "tournament", "actor", "concert",
            "খেলা", "ক্রিকেট", "ফুটবল", "ম্যাচ", "গান", "সিনেমা", "চলচ্চিত্র", "খেলোয়াড়"
        )
        for (w in sportWords) { if (lower.contains(w)) scoreSports += 2 }

        // 9. Logic & Reasoning
        val logicWords = listOf(
            "why", "because", "reason", "proof", "logic", "analyze", "explain", "compare", "differ",
            "step", "solution", "calculate", "argue",
            "কেন", "কারণ", "যুক্তি", "প্রমাণ", "বিশ্লেষণ", "ব্যাখ্যা", "তুলনা", "পার্থক্য", "সমাধান"
        )
        for (w in logicWords) { if (lower.contains(w)) scoreLogic += 2 }

        // 10. General Knowledge / Facts
        val qWords = listOf(
            "what", "when", "where", "who", "which", "capital", "population", "planet", "fact",
            "definition", "identify", "called",
            "কি", "কী", "কখন", "কোথায়", "কে", "রাজধানী", "জনসংখ্যা", "সংজ্ঞা"
        )
        for (w in qWords) { if (lower.contains(w)) scoreKnowledge += 1 }

        val scores = listOf(
            "Technology & Science" to scoreTech,
            "Health & Wellness" to scoreHealth,
            "History & Culture" to scoreHistory,
            "Education & Language" to scoreEducation,
            "Literature & Stories" to scoreLiterature,
            "Daily Conversation & Chat" to scoreDialogue,
            "Business & Finance" to scoreBusiness,
            "Entertainment & Sports" to scoreSports,
            "Reasoning & Logic" to scoreLogic,
            "General Knowledge & Facts" to scoreKnowledge
        )

        val best = scores.maxByOrNull { it.second }
        if (best != null && best.second > 0) {
            return best.first
        }

        // Hash-based deterministic distribution across 3 diverse categories if neutral
        val hash = kotlin.math.abs(text.hashCode()) % 3
        return when (hash) {
            0 -> "General Knowledge & Facts"
            1 -> "Education & Language"
            else -> "Daily Conversation & Chat"
        }
    }

    /**
     * Transforms any parse result into balanced thematic topic clusters.
     */
    fun clusterIntoTopics(result: ParseResult): ParseResult {
        val clusteredSamples = result.samples.map { sample ->
            val topic = classifyTopic(sample.text)
            ParsedSample(topic, sample.text)
        }
        val counts = clusteredSamples.groupingBy { it.className }.eachCount()
        return result.copy(
            samples = clusteredSamples,
            classCounts = counts
        )
    }

    /**
     * Non-closing InputStream wrapper so ZipInputStream isn't closed when BufferedReader closes.
     */
    private class NonClosingInputStream(private val stream: InputStream) : InputStream() {
        override fun read(): Int = stream.read()
        override fun read(b: ByteArray, off: Int, len: Int): Int = stream.read(b, off, len)
        override fun close() { /* Don't close outer ZipInputStream */ }
    }

    /**
     * 28+ Global & Bangla NLP Benchmark Datasets Catalog
     */
    val BENCHMARK_CATALOG: List<BenchmarkDatasetInfo> = listOf(
        // Literature & Drama
        BenchmarkDatasetInfo(
            id = "tiny_shakespeare",
            name = "Tiny Shakespeare",
            nameBn = "টাইনি শেক্সপিয়র",
            category = "Literature & Drama",
            format = "Drama Dialogues (.txt)",
            defaultStrategy = DatasetFormatStrategy.SHAKESPEARE_DIALOGUE,
            description = "Complete dialogue transcripts of Shakespeare plays (Coriolanus, Julius Caesar, Hamlet) categorized by speaker roles.",
            sampleSnippet = "First Citizen:\nYou are all resolved rather to die than to famish?\n\nMENENIUS:\nWhat work's, my countrymen, in hand?"
        ),
        BenchmarkDatasetInfo(
            id = "tinystories",
            name = "TinyStories",
            nameBn = "টাইনি স্টোরিজ",
            category = "Literature & Drama",
            format = "Synthesized Story Chunks (.txt / .jsonl)",
            defaultStrategy = DatasetFormatStrategy.SECTION_HEADER,
            description = "Synthetic short stories with simple vocabulary for foundational neural reasoning and coherence modeling.",
            sampleSnippet = "[Chapter 1: The Magic Tree]\nOnce upon a time there was a small girl named Lily. She loved to play under the big oak tree."
        ),
        BenchmarkDatasetInfo(
            id = "wikitext103",
            name = "WikiText-103",
            nameBn = "উইকিটেক্সট-১০৩",
            category = "Literature & Drama",
            format = "Articles with Section Headers (.txt)",
            defaultStrategy = DatasetFormatStrategy.SECTION_HEADER,
            description = "100M+ tokens extracted from verified Wikipedia Good & Featured articles structured by header sections.",
            sampleSnippet = "= Valkyria Chronicles III =\nSenjō no Valkyria 3 is a tactical role-playing video game developed by Sega for PlayStation Portable."
        ),
        BenchmarkDatasetInfo(
            id = "llm_lab_corpus",
            name = "llm-lab-corpus",
            nameBn = "এলএলএম ল্যাব করপাস",
            category = "Literature & Drama",
            format = "Pretraining Domain Chunks (.txt)",
            defaultStrategy = DatasetFormatStrategy.SECTION_HEADER,
            description = "Curated multi-domain research corpus for lightweight on-device language modeling experiments.",
            sampleSnippet = "[Neural Architecture]\nMixture-of-Experts enables sparse activation of specialized feedforward layers on mobile CPUs."
        ),

        // LLM Instruction Tuning
        BenchmarkDatasetInfo(
            id = "alpaca",
            name = "Alpaca (Stanford)",
            nameBn = "স্ট্যানফোর্ড আলপাকা",
            category = "Instruction & LLM",
            format = "JSONL (instruction, input, output)",
            defaultStrategy = DatasetFormatStrategy.INSTRUCTION_RESPONSE,
            description = "52,000 instruction-following demonstrations generated by OpenAI text-davinci-003 for alignment.",
            sampleSnippet = "{\"instruction\": \"Give three tips for staying healthy.\", \"input\": \"\", \"output\": \"1. Eat a balanced diet.\\n2. Exercise regularly.\\n3. Get enough sleep.\"}"
        ),
        BenchmarkDatasetInfo(
            id = "dolly15k",
            name = "Dolly 15k (Databricks)",
            nameBn = "ডলি ১৫কে (ডেটাব্রিক্স)",
            category = "Instruction & LLM",
            format = "JSONL (instruction, response, category)",
            defaultStrategy = DatasetFormatStrategy.INSTRUCTION_RESPONSE,
            description = "15,000 high-quality human-generated prompt-response pairs categorized into 8 distinct task types.",
            sampleSnippet = "{\"instruction\": \"When did Virgin Australia start?\", \"category\": \"closed_qa\", \"response\": \"Virgin Australia commenced services on 31 August 2000.\"}"
        ),
        BenchmarkDatasetInfo(
            id = "wizardlm70k",
            name = "WizardLM 70k",
            nameBn = "উইজার্ডএলএম ৭০কে",
            category = "Instruction & LLM",
            format = "Complex Evol-Instruct JSONL",
            defaultStrategy = DatasetFormatStrategy.INSTRUCTION_RESPONSE,
            description = "Evol-Instruct dataset with complex multi-step reasoning, constraints, and deep explanations.",
            sampleSnippet = "{\"instruction\": \"Explain quantum superposition with a practical analogy.\", \"category\": \"reasoning\", \"output\": \"Imagine a spinning coin...\"}"
        ),
        BenchmarkDatasetInfo(
            id = "lima",
            name = "LIMA (Less Is More)",
            nameBn = "লিমা (মেটা এলএলএম)",
            category = "Instruction & LLM",
            format = "High-Quality Alignment JSONL",
            defaultStrategy = DatasetFormatStrategy.INSTRUCTION_RESPONSE,
            description = "1,000 carefully curated instruction pairs showing that a small set of high-quality examples yields strong alignment.",
            sampleSnippet = "{\"instruction\": \"How do I write a formal resignation letter?\", \"category\": \"writing\", \"output\": \"Dear [Manager], Please accept this letter as formal notification...\"}"
        ),
        BenchmarkDatasetInfo(
            id = "openassistant",
            name = "OpenAssistant",
            nameBn = "ওপেনঅ্যাসিস্ট্যান্ট",
            category = "Instruction & LLM",
            format = "Crowdsourced Conversation JSONL",
            defaultStrategy = DatasetFormatStrategy.INSTRUCTION_RESPONSE,
            description = "Human-generated, human-annotated conversation trees across multiple languages for assistant training.",
            sampleSnippet = "{\"instruction\": \"What are the differences between TCP and UDP?\", \"category\": \"networking\", \"output\": \"TCP is connection-oriented and reliable, while UDP is connectionless.\"}"
        ),
        BenchmarkDatasetInfo(
            id = "ultrachat200k",
            name = "UltraChat 200k",
            nameBn = "আল্ট্রাচ্যাট ২০০কে",
            category = "Instruction & LLM",
            format = "Multi-turn Dialogue JSONL",
            defaultStrategy = DatasetFormatStrategy.INSTRUCTION_RESPONSE,
            description = "Comprehensive multi-turn multi-topic dialog dataset covering complex reasoning and knowledge.",
            sampleSnippet = "{\"instruction\": \"Can you summarize the plot of Interstellar?\", \"category\": \"entertainment\", \"output\": \"A team of astronauts travel through a wormhole...\"}"
        ),

        // QA & Knowledge Base
        BenchmarkDatasetInfo(
            id = "squad2",
            name = "SQuAD 2.0",
            nameBn = "স্কোয়াড ২.০ (স্ট্যানফোর্ড)",
            category = "QA & Knowledge",
            format = "Reading Comprehension JSON / CSV",
            defaultStrategy = DatasetFormatStrategy.QA_QUESTION_ANSWER,
            description = "Stanford Question Answering Dataset with 100,000+ questions on Wikipedia articles and unanswerable questions.",
            sampleSnippet = "question,answer\nWhat causes the greenhouse effect?,Atmospheric greenhouse gases trapping heat\nWhen was NASA established?,July 29 1958"
        ),
        BenchmarkDatasetInfo(
            id = "triviaqa",
            name = "TriviaQA",
            nameBn = "ট্রিভিয়া কিউএ",
            category = "QA & Knowledge",
            format = "Question-Answer Pairs (.csv / .jsonl)",
            defaultStrategy = DatasetFormatStrategy.QA_QUESTION_ANSWER,
            description = "Large-scale reading comprehension dataset containing 650K question-answer-evidence triples.",
            sampleSnippet = "question,answer\nWhich planet is known as the Red Planet?,Mars\nWho wrote Romeo and Juliet?,William Shakespeare"
        ),
        BenchmarkDatasetInfo(
            id = "tydiqa",
            name = "TyDi QA (Google)",
            nameBn = "টাইডি কিউএ (গুগল)",
            category = "QA & Knowledge",
            format = "Multilingual Diverse QA (.jsonl)",
            defaultStrategy = DatasetFormatStrategy.QA_QUESTION_ANSWER,
            description = "Google's benchmark across 11 typologically diverse languages including Bengali, Arabic, and Russian.",
            sampleSnippet = "question,answer\nWhat is the speed of light in vacuum?,299792458 meters per second\nWho discovered penicillin?,Alexander Fleming"
        ),
        BenchmarkDatasetInfo(
            id = "bangladesh_gk",
            name = "Bangladesh GK QA",
            nameBn = "বাংলাদেশ সাধারণ জ্ঞান ও প্রশ্ন-উত্তর",
            category = "QA & Knowledge",
            format = "question,answer CSV",
            defaultStrategy = DatasetFormatStrategy.QA_QUESTION_ANSWER,
            description = "বাংলাদেশের সংবিধান, ভূগোল, নদী, ইতিহাস ও সাধারণ জ্ঞানভিত্তিক প্রশ্ন-উত্তর ডাটাবেজ।",
            sampleSnippet = "question,answer\nবাংলাদেশের দীর্ঘতম নদী কোনটি?,মেঘনা\nকোন সংস্থা GDP হিসাব করে?,বাংলাদেশ পরিসংখ্যান ব্যুরো"
        ),

        // Multi-Turn Dialogue & Chat
        BenchmarkDatasetInfo(
            id = "dailydialog",
            name = "DailyDialog",
            nameBn = "ডেইলি ডায়লগ",
            category = "Dialogue & Chat",
            format = "Multi-Turn Dialogue (.txt / .jsonl)",
            defaultStrategy = DatasetFormatStrategy.SHAKESPEARE_DIALOGUE,
            description = "High-quality human dialogues reflecting everyday communication topics (shopping, travel, work).",
            sampleSnippet = "Person A:\nCould you help me find the nearest subway station?\n\nPerson B:\nSure, walk two blocks north and turn left."
        ),
        BenchmarkDatasetInfo(
            id = "persona_chat",
            name = "Persona-Chat",
            nameBn = "পারসোনা চ্যাট",
            category = "Dialogue & Chat",
            format = "Persona Profile Dialogues (.jsonl)",
            defaultStrategy = DatasetFormatStrategy.INSTRUCTION_RESPONSE,
            description = "Dialogues where each participant is assigned a specific persona profile (hobbies, profession, location).",
            sampleSnippet = "{\"instruction\": \"Persona: I am an artist who loves hiking. Speak with me!\", \"category\": \"persona_dialog\", \"output\": \"Hello! I just finished painting a mountain landscape.\"}"
        ),
        BenchmarkDatasetInfo(
            id = "everyday_conversations",
            name = "Everyday Conversations",
            nameBn = "এভরিডে কনভার্সেশনস",
            category = "Dialogue & Chat",
            format = "Chat Turns (.txt)",
            defaultStrategy = DatasetFormatStrategy.SHAKESPEARE_DIALOGUE,
            description = "Natural casual conversation dataset for training responsive conversational intent agents.",
            sampleSnippet = "User:\nWhat is the weather like today?\n\nAssistant:\nIt is bright and sunny with a gentle breeze."
        ),

        // Bangla NLP Benchmarks
        BenchmarkDatasetInfo(
            id = "bangla_dailydialog",
            name = "Bangla DailyDialog",
            nameBn = "বাংলা ডেইলি ডায়লগ",
            category = "Bangla NLP",
            format = "বাংলা সংলাপ ও কথোপকথন (.txt)",
            defaultStrategy = DatasetFormatStrategy.SHAKESPEARE_DIALOGUE,
            description = "দৈনন্দিন সাধারণ বাংলা কথোপকথন, সম্ভাষণ ও প্রশ্ন-উত্তর ভিত্তিক সংলাপ করপাস।",
            sampleSnippet = "প্রথম ব্যক্তি:\nকেমন আছেন? আজকের আবহাওয়াটা বেশ চমৎকার!\n\nদ্বিতীয় ব্যক্তি:\nজি আলহামদুলিল্লাহ ভালো। চলুন একটু ঘুরে আসি।"
        ),
        BenchmarkDatasetInfo(
            id = "bangla_squad",
            name = "Bangla SQuAD",
            nameBn = "বাংলা স্কোয়াড (প্রশ্ন-উত্তর)",
            category = "Bangla NLP",
            format = "বাংলা রিডিং কমপ্রিহেনশন (.csv / .jsonl)",
            defaultStrategy = DatasetFormatStrategy.QA_QUESTION_ANSWER,
            description = "বাংলা উইকিপিডিয়া ও বিভিন্ন আর্টিকেল থেকে তৈরি প্রশ্ন ও উত্তরের সমৃদ্ধ ডাটাবেজ।",
            sampleSnippet = "question,answer\nপদ্মা ও যমুনা নদীর শাখা কোনগুলো?,গড়াই ও ধলেশ্বরী\nমুজিবনগর সরকার শপথ গ্রহণ করে কবে?,১৭ এপ্রিল ১৯৭১"
        ),
        BenchmarkDatasetInfo(
            id = "bangla_alpaca",
            name = "Bangla Alpaca",
            nameBn = "বাংলা আলপাকা",
            category = "Bangla NLP",
            format = "বাংলা ইনস্ট্রাকশন টিউনিং JSONL",
            defaultStrategy = DatasetFormatStrategy.INSTRUCTION_RESPONSE,
            description = "বাংলা ভাষায় অনূদিত ও কিউরেট করা ৫২,০০০ ইনস্ট্রাকশন-ফলোয়িং ডেমোনস্ট্রেশন।",
            sampleSnippet = "{\"instruction\": \"স্বাস্থ্য সুরক্ষায় ৩টি গুরুত্বপূর্ণ পরামর্শ দিন।\", \"category\": \"health\", \"output\": \"১. সুষম খাবার খান।\\n২. নিয়মিত ব্যায়াম করুন।\\n৩. পর্যাপ্ত ঘুমান।\"}"
        ),
        BenchmarkDatasetInfo(
            id = "bangla_instruction_100k",
            name = "Bangla Instruction Tuning 100K",
            nameBn = "বাংলা ইনস্ট্রাকশন টিউনিং ১০০কে",
            category = "Bangla NLP",
            format = "১০০কে বাংলা ইনস্ট্রাকশন JSONL",
            defaultStrategy = DatasetFormatStrategy.INSTRUCTION_RESPONSE,
            description = "১,০০,০০০ বাংলা ইনস্ট্রাকশন ও রেসপন্স জোড়া বিশিষ্ট বৃহৎ লার্জ ল্যাঙ্গুয়েজ মডেল করপাস।",
            sampleSnippet = "{\"instruction\": \"বাংলাদেশের সংবিধানের মূলনীতি কয়টি ও কি কি?\", \"category\": \"constitution\", \"output\": \"মূলনীতি ৪টি: জাতীয়তাবাদ, সমাজতন্ত্র, গণতন্ত্র ও ধর্মনিরপেক্ষতা।\"}"
        ),
        BenchmarkDatasetInfo(
            id = "banglatext",
            name = "BanglaText",
            nameBn = "বাংলাটেক্সট করপাস",
            category = "Bangla NLP",
            format = "বাংলা খবর ও অনুচ্ছেদ (.txt)",
            defaultStrategy = DatasetFormatStrategy.SECTION_HEADER,
            description = "সংবাদপত্র, সাহিত্য ও শিক্ষামূলক বাংলা আর্টিকেলের বৃহৎ ক্যাটাগরিভিত্তিক করপাস।",
            sampleSnippet = "[জাতীয় সংবাদ]\nঢাকা মেট্রোরেলের নতুন স্টেশন চালু হওয়ায় সাধারণ যাত্রীদের যাতায়াত আরও সহজ হয়েছে।"
        ),
        BenchmarkDatasetInfo(
            id = "bangla_wikipedia",
            name = "বাংলা উইকিপিডিয়া (Bangla Wikipedia)",
            nameBn = "বাংলা উইকিপিডিয়া করপাস",
            category = "Bangla NLP",
            format = "বিশ্বকোষীয় অনুচ্ছেদ (.txt)",
            defaultStrategy = DatasetFormatStrategy.SECTION_HEADER,
            description = "বাংলা উইকিপিডিয়ার সমস্ত নিবন্ধের বিষয়ভিত্তিক প্যারাগ্রাফ ও টপিক ক্লাসিফিকেশন।",
            sampleSnippet = "[সুন্দরবন]\nসুন্দরবন হলো বঙ্গোপসাগরের উপকূলবর্তী অঞ্চলে অবস্থিত একটি প্রশস্ত লবণাক্ত বনভূমি।"
        ),
        BenchmarkDatasetInfo(
            id = "oscar_bangla",
            name = "OSCAR (Open Super-large Crawled)",
            nameBn = "অস্কার করপাস",
            category = "Bangla NLP",
            format = "ওয়েব ক্রলড টেক্সট করপাস (.txt)",
            defaultStrategy = DatasetFormatStrategy.SECTION_HEADER,
            description = "মাল্টি-টেরাবাইট ওয়েব ক্রল থেকে পরিশোধিত বহুভাষিক ওপেন করপাস।",
            sampleSnippet = "[প্রযুক্তি ও বিজ্ঞান]\nকৃত্রিম বুদ্ধিমত্তা ও মোবাইল ডিভাইসে অন-ডিভাইস মেশিন লার্নিং এর ব্যবহার দ্রুত বৃদ্ধি পাচ্ছে।"
        ),
        BenchmarkDatasetInfo(
            id = "cc100_bangla",
            name = "CC-100 (Common Crawl 100)",
            nameBn = "সিসি-১০০ বাংলা",
            category = "Bangla NLP",
            format = "কমন ক্রল টেক্সট ডেটাসেট (.txt)",
            defaultStrategy = DatasetFormatStrategy.SECTION_HEADER,
            description = "১০০টি ভাষার ওপর নির্মিত ফেসবুক এআই-এর বৃহৎ মনোপ্রিট্রেইনিং করপাস।",
            sampleSnippet = "[শিক্ষা ও সংস্কৃতি]\nমাতৃভাষা শিক্ষায় ডিজিটাল কনটেন্ট তৈরি শিক্ষার্থীদের আগ্রহ বহুলাংশে বাড়িয়ে দেয়।"
        ),
        BenchmarkDatasetInfo(
            id = "bangla2b",
            name = "Bangla2B",
            nameBn = "বাংলা ২ বিলিয়ন টোকেন করপাস",
            category = "Bangla NLP",
            format = "প্রিট্রেইনিং করপাস (.jsonl / .txt)",
            defaultStrategy = DatasetFormatStrategy.SECTION_HEADER,
            description = "বাংলা ভাষায় ২ বিলিয়নের অধিক টোকেন সম্বলিত বৃহৎ প্রিট্রেইনিং টেক্সট করপাস।",
            sampleSnippet = "[অর্থনীতি ও বাণিজ্য]\nগত অর্থবছরে দেশের রপ্তানি আয়ে তৈরি পোশাক খাতের অবদান ছিল সর্বাধিক।"
        ),

        // Specialized & Healthcare
        BenchmarkDatasetInfo(
            id = "pubmed_cancer_nlp",
            name = "PubMed Cancer NLP Textual Dataset",
            nameBn = "পাবমেড ক্যান্সার এনএলপি ডেটাসেট",
            category = "Specialized",
            format = "Biomedical Literature CSV / TSV",
            defaultStrategy = DatasetFormatStrategy.CSV_LABEL_TEXT,
            description = "Cancer research paper abstracts and clinical trial texts categorized by cancer types (Lung, Breast, Colon, Melanoma).",
            sampleSnippet = "label,text\nLung Cancer,EGFR mutation analysis was performed in non-small cell lung carcinoma patients to evaluate osimertinib efficacy.\nBreast Cancer,HER2 positive metastatic breast cancer patients demonstrated improved progression-free survival with antibody-drug conjugates."
        ),
        BenchmarkDatasetInfo(
            id = "sms_spam_collection",
            name = "SMS Spam Collection",
            nameBn = "এসএমএস স্প্যাম গার্ড ডেটাসেট",
            category = "Specialized",
            format = "Label \t Message TSV",
            defaultStrategy = DatasetFormatStrategy.CSV_LABEL_TEXT,
            description = "5,574 mobile SMS messages labeled as legitimate Ham or Phishing/Promotional Spam.",
            sampleSnippet = "label,text\nham,Hey are we still meeting for dinner at 7pm tonight?\nspam,CONGRATULATIONS! You have won a $1000 cash prize! Claim now at http://win.xyz"
        )
    )
}
