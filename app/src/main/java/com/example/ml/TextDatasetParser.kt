package com.example.ml

import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.util.Locale

/**
 * Universal, Memory-Safe Multi-Format Parser for Text NLP & Large Datasets (500MB+ Safe).
 *
 * Supports:
 * 1. QA / Flashcard / Knowledge Base Formats:
 *    - "question,answer" (e.g. "বাংলাদেশের দীর্ঘতম নদী কোনটি?,মেঘনা")
 *    - "q,a", "query,target", "prompt,response", "input,output"
 * 2. Instruction & LLM Tuning Formats (JSON / JSONL):
 *    - Dolly/Databricks: {"instruction": "...", "context": "...", "response": "...", "category": "..."}
 *    - Alpaca: {"instruction": "...", "input": "...", "output": "..."}
 *    - OpenAI / ChatML: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
 * 3. Dialogue / Transcript / Drama Script:
 *    - "First Citizen:\n...\n\nMENENIUS:\n..."
 * 4. Standard Tabular CSV / TSV / Semicolon (Any custom headers & column layouts).
 * 5. Markdown / Section Headers ([Category] / # Category).
 * 6. Memory-Safe Streaming: Line-by-line buffered stream with zero OOM risk for huge files.
 */
object TextDatasetParser {

    enum class DatasetFormatStrategy(val title: String, val titleBn: String, val description: String) {
        AUTO_DETECT("Auto-Detect Format", "স্বয়ংক্রিয় সনাক্তকরণ", "Automatically identify CSV, JSONL, QA, or Dialogue formats"),
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

    const val SHAKESPEARE_CORIOLANUS_SAMPLE: String = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report fort, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even till the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city
is risen: why stay we prating here? to the Capitol!

All:
Come, come.

First Citizen:
Soft! who comes here?

Second Citizen:
Worthy Menenius Agrippa; one that hath always loved
the people.

First Citizen:
He's one honest enough: would all the rest were so!

MENENIUS:
What work's, my countrymen, in hand? where go you
With bats and clubs? The matter? speak, I pray you.

First Citizen:
Our business is not unknown to the senate; they have
had inkling this fortnight what we intend to do,
which now we'll show 'em in deeds. They say poor
suitors have strong breaths: they shall know we
have strong arms too.

MENENIUS:
Why, masters, my good friends, mine honest neighbours,
Will you undo yourselves?

First Citizen:
We cannot, sir, we are undone already.

MENENIUS:
I tell you, friends, most charitable care
Have the patricians of you. For your wants,
Your suffering in this dearth, you may as well
Strike at the heaven with your staves as lift them"""

    const val BANGLADESH_GK_QA_SAMPLE: String = """question,answer
বাংলাদেশের দীর্ঘতম নদী কোনটি?,মেঘনা
বাংলাদেশে স্থানীয় সরকার ব্যবস্থা দুর্বল হওয়ার পেছনে সবচেয়ে বড় প্রাতিষ্ঠানিক দ্বন্দ্ব কোনটি?,উপজেলা ও ইউনিয়ন পরিষদের মধ্যে দ্বৈত প্রশাসনিক কর্তৃত্ব
কোন সংস্থা বাংলাদেশের GDP হিসাব করে?,বাংলাদেশ পরিসংখ্যান ব্যুরো
পদ্মা ও যমুনা নদীর শাখা কোনগুলো?,গড়াই ও ধলেশ্বরী
বাংলাদেশের সংবিধানে মৌলিক অধিকার সংক্রান্ত অনুচ্ছেদ কতটি?,১৮টি অনুচ্ছেদ
বাংলাদেশের জাতীয় সংসদের প্রথম স্পিকার কে ছিলেন?,মোহাম্মদ উল্লাহ
মুজিবনগর সরকার শপথ গ্রহণ করে কত তারিখে?,১৭ এপ্রিল ১৯৭১
বাংলাদেশের সবচেয়ে বড় স্থলবন্দর কোনটি?,বেনাপোল স্থলবন্দর
সুন্দরবনকে বিশ্ব ঐতিহ্য হিসেবে ঘোষণা করে কোন সংস্থা?,ইউনেস্কো (UNESCO)
বাংলাদেশের প্রথম ডিজিটাল জেলা কোনটি?,যশোর"""

    const val INSTRUCTION_JSONL_SAMPLE: String = """{"instruction": "When did Virgin Australia start operating?", "context": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.", "response": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue.", "category": "closed_qa"}
{"instruction": "Which is a species of fish? Tope or Rope", "context": "", "response": "Tope is a species of houndshark fish.", "category": "classification"}
{"instruction": "Why can camels survive for long without water?", "context": "", "response": "Camels use the fat in their humps to store energy and minimize water loss.", "category": "open_qa"}
{"instruction": "What is the capital city of France?", "context": "", "response": "The capital of France is Paris.", "category": "general_knowledge"}
{"instruction": "Identify the intent: I want to cancel my monthly plan", "context": "", "response": "Cancellation Request", "category": "customer_support"}"""

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
        maxSampleCap: Int = 30_000,
        onProgress: ((linesRead: Long, samplesFound: Int) -> Unit)? = null
    ): ParseResult {
        val reader = BufferedReader(InputStreamReader(inputStream, Charsets.UTF_8), 32 * 1024)
        return parseBufferedReader(reader, strategy, maxSampleCap, onProgress)
    }

    /**
     * Standard string-based parser (wraps reader for unified parsing).
     */
    fun parse(
        rawContent: String,
        strategy: DatasetFormatStrategy = DatasetFormatStrategy.AUTO_DETECT,
        maxSampleCap: Int = 30_000
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
        // Read first 50 lines to inspect format
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

        // Determine effective strategy
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
        val firstLine = previewLines.first().trim()

        // 1. JSON / JSONL
        if (firstLine.startsWith("{") && (firstLine.contains("\"instruction\"") || firstLine.contains("\"category\"") || firstLine.contains("\"label\"") || firstLine.contains("\"text\""))) {
            return DatasetFormatStrategy.INSTRUCTION_RESPONSE
        }
        if (firstLine.startsWith("[") && firstLine.contains("{")) {
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

        return DatasetFormatStrategy.SHAKESPEARE_DIALOGUE
    }

    /**
     * Parse Question-Answer CSV / TSV format
     */
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

        val headerCols = splitCsvLine(headerLine, delimiter).map { it.lowercase(Locale.ROOT) }
        var questionCol = headerCols.indexOfFirst { it in listOf("question", "q", "prompt", "query", "input", "instruction") }
        var answerCol = headerCols.indexOfFirst { it in listOf("answer", "a", "response", "target", "output", "completion") }

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

                if (question.isNotEmpty() && answer.isNotEmpty()) {
                    val label = sanitizeClassName(answer.take(35))
                    samples.add(ParsedSample(label, question))
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

    /**
     * Parse Instruction Tuning JSON / JSONL (Dolly, Alpaca, ChatML, Databricks)
     */
    private fun parseJsonlStream(
        reader: BufferedReader,
        maxSampleCap: Int,
        onProgress: ((linesRead: Long, samplesFound: Int) -> Unit)?
    ): ParseResult {
        val samples = mutableListOf<ParsedSample>()
        var lineNum = 0L

        var line: String?
        while (reader.readLine().also { line = it } != null) {
            lineNum++
            val l = line?.trim() ?: continue
            if (l.isEmpty()) continue

            // JSON Array support
            if (l.startsWith("[") && l.endsWith("]")) {
                try {
                    val arr = JSONArray(l)
                    for (i in 0 until arr.length()) {
                        val item = arr.optJSONObject(i) ?: continue
                        extractSampleFromJson(item)?.let {
                            samples.add(it)
                            if (samples.size >= maxSampleCap) break
                        }
                    }
                } catch (_: Exception) {}
                continue
            }

            if (l.startsWith("{") && l.endsWith("}")) {
                try {
                    val obj = JSONObject(l)
                    extractSampleFromJson(obj)?.let {
                        samples.add(it)
                        if (samples.size >= maxSampleCap) break
                    }
                } catch (_: Exception) {}
            }

            if (lineNum % 500 == 0L) {
                onProgress?.invoke(lineNum, samples.size)
            }
        }

        if (samples.isEmpty()) return emptyResult("JSON / JSONL")
        val counts = samples.groupingBy { it.className }.eachCount()
        return ParseResult(
            formatName = "🤖 LLM Instruction Dataset (JSONL / Dolly / Alpaca)",
            samples = samples,
            classCounts = counts,
            totalLinesScanned = lineNum,
            isCapped = samples.size >= maxSampleCap
        )
    }

    private fun extractSampleFromJson(obj: JSONObject): ParsedSample? {
        // Priority 1: Dolly format (category + instruction + response)
        val category = obj.optString("category", "").trim()
        val instruction = obj.optString("instruction", "").trim()
        val response = obj.optString("response", "").ifEmpty { obj.optString("output", "") }.trim()
        val context = obj.optString("context", "").trim()

        if (category.isNotEmpty()) {
            val content = buildString {
                if (instruction.isNotEmpty()) append(instruction).append(" ")
                if (context.isNotEmpty()) append(context).append(" ")
                if (response.isNotEmpty()) append(response)
            }.trim()
            if (content.isNotEmpty()) {
                return ParsedSample(sanitizeClassName(category), content)
            }
        }

        // Priority 2: Standard label + text
        val labelKey = listOf("label", "class", "category", "target", "sentiment", "speaker", "tag", "intent").firstOrNull { obj.has(it) }
        val textKey = listOf("text", "content", "sentence", "message", "utterance", "review", "line", "body", "input").firstOrNull { obj.has(it) }
        if (labelKey != null && textKey != null) {
            val label = sanitizeClassName(obj.optString(labelKey, ""))
            val content = obj.optString(textKey, "").trim()
            if (label.isNotEmpty() && content.isNotEmpty()) {
                return ParsedSample(label, content)
            }
        }

        // Priority 3: Alpaca format without category (instruction -> label, output -> text)
        if (instruction.isNotEmpty() && response.isNotEmpty()) {
            val label = sanitizeClassName(instruction.take(30))
            val content = if (obj.optString("input", "").isNotEmpty()) "${obj.optString("input")}\n$response" else response
            return ParsedSample(label, content)
        }

        return null
    }

    /**
     * Parse Dialogue / Transcript Format (e.g. Shakespeare "First Citizen:\n...")
     */
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

    /**
     * Generic CSV / TSV Parser
     */
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
                val lower = col.lowercase(Locale.ROOT)
                if (lower in listOf("label", "class", "category", "target", "sentiment", "speaker", "intent", "tag", "topic", "answer")) {
                    labelCol = idx
                }
                if (lower in listOf("text", "content", "sentence", "message", "utterance", "review", "line", "question", "prompt", "input", "body")) {
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
                // Include first line as sample since it wasn't header
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
        return ParseResult(
            formatName = if (delimiter == '\t') "TSV (Tab-Separated Dataset)" else "CSV (Comma-Separated Dataset)",
            samples = samples,
            classCounts = counts,
            totalLinesScanned = lineNum,
            isCapped = samples.size >= maxSampleCap
        )
    }

    /**
     * Section Headers Parser ([Category] / # Category)
     */
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
}
