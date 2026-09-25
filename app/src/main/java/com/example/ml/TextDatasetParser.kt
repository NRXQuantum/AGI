package com.example.ml

import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale

/**
 * Universal Parser for text classification datasets.
 *
 * Supports:
 * 1. Dialogue / Speaker Colon Format (e.g. Shakespeare "First Citizen:\n...\n\nAll:\n...")
 * 2. Inline Colon / Hyphen Format (e.g. "Spam: Buy cheap pills now")
 * 3. CSV / TSV Format (e.g. "label,text" or "text,label")
 * 4. JSON / JSONL Format (e.g. [{"label": "...", "text": "..."}] or one json per line)
 * 5. Section Header Format (e.g. "[Positive]\nSample 1\nSample 2")
 */
object TextDatasetParser {

    data class ParsedSample(
        val className: String,
        val text: String
    )

    data class ParseResult(
        val formatName: String,
        val samples: List<ParsedSample>,
        val classCounts: Map<String, Int>,
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

    private val CLASS_PALETTE = listOf(
        "#3B82F6", "#10B981", "#F59E0B", "#EF4444",
        "#8B5CF6", "#EC4899", "#06B6D4", "#F97316",
        "#14B8A6", "#6366F1", "#84CC16", "#64748B"
    )

    fun getColorForIndex(index: Int): String {
        return CLASS_PALETTE[index % CLASS_PALETTE.size]
    }

    fun parse(rawContent: String): ParseResult {
        val trimmed = rawContent.trim()
        if (trimmed.isBlank()) {
            return ParseResult(
                formatName = "Empty",
                samples = emptyList(),
                classCounts = emptyMap(),
                errorMessage = "Content is empty."
            )
        }

        // 1. Try JSON or JSONL format
        if (trimmed.startsWith("[") || (trimmed.startsWith("{") && trimmed.contains("\""))) {
            val jsonRes = tryParseJsonOrJsonl(trimmed)
            if (jsonRes != null && jsonRes.samples.isNotEmpty()) {
                return jsonRes
            }
        }

        // 2. Try Dialogue / Transcript Colon Format (e.g. "First Citizen:\n...\n\nAll:\n...")
        val dialogueRes = tryParseDialogueFormat(trimmed)
        if (dialogueRes != null && dialogueRes.samples.isNotEmpty() && dialogueRes.classCounts.size >= 2) {
            return dialogueRes
        }

        // 3. Try CSV / TSV
        val csvRes = tryParseCsvOrTsv(trimmed)
        if (csvRes != null && csvRes.samples.isNotEmpty() && csvRes.classCounts.size >= 2) {
            return csvRes
        }

        // 4. Try Section Headers (e.g. "[Class Name]" or "### Class Name")
        val sectionRes = tryParseSectionHeaders(trimmed)
        if (sectionRes != null && sectionRes.samples.isNotEmpty() && sectionRes.classCounts.size >= 2) {
            return sectionRes
        }

        // 5. Try Inline Colon Format (e.g. "Label: Text content here")
        val inlineColonRes = tryParseInlineColon(trimmed)
        if (inlineColonRes != null && inlineColonRes.samples.isNotEmpty() && inlineColonRes.classCounts.size >= 2) {
            return inlineColonRes
        }

        // If dialogue parser found at least some samples
        if (dialogueRes != null && dialogueRes.samples.isNotEmpty()) {
            return dialogueRes
        }

        return ParseResult(
            formatName = "Unknown",
            samples = emptyList(),
            classCounts = emptyMap(),
            errorMessage = "Could not automatically identify classes and samples. Supported formats: Speaker colon blocks (Label:\\nText), CSV/TSV, JSON, or Section Headers."
        )
    }

    /**
     * Parses dialogue / transcript speaker format:
     * Label:
     * Text paragraph...
     *
     * Next Label:
     * Text paragraph...
     */
    private fun tryParseDialogueFormat(text: String): ParseResult? {
        val lines = text.lines()
        val speakerRegex = "^(?:\\[([A-Za-z0-9_\u0980-\u09FF\\s\\-\\.\\(\\)'\"]{1,45})\\]|([A-Za-z0-9_\u0980-\u09FF\\s\\-\\.\\(\\)'\"]{1,45}))\\s*[:\\-]\\s*(.*)$".toRegex()

        val samples = mutableListOf<ParsedSample>()
        var currentSpeaker: String? = null
        val currentTextBuilder = StringBuilder()

        for (line in lines) {
            val trimmedLine = line.trim()
            if (trimmedLine.isEmpty()) continue

            val match = speakerRegex.matchEntire(trimmedLine)
            if (match != null) {
                val rawCandidate = (match.groupValues[1].ifEmpty { match.groupValues[2] }).trim()
                val inlineText = match.groupValues[3].trim()

                // Check that candidate looks like a persona / speaker label, not normal narrative
                if (rawCandidate.length in 1..40 &&
                    !rawCandidate.contains("?") &&
                    !rawCandidate.contains("!") &&
                    !rawCandidate.contains(";") &&
                    rawCandidate.split(" ").size <= 5
                ) {
                    // Flush previous speaker's accumulated dialogue
                    if (currentSpeaker != null && currentTextBuilder.isNotBlank()) {
                        samples.add(ParsedSample(currentSpeaker, currentTextBuilder.toString().trim()))
                        currentTextBuilder.clear()
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
        }

        // Flush last sample
        if (currentSpeaker != null && currentTextBuilder.isNotBlank()) {
            samples.add(ParsedSample(currentSpeaker, currentTextBuilder.toString().trim()))
        }

        if (samples.isEmpty()) return null

        val counts = samples.groupingBy { it.className }.eachCount()
        return ParseResult(
            formatName = "🎭 Shakespeare / Drama Dialogue Transcript",
            samples = samples,
            classCounts = counts
        )
    }

    /**
     * Parses inline colon format:
     * Label: text here...
     * Label2: text here...
     */
    private fun tryParseInlineColon(text: String): ParseResult? {
        val lines = text.lines()
        val samples = mutableListOf<ParsedSample>()

        for (line in lines) {
            val trimmed = line.trim()
            if (trimmed.isEmpty()) continue

            val colonIdx = trimmed.indexOf(':')
            if (colonIdx in 1..35) {
                val label = sanitizeClassName(trimmed.substring(0, colonIdx).trim())
                val content = trimmed.substring(colonIdx + 1).trim()
                if (content.isNotEmpty()) {
                    samples.add(ParsedSample(label, content))
                }
            }
        }

        if (samples.size < 2) return null
        val counts = samples.groupingBy { it.className }.eachCount()
        return ParseResult(
            formatName = "Inline Labelled Lines (Label: Text)",
            samples = samples,
            classCounts = counts
        )
    }

    /**
     * Parses CSV or TSV files.
     */
    private fun tryParseCsvOrTsv(text: String): ParseResult? {
        val lines = text.lines().map { it.trim() }.filter { it.isNotEmpty() }
        if (lines.size < 2) return null

        val delimiter = if (lines[0].contains("\t")) '\t' else if (lines[0].contains(";")) ';' else ','
        val samples = mutableListOf<ParsedSample>()

        var labelCol = -1
        var textCol = -1

        // Check header line
        val firstCols = splitCsvLine(lines[0], delimiter)
        if (firstCols.size >= 2) {
            for ((idx, col) in firstCols.withIndex()) {
                val lower = col.lowercase(Locale.ROOT)
                if (lower in listOf("label", "class", "category", "target", "sentiment", "speaker")) {
                    labelCol = idx
                }
                if (lower in listOf("text", "content", "sentence", "message", "utterance", "review", "line")) {
                    textCol = idx
                }
            }
        }

        val startIdx = if (labelCol != -1 && textCol != -1) 1 else 0

        // If not found from header, heuristically decide by length
        if (labelCol == -1 || textCol == -1) {
            val sampleCols = splitCsvLine(lines[startIdx], delimiter)
            if (sampleCols.size < 2) return null
            if (sampleCols[0].length <= sampleCols[1].length) {
                labelCol = 0
                textCol = 1
            } else {
                labelCol = 1
                textCol = 0
            }
        }

        for (i in startIdx until lines.size) {
            val cols = splitCsvLine(lines[i], delimiter)
            if (cols.size > maxOf(labelCol, textCol)) {
                val label = sanitizeClassName(cols[labelCol].trim())
                val content = cols[textCol].trim()
                if (label.isNotEmpty() && content.isNotEmpty()) {
                    samples.add(ParsedSample(label, content))
                }
            }
        }

        if (samples.size < 2) return null
        val counts = samples.groupingBy { it.className }.eachCount()
        return ParseResult(
            formatName = if (delimiter == '\t') "TSV (Tab-Separated)" else "CSV (Comma-Separated)",
            samples = samples,
            classCounts = counts
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

    /**
     * Parses Section Header format:
     * [Positive]
     * Great app!
     * Love this!
     *
     * [Negative]
     * Awful crash
     */
    private fun tryParseSectionHeaders(text: String): ParseResult? {
        val lines = text.lines()
        val samples = mutableListOf<ParsedSample>()
        var currentSection: String? = null

        val sectionRegex = "^(?:\\[(.*)\\]|#+\\s+(.*)|===+\\s*(.*?)\\s*===+)$".toRegex()

        for (line in lines) {
            val trimmed = line.trim()
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
            }
        }

        if (samples.size < 2) return null
        val counts = samples.groupingBy { it.className }.eachCount()
        return ParseResult(
            formatName = "Section Headers ([Category] / # Category)",
            samples = samples,
            classCounts = counts
        )
    }

    /**
     * Parses JSON Array or JSONL.
     */
    private fun tryParseJsonOrJsonl(text: String): ParseResult? {
        val samples = mutableListOf<ParsedSample>()

        fun extractSample(obj: JSONObject): ParsedSample? {
            val labelKey = listOf("label", "class", "category", "target", "sentiment", "speaker", "tag").firstOrNull { obj.has(it) }
            val textKey = listOf("text", "content", "sentence", "message", "utterance", "review", "line", "body").firstOrNull { obj.has(it) }
            if (labelKey != null && textKey != null) {
                val label = sanitizeClassName(obj.optString(labelKey, ""))
                val content = obj.optString(textKey, "").trim()
                if (label.isNotEmpty() && content.isNotEmpty()) {
                    return ParsedSample(label, content)
                }
            }
            return null
        }

        // Try JSON Array
        if (text.startsWith("[")) {
            try {
                val arr = JSONArray(text)
                for (i in 0 until arr.length()) {
                    val item = arr.optJSONObject(i) ?: continue
                    extractSample(item)?.let { samples.add(it) }
                }
            } catch (_: Exception) {}
        }

        // Try JSONL
        if (samples.isEmpty()) {
            val lines = text.lines()
            for (line in lines) {
                val trimmed = line.trim()
                if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
                    try {
                        val obj = JSONObject(trimmed)
                        extractSample(obj)?.let { samples.add(it) }
                    } catch (_: Exception) {}
                }
            }
        }

        if (samples.isEmpty()) return null
        val counts = samples.groupingBy { it.className }.eachCount()
        return ParseResult(
            formatName = if (text.startsWith("[")) "JSON Array" else "JSONL (Lines)",
            samples = samples,
            classCounts = counts
        )
    }

    private fun sanitizeClassName(name: String): String {
        return name
            .replace("[^a-zA-Z0-9\\s\\-_\u0980-\u09FF]".toRegex(), "")
            .trim()
            .take(30)
            .ifEmpty { "Class" }
    }
}
