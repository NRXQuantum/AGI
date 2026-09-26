package com.example.ui.components

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.widget.Toast
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Code
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.TableChart
import androidx.compose.material.icons.outlined.FileDownload
import androidx.compose.material.icons.outlined.TableRows
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.*
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * High-Quality Markdown, Code Block & Table Renderer for AI Chat Dialogue.
 * Supports:
 * 1. Markdown Tables (| Col1 | Col2 | with horizontal scroll, zebra rows, TSV/CSV copy).
 * 2. Multi-language Code Blocks (```scala, ```python, ```json, etc. with syntax highlighting, line numbers & copy code).
 * 3. Bullet & Numbered lists.
 * 4. Headings & Blockquotes.
 * 5. Inline formatting (bold, italic, inline `code`).
 */
@Composable
fun RichChatContent(
    text: String,
    modifier: Modifier = Modifier,
    isUser: Boolean = false,
    accentColor: Color = MaterialTheme.colorScheme.primary,
    onCopySnippet: ((String) -> Unit)? = null
) {
    val context = LocalContext.current
    val blocks = remember(text) { parseChatMarkdownBlocks(text) }

    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        blocks.forEach { block ->
            when (block) {
                is ChatContentBlock.Code -> {
                    ChatCodeBlockCard(
                        language = block.language,
                        code = block.code,
                        accentColor = accentColor,
                        onCopy = { codeToCopy ->
                            copyToClipboard(context, "Code Snippet", codeToCopy)
                            onCopySnippet?.invoke(codeToCopy)
                        }
                    )
                }

                is ChatContentBlock.Table -> {
                    ChatTableCard(
                        headers = block.headers,
                        rows = block.rows,
                        accentColor = accentColor,
                        onCopyTsv = {
                            val tableTsv = buildString {
                                append(block.headers.joinToString("\t")).append("\n")
                                block.rows.forEach { r -> append(r.joinToString("\t")).append("\n") }
                            }
                            copyToClipboard(context, "Excel/TSV Table", tableTsv)
                        },
                        onCopyCsv = {
                            val tableCsv = buildString {
                                append(block.headers.joinToString(",") { "\"${it.replace("\"", "\"\"")}\"" }).append("\n")
                                block.rows.forEach { r ->
                                    append(r.joinToString(",") { "\"${it.replace("\"", "\"\"")}\"" }).append("\n")
                                }
                            }
                            copyToClipboard(context, "CSV Table", tableCsv)
                        }
                    )
                }

                is ChatContentBlock.Heading -> {
                    val fontSize = when (block.level) {
                        1 -> 18.sp
                        2 -> 16.sp
                        else -> 14.5.sp
                    }
                    Text(
                        text = buildInlineAnnotatedString(block.text, isUser, accentColor),
                        style = MaterialTheme.typography.titleMedium.copy(
                            fontSize = fontSize,
                            fontWeight = FontWeight.Bold
                        ),
                        color = if (isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurface,
                        modifier = Modifier.padding(top = 4.dp, bottom = 2.dp)
                    )
                }

                is ChatContentBlock.NumberedList -> {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(start = 4.dp, top = 2.dp, bottom = 2.dp),
                        verticalAlignment = Alignment.Top
                    ) {
                        Surface(
                            shape = CircleShape,
                            color = accentColor.copy(alpha = 0.15f),
                            modifier = Modifier
                                .size(20.dp)
                                .padding(top = 1.dp)
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Text(
                                    text = block.number,
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        fontSize = 10.sp,
                                        fontWeight = FontWeight.Bold
                                    ),
                                    color = accentColor
                                )
                            }
                        }
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = buildInlineAnnotatedString(block.text, isUser, accentColor),
                            style = MaterialTheme.typography.bodyMedium.copy(fontSize = 13.5.sp, lineHeight = 19.sp),
                            color = if (isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.weight(1f)
                        )
                    }
                }

                is ChatContentBlock.BulletList -> {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(start = 6.dp, top = 2.dp, bottom = 2.dp),
                        verticalAlignment = Alignment.Top
                    ) {
                        Surface(
                            shape = CircleShape,
                            color = accentColor,
                            modifier = Modifier
                                .size(6.dp)
                                .padding(top = 7.dp)
                        ) {}
                        Spacer(modifier = Modifier.width(10.dp))
                        Text(
                            text = buildInlineAnnotatedString(block.text, isUser, accentColor),
                            style = MaterialTheme.typography.bodyMedium.copy(fontSize = 13.5.sp, lineHeight = 19.sp),
                            color = if (isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.weight(1f)
                        )
                    }
                }

                is ChatContentBlock.Blockquote -> {
                    Surface(
                        shape = RoundedCornerShape(topStart = 2.dp, bottomStart = 2.dp, topEnd = 8.dp, bottomEnd = 8.dp),
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.3f)),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(modifier = Modifier.padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
                            Box(
                                modifier = Modifier
                                    .width(3.5.dp)
                                    .height(28.dp)
                                    .background(accentColor, RoundedCornerShape(2.dp))
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = buildInlineAnnotatedString(block.text, isUser, accentColor),
                                style = MaterialTheme.typography.bodyMedium.copy(
                                    fontSize = 13.sp,
                                    fontStyle = FontStyle.Italic,
                                    lineHeight = 18.sp
                                ),
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }

                is ChatContentBlock.Paragraph -> {
                    Text(
                        text = buildInlineAnnotatedString(block.text, isUser, accentColor),
                        style = MaterialTheme.typography.bodyMedium.copy(fontSize = 13.5.sp, lineHeight = 20.sp),
                        color = if (isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

/**
 * Beautiful Dark Theme Code Block with Syntax Highlighting, Line Numbers & One-Click Copy
 */
@Composable
fun ChatCodeBlockCard(
    language: String,
    code: String,
    accentColor: Color,
    onCopy: (String) -> Unit
) {
    val coroutineScope = rememberCoroutineScope()
    var isCopied by remember { mutableStateOf(false) }

    val displayLang = remember(language) {
        if (language.isBlank()) "CODE" else language.uppercase(Locale.ROOT)
    }

    val codeLines = remember(code) { code.lines() }

    Surface(
        shape = RoundedCornerShape(10.dp),
        color = Color(0xFF1E1E2E), // Catppuccin / VS Code Dark background
        border = BorderStroke(1.dp, Color(0xFF313244)),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column {
            // Header Bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF181825))
                    .padding(horizontal = 10.dp, vertical = 6.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Code,
                        contentDescription = null,
                        tint = Color(0xFF89B4FA),
                        modifier = Modifier.size(14.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = displayLang,
                        style = MaterialTheme.typography.labelSmall.copy(
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold,
                            fontSize = 11.sp
                        ),
                        color = Color(0xFFCDD6F4)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "${codeLines.size} lines",
                        style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp),
                        color = Color(0xFF6C7086)
                    )
                }

                Surface(
                    onClick = {
                        onCopy(code)
                        isCopied = true
                        coroutineScope.launch {
                            delay(2000)
                            isCopied = false
                        }
                    },
                    shape = RoundedCornerShape(6.dp),
                    color = if (isCopied) Color(0xFF10B981).copy(alpha = 0.25f) else Color(0xFF313244).copy(alpha = 0.7f),
                    modifier = Modifier.height(26.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Icon(
                            imageVector = if (isCopied) Icons.Default.Check else Icons.Default.ContentCopy,
                            contentDescription = "Copy Code",
                            tint = if (isCopied) Color(0xFFA6E3A1) else Color(0xFFBAC2DE),
                            modifier = Modifier.size(12.dp)
                        )
                        Text(
                            text = if (isCopied) "Copied!" else "Copy",
                            style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.5.sp, fontWeight = FontWeight.Medium),
                            color = if (isCopied) Color(0xFFA6E3A1) else Color(0xFFBAC2DE)
                        )
                    }
                }
            }

            // Code Content with Line Numbers & Syntax Styling
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
                    .padding(vertical = 8.dp, horizontal = 10.dp)
            ) {
                Row {
                    // Line numbers column
                    Column(
                        horizontalAlignment = Alignment.End,
                        modifier = Modifier.padding(end = 12.dp)
                    ) {
                        for (idx in 1..codeLines.size) {
                            Text(
                                text = "$idx",
                                style = MaterialTheme.typography.bodySmall.copy(
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 11.5.sp,
                                    lineHeight = 18.sp
                                ),
                                color = Color(0xFF585B70)
                            )
                        }
                    }

                    // Vertical divider line in code block
                    Box(
                        modifier = Modifier
                            .width(1.dp)
                            .height((codeLines.size * 18).dp)
                            .background(Color(0xFF313244))
                    )

                    Spacer(modifier = Modifier.width(10.dp))

                    // Syntax-highlighted code lines
                    Column {
                        codeLines.forEach { line ->
                            Text(
                                text = buildSyntaxAnnotatedString(line, language),
                                style = MaterialTheme.typography.bodySmall.copy(
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 11.5.sp,
                                    lineHeight = 18.sp
                                )
                            )
                        }
                    }
                }
            }
        }
    }
}

/**
 * Modern Markdown Table Card with Responsive Horizontal Scroll & Header Highlighting
 */
@Composable
fun ChatTableCard(
    headers: List<String>,
    rows: List<List<String>>,
    accentColor: Color,
    onCopyTsv: () -> Unit,
    onCopyCsv: () -> Unit
) {
    val coroutineScope = rememberCoroutineScope()
    var isTsvCopied by remember { mutableStateOf(false) }

    Surface(
        shape = RoundedCornerShape(12.dp),
        color = MaterialTheme.colorScheme.surface,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.7f)),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column {
            // Table Top Header Banner
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        Brush.horizontalGradient(
                            listOf(
                                accentColor.copy(alpha = 0.12f),
                                MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.08f)
                            )
                        )
                    )
                    .padding(horizontal = 10.dp, vertical = 6.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.TableChart,
                        contentDescription = null,
                        tint = accentColor,
                        modifier = Modifier.size(15.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "Table Grid (${rows.size} rows • ${headers.size} cols)",
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold, fontSize = 11.sp),
                        color = accentColor
                    )
                }

                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    Surface(
                        onClick = {
                            onCopyTsv()
                            isTsvCopied = true
                            coroutineScope.launch {
                                delay(2000)
                                isTsvCopied = false
                            }
                        },
                        shape = RoundedCornerShape(6.dp),
                        color = if (isTsvCopied) Color(0xFF10B981).copy(alpha = 0.2f) else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.7f),
                        modifier = Modifier.height(24.dp)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Icon(
                                imageVector = if (isTsvCopied) Icons.Default.Check else Icons.Default.ContentCopy,
                                contentDescription = "Copy Table",
                                tint = if (isTsvCopied) Color(0xFF10B981) else MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.size(11.dp)
                            )
                            Text(
                                text = if (isTsvCopied) "Copied!" else "Copy (Excel)",
                                style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp),
                                color = if (isTsvCopied) Color(0xFF10B981) else MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }

            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f), thickness = 1.dp)

            // Scrollable Table Grid
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
            ) {
                Column(modifier = Modifier.padding(8.dp)) {
                    // Header Row
                    Row(
                        modifier = Modifier
                            .background(accentColor.copy(alpha = 0.14f), RoundedCornerShape(6.dp))
                            .padding(vertical = 6.dp, horizontal = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        headers.forEach { headerText ->
                            Text(
                                text = headerText.trim(),
                                style = MaterialTheme.typography.labelMedium.copy(
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 12.sp
                                ),
                                color = accentColor,
                                modifier = Modifier
                                    .widthIn(min = 120.dp, max = 240.dp)
                                    .padding(horizontal = 8.dp)
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    // Data Rows (Zebra striped)
                    rows.forEachIndexed { idx, rowCells ->
                        val rowBg = if (idx % 2 == 1) MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f) else Color.Transparent
                        Row(
                            modifier = Modifier
                                .background(rowBg, RoundedCornerShape(4.dp))
                                .padding(vertical = 6.dp, horizontal = 4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            headers.indices.forEach { colIdx ->
                                val cellText = rowCells.getOrNull(colIdx) ?: ""
                                Text(
                                    text = cellText.trim(),
                                    style = MaterialTheme.typography.bodySmall.copy(
                                        fontSize = 12.sp,
                                        lineHeight = 16.sp
                                    ),
                                    color = MaterialTheme.colorScheme.onSurface,
                                    modifier = Modifier
                                        .widthIn(min = 120.dp, max = 240.dp)
                                        .padding(horizontal = 8.dp)
                                )
                            }
                        }
                        if (idx < rows.size - 1) {
                            HorizontalDivider(
                                color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.25f),
                                thickness = 0.5.dp
                            )
                        }
                    }
                }
            }
        }
    }
}

/**
 * Parses raw text into structured blocks (Code, Table, Lists, Headings, Quotes, Paragraphs)
 */
sealed class ChatContentBlock {
    data class Code(val language: String, val code: String) : ChatContentBlock()
    data class Table(val headers: List<String>, val rows: List<List<String>>) : ChatContentBlock()
    data class Heading(val level: Int, val text: String) : ChatContentBlock()
    data class NumberedList(val number: String, val text: String) : ChatContentBlock()
    data class BulletList(val text: String) : ChatContentBlock()
    data class Blockquote(val text: String) : ChatContentBlock()
    data class Paragraph(val text: String) : ChatContentBlock()
}

fun parseChatMarkdownBlocks(rawText: String): List<ChatContentBlock> {
    val blocks = mutableListOf<ChatContentBlock>()
    val lines = rawText.lines()
    var i = 0

    while (i < lines.size) {
        val line = lines[i]
        val trimmed = line.trim()

        // 1. Code Block (```lang ... ```)
        if (trimmed.startsWith("```")) {
            val lang = trimmed.removePrefix("```").trim()
            val codeBuilder = StringBuilder()
            i++
            while (i < lines.size && !lines[i].trim().startsWith("```")) {
                codeBuilder.append(lines[i]).append("\n")
                i++
            }
            if (i < lines.size && lines[i].trim().startsWith("```")) {
                i++
            }
            blocks.add(ChatContentBlock.Code(lang, codeBuilder.toString().trimEnd()))
            continue
        }

        // 2. Markdown Table (| col1 | col2 | ... |)
        if (trimmed.startsWith("|") && trimmed.count { it == '|' } >= 2) {
            val tableLines = mutableListOf<String>()
            while (i < lines.size && lines[i].trim().startsWith("|") && lines[i].trim().count { it == '|' } >= 2) {
                tableLines.add(lines[i].trim())
                i++
            }
            if (tableLines.size >= 2) {
                val headerCells = splitTableLine(tableLines[0])
                val startIndex = if (tableLines.size > 1 && isTableDivider(tableLines[1])) 2 else 1
                val dataRows = mutableListOf<List<String>>()
                for (r in startIndex until tableLines.size) {
                    val rowCells = splitTableLine(tableLines[r])
                    if (rowCells.isNotEmpty()) dataRows.add(rowCells)
                }
                if (headerCells.isNotEmpty()) {
                    blocks.add(ChatContentBlock.Table(headerCells, dataRows))
                    continue
                }
            }
        }

        // 3. Headings (#, ##, ###)
        if (trimmed.startsWith("### ")) {
            blocks.add(ChatContentBlock.Heading(3, trimmed.removePrefix("### ")))
            i++
            continue
        }
        if (trimmed.startsWith("## ")) {
            blocks.add(ChatContentBlock.Heading(2, trimmed.removePrefix("## ")))
            i++
            continue
        }
        if (trimmed.startsWith("# ")) {
            blocks.add(ChatContentBlock.Heading(1, trimmed.removePrefix("# ")))
            i++
            continue
        }

        // 4. Blockquote (> quote)
        if (trimmed.startsWith("> ")) {
            blocks.add(ChatContentBlock.Blockquote(trimmed.removePrefix("> ")))
            i++
            continue
        }

        // 5. Numbered List (e.g. "1. Step", "2. Import")
        val numMatch = Regex("^([0-9]+)[.)]\\s+(.+)").find(trimmed)
        if (numMatch != null) {
            val num = numMatch.groupValues[1]
            val content = numMatch.groupValues[2]
            blocks.add(ChatContentBlock.NumberedList(num, content))
            i++
            continue
        }

        // 6. Bullet List (e.g. "- item", "* item", "• item")
        if (trimmed.startsWith("- ") || trimmed.startsWith("* ") || trimmed.startsWith("• ")) {
            val bulletContent = trimmed.substring(2).trim()
            blocks.add(ChatContentBlock.BulletList(bulletContent))
            i++
            continue
        }

        // 7. Regular Paragraph
        if (trimmed.isNotEmpty()) {
            blocks.add(ChatContentBlock.Paragraph(line))
        }
        i++
    }

    return if (blocks.isEmpty()) listOf(ChatContentBlock.Paragraph(rawText)) else blocks
}

private fun splitTableLine(line: String): List<String> {
    val clean = line.trim().removeSurrounding("|", "|")
    return clean.split("|").map { it.trim() }
}

private fun isTableDivider(line: String): Boolean {
    val clean = line.trim().replace("|", "").replace("-", "").replace(":", "").trim()
    return clean.isEmpty()
}

/**
 * Builds AnnotatedString supporting inline **bold**, *italic*, and `code` styling.
 */
private fun buildInlineAnnotatedString(
    raw: String,
    isUser: Boolean,
    accentColor: Color
): AnnotatedString {
    return buildAnnotatedString {
        var cursor = 0
        val len = raw.length

        while (cursor < len) {
            // Inline Code `code`
            if (raw[cursor] == '`') {
                val endIdx = raw.indexOf('`', cursor + 1)
                if (endIdx != -1) {
                    val codeSnippet = raw.substring(cursor + 1, endIdx)
                    pushStyle(
                        SpanStyle(
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.SemiBold,
                            color = if (isUser) Color(0xFFFFD54F) else Color(0xFF9333EA),
                            background = if (isUser) Color.Black.copy(alpha = 0.2f) else accentColor.copy(alpha = 0.12f)
                        )
                    )
                    append(" $codeSnippet ")
                    pop()
                    cursor = endIdx + 1
                    continue
                }
            }

            // Bold **text**
            if (cursor + 1 < len && raw[cursor] == '*' && raw[cursor + 1] == '*') {
                val endIdx = raw.indexOf("**", cursor + 2)
                if (endIdx != -1) {
                    val boldText = raw.substring(cursor + 2, endIdx)
                    pushStyle(SpanStyle(fontWeight = FontWeight.Bold))
                    append(boldText)
                    pop()
                    cursor = endIdx + 2
                    continue
                }
            }

            // Italic *text*
            if (raw[cursor] == '*') {
                val endIdx = raw.indexOf('*', cursor + 1)
                if (endIdx != -1) {
                    val italicText = raw.substring(cursor + 1, endIdx)
                    pushStyle(SpanStyle(fontStyle = FontStyle.Italic))
                    append(italicText)
                    pop()
                    cursor = endIdx + 1
                    continue
                }
            }

            append(raw[cursor])
            cursor++
        }
    }
}

/**
 * Syntax highlighting for common programming languages (Scala, Python, Kotlin, Java, JS, JSON, SQL, Bash)
 */
private fun buildSyntaxAnnotatedString(line: String, language: String): AnnotatedString {
    val keywords = when (language.lowercase(Locale.ROOT)) {
        "scala" -> setOf("val", "var", "def", "class", "object", "trait", "import", "package", "new", "println", "if", "else", "match", "case", "for", "while", "return", "true", "false", "null", "type", "override", "extends", "with")
        "python", "py" -> setOf("def", "class", "import", "from", "as", "return", "if", "elif", "else", "for", "while", "in", "with", "try", "except", "finally", "lambda", "yield", "pass", "True", "False", "None", "print")
        "kotlin", "kt" -> setOf("val", "var", "fun", "class", "object", "interface", "import", "package", "return", "if", "else", "when", "for", "while", "null", "true", "false", "data", "sealed", "suspend", "companion")
        "java" -> setOf("public", "private", "protected", "class", "interface", "void", "static", "final", "return", "if", "else", "for", "while", "new", "import", "package", "try", "catch", "throws", "null", "true", "false")
        "sql" -> setOf("select", "from", "where", "insert", "into", "update", "delete", "join", "inner", "left", "right", "group", "by", "order", "table", "create", "alter", "drop")
        "json" -> setOf("true", "false", "null")
        else -> setOf("val", "var", "def", "fun", "class", "import", "return", "if", "else", "new", "println", "print", "true", "false", "null")
    }

    return buildAnnotatedString {
        val trimmed = line.trimStart()
        // Check for comment
        if (trimmed.startsWith("//") || trimmed.startsWith("#") || trimmed.startsWith("--")) {
            pushStyle(SpanStyle(color = Color(0xFF6C7086), fontStyle = FontStyle.Italic))
            append(line)
            pop()
            return@buildAnnotatedString
        }

        // Tokenize line words, strings, and operators
        var cursor = 0
        val len = line.length
        while (cursor < len) {
            val char = line[cursor]

            // String literal "..."
            if (char == '"') {
                val endQuote = line.indexOf('"', cursor + 1)
                if (endQuote != -1) {
                    val strLit = line.substring(cursor, endQuote + 1)
                    pushStyle(SpanStyle(color = Color(0xFFA6E3A1))) // Green string
                    append(strLit)
                    pop()
                    cursor = endQuote + 1
                    continue
                }
            }

            // Word token
            if (char.isLetterOrDigit() || char == '_') {
                val start = cursor
                while (cursor < len && (line[cursor].isLetterOrDigit() || line[cursor] == '_')) {
                    cursor++
                }
                val word = line.substring(start, cursor)
                if (word in keywords || word.lowercase(Locale.ROOT) in keywords) {
                    pushStyle(SpanStyle(color = Color(0xFFCBA6F7), fontWeight = FontWeight.Bold)) // Purple keyword
                    append(word)
                    pop()
                } else if (word.all { it.isDigit() }) {
                    pushStyle(SpanStyle(color = Color(0xFFFAB387))) // Orange number
                    append(word)
                    pop()
                } else if (word.first().isUpperCase()) {
                    pushStyle(SpanStyle(color = Color(0xFF89DCEB))) // Cyan Type/Class
                    append(word)
                    pop()
                } else {
                    pushStyle(SpanStyle(color = Color(0xFFCDD6F4))) // Base text
                    append(word)
                    pop()
                }
                continue
            }

            // Punctuation / Operator
            pushStyle(SpanStyle(color = Color(0xFF89B4FA)))
            append(char)
            pop()
            cursor++
        }
    }
}

private fun copyToClipboard(context: Context, label: String, text: String) {
    try {
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText(label, text)
        clipboard.setPrimaryClip(clip)
        Toast.makeText(context, "$label copied to clipboard!", Toast.LENGTH_SHORT).show()
    } catch (_: Exception) {}
}
