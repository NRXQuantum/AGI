package com.example.util

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.InputStream
import kotlin.math.max

object ImageUtils {

    /**
     * Reads EXIF orientation from an InputStream.
     */
    fun getExifOrientation(inputStream: InputStream): Int {
        return try {
            val exif = ExifInterface(inputStream)
            exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } catch (_: Throwable) {
            ExifInterface.ORIENTATION_NORMAL
        }
    }

    /**
     * Reads EXIF orientation from a File.
     */
    fun getExifOrientation(file: File): Int {
        return try {
            val exif = ExifInterface(file.absolutePath)
            exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } catch (_: Throwable) {
            ExifInterface.ORIENTATION_NORMAL
        }
    }

    /**
     * Rotates or flips a Bitmap according to its EXIF orientation tag.
     */
    fun applyExifOrientation(bitmap: Bitmap, orientation: Int): Bitmap {
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postRotate(90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postRotate(270f)
                matrix.postScale(-1f, 1f)
            }
            else -> return bitmap
        }

        return try {
            val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (rotated != bitmap) {
                bitmap.recycle()
            }
            rotated
        } catch (_: Throwable) {
            bitmap
        }
    }

    /**
     * Rotates a Bitmap by arbitrary degrees (e.g. 90, 180, 270).
     */
    fun rotateBitmapByDegrees(bitmap: Bitmap, degrees: Float): Bitmap {
        if (degrees % 360f == 0f) return bitmap
        val matrix = Matrix().apply { postRotate(degrees) }
        return try {
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } catch (_: Throwable) {
            bitmap
        }
    }

    /**
     * Decodes and automatically corrects the orientation of an image from a Content Uri.
     */
    fun decodeOrientedBitmap(context: Context, uri: Uri, maxDim: Int = 1280): Bitmap? {
        return decodeOrientedBitmapFromStream(
            inputStreamProvider = { context.contentResolver.openInputStream(uri) },
            maxDim = maxDim
        )
    }

    /**
     * Decodes and automatically corrects the orientation of an image from a local File.
     */
    fun decodeOrientedBitmap(file: File, maxDim: Int = 1280): Bitmap? {
        if (!file.exists()) return null
        val orientation = getExifOrientation(file)

        val boundsOpts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, boundsOpts)
        val largest = max(boundsOpts.outWidth, boundsOpts.outHeight)

        var sampleSize = 1
        if (maxDim > 0) {
            while (largest / (sampleSize * 2) >= maxDim) {
                sampleSize *= 2
            }
        }

        val decodeOpts = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }

        val rawBmp = try {
            BitmapFactory.decodeFile(file.absolutePath, decodeOpts)
        } catch (_: Throwable) {
            null
        } ?: return null

        return applyExifOrientation(rawBmp, orientation)
    }

    /**
     * Decodes and corrects orientation from a stream provider (which can provide fresh streams for bounds, EXIF, and decode).
     */
    fun decodeOrientedBitmapFromStream(
        inputStreamProvider: () -> InputStream?,
        maxDim: Int = 1280
    ): Bitmap? {
        val orientation = try {
            inputStreamProvider()?.use { getExifOrientation(it) } ?: ExifInterface.ORIENTATION_NORMAL
        } catch (_: Throwable) {
            ExifInterface.ORIENTATION_NORMAL
        }

        val boundsOpts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        try {
            inputStreamProvider()?.use {
                BitmapFactory.decodeStream(it, null, boundsOpts)
            }
        } catch (_: Throwable) {
            return null
        }

        val largest = max(boundsOpts.outWidth, boundsOpts.outHeight)
        var sampleSize = 1
        if (maxDim > 0) {
            while (largest / (sampleSize * 2) >= maxDim) {
                sampleSize *= 2
            }
        }

        val decodeOpts = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }

        val rawBmp = try {
            inputStreamProvider()?.use {
                BitmapFactory.decodeStream(it, null, decodeOpts)
            }
        } catch (_: Throwable) {
            null
        } ?: return null

        return applyExifOrientation(rawBmp, orientation)
    }
}
