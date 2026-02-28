// SPDX-License-Identifier: GPL-3.0-only

package helium314.keyboard.latin;

import android.Manifest;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Handler;
import android.os.Looper;
import android.view.WindowManager;
import android.view.inputmethod.InputConnection;

import androidx.core.content.ContextCompat;

import helium314.keyboard.latin.settings.Settings;
import helium314.keyboard.latin.utils.DeviceProtectedUtils;
import helium314.keyboard.latin.utils.Log;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Handles voice input via a self-hosted Whisper-compatible ASR server.
 * Records audio from the mic, sends WAV to the configured endpoint,
 * and inserts the transcribed text via InputConnection.
 */
public class WhisperVoiceInput {
    private static final String TAG = "WhisperVoiceInput";
    private static final int SAMPLE_RATE = 16000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;

    public enum State { IDLE, RECORDING, PROCESSING, FAILED }

    public interface Callback {
        void onStateChanged(State state);
        void onError(String message);
    }

    private final LatinIME mLatinIME;
    private final Handler mHandler = new Handler(Looper.getMainLooper());
    private final ExecutorService mExecutor = Executors.newSingleThreadExecutor();

    private AudioRecord mAudioRecord;
    private ByteArrayOutputStream mAudioBuffer;
    private volatile State mState = State.IDLE;
    private Callback mCallback;
    private byte[] mLastWavData;  // Retained on failure for retry
    private long mLastFailedTapTime = 0;  // For double-tap-to-cancel

    public WhisperVoiceInput(LatinIME latinIME) {
        mLatinIME = latinIME;
    }

    public void setCallback(Callback callback) {
        mCallback = callback;
    }

    public State getState() {
        return mState;
    }

    /**
     * Toggle recording: if idle, start recording. If recording, stop and transcribe.
     */
    public void toggle() {
        if (mState == State.IDLE) {
            startRecording();
        } else if (mState == State.RECORDING) {
            stopRecordingAndTranscribe();
        } else if (mState == State.FAILED && mLastWavData != null) {
            // Single tap = retry, double tap (within 500ms) = cancel
            long now = System.currentTimeMillis();
            if (now - mLastFailedTapTime < 500) {
                cancel();
                mLastFailedTapTime = 0;
            } else {
                mLastFailedTapTime = now;
                retryTranscription();
            }
        } else if (mState == State.PROCESSING) {
            // Tap while processing = cancel
            cancel();
        }
    }

    /** Discard saved audio and return to idle (e.g. user wants to start fresh) */
    public void dismissFailure() {
        mLastWavData = null;
        setState(State.IDLE);
    }

    private void startRecording() {
        if (ContextCompat.checkSelfPermission(mLatinIME, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            Log.w(TAG, "RECORD_AUDIO permission not granted");
            if (mCallback != null) mCallback.onError("Microphone permission required");
            return;
        }

        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = SAMPLE_RATE * 2; // fallback: 1 second
        }

        try {
            mAudioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC,
                    SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufferSize);
            if (mAudioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize");
                if (mCallback != null) mCallback.onError("Failed to start recording");
                return;
            }
        } catch (SecurityException e) {
            Log.e(TAG, "SecurityException creating AudioRecord", e);
            if (mCallback != null) mCallback.onError("Microphone permission required");
            return;
        }

        mAudioBuffer = new ByteArrayOutputStream();
        mAudioRecord.startRecording();
        setState(State.RECORDING);

        // Read audio in background
        final int readBufferSize = bufferSize;
        mExecutor.execute(() -> {
            byte[] buffer = new byte[readBufferSize];
            while (mState == State.RECORDING) {
                int read = mAudioRecord.read(buffer, 0, buffer.length);
                if (read > 0) {
                    synchronized (mAudioBuffer) {
                        mAudioBuffer.write(buffer, 0, read);
                    }
                }
            }
        });
    }

    private void stopRecordingAndTranscribe() {
        if (mAudioRecord == null) return;

        setState(State.PROCESSING);

        try {
            mAudioRecord.stop();
            mAudioRecord.release();
        } catch (Exception e) {
            Log.e(TAG, "Error stopping AudioRecord", e);
        }
        mAudioRecord = null;

        final byte[] pcmData;
        synchronized (mAudioBuffer) {
            pcmData = mAudioBuffer.toByteArray();
        }

        // Minimum audio check (~0.5 seconds)
        if (pcmData.length < SAMPLE_RATE) {
            Log.d(TAG, "Audio too short, ignoring");
            setState(State.IDLE);
            return;
        }

        // Transcribe in background
        mExecutor.execute(() -> {
            byte[] wavData = pcmToWav(pcmData);
            String result = null;
            Exception lastError = null;

            // Try up to 2 times
            for (int attempt = 0; attempt < 2; attempt++) {
                try {
                    if (attempt > 0) Thread.sleep(1000);
                    result = sendToWhisper(wavData);
                    if (result != null && !result.isEmpty()) break;
                } catch (Exception e) {
                    lastError = e;
                    Log.e(TAG, "Transcription attempt " + (attempt + 1) + " failed", e);
                }
            }

            if (result != null && !result.isEmpty()) {
                String cleaned = result.replace("\n", " ").replace("\r", " ").trim();
                if (!cleaned.isEmpty()) {
                    final String finalText = cleaned;
                    mLastWavData = null;  // Success — discard saved audio
                    mHandler.post(() -> insertText(finalText));
                    return;
                }
            }

            // Keep WAV data for retry
            mLastWavData = wavData;

            final String errorMsg = lastError != null ? lastError.getMessage() : "No speech detected";
            mHandler.post(() -> {
                if (mCallback != null) mCallback.onError(errorMsg);
                setState(State.FAILED);
            });
        });
    }

    private void retryTranscription() {
        if (mLastWavData == null) {
            setState(State.IDLE);
            return;
        }
        setState(State.PROCESSING);
        final byte[] wavData = mLastWavData;

        mExecutor.execute(() -> {
            String result = null;
            Exception lastError = null;

            for (int attempt = 0; attempt < 2; attempt++) {
                try {
                    if (attempt > 0) Thread.sleep(1000);
                    result = sendToWhisper(wavData);
                    if (result != null && !result.isEmpty()) break;
                } catch (Exception e) {
                    lastError = e;
                    Log.e(TAG, "Retry attempt " + (attempt + 1) + " failed", e);
                }
            }

            if (result != null && !result.isEmpty()) {
                String cleaned = result.replace("\n", " ").replace("\r", " ").trim();
                if (!cleaned.isEmpty()) {
                    final String finalText = cleaned;
                    mLastWavData = null;
                    mHandler.post(() -> insertText(finalText));
                    return;
                }
            }

            // Still failed — keep WAV data for another retry
            final String errorMsg = lastError != null ? lastError.getMessage() : "No speech detected";
            mHandler.post(() -> {
                if (mCallback != null) mCallback.onError(errorMsg);
                setState(State.FAILED);
            });
        });
    }

    private String sendToWhisper(byte[] wavData) throws IOException {
        SharedPreferences prefs = DeviceProtectedUtils.getSharedPreferences(mLatinIME);
        String serverUrl = prefs.getString(Settings.PREF_WHISPER_SERVER_URL, "");
        String language = prefs.getString(Settings.PREF_WHISPER_LANGUAGE, "en");

        if (serverUrl.isEmpty()) {
            throw new IOException("Whisper server URL not configured");
        }

        String boundary = UUID.randomUUID().toString();
        URL url = new URL(serverUrl);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setDoOutput(true);
        conn.setConnectTimeout(10000);
        conn.setReadTimeout(300000); // 5 min — long recordings can take a while
        conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

        try (DataOutputStream out = new DataOutputStream(conn.getOutputStream())) {
            // File part
            out.writeBytes("--" + boundary + "\r\n");
            out.writeBytes("Content-Disposition: form-data; name=\"file\"; filename=\"recording.wav\"\r\n");
            out.writeBytes("Content-Type: audio/wav\r\n\r\n");
            out.write(wavData);
            out.writeBytes("\r\n");

            // response_format part
            out.writeBytes("--" + boundary + "\r\n");
            out.writeBytes("Content-Disposition: form-data; name=\"response_format\"\r\n\r\n");
            out.writeBytes("json\r\n");

            // language part
            out.writeBytes("--" + boundary + "\r\n");
            out.writeBytes("Content-Disposition: form-data; name=\"language\"\r\n\r\n");
            out.writeBytes(language + "\r\n");

            out.writeBytes("--" + boundary + "--\r\n");
            out.flush();
        }

        int responseCode = conn.getResponseCode();
        if (responseCode != 200) {
            throw new IOException("Server returned " + responseCode);
        }

        try (InputStream in = conn.getInputStream()) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buf = new byte[1024];
            int len;
            while ((len = in.read(buf)) != -1) {
                baos.write(buf, 0, len);
            }
            JSONObject json = new JSONObject(baos.toString("UTF-8"));
            return json.optString("text", "").trim();
        } catch (org.json.JSONException e) {
            throw new IOException("Invalid JSON response", e);
        } finally {
            conn.disconnect();
        }
    }

    private void insertText(String text) {
        Log.i(TAG, "insertText called with: \"" + text + "\"");

        // Whisper returns capitalized text — normalize to lowercase so we control casing
        if (text.length() > 0) {
            text = Character.toLowerCase(text.charAt(0)) + text.substring(1);
        }

        // Strip trailing period from short phrases (≤4 words) — Whisper always adds one
        if (text.endsWith(".")) {
            int wordCount = text.trim().split("\\s+").length;
            if (wordCount <= 4) {
                text = text.substring(0, text.length() - 1).trim();
            }
        }

        final InputConnection ic = mLatinIME.getCurrentInputConnection();

        if (ic != null) {
            CharSequence before = ic.getTextBeforeCursor(50, 0);
            CharSequence after = ic.getTextAfterCursor(50, 0);

            // --- Smart leading space ---
            // If cursor is right after a non-space character, prepend space
            if (before != null && before.length() > 0) {
                char lastBefore = before.charAt(before.length() - 1);
                if (!Character.isWhitespace(lastBefore)) {
                    text = " " + text;
                }
            }

            // --- Context-aware capitalization ---
            text = adjustCapitalization(before != null ? before.toString() : null, text);

            // --- Smart trailing space ---
            // Add trailing space after insertion, but not if next char is space/punctuation
            boolean addTrailingSpace = true;
            if (after != null && after.length() > 0) {
                char nextChar = after.charAt(0);
                if (nextChar == ' ' || nextChar == '.' || nextChar == ','
                        || nextChar == '!' || nextChar == '?' || nextChar == ';'
                        || nextChar == ':' || nextChar == ')') {
                    addTrailingSpace = false;
                }
            }
            // Don't add trailing space if text already ends with space or punctuation
            if (text.length() > 0) {
                char lastChar = text.charAt(text.length() - 1);
                if (lastChar == ' ' || lastChar == '.' || lastChar == '!' || lastChar == '?') {
                    addTrailingSpace = false;
                }
            }
            // If inserting mid-word (next char is a letter), add space to separate
            if (after != null && after.length() > 0 && Character.isLetterOrDigit(after.charAt(0))) {
                text = text + " ";
                addTrailingSpace = false;
            }
        } else {
            // No InputConnection — capitalize first letter
            text = capitalizeFirst(text);
        }

        // Use LatinIME's onTextInput which handles InputConnection properly
        try {
            Log.i(TAG, "Inserting cleaned text: \"" + text + "\"");
            mLatinIME.onTextInput(text);
        } catch (Exception e) {
            Log.e(TAG, "Error inserting text", e);
            if (mCallback != null) mCallback.onError("Failed to insert text");
        }
        setState(State.IDLE);
    }

    /**
     * Context-aware capitalization matching VoiceMic logic.
     * Capitalizes after sentence-ending punctuation, newlines, bullet markers.
     * Forces lowercase mid-sentence.
     */
    private String adjustCapitalization(String before, String text) {
        if (text == null || text.isEmpty()) return text;

        boolean shouldCapitalize = false;

        if (before == null || before.isEmpty()) {
            shouldCapitalize = true; // Start of field
        } else {
            // Find last non-space character
            int lookBack = before.length() - 1;
            while (lookBack >= 0 && before.charAt(lookBack) == ' ') lookBack--;

            if (lookBack < 0) {
                shouldCapitalize = true; // Only spaces before cursor
            } else {
                char lastChar = before.charAt(lookBack);
                shouldCapitalize = (lastChar == '.' || lastChar == '!' || lastChar == '?'
                        || lastChar == '\n' || lastChar == '\r'
                        || lastChar == '-' || lastChar == '*' || lastChar == '•');
            }
        }

        // Find first letter in text (not just first char — handles leading spaces/punctuation)
        int firstLetter = -1;
        for (int i = 0; i < text.length(); i++) {
            if (Character.isLetter(text.charAt(i))) {
                firstLetter = i;
                break;
            }
        }
        if (firstLetter < 0) return text;

        char c = text.charAt(firstLetter);
        if (shouldCapitalize && Character.isLowerCase(c)) {
            return text.substring(0, firstLetter) + Character.toUpperCase(c) + text.substring(firstLetter + 1);
        } else if (!shouldCapitalize && Character.isUpperCase(c)) {
            return text.substring(0, firstLetter) + Character.toLowerCase(c) + text.substring(firstLetter + 1);
        }
        return text;
    }

    private String capitalizeFirst(String text) {
        if (text == null || text.isEmpty()) return text;
        for (int i = 0; i < text.length(); i++) {
            if (Character.isLetter(text.charAt(i))) {
                if (Character.isLowerCase(text.charAt(i))) {
                    return text.substring(0, i) + Character.toUpperCase(text.charAt(i)) + text.substring(i + 1);
                }
                break;
            }
        }
        return text;
    }

    private void setState(State state) {
        mState = state;
        // Keep screen on while recording or processing
        setKeepScreenOn(state == State.RECORDING || state == State.PROCESSING);
        if (mCallback != null) {
            mHandler.post(() -> mCallback.onStateChanged(state));
        }
    }

    private void setKeepScreenOn(boolean on) {
        mHandler.post(() -> {
            try {
                android.view.Window window = mLatinIME.getWindow().getWindow();
                if (window != null) {
                    if (on) {
                        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                    } else {
                        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                    }
                }
            } catch (Exception e) {
                Log.w(TAG, "Could not set keep screen on", e);
            }
        });
    }

    /**
     * Cancel recording without transcribing.
     */
    public void cancel() {
        if (mAudioRecord != null) {
            try {
                mAudioRecord.stop();
                mAudioRecord.release();
            } catch (Exception e) {
                // ignore
            }
            mAudioRecord = null;
        }
        mLastWavData = null;
        setState(State.IDLE);
    }

    private void saveAudioForRecovery(byte[] wavData) {
        try {
            java.io.File dir = new java.io.File(mLatinIME.getFilesDir(), "voice_recovery");
            if (!dir.exists()) dir.mkdirs();
            String filename = "voice_" + new java.text.SimpleDateFormat("yyyyMMdd_HHmmss",
                    java.util.Locale.US).format(new java.util.Date()) + ".wav";
            java.io.File file = new java.io.File(dir, filename);
            java.io.FileOutputStream fos = new java.io.FileOutputStream(file);
            fos.write(wavData);
            fos.close();
            Log.i(TAG, "Saved recovery audio: " + file.getAbsolutePath());
        } catch (Exception e) {
            Log.e(TAG, "Failed to save recovery audio", e);
        }
    }

    private static byte[] pcmToWav(byte[] pcmData) {
        int totalDataLen = pcmData.length + 36;
        int channels = 1;
        int byteRate = SAMPLE_RATE * channels * 2;

        ByteBuffer header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN);
        header.put("RIFF".getBytes());
        header.putInt(totalDataLen);
        header.put("WAVE".getBytes());
        header.put("fmt ".getBytes());
        header.putInt(16);
        header.putShort((short) 1);
        header.putShort((short) channels);
        header.putInt(SAMPLE_RATE);
        header.putInt(byteRate);
        header.putShort((short) (channels * 2));
        header.putShort((short) 16);
        header.put("data".getBytes());
        header.putInt(pcmData.length);

        byte[] wav = new byte[44 + pcmData.length];
        System.arraycopy(header.array(), 0, wav, 0, 44);
        System.arraycopy(pcmData, 0, wav, 44, pcmData.length);
        return wav;
    }
}
