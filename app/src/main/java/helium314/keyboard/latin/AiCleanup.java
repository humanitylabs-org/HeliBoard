package helium314.keyboard.latin;

import android.content.SharedPreferences;
import android.os.Handler;
import android.os.Looper;
import android.view.inputmethod.ExtractedText;
import android.view.inputmethod.ExtractedTextRequest;
import android.view.inputmethod.InputConnection;

import helium314.keyboard.latin.settings.Settings;
import helium314.keyboard.latin.utils.DeviceProtectedUtils;
import helium314.keyboard.latin.utils.Log;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * AI text cleanup: sends selected or full field text to a local LLM (Ollama)
 * for punctuation, formatting, and grammar cleanup, then replaces the text.
 */
public class AiCleanup {
    private static final String TAG = "AiCleanup";
    private static final String DEFAULT_SERVER_URL = "http://100.83.57.121:11434";
    private static final String DEFAULT_MODEL = "qwen2.5:14b";

    public enum State { IDLE, PROCESSING }

    public interface Callback {
        void onStateChanged(State state);
        void onError(String message);
    }

    private final LatinIME mLatinIME;
    private final Handler mHandler = new Handler(Looper.getMainLooper());
    private final ExecutorService mExecutor = Executors.newSingleThreadExecutor();

    private volatile State mState = State.IDLE;
    private Callback mCallback;

    public AiCleanup(LatinIME latinIME) {
        mLatinIME = latinIME;
    }

    public void setCallback(Callback callback) {
        mCallback = callback;
    }

    public State getState() {
        return mState;
    }

    /**
     * Run cleanup on the current text field content (or selection).
     */
    public void cleanup() {
        if (mState == State.PROCESSING) return;

        final InputConnection ic = mLatinIME.getCurrentInputConnection();
        if (ic == null) {
            if (mCallback != null) mCallback.onError("No input connection");
            return;
        }

        // Check if there's a selection first
        CharSequence selected = ic.getSelectedText(0);
        final boolean hasSelection = selected != null && selected.length() > 0;
        final String textToClean;
        final int selStart;

        if (hasSelection) {
            textToClean = selected.toString();
            // We'll replace just the selection
            selStart = -1; // sentinel: replace selection
        } else {
            // Get entire field content
            ExtractedTextRequest req = new ExtractedTextRequest();
            req.hintMaxChars = 10000;
            ExtractedText extracted = ic.getExtractedText(req, 0);
            if (extracted == null || extracted.text == null || extracted.text.length() == 0) {
                if (mCallback != null) mCallback.onError("No text to clean up");
                return;
            }
            textToClean = extracted.text.toString();
            selStart = extracted.startOffset;
        }

        if (textToClean.trim().isEmpty()) {
            if (mCallback != null) mCallback.onError("No text to clean up");
            return;
        }

        setState(State.PROCESSING);

        mExecutor.execute(() -> {
            try {
                String cleaned = sendToLlm(textToClean);
                if (cleaned != null && !cleaned.isEmpty()) {
                    final String result = cleaned;
                    mHandler.post(() -> {
                        replaceText(ic, result, hasSelection, textToClean.length());
                        setState(State.IDLE);
                    });
                } else {
                    mHandler.post(() -> {
                        if (mCallback != null) mCallback.onError("Empty response from AI");
                        setState(State.IDLE);
                    });
                }
            } catch (Exception e) {
                Log.e(TAG, "AI cleanup failed", e);
                final String msg = e.getMessage();
                mHandler.post(() -> {
                    if (mCallback != null) mCallback.onError(msg != null ? msg : "Cleanup failed");
                    setState(State.IDLE);
                });
            }
        });
    }

    private void replaceText(InputConnection ic, String newText, boolean wasSelection, int originalLength) {
        if (wasSelection) {
            // Replace selection — commitText replaces selected text
            ic.commitText(newText, 1);
        } else {
            // Replace entire field: select all, then commit
            ic.performContextMenuAction(android.R.id.selectAll);
            ic.commitText(newText, 1);
        }
    }

    private String sendToLlm(String text) throws IOException {
        SharedPreferences prefs = DeviceProtectedUtils.getSharedPreferences(mLatinIME);
        String serverUrl = prefs.getString(Settings.PREF_AI_CLEANUP_SERVER_URL, DEFAULT_SERVER_URL);
        String model = prefs.getString(Settings.PREF_AI_CLEANUP_MODEL, DEFAULT_MODEL);
        if (model.isEmpty()) model = DEFAULT_MODEL;

        String prompt = "Clean up this voice-transcribed text. Fix punctuation, capitalization, and grammar. "
                + "Add paragraph breaks where topic shifts. Keep the conversational tone — don't make it formal. "
                + "Return ONLY the cleaned text with no explanation or preamble:\n\n" + text;

        JSONObject requestBody = new JSONObject();
        try {
            requestBody.put("model", model);
            requestBody.put("prompt", prompt);
            requestBody.put("stream", false);
        } catch (org.json.JSONException e) {
            throw new IOException("Failed to build request", e);
        }

        URL url = new URL(serverUrl + "/api/generate");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        conn.setConnectTimeout(10000);
        conn.setReadTimeout(60000); // LLM can be slow

        byte[] body = requestBody.toString().getBytes(StandardCharsets.UTF_8);
        conn.setFixedLengthStreamingMode(body.length);

        try (OutputStream out = conn.getOutputStream()) {
            out.write(body);
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
            return json.optString("response", "").trim();
        } catch (org.json.JSONException e) {
            throw new IOException("Invalid JSON response", e);
        } finally {
            conn.disconnect();
        }
    }

    private void setState(State state) {
        mState = state;
        if (mCallback != null) {
            mHandler.post(() -> mCallback.onStateChanged(state));
        }
    }
}
