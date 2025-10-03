// Shim for scripts/app.ts
export const ANIM_PREVIEW_WIDGET = window.comfyAPI.app.ANIM_PREVIEW_WIDGET;
export const ComfyApp = window.comfyAPI.app.ComfyApp;
export const app = window.comfyAPI.app.app;

// In web_custom_versions/Comfy-Org_ComfyUI_frontend/1.28.4/scripts/app.js

// ... (original content of app.js)

// Add this code to the end of the file
document.addEventListener('DOMContentLoaded', () => {
    const startButton = document.getElementById('live-processing-start');
    const stopButton = document.getElementById('live-processing-stop');

    if (startButton && stopButton) {
        startButton.addEventListener('click', () => {
            console.log("Starting live processing...");
            window.comfyAPI.api.send(JSON.stringify({ type: 'live_processing_start' }));
        });

        stopButton.addEventListener('click', () => {
            console.log("Stopping live processing...");
            window.comfyAPI.api.send(JSON.stringify({ type: 'live_processing_stop' }));
        });
    }
});