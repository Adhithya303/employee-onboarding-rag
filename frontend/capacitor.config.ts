import type { CapacitorConfig } from "@capacitor/cli";

const config: CapacitorConfig = {
  appId: "com.facultyonboarding.assistant",
  appName: "Faculty Onboarding Assistant",
  webDir: "dist",
  server: {
    // The deployed Render backend — HTTPS, no cleartext needed
    url: "https://employee-onboarding-rag.onrender.com",
    cleartext: false,
  },
  android: {
    // Allow Capacitor WebView to make requests to the Render backend
    allowMixedContent: false,
  },
};

export default config;

