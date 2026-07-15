/**
 * welcome.js
 *
 * Handles the anonymous session gate:
 *  - Detects a returning session_id saved in localStorage and offers to
 *    resume it.
 *  - On "Continue completely anonymously", calls POST /api/session/start
 *    with the optional age_range / gender fields, stores the returned
 *    session_id in localStorage, and redirects to /chat?sid=<id>.
 */

(function () {
  const STORAGE_KEY = "mindcare_session_id";

  const returningBanner = document.getElementById("returningBanner");
  const returningContinueBtn = document.getElementById("returningContinueBtn");
  const startFreshBtn = document.getElementById("startFreshBtn");
  const continueBtn = document.getElementById("continueBtn");
  const ageRangeSelect = document.getElementById("ageRange");
  const genderSelect = document.getElementById("gender");

  function goToChat(sessionId) {
    window.location.href = `/chat?sid=${encodeURIComponent(sessionId)}`;
  }

  function getSavedSessionId() {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      return null;
    }
  }

  function saveSessionId(sessionId) {
    try {
      window.localStorage.setItem(STORAGE_KEY, sessionId);
    } catch (e) {
      /* localStorage unavailable (e.g. private browsing) -- non-fatal */
    }
  }

  function clearSavedSessionId() {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      /* no-op */
    }
  }

  async function startNewSession() {
    continueBtn.disabled = true;
    continueBtn.textContent = "Starting your session...";

    try {
      const response = await fetch("/api/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          age_range: ageRangeSelect.value || null,
          gender: genderSelect.value || null,
        }),
      });

      if (!response.ok) {
        throw new Error(`Session start failed with status ${response.status}`);
      }

      const data = await response.json();
      saveSessionId(data.session_id);
      goToChat(data.session_id);
    } catch (err) {
      console.error("MindCare: failed to start session", err);
      continueBtn.disabled = false;
      continueBtn.textContent = "Continue completely anonymously";
      alert("Something went wrong starting your session. Please try again.");
    }
  }

  // -- Returning-session banner ------------------------------------------

  const savedSessionId = getSavedSessionId();
  if (savedSessionId) {
    returningBanner.hidden = false;
  }

  if (returningContinueBtn) {
    returningContinueBtn.addEventListener("click", () => {
      const sid = getSavedSessionId();
      if (sid) {
        goToChat(sid);
      }
    });
  }

  if (startFreshBtn) {
    startFreshBtn.addEventListener("click", () => {
      clearSavedSessionId();
      returningBanner.hidden = true;
    });
  }

  continueBtn.addEventListener("click", startNewSession);
})();
